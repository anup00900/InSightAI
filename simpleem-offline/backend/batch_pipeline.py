"""Batch analysis pipeline for offline video analysis — PARALLELIZED.

Runs AFTER transcription is complete.  Extracts visual signals from frames,
audio features with librosa, text/voice signals with GPT-4.1, personality
analysis, correlations, engagement scores, summary, flags, and coaching.

KEY OPTIMIZATION: Frame, audio, and text analysis run concurrently via
asyncio.gather.  Partial results are saved to DB as each phase completes,
enabling progressive frontend display.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import Counter
from typing import List

import aiosqlite

from .models import DB_PATH
from .frame_extractor import extract_single_frame, extract_full_audio, extract_audio_chunk
from .video_processor import get_video_duration
from .core42_client import (
    analyze_frame_all_signals,
    analyze_frame_triple_vote,
    analyze_voice_signal,
    analyze_voice_signal_dual,
    analyze_words_signal,
    analyze_words_signal_dual,
    analyze_personality,
    build_correlations,
    generate_coaching,
    generate_meeting_summary,
    generate_meeting_notes,
    generate_flags_and_scores,
    extract_names_from_frame,
    assign_speakers_to_names,
)
from .voting_engine import cross_signal_validate
from .pipelines.audio import AudioAnalyzer
from .engagement_engine import EngagementEngine, EngagementAlertDetector, EngagementScore

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
FRAME_INTERVAL = 20.0      # Analyze a frame every 20 seconds — balance speed vs quality
AUDIO_CHUNK_SEC = 30.0      # Audio feature extraction in 30-second chunks
FRAME_BATCH_SIZE = 30       # Parallel frame analysis batch size — bigger batches for more parallelism
NAME_EXTRACT_COUNT = 5      # Sample 5 frames for name extraction — need enough to find all participants

# ---------------------------------------------------------------------------
# SSE event system — per-video event queues for streaming
# ---------------------------------------------------------------------------

# Per-video maximum progress (monotonic enforcement)
_max_progress: dict[str, float] = {}

# Per-video active stage tracking for concurrent phase awareness
_active_stages: dict[str, set[str]] = {}

# Per-video event queues for SSE streaming
_event_queues: dict[str, list[asyncio.Queue]] = {}


def _mark_stage_active(video_id: str, stage: str):
    """Mark a stage as actively running for a video."""
    if video_id not in _active_stages:
        _active_stages[video_id] = set()
    _active_stages[video_id].add(stage)


def _mark_stage_done(video_id: str, stage: str):
    """Mark a stage as completed (remove from active set)."""
    if video_id in _active_stages:
        _active_stages[video_id].discard(stage)


def subscribe_events(video_id: str) -> asyncio.Queue:
    """Subscribe to analysis events for a video. Returns an asyncio.Queue."""
    q: asyncio.Queue = asyncio.Queue()
    if video_id not in _event_queues:
        _event_queues[video_id] = []
    _event_queues[video_id].append(q)
    return q


def unsubscribe_events(video_id: str, q: asyncio.Queue):
    """Remove a subscriber queue."""
    if video_id in _event_queues:
        _event_queues[video_id] = [x for x in _event_queues[video_id] if x is not q]
        if not _event_queues[video_id]:
            del _event_queues[video_id]


async def _emit_event(video_id: str, event: dict):
    """Push an event to all subscribers for this video."""
    for q in _event_queues.get(video_id, []):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if subscriber is too slow


# ---------------------------------------------------------------------------
# Helper: update progress in DB
# ---------------------------------------------------------------------------

async def _update_progress(
    video_id: str,
    wizard_step: str,
    progress: float,
    detail: str,
):
    """Write analysis progress — never goes backward."""
    # Enforce monotonic progress
    prev = _max_progress.get(video_id, 0)
    if wizard_step == "complete":
        progress = 100.0
    else:
        progress = max(progress, prev)
    _max_progress[video_id] = progress

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE videos
               SET wizard_step = ?, analyze_progress = ?, progress_detail = ?
               WHERE id = ?""",
            (wizard_step, round(progress, 2), detail, video_id),
        )
        await db.commit()

    # Also emit SSE event (include active_stages for concurrent phase tracking)
    await _emit_event(video_id, {
        "type": "progress",
        "progress": round(progress, 2),
        "detail": detail,
        "stage": wizard_step,
        "active_stages": list(_active_stages.get(video_id, set())),
    })


# ---------------------------------------------------------------------------
# Sub-step: Extract participant names from frames
# ---------------------------------------------------------------------------

async def _extract_single_name(video_path: str, ts: float, output_dir: str) -> list[dict]:
    """Extract names from a single frame at the given timestamp."""
    try:
        frame_path = await extract_single_frame(video_path, ts, output_dir)
        participants = await extract_names_from_frame(frame_path)
        try:
            os.remove(frame_path)
        except OSError:
            pass
        return participants
    except Exception as e:
        logger.warning(f"Name extraction failed at {ts:.1f}s: {e}")
        return []


async def _extract_names(
    video_path: str, duration: float, output_dir: str
) -> tuple[list[str], list[str]]:
    """Sample frames and use GPT-4o to OCR participant names + camera status.

    Prioritizes first 5 seconds (names most visible at video start in
    Zoom/Teams/Meet), plus 3 samples from the body of the video.

    Returns (all_names, camera_on_names) — deduplicated.
    """
    # Sample 2 frames from first 5 seconds (names most visible at start)
    sample_times = [1.0, 3.0]
    # Plus 3 samples from body of video
    for frac in [0.15, 0.35, 0.55]:
        t = duration * frac
        if t > 5.0:
            sample_times.append(t)

    if not sample_times:
        sample_times = [min(1.0, duration * 0.5)]

    # Collect names from all sampled frames
    name_camera_counts: dict[str, dict] = {}  # name -> {"count": N, "camera_on_count": N}

    # Run all frame extractions + OCR concurrently
    valid_times = [ts for ts in sample_times[:NAME_EXTRACT_COUNT] if ts <= duration]
    results = await asyncio.gather(
        *[_extract_single_name(video_path, ts, output_dir) for ts in valid_times],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            continue
        for p in result:
            name = p.get("name", "").strip()
            if not name:
                continue
            norm = _normalize_name(name)
            if norm not in name_camera_counts:
                name_camera_counts[norm] = {"name": name, "count": 0, "camera_on_count": 0}
            name_camera_counts[norm]["count"] += 1
            if p.get("camera_on", True):
                name_camera_counts[norm]["camera_on_count"] += 1
            if len(name) > len(name_camera_counts[norm]["name"]):
                name_camera_counts[norm]["name"] = name

    # Merge near-duplicate names (e.g. "Thekkeppat" vs "Thekkepat")
    merged: dict[str, dict] = {}
    for norm_key, info in name_camera_counts.items():
        matched_to = None
        for existing_key in merged:
            # Compare last parts (surnames most prone to OCR variation)
            if _edit_distance(norm_key, existing_key) <= 3 and max(len(norm_key), len(existing_key)) >= 6:
                matched_to = existing_key
                break
        if matched_to:
            merged[matched_to]["count"] += info["count"]
            merged[matched_to]["camera_on_count"] += info["camera_on_count"]
            # Keep the longer (likely more complete) name
            if len(info["name"]) > len(merged[matched_to]["name"]):
                merged[matched_to]["name"] = info["name"]
        else:
            merged[norm_key] = dict(info)

    # All participants found in at least 1 frame
    all_names = [info["name"] for info in merged.values()]
    # Camera-on = found with camera on in majority of their appearances
    camera_on_names = [
        info["name"]
        for info in merged.values()
        if info["camera_on_count"] > info["count"] / 2
    ]

    return all_names, camera_on_names


def _normalize_name(name: str) -> str:
    """Normalize a name for deduplication (lowercase, strip whitespace/punctuation)."""
    import re
    name = name.lower().strip()
    # Remove trailing ellipsis or dots from truncated names
    name = re.sub(r'[.\u2026]+$', '', name).strip()
    # Remove parenthetical suffixes like "(E)", "(Host)", etc.
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
    # Remove commas (normalize "Last, First" to "last first")
    name = name.replace(',', ' ')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name


def _build_name_lookup(canonical_names: list[str]) -> dict[str, str]:
    """Build a normalized-name -> canonical-name lookup.

    Creates entries for the full name AND individual name parts (first name,
    last name, etc.) so that partial matches like "Sam" -> "Chong, Sam" work.
    """
    lookup = {}
    for name in canonical_names:
        # Full normalized name
        lookup[_normalize_name(name)] = name
        # Also index by each individual word (for partial first/last name matches)
        parts = _normalize_name(name).split()
        for part in parts:
            if len(part) >= 3:  # Skip very short parts to avoid false matches
                if part not in lookup:
                    lookup[part] = name
                # If part already exists, it's ambiguous — keep the first match
    return lookup


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _split_name_parts(name: str) -> list[str]:
    """Split a name into parts, handling commas, parentheses, and spaces."""
    import re
    # Remove parenthetical suffixes
    name = re.sub(r'\s*\([^)]*\)', '', name)
    # Split on commas and spaces, filter empty parts
    parts = re.split(r'[,\s]+', name.lower().strip())
    return [p for p in parts if p]


def _resolve_label_to_name(
    label: str,
    canonical_names: list[str],
    name_lookup: dict[str, str],
) -> str | None:
    """Resolve a frame label to a canonical participant name.

    Handles all cases:
    - Exact match via name_lookup
    - Fuzzy match via _fuzzy_match_name
    - Positional "Person N" labels → canonical_names[N-1]
    - Numeric-only labels like "1", "2"
    - Single-participant fallback
    """
    import re

    if not label or not canonical_names:
        return canonical_names[0] if canonical_names else None

    # Try direct lookup first (handles normalized names + partial names)
    matched = name_lookup.get(_normalize_name(label))
    if matched:
        return matched

    # Try fuzzy match
    matched = _fuzzy_match_name(label, canonical_names)
    if matched:
        return matched

    # Handle "Person N", "Speaker N", "Participant N", or just "N" labels
    m = re.match(r'(?:person|speaker|participant|p)?\s*(\d+)', label.lower().strip())
    if m:
        idx = int(m.group(1)) - 1  # "Person 1" → index 0
        if 0 <= idx < len(canonical_names):
            return canonical_names[idx]

    # Last resort: if only 1 participant, everything goes to them
    if len(canonical_names) == 1:
        return canonical_names[0]

    return None


def _fuzzy_match_name(label: str, canonical_names: list[str]) -> str | None:
    """Try to fuzzy-match a frame label to a canonical name.

    Handles cases like 'Person 1' labels, partial names ('Sam' -> 'Chong, Sam'),
    slight name variations ('Thekkeppat' vs 'Thekkepat'), and names with commas
    or parenthetical suffixes.
    Returns the matched canonical name, or None.
    """
    if not canonical_names:
        return None

    label_lower = label.lower().strip()
    label_norm = _normalize_name(label)

    # Exact match (after normalization)
    for name in canonical_names:
        if _normalize_name(name) == label_norm:
            return name

    # Substring match (label contains name or name contains label)
    # Also compare normalized versions
    for name in canonical_names:
        name_norm = _normalize_name(name)
        if name_norm in label_norm or label_norm in name_norm:
            return name
        if name.lower() in label_lower or label_lower in name.lower():
            return name

    # Single-word match against name parts (e.g., "Sam" -> "Chong, Sam")
    label_parts = _split_name_parts(label)
    if len(label_parts) == 1 and len(label_parts[0]) >= 3:
        for name in canonical_names:
            name_parts = _split_name_parts(name)
            if label_parts[0] in name_parts:
                return name

    # First/last name overlap — with edit-distance on individual parts
    best_match = None
    best_score = 0
    for name in canonical_names:
        name_parts = _split_name_parts(name)
        # Exact word overlap
        exact_overlap = len(set(label_parts) & set(name_parts))
        # Fuzzy word overlap: match parts within edit distance ≤ 2
        fuzzy_overlap = 0
        for lp in label_parts:
            for np in name_parts:
                if lp == np:
                    continue  # already counted
                if _edit_distance(lp, np) <= 2 and max(len(lp), len(np)) >= 4:
                    fuzzy_overlap += 0.8
                    break
        score = exact_overlap + fuzzy_overlap
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= 0.8:
        return best_match

    # Full-string edit distance as last resort (for very close names)
    best_dist = 999
    best_ed_match = None
    for name in canonical_names:
        dist = _edit_distance(label_lower, name.lower())
        # Allow up to 3 edits for strings of length >= 8
        threshold = 3 if max(len(label_lower), len(name.lower())) >= 8 else 2
        if dist <= threshold and dist < best_dist:
            best_dist = dist
            best_ed_match = name

    if best_ed_match:
        return best_ed_match

    return None


# ---------------------------------------------------------------------------
# Sub-step: Analyze frames in parallel batches
# ---------------------------------------------------------------------------

async def _analyze_frames_batch(
    video_path: str,
    duration: float,
    output_dir: str,
    video_id: str,
    known_names: list[str] | None = None,
    camera_on_names: list[str] | None = None,
) -> list[dict]:
    """Extract and analyze frames every FRAME_INTERVAL seconds.

    Returns list of analysis results (one per frame).
    """
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += FRAME_INTERVAL

    if not timestamps:
        return []

    total = len(timestamps)
    results = []

    for batch_start in range(0, total, FRAME_BATCH_SIZE):
        batch_ts = timestamps[batch_start: batch_start + FRAME_BATCH_SIZE]

        # Extract frames in parallel
        frame_tasks = [
            extract_single_frame(video_path, ts, output_dir)
            for ts in batch_ts
        ]
        frame_paths = await asyncio.gather(*frame_tasks, return_exceptions=True)

        # Analyze frames in parallel
        analyze_tasks = []
        valid_frames = []
        for i, (fp, ts) in enumerate(zip(frame_paths, batch_ts)):
            if isinstance(fp, Exception):
                logger.warning(f"Frame extraction failed at {ts:.1f}s: {fp}")
                continue
            analyze_tasks.append(analyze_frame_triple_vote(fp, ts, known_names, camera_on_names))
            valid_frames.append((fp, ts))

        if analyze_tasks:
            analyses = await asyncio.gather(*analyze_tasks, return_exceptions=True)
            for j, (analysis, (fp, ts)) in enumerate(zip(analyses, valid_frames)):
                if isinstance(analysis, Exception):
                    logger.warning(f"Frame analysis failed at {ts:.1f}s: {analysis}")
                    analysis = (
                        {
                            "participants": [{
                                "label": "Person 1",
                                "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
                                "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "unable to analyze"},
                                "gestures": ["still posture", "resting hands"],
                                "reactions": ["neutral gaze", "steady expression"],
                            }]
                        },
                        0.0,
                        [],
                    )

                # Unpack tuple from triple-vote: (consensus, confidence, log_entries)
                if isinstance(analysis, tuple):
                    consensus, confidence, log_entries = analysis
                else:
                    consensus, confidence, log_entries = analysis, 0.5, []

                consensus["timestamp"] = ts
                consensus["_confidence"] = confidence
                consensus["_voting_log"] = log_entries
                results.append(consensus)

                # Clean up temp frame
                try:
                    os.remove(fp)
                except OSError:
                    pass

        # Update progress
        pct_done = min(1.0, (batch_start + len(batch_ts)) / total)
        await _update_progress(
            video_id, "analyzing",
            10 + pct_done * 35,
            f"Analyzing frames: {batch_start + len(batch_ts)}/{total}",
        )

    _mark_stage_done(video_id, "frames")
    return results


# ---------------------------------------------------------------------------
# Sub-step: Analyze audio features with librosa
# ---------------------------------------------------------------------------

async def _analyze_audio_features(
    audio_path: str,
    duration: float,
    output_dir: str,
    video_id: str,
) -> list[dict]:
    """Extract audio features in AUDIO_CHUNK_SEC chunks using librosa.

    Returns list of dicts with timestamp + AudioFeatures fields.
    """
    if not audio_path or not os.path.exists(audio_path):
        return []

    analyzer = AudioAnalyzer()
    results = []
    t = 0.0
    total_chunks = max(1, int(duration / AUDIO_CHUNK_SEC))
    chunk_idx = 0

    while t < duration:
        end_t = min(t + AUDIO_CHUNK_SEC, duration)
        try:
            chunk_path = await extract_audio_chunk(audio_path, t, end_t, output_dir)
            # Run librosa analysis in executor (CPU-bound)
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(None, analyzer.analyze_file, chunk_path)
            results.append({
                "timestamp": t,
                "energy": features.engagement_score,
                "pitch_mean": features.pitch_mean,
                "pitch_std": features.pitch_std,
                "volume_energy": features.volume_energy,
                "speaking_rate": features.speaking_rate,
                "pause_ratio": features.pause_ratio,
            })
            try:
                os.remove(chunk_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"Audio analysis failed at {t:.1f}s: {e}")

        chunk_idx += 1
        # Update progress: audio uses 46-52% range
        pct = 46 + int((chunk_idx / total_chunks) * 6)
        await _update_progress(video_id, "analyzing", pct, f"Processing audio: {chunk_idx}/{total_chunks} chunks")

        t += AUDIO_CHUNK_SEC

    _mark_stage_done(video_id, "audio")
    return results


# ---------------------------------------------------------------------------
# Sub-step: Analyze text signals (voice + words) — concurrent per chunk
# ---------------------------------------------------------------------------

async def _analyze_text_signals(
    video_id: str,
    transcript_segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Analyze voice and word signals from transcript chunks using GPT-4.1.

    Voice and word analysis for each chunk run concurrently.
    Returns (voice_signals, word_signals).
    """
    if not transcript_segments:
        return [], []

    # Group segments into ~30s chunks for analysis
    CHUNK_DURATION = 120.0  # 120s per chunk (was 60) — fewer API calls with larger chunks
    chunks: list[tuple[float, str]] = []
    current_text = []
    chunk_start = transcript_segments[0].get("start_time", 0)

    for seg in transcript_segments:
        current_text.append(seg.get("text", ""))
        seg_end = seg.get("end_time", seg.get("start_time", 0))
        if seg_end - chunk_start >= CHUNK_DURATION:
            chunks.append((chunk_start, " ".join(current_text)))
            current_text = []
            chunk_start = seg_end

    if current_text:
        chunks.append((chunk_start, " ".join(current_text)))

    voice_signals = []
    word_signals = []

    async def _analyze_voice(txt, timestamp):
        try:
            result = await analyze_voice_signal_dual(txt)
            return {
                "timestamp": timestamp,
                "tone": result.get("tone", "neutral"),
                "pace": result.get("pace", "moderate"),
                "energy": result.get("energy", 50),
                "dynamics": result.get("dynamics", ""),
            }
        except Exception as e:
            logger.warning(f"Voice signal analysis failed at {timestamp:.1f}s: {e}")
            return None

    async def _analyze_words(txt, timestamp):
        try:
            result = await analyze_words_signal_dual(txt, timestamp)
            return {
                "timestamp": timestamp,
                "sentiment": result.get("sentiment", "neutral"),
                "sentiment_score": result.get("sentiment_score", 0.5),
                "keywords": result.get("keywords", []),
                "key_phrases": result.get("key_phrases", []),
            }
        except Exception as e:
            logger.warning(f"Word signal analysis failed at {timestamp:.1f}s: {e}")
            return None

    # Use semaphore to limit concurrent API calls
    TEXT_ANALYSIS_SEM = asyncio.Semaphore(8)
    _text_done_count = 0
    _text_total = len(chunks)

    async def _analyze_chunk_pair(ts: float, text: str):
        """Analyze both voice and word signals for a chunk, with concurrency limit."""
        nonlocal _text_done_count
        if not text.strip():
            _text_done_count += 1
            return None, None
        async with TEXT_ANALYSIS_SEM:
            voice_res, word_res = await asyncio.gather(
                _analyze_voice(text, ts),
                _analyze_words(text, ts),
            )
            _text_done_count += 1
            # Update progress: text uses 53-60% range
            pct = 53 + int((_text_done_count / max(_text_total, 1)) * 7)
            await _update_progress(video_id, "analyzing", pct, f"Analyzing text: {_text_done_count}/{_text_total} chunks")
            return voice_res, word_res

    # Run all chunks concurrently (semaphore limits to 5 at a time)
    chunk_results = await asyncio.gather(
        *[_analyze_chunk_pair(ts, text) for ts, text in chunks],
        return_exceptions=True,
    )

    for result in chunk_results:
        if isinstance(result, Exception):
            logger.warning(f"Text analysis chunk failed: {result}")
            continue
        voice_res, word_res = result
        if voice_res:
            voice_signals.append(voice_res)
        if word_res:
            word_signals.append(word_res)

    _mark_stage_done(video_id, "text")
    return voice_signals, word_signals


# ---------------------------------------------------------------------------
# Partial result saving helpers
# ---------------------------------------------------------------------------

async def _save_frame_results_to_db(
    video_id: str,
    frame_results: list[dict],
    participant_names: list[str],
    participant_id_map: dict[str, str],
):
    """Save frame analysis results (emotions + signal_snapshots) to DB immediately."""
    name_lookup = _build_name_lookup(participant_names)

    async with aiosqlite.connect(DB_PATH) as db:
        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            for p_idx, p in enumerate(fr.get("participants", [])):
                label = p.get("label", f"Person {p_idx + 1}")
                name = _resolve_label_to_name(label, participant_names, name_lookup)
                if not name:
                    # Positional fallback: assign by index in frame participants list
                    name = participant_names[p_idx] if p_idx < len(participant_names) else participant_names[0]
                pid = participant_id_map.get(name)
                if not pid:
                    pid = list(participant_id_map.values())[0] if participant_id_map else "unknown"

                emotions_data = p.get("emotions", {})
                await db.execute(
                    """INSERT INTO emotions
                       (video_id, participant_id, timestamp, emotion, confidence, engagement)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        video_id, pid, ts,
                        emotions_data.get("primary", "neutral"),
                        emotions_data.get("confidence", 0.5),
                        emotions_data.get("engagement", 50),
                    ),
                )

                bl = p.get("body_language", {})
                confidence = fr.get("_confidence", 1.0)
                # Sanitize body language description — hide error/fallback text
                bl_desc = bl.get("description", "")
                if bl_desc and bl_desc.lower().strip() in ("model error", "error", "unknown", "n/a"):
                    bl_desc = ""
                await db.execute(
                    """INSERT INTO signal_snapshots
                       (video_id, participant_id, timestamp, emotion, emotion_confidence,
                        engagement, posture, openness, leaning, mirroring,
                        body_language_desc, gestures, reactions, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        video_id, pid, ts,
                        emotions_data.get("primary", "neutral"),
                        emotions_data.get("confidence", 0.5),
                        emotions_data.get("engagement", 50),
                        bl.get("posture", "upright"),
                        bl.get("openness", "mixed"),
                        bl.get("leaning", "neutral"),
                        1 if bl.get("mirroring") else 0,
                        bl_desc,
                        json.dumps(p.get("gestures", [])),
                        json.dumps(p.get("reactions", [])),
                        confidence,
                    ),
                )

        await db.commit()
    logger.info(f"[{video_id[:8]}] Saved {len(frame_results)} frame results to DB")

    # Save voting logs for audit trail
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM voting_log WHERE video_id = ?", (video_id,))
        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            confidence = fr.get("_confidence", 1.0)
            log_entries = fr.get("_voting_log", [])
            if log_entries:
                await db.execute(
                    """INSERT INTO voting_log
                       (video_id, timestamp, signal_type, consensus_result, confidence, disagreements)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        video_id, ts, "visual",
                        json.dumps(fr.get("participants", [])),
                        confidence,
                        json.dumps(log_entries),
                    ),
                )
        await db.commit()
    logger.info(f"[{video_id[:8]}] Saved voting logs for {len(frame_results)} frames")


async def _save_text_results_to_db(
    video_id: str,
    voice_signals: list[dict],
    word_signals: list[dict],
):
    """Save voice and word signals to DB immediately."""
    async with aiosqlite.connect(DB_PATH) as db:
        for vs in voice_signals:
            await db.execute(
                """INSERT INTO voice_signals
                   (video_id, timestamp, tone, pace, energy, dynamics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    video_id, vs["timestamp"],
                    vs.get("tone", "neutral"),
                    vs.get("pace", "moderate"),
                    vs.get("energy", 50),
                    vs.get("dynamics", ""),
                ),
            )

        for ws in word_signals:
            await db.execute(
                """INSERT INTO word_signals
                   (video_id, timestamp, sentiment, sentiment_score, keywords, key_phrases)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    video_id, ws["timestamp"],
                    ws.get("sentiment", "neutral"),
                    ws.get("sentiment_score", 0.5),
                    json.dumps(ws.get("keywords", [])),
                    json.dumps(ws.get("key_phrases", [])),
                ),
            )

        await db.commit()
    logger.info(f"[{video_id[:8]}] Saved {len(voice_signals)} voice + {len(word_signals)} word signals to DB")


# ---------------------------------------------------------------------------
# Main entry point — PARALLELIZED
# ---------------------------------------------------------------------------

async def run_analysis(video_id: str, video_path: str):
    """Full batch analysis pipeline — parallelized for speed.

    Assumes transcription is already complete (transcript_segments populated).

    Pipeline phases:
    1. Extract names + audio (concurrent)
    2. Frame + audio + text analysis (ALL concurrent via asyncio.gather)
    3. Save partial results to DB as each completes
    4. Personality + correlations (depend on phase 2)
    5. Summary + flags + coaching

    Updates wizard_step and analyze_progress in the DB for frontend polling.
    """
    _max_progress.pop(video_id, None)  # Reset for fresh analysis

    output_dir = os.path.join(PROCESSED_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        await _update_progress(video_id, "analyzing", 2, "Starting analysis pipeline...")

        # ---- Get video duration ----
        duration = await get_video_duration(video_path)
        if duration <= 0:
            duration = 60.0  # fallback

        # ---- Load transcript from DB ----
        transcript_segments = []
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            )
            rows = await cursor.fetchall()
            transcript_segments = [
                {
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "text": r["text"],
                    "speaker": r["speaker"],
                }
                for r in rows
            ]

        # ---- Phase 1: Extract names + audio concurrently (2-10%) ----
        _mark_stage_active(video_id, "participants")
        await _update_progress(video_id, "analyzing", 5, "Extracting participant names and audio...")

        name_task = asyncio.create_task(_extract_names(video_path, duration, output_dir))
        audio_extract_task = asyncio.create_task(extract_full_audio(video_path, output_dir))

        (known_names, camera_on_names), audio_path = await asyncio.gather(name_task, audio_extract_task)

        logger.info(f"[{video_id[:8]}] Detected {len(known_names)} participants "
                     f"({len(camera_on_names)} camera-on): {known_names}")

        # Assign real names to transcript speakers
        if known_names and transcript_segments:
            await _update_progress(video_id, "analyzing", 8, "Matching speakers to participant names...")
            seg_dicts = [
                {"start": s["start_time"], "end": s["end_time"], "text": s["text"], "speaker": s["speaker"]}
                for s in transcript_segments
            ]
            seg_dicts = await assign_speakers_to_names(seg_dicts, known_names)
            for i, seg in enumerate(transcript_segments):
                if i < len(seg_dicts):
                    seg["speaker"] = seg_dicts[i].get("speaker", seg["speaker"])

            # Update transcript in DB with real names
            async with aiosqlite.connect(DB_PATH) as db:
                for seg in transcript_segments:
                    await db.execute(
                        """UPDATE transcript_segments SET speaker = ?
                           WHERE video_id = ? AND start_time = ? AND end_time = ?""",
                        (seg["speaker"], video_id, seg["start_time"], seg["end_time"]),
                    )
                await db.commit()

        # Save participants to DB immediately so frontend can show them
        async with aiosqlite.connect(DB_PATH) as db:
            # Clear old participants first
            await db.execute("DELETE FROM participants WHERE video_id = ?", (video_id,))
            participant_id_map_early: dict[str, str] = {}
            for name in (known_names if known_names else ["Person 1"]):
                pid = str(uuid.uuid4())
                participant_id_map_early[name] = pid
                await db.execute(
                    """INSERT INTO participants
                       (id, video_id, name, engagement_score, sentiment_score,
                        speaking_pct, clarity_score, rapport_score, energy_score)
                       VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)""",
                    (pid, video_id, name),
                )
            await db.commit()
        logger.info(f"[{video_id[:8]}] Saved {len(participant_id_map_early)} participants early for live display")

        _mark_stage_done(video_id, "participants")

        # ---- Phase 2: Run frame + audio + text analysis CONCURRENTLY (10-60%) ----
        await _update_progress(video_id, "analyzing", 10, "Running visual, audio, and text analysis concurrently...")

        # Mark all three concurrent stages as active
        _mark_stage_active(video_id, "frames")
        _mark_stage_active(video_id, "audio")
        _mark_stage_active(video_id, "text")

        frame_task = asyncio.create_task(
            _analyze_frames_batch(video_path, duration, output_dir, video_id, known_names, camera_on_names)
        )
        audio_task = asyncio.create_task(
            _analyze_audio_features(audio_path, duration, output_dir, video_id)
        )
        text_task = asyncio.create_task(
            _analyze_text_signals(video_id, transcript_segments)
        )

        # Wait for all 3 to complete (they run concurrently)
        frame_results, audio_features, (voice_signals, word_signals) = await asyncio.gather(
            frame_task, audio_task, text_task
        )

        await _update_progress(video_id, "analyzing", 61, "All signal analysis complete. Computing engagement scores...")

        # ---- Phase 3: Build participant map + save partial results (60-70%) ----
        # Use OCR-extracted names as the canonical participant list
        if known_names:
            participant_names = known_names
        else:
            # Fallback: collect unique labels from frames
            label_counts = Counter()
            for fr in frame_results:
                for p in fr.get("participants", []):
                    label_counts[p.get("label", "Person 1")] += 1
            participant_names = sorted(label_counts.keys()) if label_counts else ["Person 1"]

        logger.info(f"[{video_id[:8]}] Final participant list ({len(participant_names)}): {participant_names}")

        # Build a fuzzy lookup for matching frame labels to canonical names
        name_lookup = _build_name_lookup(participant_names)

        # Save participants + frame results to DB immediately for live display
        participant_id_map: dict[str, str] = {}
        async with aiosqlite.connect(DB_PATH) as db:
            # Update participant list (might have changed from early save)
            await db.execute("DELETE FROM participants WHERE video_id = ?", (video_id,))
            await db.execute("DELETE FROM emotions WHERE video_id = ?", (video_id,))
            await db.execute("DELETE FROM signal_snapshots WHERE video_id = ?", (video_id,))
            for name in participant_names:
                pid = str(uuid.uuid4())
                participant_id_map[name] = pid
                await db.execute(
                    """INSERT INTO participants
                       (id, video_id, name, engagement_score, sentiment_score,
                        speaking_pct, clarity_score, rapport_score, energy_score)
                       VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)""",
                    (pid, video_id, name),
                )
            await db.commit()

        # Save frame results (emotions) immediately
        await _save_frame_results_to_db(video_id, frame_results, participant_names, participant_id_map)
        # Save text results immediately
        await _save_text_results_to_db(video_id, voice_signals, word_signals)

        # Compute engagement scores — with speaking activity boost
        from .signal_bus import SignalEvent, SignalType
        engine = EngagementEngine()
        participant_engagement: dict[str, list[float]] = {name: [] for name in participant_names}

        # Pre-compute speaking activity map: for each participant, which time ranges are they speaking?
        speaker_activity: dict[str, list[tuple[float, float]]] = {name: [] for name in participant_names}
        # Also compute total speaking time per participant for engagement weighting
        speaker_total_time: dict[str, float] = {name: 0.0 for name in participant_names}
        for seg in transcript_segments:
            spk = seg.get("speaker", "")
            matched = name_lookup.get(_normalize_name(spk)) or _fuzzy_match_name(spk, participant_names)
            if matched:
                speaker_activity[matched].append((seg["start_time"], seg["end_time"]))
                speaker_total_time[matched] += max(0, seg["end_time"] - seg["start_time"])

        max_speaking_time = max(speaker_total_time.values()) if speaker_total_time else 1.0
        max_speaking_time = max(max_speaking_time, 1.0)

        def _is_speaking_at(name: str, ts: float, window: float = 15.0) -> bool:
            """Check if participant is speaking near timestamp (within window)."""
            for start, end in speaker_activity.get(name, []):
                if abs(ts - start) < window or abs(ts - end) < window or (start <= ts <= end):
                    return True
            return False

        def _speaking_engagement(name: str) -> float:
            """Compute engagement score from speaking time relative to top speaker."""
            if max_speaking_time <= 0:
                return 30.0
            ratio = speaker_total_time.get(name, 0) / max_speaking_time
            # Scale: top speaker = 85, proportional down to 30 minimum
            return max(30.0, 30.0 + ratio * 55.0)

        # Pre-compute per-speaker word counts in time windows for text participation
        speaker_words_in_window: dict[str, dict[float, int]] = {name: {} for name in participant_names}
        speaker_questions_in_window: dict[str, dict[float, int]] = {name: {} for name in participant_names}
        for seg in transcript_segments:
            spk = seg.get("speaker", "")
            matched = name_lookup.get(_normalize_name(spk)) or _fuzzy_match_name(spk, participant_names)
            if not matched:
                continue
            text = seg.get("text", "")
            words = len(text.split())
            questions = text.count("?")
            # Bucket by 60s windows
            bucket = int(seg["start_time"] / 60) * 60
            speaker_words_in_window[matched][bucket] = speaker_words_in_window[matched].get(bucket, 0) + words
            speaker_questions_in_window[matched][bucket] = speaker_questions_in_window[matched].get(bucket, 0) + questions

        # Max words in any window by any speaker (for normalization)
        all_window_words = [wc for sw in speaker_words_in_window.values() for wc in sw.values()]
        max_window_words = max(all_window_words) if all_window_words else 1

        def _speaker_audio_energy(name: str, ts: float) -> float:
            """Per-speaker audio engagement: speaking now + frequency + word rate."""
            speaking_now = 1.0 if _is_speaking_at(name, ts, window=10.0) else 0.0
            # Speaking frequency: segments in last 60s
            seg_count = 0
            for start, end in speaker_activity.get(name, []):
                if ts - 60 <= start <= ts + 10:
                    seg_count += 1
            freq_score = min(1.0, seg_count / 5.0)  # 5+ segments = max
            # Overall speaking ratio
            speak_ratio = speaker_total_time.get(name, 0) / max_speaking_time
            # Composite: 50% speaking now, 30% frequency, 20% overall ratio
            energy = (speaking_now * 50 + freq_score * 30 + speak_ratio * 20)
            return max(5.0, min(100.0, energy))

        def _text_participation(name: str, ts: float, sentiment_score: float) -> float:
            """Text participation: speaking volume + questions + sentiment magnitude."""
            bucket = int(ts / 60) * 60
            words_in_bucket = speaker_words_in_window.get(name, {}).get(bucket, 0)
            questions_in_bucket = speaker_questions_in_window.get(name, {}).get(bucket, 0)
            # 40%: speaking volume relative to most active speaker in any window
            volume_score = (words_in_bucket / max(max_window_words, 1)) * 100
            # 30%: question asking (1+ questions = engaged)
            question_score = min(100, questions_in_bucket * 50)
            # 30%: sentiment magnitude (strong feelings either way = engaged)
            magnitude_score = abs(sentiment_score - 0.5) * 200
            return 0.4 * volume_score + 0.3 * question_score + 0.3 * magnitude_score

        # Determine camera-on status per participant from name extraction
        camera_on_set = set(camera_on_names) if camera_on_names else set(participant_names)

        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            for p_idx, p in enumerate(fr.get("participants", [])):
                label = p.get("label", f"Person {p_idx + 1}")
                name = _resolve_label_to_name(label, participant_names, name_lookup)
                if not name:
                    # Positional fallback: assign by index in frame participants list
                    name = participant_names[p_idx] if p_idx < len(participant_names) else participant_names[0] if participant_names else None
                if name is None or name not in participant_engagement:
                    continue

                p_camera_on = p.get("camera_on", name in camera_on_set)
                visual_engagement = p.get("emotions", {}).get("engagement", 50)
                is_speaking = _is_speaking_at(name, ts)

                # For camera-on participants who never speak at all, cap slightly
                if p_camera_on and not is_speaking and not speaker_activity.get(name):
                    visual_engagement = min(visual_engagement, 45)

                visual_signal = SignalEvent(
                    signal_type=SignalType.VISUAL,
                    participant_id=name,
                    timestamp=ts,
                    data={"engagement": visual_engagement},
                )

                # Per-speaker audio engagement (not global librosa energy)
                per_speaker_energy = _speaker_audio_energy(name, ts)
                # Blend with global librosa energy if available (librosa adds real audio info)
                if audio_features:
                    closest_audio = min(audio_features, key=lambda a: abs(a["timestamp"] - ts))
                    if abs(closest_audio["timestamp"] - ts) < AUDIO_CHUNK_SEC:
                        librosa_energy = closest_audio["energy"]
                        # 60% per-speaker transcript-based, 40% global librosa
                        blended_energy = per_speaker_energy * 0.6 + librosa_energy * 0.4
                    else:
                        blended_energy = per_speaker_energy
                else:
                    blended_energy = per_speaker_energy

                audio_signal = SignalEvent(
                    signal_type=SignalType.AUDIO,
                    participant_id=name,
                    timestamp=ts,
                    data={"energy": blended_energy},
                )

                # Text participation signal (not just sentiment)
                text_signal = None
                if word_signals:
                    closest_word = min(word_signals, key=lambda w: abs(w["timestamp"] - ts))
                    if abs(closest_word["timestamp"] - ts) < 60:
                        participation = _text_participation(name, ts, closest_word["sentiment_score"])
                        text_signal = SignalEvent(
                            signal_type=SignalType.TEXT,
                            participant_id=name,
                            timestamp=ts,
                            data={"engagement": participation},
                        )

                score = engine.compute(name, ts, visual_signal, audio_signal, text_signal, camera_on=p_camera_on)
                participant_engagement[name].append(score.holistic)

        # Run engagement alert detection
        alert_detector = EngagementAlertDetector(threshold=40, min_duration=30)
        engagement_alerts = []

        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            for p_idx, p in enumerate(fr.get("participants", [])):
                label = p.get("label", f"Person {p_idx + 1}")
                name = _resolve_label_to_name(label, participant_names, name_lookup)
                if not name:
                    name = participant_names[p_idx] if p_idx < len(participant_names) else None
                if name and name in participant_engagement:
                    scores = participant_engagement[name]
                    if scores:
                        score = EngagementScore(
                            participant_id=name, timestamp=ts,
                            holistic=scores[-1],
                            visual=p.get("emotions", {}).get("engagement", 50),
                            audio=0, verbal=0,
                        )
                        alert = alert_detector.check(score)
                        if alert:
                            engagement_alerts.append(alert)

        if engagement_alerts:
            logger.info(f"[{video_id[:8]}] Detected {len(engagement_alerts)} engagement alerts")

        await _update_progress(video_id, "analyzing", 65, "Engagement scores computed. Building personality profiles...")

        # Mark profiles stage active
        _mark_stage_active(video_id, "profiles")

        # ---- Phase 4: Personality + correlations + summary (65-85%) ----

        transcript_text = "\n".join(
            f"[{s['start_time']:.1f}s - {s['end_time']:.1f}s] {s.get('speaker', '')}: {s['text']}"
            for s in transcript_segments
        )

        signals_summary = json.dumps({
            "frame_analyses": frame_results,
            "audio_features": audio_features,
            "voice_signals": voice_signals,
        }, indent=1)

        all_signals_text = json.dumps({
            "visual": frame_results,
            "audio": audio_features,
            "voice": voice_signals,
            "words": word_signals,
        }, indent=1)

        emotions_summary = json.dumps(
            [
                {
                    "timestamp": fr.get("timestamp", 0),
                    "participants": fr.get("participants", []),
                }
                for fr in frame_results
            ],
            indent=1,
        )

        # Run personality + correlations + summary + notes + flags ALL concurrently (5-way gather)
        await _update_progress(video_id, "analyzing", 70, "Running personality + correlations + summary + notes in parallel...")

        personality_result, correlations_result, summary_result, notes_result, scores_result = await asyncio.gather(
            analyze_personality(signals_summary, transcript_text),
            build_correlations(all_signals_text, transcript_text),
            generate_meeting_summary(transcript_text, participant_names),
            generate_meeting_notes(transcript_text),
            generate_flags_and_scores(transcript_text, emotions_summary, participant_names),
        )

        # Defensive: ensure all results are dicts
        if isinstance(personality_result, list):
            # GPT sometimes returns a list directly instead of {"participants": [...]}
            logger.info(f"[{video_id[:8]}] personality_result is list ({len(personality_result)} items), wrapping")
            personality_result = {"participants": personality_result}
        elif not isinstance(personality_result, dict):
            logger.warning(f"[{video_id[:8]}] personality_result is {type(personality_result)}, using fallback")
            personality_result = {"participants": []}
        if not isinstance(correlations_result, dict):
            logger.warning(f"[{video_id[:8]}] correlations_result is {type(correlations_result)}, using fallback")
            correlations_result = {"correlations": []}
        if not isinstance(summary_result, dict):
            logger.warning(f"[{video_id[:8]}] summary_result is {type(summary_result)}, using fallback")
            summary_result = {"summary": "", "key_topics": [], "overall_sentiment": "neutral"}
        if not isinstance(notes_result, dict):
            logger.warning(f"[{video_id[:8]}] notes_result is {type(notes_result)}, using fallback")
            notes_result = {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}
        if not isinstance(scores_result, dict):
            logger.warning(f"[{video_id[:8]}] scores_result is {type(scores_result)}, using fallback")
            scores_result = {"flags": [], "participant_scores": []}

        _mark_stage_done(video_id, "profiles")
        await _update_progress(video_id, "analyzing", 82, "Personality, correlations, summary, and notes complete.")

        # Mark summary stage active
        _mark_stage_active(video_id, "summary")

        # ---- Phase 6: Persist everything to SQLite (85-93%) ----
        await _update_progress(video_id, "analyzing", 85, "Saving results to database...")

        async with aiosqlite.connect(DB_PATH) as db:
            # Clear old analysis data for re-analysis support
            # (participants, emotions, signal_snapshots, voice_signals, word_signals
            #  are already saved earlier for live display)
            for table in [
                "flags", "coaching", "summaries", "meeting_notes",
                "personality_signals", "correlations", "meeting_analytics",
            ]:
                await db.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))

            # --- Compute ACTUAL speaking percentages from transcript using TIME DURATION ---
            speaker_speaking_time: dict[str, float] = {}
            unmatched_time = 0.0
            unmatched_speakers: dict[str, float] = {}  # Track unmatched for debugging
            for seg in transcript_segments:
                speaker = seg.get("speaker", "")
                seg_duration = max(0, seg.get("end_time", 0) - seg.get("start_time", 0))
                if seg_duration <= 0:
                    # Fallback: estimate from word count (~2.5 words/sec)
                    seg_duration = len(seg.get("text", "").split()) / 2.5
                if speaker:
                    matched_name = name_lookup.get(_normalize_name(speaker)) or _fuzzy_match_name(speaker, participant_names)
                    if matched_name:
                        speaker_speaking_time[matched_name] = speaker_speaking_time.get(matched_name, 0) + seg_duration
                    else:
                        unmatched_time += seg_duration
                        unmatched_speakers[speaker] = unmatched_speakers.get(speaker, 0) + seg_duration
                else:
                    unmatched_time += seg_duration

            if unmatched_speakers:
                logger.warning(f"[{video_id[:8]}] Unmatched transcript speakers: {unmatched_speakers}")
                # Second pass: try harder to match unmatched speakers
                # For each unmatched speaker, check if it's a first/last name
                # of any participant that hasn't been matched yet
                still_unmatched_time = 0.0
                already_matched = set(speaker_speaking_time.keys())
                for spk_label, spk_time in list(unmatched_speakers.items()):
                    spk_parts = _split_name_parts(spk_label)
                    best_candidate = None
                    best_overlap = 0
                    for name in participant_names:
                        name_parts = _split_name_parts(name)
                        # Check if any word in the speaker label matches any part of this name
                        overlap = 0
                        for sp in spk_parts:
                            for np in name_parts:
                                if sp == np:
                                    overlap += 1
                                elif len(sp) >= 4 and len(np) >= 4 and _edit_distance(sp, np) <= 2:
                                    overlap += 0.7
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_candidate = name
                    if best_candidate and best_overlap >= 0.7:
                        speaker_speaking_time[best_candidate] = speaker_speaking_time.get(best_candidate, 0) + spk_time
                        unmatched_time -= spk_time
                        logger.info(f"[{video_id[:8]}] Second-pass match: '{spk_label}' -> '{best_candidate}'")
                    else:
                        still_unmatched_time += spk_time
                if still_unmatched_time != unmatched_time:
                    unmatched_time = still_unmatched_time

            # Distribute unmatched time proportionally among matched speakers
            if unmatched_time > 0 and speaker_speaking_time:
                total_matched = sum(speaker_speaking_time.values())
                for name in speaker_speaking_time:
                    share = speaker_speaking_time[name] / max(total_matched, 1)
                    speaker_speaking_time[name] += unmatched_time * share
            elif unmatched_time > 0 and not speaker_speaking_time:
                # No speaker matched at all — distribute equally
                per_person = max(0.1, unmatched_time / max(len(participant_names), 1))
                for name in participant_names:
                    speaker_speaking_time[name] = per_person

            logger.info(f"[{video_id[:8]}] Speaker speaking time (sec): {speaker_speaking_time} (unmatched: {unmatched_time:.1f}s)")
            total_time = max(sum(speaker_speaking_time.values()), 0.1)

            # Normalize to exactly 100%
            raw_pcts = {name: (t / total_time) * 100 for name, t in speaker_speaking_time.items()}
            # Ensure all participants are included (even those who didn't speak)
            for name in participant_names:
                if name not in raw_pcts:
                    raw_pcts[name] = 0

            # Give every participant a minimum floor of 1% —
            # in a real meeting everyone participates at least minimally
            # and diarization often misses short utterances.
            MIN_SPEAKING_PCT = 1.0
            zero_pct_names = [n for n in raw_pcts if raw_pcts[n] < MIN_SPEAKING_PCT]
            if zero_pct_names and len(zero_pct_names) < len(participant_names):
                # Boost each 0% participant to MIN_SPEAKING_PCT
                boost_total = sum(max(0, MIN_SPEAKING_PCT - raw_pcts[n]) for n in zero_pct_names)
                for n in zero_pct_names:
                    raw_pcts[n] = MIN_SPEAKING_PCT
                # Subtract the boost proportionally from speakers who are above minimum
                speakers_above = {n: raw_pcts[n] for n in raw_pcts if n not in zero_pct_names}
                speaker_total = sum(speakers_above.values())
                if speaker_total > boost_total:
                    for n in speakers_above:
                        raw_pcts[n] -= boost_total * (speakers_above[n] / speaker_total)

            # Round and adjust to sum to exactly 100
            rounded_pcts = {name: max(1, round(pct)) for name, pct in raw_pcts.items()}
            total_rounded = sum(rounded_pcts.values())
            if total_rounded != 100 and total_rounded > 0:
                # Add/subtract difference to the largest speaker
                diff = 100 - total_rounded
                largest = max(rounded_pcts, key=lambda k: rounded_pcts[k])
                rounded_pcts[largest] = max(1, rounded_pcts[largest] + diff)

            actual_speaking_pct = rounded_pcts
            logger.info(f"[{video_id[:8]}] Speaking distribution (normalized): {actual_speaking_pct}")

            # --- Update participants with final scores (already saved earlier) ---
            p_scores = scores_result.get("participant_scores", [])

            # Build a lookup from GPT scores by name
            gpt_score_lookup: dict[str, dict] = {}
            for ps_idx, ps in enumerate(p_scores):
                gpt_name = ps.get("name", "")
                matched = _resolve_label_to_name(gpt_name, participant_names, name_lookup)
                if matched:
                    gpt_score_lookup[matched] = ps

            for name in participant_names:
                pid = participant_id_map.get(name, str(uuid.uuid4()))
                gpt = gpt_score_lookup.get(name, {})

                # Compute avg engagement from saved emotion data (most reliable source)
                row = await db.execute(
                    """SELECT AVG(engagement) FROM emotions
                       WHERE video_id = ? AND participant_id = ?
                       AND engagement > 0""",
                    (video_id, pid),
                )
                db_avg_row = await row.fetchone()
                db_avg = db_avg_row[0] if db_avg_row else None

                # --- Compute final engagement: blend frame-based + speaking + text ---
                # 1) Frame-based engagement (visual + audio + text per-frame)
                if name in participant_engagement and participant_engagement[name]:
                    frame_eng = sum(participant_engagement[name]) / len(participant_engagement[name])
                elif db_avg is not None and db_avg > 0:
                    frame_eng = db_avg
                else:
                    frame_eng = max(gpt.get("engagement", 50), 30)

                # 2) Speaking contribution score (non-linear: any speaking is valuable)
                spk_pct = actual_speaking_pct.get(name, 0)
                if spk_pct >= 40:
                    speak_score = 90.0
                elif spk_pct >= 15:
                    speak_score = 65.0 + (spk_pct - 15) * (25.0 / 25)
                elif spk_pct >= 5:
                    speak_score = 50.0 + (spk_pct - 5) * (15.0 / 10)
                elif spk_pct >= 2:
                    speak_score = 35.0 + (spk_pct - 2) * (15.0 / 3)
                else:
                    speak_score = 20.0 + spk_pct * 15.0  # 1% → 35

                # 3) Text/GPT engagement estimate
                text_eng = max(gpt.get("engagement", 50), 30)

                # Weighted blend: 40% speaking, 35% frame-based (visual+audio), 25% text
                avg_eng = 0.40 * speak_score + 0.35 * frame_eng + 0.25 * text_eng
                avg_eng = max(15.0, min(100.0, avg_eng))  # Clamp

                # Convert sentiment from -1..1 scale to 0..100 for display
                raw_sentiment = gpt.get("sentiment", 0.0)
                if isinstance(raw_sentiment, (int, float)):
                    # GPT returns -1 to 1; convert to 0-100 percentage
                    sentiment_pct = round(((raw_sentiment + 1) / 2) * 100, 1)
                else:
                    sentiment_pct = 50.0  # neutral fallback

                await db.execute(
                    """UPDATE participants SET
                       engagement_score = ?, sentiment_score = ?,
                       speaking_pct = ?, clarity_score = ?, rapport_score = ?, energy_score = ?
                       WHERE id = ? AND video_id = ?""",
                    (
                        round(avg_eng, 1),
                        sentiment_pct,
                        spk_pct,
                        gpt.get("clarity", 50),
                        gpt.get("rapport", 50),
                        gpt.get("energy", 50),
                        pid, video_id,
                    ),
                )

            await db.commit()

        # Save personality, correlations, flags, summary
        async with aiosqlite.connect(DB_PATH) as db:
            # --- Store personality signals ---
            name_lookup = _build_name_lookup(participant_names)
            for pp_idx, pp in enumerate(personality_result.get("participants", [])):
                label = pp.get("label", f"Person {pp_idx + 1}")
                name = _resolve_label_to_name(label, participant_names, name_lookup)
                if not name:
                    name = participant_names[pp_idx] if pp_idx < len(participant_names) else participant_names[0] if participant_names else None
                pid = participant_id_map.get(name) if name else None
                if not pid:
                    pid = list(participant_id_map.values())[0] if participant_id_map else "unknown"
                await db.execute(
                    """INSERT INTO personality_signals
                       (video_id, participant_id, traits, communication_style, confidence_level)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        video_id, pid,
                        json.dumps(pp.get("traits", [])),
                        pp.get("communication_style", ""),
                        pp.get("confidence_level", ""),
                    ),
                )

            # --- Store correlations ---
            for corr in correlations_result.get("correlations", []):
                await db.execute(
                    """INSERT INTO correlations
                       (video_id, signal_pair, pattern, confidence, implication)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        video_id,
                        json.dumps(corr.get("signal_pair", [])),
                        corr.get("pattern", ""),
                        corr.get("confidence", 0),
                        corr.get("implication", ""),
                    ),
                )

            # --- Store flags (from scores call) ---
            raw_flags = scores_result.get("flags", [])
            if not isinstance(raw_flags, list):
                raw_flags = []
            for flag in raw_flags:
                if not isinstance(flag, dict):
                    continue
                await db.execute(
                    """INSERT INTO flags
                       (video_id, timestamp, type, description, severity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        video_id,
                        flag.get("timestamp", 0),
                        flag.get("type", "neutral"),
                        flag.get("description", ""),
                        flag.get("severity", "medium"),
                    ),
                )

            # --- Store engagement alerts as flags ---
            for alert in engagement_alerts:
                await db.execute(
                    """INSERT INTO flags (video_id, timestamp, type, description, severity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (video_id, alert.timestamp, "engagement_alert",
                     alert.suggested_action, "high"),
                )

            # --- Store summary (from dedicated call) ---
            await db.execute(
                """INSERT OR REPLACE INTO summaries
                   (video_id, summary, key_topics, overall_sentiment)
                   VALUES (?, ?, ?, ?)""",
                (
                    video_id,
                    summary_result.get("summary", ""),
                    json.dumps(summary_result.get("key_topics", [])),
                    summary_result.get("overall_sentiment", "neutral"),
                ),
            )

            # --- Store meeting notes ---
            await db.execute(
                """INSERT OR REPLACE INTO meeting_notes
                   (video_id, action_items, decisions, follow_ups, key_questions)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    video_id,
                    json.dumps(notes_result.get("action_items", [])),
                    json.dumps(notes_result.get("decisions", [])),
                    json.dumps(notes_result.get("follow_ups", [])),
                    json.dumps(notes_result.get("key_questions", [])),
                ),
            )

            # --- Store meeting analytics ---
            # Use FINAL blended participant engagement scores (not raw frame scores)
            cursor = await db.execute(
                "SELECT AVG(engagement_score) FROM participants WHERE video_id = ? AND engagement_score > 0",
                (video_id,),
            )
            row = await cursor.fetchone()
            avg_engagement = row[0] if row and row[0] else 0

            avg_visual = 0
            avg_audio = 0
            avg_verbal = 0

            if frame_results:
                visual_scores = []
                for fr in frame_results:
                    for p in fr.get("participants", []):
                        visual_scores.append(p.get("emotions", {}).get("engagement", 50))
                if visual_scores:
                    avg_visual = sum(visual_scores) / len(visual_scores)

            if audio_features:
                avg_audio = sum(a["energy"] for a in audio_features) / len(audio_features)

            if word_signals:
                avg_verbal = sum(w["sentiment_score"] * 100 for w in word_signals) / len(word_signals)

            await db.execute(
                """INSERT OR REPLACE INTO meeting_analytics
                   (video_id, avg_engagement, avg_visual_engagement, avg_audio_engagement,
                    avg_verbal_engagement, participant_count, duration, overall_sentiment, meeting_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    video_id,
                    round(avg_engagement, 1),
                    round(avg_visual, 1),
                    round(avg_audio, 1),
                    round(avg_verbal, 1),
                    len(participant_id_map),
                    duration,
                    summary_result.get("overall_sentiment", "neutral"),
                ),
            )

            # --- Update video status ---
            overall_score = avg_engagement if avg_engagement > 0 else 0
            await db.execute(
                """UPDATE videos
                   SET duration = ?, overall_score = ?, status = 'complete'
                   WHERE id = ?""",
                (duration, round(overall_score, 1), video_id),
            )

            await db.commit()

        # ---- Phase 7: Generate coaching for all participants CONCURRENTLY (93-98%) ----
        await _update_progress(video_id, "analyzing", 93, "Generating coaching recommendations...")

        _coaching_name_lookup = _build_name_lookup(participant_names)

        async def _generate_and_save_coaching(name: str, pid: str):
            """Generate coaching for one participant and save to DB."""
            p_emotions = [
                fr for fr in frame_results
                if any(
                    _resolve_label_to_name(p.get("label", f"Person {p_idx + 1}"), participant_names, _coaching_name_lookup) == name
                    for p_idx, p in enumerate(fr.get("participants", []))
                )
            ]
            p_emo_summary = json.dumps(p_emotions, indent=1)

            coaching_items = await generate_coaching(
                transcript_text, p_emo_summary, name
            )
            if not isinstance(coaching_items, list):
                coaching_items = []
            async with aiosqlite.connect(DB_PATH) as db:
                for item in coaching_items:
                    if not isinstance(item, dict):
                        continue
                    await db.execute(
                        """INSERT INTO coaching
                           (video_id, participant_id, recommendation, category, priority)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            video_id, pid,
                            item.get("recommendation", ""),
                            item.get("category", "general"),
                            item.get("priority", 3),
                        ),
                    )
                await db.commit()

        # Generate coaching for ALL participants concurrently
        coaching_tasks = [
            _generate_and_save_coaching(name, pid)
            for name, pid in participant_id_map.items()
        ]
        await asyncio.gather(*coaching_tasks, return_exceptions=True)

        _mark_stage_done(video_id, "summary")

        # ---- Mark complete ----
        await _emit_event(video_id, {"type": "complete"})
        _max_progress.pop(video_id, None)  # Cleanup memory
        _active_stages.pop(video_id, None)  # Cleanup stage tracking

        await _update_progress(
            video_id, "complete", 100,
            f"Analysis complete: {len(participant_id_map)} participants, "
            f"{len(frame_results)} frames, {len(voice_signals)} voice segments analyzed",
        )
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET wizard_step = 'complete' WHERE id = ?",
                (video_id,),
            )
            await db.commit()

        logger.info(f"[{video_id[:8]}] Analysis complete")

    except Exception as e:
        logger.exception(f"[{video_id[:8]}] Analysis failed: {e}")
        await _emit_event(video_id, {"type": "error", "detail": str(e)})
        _max_progress.pop(video_id, None)  # Cleanup memory
        _active_stages.pop(video_id, None)  # Cleanup stage tracking
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """UPDATE videos
                   SET wizard_step = 'error',
                       progress_detail = ?,
                       status = ?
                   WHERE id = ?""",
                (f"Analysis error: {str(e)[:300]}", f"error: {str(e)[:200]}", video_id),
            )
            await db.commit()
