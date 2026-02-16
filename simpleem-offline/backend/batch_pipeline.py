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
    analyze_voice_signal,
    analyze_words_signal,
    analyze_personality,
    build_correlations,
    generate_coaching,
    generate_summary_and_flags,
    extract_names_from_frame,
    assign_speakers_to_names,
)
from .pipelines.audio import AudioAnalyzer
from .engagement_engine import EngagementEngine, EngagementAlertDetector, EngagementScore

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
FRAME_INTERVAL = 5.0       # Analyze a frame every 5 seconds (was 3)
AUDIO_CHUNK_SEC = 10.0      # Audio feature extraction in 10-second chunks
FRAME_BATCH_SIZE = 10       # Parallel frame analysis batch size (was 5)
NAME_EXTRACT_COUNT = 5      # Number of frames to sample for name extraction


# ---------------------------------------------------------------------------
# Helper: update progress in DB
# ---------------------------------------------------------------------------

async def _update_progress(
    video_id: str,
    wizard_step: str,
    progress: float,
    detail: str,
):
    """Write analysis progress to the videos table for frontend polling."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE videos
               SET wizard_step = ?, analyze_progress = ?, progress_detail = ?
               WHERE id = ?""",
            (wizard_step, round(progress, 2), detail, video_id),
        )
        await db.commit()


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

    # All participants found in at least 1 frame
    all_names = [info["name"] for info in name_camera_counts.values()]
    # Camera-on = found with camera on in majority of their appearances
    camera_on_names = [
        info["name"]
        for info in name_camera_counts.values()
        if info["camera_on_count"] > info["count"] / 2
    ]

    return all_names, camera_on_names


def _normalize_name(name: str) -> str:
    """Normalize a name for deduplication (lowercase, strip whitespace/punctuation)."""
    import re
    name = name.lower().strip()
    # Remove trailing ellipsis or dots from truncated names
    name = re.sub(r'[.\u2026]+$', '', name).strip()
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name


def _build_name_lookup(canonical_names: list[str]) -> dict[str, str]:
    """Build a normalized-name -> canonical-name lookup."""
    lookup = {}
    for name in canonical_names:
        lookup[_normalize_name(name)] = name
    return lookup


def _fuzzy_match_name(label: str, canonical_names: list[str]) -> str | None:
    """Try to fuzzy-match a frame label to a canonical name.

    Handles cases like 'Person 1' labels or slight name variations.
    Returns the matched canonical name, or None.
    """
    if not canonical_names:
        return None

    label_lower = label.lower().strip()

    # Exact match
    for name in canonical_names:
        if name.lower() == label_lower:
            return name

    # Substring match (label contains name or name contains label)
    for name in canonical_names:
        if name.lower() in label_lower or label_lower in name.lower():
            return name

    # First/last name match
    label_parts = set(label_lower.split())
    best_match = None
    best_overlap = 0
    for name in canonical_names:
        name_parts = set(name.lower().split())
        overlap = len(label_parts & name_parts)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = name

    if best_overlap > 0:
        return best_match

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
            analyze_tasks.append(analyze_frame_all_signals(fp, ts, known_names, camera_on_names))
            valid_frames.append((fp, ts))

        if analyze_tasks:
            analyses = await asyncio.gather(*analyze_tasks, return_exceptions=True)
            for j, (analysis, (fp, ts)) in enumerate(zip(analyses, valid_frames)):
                if isinstance(analysis, Exception):
                    logger.warning(f"Frame analysis failed at {ts:.1f}s: {analysis}")
                    analysis = {
                        "participants": [{
                            "label": "Person 1",
                            "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
                            "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "unable to analyze"},
                            "gestures": ["still posture", "resting hands"],
                            "reactions": ["neutral gaze", "steady expression"],
                        }]
                    }
                analysis["timestamp"] = ts
                results.append(analysis)

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

        t += AUDIO_CHUNK_SEC

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
    CHUNK_DURATION = 30.0
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
            result = await analyze_voice_signal(txt[:1500])
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
            result = await analyze_words_signal(txt[:1500], timestamp)
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
    TEXT_ANALYSIS_SEM = asyncio.Semaphore(5)

    async def _analyze_chunk_pair(ts: float, text: str):
        """Analyze both voice and word signals for a chunk, with concurrency limit."""
        if not text.strip():
            return None, None
        async with TEXT_ANALYSIS_SEM:
            voice_res, word_res = await asyncio.gather(
                _analyze_voice(text, ts),
                _analyze_words(text, ts),
            )
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
            for p in fr.get("participants", []):
                label = p.get("label", "Person 1")
                name = name_lookup.get(_normalize_name(label))
                if not name:
                    name = _fuzzy_match_name(label, participant_names)
                pid = participant_id_map.get(name) if name else None
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
                await db.execute(
                    """INSERT INTO signal_snapshots
                       (video_id, participant_id, timestamp, emotion, emotion_confidence,
                        engagement, posture, openness, leaning, mirroring,
                        body_language_desc, gestures, reactions)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        video_id, pid, ts,
                        emotions_data.get("primary", "neutral"),
                        emotions_data.get("confidence", 0.5),
                        emotions_data.get("engagement", 50),
                        bl.get("posture", "upright"),
                        bl.get("openness", "mixed"),
                        bl.get("leaning", "neutral"),
                        1 if bl.get("mirroring") else 0,
                        bl.get("description", ""),
                        json.dumps(p.get("gestures", [])),
                        json.dumps(p.get("reactions", [])),
                    ),
                )

        await db.commit()
    logger.info(f"[{video_id[:8]}] Saved {len(frame_results)} frame results to DB")


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

        # ---- Phase 2: Run frame + audio + text analysis CONCURRENTLY (10-60%) ----
        await _update_progress(video_id, "analyzing", 10, "Running visual, audio, and text analysis concurrently...")

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

        await _update_progress(video_id, "analyzing", 60, "All signal analysis complete. Processing results...")

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

        # Compute engagement scores
        from .signal_bus import SignalEvent, SignalType
        engine = EngagementEngine()
        participant_engagement: dict[str, list[float]] = {name: [] for name in participant_names}

        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            for p in fr.get("participants", []):
                label = p.get("label", "Person 1")
                name = name_lookup.get(_normalize_name(label), label)
                if name not in participant_engagement:
                    # Try partial match
                    name = _fuzzy_match_name(label, participant_names)
                    if name is None:
                        continue

                visual_engagement = p.get("emotions", {}).get("engagement", 50)
                visual_signal = SignalEvent(
                    signal_type=SignalType.VISUAL,
                    participant_id=name,
                    timestamp=ts,
                    data={"engagement": visual_engagement},
                )

                audio_signal = None
                if audio_features:
                    closest_audio = min(audio_features, key=lambda a: abs(a["timestamp"] - ts))
                    if abs(closest_audio["timestamp"] - ts) < AUDIO_CHUNK_SEC:
                        audio_signal = SignalEvent(
                            signal_type=SignalType.AUDIO,
                            participant_id=name,
                            timestamp=ts,
                            data={"energy": closest_audio["energy"]},
                        )

                text_signal = None
                if word_signals:
                    closest_word = min(word_signals, key=lambda w: abs(w["timestamp"] - ts))
                    if abs(closest_word["timestamp"] - ts) < 60:
                        text_signal = SignalEvent(
                            signal_type=SignalType.TEXT,
                            participant_id=name,
                            timestamp=ts,
                            data={"sentiment_score": closest_word["sentiment_score"]},
                        )

                score = engine.compute(name, ts, visual_signal, audio_signal, text_signal)
                participant_engagement[name].append(score.holistic)

        # Run engagement alert detection
        alert_detector = EngagementAlertDetector(threshold=40, min_duration=30)
        engagement_alerts = []

        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            for p in fr.get("participants", []):
                label = p.get("label", "Person 1")
                name = name_lookup.get(_normalize_name(label)) or _fuzzy_match_name(label, participant_names)
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

        # ---- Phase 4: Personality + correlations (65-75%) ----
        await _update_progress(video_id, "analyzing", 65, "Analyzing personality traits and correlations...")

        transcript_text = "\n".join(
            f"[{s['start_time']:.1f}s - {s['end_time']:.1f}s] {s.get('speaker', '')}: {s['text']}"
            for s in transcript_segments
        )

        signals_summary = json.dumps({
            "frame_analyses": frame_results[:20],
            "audio_features": audio_features[:10],
            "voice_signals": voice_signals[:10],
        }, indent=1)[:4000]

        all_signals_text = json.dumps({
            "visual": frame_results[:15],
            "audio": audio_features[:8],
            "voice": voice_signals[:8],
            "words": word_signals[:8],
        }, indent=1)[:4000]

        emotions_summary = json.dumps(
            [
                {
                    "timestamp": fr.get("timestamp", 0),
                    "participants": fr.get("participants", []),
                }
                for fr in frame_results[:30]
            ],
            indent=1,
        )

        # Run personality + correlations + summary/flags ALL concurrently (3-way gather)
        personality_result, correlations_result, analysis = await asyncio.gather(
            analyze_personality(signals_summary, transcript_text[:3000]),
            build_correlations(all_signals_text, transcript_text[:2000]),
            generate_summary_and_flags(transcript_text[:8000], emotions_summary[:4000], participant_names),
        )

        # ---- Phase 6: Persist everything to SQLite (85-93%) ----
        await _update_progress(video_id, "analyzing", 85, "Saving analysis results to database...")

        async with aiosqlite.connect(DB_PATH) as db:
            # Clear old analysis data for re-analysis support
            for table in [
                "emotions", "flags", "coaching", "summaries",
                "signal_snapshots", "voice_signals", "word_signals",
                "personality_signals", "correlations", "meeting_analytics",
                "participants",
            ]:
                await db.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))

            # --- Store participants ---
            participant_id_map: dict[str, str] = {}
            p_scores = analysis.get("participant_scores", [])

            scored_names = {ps.get("name") for ps in p_scores}
            for name in participant_names:
                if name not in scored_names:
                    p_scores.append({
                        "name": name, "engagement": 60, "sentiment": 0.3,
                        "speaking_pct": round(100 / max(len(participant_names), 1)),
                        "clarity": 65, "rapport": 65, "energy": 65,
                    })

            for ps in p_scores:
                pid = str(uuid.uuid4())
                name = ps.get("name", "Unknown")
                participant_id_map[name] = pid

                avg_eng = ps.get("engagement", 70)
                if name in participant_engagement and participant_engagement[name]:
                    avg_eng = sum(participant_engagement[name]) / len(participant_engagement[name])

                await db.execute(
                    """INSERT INTO participants
                       (id, video_id, name, engagement_score, sentiment_score,
                        speaking_pct, clarity_score, rapport_score, energy_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid, video_id, name,
                        round(avg_eng, 1),
                        ps.get("sentiment", 0.3),
                        ps.get("speaking_pct", 25),
                        ps.get("clarity", 70),
                        ps.get("rapport", 70),
                        ps.get("energy", 70),
                    ),
                )

            await db.commit()

        # Save frame results (emotions + signal_snapshots)
        await _save_frame_results_to_db(video_id, frame_results, participant_names, participant_id_map)

        # Save text results (voice + word signals)
        await _save_text_results_to_db(video_id, voice_signals, word_signals)

        # Save personality, correlations, flags, summary
        async with aiosqlite.connect(DB_PATH) as db:
            # --- Store personality signals ---
            name_lookup = _build_name_lookup(participant_names)
            for pp in personality_result.get("participants", []):
                label = pp.get("label", "Person 1")
                name = name_lookup.get(_normalize_name(label)) or _fuzzy_match_name(label, participant_names)
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

            # --- Store flags ---
            for flag in analysis.get("flags", []):
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

            # --- Store summary ---
            await db.execute(
                """INSERT OR REPLACE INTO summaries
                   (video_id, summary, key_topics, overall_sentiment)
                   VALUES (?, ?, ?, ?)""",
                (
                    video_id,
                    analysis.get("summary", ""),
                    json.dumps(analysis.get("key_topics", [])),
                    analysis.get("overall_sentiment", "neutral"),
                ),
            )

            # --- Store meeting analytics ---
            avg_engagement = 0
            avg_visual = 0
            avg_audio = 0
            avg_verbal = 0

            all_engagement_scores = []
            for name, scores in participant_engagement.items():
                all_engagement_scores.extend(scores)
            if all_engagement_scores:
                avg_engagement = sum(all_engagement_scores) / len(all_engagement_scores)

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
                    analysis.get("overall_sentiment", "neutral"),
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
                    (_coaching_name_lookup.get(_normalize_name(p.get("label", "")))
                     or _fuzzy_match_name(p.get("label", ""), participant_names)) == name
                    for p in fr.get("participants", [])
                )
            ]
            p_emo_summary = json.dumps(p_emotions[:15], indent=1)

            coaching_items = await generate_coaching(
                transcript_text[:4000], p_emo_summary[:2000], name
            )
            async with aiosqlite.connect(DB_PATH) as db:
                for item in coaching_items:
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

        # ---- Mark complete ----
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
