"""API-based transcription engine using Core42 Whisper API.

Extracts full audio via FFmpeg, splits into chunks for the API size limit,
transcribes with Core42 Whisper, optionally diarizes speakers, and stores
everything in the DB.  Progress is updated for frontend polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

import aiosqlite

from .models import DB_PATH
from .frame_extractor import extract_full_audio, extract_audio_chunk
from .core42_client import transcribe_audio
from .pipelines.diarization import diarize_audio, assign_speakers_to_transcript

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")

# Whisper API accepts up to 25MB.  10-minute WAV chunks stay well under.
CHUNK_DURATION_SECONDS = 600  # 10 minutes per chunk


# ---------------------------------------------------------------------------
# Helper: update progress in DB
# ---------------------------------------------------------------------------

async def _update_progress(
    video_id: str,
    wizard_step: str,
    progress: float,
    detail: str,
):
    """Write transcription progress to the videos table for frontend polling."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE videos
               SET wizard_step = ?, transcribe_progress = ?, progress_detail = ?
               WHERE id = ?""",
            (wizard_step, round(progress, 2), detail, video_id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Transcribe audio via Core42 Whisper API (chunked for long files)
# ---------------------------------------------------------------------------

async def _get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", audio_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    import json
    try:
        info = json.loads(stdout)
        return float(info.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


async def _transcribe_via_api(
    video_id: str, audio_path: str, output_dir: str
) -> List[dict]:
    """Transcribe audio using the Core42 Whisper API.

    Splits long audio into chunks to stay within the API file size limit.
    Returns a list of segment dicts with keys: start, end, text.
    """
    duration = await _get_audio_duration(audio_path)
    if duration <= 0:
        logger.warning(f"[{video_id[:8]}] Could not determine audio duration")
        return []

    # If short enough, send the whole file
    if duration <= CHUNK_DURATION_SECONDS:
        result = await transcribe_audio(audio_path)
        short_segments = _parse_whisper_response(result, offset=0)
        # Save segments incrementally for the single-file case
        if short_segments:
            async with aiosqlite.connect(DB_PATH) as db:
                for seg in short_segments:
                    await db.execute(
                        """INSERT INTO transcript_segments
                           (video_id, start_time, end_time, text, speaker)
                           VALUES (?, ?, ?, ?, ?)""",
                        (video_id, seg.get("start", 0), seg.get("end", 0),
                         seg.get("text", ""), seg.get("speaker", "")),
                    )
                await db.commit()
        return short_segments

    # Split into chunks — process up to 3 concurrently for speed
    MAX_CONCURRENT_CHUNKS = 3
    segments = []
    chunk_starts = list(range(0, int(duration), CHUNK_DURATION_SECONDS))
    total_chunks = len(chunk_starts)

    for batch_start in range(0, len(chunk_starts), MAX_CONCURRENT_CHUNKS):
        batch = chunk_starts[batch_start:batch_start + MAX_CONCURRENT_CHUNKS]
        pct = 15 + int((batch_start / total_chunks) * 45)  # 15% -> 60%
        await _update_progress(
            video_id, "transcribing", pct,
            f"Transcribing chunks {batch_start + 1}-{min(batch_start + len(batch), total_chunks)}/{total_chunks} ({len(segments)} segments so far)"
        )

        # Extract all chunks in this batch
        chunk_info = []
        for start in batch:
            end = min(start + CHUNK_DURATION_SECONDS, duration)
            chunk_path = await extract_audio_chunk(audio_path, start, end, output_dir)
            chunk_info.append((start, chunk_path))

        # Transcribe all chunks in this batch concurrently
        transcribe_tasks = [transcribe_audio(cp) for _, cp in chunk_info]
        results = await asyncio.gather(*transcribe_tasks, return_exceptions=True)

        chunk_segments_batch = []
        for (start, chunk_path), result in zip(chunk_info, results):
            if isinstance(result, Exception):
                logger.error(f"[{video_id[:8]}] Chunk at {start}s transcription failed: {result}")
            else:
                chunk_segments = _parse_whisper_response(result, offset=start)
                chunk_segments_batch.extend(chunk_segments)
                segments.extend(chunk_segments)
            # Clean up chunk file
            try:
                os.remove(chunk_path)
            except OSError:
                pass

        # Save new segments from this batch incrementally
        if chunk_segments_batch:
            async with aiosqlite.connect(DB_PATH) as db:
                for seg in chunk_segments_batch:
                    await db.execute(
                        """INSERT INTO transcript_segments
                           (video_id, start_time, end_time, text, speaker)
                           VALUES (?, ?, ?, ?, ?)""",
                        (video_id, seg.get("start", 0), seg.get("end", 0),
                         seg.get("text", ""), seg.get("speaker", "")),
                    )
                await db.commit()

    return segments


def _parse_whisper_response(result: dict, offset: float) -> List[dict]:
    """Parse the Whisper API verbose_json response into segment dicts."""
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": seg.get("start", 0) + offset,
            "end": seg.get("end", 0) + offset,
            "text": text,
        })

    # Fallback: if no segments but there's text, create a single segment
    if not segments and result.get("text", "").strip():
        segments.append({
            "start": offset,
            "end": offset + 30,  # approximate
            "text": result["text"].strip(),
        })

    return segments


def _post_process_segments(segments: List[dict]) -> List[dict]:
    """Clean up transcription artifacts for better quality."""
    if not segments:
        return segments

    cleaned = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Remove repeated phrases (common Whisper artifact)
        # e.g., "Thank you. Thank you. Thank you." -> "Thank you."
        import re
        text = re.sub(r'(\b\w+(?:\s+\w+){0,3})\s*(?:\.\s*\1){2,}', r'\1.', text)

        # Remove leading/trailing artifacts
        text = text.strip(' -—')

        if not text:
            continue

        seg["text"] = text
        cleaned.append(seg)

    # Merge very short segments (< 0.5s) into adjacent ones
    merged = []
    for seg in cleaned:
        duration = seg.get("end", 0) - seg.get("start", 0)
        if merged and duration < 0.5:
            # Merge into previous segment
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg.get("end", merged[-1]["end"])
        else:
            merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_transcription(video_id: str, video_path: str):
    """Full transcription pipeline: extract audio -> Whisper API -> diarize -> store.

    Updates wizard_step and transcribe_progress in the DB so the frontend
    can poll for status.
    """
    output_dir = os.path.join(PROCESSED_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ---- Phase 0: Clear any existing transcript segments ----
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "DELETE FROM transcript_segments WHERE video_id = ?", (video_id,)
            )
            await db.commit()

        # ---- Phase 1: Extract audio ----
        await _update_progress(video_id, "transcribing", 5, "Extracting audio from video...")
        audio_path = await extract_full_audio(video_path, output_dir)

        if audio_path is None:
            await _update_progress(video_id, "transcribed", 100, "No audio track found -- skipping transcription")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE videos SET wizard_step = ? WHERE id = ?",
                    ("transcribed", video_id),
                )
                await db.commit()
            return

        # ---- Phase 2: Transcribe via Core42 Whisper API ----
        await _update_progress(video_id, "transcribing", 15, "Sending audio to Whisper API...")

        segments = await _transcribe_via_api(video_id, audio_path, output_dir)

        segments = _post_process_segments(segments)

        if not segments:
            await _update_progress(video_id, "transcribed", 100, "Transcription complete (no speech detected)")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE videos SET wizard_step = ? WHERE id = ?",
                    ("transcribed", video_id),
                )
                await db.commit()
            return

        await _update_progress(
            video_id, "transcribing", 60,
            f"Transcribed {len(segments)} segments. Running speaker diarization..."
        )

        # ---- Phase 3: Diarize speakers (gracefully skips if pyannote not installed) ----
        diarization_segments = await diarize_audio(audio_path)

        await _update_progress(video_id, "transcribing", 80, "Assigning speakers to transcript segments...")

        # ---- Phase 4: Assign speakers ----
        segments = assign_speakers_to_transcript(segments, diarization_segments)

        # ---- Phase 5: Update speaker assignments and finalize ----
        # Segments were already saved incrementally during transcription.
        # Now update them with speaker assignments from diarization.
        await _update_progress(video_id, "transcribing", 90, "Updating speaker assignments in database...")

        async with aiosqlite.connect(DB_PATH) as db:
            for seg in segments:
                await db.execute(
                    """UPDATE transcript_segments
                       SET speaker = ?
                       WHERE video_id = ? AND start_time = ? AND end_time = ?""",
                    (
                        seg.get("speaker", ""),
                        video_id,
                        seg.get("start", 0),
                        seg.get("end", 0),
                    ),
                )

            await db.execute(
                """UPDATE videos
                   SET wizard_step = 'transcribed',
                       transcribe_progress = 100,
                       progress_detail = ?
                   WHERE id = ?""",
                (f"Transcription complete: {len(segments)} segments, {len(set(s.get('speaker', '') for s in segments))} speakers", video_id),
            )
            await db.commit()

        logger.info(f"[{video_id[:8]}] Transcription complete: {len(segments)} segments")

    except Exception as e:
        logger.exception(f"[{video_id[:8]}] Transcription failed: {e}")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """UPDATE videos
                   SET wizard_step = 'error',
                       progress_detail = ?,
                       status = ?
                   WHERE id = ?""",
                (f"Transcription error: {str(e)[:300]}", f"error: {str(e)[:200]}", video_id),
            )
            await db.commit()
