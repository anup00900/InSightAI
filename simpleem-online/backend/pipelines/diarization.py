"""Speaker diarization pipeline using pyannote.audio.

Identifies who spoke when in an audio file. Results are used to
assign speaker labels to Whisper transcript segments.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load pyannote pipeline (cached -- only loads once)."""
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HUGGINGFACE_TOKEN or None,
        )
        logger.info("pyannote diarization pipeline loaded")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}")
        return None


async def diarize_audio(audio_path: str) -> List[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Returns a list of segments with speaker labels.
    Runs in a thread pool since pyannote is CPU-bound.
    """
    pipeline = _load_pipeline()
    if pipeline is None:
        logger.warning("Diarization skipped -- pipeline not available")
        return []

    def _run():
        result = pipeline(audio_path)
        # pyannote 4.x returns DiarizeOutput; extract the Annotation object
        annotation = getattr(result, "speaker_diarization", result)
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
        return segments

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def assign_speakers_to_transcript(
    transcript_segments: List[dict],
    diarization_segments: List[DiarizationSegment],
) -> List[dict]:
    """Assign speaker labels from diarization to transcript segments.

    Uses maximum overlap to determine which speaker is most likely for
    each transcript segment.
    """
    if not diarization_segments:
        return transcript_segments

    # Build speaker label mapping (SPEAKER_00 -> Speaker 1, etc.)
    unique_speakers = sorted(set(d.speaker for d in diarization_segments))
    speaker_map = {s: f"Speaker {i + 1}" for i, s in enumerate(unique_speakers)}

    for seg in transcript_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # Find diarization segment(s) overlapping this transcript segment
        best_speaker = ""
        best_overlap = 0

        for d in diarization_segments:
            overlap_start = max(seg_start, d.start)
            overlap_end = min(seg_end, d.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.speaker

        if best_speaker:
            seg["speaker"] = speaker_map.get(best_speaker, best_speaker)

    return transcript_segments
