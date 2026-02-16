"""On-demand frame and audio chunk extraction using FFmpeg."""
from __future__ import annotations

import asyncio
import os
import uuid


async def extract_single_frame(video_path: str, timestamp: float, output_dir: str) -> str:
    """Extract a single frame at the given timestamp using FFmpeg -ss seek.

    Returns the path to the extracted JPEG frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_name = f"rt_frame_{timestamp:.2f}_{uuid.uuid4().hex[:8]}.jpg"
    output_path = os.path.join(output_dir, frame_name)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y", output_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Frame extraction failed at {timestamp}s: {stderr.decode()[:200]}")
    return output_path


async def extract_audio_chunk(
    audio_path: str, start: float, end: float, output_dir: str
) -> str:
    """Extract an audio segment from start to end seconds.

    Returns the path to the extracted WAV chunk.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_name = f"chunk_{start:.1f}_{end:.1f}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(output_dir, chunk_name)

    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", audio_path,
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y", output_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Audio chunk extraction failed [{start}-{end}]: {stderr.decode()[:200]}")
    return output_path


async def extract_audio_chunk_from_video(
    video_path: str, start: float, end: float, output_dir: str
) -> str:
    """Extract an audio segment directly from the VIDEO file (no pre-extracted audio needed).

    Returns the path to the extracted WAV chunk.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_name = f"vchunk_{start:.1f}_{end:.1f}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(output_dir, chunk_name)

    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y", output_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Audio chunk from video failed [{start}-{end}]: {stderr.decode()[:200]}")
    return output_path


async def extract_full_audio(video_path: str, output_dir: str) -> str | None:
    """Extract the full audio track from a video file. Returns None if no audio."""
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "rt_audio_full.wav")

    # Check for audio stream first
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "a",
        video_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return None

    import json
    try:
        info = json.loads(stdout.decode())
        if len(info.get("streams", [])) == 0:
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", audio_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        return None
    return audio_path
