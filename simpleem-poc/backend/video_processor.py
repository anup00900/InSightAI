from __future__ import annotations

import asyncio
import os
import json
import subprocess


async def has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "a",
        video_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return False
    try:
        info = json.loads(stdout.decode())
        return len(info.get("streams", [])) > 0
    except (json.JSONDecodeError, ValueError):
        return False


async def extract_audio(video_path: str, output_dir: str) -> str | None:
    """Extract audio track from video file using FFmpeg. Returns None if no audio."""
    if not await has_audio_stream(video_path):
        return None

    audio_path = os.path.join(output_dir, "audio.wav")
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


async def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list[tuple[str, float]]:
    """Extract frames from video at specified FPS. Returns list of (frame_path, timestamp)."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-y", os.path.join(frames_dir, "frame_%04d.jpg"),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {stderr.decode()}")

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    )
    frames = []
    for i, fname in enumerate(frame_files):
        timestamp = i / fps
        frames.append((os.path.join(frames_dir, fname), timestamp))

    return frames


async def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using FFprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", video_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return 0.0
    try:
        info = json.loads(stdout.decode())
        return float(info.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def is_video_file(filename: str) -> bool:
    """Check if a file is a supported video format."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    return os.path.splitext(filename)[1].lower() in video_exts


def is_audio_file(filename: str) -> bool:
    """Check if a file is a supported audio format."""
    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
    return os.path.splitext(filename)[1].lower() in audio_exts
