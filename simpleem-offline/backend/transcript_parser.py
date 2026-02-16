"""Parse transcript files (VTT, SRT, plain text) into segment dicts."""

from __future__ import annotations
import re
from typing import List


def _parse_timestamp_vtt(ts: str) -> float:
    """Parse VTT timestamp like '00:01:23.456' to seconds."""
    parts = ts.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0


def _parse_timestamp_srt(ts: str) -> float:
    """Parse SRT timestamp like '00:01:23,456' to seconds."""
    return _parse_timestamp_vtt(ts.replace(',', '.'))


def _extract_speaker(text: str) -> tuple[str, str]:
    """Extract speaker label from text prefix. Returns (speaker, cleaned_text)."""
    # Match patterns like "Speaker 1: text" or "John: text" or "<v Speaker>text"
    m = re.match(r'^<v\s+([^>]+)>\s*(.*)', text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r'^([A-Za-z][A-Za-z0-9 _.\'"-]{0,40}):\s+(.*)', text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


def parse_vtt(content: str) -> List[dict]:
    """Parse WebVTT format into segment dicts."""
    segments = []
    lines = content.strip().split('\n')
    i = 0
    # Skip WEBVTT header
    while i < len(lines) and not re.match(r'\d{2}:\d{2}', lines[i]):
        i += 1

    while i < len(lines):
        line = lines[i].strip()
        # Look for timestamp line
        m = re.match(r'(\d{2}:\d{2}[:\.][\d.]+)\s*-->\s*(\d{2}:\d{2}[:\.][\d.]+)', line)
        if m:
            start = _parse_timestamp_vtt(m.group(1))
            end = _parse_timestamp_vtt(m.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = ' '.join(text_lines)
            speaker, text = _extract_speaker(text)
            if text:
                segments.append({"start": start, "end": end, "text": text, "speaker": speaker})
        else:
            i += 1

    return segments


def parse_srt(content: str) -> List[dict]:
    """Parse SRT format into segment dicts."""
    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        # Find timestamp line
        ts_line = None
        text_start = 0
        for j, line in enumerate(lines):
            m = re.match(r'(\d{2}:\d{2}:\d{2}[,.]?\d*)\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]?\d*)', line)
            if m:
                ts_line = m
                text_start = j + 1
                break

        if not ts_line:
            continue

        start = _parse_timestamp_srt(ts_line.group(1))
        end = _parse_timestamp_srt(ts_line.group(2))
        text = ' '.join(l.strip() for l in lines[text_start:] if l.strip())
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        speaker, text = _extract_speaker(text)
        if text:
            segments.append({"start": start, "end": end, "text": text, "speaker": speaker})

    return segments


def parse_plain_text(content: str) -> List[dict]:
    """Parse plain text transcript, with optional timestamps and speaker labels."""
    segments = []
    lines = content.strip().split('\n')
    current_time = 0.0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to extract timestamp: [00:01] or [00:01:23] or (00:01)
        ts_match = re.match(r'[\[\(](\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)[\]\)]\s*(.*)', line)
        if ts_match:
            current_time = _parse_timestamp_vtt(ts_match.group(1))
            line = ts_match.group(2)

        speaker, text = _extract_speaker(line)
        if text:
            segments.append({
                "start": current_time,
                "end": current_time + 5.0,  # approximate 5s per segment
                "text": text,
                "speaker": speaker,
            })
            current_time += 5.0

    return segments


def parse_transcript(content: str, filename: str = "") -> List[dict]:
    """Auto-detect format and parse transcript."""
    if filename.lower().endswith('.vtt') or content.strip().startswith('WEBVTT'):
        return parse_vtt(content)
    elif filename.lower().endswith('.srt') or re.match(r'^\d+\s*\n\d{2}:', content.strip()):
        return parse_srt(content)
    else:
        return parse_plain_text(content)
