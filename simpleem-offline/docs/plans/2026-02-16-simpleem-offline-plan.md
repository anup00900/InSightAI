# simpleem-offline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a batch video analysis tool with premium glassmorphism UI, local Whisper transcription, and wizard-based UX — as a sibling to simpleem-poc.

**Architecture:** REST-only FastAPI backend with background jobs + polling. Local faster-whisper for transcription, pyannote for diarization. Same Core42 API for GPT-4o vision and GPT-4.1 text analysis. React + Vite + Tailwind frontend with dark glassmorphism theme and 4-step wizard flow.

**Tech Stack:** Python 3.9+, FastAPI, faster-whisper, pyannote-audio, SQLite, React 19, TypeScript, Vite 5, Tailwind CSS 3, Recharts 3, Lucide React.

**Source project:** `/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc`
**Target project:** `/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline`

---

## Task 1: Scaffold Project Structure

**Files:**
- Create: `simpleem-offline/backend/__init__.py`
- Create: `simpleem-offline/backend/requirements.txt`
- Create: `simpleem-offline/.env`
- Create: `simpleem-offline/run.sh`
- Create: `simpleem-offline/.gitignore`

**Step 1: Create directory structure**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work"
mkdir -p simpleem-offline/{backend/pipelines,backend/tests,frontend/src/{components/{ui,wizard,dashboard,signals,layout},lib,hooks,theme},uploads,processed,docs/plans}
```

**Step 2: Create backend requirements.txt**

```
fastapi==0.115.6
uvicorn==0.34.0
python-multipart==0.0.20
openai==1.59.5
python-dotenv==1.0.1
aiosqlite==0.20.0
aiofiles==24.1.0
pydantic==2.10.4
httpx==0.28.1
librosa>=0.10.0
numpy>=1.26.0
soundfile>=0.12.0
faster-whisper>=1.0.0
pyannote.audio>=3.1
torch>=2.0
fpdf2>=2.7
matplotlib>=3.8
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Step 3: Copy .env from simpleem-poc (identical)**

```env
# Azure OpenAI Configuration (Core42 Compass)
AZURE_OPENAI_API_KEY=3699a514e0b345afbd14492582322e28
AZURE_OPENAI_ENDPOINT=https://api.core42.ai/
AZURE_OPENAI_API_VERSION=2024-10-21

# Model deployments on Core42
VISION_MODEL=gpt-4o
CHAT_MODEL=gpt-4.1
WHISPER_MODEL=whisper-1
```

**Step 4: Create run.sh**

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== InsightAI Offline — Starting ==="

# Backend
cd backend
pip install -r requirements.txt -q 2>/dev/null
cd ..
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

# Frontend
cd frontend
npm install --silent 2>/dev/null
npm run dev &
FRONTEND_PID=$!

echo "Backend: http://localhost:8001"
echo "Frontend: http://localhost:5174"

wait $BACKEND_PID $FRONTEND_PID
```

**Step 5: Create .gitignore**

Copy from simpleem-poc, add: `data.db`, `uploads/`, `processed/`, `node_modules/`, `dist/`, `__pycache__/`, `.env`

**Step 6: Commit**

```bash
git add simpleem-offline/
git commit -m "feat: scaffold simpleem-offline project structure"
```

---

## Task 2: Copy Shared Backend Modules (unchanged from simpleem-poc)

These files are copied verbatim — they contain reusable logic that doesn't change.

**Files to copy:**
- Copy: `simpleem-poc/backend/models.py` → `simpleem-offline/backend/models.py`
- Copy: `simpleem-poc/backend/schemas.py` → `simpleem-offline/backend/schemas.py`
- Copy: `simpleem-poc/backend/core42_client.py` → `simpleem-offline/backend/core42_client.py`
- Copy: `simpleem-poc/backend/engagement_engine.py` → `simpleem-offline/backend/engagement_engine.py`
- Copy: `simpleem-poc/backend/frame_extractor.py` → `simpleem-offline/backend/frame_extractor.py`
- Copy: `simpleem-poc/backend/video_processor.py` → `simpleem-offline/backend/video_processor.py`
- Copy: `simpleem-poc/backend/signal_bus.py` → `simpleem-offline/backend/signal_bus.py`
- Copy: `simpleem-poc/backend/api_utils.py` → `simpleem-offline/backend/api_utils.py`
- Copy: `simpleem-poc/backend/export.py` → `simpleem-offline/backend/export.py`
- Copy: `simpleem-poc/backend/pipelines/__init__.py` → `simpleem-offline/backend/pipelines/__init__.py`
- Copy: `simpleem-poc/backend/pipelines/audio.py` → `simpleem-offline/backend/pipelines/audio.py`
- Copy: `simpleem-poc/backend/pipelines/visual.py` → `simpleem-offline/backend/pipelines/visual.py`
- Copy: `simpleem-poc/backend/pipelines/text.py` → `simpleem-offline/backend/pipelines/text.py`
- Copy: `simpleem-poc/backend/pipelines/diarization.py` → `simpleem-offline/backend/pipelines/diarization.py`
- Copy: `simpleem-poc/backend/__init__.py` → `simpleem-offline/backend/__init__.py`
- Copy: `simpleem-poc/backend/tests/__init__.py` → `simpleem-offline/backend/tests/__init__.py`

**Step 1: Copy all shared backend files**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work"
for f in models.py schemas.py core42_client.py engagement_engine.py frame_extractor.py video_processor.py signal_bus.py api_utils.py export.py __init__.py; do
  cp simpleem-poc/backend/$f simpleem-offline/backend/$f
done
for f in __init__.py audio.py visual.py text.py diarization.py; do
  cp simpleem-poc/backend/pipelines/$f simpleem-offline/backend/pipelines/$f
done
cp simpleem-poc/backend/tests/__init__.py simpleem-offline/backend/tests/__init__.py
```

**Step 2: Add a `wizard_step` column to videos table in models.py**

In `simpleem-offline/backend/models.py`, after `source_url TEXT DEFAULT ''`, add:

```python
                wizard_step TEXT DEFAULT 'upload',
                transcribe_progress REAL DEFAULT 0,
                analyze_progress REAL DEFAULT 0,
                progress_detail TEXT DEFAULT ''
```

This tracks wizard state: `upload` → `transcribing` → `transcribed` → `analyzing` → `complete` → `error`.

**Step 3: Add new schema models in schemas.py**

Add to `simpleem-offline/backend/schemas.py`:

```python
class VideoStatus(BaseModel):
    step: str          # upload, transcribing, transcribed, analyzing, complete, error
    progress: float    # 0-100
    detail: str        # "Extracting audio...", "Transcribing... 45%", etc.

class TranscribeRequest(BaseModel):
    pass  # No body needed, just triggers the job

class AnalyzeRequest(BaseModel):
    pass  # No body needed, just triggers the job
```

**Step 4: Commit**

```bash
git add simpleem-offline/backend/
git commit -m "feat: copy shared backend modules with wizard state extensions"
```

---

## Task 3: Create Transcription Engine (faster-whisper + pyannote)

**Files:**
- Create: `simpleem-offline/backend/transcription_engine.py`

**Step 1: Write transcription engine**

```python
"""Local transcription engine using faster-whisper + pyannote diarization.

Replaces the real-time Whisper API calls with high-quality local models.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import aiosqlite
from .models import DB_PATH
from .frame_extractor import extract_full_audio
from .pipelines.diarization import diarize_audio, assign_speakers_to_transcript

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


@dataclass
class TranscriptionProgress:
    step: str       # "extracting_audio", "transcribing", "diarizing", "assigning_speakers", "done"
    progress: float # 0-100
    detail: str


async def _update_progress(video_id: str, step: str, progress: float, detail: str):
    """Update transcription progress in DB for polling."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET wizard_step = ?, transcribe_progress = ?, progress_detail = ? WHERE id = ?",
            (step, progress, detail, video_id),
        )
        await db.commit()


async def run_transcription(video_id: str, video_path: str):
    """Full transcription pipeline: extract audio → faster-whisper → pyannote → assign speakers.

    Updates progress in DB so frontend can poll.
    """
    output_dir = os.path.join(PROCESSED_DIR, video_id, "offline")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Extract full audio
        await _update_progress(video_id, "transcribing", 5, "Extracting audio track...")
        audio_path = await extract_full_audio(video_path, output_dir)
        if not audio_path:
            await _update_progress(video_id, "error", 0, "No audio track found in video")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE videos SET status = 'error' WHERE id = ?", (video_id,))
                await db.commit()
            return

        # Step 2: Transcribe with faster-whisper
        await _update_progress(video_id, "transcribing", 15, "Loading Whisper model...")
        segments = await _transcribe_with_faster_whisper(audio_path, video_id)

        # Step 3: Speaker diarization with pyannote
        await _update_progress(video_id, "transcribing", 70, "Running speaker diarization...")
        diarization_segments = await diarize_audio(audio_path)

        # Step 4: Assign speakers to transcript segments
        await _update_progress(video_id, "transcribing", 85, "Assigning speakers...")
        if diarization_segments:
            segments = assign_speakers_to_transcript(segments, diarization_segments)

        # Step 5: Store in DB
        await _update_progress(video_id, "transcribing", 95, "Saving transcript...")
        await _store_transcript(video_id, segments)

        # Done
        await _update_progress(video_id, "transcribed", 100, "Transcription complete")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET wizard_step = 'transcribed' WHERE id = ?", (video_id,)
            )
            await db.commit()

        logger.info(f"[{video_id[:8]}] Transcription complete: {len(segments)} segments")

    except Exception as e:
        logger.error(f"[{video_id[:8]}] Transcription failed: {e}")
        await _update_progress(video_id, "error", 0, f"Transcription failed: {str(e)[:200]}")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET status = 'error', wizard_step = 'error' WHERE id = ?",
                (video_id,),
            )
            await db.commit()


async def _transcribe_with_faster_whisper(audio_path: str, video_id: str) -> list[dict]:
    """Transcribe audio using faster-whisper large-v3 locally."""

    def _run():
        from faster_whisper import WhisperModel

        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=None,  # auto-detect
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        logger.info(f"[{video_id[:8]}] Detected language: {info.language} (prob={info.language_probability:.2f})")

        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            # Filter common Whisper hallucinations
            if text.lower().rstrip('.!?,') in (
                "you", "bye", "bye-bye", "thank you", "thanks",
                "thanks for watching", "subscribe", "like and subscribe",
            ):
                continue
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "speaker": "",
            })
        return segments

    await _update_progress(video_id, "transcribing", 20, "Transcribing audio with Whisper large-v3...")
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(None, _run)

    await _update_progress(video_id, "transcribing", 65, f"Transcribed {len(segments)} segments")
    return segments


async def _store_transcript(video_id: str, segments: list[dict]):
    """Store transcript segments in the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Clear any existing transcript for this video
        await db.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
        for seg in segments:
            await db.execute(
                """INSERT INTO transcript_segments
                   (video_id, start_time, end_time, text, speaker)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, seg["start"], seg["end"], seg["text"], seg.get("speaker", "")),
            )
        await db.commit()
```

**Step 2: Commit**

```bash
git add simpleem-offline/backend/transcription_engine.py
git commit -m "feat: add local transcription engine with faster-whisper + pyannote"
```

---

## Task 4: Create Batch Analysis Pipeline

**Files:**
- Create: `simpleem-offline/backend/batch_pipeline.py`

**Step 1: Write batch pipeline**

This orchestrates the full analysis after transcription: frame sampling → GPT-4o vision → librosa audio → GPT-4.1 text → engagement scoring → coaching → summary → flags → persistence.

```python
"""Batch analysis pipeline — runs full analysis on a pre-transcribed video.

No WebSocket. Updates progress in DB for frontend polling.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

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
from .signal_bus import SignalBus, SignalEvent, SignalType
from .engagement_engine import EngagementEngine

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
FRAME_INTERVAL = 3.0  # Sample a frame every 3 seconds
MAX_CONCURRENT_API = 5  # Max parallel API calls


async def _update_progress(video_id: str, progress: float, detail: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET wizard_step = 'analyzing', analyze_progress = ?, progress_detail = ? WHERE id = ?",
            (progress, detail, video_id),
        )
        await db.commit()


async def run_analysis(video_id: str, video_path: str):
    """Full batch analysis pipeline. Assumes transcription is already complete."""
    output_dir = os.path.join(PROCESSED_DIR, video_id, "offline")
    os.makedirs(output_dir, exist_ok=True)

    try:
        duration = await get_video_duration(video_path)
        await _update_progress(video_id, 5, "Starting analysis...")

        # Step 1: Extract names from multiple frames
        await _update_progress(video_id, 8, "Detecting participants...")
        name_map = await _extract_names(video_path, output_dir, duration)
        participant_names = sorted(set(name_map.values())) if name_map else []
        logger.info(f"[{video_id[:8]}] Found {len(participant_names)} participants: {participant_names}")

        # Step 2: Analyze frames (GPT-4o vision) — batched
        await _update_progress(video_id, 12, "Extracting and analyzing frames...")
        timestamps = [t for t in _frange(0, duration, FRAME_INTERVAL)]
        signal_snapshots = await _analyze_frames_batch(
            video_id, video_path, output_dir, timestamps, name_map
        )
        logger.info(f"[{video_id[:8]}] Analyzed {len(signal_snapshots)} frames")

        # Step 3: Audio feature analysis (librosa)
        await _update_progress(video_id, 55, "Analyzing audio features...")
        audio_path = os.path.join(output_dir, "rt_audio_full.wav")
        if not os.path.exists(audio_path):
            audio_path_new = await extract_full_audio(video_path, output_dir)
            if audio_path_new:
                audio_path = audio_path_new
        audio_features = await _analyze_audio_batch(video_id, audio_path, output_dir, duration)

        # Step 4: Text analysis (GPT-4.1: voice, words, personality, correlations)
        await _update_progress(video_id, 65, "Analyzing text signals...")
        transcript_segments = await _get_transcript(video_id)
        transcript_text = "\n".join(
            f"[{s['start_time']:.1f}s]: {s['text']}" for s in transcript_segments
        )
        voice_signals, word_signals = await _analyze_text_signals(
            video_id, transcript_segments
        )

        await _update_progress(video_id, 75, "Analyzing personality and correlations...")
        personality_data = await _analyze_personality(signal_snapshots, transcript_text)
        correlations = await _analyze_correlations(signal_snapshots, voice_signals, word_signals, transcript_text)

        # Step 5: Engagement scoring
        await _update_progress(video_id, 80, "Computing engagement scores...")
        engagement_engine = EngagementEngine()

        # Step 6: Generate summary, flags, coaching
        await _update_progress(video_id, 85, "Generating summary and insights...")
        emotions_summary = json.dumps(
            [{"timestamp": s.get("timestamp", 0), "participants": s.get("participants", [])}
             for s in signal_snapshots[:30]]
        )
        analysis = await generate_summary_and_flags(
            transcript_text[:8000], emotions_summary[:4000],
            participant_names or ["Person 1"],
        )

        await _update_progress(video_id, 90, "Generating coaching recommendations...")
        coaching_per_participant = {}
        for name in (participant_names or ["Person 1"]):
            p_emotions = json.dumps(
                [s for s in signal_snapshots if name in str(s)][:15]
            )
            coaching_items = await generate_coaching(
                transcript_text[:4000], p_emotions[:2000], name
            )
            coaching_per_participant[name] = coaching_items

        # Step 7: Persist everything
        await _update_progress(video_id, 95, "Saving results...")
        await _persist_all(
            video_id, video_path, duration, analysis,
            signal_snapshots, voice_signals, word_signals,
            personality_data, correlations, coaching_per_participant,
            participant_names, name_map, audio_features,
        )

        await _update_progress(video_id, 100, "Analysis complete")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET wizard_step = 'complete', status = 'complete' WHERE id = ?",
                (video_id,),
            )
            await db.commit()

        logger.info(f"[{video_id[:8]}] Batch analysis complete")

    except Exception as e:
        logger.error(f"[{video_id[:8]}] Analysis failed: {e}")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET wizard_step = 'error', status = 'error', progress_detail = ? WHERE id = ?",
                (f"Analysis failed: {str(e)[:200]}", video_id),
            )
            await db.commit()


# --- Helper functions ---

def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    val = start
    while val < stop:
        yield round(val, 2)
        val += step


async def _extract_names(video_path: str, output_dir: str, duration: float) -> dict[str, str]:
    """Extract participant names from multiple frames."""
    name_map = {}
    scan_timestamps = [5.0, 10.0, 15.0, 30.0, 60.0, 120.0]
    for ts in scan_timestamps:
        if ts > duration:
            break
        try:
            frame_path = await extract_single_frame(video_path, ts, output_dir)
            result = await extract_names_from_frame(frame_path)
            if isinstance(result, dict) and result:
                for k, v in result.items():
                    if v not in name_map.values():
                        name_map[k] = v
            try:
                os.remove(frame_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"Name extraction at {ts}s failed: {e}")
    return name_map


async def _analyze_frames_batch(
    video_id: str, video_path: str, output_dir: str,
    timestamps: list[float], name_map: dict[str, str],
) -> list[dict]:
    """Analyze frames in parallel batches."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API)
    snapshots = []
    total = len(timestamps)

    async def _analyze_one(ts: float, idx: int):
        async with semaphore:
            try:
                frame_path = await extract_single_frame(video_path, ts, output_dir)
                result = await analyze_frame_all_signals(frame_path, ts)
                result["timestamp"] = ts

                # Map Person N labels to real names
                for p in result.get("participants", []):
                    raw_label = p.get("label", "Person 1")
                    if raw_label in name_map:
                        p["label"] = name_map[raw_label]

                snapshots.append(result)
                pct = 12 + int((idx / total) * 40)
                await _update_progress(video_id, pct, f"Analyzing frame {idx+1}/{total}...")

                try:
                    os.remove(frame_path)
                except OSError:
                    pass
            except Exception as e:
                logger.error(f"[{video_id[:8]}] Frame {ts:.1f}s failed: {e}")

    await asyncio.gather(*[_analyze_one(ts, i) for i, ts in enumerate(timestamps)])
    snapshots.sort(key=lambda s: s.get("timestamp", 0))
    return snapshots


async def _analyze_audio_batch(
    video_id: str, audio_path: str, output_dir: str, duration: float
) -> list[dict]:
    """Analyze audio in chunks using librosa."""
    if not audio_path or not os.path.exists(audio_path):
        return []

    analyzer = AudioAnalyzer()
    features = []
    chunk_duration = 10.0  # 10 second chunks
    timestamps = list(_frange(0, duration, chunk_duration))

    for ts in timestamps:
        try:
            end = min(ts + chunk_duration, duration)
            chunk_path = await extract_audio_chunk(audio_path, ts, end, output_dir)
            result = analyzer.analyze_file(chunk_path)
            features.append({
                "timestamp": (ts + end) / 2,
                "energy": result.engagement_score,
                "pitch_mean": result.pitch_mean,
                "pitch_std": result.pitch_std,
                "volume_energy": result.volume_energy,
                "speaking_rate": result.speaking_rate,
                "pause_count": result.pause_count,
            })
            try:
                os.remove(chunk_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"[{video_id[:8]}] Audio chunk at {ts:.1f}s failed: {e}")

    return features


async def _get_transcript(video_id: str) -> list[dict]:
    """Get transcript segments from DB."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
            (video_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def _analyze_text_signals(
    video_id: str, transcript_segments: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Analyze voice and word signals from transcript chunks."""
    voice_signals = []
    word_signals = []

    # Group transcript into ~30 second chunks
    chunks = []
    current_chunk = []
    chunk_start = 0

    for seg in transcript_segments:
        current_chunk.append(seg["text"])
        if seg["end_time"] - chunk_start >= 30 or seg == transcript_segments[-1]:
            chunks.append({
                "text": " ".join(current_chunk),
                "timestamp": (chunk_start + seg["end_time"]) / 2,
            })
            chunk_start = seg["end_time"]
            current_chunk = []

    for chunk in chunks:
        try:
            voice, words = await asyncio.gather(
                analyze_voice_signal(chunk["text"]),
                analyze_words_signal(chunk["text"], chunk["timestamp"]),
                return_exceptions=True,
            )
            if not isinstance(voice, Exception):
                voice["timestamp"] = chunk["timestamp"]
                voice_signals.append(voice)
            if not isinstance(words, Exception):
                words["timestamp"] = chunk["timestamp"]
                word_signals.append(words)
        except Exception as e:
            logger.warning(f"[{video_id[:8]}] Text analysis failed: {e}")

    return voice_signals, word_signals


async def _analyze_personality(signal_snapshots: list[dict], transcript_text: str) -> dict | None:
    """Analyze personality from accumulated signals."""
    if len(signal_snapshots) < 2:
        return None
    try:
        signals_summary = json.dumps(signal_snapshots[-20:])
        return await analyze_personality(signals_summary, transcript_text[:3000])
    except Exception as e:
        logger.error(f"Personality analysis failed: {e}")
        return None


async def _analyze_correlations(
    signal_snapshots: list[dict], voice_signals: list[dict],
    word_signals: list[dict], transcript_text: str,
) -> list[dict]:
    """Build cross-signal correlations."""
    if len(signal_snapshots) < 3:
        return []
    try:
        all_signals = json.dumps({
            "visual": signal_snapshots[-15:],
            "voice": voice_signals[-5:],
            "words": word_signals[-5:],
        })
        result = await build_correlations(all_signals, transcript_text[:2000])
        return result.get("correlations", [])
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        return []


async def _persist_all(
    video_id: str, video_path: str, duration: float, analysis: dict,
    signal_snapshots: list[dict], voice_signals: list[dict],
    word_signals: list[dict], personality_data: dict | None,
    correlations: list[dict], coaching_per_participant: dict[str, list[dict]],
    participant_names: list[str], name_map: dict[str, str],
    audio_features: list[dict],
):
    """Persist all analysis results to SQLite."""
    async with aiosqlite.connect(DB_PATH) as db:
        p_scores = analysis.get("participant_scores", [])
        avg_engagement = 0.0
        if p_scores:
            avg_engagement = sum(p.get("engagement", 70) for p in p_scores) / len(p_scores)

        await db.execute(
            "UPDATE videos SET duration = ?, overall_score = ? WHERE id = ?",
            (duration, avg_engagement, video_id),
        )

        # Participants
        participant_id_map = {}
        for ps in p_scores:
            pid = str(uuid.uuid4())
            name = ps.get("name", "Unknown")
            participant_id_map[name] = pid
            await db.execute(
                """INSERT INTO participants (id, video_id, name, engagement_score,
                   sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, video_id, name,
                 ps.get("engagement", 70), ps.get("sentiment", 0.3),
                 ps.get("speaking_pct", 25), ps.get("clarity", 70),
                 ps.get("rapport", 70), ps.get("energy", 70)),
            )

        # Signal snapshots + emotions
        for snap in signal_snapshots:
            ts = snap.get("timestamp", 0)
            for p in snap.get("participants", []):
                label = p.get("label", "Person 1")
                pid = participant_id_map.get(label)
                if not pid:
                    pid = str(uuid.uuid4())
                    participant_id_map[label] = pid
                emotions = p.get("emotions", {})
                bl = p.get("body_language", {})
                await db.execute(
                    """INSERT INTO signal_snapshots
                       (video_id, participant_id, timestamp,
                        emotion, emotion_confidence, engagement,
                        posture, openness, leaning, mirroring, body_language_desc,
                        gestures, reactions)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (video_id, pid, ts,
                     emotions.get("primary", "neutral"),
                     emotions.get("confidence", 0.5),
                     emotions.get("engagement", 50),
                     bl.get("posture", "upright"), bl.get("openness", "mixed"),
                     bl.get("leaning", "neutral"), bl.get("mirroring", False),
                     bl.get("description", ""),
                     json.dumps(p.get("gestures", [])),
                     json.dumps(p.get("reactions", []))),
                )
                await db.execute(
                    """INSERT INTO emotions (video_id, participant_id, timestamp,
                       emotion, confidence, engagement) VALUES (?, ?, ?, ?, ?, ?)""",
                    (video_id, pid, ts,
                     emotions.get("primary", "neutral"),
                     emotions.get("confidence", 0.5),
                     emotions.get("engagement", 50)),
                )

        # Voice signals
        for vs in voice_signals:
            await db.execute(
                """INSERT INTO voice_signals (video_id, timestamp, tone, pace, energy, dynamics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (video_id, vs.get("timestamp", 0), vs.get("tone", "neutral"),
                 vs.get("pace", "moderate"), vs.get("energy", 50), vs.get("dynamics", "")),
            )

        # Word signals
        for ws in word_signals:
            await db.execute(
                """INSERT INTO word_signals (video_id, timestamp, sentiment, sentiment_score, keywords, key_phrases)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (video_id, ws.get("timestamp", 0), ws.get("sentiment", "neutral"),
                 ws.get("sentiment_score", 0.5), json.dumps(ws.get("keywords", [])),
                 json.dumps(ws.get("key_phrases", []))),
            )

        # Personality
        if personality_data:
            for p in personality_data.get("participants", []):
                label = p.get("label", "Person 1")
                pid = participant_id_map.get(label)
                if not pid:
                    pid = str(uuid.uuid4())
                await db.execute(
                    """INSERT INTO personality_signals (video_id, participant_id, traits, communication_style, confidence_level)
                       VALUES (?, ?, ?, ?, ?)""",
                    (video_id, pid, json.dumps(p.get("traits", [])),
                     p.get("communication_style", ""), p.get("confidence_level", "")),
                )

        # Correlations
        for corr in correlations:
            await db.execute(
                """INSERT INTO correlations (video_id, signal_pair, pattern, confidence, implication)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, json.dumps(corr.get("signal_pair", [])),
                 corr.get("pattern", ""), corr.get("confidence", 0.0),
                 corr.get("implication", "")),
            )

        # Flags
        for flag in analysis.get("flags", []):
            await db.execute(
                """INSERT INTO flags (video_id, timestamp, type, description, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, flag.get("timestamp", 0), flag.get("type", "neutral"),
                 flag.get("description", ""), flag.get("severity", "medium")),
            )

        # Summary
        summary_text = analysis.get("summary", "")
        if summary_text:
            await db.execute(
                """INSERT OR REPLACE INTO summaries (video_id, summary, key_topics, overall_sentiment)
                   VALUES (?, ?, ?, ?)""",
                (video_id, summary_text, json.dumps(analysis.get("key_topics", [])),
                 analysis.get("overall_sentiment", "neutral")),
            )

        # Coaching
        for name, pid in participant_id_map.items():
            for item in coaching_per_participant.get(name, []):
                await db.execute(
                    """INSERT INTO coaching (video_id, participant_id, recommendation, category, priority)
                       VALUES (?, ?, ?, ?, ?)""",
                    (video_id, pid, item.get("recommendation", ""),
                     item.get("category", "general"), item.get("priority", 3)),
                )

        # Meeting analytics
        await db.execute(
            """INSERT OR REPLACE INTO meeting_analytics
               (video_id, avg_engagement, participant_count, duration, overall_sentiment)
               VALUES (?, ?, ?, ?, ?)""",
            (video_id, avg_engagement, len(participant_names),
             duration, analysis.get("overall_sentiment", "neutral")),
        )

        await db.commit()
        logger.info(f"[{video_id[:8]}] All results persisted to DB")
```

**Step 2: Commit**

```bash
git add simpleem-offline/backend/batch_pipeline.py
git commit -m "feat: add batch analysis pipeline for offline processing"
```

---

## Task 5: Create Backend main.py (REST-only API)

**Files:**
- Create: `simpleem-offline/backend/main.py`

**Step 1: Write main.py**

Copy the structure from simpleem-poc's main.py but:
- Remove WebSocket endpoint
- Remove Recall.ai meeting bot endpoints
- Add `POST /api/videos/{id}/transcribe`
- Add `POST /api/videos/{id}/analyze`
- Add `GET /api/videos/{id}/status` (returns wizard step + progress)
- Keep all other GET endpoints identical

The main.py should be similar to simpleem-poc's but with these key differences:
1. No `from .realtime_ws import websocket_analyze`
2. No `from .recall_adapter import RecallAiAdapter`
3. No WebSocket route
4. No meeting bot routes
5. Add transcribe and analyze routes that launch background tasks
6. Add status endpoint that reads wizard_step + progress from DB
7. Port changed to 8001 (in run.sh, not in main.py)

**Step 2: Commit**

```bash
git add simpleem-offline/backend/main.py
git commit -m "feat: add REST-only FastAPI backend with batch endpoints"
```

---

## Task 6: Scaffold Frontend Project

**Files:**
- Create: `simpleem-offline/frontend/package.json`
- Create: `simpleem-offline/frontend/vite.config.ts`
- Create: `simpleem-offline/frontend/tsconfig.json`
- Create: `simpleem-offline/frontend/tsconfig.app.json`
- Create: `simpleem-offline/frontend/tsconfig.node.json`
- Create: `simpleem-offline/frontend/tailwind.config.js`
- Create: `simpleem-offline/frontend/postcss.config.js`
- Create: `simpleem-offline/frontend/eslint.config.js`
- Create: `simpleem-offline/frontend/index.html`
- Create: `simpleem-offline/frontend/src/main.tsx`
- Create: `simpleem-offline/frontend/src/vite-env.d.ts`

**Step 1: Copy base frontend config files from simpleem-poc**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work"
for f in tsconfig.json tsconfig.app.json tsconfig.node.json postcss.config.js eslint.config.js; do
  cp simpleem-poc/frontend/$f simpleem-offline/frontend/$f
done
cp simpleem-poc/frontend/src/vite-env.d.ts simpleem-offline/frontend/src/vite-env.d.ts
```

**Step 2: Create package.json (same deps + framer-motion)**

Same as simpleem-poc but add `framer-motion` for wizard transitions:

```json
{
  "name": "simpleem-offline-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5174",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "clsx": "^2.1.1",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.563.0",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "recharts": "^3.7.0",
    "tailwind-merge": "^3.4.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.39.1",
    "@types/node": "^24.10.1",
    "@types/react": "^19.2.7",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^4.7.0",
    "autoprefixer": "^10.4.24",
    "eslint": "^9.39.1",
    "eslint-plugin-react-hooks": "^7.0.1",
    "eslint-plugin-react-refresh": "^0.4.24",
    "globals": "^16.5.0",
    "postcss": "^8.5.6",
    "tailwindcss": "^3.4.19",
    "typescript": "~5.9.3",
    "typescript-eslint": "^8.48.0",
    "vite": "^5.4.21"
  }
}
```

**Step 3: Create vite.config.ts (proxy to port 8001, no WS)**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        timeout: 600000,
        proxyTimeout: 600000,
      },
      '/uploads': 'http://127.0.0.1:8001',
    },
  },
})
```

**Step 4: Create tailwind.config.js with glassmorphism theme**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0a1a',
        'bg-secondary': '#0f0f2e',
        'bg-card': 'rgba(255,255,255,0.05)',
        'bg-card-hover': 'rgba(255,255,255,0.08)',
        'bg-card-solid': '#141430',
        'border': 'rgba(255,255,255,0.1)',
        'border-light': 'rgba(255,255,255,0.15)',
        'border-glow': 'rgba(139,92,246,0.3)',
        'accent': '#8b5cf6',
        'accent-light': '#a78bfa',
        'accent-blue': '#3b82f6',
        'accent-cyan': '#06b6d4',
        'accent-glow': 'rgba(139,92,246,0.4)',
        'success': '#10b981',
        'success-glow': 'rgba(16,185,129,0.3)',
        'warning': '#f59e0b',
        'warning-glow': 'rgba(245,158,11,0.3)',
        'danger': '#ef4444',
        'danger-glow': 'rgba(239,68,68,0.3)',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      backdropBlur: {
        'glass': '20px',
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0,0,0,0.3)',
        'glow-purple': '0 0 15px rgba(139,92,246,0.3)',
        'glow-blue': '0 0 15px rgba(59,130,246,0.3)',
        'glow-cyan': '0 0 15px rgba(6,182,212,0.3)',
        'glow-success': '0 0 15px rgba(16,185,129,0.3)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7)',
        'gradient-secondary': 'linear-gradient(135deg, #06b6d4, #3b82f6)',
        'gradient-success': 'linear-gradient(135deg, #10b981, #06b6d4)',
        'gradient-warm': 'linear-gradient(135deg, #f59e0b, #ef4444)',
        'gradient-mesh': 'radial-gradient(ellipse at 20% 50%, rgba(139,92,246,0.15) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(59,130,246,0.1) 0%, transparent 50%), radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.08) 0%, transparent 50%)',
      },
    },
  },
  plugins: [],
}
```

**Step 5: Create index.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <title>InsightAI Offline — Meeting Analysis</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**Step 6: Create src/main.tsx**

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

**Step 7: Commit**

```bash
git add simpleem-offline/frontend/
git commit -m "feat: scaffold frontend with glassmorphism Tailwind theme"
```

---

## Task 7: Create Frontend Theme, Global CSS, and Reusable UI Components

**Files:**
- Create: `simpleem-offline/frontend/src/index.css`
- Create: `simpleem-offline/frontend/src/theme/colors.ts`
- Create: `simpleem-offline/frontend/src/components/ui/GlassCard.tsx`
- Create: `simpleem-offline/frontend/src/components/ui/GradientButton.tsx`
- Create: `simpleem-offline/frontend/src/components/ui/ProgressRing.tsx`
- Create: `simpleem-offline/frontend/src/components/ui/AnimatedCounter.tsx`
- Create: `simpleem-offline/frontend/src/components/ui/GlowBadge.tsx`

These are the reusable premium UI primitives used by all pages. Each component should have:
- Glassmorphism styling (backdrop-blur, semi-transparent backgrounds, glow borders)
- Gradient accents (purple/blue/cyan)
- Smooth animations via framer-motion
- TypeScript props interfaces

**Step 1: Create index.css** — Global styles including glassmorphism base classes, animations, scrollbar styling, gradient text utility.

**Step 2: Create colors.ts** — Theme constants exported for use in charts/recharts.

**Step 3: Create GlassCard.tsx** — A card wrapper with frosted glass effect, optional glow border, gradient header option.

**Step 4: Create GradientButton.tsx** — Button with gradient background, hover glow, loading spinner state.

**Step 5: Create ProgressRing.tsx** — SVG circular progress indicator with animated stroke, percentage text in center, gradient stroke color.

**Step 6: Create AnimatedCounter.tsx** — Number that animates from 0 to target value on mount using requestAnimationFrame.

**Step 7: Create GlowBadge.tsx** — Small badge/pill with glow effect, used for engagement levels and emotion labels.

**Step 8: Commit**

```bash
git add simpleem-offline/frontend/src/
git commit -m "feat: add glassmorphism UI primitives and global theme"
```

---

## Task 8: Create Frontend API Client and Polling Hook

**Files:**
- Create: `simpleem-offline/frontend/src/lib/api.ts`
- Create: `simpleem-offline/frontend/src/hooks/usePolling.ts`

**Step 1: Create api.ts** — Copy interfaces from simpleem-poc's api.ts. Add:
- `triggerTranscription(videoId)` → `POST /api/videos/{id}/transcribe`
- `triggerAnalysis(videoId)` → `POST /api/videos/{id}/analyze`
- `getVideoStatus(videoId)` → `GET /api/videos/{id}/status`
- `getTranscript(videoId)` → `GET /api/videos/{id}/transcript`
- Remove WebSocket-related code
- Keep all other functions identical

**Step 2: Create usePolling.ts** — React hook that polls a URL at an interval and returns the latest data. Used for tracking transcription/analysis progress.

```typescript
import { useState, useEffect, useRef } from 'react';

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled: boolean = true,
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const timerRef = useRef<number>();

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const result = await fetcher();
        if (!cancelled) setData(result);
      } catch (e) {
        if (!cancelled) setError(e as Error);
      }
    };

    poll(); // immediate first fetch
    timerRef.current = window.setInterval(poll, intervalMs);

    return () => {
      cancelled = true;
      clearInterval(timerRef.current);
    };
  }, [fetcher, intervalMs, enabled]);

  return { data, error };
}
```

**Step 3: Commit**

```bash
git add simpleem-offline/frontend/src/lib/ simpleem-offline/frontend/src/hooks/
git commit -m "feat: add REST API client and polling hook"
```

---

## Task 9: Create Wizard Components

**Files:**
- Create: `simpleem-offline/frontend/src/components/wizard/WizardShell.tsx`
- Create: `simpleem-offline/frontend/src/components/wizard/UploadStep.tsx`
- Create: `simpleem-offline/frontend/src/components/wizard/TranscriptStep.tsx`
- Create: `simpleem-offline/frontend/src/components/wizard/AnalysisStep.tsx`

**Step 1: Create WizardShell.tsx** — Horizontal step indicator (Upload → Transcribe → Analyze → Results) with animated active/completed states, gradient connector lines. Uses framer-motion AnimatePresence for step transitions.

**Step 2: Create UploadStep.tsx** — Drag-and-drop zone with animated gradient dashed border. File type icon, name, and size display. Upload progress via ProgressRing. "Continue" button auto-advances after upload. Glassmorphism card styling.

**Step 3: Create TranscriptStep.tsx** — Auto-triggers `POST /api/videos/{id}/transcribe` on mount. Shows ProgressRing with status text during transcription. When done, shows scrollable transcript with:
- Colored speaker labels (each speaker gets a unique gradient color)
- Timestamps left-aligned
- Smooth scroll
- "Continue to Analysis" GradientButton at bottom

**Step 4: Create AnalysisStep.tsx** — Auto-triggers `POST /api/videos/{id}/analyze` on mount. Shows multi-stage progress:
- ProgressRing with overall percentage
- Status text updates from polling ("Detecting participants...", "Analyzing frame 12/45...", etc.)
- Sub-stage pills showing completed stages
- Auto-advances to results when complete

**Step 5: Commit**

```bash
git add simpleem-offline/frontend/src/components/wizard/
git commit -m "feat: add wizard step components with glassmorphism styling"
```

---

## Task 10: Create Dashboard Components (Glassmorphism Version)

**Files:**
- Create: `simpleem-offline/frontend/src/components/dashboard/AnalysisDashboard.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/EmotionTimeline.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/ParticipantSidebar.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/CoachingPanel.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/SpeakingDonut.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/SkillRadar.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/FlagTimeline.tsx`
- Create: `simpleem-offline/frontend/src/components/dashboard/SignalWeightCard.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/SignalPanel.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/EmotionsSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/BodyLanguageSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/GesturesSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/ReactionsSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/VoiceSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/WordsSignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/PersonalitySignal.tsx`
- Create: `simpleem-offline/frontend/src/components/signals/CorrelationPanel.tsx`

These are glassmorphism-styled versions of the simpleem-poc dashboard components. Key differences:
- All cards use GlassCard wrapper
- Gradient text headings
- GlowBadge for scores/emotions
- AnimatedCounter for numbers
- All data loaded from REST (no WebSocket state)
- Static view (no real-time updates)
- Recharts with theme-matching colors (gradient fills, glow)

**Step 1: Create AnalysisDashboard.tsx** — Main results page. Fetches all data from `GET /api/videos/{id}/results` and related endpoints. Renders:
- Video info header with AnimatedCounter scores
- Video player (static, for review)
- Tab panel: Summary | Coaching | Transcript | Insights
- Signal panels grid (7 signals)
- EmotionTimeline chart
- ParticipantSidebar
- SpeakingDonut + SkillRadar
- FlagTimeline
- SignalWeightCard
- Export buttons (PDF/CSV)

**Step 2-8: Create each signal panel and chart component** — Each wraps in GlassCard, uses glassmorphism styling, gradient accents. Logic is same as simpleem-poc but reads from props (data passed from AnalysisDashboard) instead of WebSocket state.

**Step 9: Commit**

```bash
git add simpleem-offline/frontend/src/components/dashboard/ simpleem-offline/frontend/src/components/signals/
git commit -m "feat: add glassmorphism dashboard and signal panel components"
```

---

## Task 11: Create Library and Analytics Views

**Files:**
- Create: `simpleem-offline/frontend/src/components/VideoLibrary.tsx`
- Create: `simpleem-offline/frontend/src/components/VideoUpload.tsx`
- Create: `simpleem-offline/frontend/src/components/CrossMeetingAnalytics.tsx`

**Step 1: Create VideoLibrary.tsx** — Grid of uploaded videos with glassmorphism cards. Status badges (Ready, Transcribed, Complete, Error). Delete button. Score display with GlowBadge.

**Step 2: Create VideoUpload.tsx** — Simplified version (no URL import, no meeting bot). Just drag-and-drop file upload with animated gradient border and ProgressRing.

**Step 3: Create CrossMeetingAnalytics.tsx** — Copy logic from simpleem-poc, restyle with glassmorphism.

**Step 4: Commit**

```bash
git add simpleem-offline/frontend/src/components/
git commit -m "feat: add video library and analytics with glassmorphism styling"
```

---

## Task 12: Create App.tsx and Layout Components

**Files:**
- Create: `simpleem-offline/frontend/src/App.tsx`
- Create: `simpleem-offline/frontend/src/components/layout/Sidebar.tsx`
- Create: `simpleem-offline/frontend/src/components/layout/TopBar.tsx`

**Step 1: Create App.tsx** — Root component with views:
- `library` → VideoUpload + VideoLibrary + CrossMeetingAnalytics
- `wizard` → WizardShell with 4 steps (UploadStep → TranscriptStep → AnalysisStep → Results/Dashboard)
- `dashboard` → AnalysisDashboard (for reviewing previously completed analyses)

State management:
- `view`: 'library' | 'wizard' | 'dashboard'
- `wizardStep`: 1 | 2 | 3 | 4
- `selectedVideoId`: string | null
- `selectedVideo`: Video | null
- `results`: AnalysisResults | null

Flow:
1. User uploads video → auto-enters wizard at step 1
2. Upload completes → advance to step 2 (TranscriptStep)
3. Transcription completes → user reviews, clicks Continue → step 3 (AnalysisStep)
4. Analysis completes → step 4 shows AnalysisDashboard
5. From library, clicking a "complete" video goes straight to dashboard view

**Step 2: Create TopBar.tsx** — App header with gradient logo, "InsightAI Offline" title, back button. Glassmorphism navbar with backdrop blur.

**Step 3: Create Sidebar.tsx** — Optional left nav (if needed, otherwise TopBar is sufficient).

**Step 4: Commit**

```bash
git add simpleem-offline/frontend/src/
git commit -m "feat: add App root with wizard flow and layout components"
```

---

## Task 13: Install Dependencies and Test Boot

**Step 1: Install backend dependencies**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
pip install -r backend/requirements.txt
```

**Step 2: Install frontend dependencies**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline/frontend"
npm install
```

**Step 3: Test backend starts**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

Expected: FastAPI starts, DB initializes, no errors.

**Step 4: Test frontend starts**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline/frontend"
npm run dev
```

Expected: Vite dev server starts on port 5174, no compile errors.

**Step 5: Test basic flow**

1. Open http://localhost:5174
2. Upload a small video
3. Check wizard transitions to transcript step
4. Check transcription runs (may take a while on first run as faster-whisper downloads the model)
5. Check analysis runs after clicking continue
6. Check results dashboard shows

**Step 6: Fix any issues found during testing**

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: complete simpleem-offline MVP with wizard flow and glassmorphism UI"
```

---

## Summary of Key Architecture Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| No WebSocket | REST + polling | Batch processing doesn't need bidirectional real-time |
| Local Whisper | faster-whisper large-v3 | Better quality than API, no per-call cost, works offline |
| Local diarization | pyannote-audio 3.1 | Accurate speaker identification, pairs with whisper |
| Wizard UX | 4-step flow | Gives user control, transparency into processing |
| Glassmorphism UI | Tailwind + framer-motion | Premium feel without heavy CSS framework |
| Port 8001/5174 | Different from simpleem-poc | Can run both simultaneously |
| Same DB schema | SQLite + same tables | Maximum compatibility with export/analytics |
| Same API prompts | Identical GPT-4o/4.1 prompts | Proven quality, no regression risk |
