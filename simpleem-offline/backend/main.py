from __future__ import annotations

import os
import uuid
import json
import asyncio
import logging
import aiosqlite
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .models import init_db, DB_PATH
from .schemas import (
    VideoOut, ParticipantOut, EmotionPoint, TranscriptSegment,
    Flag, CoachingItem, SummaryOut, AnalysisResults, ImportUrlRequest,
    MeetingAnalytics, RenameParticipantRequest, VideoStatus, MeetingNotesOut,
)
from .transcription_engine import run_transcription
from .transcript_parser import parse_transcript
from .batch_pipeline import run_analysis, subscribe_events, unsubscribe_events
from .video_processor import is_video_file, is_audio_file, get_video_duration
from .media_ingestion import MediaIngestion, extract_name_from_url, COOKIE_DIR
from .export import generate_pdf_report, generate_csv_export
from .frame_extractor import extract_single_frame

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="InsightAI - Offline Conversation Intelligence", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Upload / Import endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video or audio file for analysis. Streams to disk for speed."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    if not (is_video_file(file.filename) or is_audio_file(file.filename)):
        raise HTTPException(400, "Unsupported file format. Use video (mp4, avi, mov, mkv, webm) or audio (mp3, wav, m4a, flac)")

    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")

    # Stream to disk in chunks (no full-file memory load)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    total_size = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                f.close()
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="File too large. Maximum 2GB.")
            f.write(chunk)

    # Insert immediately — duration computed in background
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO videos
               (id, name, filename, file_path, duration, status, wizard_step)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (video_id, file.filename, file.filename, file_path, 0, "ready", "upload"),
        )
        await db.commit()

    # Compute duration in background (ffprobe) — doesn't block response
    background_tasks.add_task(_update_duration, video_id, file_path)

    return {"id": video_id, "status": "ready", "message": "Ready for transcription and analysis"}


async def _update_duration(video_id: str, file_path: str):
    """Background: compute video duration via ffprobe and update DB."""
    try:
        duration = await get_video_duration(file_path)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE videos SET duration = ? WHERE id = ?", (duration, video_id))
            await db.commit()
    except Exception as e:
        logger.warning(f"Duration update failed for {video_id}: {e}")


@app.post("/api/import-url")
async def import_video_url(req: ImportUrlRequest, background_tasks: BackgroundTasks):
    """Import a video from URL — downloads using multi-strategy downloader."""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL is required")

    video_id = str(uuid.uuid4())
    name = extract_name_from_url(url)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO videos
               (id, name, filename, file_path, duration, status, source_url, wizard_step)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_id, name, "url_import", "", 0, "downloading", url, "upload"),
        )
        await db.commit()

    background_tasks.add_task(_download_video_from_url, video_id, url, name)

    logger.info(f"[{video_id[:8]}] URL import started: {name}")
    return {"id": video_id, "status": "downloading", "message": "Downloading video..."}


async def _download_video_from_url(video_id: str, url: str, name: str):
    """Background: download video using MediaIngestion strategies."""
    # Check for user-uploaded cookies
    cookie_path = os.path.join(COOKIE_DIR, "cookies.txt")
    if not os.path.isfile(cookie_path):
        cookie_path = None

    ingestion = MediaIngestion(cookie_path=cookie_path)
    result = await ingestion.download(url, UPLOAD_DIR)

    async with aiosqlite.connect(DB_PATH) as db:
        if result.success:
            # Rename to include video_id prefix
            ext = os.path.splitext(result.file_path)[1]
            safe_name = f"{name.rsplit('.', 1)[0] if '.' in name else name}{ext}"
            final_path = os.path.join(UPLOAD_DIR, f"{video_id}_{safe_name}")
            os.rename(result.file_path, final_path)

            duration = 0
            try:
                duration = await get_video_duration(final_path)
            except Exception:
                pass

            await db.execute(
                "UPDATE videos SET filename = ?, file_path = ?, duration = ?, status = ? WHERE id = ?",
                (safe_name, final_path, duration, "ready", video_id),
            )
            logger.info(f"[{video_id[:8]}] Video ready (via {result.strategy})")
        else:
            await db.execute(
                "UPDATE videos SET status = ?, filename = ? WHERE id = ?",
                ("url", f"error: {result.error[:200]}", video_id),
            )
            logger.warning(f"[{video_id[:8]}] Download failed: {result.error}")
        await db.commit()


@app.post("/api/upload-cookies")
async def upload_cookies(file: UploadFile = File(...)):
    """Upload a cookies.txt file for authenticated video downloads."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    os.makedirs(COOKIE_DIR, exist_ok=True)
    cookie_path = os.path.join(COOKIE_DIR, "cookies.txt")

    content = await file.read()
    if len(content) > 1024 * 1024:  # Max 1MB
        raise HTTPException(413, "Cookie file too large")

    with open(cookie_path, "wb") as f:
        f.write(content)

    logger.info(f"Cookies uploaded: {len(content)} bytes")
    return {"status": "ok", "message": "Cookies saved. URL imports will now use them."}


# ---------------------------------------------------------------------------
# Transcription & Analysis (offline batch pipeline)
# ---------------------------------------------------------------------------


@app.post("/api/videos/{video_id}/transcribe")
async def transcribe_video(video_id: str, background_tasks: BackgroundTasks):
    """Start transcription job (faster-whisper + pyannote)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")
        if row["status"] not in ("ready", "complete"):
            raise HTTPException(400, f"Video not ready for transcription (status={row['status']})")

        file_path = row["file_path"]
        if not file_path or not os.path.isfile(file_path):
            raise HTTPException(400, "Video file not found on disk")

        await db.execute(
            """UPDATE videos
               SET wizard_step = 'transcribing', transcribe_progress = 0, progress_detail = 'Starting transcription...'
               WHERE id = ?""",
            (video_id,),
        )
        await db.commit()

    background_tasks.add_task(run_transcription, video_id, file_path)
    return {"status": "transcribing", "message": "Transcription started"}


@app.post("/api/videos/{video_id}/upload-transcript")
async def upload_transcript(
    video_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
):
    """Upload an existing transcript (VTT/SRT/text file) instead of running Whisper."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

    if not file:
        raise HTTPException(400, "Transcript file is required")

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(400, "Transcript file is empty")

    segments = parse_transcript(content, file.filename or "")
    if not segments:
        raise HTTPException(400, "Could not parse any segments from the transcript")

    # Store segments in DB
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
        for seg in segments:
            await db.execute(
                """INSERT INTO transcript_segments
                   (video_id, start_time, end_time, text, speaker)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, seg["start"], seg["end"], seg["text"], seg.get("speaker", "")),
            )

        await db.execute(
            """UPDATE videos
               SET wizard_step = 'transcribed',
                   transcribe_progress = 100,
                   progress_detail = ?
               WHERE id = ?""",
            (f"Transcript uploaded: {len(segments)} segments", video_id),
        )
        await db.commit()

    return {"status": "transcribed", "segments": len(segments), "message": f"Uploaded {len(segments)} transcript segments"}


@app.post("/api/videos/{video_id}/upload-transcript-text")
async def upload_transcript_text(video_id: str, body: dict):
    """Upload transcript as plain text (pasted by user)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "Transcript text is required")

    segments = parse_transcript(text)
    if not segments:
        raise HTTPException(400, "Could not parse any segments from the text")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
        for seg in segments:
            await db.execute(
                """INSERT INTO transcript_segments
                   (video_id, start_time, end_time, text, speaker)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, seg["start"], seg["end"], seg["text"], seg.get("speaker", "")),
            )

        await db.execute(
            """UPDATE videos
               SET wizard_step = 'transcribed',
                   transcribe_progress = 100,
                   progress_detail = ?
               WHERE id = ?""",
            (f"Transcript pasted: {len(segments)} segments", video_id),
        )
        await db.commit()

    return {"status": "transcribed", "segments": len(segments)}


@app.post("/api/videos/{video_id}/analyze")
async def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    """Start batch analysis job (after transcription)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")
        if row["wizard_step"] not in ("transcribed", "complete", "error"):
            raise HTTPException(
                400,
                f"Transcription must complete before analysis (current step={row['wizard_step']})",
            )

        file_path = row["file_path"]
        if not file_path or not os.path.isfile(file_path):
            raise HTTPException(400, "Video file not found on disk")

        await db.execute(
            """UPDATE videos
               SET wizard_step = 'analyzing', analyze_progress = 0, progress_detail = 'Starting analysis...'
               WHERE id = ?""",
            (video_id,),
        )
        await db.commit()

    background_tasks.add_task(run_analysis, video_id, file_path)
    return {"status": "analyzing", "message": "Analysis started"}


@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str) -> VideoStatus:
    """Get current wizard step and progress for polling."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

        step = row["wizard_step"] if "wizard_step" in row.keys() else "upload"

        # Return the appropriate progress depending on the step
        if step in ("transcribing",):
            progress = row["transcribe_progress"] if "transcribe_progress" in row.keys() else 0
        elif step in ("analyzing",):
            progress = row["analyze_progress"] if "analyze_progress" in row.keys() else 0
        elif step == "complete":
            progress = 100
        else:
            progress = 0

        detail = row["progress_detail"] if "progress_detail" in row.keys() else ""

        return VideoStatus(step=step, progress=progress, detail=detail)


@app.get("/api/videos/{video_id}/analysis-stream")
async def analysis_stream(video_id: str):
    """SSE endpoint for real-time analysis events."""
    q = subscribe_events(video_id)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            unsubscribe_events(video_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Partial results (progressive display during analysis)
# ---------------------------------------------------------------------------


@app.get("/api/videos/{video_id}/partial-results")
async def get_partial_results(video_id: str):
    """Return whatever analysis data is available so far for progressive display."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = await cursor.fetchone()
        if not video:
            raise HTTPException(404, "Video not found")

        ready_sections: list[str] = []
        result: dict = {"video": _build_video_out(video).model_dump(), "ready_sections": ready_sections}

        # Transcript
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM transcript_segments WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("transcript")
            cursor = await db.execute(
                "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time", (video_id,)
            )
            rows = await cursor.fetchall()
            result["transcript"] = [
                {"start_time": r["start_time"], "end_time": r["end_time"],
                 "text": r["text"], "speaker": r["speaker"]}
                for r in rows
            ]

        # Participants
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM participants WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("participants")
            cursor = await db.execute(
                "SELECT * FROM participants WHERE video_id = ?", (video_id,)
            )
            rows = await cursor.fetchall()
            result["participants"] = [
                {
                    "id": r["id"], "video_id": r["video_id"], "name": r["name"],
                    "engagement_score": r["engagement_score"],
                    "sentiment_score": r["sentiment_score"],
                    "speaking_pct": r["speaking_pct"],
                    "clarity_score": r["clarity_score"],
                    "rapport_score": r["rapport_score"],
                    "energy_score": r["energy_score"],
                }
                for r in rows
            ]

        # Emotions (from frame analysis)
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM emotions WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("emotions")
            cursor = await db.execute(
                "SELECT * FROM emotions WHERE video_id = ? ORDER BY timestamp", (video_id,)
            )
            rows = await cursor.fetchall()
            result["emotions"] = [
                {"timestamp": r["timestamp"], "emotion": r["emotion"],
                 "confidence": r["confidence"], "engagement": r["engagement"],
                 "participant_id": r["participant_id"]}
                for r in rows
            ]

        # Voice signals
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM voice_signals WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("voice_signals")

        # Word signals
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM word_signals WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("word_signals")

        # Summary
        cursor = await db.execute(
            "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
        )
        row = await cursor.fetchone()
        if row:
            ready_sections.append("summary")
            result["summary"] = {
                "summary": row["summary"],
                "key_topics": json.loads(row["key_topics"]),
                "overall_sentiment": row["overall_sentiment"],
            }

        # Flags
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM flags WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("flags")
            cursor = await db.execute(
                "SELECT * FROM flags WHERE video_id = ? ORDER BY timestamp", (video_id,)
            )
            rows = await cursor.fetchall()
            result["flags"] = [
                {"id": r["id"], "timestamp": r["timestamp"], "type": r["type"],
                 "description": r["description"], "severity": r["severity"]}
                for r in rows
            ]

        # Coaching
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM coaching WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("coaching")

        return result


# ---------------------------------------------------------------------------
# Signal panel API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/videos/{video_id}/signals/snapshots")
async def get_signal_snapshots(video_id: str):
    """Frame-by-frame visual signals — emotions, body language, gestures, reactions per participant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT timestamp, participant_id, emotion, emotion_confidence, engagement,
                      posture, openness, leaning, mirroring, body_language_desc,
                      gestures, reactions, confidence
               FROM signal_snapshots WHERE video_id = ?
               ORDER BY timestamp""",
            (video_id,),
        )
        rows = await cursor.fetchall()

    # Get participant names
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        p_cursor = await db.execute("SELECT id, name FROM participants WHERE video_id = ?", (video_id,))
        p_rows = await p_cursor.fetchall()
    pid_to_name = {r["id"]: r["name"] for r in p_rows}

    # Group by timestamp into SignalSnapshot format
    from collections import defaultdict
    by_ts = defaultdict(list)

    for r in rows:
        conf = 1.0
        try:
            conf = r["confidence"] if r["confidence"] is not None else 1.0
        except (IndexError, KeyError):
            pass
        by_ts[r["timestamp"]].append({
            "label": pid_to_name.get(r["participant_id"], r["participant_id"]),
            "emotions": {
                "primary": r["emotion"],
                "confidence": r["emotion_confidence"],
                "engagement": r["engagement"],
            },
            "body_language": {
                "posture": r["posture"],
                "openness": r["openness"],
                "leaning": r["leaning"],
                "mirroring": bool(r["mirroring"]),
                "description": r["body_language_desc"] or "",
            },
            "gestures": json.loads(r["gestures"]) if r["gestures"] else [],
            "reactions": json.loads(r["reactions"]) if r["reactions"] else [],
            "confidence": conf,
        })

    return [{"timestamp": ts, "participants": parts} for ts, parts in sorted(by_ts.items())]


@app.get("/api/videos/{video_id}/signals/voice")
async def get_voice_signals(video_id: str):
    """Voice signal timeline — tone, pace, energy, dynamics."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, tone, pace, energy, dynamics FROM voice_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        return [dict(r) for r in await cursor.fetchall()]


@app.get("/api/videos/{video_id}/signals/words")
async def get_word_signals(video_id: str):
    """Word-level signals — sentiment, keywords, key phrases."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, sentiment, sentiment_score, keywords, key_phrases FROM word_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [{
        "timestamp": r["timestamp"],
        "sentiment": r["sentiment"],
        "sentiment_score": r["sentiment_score"],
        "keywords": json.loads(r["keywords"]) if r["keywords"] else [],
        "key_phrases": json.loads(r["key_phrases"]) if r["key_phrases"] else [],
    } for r in rows]


@app.get("/api/videos/{video_id}/signals/personality")
async def get_personality_signals(video_id: str):
    """Personality profiles per participant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT ps.*, p.name FROM personality_signals ps JOIN participants p ON ps.participant_id = p.id WHERE ps.video_id = ?",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return {
        "participants": [{
            "label": r["name"],
            "traits": json.loads(r["traits"]) if r["traits"] else [],
            "communication_style": r["communication_style"] or "",
            "confidence_level": r["confidence_level"] or "",
        } for r in rows]
    }


@app.get("/api/videos/{video_id}/signals/correlations")
async def get_correlations(video_id: str):
    """Cross-signal correlation patterns."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT signal_pair, pattern, confidence, implication FROM correlations WHERE video_id = ?",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [{
        "signal_pair": json.loads(r["signal_pair"]) if r["signal_pair"] else [],
        "pattern": r["pattern"],
        "confidence": r["confidence"],
        "implication": r["implication"],
    } for r in rows]


@app.get("/api/videos/{video_id}/signals/audio-features")
async def get_audio_features(video_id: str):
    """Audio features from librosa analysis (voice signals as proxy)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, tone, pace, energy, dynamics FROM voice_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Video thumbnail
# ---------------------------------------------------------------------------


@app.get("/api/videos/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Return a JPEG thumbnail from 25% into the video."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT file_path, duration FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

    file_path = row["file_path"]
    duration = row["duration"] or 10

    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(404, "Video file not found")

    thumb_dir = os.path.join(PROCESSED_DIR, video_id)
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, "thumb.jpg")

    if not os.path.exists(thumb_path):
        ts = max(1.0, duration * 0.25)
        try:
            extracted = await extract_single_frame(file_path, ts, thumb_dir)
            # Rename to thumb.jpg
            if extracted and extracted != thumb_path:
                os.rename(extracted, thumb_path)
        except Exception as e:
            logger.warning(f"Thumbnail extraction failed for {video_id}: {e}")
            raise HTTPException(500, "Failed to generate thumbnail")

    return FileResponse(thumb_path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Video CRUD
# ---------------------------------------------------------------------------


def _build_video_out(r) -> VideoOut:
    """Construct a VideoOut from a DB row, handling missing columns gracefully."""
    keys = r.keys()
    return VideoOut(
        id=r["id"],
        name=r["name"],
        filename=r["filename"],
        duration=r["duration"],
        status=r["status"],
        created_at=r["created_at"],
        overall_score=r["overall_score"],
        source_url=r["source_url"] if "source_url" in keys else "",
        wizard_step=r["wizard_step"] if "wizard_step" in keys else "upload",
        transcribe_progress=r["transcribe_progress"] if "transcribe_progress" in keys else 0,
        analyze_progress=r["analyze_progress"] if "analyze_progress" in keys else 0,
        progress_detail=r["progress_detail"] if "progress_detail" in keys else "",
    )


@app.get("/api/videos")
async def list_videos() -> list[VideoOut]:
    """List all uploaded videos."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM videos ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [_build_video_out(r) for r in rows]


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its associated data."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Check video exists
        cursor = await db.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

        file_path = row[0]

        # Delete all related data
        for table in [
            "emotions", "transcript_segments", "flags", "coaching",
            "summaries", "signal_snapshots", "voice_signals", "word_signals",
            "personality_signals", "pre_analysis_cache", "correlations",
            "meeting_analytics", "face_mappings",
            "meeting_notes", "voting_log", "speaker_audio_features",
        ]:
            await db.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))
        await db.execute("DELETE FROM participants WHERE video_id = ?", (video_id,))
        await db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        await db.commit()

    # Delete the file from disk
    if file_path and os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    # Delete processed directory (extracted frames, audio chunks, etc.)
    import shutil
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "processed", video_id)
    if os.path.isdir(processed_dir):
        try:
            shutil.rmtree(processed_dir)
        except OSError:
            pass

    return {"status": "ok", "message": "Video deleted"}


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str) -> VideoOut:
    """Get video details and processing status."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        r = await cursor.fetchone()
        if not r:
            raise HTTPException(404, "Video not found")
        return _build_video_out(r)


# ---------------------------------------------------------------------------
# Data retrieval endpoints (unchanged from original)
# ---------------------------------------------------------------------------


@app.get("/api/videos/{video_id}/participants")
async def get_participants(video_id: str) -> list[ParticipantOut]:
    """List all participants for a video."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM participants WHERE video_id = ?", (video_id,)
        )
        rows = await cursor.fetchall()
        return [
            ParticipantOut(
                id=r["id"], video_id=r["video_id"], name=r["name"],
                engagement_score=r["engagement_score"],
                sentiment_score=r["sentiment_score"],
                speaking_pct=r["speaking_pct"],
                clarity_score=r["clarity_score"],
                rapport_score=r["rapport_score"],
                energy_score=r["energy_score"],
            )
            for r in rows
        ]


@app.get("/api/videos/{video_id}/emotions")
async def get_emotions(video_id: str) -> list[EmotionPoint]:
    """Get emotion timeline data for all participants."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Join with participants to return name instead of raw UUID
        cursor = await db.execute(
            """SELECT e.timestamp, e.emotion, e.confidence, e.engagement,
                      COALESCE(p.name, e.participant_id) as participant_name
               FROM emotions e
               LEFT JOIN participants p ON e.participant_id = p.id
               WHERE e.video_id = ?
               ORDER BY e.timestamp""",
            (video_id,),
        )
        rows = await cursor.fetchall()
        return [
            EmotionPoint(
                timestamp=r["timestamp"], emotion=r["emotion"],
                confidence=r["confidence"], engagement=r["engagement"],
                participant_id=r["participant_name"],
            )
            for r in rows
        ]


@app.get("/api/videos/{video_id}/transcript")
async def get_transcript(video_id: str) -> list[TranscriptSegment]:
    """Get full transcript with timestamps."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
            (video_id,),
        )
        rows = await cursor.fetchall()
        return [
            TranscriptSegment(
                start_time=r["start_time"], end_time=r["end_time"],
                text=r["text"], speaker=r["speaker"],
                participant_id=r["participant_id"],
            )
            for r in rows
        ]


@app.get("/api/videos/{video_id}/flags")
async def get_flags(video_id: str) -> list[Flag]:
    """Get critical moment flags."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM flags WHERE video_id = ? ORDER BY timestamp", (video_id,)
        )
        rows = await cursor.fetchall()
        return [
            Flag(
                id=r["id"], timestamp=r["timestamp"], type=r["type"],
                description=r["description"], severity=r["severity"],
            )
            for r in rows
        ]


@app.get("/api/videos/{video_id}/coaching/{participant_id}")
async def get_coaching(video_id: str, participant_id: str) -> list[CoachingItem]:
    """Get coaching recommendations for a participant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM coaching WHERE video_id = ? AND participant_id = ? ORDER BY priority",
            (video_id, participant_id),
        )
        rows = await cursor.fetchall()
        return [
            CoachingItem(
                id=r["id"], recommendation=r["recommendation"],
                category=r["category"], priority=r["priority"],
            )
            for r in rows
        ]


@app.get("/api/videos/{video_id}/summary")
async def get_summary(video_id: str) -> SummaryOut:
    """Get AI-generated meeting summary."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
        )
        r = await cursor.fetchone()
        if not r:
            raise HTTPException(404, "Summary not found")
        return SummaryOut(
            summary=r["summary"],
            key_topics=json.loads(r["key_topics"]),
            overall_sentiment=r["overall_sentiment"],
        )


@app.get("/api/videos/{video_id}/meeting-notes")
async def get_meeting_notes(video_id: str):
    """Get structured meeting notes (action items, decisions, follow-ups)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM meeting_notes WHERE video_id = ?", (video_id,)
        )
        r = await cursor.fetchone()
        if not r:
            return {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}
        return {
            "action_items": json.loads(r["action_items"]),
            "decisions": json.loads(r["decisions"]),
            "follow_ups": json.loads(r["follow_ups"]),
            "key_questions": json.loads(r["key_questions"]),
        }


@app.get("/api/videos/{video_id}/signals/speaker-audio")
async def get_speaker_audio_features(video_id: str):
    """Per-speaker audio features from librosa analysis."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT saf.*, p.name as participant_name
               FROM speaker_audio_features saf
               JOIN participants p ON saf.participant_id = p.id
               WHERE saf.video_id = ? ORDER BY p.name, saf.timestamp""",
            (video_id,),
        )
        rows = await cursor.fetchall()

    from collections import defaultdict
    by_participant = defaultdict(list)
    for r in rows:
        by_participant[r["participant_name"]].append({
            "timestamp": r["timestamp"],
            "pitch_mean": r["pitch_mean"],
            "pitch_std": r["pitch_std"],
            "volume_energy": r["volume_energy"],
            "speaking_rate": r["speaking_rate"],
            "pause_ratio": r["pause_ratio"],
            "spectral_centroid": r["spectral_centroid"],
            "engagement_score": r["engagement_score"],
        })

    return {
        "participants": [
            {"name": name, "features": feats}
            for name, feats in by_participant.items()
        ]
    }


@app.get("/api/videos/{video_id}/signals/confidence")
async def get_confidence_summary(video_id: str):
    """Get voting confidence summary for a video."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT AVG(confidence) as avg_conf, MIN(confidence) as min_conf, COUNT(*) as total FROM signal_snapshots WHERE video_id = ?",
            (video_id,),
        )
        row = await cursor.fetchone()
        avg_conf = row["avg_conf"] if row and row["avg_conf"] else 1.0
        min_conf = row["min_conf"] if row and row["min_conf"] else 1.0
        total = row["total"] if row else 0

        disagreement_count = 0
        try:
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt FROM voting_log WHERE video_id = ? AND disagreements != '[]'",
                (video_id,),
            )
            row = await cursor.fetchone()
            disagreement_count = row["cnt"] if row else 0
        except Exception:
            pass

    return {
        "avg_confidence": round(avg_conf, 2),
        "min_confidence": round(min_conf, 2),
        "total_snapshots": total,
        "disagreement_count": disagreement_count,
        "accuracy_grade": "A" if avg_conf >= 0.8 else "B" if avg_conf >= 0.6 else "C" if avg_conf >= 0.4 else "D",
    }


@app.get("/api/videos/{video_id}/results")
async def get_results(video_id: str) -> AnalysisResults:
    """Get full analysis results for a video."""
    video = await get_video(video_id)
    participants = await get_participants(video_id)
    emotions = await get_emotions(video_id)
    transcript = await get_transcript(video_id)
    flags = await get_flags(video_id)
    try:
        summary = await get_summary(video_id)
    except HTTPException:
        summary = None
    notes = await get_meeting_notes(video_id)

    return AnalysisResults(
        video=video,
        participants=participants,
        emotions=emotions,
        transcript=transcript,
        flags=flags,
        summary=summary,
        meeting_notes=MeetingNotesOut(**notes) if notes else None,
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@app.get("/api/analytics")
async def list_analytics():
    """List analytics for all analyzed meetings."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT ma.*, v.name as video_name
            FROM meeting_analytics ma
            JOIN videos v ON ma.video_id = v.id
            ORDER BY ma.meeting_date DESC
        """)
        rows = await cursor.fetchall()
        return [
            MeetingAnalytics(
                video_id=r["video_id"],
                video_name=r["video_name"],
                avg_engagement=r["avg_engagement"],
                avg_visual_engagement=r["avg_visual_engagement"],
                avg_audio_engagement=r["avg_audio_engagement"],
                avg_verbal_engagement=r["avg_verbal_engagement"],
                participant_count=r["participant_count"],
                duration=r["duration"],
                overall_sentiment=r["overall_sentiment"],
                meeting_date=r["meeting_date"],
            )
            for r in rows
        ]


@app.get("/api/analytics/compare")
async def compare_meetings(ids: str):
    """Compare two meetings side by side. ids=uuid1,uuid2"""
    video_ids = [vid.strip() for vid in ids.split(",") if vid.strip()]
    if len(video_ids) < 2:
        raise HTTPException(400, "Provide at least 2 video IDs separated by commas")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ",".join("?" * len(video_ids))
        cursor = await db.execute(
            f"""SELECT ma.*, v.name as video_name
                FROM meeting_analytics ma
                JOIN videos v ON ma.video_id = v.id
                WHERE ma.video_id IN ({placeholders})""",
            video_ids,
        )
        rows = await cursor.fetchall()

        meetings = [
            MeetingAnalytics(
                video_id=r["video_id"],
                video_name=r["video_name"],
                avg_engagement=r["avg_engagement"],
                avg_visual_engagement=r["avg_visual_engagement"],
                avg_audio_engagement=r["avg_audio_engagement"],
                avg_verbal_engagement=r["avg_verbal_engagement"],
                participant_count=r["participant_count"],
                duration=r["duration"],
                overall_sentiment=r["overall_sentiment"],
                meeting_date=r["meeting_date"],
            )
            for r in rows
        ]

        # Compute deltas if exactly 2 meetings
        comparison = {"meetings": [m.model_dump() for m in meetings]}
        if len(meetings) == 2:
            a, b = meetings[0], meetings[1]
            comparison["deltas"] = {
                "engagement": round(b.avg_engagement - a.avg_engagement, 1),
                "visual": round(b.avg_visual_engagement - a.avg_visual_engagement, 1),
                "audio": round(b.avg_audio_engagement - a.avg_audio_engagement, 1),
                "verbal": round(b.avg_verbal_engagement - a.avg_verbal_engagement, 1),
                "participants": b.participant_count - a.participant_count,
            }

        return comparison


# ---------------------------------------------------------------------------
# Participant management
# ---------------------------------------------------------------------------


@app.post("/api/videos/{video_id}/participants/{participant_id}/rename")
async def rename_participant(video_id: str, participant_id: str, req: RenameParticipantRequest):
    """Assign a custom name to a participant."""
    new_name = req.name.strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")
    if len(new_name) > 100:
        raise HTTPException(400, "Name too long (max 100 characters)")

    async with aiosqlite.connect(DB_PATH) as db:
        # Update participant name
        result = await db.execute(
            "UPDATE participants SET name = ? WHERE id = ? AND video_id = ?",
            (new_name, participant_id, video_id),
        )
        if result.rowcount == 0:
            raise HTTPException(404, "Participant not found")

        # Store in face_mappings for future reference
        await db.execute(
            """INSERT OR REPLACE INTO face_mappings (video_id, participant_label, custom_name)
               VALUES (?, ?, ?)""",
            (video_id, participant_id, new_name),
        )
        await db.commit()

    logger.info(f"Renamed participant {participant_id} to '{new_name}' in video {video_id[:8]}")
    return {"status": "ok", "name": new_name}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@app.get("/api/videos/{video_id}/export/pdf")
async def export_pdf(video_id: str):
    """Export analysis results as PDF report."""
    try:
        pdf_bytes = await generate_pdf_report(video_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=insightai_report_{video_id[:8]}.pdf"},
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/videos/{video_id}/export/csv")
async def export_csv(video_id: str):
    """Export analysis results as ZIP of CSV files."""
    try:
        zip_bytes = await generate_csv_export(video_id)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=insightai_data_{video_id[:8]}.zip"},
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


# Serve uploaded files for video playback
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
