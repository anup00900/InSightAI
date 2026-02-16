from __future__ import annotations

import os
import uuid
import json
import asyncio
import logging
import aiosqlite
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .models import init_db, DB_PATH
from .schemas import (
    VideoOut, ParticipantOut, EmotionPoint, TranscriptSegment,
    Flag, CoachingItem, SummaryOut, AnalysisResults, ImportUrlRequest,
    MeetingAnalytics, JoinMeetingRequest, RenameParticipantRequest,
)
from .recall_adapter import RecallAiAdapter
from .pipeline import run_analysis_pipeline
from .realtime_ws import websocket_analyze
from .video_processor import is_video_file, is_audio_file, get_video_duration
from .media_ingestion import MediaIngestion, extract_name_from_url, COOKIE_DIR
from .export import generate_pdf_report, generate_csv_export

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="InsightAI - Conversation Intelligence", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "INSERT INTO videos (id, name, filename, file_path, duration, status) VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, file.filename, file.filename, file_path, 0, "ready"),
        )
        await db.commit()

    # Compute duration in background (ffprobe) — doesn't block response
    background_tasks.add_task(_update_duration, video_id, file_path)

    return {"id": video_id, "status": "ready", "message": "Ready for live analysis"}


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
            "INSERT INTO videos (id, name, filename, file_path, duration, status, source_url) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, name, "url_import", "", 0, "downloading", url),
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


@app.get("/api/videos")
async def list_videos() -> list[VideoOut]:
    """List all uploaded videos."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM videos ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            VideoOut(
                id=r["id"], name=r["name"], filename=r["filename"],
                duration=r["duration"], status=r["status"],
                created_at=r["created_at"], overall_score=r["overall_score"],
                source_url=r["source_url"] if "source_url" in r.keys() else "",
            )
            for r in rows
        ]


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
        return VideoOut(
            id=r["id"], name=r["name"], filename=r["filename"],
            duration=r["duration"], status=r["status"],
            created_at=r["created_at"], overall_score=r["overall_score"],
            source_url=r["source_url"] if "source_url" in r.keys() else "",
        )


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
        cursor = await db.execute(
            "SELECT * FROM emotions WHERE video_id = ? ORDER BY timestamp", (video_id,)
        )
        rows = await cursor.fetchall()
        return [
            EmotionPoint(
                timestamp=r["timestamp"], emotion=r["emotion"],
                confidence=r["confidence"], engagement=r["engagement"],
                participant_id=r["participant_id"],
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

    return AnalysisResults(
        video=video,
        participants=participants,
        emotions=emotions,
        transcript=transcript,
        flags=flags,
        summary=summary,
    )


@app.websocket("/ws/analyze/{video_id}")
async def ws_analyze(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time video analysis."""
    await websocket_analyze(websocket, video_id)



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


# ─── Live Meeting Integration ─────────────────────────────────────

_recall = RecallAiAdapter()


@app.post("/api/meetings/join")
async def join_meeting(req: JoinMeetingRequest):
    """Send a bot to join a live meeting for analysis."""
    if not _recall.is_configured:
        raise HTTPException(503, "Live meeting integration not configured. Set RECALL_API_KEY.")

    bot = await _recall.join_meeting(req.meeting_url, req.bot_name)
    if bot.status == "error":
        raise HTTPException(502, f"Failed to join meeting: {bot.error}")

    return {"bot_id": bot.id, "status": bot.status, "message": "Bot is joining the meeting..."}


@app.get("/api/meetings/{bot_id}/status")
async def get_meeting_status(bot_id: str):
    """Get the current status of a meeting bot."""
    status = await _recall.get_status(bot_id)
    return {
        "bot_id": status.bot_id,
        "status": status.status,
        "participants": status.participants,
        "duration": status.duration,
        "recording_available": status.recording_available,
    }


@app.post("/api/meetings/{bot_id}/leave")
async def leave_meeting(bot_id: str):
    """Tell the bot to leave the meeting."""
    success = await _recall.leave_meeting(bot_id)
    if not success:
        raise HTTPException(502, "Failed to remove bot from meeting")
    return {"status": "ok", "message": "Bot is leaving the meeting"}


@app.get("/api/meetings/{bot_id}/recording")
async def get_meeting_recording(bot_id: str, background_tasks: BackgroundTasks):
    """Get the recording URL and optionally import it for analysis."""
    recording_url = await _recall.get_recording(bot_id)
    if not recording_url:
        raise HTTPException(404, "Recording not yet available")
    return {"recording_url": recording_url}


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
