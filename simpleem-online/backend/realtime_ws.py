"""WebSocket endpoint handler for real-time video analysis."""

import json
import logging

import aiosqlite
from fastapi import WebSocket, WebSocketDisconnect

from .models import DB_PATH
from .realtime_pipeline import RealtimeSession

logger = logging.getLogger("realtime_ws")


async def websocket_analyze(ws: WebSocket, video_id: str):
    """WebSocket handler at /ws/analyze/{video_id}.

    Protocol messages (client → server):
      {"action": "tick", "timestamp": 12.5}
      {"action": "play"}
      {"action": "pause"}
      {"action": "seek", "timestamp": 30.0}
      {"action": "video_ended"}

    Server → client messages have {"type": ..., "data": ...} shape.
    """
    await ws.accept()

    # Look up video file path from DB
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT file_path, status FROM videos WHERE id = ?", (video_id,)
        )
        row = await cursor.fetchone()

    if not row:
        await ws.send_json({"type": "error", "data": {"message": "Video not found"}})
        await ws.close()
        return

    video_path = row["file_path"]

    # Create session
    session = RealtimeSession(ws, video_id, video_path)
    await session.initialize()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action", "")
            logger.info(f"WS [{video_id[:8]}] action={action} {msg}")

            if action == "tick":
                ts = float(msg.get("timestamp", 0))
                await session.handle_tick(ts)

            elif action == "play":
                await session.handle_play()

            elif action == "pause":
                await session.handle_pause()

            elif action == "seek":
                ts = float(msg.get("timestamp", 0))
                await session.handle_seek(ts)

            elif action == "video_ended":
                await session.finalize()

    except WebSocketDisconnect:
        logger.info(f"WS [{video_id[:8]}] client disconnected")
    except Exception as e:
        logger.error(f"WS [{video_id[:8]}] error: {e}")
    finally:
        await session.cleanup()
