"""Recall.ai adapter for live meeting integration.

Wraps the Recall.ai REST API for:
- Creating meeting bots that join Zoom/Teams/Meet calls
- Polling bot status
- Retrieving recordings after meetings end
- Webhook handling for real-time audio/video streams

Requires RECALL_API_KEY environment variable.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

import httpx

logger = logging.getLogger(__name__)

RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://us-west-2.recall.ai/api/v1")


@dataclass
class MeetingBot:
    id: str
    meeting_url: str
    bot_name: str
    status: str
    recording_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MeetingStatus:
    bot_id: str
    status: str  # joining, in_call, recording, done, error
    participants: List[str] = field(default_factory=list)
    duration: float = 0
    recording_available: bool = False


class RecallAiAdapter:
    """Adapter for the Recall.ai meeting bot API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or RECALL_API_KEY
        if not self.api_key:
            logger.warning("RECALL_API_KEY not set — live meeting integration disabled")

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def join_meeting(self, meeting_url: str, bot_name: str = "LensAI Bot") -> MeetingBot:
        """Create a bot to join the meeting.

        Args:
            meeting_url: The Zoom/Teams/Meet meeting URL
            bot_name: Display name for the bot in the meeting

        Returns:
            MeetingBot with the bot ID and initial status
        """
        if not self.is_configured:
            return MeetingBot(
                id="", meeting_url=meeting_url, bot_name=bot_name,
                status="error", error="Recall.ai API key not configured",
            )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{RECALL_BASE_URL}/bot",
                    headers=self._headers,
                    json={
                        "meeting_url": meeting_url,
                        "bot_name": bot_name,
                        "transcription_options": {
                            "provider": "default",
                        },
                        "recording_mode": "speaker_view",
                    },
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    bot = MeetingBot(
                        id=data.get("id", ""),
                        meeting_url=meeting_url,
                        bot_name=bot_name,
                        status=data.get("status", {}).get("code", "joining"),
                    )
                    logger.info(f"Recall.ai bot created: {bot.id} for {meeting_url}")
                    return bot
                else:
                    error = resp.text[:200]
                    logger.error(f"Recall.ai bot creation failed: {resp.status_code} {error}")
                    return MeetingBot(
                        id="", meeting_url=meeting_url, bot_name=bot_name,
                        status="error", error=f"API error: {resp.status_code}",
                    )
        except Exception as e:
            logger.error(f"Recall.ai request failed: {e}")
            return MeetingBot(
                id="", meeting_url=meeting_url, bot_name=bot_name,
                status="error", error=str(e)[:200],
            )

    async def get_status(self, bot_id: str) -> MeetingStatus:
        """Get the current status of a meeting bot."""
        if not self.is_configured:
            return MeetingStatus(bot_id=bot_id, status="error")

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{RECALL_BASE_URL}/bot/{bot_id}",
                    headers=self._headers,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    status_data = data.get("status", {})
                    return MeetingStatus(
                        bot_id=bot_id,
                        status=status_data.get("code", "unknown"),
                        participants=[
                            p.get("name", "Unknown")
                            for p in data.get("meeting_participants", [])
                        ],
                        duration=data.get("meeting_metadata", {}).get("duration", 0),
                        recording_available=bool(data.get("video_url")),
                    )
                else:
                    return MeetingStatus(bot_id=bot_id, status="error")
        except Exception as e:
            logger.error(f"Recall.ai status check failed: {e}")
            return MeetingStatus(bot_id=bot_id, status="error")

    async def get_recording(self, bot_id: str) -> Optional[str]:
        """Get the recording URL for a completed meeting."""
        if not self.is_configured:
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{RECALL_BASE_URL}/bot/{bot_id}",
                    headers=self._headers,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    video_url = data.get("video_url")
                    if video_url:
                        logger.info(f"Recording available for bot {bot_id}")
                        return video_url
                return None
        except Exception as e:
            logger.error(f"Recall.ai recording fetch failed: {e}")
            return None

    async def leave_meeting(self, bot_id: str) -> bool:
        """Tell the bot to leave the meeting."""
        if not self.is_configured:
            return False

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{RECALL_BASE_URL}/bot/{bot_id}/leave_call",
                    headers=self._headers,
                )
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Recall.ai leave failed: {e}")
            return False
