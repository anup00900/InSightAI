# LensAI v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade LensAI POC to production-grade conversation intelligence with Simpleem-style 55/38/7 signal weighting, robust URL import, real audio analysis, speaker diarization, engagement alerts, live meeting integration, and export capabilities.

**Architecture:** Three independent signal pipelines (Visual 55%, Audio 38%, Text 7%) emit normalized events to a central SignalBus. The EngagementEngine subscribes to the bus and computes holistic per-participant engagement scores with configurable threshold alerts.

**Tech Stack:** FastAPI + SQLite (backend), React + Vite + Tailwind + Recharts (frontend), Core42/Azure OpenAI (GPT-4o Vision + Whisper), librosa (audio analysis), pyannote.audio (speaker diarization), Recall.ai (live meetings), weasyprint (PDF export).

**Base directory:** `/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc`

---

## Phase 1: Core Pipeline Refactor

### Task 1: Install New Dependencies

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Update requirements.txt**

Add these lines to `backend/requirements.txt`:

```
httpx==0.28.1
librosa>=0.10.0
numpy>=1.26.0
soundfile>=0.12.0
```

Note: pyannote.audio, torch, weasyprint, matplotlib, and face_recognition are deferred to later phases to keep Phase 1 lightweight.

**Step 2: Install dependencies**

Run: `pip3 install -r backend/requirements.txt`
Expected: All packages install successfully. librosa pulls in numpy, soundfile, etc.

**Step 3: Verify imports**

Run: `python3 -c "import librosa; import httpx; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add librosa and httpx dependencies for Phase 1"
```

---

### Task 2: Create SignalBus

The SignalBus is the central event bus that decouples signal producers (pipelines) from consumers (EngagementEngine, WebSocket, DB).

**Files:**
- Create: `backend/signal_bus.py`
- Create: `backend/tests/test_signal_bus.py`

**Step 1: Write the failing test**

Create `backend/tests/__init__.py` (empty) and `backend/tests/test_signal_bus.py`:

```python
import asyncio
import pytest
from backend.signal_bus import SignalBus, SignalEvent, SignalType


@pytest.mark.asyncio
async def test_publish_and_subscribe():
    bus = SignalBus()
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    bus.subscribe(SignalType.VISUAL, handler)
    event = SignalEvent(
        signal_type=SignalType.VISUAL,
        participant_id="Person 1",
        timestamp=10.0,
        data={"engagement": 75, "emotion": "engaged"},
    )
    await bus.publish(event)
    await asyncio.sleep(0.05)  # Let handler run

    assert len(received) == 1
    assert received[0].participant_id == "Person 1"
    assert received[0].data["engagement"] == 75


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = SignalBus()
    results_a = []
    results_b = []

    async def handler_a(event: SignalEvent):
        results_a.append(event)

    async def handler_b(event: SignalEvent):
        results_b.append(event)

    bus.subscribe(SignalType.AUDIO, handler_a)
    bus.subscribe(SignalType.AUDIO, handler_b)

    event = SignalEvent(
        signal_type=SignalType.AUDIO,
        participant_id="Person 1",
        timestamp=5.0,
        data={"energy": 60},
    )
    await bus.publish(event)
    await asyncio.sleep(0.05)

    assert len(results_a) == 1
    assert len(results_b) == 1


@pytest.mark.asyncio
async def test_subscribe_all():
    bus = SignalBus()
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    bus.subscribe_all(handler)

    await bus.publish(SignalEvent(
        signal_type=SignalType.VISUAL, participant_id="P1", timestamp=1.0, data={}
    ))
    await bus.publish(SignalEvent(
        signal_type=SignalType.AUDIO, participant_id="P1", timestamp=2.0, data={}
    ))
    await bus.publish(SignalEvent(
        signal_type=SignalType.TEXT, participant_id="P1", timestamp=3.0, data={}
    ))
    await asyncio.sleep(0.05)

    assert len(received) == 3


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = SignalBus()
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    unsub = bus.subscribe(SignalType.VISUAL, handler)
    await bus.publish(SignalEvent(
        signal_type=SignalType.VISUAL, participant_id="P1", timestamp=1.0, data={}
    ))
    await asyncio.sleep(0.05)
    assert len(received) == 1

    unsub()
    await bus.publish(SignalEvent(
        signal_type=SignalType.VISUAL, participant_id="P1", timestamp=2.0, data={}
    ))
    await asyncio.sleep(0.05)
    assert len(received) == 1  # No new events after unsubscribe
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_signal_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.signal_bus'`

**Step 3: Implement SignalBus**

Create `backend/signal_bus.py`:

```python
"""Central event bus for signal pipelines.

All three pipelines (Visual, Audio, Text) publish normalized SignalEvents here.
The EngagementEngine and WebSocket handler subscribe to receive them.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

SignalHandler = Callable[["SignalEvent"], Awaitable[None]]


class SignalType(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    ENGAGEMENT = "engagement"  # Computed by EngagementEngine, published back


@dataclass
class SignalEvent:
    signal_type: SignalType
    participant_id: str
    timestamp: float
    data: dict = field(default_factory=dict)


class SignalBus:
    """Async publish-subscribe event bus for signal events."""

    def __init__(self):
        self._subscribers: dict[SignalType, list[SignalHandler]] = {
            st: [] for st in SignalType
        }
        self._all_subscribers: list[SignalHandler] = []

    def subscribe(self, signal_type: SignalType, handler: SignalHandler) -> Callable[[], None]:
        """Subscribe to events of a specific signal type. Returns an unsubscribe callable."""
        self._subscribers[signal_type].append(handler)

        def unsubscribe():
            try:
                self._subscribers[signal_type].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def subscribe_all(self, handler: SignalHandler) -> Callable[[], None]:
        """Subscribe to ALL signal types."""
        self._all_subscribers.append(handler)

        def unsubscribe():
            try:
                self._all_subscribers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    async def publish(self, event: SignalEvent):
        """Publish a signal event to all matching subscribers."""
        handlers = list(self._subscribers.get(event.signal_type, []))
        handlers.extend(self._all_subscribers)

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"SignalBus handler error: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_signal_bus.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add backend/signal_bus.py backend/tests/__init__.py backend/tests/test_signal_bus.py
git commit -m "feat: add SignalBus event bus for pipeline decoupling"
```

---

### Task 3: Create EngagementEngine

**Files:**
- Create: `backend/engagement_engine.py`
- Create: `backend/tests/test_engagement_engine.py`

**Step 1: Write the failing test**

Create `backend/tests/test_engagement_engine.py`:

```python
import pytest
from backend.engagement_engine import EngagementEngine, EngagementScore
from backend.signal_bus import SignalEvent, SignalType


class TestEngagementEngine:
    def setup_method(self):
        self.engine = EngagementEngine()

    def test_compute_with_all_signals(self):
        visual = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="Person 1",
            timestamp=10.0,
            data={"engagement": 80},
        )
        audio = SignalEvent(
            signal_type=SignalType.AUDIO,
            participant_id="Person 1",
            timestamp=10.0,
            data={"energy": 70},
        )
        text = SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="Person 1",
            timestamp=10.0,
            data={"sentiment_score": 0.8},  # 0-1 scale → 80
        )

        score = self.engine.compute(
            participant_id="Person 1",
            timestamp=10.0,
            visual_signal=visual,
            audio_signal=audio,
            text_signal=text,
        )

        assert isinstance(score, EngagementScore)
        assert score.participant_id == "Person 1"
        assert score.timestamp == 10.0
        # 0.55*80 + 0.38*70 + 0.07*80 = 44 + 26.6 + 5.6 = 76.2
        assert abs(score.holistic - 76.2) < 0.5
        assert score.visual == 80
        assert score.audio == 70
        assert abs(score.verbal - 80) < 0.5

    def test_compute_without_audio(self):
        """When audio is not available, redistribute weight to visual."""
        visual = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="P1",
            timestamp=5.0,
            data={"engagement": 60},
        )
        text = SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="P1",
            timestamp=5.0,
            data={"sentiment_score": 0.5},
        )

        score = self.engine.compute(
            participant_id="P1",
            timestamp=5.0,
            visual_signal=visual,
            audio_signal=None,
            text_signal=text,
        )

        # Without audio: visual gets 55+38=93% weight, text gets 7%
        # Normalized: visual=93/100*60 + text=7/100*50
        assert score.holistic > 0
        assert score.audio == 0

    def test_compute_visual_only(self):
        """When only visual is available."""
        visual = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="P1",
            timestamp=1.0,
            data={"engagement": 50},
        )

        score = self.engine.compute(
            participant_id="P1",
            timestamp=1.0,
            visual_signal=visual,
            audio_signal=None,
            text_signal=None,
        )

        assert score.holistic == 50  # Falls back to visual score directly


class TestEngagementAlertDetector:
    def test_alert_fires_after_duration(self):
        from backend.engagement_engine import EngagementAlertDetector

        detector = EngagementAlertDetector(threshold=40, min_duration=5)

        # Score below threshold for 6 seconds
        alert = detector.check(EngagementScore("P1", 10.0, 30, 30, 30, 30))
        assert alert is None  # First time — just starts tracking

        alert = detector.check(EngagementScore("P1", 16.0, 35, 35, 35, 35))
        assert alert is not None  # 6 seconds below threshold
        assert alert.participant_id == "P1"
        assert alert.duration_below >= 5

    def test_no_alert_when_above_threshold(self):
        from backend.engagement_engine import EngagementAlertDetector

        detector = EngagementAlertDetector(threshold=40, min_duration=5)

        alert = detector.check(EngagementScore("P1", 10.0, 30, 30, 30, 30))
        assert alert is None

        # Score goes back above threshold — resets tracker
        alert = detector.check(EngagementScore("P1", 12.0, 60, 60, 60, 60))
        assert alert is None

        # Below again but not long enough
        alert = detector.check(EngagementScore("P1", 14.0, 30, 30, 30, 30))
        assert alert is None

    def test_no_alert_when_duration_too_short(self):
        from backend.engagement_engine import EngagementAlertDetector

        detector = EngagementAlertDetector(threshold=40, min_duration=30)

        alert = detector.check(EngagementScore("P1", 10.0, 30, 30, 30, 30))
        assert alert is None

        # Only 10 seconds — not enough
        alert = detector.check(EngagementScore("P1", 20.0, 35, 35, 35, 35))
        assert alert is None
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_engagement_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement EngagementEngine**

Create `backend/engagement_engine.py`:

```python
"""Weighted engagement scoring engine (Simpleem 55/38/7 model).

Combines visual (55%), para-verbal/audio (38%), and verbal/text (7%)
signals into a holistic per-participant engagement score.
"""

import logging
from dataclasses import dataclass
from backend.signal_bus import SignalEvent

logger = logging.getLogger(__name__)


@dataclass
class EngagementScore:
    participant_id: str
    timestamp: float
    holistic: float  # 0-100 weighted combination
    visual: float    # 0-100 from visual pipeline
    audio: float     # 0-100 from audio pipeline
    verbal: float    # 0-100 from text pipeline


@dataclass
class EngagementAlert:
    participant_id: str
    timestamp: float
    score: float
    duration_below: float
    suggested_action: str


class EngagementEngine:
    """Compute holistic engagement using Simpleem-style weighting."""

    VISUAL_WEIGHT = 0.55
    AUDIO_WEIGHT = 0.38
    VERBAL_WEIGHT = 0.07

    def compute(
        self,
        participant_id: str,
        timestamp: float,
        visual_signal: SignalEvent | None = None,
        audio_signal: SignalEvent | None = None,
        text_signal: SignalEvent | None = None,
    ) -> EngagementScore:
        """Compute weighted engagement score from available signals."""
        visual_score = 0.0
        audio_score = 0.0
        verbal_score = 0.0

        has_visual = visual_signal is not None
        has_audio = audio_signal is not None
        has_text = text_signal is not None

        if has_visual:
            visual_score = float(visual_signal.data.get("engagement", 50))

        if has_audio:
            audio_score = float(audio_signal.data.get("energy", 50))

        if has_text:
            raw = float(text_signal.data.get("sentiment_score", 0.5))
            # sentiment_score is 0-1, normalize to 0-100
            verbal_score = raw * 100 if raw <= 1.0 else raw

        # Determine active weights based on available signals
        active_weights = {}
        if has_visual:
            active_weights["visual"] = self.VISUAL_WEIGHT
        if has_audio:
            active_weights["audio"] = self.AUDIO_WEIGHT
        if has_text:
            active_weights["verbal"] = self.VERBAL_WEIGHT

        if not active_weights:
            return EngagementScore(participant_id, timestamp, 0, 0, 0, 0)

        # If only one signal, use it directly
        if len(active_weights) == 1:
            scores = {"visual": visual_score, "audio": audio_score, "verbal": verbal_score}
            key = next(iter(active_weights))
            direct = scores[key]
            return EngagementScore(
                participant_id, timestamp, direct,
                visual_score, audio_score, verbal_score,
            )

        # Normalize weights to sum to 1.0
        total_weight = sum(active_weights.values())
        norm = {k: v / total_weight for k, v in active_weights.items()}

        holistic = (
            norm.get("visual", 0) * visual_score +
            norm.get("audio", 0) * audio_score +
            norm.get("verbal", 0) * verbal_score
        )

        return EngagementScore(
            participant_id=participant_id,
            timestamp=timestamp,
            holistic=round(holistic, 1),
            visual=round(visual_score, 1),
            audio=round(audio_score, 1),
            verbal=round(verbal_score, 1),
        )


class EngagementAlertDetector:
    """Detect sustained low engagement and emit alerts."""

    def __init__(self, threshold: float = 40, min_duration: float = 30):
        self.threshold = threshold
        self.min_duration = min_duration
        self._below_since: dict[str, float] = {}
        self._alerted: set[str] = set()  # Don't repeat alerts for same drop

    def check(self, score: EngagementScore) -> EngagementAlert | None:
        """Check if engagement has been below threshold long enough to alert."""
        pid = score.participant_id

        if score.holistic >= self.threshold:
            # Above threshold — reset tracking
            self._below_since.pop(pid, None)
            self._alerted.discard(pid)
            return None

        # Below threshold
        if pid not in self._below_since:
            self._below_since[pid] = score.timestamp
            return None

        duration = score.timestamp - self._below_since[pid]
        if duration >= self.min_duration and pid not in self._alerted:
            self._alerted.add(pid)
            return EngagementAlert(
                participant_id=pid,
                timestamp=score.timestamp,
                score=score.holistic,
                duration_below=duration,
                suggested_action=f"Consider asking {pid} a direct question to re-engage",
            )

        return None

    def reset(self):
        """Reset all tracking state."""
        self._below_since.clear()
        self._alerted.clear()
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_engagement_engine.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add backend/engagement_engine.py backend/tests/test_engagement_engine.py
git commit -m "feat: add EngagementEngine with 55/38/7 weighted scoring and alert detection"
```

---

### Task 4: Create MediaIngestion Module (URL Import Fix)

**Files:**
- Create: `backend/media_ingestion.py`
- Create: `backend/tests/test_media_ingestion.py`
- Modify: `backend/main.py:96-207` (replace inline download logic)

**Step 1: Write the failing test**

Create `backend/tests/test_media_ingestion.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.media_ingestion import (
    MediaIngestion,
    DownloadResult,
    extract_name_from_url,
)


class TestExtractNameFromUrl:
    def test_youtube_url(self):
        name = extract_name_from_url("https://www.youtube.com/watch?v=abc123")
        assert name  # Should extract something meaningful
        assert len(name) > 0

    def test_direct_mp4_url(self):
        name = extract_name_from_url("https://example.com/videos/meeting.mp4")
        assert name == "meeting.mp4"

    def test_sharepoint_url(self):
        name = extract_name_from_url(
            "https://company.sharepoint.com/_layouts/15/stream.aspx?id=%2Fsites%2Fteam%2FRecording.mp4"
        )
        assert "Recording.mp4" in name

    def test_empty_url(self):
        name = extract_name_from_url("")
        assert name == "Imported Recording"


class TestDownloadResult:
    def test_success_result(self):
        result = DownloadResult(
            success=True,
            file_path="/tmp/video.mp4",
            strategy="yt-dlp",
        )
        assert result.success
        assert result.file_path == "/tmp/video.mp4"

    def test_failure_result(self):
        result = DownloadResult(
            success=False,
            file_path="",
            strategy="http",
            error="404 Not Found",
        )
        assert not result.success
        assert result.error == "404 Not Found"


class TestMediaIngestion:
    @pytest.mark.asyncio
    async def test_download_tries_strategies_in_order(self):
        ingestion = MediaIngestion(cookie_path=None)

        # Both strategies fail → final result is failure with combined error
        with patch.object(ingestion, '_try_ytdlp', new_callable=AsyncMock) as mock_yt, \
             patch.object(ingestion, '_try_http', new_callable=AsyncMock) as mock_http:
            mock_yt.return_value = DownloadResult(False, "", "yt-dlp", "not installed")
            mock_http.return_value = DownloadResult(False, "", "http", "404")

            result = await ingestion.download("https://example.com/vid.mp4", "/tmp")
            assert not result.success
            assert "yt-dlp" in result.error
            assert "404" in result.error
            mock_yt.assert_called_once()
            mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_stops_on_first_success(self):
        ingestion = MediaIngestion(cookie_path=None)

        with patch.object(ingestion, '_try_ytdlp', new_callable=AsyncMock) as mock_yt, \
             patch.object(ingestion, '_try_http', new_callable=AsyncMock) as mock_http:
            mock_yt.return_value = DownloadResult(True, "/tmp/vid.mp4", "yt-dlp")

            result = await ingestion.download("https://youtube.com/watch?v=abc", "/tmp")
            assert result.success
            assert result.file_path == "/tmp/vid.mp4"
            mock_yt.assert_called_once()
            mock_http.assert_not_called()  # Didn't need to try HTTP
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_media_ingestion.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement MediaIngestion**

Create `backend/media_ingestion.py`:

```python
"""Multi-strategy video downloader with cookie support.

Tries strategies in order: yt-dlp (with optional cookies) → direct HTTP.
Each strategy returns a DownloadResult. Stops on first success.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote, parse_qs

import httpx

logger = logging.getLogger(__name__)

COOKIE_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "cookies")


@dataclass
class DownloadResult:
    success: bool
    file_path: str
    strategy: str
    error: str = ""


def extract_name_from_url(url: str) -> str:
    """Extract a human-readable name from a URL."""
    if not url:
        return "Imported Recording"

    parsed = urlparse(url)

    # SharePoint: try 'id' query parameter
    qs = parse_qs(parsed.query)
    if "id" in qs:
        path = unquote(qs["id"][0])
        parts = path.rstrip("/").split("/")
        for part in reversed(parts):
            if part and "." in part:
                return part

    # Generic: try path segments
    path = unquote(parsed.path)
    parts = path.rstrip("/").split("/")
    for part in reversed(parts):
        if part and "." in part:
            return part
    for part in reversed(parts):
        if part and part not in ("stream.aspx", "_layouts", "15", "embed.aspx", "watch"):
            return part[:60]

    # YouTube: use video ID
    if "v" in qs:
        return f"YouTube_{qs['v'][0]}"

    return "Imported Recording"


class MediaIngestion:
    """Multi-strategy video downloader."""

    def __init__(self, cookie_path: str | None = None):
        self.cookie_path = cookie_path  # Path to Netscape cookies.txt

    async def download(self, url: str, output_dir: str) -> DownloadResult:
        """Try each download strategy in order. Return first success or combined error."""
        errors = []

        # Strategy 1: yt-dlp (handles YouTube, Vimeo, many streaming sites)
        result = await self._try_ytdlp(url, output_dir)
        if result.success:
            return result
        errors.append(f"yt-dlp: {result.error}")

        # Strategy 2: Direct HTTP download
        result = await self._try_http(url, output_dir)
        if result.success:
            return result
        errors.append(f"http: {result.error}")

        # All strategies failed
        combined_error = " | ".join(errors)
        return DownloadResult(
            success=False,
            file_path="",
            strategy="all",
            error=combined_error,
        )

    async def _try_ytdlp(self, url: str, output_dir: str) -> DownloadResult:
        """Download using yt-dlp with optional cookie support."""
        file_id = uuid.uuid4().hex[:12]
        output_template = os.path.join(output_dir, f"dl_{file_id}.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", output_template,
            "--no-playlist",
            "--socket-timeout", "30",
            "--retries", "3",
        ]

        # Add cookie support if available
        if self.cookie_path and os.path.isfile(self.cookie_path):
            cmd.extend(["--cookies", self.cookie_path])
            logger.info(f"Using cookies from {self.cookie_path}")

        cmd.append(url)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

            if proc.returncode == 0:
                # Find the downloaded file (yt-dlp may change extension)
                for fname in os.listdir(output_dir):
                    if fname.startswith(f"dl_{file_id}"):
                        file_path = os.path.join(output_dir, fname)
                        if os.path.getsize(file_path) > 1000:
                            logger.info(f"yt-dlp OK: {file_path} ({os.path.getsize(file_path)} bytes)")
                            return DownloadResult(True, file_path, "yt-dlp")

            error_text = stderr.decode()[:300] if stderr else "unknown error"
            return DownloadResult(False, "", "yt-dlp", error_text)

        except FileNotFoundError:
            return DownloadResult(False, "", "yt-dlp", "yt-dlp not installed")
        except asyncio.TimeoutError:
            return DownloadResult(False, "", "yt-dlp", "timed out after 180s")
        except Exception as e:
            return DownloadResult(False, "", "yt-dlp", str(e)[:200])

    async def _try_http(self, url: str, output_dir: str) -> DownloadResult:
        """Direct HTTP download with retry and redirect following."""
        # Determine file extension from URL
        ext = "mp4"
        path = unquote(urlparse(url).path)
        for e in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"):
            if path.lower().endswith(e):
                ext = e.lstrip(".")
                break

        file_id = uuid.uuid4().hex[:12]
        file_path = os.path.join(output_dir, f"dl_{file_id}.{ext}")

        for attempt in range(3):  # Retry up to 3 times
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(60.0, connect=15.0),
                    max_redirects=5,
                ) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            return DownloadResult(
                                False, "", "http",
                                f"HTTP {resp.status_code}",
                            )

                        # Check content-type if available
                        content_type = resp.headers.get("content-type", "")
                        if content_type and "text/html" in content_type:
                            return DownloadResult(
                                False, "", "http",
                                "URL returned HTML page, not a video file. Authentication may be required.",
                            )

                        total = 0
                        with open(file_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                                f.write(chunk)
                                total += len(chunk)

                        if total > 1000:
                            logger.info(f"HTTP download OK: {file_path} ({total} bytes)")
                            return DownloadResult(True, file_path, "http")
                        else:
                            return DownloadResult(
                                False, "", "http",
                                f"Download too small: {total} bytes",
                            )

            except httpx.TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return DownloadResult(False, "", "http", "timed out")
            except Exception as e:
                return DownloadResult(False, "", "http", str(e)[:200])

        return DownloadResult(False, "", "http", "all retries exhausted")
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_media_ingestion.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add backend/media_ingestion.py backend/tests/test_media_ingestion.py
git commit -m "feat: add MediaIngestion multi-strategy downloader with cookie support"
```

---

### Task 5: Integrate MediaIngestion into main.py

**Files:**
- Modify: `backend/main.py:96-207` (replace `import_video_url` and `_download_video_from_url`)

**Step 1: Update import_video_url endpoint**

In `backend/main.py`, replace the `import_video_url` endpoint and `_download_video_from_url` function (lines 96-232) with:

```python
from .media_ingestion import MediaIngestion, extract_name_from_url, COOKIE_DIR

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
```

Also remove the old `_extract_name_from_url` function (lines 209-232) since it's now in `media_ingestion.py`.

**Step 2: Test manually**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m uvicorn backend.main:app --port 8000`
Expected: Server starts without import errors.

Test the new cookies endpoint:
Run: `curl -X POST http://localhost:8000/api/upload-cookies -F "file=@/dev/null"`
Expected: 400 or similar (empty file), but confirms endpoint exists.

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: integrate MediaIngestion into URL import with cookie support"
```

---

### Task 6: Create pipelines/ Directory Structure

Reorganize existing analysis code into the new pipelines/ structure. This is a refactor — no new functionality.

**Files:**
- Create: `backend/pipelines/__init__.py`
- Create: `backend/pipelines/visual.py` (extract from `core42_client.py`)
- Create: `backend/pipelines/text.py` (extract from `core42_client.py`)
- Modify: `backend/realtime_pipeline.py` (update imports)

**Step 1: Create pipelines/__init__.py**

```python
"""Signal processing pipelines for LensAI v2.

- visual.py: Frame analysis via GPT-4o Vision (55% weight)
- audio.py: Audio feature extraction via librosa (38% weight)
- text.py: Transcription + NLP via Whisper + GPT (7% weight)
- diarization.py: Speaker diarization via pyannote (Phase 3)
"""
```

**Step 2: Create pipelines/visual.py**

This is a thin wrapper that re-exports the existing `analyze_frame_all_signals` and `analyze_frame_emotions` from `core42_client.py` and adds SignalBus integration:

```python
"""Visual signal pipeline — GPT-4o Vision frame analysis.

Analyzes video frames for emotions, body language, gestures, and reactions.
Contributes 55% to the holistic engagement score.
"""

from backend.core42_client import analyze_frame_all_signals, analyze_frame_emotions
from backend.frame_extractor import extract_single_frame
from backend.signal_bus import SignalBus, SignalEvent, SignalType


async def analyze_frame_and_publish(
    frame_path: str,
    timestamp: float,
    signal_bus: SignalBus,
) -> dict:
    """Analyze a frame and publish visual signals to the bus."""
    result = await analyze_frame_all_signals(frame_path, timestamp)
    result["timestamp"] = timestamp

    # Publish per-participant visual signals
    for p in result.get("participants", []):
        label = p.get("label", "Person 1")
        emotions = p.get("emotions", {})

        await signal_bus.publish(SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id=label,
            timestamp=timestamp,
            data={
                "engagement": emotions.get("engagement", 50),
                "emotion": emotions.get("primary", "neutral"),
                "confidence": emotions.get("confidence", 0.5),
                "body_language": p.get("body_language", {}),
                "gestures": p.get("gestures", []),
                "reactions": p.get("reactions", []),
            },
        ))

    return result
```

**Step 3: Create pipelines/text.py**

```python
"""Text signal pipeline — Whisper transcription + GPT NLP analysis.

Transcribes audio and analyzes sentiment/keywords.
Contributes 7% to the holistic engagement score.
"""

from backend.core42_client import transcribe_audio, analyze_words_signal, analyze_voice_signal
from backend.signal_bus import SignalBus, SignalEvent, SignalType


async def transcribe_and_publish(
    audio_chunk_path: str,
    chunk_start: float,
    signal_bus: SignalBus,
) -> dict:
    """Transcribe an audio chunk and publish text signals to the bus."""
    transcript_data = await transcribe_audio(audio_chunk_path)
    segments = transcript_data.get("segments", [])

    # Analyze words for this chunk
    chunk_text = " ".join(s.get("text", "") for s in segments)
    if chunk_text.strip():
        words_result = await analyze_words_signal(chunk_text, chunk_start)

        await signal_bus.publish(SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="all",  # Text signal applies to all for now
            timestamp=chunk_start,
            data={
                "sentiment": words_result.get("sentiment", "neutral"),
                "sentiment_score": words_result.get("sentiment_score", 0.5),
                "keywords": words_result.get("keywords", []),
                "key_phrases": words_result.get("key_phrases", []),
                "text": chunk_text,
            },
        ))

    return transcript_data
```

**Step 4: Verify imports work**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -c "from backend.pipelines.visual import analyze_frame_and_publish; from backend.pipelines.text import transcribe_and_publish; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add backend/pipelines/
git commit -m "refactor: create pipelines/ directory with visual and text modules"
```

---

## Phase 2: Real Audio Analysis

### Task 7: Create Audio Analysis Pipeline

**Files:**
- Create: `backend/pipelines/audio.py`
- Create: `backend/tests/test_audio_pipeline.py`

**Step 1: Write the failing test**

Create `backend/tests/test_audio_pipeline.py`:

```python
import os
import numpy as np
import pytest
from backend.pipelines.audio import AudioAnalyzer, AudioFeatures


class TestAudioAnalyzer:
    def setup_method(self):
        self.analyzer = AudioAnalyzer()

    def test_analyze_silence(self):
        """Silent audio should produce low engagement."""
        # Create a silent WAV-like array
        sr = 16000
        duration = 2.0
        silence = np.zeros(int(sr * duration))

        features = self.analyzer.analyze_array(silence, sr)
        assert isinstance(features, AudioFeatures)
        assert features.volume_energy < 10  # Very low for silence
        assert features.speaking_rate == 0  # No speech in silence

    def test_analyze_tone(self):
        """A pure tone should have consistent pitch."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        tone = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200 Hz tone

        features = self.analyzer.analyze_array(tone, sr)
        assert features.pitch_mean > 0  # Should detect pitch
        assert features.volume_energy > 20  # Audible

    def test_engagement_score_range(self):
        """Engagement score should always be 0-100."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        tone = 0.3 * np.sin(2 * np.pi * 300 * t)

        features = self.analyzer.analyze_array(tone, sr)
        score = features.engagement_score
        assert 0 <= score <= 100

    def test_features_from_file(self, tmp_path):
        """Test loading from a WAV file."""
        import soundfile as sf

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.4 * np.sin(2 * np.pi * 250 * t)

        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, sr)

        features = self.analyzer.analyze_file(wav_path)
        assert isinstance(features, AudioFeatures)
        assert features.duration > 1.5
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_audio_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement AudioAnalyzer**

Create `backend/pipelines/audio.py`:

```python
"""Audio signal pipeline — librosa-based audio feature extraction.

Extracts real acoustic features from audio waveform: pitch, volume, speaking rate,
pauses, spectral brightness. Contributes 38% to holistic engagement score.
"""

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from backend.signal_bus import SignalBus, SignalEvent, SignalType

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Extracted audio features for a chunk."""
    duration: float
    pitch_mean: float          # Average F0 in Hz (0 if no pitch detected)
    pitch_std: float           # Pitch variation (std dev of F0)
    volume_energy: float       # RMS energy normalized to 0-100
    volume_dynamics: float     # Dynamic range of volume (0-100)
    speaking_rate: float       # Estimated syllables per second
    pause_count: int           # Number of silence gaps > 0.3s
    pause_ratio: float         # Fraction of time in silence (0-1)
    spectral_centroid: float   # Average spectral centroid (Hz)
    zcr_mean: float            # Mean zero-crossing rate

    @property
    def pitch_variation_normalized(self) -> float:
        """Pitch variation normalized to 0-100. Higher variation = more engaged."""
        if self.pitch_std <= 0:
            return 0
        # Typical speech pitch std is 20-60 Hz
        return min(100, (self.pitch_std / 60.0) * 100)

    @property
    def volume_energy_normalized(self) -> float:
        """Already 0-100."""
        return self.volume_energy

    @property
    def speaking_rate_normalized(self) -> float:
        """Natural speaking rate scores highest (3-5 syl/s). Too fast or slow penalized."""
        if self.speaking_rate <= 0:
            return 0
        # Bell curve centered around 4 syllables/second
        optimal = 4.0
        deviation = abs(self.speaking_rate - optimal)
        return max(0, min(100, 100 - (deviation / optimal) * 60))

    @property
    def pause_pattern_normalized(self) -> float:
        """Natural pauses (10-30% of speech) score high. Too many/few penalized."""
        # Optimal pause ratio: 0.15-0.25
        if self.pause_ratio < 0.05:
            return 70  # Almost no pauses — still ok
        if self.pause_ratio < 0.15:
            return 90
        if self.pause_ratio < 0.30:
            return 80
        if self.pause_ratio < 0.50:
            return 50
        return 20  # Mostly silence — disengaged

    @property
    def spectral_brightness_normalized(self) -> float:
        """Spectral centroid normalized to 0-100. Higher = more alert/aroused."""
        if self.spectral_centroid <= 0:
            return 0
        # Typical speech centroid: 500-3000 Hz
        return min(100, (self.spectral_centroid / 3000.0) * 100)

    @property
    def engagement_score(self) -> float:
        """Weighted audio engagement sub-score (0-100)."""
        return (
            0.30 * self.volume_energy_normalized +
            0.25 * self.pitch_variation_normalized +
            0.20 * self.speaking_rate_normalized +
            0.15 * self.pause_pattern_normalized +
            0.10 * self.spectral_brightness_normalized
        )


class AudioAnalyzer:
    """Extract acoustic features from audio using librosa."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze_file(self, audio_path: str) -> AudioFeatures:
        """Load a WAV file and extract features."""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.analyze_array(y, sr)

    def analyze_array(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract features from a numpy audio array."""
        duration = len(y) / sr
        if duration < 0.1 or np.max(np.abs(y)) < 1e-6:
            return AudioFeatures(
                duration=duration, pitch_mean=0, pitch_std=0,
                volume_energy=0, volume_dynamics=0,
                speaking_rate=0, pause_count=0, pause_ratio=1.0,
                spectral_centroid=0, zcr_mean=0,
            )

        # Pitch (F0) via pyin
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        if len(voiced_f0) == 0:
            voiced_f0 = np.array([0.0])

        pitch_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0
        pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 1 else 0

        # Volume (RMS energy)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms))
        rms_max = float(np.max(rms))
        # Normalize to 0-100 (typical speech RMS is 0.01-0.3)
        volume_energy = min(100, (rms_mean / 0.15) * 100)
        volume_dynamics = min(100, ((rms_max - float(np.min(rms))) / max(rms_max, 1e-6)) * 100)

        # Speaking rate (syllable estimation using onset detection)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        speaking_rate = len(onsets) / max(duration, 0.1)

        # Pause detection (silence segments > 0.3s)
        frame_duration = 512 / sr  # hop_length / sr
        silence_threshold = 0.01
        is_silent = rms < silence_threshold
        pause_count = 0
        pause_frames = 0
        in_pause = False
        current_pause_len = 0
        min_pause_frames = int(0.3 / frame_duration)

        for silent in is_silent[0]:
            if silent:
                current_pause_len += 1
                if not in_pause and current_pause_len >= min_pause_frames:
                    in_pause = True
                    pause_count += 1
                if in_pause:
                    pause_frames += 1
            else:
                in_pause = False
                current_pause_len = 0

        total_frames = len(rms[0]) if len(rms.shape) > 1 else len(rms)
        pause_ratio = pause_frames / max(total_frames, 1)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = float(np.mean(centroid))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        return AudioFeatures(
            duration=duration,
            pitch_mean=round(pitch_mean, 1),
            pitch_std=round(pitch_std, 1),
            volume_energy=round(volume_energy, 1),
            volume_dynamics=round(volume_dynamics, 1),
            speaking_rate=round(speaking_rate, 2),
            pause_count=pause_count,
            pause_ratio=round(pause_ratio, 3),
            spectral_centroid=round(spectral_centroid, 1),
            zcr_mean=round(zcr_mean, 4),
        )


async def analyze_audio_and_publish(
    audio_chunk_path: str,
    timestamp: float,
    signal_bus: SignalBus,
    participant_id: str = "all",
) -> AudioFeatures:
    """Analyze an audio chunk and publish audio signals to the bus."""
    analyzer = AudioAnalyzer()
    features = analyzer.analyze_file(audio_chunk_path)

    await signal_bus.publish(SignalEvent(
        signal_type=SignalType.AUDIO,
        participant_id=participant_id,
        timestamp=timestamp,
        data={
            "energy": features.engagement_score,
            "pitch_mean": features.pitch_mean,
            "pitch_std": features.pitch_std,
            "volume_energy": features.volume_energy,
            "volume_dynamics": features.volume_dynamics,
            "speaking_rate": features.speaking_rate,
            "pause_count": features.pause_count,
            "pause_ratio": features.pause_ratio,
            "spectral_centroid": features.spectral_centroid,
            "zcr_mean": features.zcr_mean,
        },
    ))

    return features
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_audio_pipeline.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add backend/pipelines/audio.py backend/tests/test_audio_pipeline.py
git commit -m "feat: add librosa audio analysis pipeline with engagement scoring"
```

---

### Task 8: Integrate Audio Pipeline into RealtimeSession

**Files:**
- Modify: `backend/realtime_pipeline.py` (add audio analysis alongside transcription)

**Step 1: Add audio analysis to the transcription worker**

In `backend/realtime_pipeline.py`, add import at top:

```python
from .pipelines.audio import analyze_audio_and_publish, AudioAnalyzer
from .signal_bus import SignalBus, SignalEvent, SignalType
from .engagement_engine import EngagementEngine, EngagementAlertDetector
```

In `RealtimeSession.__init__`, add:

```python
# Signal bus and engagement engine
self.signal_bus = SignalBus()
self.engagement_engine = EngagementEngine()
self.alert_detector = EngagementAlertDetector(
    threshold=float(os.getenv("ENGAGEMENT_ALERT_THRESHOLD", "40")),
    min_duration=float(os.getenv("ENGAGEMENT_ALERT_DURATION", "30")),
)
self.audio_analyzer = AudioAnalyzer()
self.audio_features: list[dict] = []
```

In the `_transcription_worker` method, after `transcript_data = await transcribe_audio(chunk_path)`, add audio analysis:

```python
# Analyze audio features from the same chunk
try:
    features = self.audio_analyzer.analyze_file(chunk_path)
    audio_data = {
        "timestamp": (start + end) / 2,
        "energy": features.engagement_score,
        "pitch_mean": features.pitch_mean,
        "pitch_std": features.pitch_std,
        "volume_energy": features.volume_energy,
        "speaking_rate": features.speaking_rate,
        "pause_count": features.pause_count,
    }
    self.audio_features.append(audio_data)
    if not self._cancelled and not self._finalizing:
        await self._send_json({"type": "audio_features", "data": audio_data})
except Exception as e:
    logger.warning(f"[{self.video_id[:8]}] Audio analysis failed: {e}")
```

**Step 2: Add engagement_alert WebSocket message type**

In the `_analyze_frame_live` method, after sending signals, compute and check engagement:

```python
# Compute holistic engagement using latest audio data
for p in result.get("participants", []):
    label = p.get("label", "Person 1")
    visual_signal = SignalEvent(
        signal_type=SignalType.VISUAL,
        participant_id=label,
        timestamp=timestamp,
        data={"engagement": p.get("emotions", {}).get("engagement", 50)},
    )
    audio_signal = None
    if self.audio_features:
        latest_audio = self.audio_features[-1]
        audio_signal = SignalEvent(
            signal_type=SignalType.AUDIO,
            participant_id=label,
            timestamp=timestamp,
            data={"energy": latest_audio.get("energy", 50)},
        )

    score = self.engagement_engine.compute(
        participant_id=label,
        timestamp=timestamp,
        visual_signal=visual_signal,
        audio_signal=audio_signal,
    )

    alert = self.alert_detector.check(score)
    if alert and not self._cancelled and not self._finalizing:
        await self._send_json({
            "type": "engagement_alert",
            "data": {
                "participant": alert.participant_id,
                "score": alert.score,
                "duration_below_threshold": alert.duration_below,
                "timestamp": alert.timestamp,
                "suggested_action": alert.suggested_action,
            },
        })
```

**Step 3: Test manually**

Run the backend and play a video. Verify:
- `audio_features` messages appear in WebSocket alongside transcript
- `engagement_alert` messages appear if a participant stays below threshold

**Step 4: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "feat: integrate audio pipeline + engagement alerts into realtime session"
```

---

### Task 9: Update Frontend for Audio Features + Engagement Alerts

**Files:**
- Modify: `frontend/src/lib/websocket.ts` (add new message types)
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts` (handle new messages)
- Modify: `frontend/src/components/signals/VoiceSignal.tsx` (show real audio data)
- Create: `frontend/src/components/EngagementAlert.tsx`

**Step 1: Add message types to websocket.ts**

In `frontend/src/lib/websocket.ts`, add to `MessageType`:

```typescript
export type MessageType =
  | 'signals'
  | 'transcript'
  | 'voice'
  | 'words'
  | 'personality'
  | 'correlation'
  | 'flag'
  | 'summary'
  | 'coaching'
  | 'status'
  | 'error'
  | 'no_audio'
  | 'detecting'
  | 'video_ended_ack'
  | 'complete'
  | 'audio_features'        // NEW
  | 'engagement_alert';     // NEW
```

**Step 2: Add state and handlers to useRealtimeAnalysis.ts**

Add to `RealtimeState`:

```typescript
audioFeatures: AudioFeatureData[];
engagementAlerts: EngagementAlertData[];
```

Add interfaces:

```typescript
export interface AudioFeatureData {
  timestamp: number;
  energy: number;
  pitch_mean: number;
  pitch_std: number;
  volume_energy: number;
  speaking_rate: number;
  pause_count: number;
}

export interface EngagementAlertData {
  participant: string;
  score: number;
  duration_below_threshold: number;
  timestamp: number;
  suggested_action: string;
}
```

Add action types and reducer cases:

```typescript
| { type: 'AUDIO_FEATURES'; data: AudioFeatureData }
| { type: 'ENGAGEMENT_ALERT'; data: EngagementAlertData }
```

Reducer cases:

```typescript
case 'AUDIO_FEATURES': {
  let newFeatures = [...state.audioFeatures, action.data];
  if (newFeatures.length > 200) newFeatures = newFeatures.slice(-200);
  return { ...state, audioFeatures: newFeatures };
}

case 'ENGAGEMENT_ALERT': {
  return {
    ...state,
    engagementAlerts: [...state.engagementAlerts, action.data],
  };
}
```

Add WebSocket subscriptions in the hook:

```typescript
ws.on('audio_features', (data) => dispatch({ type: 'AUDIO_FEATURES', data: data as AudioFeatureData }));
ws.on('engagement_alert', (data) => dispatch({ type: 'ENGAGEMENT_ALERT', data: data as EngagementAlertData }));
```

**Step 3: Create EngagementAlert toast component**

Create `frontend/src/components/EngagementAlert.tsx`:

```tsx
import { AlertTriangle, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { EngagementAlertData } from '../hooks/useRealtimeAnalysis';
import { formatTime } from '../lib/api';

interface Props {
  alerts: EngagementAlertData[];
}

export default function EngagementAlerts({ alerts }: Props) {
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  // Auto-dismiss after 10 seconds
  useEffect(() => {
    if (alerts.length === 0) return;
    const timer = setTimeout(() => {
      setDismissed(new Set(alerts.map((_, i) => i)));
    }, 10000);
    return () => clearTimeout(timer);
  }, [alerts.length]);

  const visible = alerts.filter((_, i) => !dismissed.has(i));
  if (visible.length === 0) return null;

  return (
    <div className="fixed top-20 right-6 z-50 space-y-2 max-w-sm">
      {visible.map((alert, i) => (
        <div
          key={`${alert.timestamp}-${i}`}
          className="bg-red-900/90 border border-red-500/50 rounded-xl p-4 shadow-2xl animate-fade-in backdrop-blur-sm"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-red-200">
                Low Engagement: {alert.participant}
              </p>
              <p className="text-xs text-red-300 mt-1">
                Score dropped to {Math.round(alert.score)}% for {Math.round(alert.duration_below_threshold)}s
                (at {formatTime(alert.timestamp)})
              </p>
              <p className="text-xs text-red-400 mt-1 italic">
                {alert.suggested_action}
              </p>
            </div>
            <button
              onClick={() => setDismissed(new Set([...dismissed, alerts.indexOf(alert)]))}
              className="text-red-400 hover:text-red-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
```

**Step 4: Add EngagementAlerts to AnalysisDashboard**

In `frontend/src/components/AnalysisDashboard.tsx`, import and render:

```tsx
import EngagementAlerts from './EngagementAlert';

// Inside the realtime mode JSX, after the header:
<EngagementAlerts alerts={rtState.engagementAlerts} />
```

**Step 5: Commit**

```bash
git add frontend/src/lib/websocket.ts frontend/src/hooks/useRealtimeAnalysis.ts frontend/src/components/EngagementAlert.tsx frontend/src/components/AnalysisDashboard.tsx
git commit -m "feat: add audio features display + engagement alert toasts in frontend"
```

---

## Phase 3: Transcription Upgrade

### Task 10: Add pyannote.audio Dependencies

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Add pyannote dependencies**

Add to `backend/requirements.txt`:

```
pyannote.audio>=3.1
torch>=2.0
```

**Step 2: Install**

Run: `pip3 install pyannote.audio torch --quiet`
Note: This may take a while (PyTorch is large). CPU-only is fine.

**Step 3: Verify**

Run: `python3 -c "from pyannote.audio import Pipeline; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add pyannote.audio and torch for speaker diarization"
```

---

### Task 11: Create Diarization Pipeline

**Files:**
- Create: `backend/pipelines/diarization.py`
- Create: `backend/tests/test_diarization.py`

**Step 1: Write the failing test**

Create `backend/tests/test_diarization.py`:

```python
import pytest
from backend.pipelines.diarization import (
    DiarizationSegment,
    assign_speakers_to_transcript,
)


class TestAssignSpeakers:
    def test_basic_assignment(self):
        diarization = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ]
        transcript = [
            {"start": 1.0, "end": 3.0, "text": "Hello", "speaker": ""},
            {"start": 6.0, "end": 8.0, "text": "Hi there", "speaker": ""},
        ]

        result = assign_speakers_to_transcript(transcript, diarization)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_overlap_picks_majority(self):
        diarization = [
            DiarizationSegment(start=0.0, end=3.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=3.0, end=6.0, speaker="SPEAKER_01"),
        ]
        transcript = [
            {"start": 2.0, "end": 5.0, "text": "Some overlap", "speaker": ""},
        ]

        result = assign_speakers_to_transcript(transcript, diarization)
        # 2-3s with SPEAKER_00 (1s), 3-5s with SPEAKER_01 (2s) → SPEAKER_01 wins
        assert result[0]["speaker"] == "Speaker 2"

    def test_empty_diarization(self):
        transcript = [
            {"start": 1.0, "end": 3.0, "text": "Hello", "speaker": ""},
        ]
        result = assign_speakers_to_transcript(transcript, [])
        assert result[0]["speaker"] == ""  # Unchanged
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_diarization.py -v`
Expected: FAIL

**Step 3: Implement diarization module**

Create `backend/pipelines/diarization.py`:

```python
"""Speaker diarization pipeline using pyannote.audio.

Identifies who spoke when in an audio file. Results are used to
assign speaker labels to Whisper transcript segments.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load pyannote pipeline (cached — only loads once)."""
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        logger.info("pyannote diarization pipeline loaded")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}")
        return None


async def diarize_audio(audio_path: str) -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Returns a list of segments with speaker labels.
    Runs in a thread pool since pyannote is CPU-bound.
    """
    pipeline = _load_pipeline()
    if pipeline is None:
        logger.warning("Diarization skipped — pipeline not available")
        return []

    def _run():
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
        return segments

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def assign_speakers_to_transcript(
    transcript_segments: list[dict],
    diarization_segments: list[DiarizationSegment],
) -> list[dict]:
    """Assign speaker labels from diarization to transcript segments.

    Uses maximum overlap to determine which speaker is most likely for
    each transcript segment.
    """
    if not diarization_segments:
        return transcript_segments

    # Build speaker label mapping (SPEAKER_00 → Speaker 1, etc.)
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
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && python3 -m pytest backend/tests/test_diarization.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add backend/pipelines/diarization.py backend/tests/test_diarization.py
git commit -m "feat: add speaker diarization pipeline using pyannote.audio"
```

---

### Task 12: Create Live Caption Overlay Component

**Files:**
- Create: `frontend/src/components/CaptionOverlay.tsx`
- Modify: `frontend/src/components/AnalysisDashboard.tsx` (add CC toggle + overlay)

**Step 1: Create CaptionOverlay.tsx**

```tsx
import { useState } from 'react';
import type { TranscriptSegment } from '../lib/api';
import { Captions, CaptionsOff } from 'lucide-react';

interface Props {
  transcript: TranscriptSegment[];
  currentTime: number;
  enabled: boolean;
  onToggle: () => void;
}

export default function CaptionOverlay({ transcript, currentTime, enabled, onToggle }: Props) {
  // Find transcript segments near current playback time (last 2-3 visible)
  const visibleSegments = transcript.filter(
    (seg) => seg.start_time <= currentTime && seg.end_time >= currentTime - 5
  ).slice(-3);

  return (
    <>
      {/* CC Toggle Button */}
      <button
        onClick={onToggle}
        className={`absolute top-3 right-3 z-10 p-2 rounded-lg transition-colors ${
          enabled
            ? 'bg-accent/80 text-white'
            : 'bg-black/40 text-white/50 hover:text-white/80'
        }`}
        title={enabled ? 'Hide captions' : 'Show captions'}
      >
        {enabled ? (
          <Captions className="w-4 h-4" />
        ) : (
          <CaptionsOff className="w-4 h-4" />
        )}
      </button>

      {/* Caption overlay */}
      {enabled && visibleSegments.length > 0 && (
        <div className="absolute bottom-14 left-4 right-4 z-10 pointer-events-none">
          <div className="bg-black/75 backdrop-blur-sm rounded-lg px-4 py-2 space-y-1">
            {visibleSegments.map((seg, i) => (
              <p
                key={`${seg.start_time}-${i}`}
                className="text-sm text-white animate-fade-in"
                style={{
                  opacity: i === visibleSegments.length - 1 ? 1 : 0.6,
                }}
              >
                {seg.speaker && (
                  <span className="text-accent font-semibold mr-2">{seg.speaker}:</span>
                )}
                {seg.text}
              </p>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
```

**Step 2: Integrate into AnalysisDashboard**

In the `RealtimePlayer` component inside `AnalysisDashboard.tsx`, add:

```tsx
import CaptionOverlay from './CaptionOverlay';

// Add state for captions
const [captionsEnabled, setCaptionsEnabled] = useState(false);
const [currentTime, setCurrentTime] = useState(0);

// In the video player, track currentTime via the existing attachVideo mechanism
// Add to the video element's parent div:
<CaptionOverlay
  transcript={rtState.transcript}
  currentTime={currentTime}
  enabled={captionsEnabled}
  onToggle={() => setCaptionsEnabled(!captionsEnabled)}
/>
```

**Step 3: Commit**

```bash
git add frontend/src/components/CaptionOverlay.tsx frontend/src/components/AnalysisDashboard.tsx
git commit -m "feat: add live caption overlay with CC toggle on video player"
```

---

## Phase 4: Engagement Intelligence (Frontend)

### Task 13: Per-Participant Engagement Timelines

**Files:**
- Modify: `frontend/src/components/EmotionTimeline.tsx`

**Step 1: Add view toggle**

Add a "Combined" / "Individual" toggle to EmotionTimeline. In individual mode, render one mini AreaChart per participant with engagement zone coloring.

The key changes to `EmotionTimeline.tsx`:
- Add `const [viewMode, setViewMode] = useState<'combined' | 'individual'>('combined')`
- Render toggle buttons in the header
- When `viewMode === 'individual'`, map over participants and render a separate chart for each
- Each individual chart has colored ReferenceAreas for engagement zones

**Step 2: Commit**

```bash
git add frontend/src/components/EmotionTimeline.tsx
git commit -m "feat: add per-participant individual engagement timelines"
```

---

### Task 14: Cross-Meeting Analytics Backend

**Files:**
- Modify: `backend/models.py` (add `meeting_analytics` table)
- Modify: `backend/schemas.py` (add analytics schemas)
- Modify: `backend/main.py` (add analytics endpoints)

**Step 1: Add meeting_analytics table to models.py**

Add to `init_db()`:

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS meeting_analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL UNIQUE,
        avg_engagement REAL DEFAULT 0,
        avg_visual_engagement REAL DEFAULT 0,
        avg_audio_engagement REAL DEFAULT 0,
        avg_verbal_engagement REAL DEFAULT 0,
        participant_count INTEGER DEFAULT 0,
        duration REAL DEFAULT 0,
        overall_sentiment TEXT DEFAULT 'neutral',
        meeting_date TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id)
    )
""")
```

**Step 2: Add schemas**

Add to `backend/schemas.py`:

```python
class MeetingAnalytics(BaseModel):
    video_id: str
    video_name: str
    avg_engagement: float
    avg_visual_engagement: float
    avg_audio_engagement: float
    avg_verbal_engagement: float
    participant_count: int
    duration: float
    overall_sentiment: str
    meeting_date: str
```

**Step 3: Add API endpoints to main.py**

```python
@app.get("/api/analytics")
async def list_analytics() -> list[MeetingAnalytics]:
    """List analytics for all analyzed meetings."""
    ...

@app.get("/api/analytics/compare")
async def compare_meetings(ids: str) -> dict:
    """Compare two meetings side by side. ids=uuid1,uuid2"""
    ...
```

**Step 4: Commit**

```bash
git add backend/models.py backend/schemas.py backend/main.py
git commit -m "feat: add cross-meeting analytics backend with compare endpoint"
```

---

## Phase 5: Live Meeting Integration

### Task 15: Create Recall.ai Adapter

**Files:**
- Create: `backend/recall_adapter.py`
- Modify: `backend/main.py` (add meeting endpoints)

This task requires a Recall.ai API key. The implementation wraps the Recall.ai REST API for bot creation, status polling, and webhook handling. The specific API calls depend on Recall.ai's current API documentation.

**Step 1: Implement adapter skeleton**

Create `backend/recall_adapter.py` with the `RecallAiAdapter` class that handles:
- `join_meeting(url, bot_name)` → Creates bot via POST to Recall.ai API
- `get_status(meeting_id)` → Polls bot status
- `get_recording(meeting_id)` → Downloads recording after meeting ends
- Webhook endpoint for real-time audio/video stream (if Recall.ai supports it)

**Step 2: Add meeting endpoints to main.py**

```python
@app.post("/api/meetings/join")
async def join_meeting(req: JoinMeetingRequest, background_tasks: BackgroundTasks):
    ...

@app.get("/api/meetings/{meeting_id}/status")
async def get_meeting_status(meeting_id: str):
    ...

@app.websocket("/ws/meeting/{meeting_id}")
async def ws_meeting(websocket: WebSocket, meeting_id: str):
    ...
```

**Step 3: Commit**

```bash
git add backend/recall_adapter.py backend/main.py
git commit -m "feat: add Recall.ai adapter for live meeting integration"
```

---

## Phase 6: Export & Polish

### Task 16: PDF/CSV Export

**Files:**
- Create: `backend/export.py`
- Modify: `backend/requirements.txt` (add weasyprint, matplotlib)
- Modify: `backend/main.py` (add export endpoints)

**Step 1: Add dependencies**

Add to `backend/requirements.txt`:
```
weasyprint>=60.0
matplotlib>=3.8
```

**Step 2: Implement export module**

Create `backend/export.py` with:
- `generate_pdf_report(video_id)` — Renders HTML template → PDF via weasyprint
- `generate_csv_export(video_id)` — Creates ZIP of CSV files (engagement, transcript, participants)

**Step 3: Add endpoints**

```python
@app.get("/api/videos/{video_id}/export/pdf")
async def export_pdf(video_id: str):
    ...

@app.get("/api/videos/{video_id}/export/csv")
async def export_csv(video_id: str):
    ...
```

**Step 4: Commit**

```bash
git add backend/export.py backend/requirements.txt backend/main.py
git commit -m "feat: add PDF and CSV export endpoints"
```

---

### Task 17: Participant Name Mapping

**Files:**
- Modify: `backend/models.py` (add face_profiles and face_mappings tables)
- Modify: `backend/main.py` (add rename endpoint)
- Modify: `frontend/src/components/ParticipantSidebar.tsx` (add rename UI)

**Step 1: Add DB tables**

Add to `init_db()`:

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS face_profiles (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        face_embedding TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
await db.execute("""
    CREATE TABLE IF NOT EXISTS face_mappings (
        video_id TEXT NOT NULL,
        participant_label TEXT NOT NULL,
        profile_id TEXT,
        custom_name TEXT,
        FOREIGN KEY (video_id) REFERENCES videos(id)
    )
""")
```

**Step 2: Add rename API**

```python
@app.post("/api/videos/{video_id}/participants/{participant_id}/rename")
async def rename_participant(video_id: str, participant_id: str, req: RenameRequest):
    """Assign a custom name to a participant."""
    ...
```

**Step 3: Add rename UI to ParticipantSidebar**

Add an edit icon next to each participant name. Clicking opens an inline input field to type a new name. On submit, calls the rename API.

**Step 4: Commit**

```bash
git add backend/models.py backend/main.py frontend/src/components/ParticipantSidebar.tsx
git commit -m "feat: add participant name mapping with rename UI"
```

---

## Summary of All Tasks

| Task | Phase | Description | Files |
|------|-------|-------------|-------|
| 1 | 1 | Install dependencies | requirements.txt |
| 2 | 1 | Create SignalBus | signal_bus.py + test |
| 3 | 1 | Create EngagementEngine | engagement_engine.py + test |
| 4 | 1 | Create MediaIngestion | media_ingestion.py + test |
| 5 | 1 | Integrate MediaIngestion | main.py |
| 6 | 1 | Create pipelines/ structure | pipelines/*.py |
| 7 | 2 | Audio analysis pipeline | pipelines/audio.py + test |
| 8 | 2 | Integrate audio into realtime | realtime_pipeline.py |
| 9 | 2 | Frontend audio + alerts | websocket.ts, hooks, components |
| 10 | 3 | Add pyannote dependencies | requirements.txt |
| 11 | 3 | Diarization pipeline | pipelines/diarization.py + test |
| 12 | 3 | Live caption overlay | CaptionOverlay.tsx |
| 13 | 4 | Per-participant timelines | EmotionTimeline.tsx |
| 14 | 4 | Cross-meeting analytics | models.py, main.py, schemas.py |
| 15 | 5 | Recall.ai adapter | recall_adapter.py, main.py |
| 16 | 6 | PDF/CSV export | export.py, main.py |
| 17 | 6 | Participant name mapping | models.py, main.py, frontend |
