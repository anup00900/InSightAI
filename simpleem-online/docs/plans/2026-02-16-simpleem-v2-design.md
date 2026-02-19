# InsightAI v2 — Simpleem-Grade Conversation Intelligence

**Date:** 2026-02-16
**Goal:** Transform the InsightAI POC into a production-grade internal product with Simpleem-style Artificial Emotional Intelligence (55% non-verbal / 38% para-verbal / 7% verbal signal weighting), robust URL import, real audio analysis, streaming transcription with speaker diarization, live meeting integration, and engagement intelligence.

---

## 1. Architecture: SignalBus + EngagementEngine

### Current Flow
```
Video → FFmpeg frames → GPT-4o Vision → emotions/body_language
Video → FFmpeg audio → Whisper → transcript → GPT-4.1 → voice/words (text-inferred)
All signals → flat WebSocket messages → frontend reducer
```

### New Architecture
```
                    ┌──────────────────────────────┐
                    │       Input Sources           │
                    │  File Upload | URL Import |   │
                    │  Recall.ai Meeting Bot        │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │      MediaIngestion          │
                    │  Multi-strategy downloader    │
                    │  (yt-dlp + cookies, httpx,    │
                    │   Recall.ai stream adapter)   │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                     │
    ┌─────────▼──────┐  ┌─────▼─────────┐  ┌───────▼────────┐
    │ Visual Pipeline │  │Audio Pipeline │  │ Text Pipeline  │
    │ (55% weight)   │  │(38% weight)   │  │ (7% weight)    │
    │                │  │               │  │                │
    │ Frame extract  │  │ Audio extract │  │ Transcript     │
    │ GPT-4o Vision  │  │ librosa FFT   │  │ from Whisper   │
    │ → emotions     │  │ → pitch (F0)  │  │ → sentiment    │
    │ → body lang    │  │ → volume/RMS  │  │ → keywords     │
    │ → gestures     │  │ → speak rate  │  │ → key phrases  │
    │ → reactions    │  │ → pauses      │  │                │
    │                │  │ → spectral    │  │ Diarization    │
    │                │  │ → ZCR, MFCC   │  │ (pyannote)     │
    └───────┬────────┘  └──────┬────────┘  └───────┬────────┘
            │                  │                    │
            └──────────────────┼────────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     SignalBus (in-memory)     │
                    │  Normalized signal events     │
                    │  per-participant, timestamped  │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │    EngagementEngine           │
                    │  Weighted: 55V + 38A + 7T     │
                    │  Per-participant scoring       │
                    │  Threshold alerts              │
                    │  Trend detection               │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼─────┐  ┌───────▼──────┐
    │  WebSocket     │  │ DB Store  │  │ Export       │
    │  (real-time)   │  │ (SQLite)  │  │ (PDF/CSV)    │
    └────────────────┘  └───────────┘  └──────────────┘
```

### Key Design Decisions

**SignalBus:** An in-memory event bus (asyncio queues) that all 3 pipelines emit normalized events to. The EngagementEngine subscribes to the bus and computes weighted engagement scores. This decouples signal production from consumption.

**EngagementEngine:** Computes holistic engagement per-participant per-timestamp:
```
engagement_score = 0.55 * visual + 0.38 * audio + 0.07 * verbal
```

### New Backend File Structure
```
backend/
  main.py                    # FastAPI app + routes (updated)
  models.py                  # DB schema (updated with new tables)
  schemas.py                 # Pydantic models (updated)
  media_ingestion.py         # NEW: Multi-strategy downloader
  signal_bus.py              # NEW: Central event bus
  engagement_engine.py       # NEW: Weighted scoring + alerts
  pipelines/
    __init__.py
    visual.py                # Frame extraction + GPT-4o Vision analysis
    audio.py                 # NEW: librosa audio feature extraction
    text.py                  # Whisper transcription + NLP analysis
    diarization.py           # NEW: pyannote speaker diarization
  realtime_pipeline.py       # Updated to use SignalBus
  realtime_ws.py             # Updated WebSocket protocol
  recall_adapter.py          # NEW: Recall.ai integration
  export.py                  # NEW: PDF/CSV generation
  core42_client.py           # Existing GPT/Whisper client (kept)
  api_utils.py               # Existing API wrapper (kept)
  video_processor.py         # Existing FFmpeg utils (kept)
  frame_extractor.py         # Existing frame extraction (kept)
```

---

## 2. URL Import Fix — MediaIngestion Module

### Problem
All URL types fail. Current code tries yt-dlp then direct HTTP. Fails because SharePoint/Teams need auth, YouTube gets blocked without cookies, Google Drive needs tokens.

### Solution: `media_ingestion.py`

```python
class MediaIngestion:
    """Multi-strategy video downloader with cookie support."""

    strategies = [
        YtDlpWithCookiesStrategy,   # YouTube, Vimeo + cookie jar
        DirectHttpStrategy,          # Direct .mp4/.webm links
        RecallAiStrategy,            # Live meeting streams (phase 5)
    ]
```

**Strategy 1: yt-dlp with cookie support**
- Accept a `cookies.txt` file (Netscape format) uploaded by the user
- Store at `uploads/cookies/{user_id}/cookies.txt`
- Pass `--cookies <path>` to yt-dlp for authenticated downloads
- Also support `--cookies-from-browser chrome` as an option
- Handle yt-dlp output parsing for progress reporting

**Strategy 2: Direct HTTP**
- Existing httpx-based download (kept)
- Add retry with exponential backoff
- Follow redirects up to 5 hops
- Content-Type validation

**Error handling:**
- Each strategy returns `DownloadResult(success, file_path, error_message, strategy_used)`
- On failure, try next strategy
- Final error message sent to frontend: "Download failed. Tried: yt-dlp (error: ...), HTTP (error: ...). Try uploading the file directly."

**New API endpoint:**
```
POST /api/upload-cookies   — Upload a cookies.txt file for authenticated downloads
```

**Frontend changes:**
- Add a "Cookie settings" gear icon in the URL import section
- Modal to upload cookies.txt with instructions on how to export from browser

---

## 3. Real Audio Analysis Pipeline

### Purpose
Replace text-inferred voice signals with actual audio waveform analysis using librosa.

### Features Extracted Per Audio Chunk (2-5 seconds)

| Feature | librosa Method | What It Measures | Engagement Signal |
|---------|---------------|------------------|-------------------|
| Pitch (F0) | `librosa.pyin()` | Fundamental frequency | Monotone = low, varied = high |
| Pitch variation | std dev of F0 | How much pitch changes | More variation = more engaged |
| Volume (RMS) | `librosa.feature.rms()` | Loudness energy | Higher energy = more engaged |
| Volume dynamics | range of RMS | Dynamic range | Flat = monotone/bored |
| Speaking rate | syllables / duration | Words per second | Natural pace scores high |
| Pause count | energy < threshold duration | Silence gaps | Too many long pauses = disengaged |
| Spectral centroid | `librosa.feature.spectral_centroid()` | Voice brightness | Higher = more aroused/alert |
| MFCC (13 coefficients) | `librosa.feature.mfcc()` | Voice quality fingerprint | Used for speaker consistency |
| Zero-crossing rate | `librosa.feature.zero_crossing_rate()` | Voice roughness | Breathiness/vocal fry detection |

### Audio Engagement Sub-Score Formula
```python
audio_engagement = (
    0.30 * volume_energy_normalized +      # How energetically speaking (RMS → 0-100)
    0.25 * pitch_variation_normalized +    # Monotone vs. expressive (F0 std → 0-100)
    0.20 * speaking_rate_normalized +      # Natural pace score (WPM → 0-100, bell curve)
    0.15 * pause_pattern_normalized +      # Natural pauses vs. long silences (→ 0-100)
    0.10 * spectral_brightness_normalized  # Voice alertness (centroid → 0-100)
)
```

### Integration
- Processes the same WAV chunks already extracted for Whisper
- Runs in parallel with transcription (no additional FFmpeg calls)
- Emits `AudioSignal` events to SignalBus
- WebSocket sends new `audio_features` message type to frontend

### New Frontend Panel: "Voice Analysis"
- Replaces current text-inferred VoiceSignal panel
- Shows: real-time pitch graph, volume meter, speaking rate indicator, pause counter
- Color-coded: green (energetic), yellow (moderate), red (flat/disengaged)

### Dependencies
- `librosa>=0.10.0` (pure Python + NumPy, no GPU)
- `numpy` (already likely a transitive dependency)

---

## 4. Transcription Upgrade

### 4a. Faster Chunking
- Reduce chunk size from 5s to **2 seconds** with 1s overlap
- Use word-level timestamps: `timestamp_granularities=["word"]`
- Deduplicate overlapping words by timestamp matching
- Net effect: transcript appears ~2-3 seconds after speech instead of ~5-7 seconds

### 4b. Speaker Diarization via pyannote.audio
- Run pyannote diarization on the full audio track (extracted in background)
- pyannote outputs: `[(start, end, speaker_label)]` segments
- Cross-reference Whisper transcript segments with pyannote speaker labels by timestamp overlap
- Result: each transcript segment gets an accurate speaker attribution

**Implementation:**
```python
# pipelines/diarization.py
from pyannote.audio import Pipeline

class DiarizationPipeline:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )

    async def diarize(self, audio_path: str) -> list[DiarizationSegment]:
        """Run speaker diarization on full audio. Returns speaker-labeled segments."""
        ...
```

**For live analysis:** Run diarization incrementally on accumulated audio (every 30 seconds) and update speaker labels retroactively.

**For uploaded files:** Run diarization on the full audio track once, then use results to label all transcript segments.

### 4c. Live Caption Overlay
- Semi-transparent caption bar at bottom of video player
- Shows last 2-3 transcript segments with speaker labels
- Fade-in animation for new segments
- Toggle button on video controls: CC icon
- Respects current playback position (captions sync to video time)

**Frontend component:** `CaptionOverlay.tsx`
```tsx
// Positioned absolute at bottom of video container
// Reads from rtState.transcript, filtered by current video time
```

### Dependencies
- `pyannote.audio>=3.1` (requires PyTorch)
- `torch>=2.0` (CPU-only is sufficient)
- HuggingFace auth token for pyannote model access

---

## 5. Engagement Intelligence

### 5a. Holistic Engagement Score (EngagementEngine)

```python
# engagement_engine.py

class EngagementEngine:
    VISUAL_WEIGHT = 0.55
    AUDIO_WEIGHT = 0.38
    VERBAL_WEIGHT = 0.07

    def compute_engagement(
        self,
        visual_signals: list[VisualSignal],
        audio_signals: list[AudioSignal],
        text_signals: list[TextSignal],
        participant_id: str,
        timestamp: float,
    ) -> EngagementScore:
        """Compute weighted holistic engagement score."""
        visual_score = self._compute_visual(visual_signals)   # 0-100
        audio_score = self._compute_audio(audio_signals)       # 0-100
        verbal_score = self._compute_verbal(text_signals)      # 0-100

        holistic = (
            self.VISUAL_WEIGHT * visual_score +
            self.AUDIO_WEIGHT * audio_score +
            self.VERBAL_WEIGHT * verbal_score
        )

        return EngagementScore(
            participant_id=participant_id,
            timestamp=timestamp,
            holistic=holistic,
            visual=visual_score,
            audio=audio_score,
            verbal=verbal_score,
        )
```

### 5b. Engagement Alerts

```python
class EngagementAlertDetector:
    def __init__(self, threshold=40, min_duration=30):
        self.threshold = threshold          # Score below this triggers alert
        self.min_duration = min_duration    # Must be below for this many seconds
        self._below_since: dict[str, float] = {}  # participant_id → timestamp

    def check(self, score: EngagementScore) -> EngagementAlert | None:
        """Check if engagement has been below threshold long enough to alert."""
        ...
```

**Alert data sent via WebSocket:**
```json
{
  "type": "engagement_alert",
  "data": {
    "participant": "Person 1",
    "score": 32,
    "duration_below_threshold": 45,
    "timestamp": 120.5,
    "suggested_action": "Consider asking Person 1 a direct question to re-engage"
  }
}
```

**Frontend:** Toast notification + red marker on engagement timeline.

### 5c. Per-Participant Engagement Timelines

**Frontend changes to EmotionTimeline.tsx:**
- Add toggle: "Combined" | "Individual"
- Individual mode: one mini-chart per participant, stacked vertically
- Each mini-chart shows engagement zones: red (<35), yellow (35-70), green (>70)
- Alert markers shown as red dots on the timeline
- Shows all 3 sub-scores (visual/audio/verbal) as separate light-colored lines behind the main engagement line

### 5d. Cross-Meeting Comparison

**New DB table:**
```sql
CREATE TABLE meeting_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL UNIQUE,
    avg_engagement REAL,
    avg_visual_engagement REAL,
    avg_audio_engagement REAL,
    avg_verbal_engagement REAL,
    participant_count INTEGER,
    duration REAL,
    overall_sentiment TEXT,
    meeting_date TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);
```

**New frontend page: "Analytics"**
- Line chart of engagement scores across meetings (x-axis: meeting date)
- Filter by participant (track same person across meetings)
- Side-by-side comparison of two meetings
- Trends: improving/declining engagement over time

**New API endpoints:**
```
GET /api/analytics                      — List all meeting analytics
GET /api/analytics/compare?ids=a,b      — Compare two meetings side by side
GET /api/analytics/trends?participant=X  — Get trend data for a participant
```

---

## 6. Live Meeting Integration — Recall.ai

### Overview
Use Recall.ai (or similar meeting bot service) to join Zoom/Teams/Meet calls and stream audio+video to the backend for real-time analysis.

### API Flow
```
POST /api/meetings/join  {"meeting_url": "https://zoom.us/j/123...", "bot_name": "InsightAI"}
  → Backend calls Recall.ai API to create bot
  → Returns {meeting_id, status: "joining"}

GET /api/meetings/{meeting_id}/status
  → Returns current bot status (joining, in_meeting, processing, done)

WebSocket: /ws/meeting/{meeting_id}
  → Same protocol as /ws/analyze/{video_id}
  → Recall.ai streams feed into the 3-pipeline architecture
```

### Backend Components

**`recall_adapter.py`:**
```python
class RecallAiAdapter:
    """Manages Recall.ai bot lifecycle and stream processing."""

    async def join_meeting(self, meeting_url: str, bot_name: str) -> str:
        """Deploy bot to meeting. Returns meeting_id."""
        ...

    async def get_status(self, meeting_id: str) -> MeetingStatus:
        """Check bot status."""
        ...

    async def stream_to_pipelines(self, meeting_id: str, signal_bus: SignalBus):
        """Receive Recall.ai webhook/stream and feed into pipelines."""
        ...
```

**Recall.ai advantages:**
- Provides per-speaker separated audio (solves diarization for live meetings)
- Video frames from gallery/speaker view
- Speaker identification with names (from meeting participant list)
- No need for OAuth app registrations with Zoom/Teams/Meet

### Frontend Changes
- New "Join Meeting" section alongside file upload and URL import
- Meeting URL input + "Join" button
- Status display: "Bot joining...", "In meeting — analyzing", "Meeting ended"
- Same analysis dashboard (signals, timeline, transcript) but fed from live meeting data

### Configuration
- Recall.ai API key stored in `.env`
- Bot name configurable in settings

---

## 7. Export & Polish

### 7a. PDF Export
- Library: `weasyprint` (HTML-to-PDF, supports CSS styling)
- Template: HTML report with embedded charts (matplotlib renders engagement chart as PNG)
- Sections: Executive Summary, Engagement Overview (chart), Participant Scores, Full Transcript, Coaching Recommendations, Critical Moment Flags
- Endpoint: `GET /api/videos/{id}/export/pdf` → Returns PDF file

### 7b. CSV Export
- Raw data export for spreadsheet analysis
- Separate sheets/files: engagement_timeline.csv, transcript.csv, participant_scores.csv
- Endpoint: `GET /api/videos/{id}/export/csv` → Returns ZIP of CSV files

### 7c. Participant Name Mapping
- New DB table:
```sql
CREATE TABLE face_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    face_embedding TEXT,      -- JSON array of face embedding vectors
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE face_mappings (
    video_id TEXT NOT NULL,
    participant_label TEXT NOT NULL,  -- "Person 1"
    profile_id TEXT,                   -- FK to face_profiles
    FOREIGN KEY (video_id) REFERENCES videos(id),
    FOREIGN KEY (profile_id) REFERENCES face_profiles(id)
);
```

- After analysis, users can click a participant and either:
  - Assign a new name (creates a face_profile)
  - Link to an existing profile (from dropdown of known people)
- Future meetings: compare detected faces against stored profiles for auto-naming
- Face embedding: use a lightweight face recognition model (e.g., `face_recognition` library or CLIP embeddings from GPT-4o)

---

## Phase Plan

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **Phase 1** | Core pipeline refactor: SignalBus, EngagementEngine, MediaIngestion (URL fix), new file structure | None |
| **Phase 2** | Real audio analysis: librosa pipeline, AudioSignal events, updated engagement scoring | Phase 1 |
| **Phase 3** | Transcription upgrade: 2s chunks, pyannote diarization, live caption overlay | Phase 1 |
| **Phase 4** | Engagement intelligence: alerts, per-participant timelines, cross-meeting comparison | Phases 1-3 |
| **Phase 5** | Live meeting integration: Recall.ai adapter, meeting join UI, stream-to-pipeline | Phases 1-3 |
| **Phase 6** | Export & polish: PDF/CSV export, participant name mapping, analytics dashboard | Phases 1-4 |

### New Dependencies
```
# requirements.txt additions
librosa>=0.10.0
pyannote.audio>=3.1
torch>=2.0  # CPU-only for pyannote
weasyprint>=60.0
matplotlib>=3.8  # For chart rendering in PDF export
face_recognition>=1.3.0  # Optional: for face profile matching
```

### Environment Variables
```
# .env additions
HUGGINGFACE_TOKEN=hf_xxx          # For pyannote model download
RECALL_AI_API_KEY=xxx              # For live meeting integration
ENGAGEMENT_ALERT_THRESHOLD=40     # Configurable alert threshold
ENGAGEMENT_ALERT_DURATION=30      # Seconds below threshold to trigger
```
