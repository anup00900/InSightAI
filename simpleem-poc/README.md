# InSightAI - Conversation Intelligence Platform

Real-time meeting analysis platform that extracts behavioral signals, engagement metrics, and actionable coaching from video recordings.

## What It Does

Upload a meeting recording (or paste a URL), press play, and get:

- **Real-time participant detection** — OCR extracts real names from Teams/Zoom/Meet gallery view, including camera-off participants (initials circles)
- **Live transcription** — Parallel Whisper workers transcribe audio in real-time with hallucination filtering
- **Behavioral signals** — GPT-4o vision analyzes each frame for eye contact, gestures, posture, facial expressions, reactions per participant
- **Engagement scoring** — Second-by-second engagement chart combining visual + audio + word signals
- **Voice analysis** — Pitch, energy, speaking pace, filler words, emotional tone
- **Personality insights** — DISC/Big Five personality trait estimation from communication patterns
- **AI coaching** — Personalized coaching tips per participant ("Anup, try maintaining eye contact when others speak")
- **Summary & flags** — Meeting summary, key moments, red/green flags
- **PDF/CSV export** — Downloadable reports with all metrics

## Architecture

```
Frontend (React + Vite + Tailwind)
  ├── RealtimeVideoPlayer — HTML5 video with CC overlay
  ├── AnalysisDashboard — Signals, engagement chart, transcripts
  ├── ParticipantSidebar — Per-person metrics and drill-down
  ├── CoachingPanel — AI-generated coaching per participant
  └── CrossMeetingAnalytics — Compare metrics across sessions

WebSocket (real-time bidirectional)
  └── play/tick/pause/finalize protocol

Backend (FastAPI + Python)
  ├── realtime_pipeline.py — Core session: frame analysis, transcription, engagement
  ├── core42_client.py — GPT-4o vision + Whisper + GPT-4.1 API calls
  ├── engagement_engine.py — Multi-signal engagement scoring + alerts
  ├── signal_bus.py — Event bus for cross-component signal routing
  ├── frame_extractor.py — FFmpeg frame/audio extraction
  ├── export.py — PDF (fpdf2) and CSV report generation
  └── recall_adapter.py — Recall.ai integration for live meeting bots
```

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts, Lucide |
| Backend | FastAPI, Python 3.9+, aiosqlite, asyncio |
| AI | OpenAI GPT-4o (vision), GPT-4.1, Whisper |
| Audio | librosa, pyannote.audio (diarization) |
| Video | FFmpeg (frame/audio extraction) |
| Export | fpdf2 (PDF), csv (CSV) |

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- FFmpeg (`brew install ffmpeg`)
- OpenAI API key

### Setup

```bash
# Clone
git clone https://github.com/anup00900/InSightAI.git
cd InSightAI

# Create .env
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run everything (installs deps + starts both servers)
./run.sh
```

Or manually:

```bash
# Backend
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — upload a meeting recording and press play.

### API Docs

FastAPI auto-generated docs at **http://localhost:8000/docs**

## Key Features

### Progressive Name Detection
Scans frames at [5s, 10s, 15s, 30s, 60s, 120s] to progressively discover all participants as the gallery view expands. Handles camera-off participants (initials circles), truncated names, and "Last, First" format.

### Parallel Transcription
Two concurrent Whisper workers with 2s chunks and 1s trigger threshold. Includes hallucination filtering for common Whisper artifacts on silence.

### Real-time Engagement Engine
Combines visual signals (eye contact, gestures, posture) + audio signals (pitch, energy, pace) + word signals (sentiment, filler words) into a per-second engagement score with configurable weights and alert detection.

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI app + routes
│   ├── realtime_pipeline.py    # Core real-time session engine
│   ├── realtime_ws.py          # WebSocket endpoint handler
│   ├── core42_client.py        # OpenAI API integration
│   ├── engagement_engine.py    # Multi-signal engagement scoring
│   ├── signal_bus.py           # Event bus for signal routing
│   ├── frame_extractor.py      # FFmpeg frame/audio extraction
│   ├── video_processor.py      # Video metadata + processing
│   ├── export.py               # PDF/CSV report generation
│   ├── models.py               # SQLite models
│   ├── recall_adapter.py       # Recall.ai live meeting bot
│   └── pipelines/audio.py      # Audio feature extraction
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/         # React components
│       ├── hooks/              # useRealtimeAnalysis WebSocket hook
│       └── lib/                # WebSocket client, utilities
├── docs/plans/                 # Design documents
├── run.sh                      # One-command startup script
└── .env                        # OPENAI_API_KEY (not committed)
```

## License

Private / Proprietary
