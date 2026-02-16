# Simpleem In-House POC — Design Document

**Date**: 2026-02-12
**Goal**: Build a demoable POC that replicates Simpleem's core capabilities for the (In)Sight AI Coach / Board Coach use case.

## What We're Building

An in-house conversation intelligence platform that analyzes video/audio recordings of meetings to provide:
- Emotion detection and engagement tracking per participant
- Speech transcription with speaker identification
- AI-generated coaching recommendations
- Critical moment flagging
- Meeting summaries and rapport scores

## Architecture

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + FastAPI |
| Frontend | React + TypeScript + Vite + Tailwind CSS + shadcn/ui |
| Video Processing | FFmpeg (extract frames + audio) |
| All AI Models | Core42 Compass API (OpenAI-compatible) |
| Charts | Recharts |
| Database | SQLite (POC) |
| Theme | Dark blue/black premium |

### AI Models (via Core42 Compass)

- **Whisper** — Audio transcription with timestamps
- **GPT-4o Vision** — Frame-by-frame emotion detection (1 frame/sec)
- **GPT-4o Audio** — Tone and energy analysis
- **GPT-4o** — Coaching recommendations, summaries, critical moments

### API Base: `https://api.core42.ai/v1/`
### Auth: `Authorization: Bearer <API_KEY>` from `.env`

## Processing Pipeline

```
Upload Video/Audio
    │
    ├── FFmpeg: Extract audio (.wav) + frames (1/sec as .jpg)
    │
    ├── Core42 Whisper: Audio → Transcript with timestamps
    │
    ├── Core42 GPT-4o Vision: Each frame → Emotion labels
    │   (happy, neutral, surprised, confused, engaged, disengaged)
    │   (parallel batch requests for speed)
    │
    ├── Core42 GPT-4o Audio: Audio segments → Tone/energy analysis
    │
    └── Core42 GPT-4o: Transcript + emotions + tone →
            ├── Conversation summary
            ├── Critical moment flags
            ├── Coaching recommendations per participant
            └── Engagement & rapport scores
```

## Backend API Endpoints

```
POST   /api/upload                          → Upload video/audio
GET    /api/videos                          → List all videos
GET    /api/videos/{id}                     → Video details + status
POST   /api/videos/{id}/analyze             → Trigger analysis
GET    /api/videos/{id}/results             → Full analysis results
GET    /api/videos/{id}/participants         → List participants
GET    /api/videos/{id}/participants/{pid}   → Participant detail
GET    /api/videos/{id}/emotions             → Emotion timeline
GET    /api/videos/{id}/transcript           → Full transcript
GET    /api/videos/{id}/flags                → Critical moments
GET    /api/videos/{id}/coaching/{pid}       → Coaching per participant
GET    /api/videos/{id}/summary              → AI summary
```

## Dashboard UI

### Screen 1: Upload & Library
- Drag-and-drop upload zone
- List of analyzed recordings with status badges
- Thumbnail, title, date, duration, overall score

### Screen 2: Analysis Dashboard
- Video player synced with emotion timeline
- Scrollable participant sidebar (supports 2-20+ participants)
- Engagement score bar per participant
- Emotion flow chart (multi-line, all participants overlaid)
- Speaking time donut chart
- Skill radar chart per participant (engagement, clarity, rapport, energy, sentiment)
- Clickable critical moment flags on timeline
- Tabbed content: Summary | Coaching | Transcript | Insights
- Coaching recommendations per participant with dropdown selector

### Visualizations:
| Chart | Purpose |
|-------|---------|
| Emotion Flow (multi-line area) | Emotional journey over time |
| Speaking Donut | Conversation dominance |
| Skill Radar | Per-participant strengths |
| Engagement Bars | Quick comparison |
| Flag Timeline | Critical moments |

### Theme: Dark blue/black premium palette with shadcn/ui components.

## Project Structure

```
simpleem-poc/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── pipeline.py          # Analysis pipeline orchestrator
│   ├── core42_client.py     # Core42 API wrapper
│   ├── video_processor.py   # FFmpeg frame/audio extraction
│   ├── models.py            # SQLite models
│   ├── schemas.py           # Pydantic schemas
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── VideoUpload.tsx
│   │   │   ├── VideoLibrary.tsx
│   │   │   ├── AnalysisDashboard.tsx
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── EmotionTimeline.tsx
│   │   │   ├── ParticipantSidebar.tsx
│   │   │   ├── SpeakingDonut.tsx
│   │   │   ├── SkillRadar.tsx
│   │   │   ├── FlagTimeline.tsx
│   │   │   └── CoachingPanel.tsx
│   │   ├── lib/
│   │   │   └── api.ts
│   │   └── styles/
│   │       └── theme.css
│   ├── package.json
│   └── vite.config.ts
├── .env                     # CORE42_API_KEY=...
└── README.md
```
