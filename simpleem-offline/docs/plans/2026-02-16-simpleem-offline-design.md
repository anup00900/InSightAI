# simpleem-offline Design

## Overview

Batch video analysis tool — an offline counterpart to simpleem-poc. Users upload recorded meeting videos, get high-quality transcription via local models, review transcripts, then run full AI analysis. Premium dark glassmorphism UI.

## Architecture

### Processing Model: REST-only Batch Pipeline

No WebSocket. Pure REST endpoints + polling for progress.

**Wizard flow:**
1. Upload video
2. Transcribe (local faster-whisper + pyannote diarization) → user reviews transcript
3. Analyze (GPT-4o vision + GPT-4.1 text + librosa audio + engagement engine) → progress bar
4. Results dashboard with all signals, coaching, summary, flags, export

### API Endpoints

```
POST /api/upload                    → Save file, return video_id
POST /api/videos/{id}/transcribe    → Background job: extract audio, whisper, diarize
GET  /api/videos/{id}/status        → { step, progress, detail }
GET  /api/videos/{id}/transcript    → Full transcript with speaker labels
POST /api/videos/{id}/analyze       → Background job: frames, vision, text, scoring
GET  /api/videos/{id}/results       → All signals, coaching, summary, flags, scores
GET  /api/videos                    → List all videos
DELETE /api/videos/{id}             → Delete video + data
GET  /api/videos/{id}/export/pdf    → PDF report
GET  /api/videos/{id}/export/csv    → CSV ZIP
GET  /api/analytics                 → Cross-meeting analytics
GET  /api/analytics/compare?ids=a,b → Compare meetings
POST /api/videos/{id}/participants/{pid}/rename → Rename participant
```

### Local Models

| Model | Task | Package |
|-------|------|---------|
| faster-whisper large-v3 | Transcription | `pip install faster-whisper` |
| pyannote/speaker-diarization-3.1 | Speaker diarization | `pip install pyannote-audio` |
| GPT-4o (Core42 API) | Frame analysis, name OCR | Same .env |
| GPT-4.1 (Core42 API) | Coaching, summary, personality, voice/word signals | Same .env |

### Batch Pipeline Steps

**Transcription job (`POST /api/videos/{id}/transcribe`):**
1. FFmpeg extracts full audio track → WAV
2. faster-whisper large-v3 transcribes full audio with timestamps
3. pyannote-audio diarizes speakers (who spoke when)
4. Map diarization segments to transcript segments
5. Store in transcript_segments table

**Analysis job (`POST /api/videos/{id}/analyze`):**
1. Sample frames every 2-3 seconds via FFmpeg
2. GPT-4o vision analysis on each frame (emotions, body language, gestures, reactions, name OCR)
3. librosa audio feature extraction (pitch, volume, speaking rate, pauses)
4. GPT-4.1 text analysis (sentiment, keywords, personality, voice signals, word signals)
5. Engagement engine scoring (55% visual, 38% audio, 7% verbal)
6. GPT-4.1 generates coaching per participant
7. GPT-4.1 generates meeting summary + flags
8. GPT-4.1 generates correlation patterns
9. Store everything to SQLite

## UI: Dark Glassmorphism Theme

### Color Palette

- Background: `#0a0a1a` (deep navy-black)
- Card surfaces: `rgba(255,255,255,0.05)` + `backdrop-filter: blur(20px)`
- Primary gradient: `linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7)` (indigo → violet → purple)
- Secondary gradient: `linear-gradient(135deg, #06b6d4, #3b82f6)` (cyan → blue)
- Success: `#10b981`, Warning: `#f59e0b`, Error: `#ef4444`
- Text: `#f1f5f9`, Muted: `#94a3b8`
- Glow borders: `box-shadow: 0 0 15px rgba(139,92,246,0.3)`

### Premium Elements

- Frosted glass cards with subtle border glow
- Gradient text headings
- Animated progress rings
- Smooth wizard step transitions
- Animated score counters
- Gradient engagement badges
- Hover glow intensification

### Wizard Steps

**Step 1 — Upload:** Animated gradient border drag-drop zone, file badges, upload progress ring.

**Step 2 — Transcript Review:** Auto-starts transcription, animated progress ring, scrollable transcript with colored speaker labels and timestamps. "Continue to Analysis" button.

**Step 3 — Analysis:** Multi-stage progress (extracting frames → analyzing visuals → processing text → generating insights). Auto-advances when done.

**Step 4 — Results Dashboard:** Full glassmorphism dashboard with all 7 signal panels, charts, coaching, summary, export buttons.

## What Stays the Same as simpleem-poc

- .env / API keys (Core42 Azure OpenAI)
- Database schema (14+ tables)
- Engagement engine (55/38/7 model)
- GPT-4o vision prompts
- GPT-4.1 analysis prompts
- PDF/CSV export
- Video library + cross-meeting analytics
- Tech stack: FastAPI, React, Vite, Tailwind, SQLite

## What Changes

- No WebSocket — pure REST + polling
- Local faster-whisper large-v3 instead of Whisper API
- Local pyannote-audio for speaker diarization
- Wizard-based UX instead of real-time play-along
- Dark glassmorphism premium UI
- New reusable UI components (GlassCard, GradientButton, ProgressRing, etc.)
