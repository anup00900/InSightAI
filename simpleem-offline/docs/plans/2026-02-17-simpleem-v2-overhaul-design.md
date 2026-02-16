# Simpleem Offline v2 — Comprehensive Overhaul Design

## Problem Statement

The initial simpleem-offline implementation works end-to-end but has critical issues:
1. **Too slow** — sequential pipeline, frame analysis takes forever
2. **No progressive display** — user sees nothing until ALL phases finish
3. **UI not CEO-grade** — bland empty state, generic dashboard, weak visual impact
4. **PDF/CSV exports incomplete** — missing coaching, correlations, charts
5. **Name extraction unreliable** — only samples 3 frames at 10/30/50%
6. **Frame analysis low quality** — uses `detail: "low"` image mode

## Architecture: What Changes

### Backend Pipeline Overhaul

**Before (sequential):**
```
extract_names → frame_analysis → audio_analysis → text_analysis → personality → correlations → summary → coaching
```

**After (parallelized with progressive saves):**
```
┌─ extract_names (first 5s frames + samples)
│
├─ frame_analysis (batch=10, interval=5s) ──┐
├─ audio_analysis (librosa chunks)    ──────┤─→ save partial results to DB after each
├─ text_analysis (voice + words)      ──────┘
│
├─ personality + correlations (depend on above 3)
├─ summary + flags (depend on above)
└─ coaching per participant (concurrent per participant)
```

**Key changes:**
- `asyncio.gather` for 3 independent analysis phases (frames, audio, text)
- Each phase saves to DB immediately when done (not all at end)
- Frame interval: 3s → 5s (40% fewer API calls)
- Frame batch size: 5 → 10 concurrent
- Transcription chunks: sequential → concurrent (up to 3 at once)
- Frame analysis image detail: "low" → "auto"
- Name extraction: 5 frames from first 5s + 3 samples later

### New API Endpoint

```
GET /api/videos/{id}/partial-results
```
Returns whatever analysis data is available so far:
- `transcript` (if transcription done)
- `participants` (if names extracted)
- `emotions` (if frame analysis done)
- `voice_signals` (if audio analysis done)
- `word_signals` (if text analysis done)
- `summary`, `flags`, `coaching` (if post-analysis done)
- `ready_sections: string[]` — which sections have data

### Frontend Overhaul

#### Empty State (No Videos)
Replace sad empty icon with an inviting hero:
- Animated gradient background with subtle particles
- Product tagline: "AI-Powered Meeting Intelligence"
- 3 feature highlight cards (Engagement, Coaching, Insights)
- Large CTA: "Analyze Your First Meeting" → opens wizard

#### Library View
- Video cards with thumbnail preview (extracted frame)
- Mini sparkline of engagement per completed video
- Animated status badges for in-progress items
- Quick actions on hover (view, re-analyze, export, delete)

#### Dashboard — Executive Summary First
```
┌─────────────────────────────────────────────────┐
│  [KPI] Overall  [KPI] Engagement  [KPI] Sent.   [KPI] Speaking │
├──────────────────┬──────────────────────────────┤
│  Video Player    │  Participant Sidebar         │
│  (synced seek)   │  - Cards with scores         │
│                  │  - Radar chart for selected   │
│  Emotion Timeline│  - Speaking donut            │
│  Flag Timeline   │                              │
├──────────────────┴──────────────────────────────┤
│  Tabs: Summary | Coaching | Transcript | Signals│
│  ─────────────────────────────────────────────  │
│  [Tab content area]                             │
│  Export buttons: PDF | CSV                      │
└─────────────────────────────────────────────────┘
```

**Progressive loading in dashboard:**
- Shows sections as they become available
- Skeleton loaders for sections still processing
- "Analyzing..." badges on pending sections
- Auto-refreshes via polling every 2 seconds

#### Enhanced PDF Report
- Executive summary with big score numbers
- Engagement timeline chart (matplotlib, embedded as image)
- Per-participant coaching recommendations
- Correlation insights section
- Red flags with severity badges
- Professional header with gradient styling

#### Enhanced CSV Export
Add files: coaching.csv, correlations.csv, personality.csv, signal_snapshots.csv

## Tech Stack (unchanged)
- **Backend:** FastAPI + SQLite + aiosqlite
- **Frontend:** React 19 + Vite 5 + Tailwind CSS
- **AI:** Core42 API (GPT-4.1 for chat, GPT-4o for vision, whisper-1 for transcription)
- **Audio:** librosa for acoustic features
- **Charts:** matplotlib for PDF charts

## Performance Targets
- 30-min meeting: complete analysis in ~5 min (currently ~15+ min)
- First results visible (transcript): within 60 seconds
- Frame analysis: 3x faster via larger batches + wider intervals
- Transcription: 2x faster via concurrent chunks
