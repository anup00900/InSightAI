# Simpleem Offline V3 — Full Overhaul Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make simpleem-offline production-grade — accurate per-participant analysis with proper 55/38/7 weighted signals, professional UI with full analytics, and all signal panels wired to real data.

**Architecture:** FastAPI + SQLite backend (port 8001), React 19 + Vite + Tailwind frontend (port 5174). Core42 Compass API (GPT-4o vision, GPT-4.1 text, Whisper-1).

---

## 1. Analysis Accuracy — Backend

### 1A. Better Frame Analysis Prompts
- Calibrated engagement scale with anchors: 80+ = active nodding/smiling, 50-70 = neutral attention, 30-50 = passive/distracted, <30 = disengaged
- Camera-ON: full visual analysis (emotions, body language, gestures, reactions)
- Camera-OFF: engagement=30, emotion='passive', minimal body language
- Use REAL NAMES from OCR instead of positional "Person N" labels (DONE in current session)

### 1B. Per-Participant Audio Analysis (NEW)
- Current: global audio analysis (one score shared by all participants)
- New: use speaker diarization timestamps to extract audio for each speaker's segments
- Run librosa on each participant's speaking segments independently
- Each participant gets their own audio engagement sub-score
- This makes the 55/38/7 weighting meaningful per person

### 1C. Engagement Alerts
- Wire EngagementAlertDetector into the pipeline
- Save alerts as flags with type="engagement_alert"
- Alert when any participant's score stays below 40 for 30+ seconds

### 1D. More Parallelization
- Phase 4: Run personality + correlations + summary/flags ALL concurrently (3-way gather)
- Currently summary/flags runs after personality. No dependency between them.

### 1E. Model Allocation
- GPT-4o: frame analysis, name extraction (vision tasks)
- GPT-4.1: summary, coaching, personality, correlations, voice signals, word signals, speaker assignment
- Whisper-1: transcription
- Increase max_tokens: summary 2000→3000, frame analysis 2000→2500

---

## 2. Frontend — UI Overhaul

### 2A. Landing Page (Videos Exist)
- Top: Quick stats bar (total meetings, avg engagement, total participants)
- Video grid with thumbnails + engagement sparkline
- Bottom: Cross-meeting engagement trend chart

### 2B. Dashboard Enhancements
- KPI row: add per-channel sub-scores (Visual/Audio/Text) below Overall Score
- NEW: Engagement Timeline line chart (engagement over time per participant)
- NEW: Speaking Distribution horizontal bar chart
- NEW: Engagement Alerts cards
- Wire all 7 signal panels to real API data

### 2C. Signal Panel API Endpoints (NEW)
- GET /api/videos/{id}/signals/snapshots — frame-by-frame visual signals
- GET /api/videos/{id}/signals/voice — voice signal timeline
- GET /api/videos/{id}/signals/words — word/text signals
- GET /api/videos/{id}/signals/personality — personality profiles
- GET /api/videos/{id}/signals/correlations — cross-signal patterns
- GET /api/videos/{id}/signals/audio-features — librosa features

### 2D. Signal Panel Components (already exist, need data wiring)
- EmotionsSignal, BodyLanguageSignal, GesturesSignal, ReactionsSignal
- VoiceSignal, WordsSignal, PersonalitySignal, CorrelationPanel

---

## 3. Out of Scope (YAGNI)
- Deal prediction / CRM integration
- Burnout detection (multi-session)
- Real-time WebSocket streaming
- Face recognition / embeddings
