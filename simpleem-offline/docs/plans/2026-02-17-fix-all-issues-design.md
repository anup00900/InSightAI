# Design: Fix PDF, Speaking Distribution, Summary, Voice Analysis, Signal Pipeline

**Date:** 2026-02-17
**Status:** Approved

## Problem Statement

Five interconnected issues across the simpleem-offline backend and frontend:

1. **PDF Report download crashes** — None guards missing, transcript layout breaks
2. **Speaking distribution incorrect** — speaker matching fails for camera-off participants, donut chart not normalized
3. **Summary not working** — overloaded single GPT call, no structured meeting notes
4. **Voice analysis feels hardcoded** — GPT infers from text instead of using real librosa audio data
5. **Signal pipeline not properly structured** — signals only show last snapshot, no per-speaker audio, channel scores are superficial

## Design Decisions

### 1. PDF Report Fix
- Add None guards on `overall_score`, `duration`, `summary`
- Truncate transcript to 50 segments (code says it does but doesn't)
- Fix `multi_cell` width going negative with long speaker names
- Add meeting notes section to PDF
- Frontend: better error handling in `handleExport`

### 2. Speaking Distribution Fix
- Backend: normalize `speaking_pct` to always sum to 100%
- Backend: distribute unmatched speaker words using fuzzy fallback + proportional distribution
- Frontend: always normalize donut chart data to 100%

### 3. Summary — Separate Calls + Meeting Notes
- New `generate_meeting_summary()` — dedicated GPT-5 call, transcript only, 3-4 paragraph summary
- New `generate_meeting_notes()` — structured action items, decisions, follow-ups
- Rename old function to `generate_flags_and_scores()` — only flags + participant scores
- New DB table `meeting_notes` for storing structured notes
- Frontend: new Meeting Notes section with structured cards
- PDF: include meeting notes

### 4. Voice Analysis — Three-Layer with Fallback
- Per-speaker audio: map transcript timestamps to audio, run librosa per speaker
- New DB table `speaker_audio_features` for per-speaker metrics
- New API endpoint for detailed audio features
- Keep GPT voice signal as qualitative overlay
- Fallback: per-speaker librosa -> global librosa -> GPT text inference
- Frontend: redesigned VoiceSignal with multi-line charts + per-speaker cards

### 5. Signal Pipeline — Full Timeline
- Signal components show aggregated data across all snapshots
- Frequency/distribution charts for gestures, reactions, emotions
- Channel score bars show actual computed values
- Remove dead code (unused SignalPanel.tsx)

## Architecture

```
Transcript --> generate_meeting_summary() --> summary text (GPT-5)
Transcript --> generate_meeting_notes()   --> action items, decisions (GPT-5)
Transcript --> generate_flags_and_scores()--> flags + scores (GPT-4.1)

Audio --> global librosa analysis (existing)
Audio --> per-speaker librosa analysis (NEW)
Audio --> GPT voice qualitative (existing, becomes overlay)

Frames --> visual signals (existing)
Text  --> word signals (existing)

All --> engagement engine (55/38/7 weighting)
```
