# Design: Wire Remaining Glass UI + Accuracy Engine

**Date:** 2026-02-17
**Status:** Approved
**Parent Plans:** `2026-02-17-glass-ui-accuracy-design.md`, `2026-02-17-glass-ui-accuracy-engine.md`

## Context

Audit found ~30% of the Glass UI + Accuracy Engine plan was created as standalone files but never wired. This design covers wiring the remaining ~70%.

## What's Already Done
- `voting_engine.py` — full module with consensus algorithms
- `ExpandableCard.tsx` — framer-motion component
- `index.css` — glass CSS enhancements
- `api.ts` — types + API functions (SpeakerAudioData, MeetingNotes, ConfidenceSummary)
- `VoiceSignal.tsx` — redesigned with speakerAudio prop
- `EmotionsSignal.tsx` — aggregates all snapshots (missing confidence dots)
- `models.py` — meeting_notes table
- `core42_client.py` — generate_meeting_summary, generate_meeting_notes, generate_flags_and_scores

## What Needs Wiring

### Backend
1. `core42_client.py` — Add 6 functions: analyze_frame_triple_vote, _analyze_frame_with_model, analyze_voice_signal_dual, _analyze_voice_with_model, analyze_words_signal_dual, _analyze_words_with_model
2. `models.py` — Add voting_log table, speaker_audio_features table, signal_snapshots.confidence migration
3. `batch_pipeline.py` — Full rewire: new imports, triple-model frames, dual-model text, 5-way gather, save meeting_notes + voting_log, cross_signal_validate, confidence in snapshots
4. `main.py` — 3 new endpoints (meeting-notes, speaker-audio, confidence), import MeetingNotesOut, wire get_results

### Frontend
5. `AnalysisDashboard.tsx` — ExpandableCard for KPI + channel cards, fetch confidenceSummary/speakerAudio/meetingNotes, Meeting Notes section, pass speakerAudio to VoiceSignal
6. `EmotionsSignal.tsx` — confidence dots
7. `api.ts` — confidence field on SignalSnapshot

## Approach
Full triple-model (3 GPT calls per frame, 2 per text signal) as specified in parent plan. No config flags — always-on voting.
