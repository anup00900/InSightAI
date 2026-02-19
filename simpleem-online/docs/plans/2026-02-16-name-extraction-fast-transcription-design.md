# Name Extraction + Fast Transcription + Per-Person Insights

## Goal

Extract real participant names from meeting recordings (via OCR + speaker diarization), make transcription faster and complete, and deliver name-specific insights.

## Problems Solved

1. **"Person 1" labels** — GPT-4o labels by position, not real name. Users see generic labels instead of "Anup Roy", "John Smith".
2. **Slow transcription** — Single serial Whisper worker falls behind real-time. First transcript appears ~5s after play.
3. **Incomplete transcription** — Sequential processing drops audio when API is slow. Gaps in transcript.
4. **Generic insights** — Coaching and insights use "Person 1" instead of real names. Not actionable.

## Architecture

### Name Extraction Pipeline

```
initialize() (WebSocket connect)
    ├── Frame 0 → GPT-4o: "Read name labels in this meeting screenshot"
    │   └── Returns: {"Person 1": "Anup Roy", "Person 2": "John Smith"}
    │   └── Stored in: self._name_map: dict[str, str]
    │
    ├── Pre-extract first 2s audio chunk (head start for first transcript)
    └── Full audio extraction (background)

handle_tick() (during playback)
    ├── Parallel transcription workers → raw segments
    ├── Speaker assignment (GPT-4.1): assign names to transcript segments
    │   └── Batched every ~10s to reduce API calls
    └── Frame analysis → uses name_map to replace "Person 1" with real name
```

### Parallel Transcription

Current: 1 serial worker, 3s chunks, 1.5s trigger threshold.

New:
- 2s chunks, 1.0s trigger threshold
- 2 parallel transcription workers
- Range-based coordination: workers claim non-overlapping ranges
- Pre-buffer first chunk during initialize()
- Gap sweep on finalize

### Speaker Assignment

Whisper returns text without speaker IDs. We add a lightweight GPT-4.1 call:

- Accumulate ~10s of raw transcript segments
- Send to GPT-4.1 with participant names: "Assign each segment to a speaker"
- Update transcript segments with real names
- Frontend displays named speakers immediately

### Per-Person Insights

With real names available:
- Participant sidebar, insights tab, coaching all use real names
- Per-person speaking time from speaker-assigned transcript duration
- Coaching is personalized: "Anup, try to maintain eye contact"
- Summary references participants by name
- PDF/CSV exports use real names

## API Cost Impact

- +1 GPT-4o call on init (name OCR) — ~$0.01
- +1 GPT-4.1 call per ~10s of audio (speaker assignment) — ~$0.005 each
- For a 30-min meeting: ~180 extra speaker assignment calls
- Total added cost per session: ~$1.00

## Key Files Modified

- `backend/realtime_pipeline.py` — parallel workers, name map, speaker assignment
- `backend/core42_client.py` — new `extract_names_from_frame()` and `assign_speakers()` functions
- `backend/frame_extractor.py` — no changes needed
- `frontend/src/hooks/useRealtimeAnalysis.ts` — handle name updates
- `frontend/src/components/AnalysisDashboard.tsx` — display real names
- `frontend/src/components/ParticipantSidebar.tsx` — display real names
