# Critical 4-Bug Fix Design

**Date**: 2026-02-19
**Status**: Approved

## Problem Statement

Four critical bugs in the online realtime analysis pipeline:

1. **Participants preloaded**: All participant names appear immediately on init, even before playback starts. Feels hardcoded — should appear dynamically as they become visible.
2. **Speaking distribution broken**: Percentages are wrong/even because the SIGNALS handler wipes transcript-calculated values every frame.
3. **Speaker names missing from transcript**: Visual `is_speaking` detection at `detail: "low"` can't see platform speaker borders. Speaking timeline is sparse, attribution fails.
4. **PDF export incomplete**: Online export.py is a 323-line skeleton missing coaching, personality, correlations, charts, meeting notes. Offline version (905 lines) is complete and battle-tested.

## Design

### Bug 1: Dynamic Participant Appearance

**Approach**: Participants appear one-by-one as they are first detected by frame analysis during playback.

**Backend** (`realtime_pipeline.py`):
- Name scanning still runs ahead (needed for name resolution when matching labels to real names)
- `_analyze_frame_live`: REMOVE the "carry forward" logic that pads `final_participants` with ALL canonical names. Only include participants that the vision model ACTUALLY detected in the current frame.
- Camera-off participants appear only when the vision model sees them (colored circle + initials)
- The `_name_map` is still prebuilt for name resolution, but does NOT drive the participant list

**Frontend** (`useRealtimeAnalysis.ts`):
- SIGNALS handler: change from `new Map()` (replace) to updating the EXISTING `state.participants` Map
- New participants get added when first seen; existing ones get updated
- A participant, once added, stays in the Map for the rest of the session

### Bug 2: Speaking Distribution Fix

**Root cause**: SIGNALS handler runs every 1s and does:
1. Creates fresh `new Map()` — wipes everything
2. Sets `speaking_pct = 100 / participantCount` — even distribution

TRANSCRIPT handler calculates correct speaking_pct from audio durations, but next SIGNAL wipes it.

**Fix** (`useRealtimeAnalysis.ts`):
- SIGNALS handler: do NOT set `speaking_pct` at all — leave it for TRANSCRIPT handler only
- SIGNALS handler: MERGE into existing Map (not replace) — preserves transcript-computed values
- TRANSCRIPT handler: use ALL transcript segments (not just last 30) for accurate distribution
- Initial `speaking_pct` for new participants: 0 (not 100/N)

### Bug 3: Better Speaker Detection

**Root cause**: `detail: "low"` makes the vision model unable to detect:
- Teams purple/blue speaker border
- Zoom green speaker border
- Mouth movement

**Fix**:
- `core42_client.py`: change `detail: "low"` to `detail: "auto"` in `analyze_frame_all_signals`
- `core42_client.py`: add relaxed fallback in `attribute_speaker_from_context` — if consensus voting fails (no entries in 1.5s window), fall back to most recently detected speaker within 3s (single match, not voting)
- This gives the best of both: accurate voting when data is dense, reasonable fallback when sparse

### Bug 4: Port Offline PDF Template

**Approach**: Replace the online `export.py` with the offline version, adapted for the online DB schema.

**Added sections**:
- Engagement timeline chart (matplotlib PNG)
- Personality traits table
- Coaching recommendations table
- Cross-signal correlations table
- Meeting notes (action items, decisions, follow-ups, key questions)
- Full transcript (no 50-segment cap)

**Quality improvements**:
- `_safe_latin1()` sanitization (prevents crash on non-ASCII names)
- `_ensure_space()` page-break logic
- `_draw_table_multicell()` with height estimation for proper text wrapping
- Landscape orientation for wider tables
- FPDF subclass with automatic page footer
- Alternating row colors, proper borders

## Files Changed

| File | Changes |
|------|---------|
| `backend/realtime_pipeline.py` | Remove carry-forward logic in `_analyze_frame_live` |
| `backend/core42_client.py` | `detail: "low"` → `detail: "auto"`, relaxed speaker fallback |
| `backend/export.py` | Replace with offline template (adapted for online DB) |
| `frontend/src/hooks/useRealtimeAnalysis.ts` | SIGNALS: merge not replace, stop setting speaking_pct; TRANSCRIPT: use all segments |
