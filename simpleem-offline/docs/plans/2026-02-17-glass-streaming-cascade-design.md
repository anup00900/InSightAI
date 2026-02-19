# Design: Deep Ocean Glass UI + Streaming Progress + Smart Cascade Fallback

**Date:** 2026-02-17
**Status:** Approved

## Problem

1. UI background is near-black (#0a0a1a) — user wants glass blueish look, not dark void
2. Progress bar goes backward (40% -> 29% -> 60%) because concurrent tasks overwrite each other
3. User stares at a spinner with nothing to see during 2-5 min analysis
4. GPT-5 fails consistently (empty responses) with no reliable fallback for most functions

## Solution

### Part A — Deep Ocean Glass UI

**Background:** `#0a1628` (deep navy) with stronger blue/cyan radial glows at 12-15% opacity.

**Glass cards:** Blue-tinted `rgba(30,58,138,0.15)` with `backdrop-filter: blur(24px)`, inner glow border `rgba(96,165,250,0.15)`.

**Changes:**
- `index.css`: bg-primary, radial gradients, glass-card classes → blue tint
- `tailwind.config.js`: bg-primary, bg-secondary, bg-card tokens → blue palette
- Hover states, borders, depth layers → blue cast

### Part B — Streaming Stage Cards

**Backend:** New SSE endpoint `GET /api/videos/{vid}/analysis-stream` streaming JSON events:
- `stage_start`, `stage_progress`, `result_preview`, `stage_complete`, `error`
- Progress: `max(current, previous_max)` — never goes backward
- Each model result streams a preview immediately

**Frontend:** Replace AnalysisStep with expandable stage cards:
- 6 stages: Participants, Video Frames, Audio, Text, Profiles, Summary
- States: Waiting (dimmed) → Active (pulsing, streaming results) → Done (green, collapsible)
- Multiple stages can be Active simultaneously (frames/audio/text run concurrently)
- Top-level progress bar only goes forward

### Part C — Smart Cascade Model Fallback

```python
MODEL_CASCADE = {
    "vision":        ["gpt-4o", "gpt-4.1", "gpt-4o-mini"],
    "summary":       ["gpt-5", "gpt-4.1", "gpt-4o"],
    "analysis":      ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
    "fast":          ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
    "transcription": ["whisper-1"],
}
```

New `cascade_api_call(category, call_factory, ...)` in api_utils.py. Each existing function gets a cascade instead of a single model.

Triple-vote for frames: gpt-4o + gpt-4.1 + gpt-4o-mini (drop gpt-5 from vision — it fails).

## Files to Change

**Part A (Glass UI):**
- `frontend/src/index.css` — background, glass classes, glows
- `frontend/tailwind.config.js` — color tokens

**Part B (Streaming Progress):**
- `backend/main.py` — new SSE endpoint
- `backend/batch_pipeline.py` — emit SSE events during analysis
- `frontend/src/components/wizard/AnalysisStep.tsx` — stage cards with streaming

**Part C (Cascade Fallback):**
- `backend/api_utils.py` — new `cascade_api_call` function
- `backend/core42_client.py` — MODEL_CASCADE config, update all functions to use cascades
