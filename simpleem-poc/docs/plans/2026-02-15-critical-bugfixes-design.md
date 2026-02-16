# Critical + High Severity Bugfixes Design

**Date:** 2026-02-15
**Scope:** Fix 14 critical and high-severity issues in simpleem-poc
**Approach:** Hybrid — targeted fixes + lightweight API wrapper

## Context

After a full codebase audit, 38 issues were identified. This design covers the 14 critical and high-severity bugs that affect data integrity, reliability, and user experience.

## Issues Addressed

### Critical (5)
1. Participant ID mapping falls back to string "unknown" — corrupts DB data
2. WebSocket `_send_json` silently swallows all errors — frontend stuck in stale state
3. Gesture/Reaction badges accumulate infinitely — memory leak in long sessions
4. Participant detection race condition — 6 fake participants shown then replaced
5. Duplicate emotions + signal_snapshots DB inserts — wasted space + inconsistency

### High (9)
6. No timeout on API calls — Core42 hang blocks all analysis permanently
7. No retry on transient API failures — single network blip loses data
8. No JSON validation on GPT responses — malformed responses crash finalization
9. Failed transcription chunks silently skipped — audio data permanently lost
10. Heartbeat engagement has artificial +-5% noise — misleading chart movement
11. Frontend emotion timeline array unbounded — memory grows indefinitely
12. Frontend voiceSignals/wordSignals arrays unbounded — same memory issue
13. CoachingPanel dropdown doesn't close on outside click
14. No error status surfaced to user when analysis fails

## Design

### Section 1: API Call Wrapper

**New file:** `backend/api_utils.py`

Provides `safe_api_call(coro, timeout=30, retries=1, required_keys=None)`:
- Wraps any async API coroutine with `asyncio.wait_for(timeout=30)`
- On timeout or transient error (status 429, 500, 502, 503, 504, network error): retry once after 2s backoff
- On success: validate response JSON has `required_keys` if specified
- On permanent failure: return a typed fallback dict instead of raising

All `core42_client.py` functions use this wrapper internally. No caller changes needed.

### Section 2: Participant Management

**Backend changes (`realtime_pipeline.py`):**
- Remove the 6 fake default participants on init
- Send `{"type": "detecting"}` status instead — frontend shows loading state
- Only send first `signals` after real GPT-4o detection completes
- Fix "unknown" fallback: generate a proper UUID for unmatched labels

**Frontend changes (`useRealtimeAnalysis.ts`):**
- Add `DETECTING` action type — sets `isDetecting: true`
- Engagement chart shows "Detecting participants..." overlay when `isDetecting && participants.size === 0`
- Participant sidebar shows loading skeleton when detecting

### Section 3: Remove Heartbeat Noise

**Backend (`realtime_pipeline.py`):**
- Remove `random.uniform(-5, 5)` from `_send_heartbeat`
- Carry forward exact last-known engagement values
- Chart only moves when real frame analysis returns new data

### Section 4: WebSocket Error Handling

**Backend (`realtime_pipeline.py`):**
- `_send_json`: catch `WebSocketDisconnect` specifically and set `_cancelled = True`
- Other exceptions: log at warning level (don't swallow)
- This stops all background analysis tasks when client disconnects

### Section 5: Memory Management

**Frontend (`useRealtimeAnalysis.ts`):**
- Cap `emotionTimeline` to last 2000 entries (splice oldest)
- Cap `voiceSignals` to last 200 entries
- Cap `wordSignals` to last 200 entries

**Frontend (`GesturesSignal.tsx`, `ReactionsSignal.tsx`):**
- Cap badges array to last 50 entries

### Section 6: DB Deduplication

**Backend (`realtime_pipeline.py`):**
- Remove the first emotions insert loop (lines 686-705)
- Keep `signal_snapshots` as the single source of truth
- Populate `emotions` table from signal_snapshots in a single derivation pass

### Section 7: Transcription Error Recovery

**Backend (`realtime_pipeline.py`):**
- On transcription chunk failure: retry once after 1s
- If retry fails: log warning, send status message to frontend, advance `_audio_transcribed_up_to` past failed chunk, continue processing next chunks (don't break the loop)

### Section 8: Minor UI Fixes

- CoachingPanel: add click-outside handler to close dropdown
- Surface error status to frontend when video analysis fails

## Files Changed

| File | Changes |
|------|---------|
| `backend/api_utils.py` | NEW — API call wrapper |
| `backend/core42_client.py` | Use wrapper in all API functions |
| `backend/realtime_pipeline.py` | Participant mgmt, heartbeat, WS errors, DB dedup, transcription retry |
| `frontend/src/hooks/useRealtimeAnalysis.ts` | Memory caps, detecting state, remove noise |
| `frontend/src/components/signals/GesturesSignal.tsx` | Badge cap |
| `frontend/src/components/signals/ReactionsSignal.tsx` | Badge cap |
| `frontend/src/components/CoachingPanel.tsx` | Click-outside dropdown close |

## Testing

- Upload a large video (>500MB) — verify upload completes with progress bar
- Play video end to end — verify engagement chart stops when video ends
- Play video — verify transcription starts within first 5s (not waiting for full audio)
- Kill backend during analysis — verify frontend shows disconnected state
- Run 30+ minute video — verify no memory growth in browser dev tools
- Check DB after analysis — verify no duplicate emotion entries
