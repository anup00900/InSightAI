# Critical Bugfixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 14 critical and high-severity bugs in simpleem-poc covering API reliability, participant management, memory leaks, data integrity, and UX issues.

**Architecture:** Hybrid approach — a new `api_utils.py` wrapper handles the 4 API-related bugs (timeout, retry, validation, error handling) as a single abstraction. All other fixes are targeted per-file edits.

**Tech Stack:** Python/FastAPI backend, React/TypeScript frontend, Core42/Azure OpenAI API, WebSocket, SQLite.

**Design doc:** `docs/plans/2026-02-15-critical-bugfixes-design.md`

---

### Task 1: Create API Call Wrapper (`backend/api_utils.py`)

Fixes issues: #6 (no timeout), #7 (no retry), #8 (no JSON validation)

**Files:**
- Create: `backend/api_utils.py`

**Step 1: Create the wrapper module**

```python
"""Lightweight wrapper for Core42 API calls with timeout, retry, and validation."""

import asyncio
import logging

logger = logging.getLogger(__name__)

API_TIMEOUT = 30  # seconds
MAX_RETRIES = 1
RETRY_DELAY = 2.0  # seconds


async def safe_api_call(
    coro_factory,
    *,
    timeout: float = API_TIMEOUT,
    retries: int = MAX_RETRIES,
    required_keys: list[str] | None = None,
    fallback: dict | None = None,
    label: str = "api_call",
) -> dict:
    """Execute an async API call with timeout, retry, and response validation.

    Args:
        coro_factory: A zero-argument callable that returns a new coroutine each call.
                      Must be a factory (lambda/function) because coroutines can only be awaited once.
        timeout: Max seconds to wait per attempt.
        retries: Number of retries on transient errors (0 = no retry).
        required_keys: If set, validate response dict contains these keys.
        fallback: Dict to return on total failure. If None, re-raises the exception.
        label: Human-readable label for log messages.
    """
    last_error = None
    for attempt in range(1 + retries):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=timeout)

            # Validate required keys if specified
            if required_keys and isinstance(result, dict):
                missing = [k for k in required_keys if k not in result]
                if missing:
                    logger.warning(f"[{label}] Response missing keys: {missing}")
                    if fallback is not None:
                        return fallback

            return result

        except asyncio.TimeoutError:
            last_error = TimeoutError(f"{label} timed out after {timeout}s")
            logger.warning(f"[{label}] Timeout (attempt {attempt + 1}/{1 + retries})")

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_transient = any(code in error_str for code in ["429", "500", "502", "503", "504", "rate", "connection", "timeout"])

            if not is_transient or attempt >= retries:
                logger.error(f"[{label}] Failed: {e}")
                if fallback is not None:
                    return fallback
                raise

            logger.warning(f"[{label}] Transient error (attempt {attempt + 1}): {e}")

        # Wait before retry
        if attempt < retries:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    # All attempts exhausted
    if fallback is not None:
        logger.error(f"[{label}] All attempts failed, using fallback")
        return fallback
    raise last_error or RuntimeError(f"{label} failed with no error captured")
```

**Step 2: Verify the file was created correctly**

Run: `python -c "import backend.api_utils; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add backend/api_utils.py
git commit -m "feat: add API call wrapper with timeout, retry, and validation"
```

---

### Task 2: Integrate API Wrapper into Core42 Client

Fixes issues: #6 (no timeout), #7 (no retry), #8 (no JSON validation)

**Files:**
- Modify: `backend/core42_client.py`

**Step 1: Add import at top of file**

After the existing imports, add:
```python
from .api_utils import safe_api_call
```

**Step 2: Wrap `transcribe_audio` function**

Replace the body of `transcribe_audio` (lines 38-56) with:

```python
async def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Core42 transcription model via OpenAI-compatible endpoint."""
    async def _call():
        with open(audio_path, "rb") as f:
            try:
                response = await audio_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            except Exception:
                f.seek(0)
                response = await audio_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                )
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    return await safe_api_call(
        _call,
        timeout=60,  # Audio transcription can take longer
        fallback={"segments": [], "text": ""},
        label="transcribe_audio",
    )
```

**Step 3: Wrap `analyze_frame_all_signals` function**

Wrap the API call portion inside `analyze_frame_all_signals` (lines 201-302). The function already has a try/except with fallback, so wrap just the `client.chat.completions.create` call:

After the `response = await client.chat.completions.create(...)` block (around line 209-275), wrap it:

```python
async def analyze_frame_all_signals(frame_path: str, timestamp: float) -> dict:
    """Analyze a single frame for ALL visual signals."""
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    fallback = {
        "participants": [{
            "label": "Person 1",
            "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
            "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "unable to analyze"},
            "gestures": ["still posture", "resting hands"],
            "reactions": ["neutral gaze", "steady expression"],
        }]
    }

    async def _call():
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[...],  # Keep existing messages exactly as-is
            max_tokens=2000,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        parsed = json.loads(content)
        for p in parsed.get("participants", []):
            if not p.get("gestures"):
                p["gestures"] = ["still posture", "resting hands"]
            if not p.get("reactions"):
                p["reactions"] = ["neutral gaze", "steady expression"]
        return parsed

    return await safe_api_call(
        _call,
        timeout=30,
        required_keys=["participants"],
        fallback=fallback,
        label="analyze_frame",
    )
```

**Step 4: Wrap remaining API functions the same way**

Apply the same pattern to:
- `analyze_voice_signal` — timeout=15, fallback=`{"tone":"neutral","pace":"moderate","energy":50,"dynamics":"unable to analyze"}`
- `analyze_words_signal` — timeout=15, fallback=`{"sentiment":"neutral","sentiment_score":0.5,"keywords":[],"key_phrases":[]}`
- `analyze_personality` — timeout=30, fallback=`{"participants":[{"label":"Person 1","traits":["neutral"],"communication_style":"conversational","confidence_level":"moderate"}]}`
- `build_correlations` — timeout=30, fallback=`{"correlations":[]}`
- `generate_summary_and_flags` — timeout=60, required_keys=`["summary","participant_scores"]`
- `generate_coaching` — timeout=30, fallback=`[{"recommendation":"Maintain engagement","category":"engagement","priority":3}]`

**Step 5: Commit**

```bash
git add backend/core42_client.py
git commit -m "feat: wrap all Core42 API calls with timeout, retry, and validation"
```

---

### Task 3: Fix Participant Management — Remove Fake Defaults

Fixes issues: #4 (race condition), #1 (unknown ID fallback)

**Files:**
- Modify: `backend/realtime_pipeline.py` (initialize method, lines 101-140)
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts` (add DETECTING state)

**Step 1: Backend — Replace fake participants with detecting status**

In `realtime_pipeline.py`, replace the `initialize` method's fake participant block (lines 107-121) with:

```python
    async def initialize(self):
        """Called on WebSocket connect. Detect participants from first frame."""
        logger.info(f"[{self.video_id[:8]}] Session initialized — detecting participants...")
        await self._send_json({"type": "detecting"})
        await self._send_status("Detecting participants...")

        # Extract audio in background
        self._audio_task = asyncio.create_task(self._extract_audio_background())

        # Detect participants from frame 0 (blocks until complete — no fake defaults)
        self._detect_task = asyncio.create_task(self._detect_participants())

        # Pre-warm: analyze frames at 3s intervals
        try:
            duration = await get_video_duration(self.video_path)
            pre_warm_timestamps = [float(t) for t in range(3, int(duration), 3)][:5]
            for ts in pre_warm_timestamps:
                task = asyncio.create_task(self._analyze_frame_live(ts))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            if pre_warm_timestamps:
                logger.info(f"[{self.video_id[:8]}] Pre-warming {len(pre_warm_timestamps)} frames")
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] Pre-warm failed: {e}")
```

**Step 2: Backend — Fix _detect_participants to not reference defaults**

In `_detect_participants` (around line 142), remove the merge block that references `self._last_known_signal` defaults (lines 158-162), and remove the neutral signal send. Just send the real detected signal:

```python
    async def _detect_participants(self):
        """Analyze frame 0 to detect all participants."""
        try:
            frame_path = await extract_single_frame(self.video_path, 0.0, self.output_dir)
            result = await analyze_frame_all_signals(frame_path, 0.0)
            result["timestamp"] = 0.0

            for p in result.get("participants", []):
                self.participant_labels.add(p.get("label", "Person 1"))
                if not p.get("gestures"):
                    p["gestures"] = ["still posture", "resting hands"]
                if not p.get("reactions"):
                    p["reactions"] = ["neutral gaze", "steady expression"]

            self.signal_snapshots.append(result)
            self._last_known_signal = result

            await self._send_json({"type": "signals", "data": result})
            logger.info(f"[{self.video_id[:8]}] Detected {len(result.get('participants', []))} participants")
            await self._send_status("Ready — press play")

            try:
                os.remove(frame_path)
            except OSError:
                pass

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Participant detection failed: {e}")
            await self._send_status("Ready — press play")
```

**Step 3: Backend — Fix "unknown" participant ID fallback in _persist_to_db**

In `_persist_to_db`, find the two blocks that fall back to `pid = "unknown"` (around lines 687-690 and 710-714). Replace with proper UUID generation:

```python
                # Instead of: pid = "unknown"
                # Use:
                if not pid:
                    pid = str(uuid.uuid4())
                    participant_id_map[label] = pid
                    # Also insert this participant
                    await db.execute(
                        """INSERT INTO participants (id, video_id, name, engagement_score,
                           sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pid, self.video_id, label, 50, 0.3, 0, 50, 50, 50),
                    )
```

**Step 4: Frontend — Add DETECTING state**

In `useRealtimeAnalysis.ts`:

Add to `RealtimeState` interface:
```typescript
  isDetecting: boolean;
```

Add to `initialState`:
```typescript
  isDetecting: true,
```

Add to Action union type:
```typescript
  | { type: 'DETECTING' }
```

Add reducer case:
```typescript
    case 'DETECTING':
      return { ...state, isDetecting: true };
```

In the `SIGNALS` case, add after the return:
```typescript
      // First real signal means detection is complete
      return {
        ...state,
        signals: newSignals,
        emotionTimeline: [...state.emotionTimeline, ...newEmotions],
        participants: newParticipants,
        isAnalyzing: true,
        isDetecting: false,  // <-- ADD THIS
      };
```

Add subscription in the hook:
```typescript
    ws.on('detecting', () => dispatch({ type: 'DETECTING' }));
```

**Step 5: Commit**

```bash
git add backend/realtime_pipeline.py frontend/src/hooks/useRealtimeAnalysis.ts
git commit -m "fix: remove fake participants, wait for real detection, fix unknown ID fallback"
```

---

### Task 4: Remove Heartbeat Artificial Noise

Fixes issue: #10 (misleading +-5% noise)

**Files:**
- Modify: `backend/realtime_pipeline.py` (`_send_heartbeat` method, around line 298)

**Step 1: Replace the heartbeat method**

Replace the engagement variation logic in `_send_heartbeat`:

```python
    async def _send_heartbeat(self, timestamp: float):
        """Send a signal carrying forward exact last known engagement.
        No random variation — chart only moves on real analysis results.
        """
        if not self._last_known_signal or self._cancelled or self._finalizing:
            return

        heartbeat = {
            "timestamp": timestamp,
            "participants": self._last_known_signal.get("participants", []),
            "is_heartbeat": True,
        }
        await self._send_json({"type": "signals", "data": heartbeat})
```

**Step 2: Remove the `import random` if no longer used elsewhere**

Check if `random` is used elsewhere in the file. It's used in `_make_neutral_signal` — but with Task 3 we removed that. If `random` is no longer used anywhere, remove the import.

**Step 3: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "fix: remove artificial engagement noise from heartbeat signals"
```

---

### Task 5: Fix WebSocket Silent Error Swallowing

Fixes issue: #2 (frontend stuck in stale state)

**Files:**
- Modify: `backend/realtime_pipeline.py` (`_send_json` method, around line 877)

**Step 1: Replace `_send_json` method**

```python
    async def _send_json(self, data: dict):
        """Send JSON over WebSocket. Cancels session on disconnect."""
        try:
            await self.ws.send_json(data)
        except (WebSocketDisconnect, RuntimeError):
            # Client disconnected — stop all analysis
            if not self._cancelled:
                logger.info(f"[{self.video_id[:8]}] WebSocket disconnected, cancelling session")
                self._cancelled = True
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] WebSocket send error: {e}")
```

**Step 2: Add WebSocketDisconnect import if not already present**

At the top of `realtime_pipeline.py`, ensure this import exists:
```python
from fastapi import WebSocket, WebSocketDisconnect
```

Note: Check if `WebSocketDisconnect` is already imported. If not, add it. Currently only `WebSocket` is imported (line 16).

**Step 3: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "fix: detect WebSocket disconnect and cancel session instead of swallowing errors"
```

---

### Task 6: Cap Frontend Memory — State Arrays

Fixes issues: #11 (unbounded emotionTimeline), #12 (unbounded voice/word signals)

**Files:**
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts`

**Step 1: Add size constants at top of file**

After the imports:
```typescript
const MAX_EMOTION_TIMELINE = 2000;
const MAX_VOICE_SIGNALS = 200;
const MAX_WORD_SIGNALS = 200;
```

**Step 2: Cap emotionTimeline in SIGNALS reducer**

In the SIGNALS case, change the return statement:

```typescript
      let newTimeline = [...state.emotionTimeline, ...newEmotions];
      if (newTimeline.length > MAX_EMOTION_TIMELINE) {
        newTimeline = newTimeline.slice(-MAX_EMOTION_TIMELINE);
      }

      return {
        ...state,
        signals: newSignals,
        emotionTimeline: newTimeline,
        participants: newParticipants,
        isAnalyzing: true,
        isDetecting: false,
      };
```

**Step 3: Cap voiceSignals in VOICE reducer**

```typescript
    case 'VOICE': {
      let newVoice = [...state.voiceSignals, action.data];
      if (newVoice.length > MAX_VOICE_SIGNALS) {
        newVoice = newVoice.slice(-MAX_VOICE_SIGNALS);
      }
      // ... rest unchanged
```

**Step 4: Cap wordSignals in WORDS reducer**

```typescript
    case 'WORDS': {
      let newWords = [...state.wordSignals, action.data];
      if (newWords.length > MAX_WORD_SIGNALS) {
        newWords = newWords.slice(-MAX_WORD_SIGNALS);
      }
      // ... rest unchanged
```

**Step 5: Commit**

```bash
git add frontend/src/hooks/useRealtimeAnalysis.ts
git commit -m "fix: cap emotion timeline and signal arrays to prevent memory leaks"
```

---

### Task 7: Cap Frontend Memory — Gesture/Reaction Badges

Fixes issue: #3 (memory leak)

**Files:**
- Modify: `frontend/src/components/signals/GesturesSignal.tsx` (line 35)
- Modify: `frontend/src/components/signals/ReactionsSignal.tsx` (line 35)

**Step 1: Fix GesturesSignal.tsx**

Replace the badge accumulation line (line 35):

```typescript
    // OLD: setBadges((prev) => [...prev, ...newBadges]);
    // NEW: Cap at 50 badges to prevent memory leaks in long sessions
    setBadges((prev) => {
      const combined = [...prev, ...newBadges];
      return combined.length > 50 ? combined.slice(-50) : combined;
    });
```

**Step 2: Fix ReactionsSignal.tsx**

Same change — replace line 35:

```typescript
    setBadges((prev) => {
      const combined = [...prev, ...newBadges];
      return combined.length > 50 ? combined.slice(-50) : combined;
    });
```

**Step 3: Commit**

```bash
git add frontend/src/components/signals/GesturesSignal.tsx frontend/src/components/signals/ReactionsSignal.tsx
git commit -m "fix: cap gesture and reaction badges at 50 to prevent memory leak"
```

---

### Task 8: Fix DB Deduplication — Remove Duplicate Emotions Insert

Fixes issue: #5 (duplicate DB inserts)

**Files:**
- Modify: `backend/realtime_pipeline.py` (`_persist_to_db` method)

**Step 1: Locate the duplicate insert blocks**

In `_persist_to_db`, there are two blocks that iterate over `self.signal_snapshots`:
1. Lines ~686-705: Insert into `emotions` table
2. Lines ~707-738: Insert into `signal_snapshots` table (richer data)

**Step 2: Remove the first emotions insert loop**

Delete the entire first block (lines ~682-705) that inserts into the `emotions` table. Instead, derive emotions from signal_snapshots after that insert:

```python
                # Insert signal snapshots (primary source of truth)
                for snap in self.signal_snapshots:
                    ts = snap.get("timestamp", 0)
                    for p in snap.get("participants", []):
                        label = p.get("label", "Person 1")
                        pid = participant_id_map.get(label)
                        if not pid:
                            pid = str(uuid.uuid4())
                            participant_id_map[label] = pid
                            await db.execute(
                                """INSERT INTO participants (id, video_id, name, engagement_score,
                                   sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (pid, self.video_id, label, 50, 0.3, 0, 50, 50, 50),
                            )
                        emotions = p.get("emotions", {})
                        bl = p.get("body_language", {})

                        # Insert into signal_snapshots (rich data)
                        await db.execute(
                            """INSERT INTO signal_snapshots ... """,  # Keep existing INSERT exactly
                            ...  # Keep existing params exactly
                        )

                        # Derive emotions entry from the same data (no duplicate loop)
                        await db.execute(
                            """INSERT INTO emotions (video_id, participant_id, timestamp,
                               emotion, confidence, engagement) VALUES (?, ?, ?, ?, ?, ?)""",
                            (self.video_id, pid, ts,
                             emotions.get("primary", "neutral"),
                             emotions.get("confidence", 0.5),
                             emotions.get("engagement", 50)),
                        )
```

**Step 3: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "fix: remove duplicate emotions insert, derive from signal_snapshots"
```

---

### Task 9: Transcription Error Recovery with Retry

Fixes issue: #9 (failed chunks silently lost)

**Files:**
- Modify: `backend/realtime_pipeline.py` (`_transcription_worker` method)

**Step 1: Replace the error handling block in the transcription worker**

Find the `except Exception` block in `_transcription_worker` (around line 444-448). Replace:

```python
                except Exception as e:
                    logger.error(f"[{self.video_id[:8]}] Transcription chunk [{start:.1f}-{end:.1f}] failed: {e}")
                    # Skip this chunk to avoid infinite retry
                    self._audio_transcribed_up_to = end
                    break
```

With retry logic:

```python
                except Exception as e:
                    logger.warning(f"[{self.video_id[:8]}] Transcription [{start:.1f}-{end:.1f}] failed, retrying: {e}")
                    # Retry once after 1s
                    await asyncio.sleep(1.0)
                    try:
                        if self._audio_ready and self._audio_path:
                            chunk_path = await extract_audio_chunk(
                                self._audio_path, start, end, self.output_dir
                            )
                        else:
                            chunk_path = await extract_audio_chunk_from_video(
                                self.video_path, start, end, self.output_dir
                            )
                        transcript_data = await transcribe_audio(chunk_path)
                        segments = transcript_data.get("segments", [])
                        for seg in segments:
                            text = seg.get("text", "").strip()
                            if not text:
                                continue
                            seg["start"] = seg.get("start", 0) + start
                            seg["end"] = seg.get("end", 0) + start
                            self._transcript_segments.append(seg)
                        self._audio_transcribed_up_to = end
                        await self._send_pending_transcript(float("inf"))
                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass
                    except Exception as e2:
                        logger.error(f"[{self.video_id[:8]}] Transcription [{start:.1f}-{end:.1f}] retry also failed: {e2}")
                        await self._send_status(f"Transcription failed at {start:.0f}s — continuing...")
                        self._audio_transcribed_up_to = end
                        # Continue to next chunk instead of breaking
                        continue
```

Note: Change `break` to `continue` in the final fallback so the worker processes the next chunk.

**Step 2: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "fix: retry failed transcription chunks once before skipping"
```

---

### Task 10: CoachingPanel Click-Outside to Close Dropdown

Fixes issue: #13

**Files:**
- Modify: `frontend/src/components/CoachingPanel.tsx`

**Step 1: Add useRef and click-outside effect**

Add `useRef` to the import:
```typescript
import { useState, useEffect, useRef } from 'react';
```

Add ref and effect after the existing state:
```typescript
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!dropdownOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);
```

**Step 2: Wrap the dropdown container with the ref**

Change the `<div className="relative mb-4">` to:
```typescript
      <div className="relative mb-4" ref={dropdownRef}>
```

**Step 3: Commit**

```bash
git add frontend/src/components/CoachingPanel.tsx
git commit -m "fix: close coaching dropdown on click outside"
```

---

### Task 11: Surface Error Status to Frontend

Fixes issue: #14

**Files:**
- Modify: `backend/realtime_pipeline.py` (`finalize` method, error handler)

**Step 1: Improve error reporting in finalize**

In the `finalize` method's `except Exception` block (around line 628):

```python
        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Finalization error: {e}")
            await self._send_json({
                "type": "error",
                "data": {"message": f"Analysis error: {str(e)[:200]}"},
            })
            await self._send_json({"type": "complete"})  # Still mark complete so UI unblocks
```

**Step 2: Commit**

```bash
git add backend/realtime_pipeline.py
git commit -m "fix: surface analysis errors to frontend instead of silently failing"
```

---

### Task 12: Final Verification

**Step 1: Check Python syntax**

Run: `python -m py_compile backend/api_utils.py && python -m py_compile backend/core42_client.py && python -m py_compile backend/realtime_pipeline.py`
Expected: No output (success)

**Step 2: Check TypeScript compilation**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

**Step 3: Start the application**

Run: `cd /Users/anup.roy/Downloads/Simpleem\ Work/simpleem-poc && ./run.sh`
Expected: Both backend and frontend start without errors

**Step 4: Manual verification checklist**

- [ ] Upload a video — see progress bar with percentage
- [ ] Press play — engagement chart starts after real detection (no 6 fake participants)
- [ ] During playback — transcription segments appear within 5s of speech
- [ ] Let video play 30s — chart moves smoothly, no artificial jitter
- [ ] Stop video — engagement chart freezes immediately
- [ ] After video ends — summary and coaching appear
- [ ] Open coaching tab — dropdown closes when clicking outside
- [ ] Check browser dev tools Memory tab — no runaway growth over time

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final verification pass for critical bugfixes"
```
