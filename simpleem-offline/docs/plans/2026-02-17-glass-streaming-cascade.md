# Glass UI + Streaming Progress + Smart Cascade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the UI to a deep ocean glass aesthetic, make analysis progress engaging with streaming stage cards, and add smart model cascade fallback for resilience.

**Architecture:** Three independent workstreams: (A) CSS/Tailwind color token changes for the glass look, (B) backend SSE endpoint + frontend EventSource stage cards replacing the current polling progress, (C) a `cascade_api_call` utility that wraps model fallback logic used by all API functions.

**Tech Stack:** Tailwind CSS, React (TSX), FastAPI SSE (sse-starlette or raw StreamingResponse), EventSource API, Python asyncio

---

### Task 1: Deep Ocean Glass — CSS + Tailwind Tokens

**Files:**
- Modify: `frontend/src/index.css`
- Modify: `frontend/tailwind.config.js`

**Step 1: Update `tailwind.config.js` color tokens**

Replace the entire colors section in `frontend/tailwind.config.js`:

```js
colors: {
  'bg-primary': '#0a1628',
  'bg-secondary': '#0d1f3c',
  'bg-card': 'rgba(30,58,138,0.15)',
  'bg-card-hover': 'rgba(30,58,138,0.25)',
  'bg-card-solid': '#111d35',
  'border': 'rgba(96,165,250,0.12)',
  'border-light': 'rgba(96,165,250,0.20)',
  'border-glow': 'rgba(59,130,246,0.4)',
  'accent': '#3b82f6',
  'accent-light': '#60a5fa',
  'accent-blue': '#3b82f6',
  'accent-cyan': '#06b6d4',
  'accent-glow': 'rgba(59,130,246,0.4)',
  'success': '#10b981',
  'success-glow': 'rgba(16,185,129,0.3)',
  'warning': '#f59e0b',
  'warning-glow': 'rgba(245,158,11,0.3)',
  'danger': '#ef4444',
  'danger-glow': 'rgba(239,68,68,0.3)',
  'text-primary': '#e2e8f0',
  'text-secondary': '#94a3b8',
  'text-muted': '#64748b',
},
```

Also update box shadows to be blue-tinted:
```js
boxShadow: {
  'glass': '0 8px 32px rgba(0,10,40,0.5)',
  'glow-purple': '0 0 15px rgba(99,102,241,0.3)',
  'glow-blue': '0 0 20px rgba(59,130,246,0.3)',
  'glow-cyan': '0 0 15px rgba(6,182,212,0.3)',
  'glow-success': '0 0 15px rgba(16,185,129,0.3)',
},
```

And update gradient images:
```js
backgroundImage: {
  'gradient-primary': 'linear-gradient(135deg, #3b82f6, #6366f1, #8b5cf6)',
  'gradient-secondary': 'linear-gradient(135deg, #06b6d4, #3b82f6)',
  'gradient-success': 'linear-gradient(135deg, #10b981, #06b6d4)',
  'gradient-warm': 'linear-gradient(135deg, #f59e0b, #ef4444)',
},
```

**Step 2: Update `index.css` body and glass classes**

Replace the `body` rule background:
```css
body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background-color: #0a1628;
  background-image:
    radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.12) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 20%, rgba(99,102,241,0.10) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.08) 0%, transparent 50%);
  color: #e2e8f0;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

Replace `.glass-card`:
```css
.glass-card {
  background: rgba(15,35,80,0.45);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(96,165,250,0.12);
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0,10,40,0.3);
}
```

Replace `.glass-card-hover:hover`:
```css
.glass-card-hover:hover {
  background: rgba(20,45,100,0.50);
  border-color: rgba(59,130,246,0.35);
  box-shadow: 0 0 20px rgba(59,130,246,0.15);
}
```

Replace `.glass-card-expanded`:
```css
.glass-card-expanded {
  background: rgba(20,45,100,0.55);
  border-color: rgba(96,165,250,0.22);
  box-shadow:
    0 0 30px rgba(59,130,246,0.12),
    0 8px 32px rgba(0,10,40,0.5),
    inset 0 1px 0 rgba(96,165,250,0.08);
}
```

Replace `.glass-depth-1`:
```css
.glass-depth-1 {
  background: rgba(15,35,80,0.25);
  border: 1px solid rgba(96,165,250,0.08);
  border-radius: 12px;
}
```

Replace `.glass-depth-2`:
```css
.glass-depth-2 {
  background: rgba(20,45,100,0.30);
  border: 1px solid rgba(96,165,250,0.12);
  border-radius: 12px;
}
```

Replace `.gradient-text`:
```css
.gradient-text {
  background: linear-gradient(135deg, #60a5fa, #818cf8, #22d3ee);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

Replace `.gradient-border::before` background:
```css
background: linear-gradient(135deg, #3b82f6, #6366f1, #06b6d4);
```

Update glow-pulse to use blue:
```css
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 5px rgba(59,130,246,0.2); }
  50% { box-shadow: 0 0 20px rgba(59,130,246,0.4); }
}
```

**Step 3: Visual verification**

Run: `open http://localhost:5174`
Expected: Deep navy background with blue-tinted glass cards, blue accent glows, cyan highlights.

**Step 4: Commit**

```bash
git add frontend/src/index.css frontend/tailwind.config.js
git commit -m "feat: deep ocean glass UI — navy background, blue-tinted glass cards"
```

---

### Task 2: Smart Cascade Model Fallback — api_utils.py

**Files:**
- Modify: `backend/api_utils.py`

**Step 1: Add `cascade_api_call` function**

Append after the existing `safe_api_call` function in `backend/api_utils.py`:

```python
async def cascade_api_call(
    call_factory,
    *,
    models: list[str],
    timeout: float = API_TIMEOUT,
    retries: int = MAX_RETRIES,
    required_keys: list[str] | None = None,
    fallback: dict | list | None = None,
    label: str = "cascade_call",
) -> dict | list:
    """Try multiple models in sequence until one succeeds.

    Args:
        call_factory: A callable that takes a model name and returns a coroutine factory.
                      Usage: call_factory(model) returns a zero-arg callable for safe_api_call.
        models: Ordered list of model names to try.
        timeout, retries, required_keys: Passed through to safe_api_call.
        fallback: Final fallback if ALL models fail.
        label: Human-readable label for logs.
    """
    for i, model in enumerate(models):
        try:
            result = await safe_api_call(
                call_factory(model),
                timeout=timeout,
                retries=retries,
                required_keys=required_keys,
                fallback=None,  # Don't use fallback yet — try next model
                label=f"{label}_{model}",
            )
            if result is not None:
                if i > 0:
                    logger.info(f"[{label}] Succeeded with fallback model {model} (attempt {i + 1})")
                return result
        except Exception as e:
            logger.warning(f"[{label}] Model {model} failed: {e}")
            if i < len(models) - 1:
                logger.info(f"[{label}] Trying next model: {models[i + 1]}")
            continue

    logger.error(f"[{label}] All {len(models)} models failed, using fallback")
    if fallback is not None:
        return fallback
    raise RuntimeError(f"{label}: all models exhausted")
```

**Step 2: Commit**

```bash
git add backend/api_utils.py
git commit -m "feat: add cascade_api_call for multi-model fallback"
```

---

### Task 3: MODEL_CASCADE Config + Update core42_client.py Functions

**Files:**
- Modify: `backend/core42_client.py`

**Step 1: Add MODEL_CASCADE dict after model constants (after line ~44)**

Add after the `WHISPER_MODEL` line:

```python
# Smart cascade: ordered fallback models per task category
MODEL_CASCADE = {
    "vision":        [VISION_MODEL, CHAT_MODEL, CHAT_MODEL_FAST],
    "summary":       [CHAT_MODEL_STRONG, CHAT_MODEL, VISION_MODEL],
    "analysis":      [CHAT_MODEL, VISION_MODEL, CHAT_MODEL_FAST],
    "fast":          [CHAT_MODEL_FAST, CHAT_MODEL, VISION_MODEL],
    "transcription": [WHISPER_MODEL],
}
```

Update the import at top of file:
```python
from .api_utils import safe_api_call, cascade_api_call
```

**Step 2: Update `generate_meeting_summary` to use `cascade_api_call`**

Replace the current model loop (try/except with for loop) in `generate_meeting_summary` with:

```python
    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                **_model_kwargs(model, max_tokens=3000, temperature=0.4),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["summary"],
        timeout=120,
        required_keys=["summary"],
        fallback=fallback,
        label="generate_meeting_summary",
    )
```

**Step 3: Update `generate_meeting_notes` the same way**

Replace the model loop with:

```python
    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                **_model_kwargs(model, max_tokens=2000, temperature=0.3),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["summary"],
        timeout=90,
        required_keys=["action_items"],
        fallback=fallback,
        label="generate_meeting_notes",
    )
```

**Step 4: Update `analyze_frame_triple_vote` to skip GPT-5 for vision**

In the `analyze_frame_triple_vote` function, change the model list from:
```python
model_tasks = [
    _analyze_frame_with_model(frame_path, timestamp, VISION_MODEL, known_names, camera_on_names),
    _analyze_frame_with_model(frame_path, timestamp, CHAT_MODEL, known_names, camera_on_names),
    _analyze_frame_with_model(frame_path, timestamp, CHAT_MODEL_STRONG, known_names, camera_on_names),
]
```

To use the vision cascade (dropping GPT-5):
```python
vision_models = MODEL_CASCADE["vision"][:3]  # gpt-4o, gpt-4.1, gpt-4o-mini
model_tasks = [
    _analyze_frame_with_model(frame_path, timestamp, m, known_names, camera_on_names)
    for m in vision_models
]
```

**Step 5: Update remaining functions that use single models**

For `generate_flags_and_scores` — change `model=CHAT_MODEL` to use cascade:
- Replace the single `safe_api_call` with `cascade_api_call` using `MODEL_CASCADE["analysis"]`.

For `analyze_voice_signal` and `analyze_word_signal` (fast tasks):
- These already use `CHAT_MODEL_FAST` via `safe_api_call`. Wrap each with `cascade_api_call` using `MODEL_CASCADE["fast"]`.

For `generate_coaching` (uses `CHAT_MODEL_FAST`):
- Wrap with `cascade_api_call` using `MODEL_CASCADE["fast"]`.

**Step 6: Commit**

```bash
git add backend/core42_client.py
git commit -m "feat: smart cascade model fallback for all API functions"
```

---

### Task 4: Backend — Monotonic Progress + SSE Event Stream

**Files:**
- Modify: `backend/batch_pipeline.py`
- Modify: `backend/main.py`

**Step 1: Fix progress going backward**

In `batch_pipeline.py`, modify `_update_progress` to track and enforce monotonic progress:

```python
# Module-level dict to track max progress per video
_max_progress: dict[str, float] = {}

async def _update_progress(
    video_id: str,
    wizard_step: str,
    progress: float,
    detail: str,
):
    """Write analysis progress — never goes backward."""
    # Enforce monotonic progress
    prev = _max_progress.get(video_id, 0)
    if wizard_step == "complete":
        progress = 100.0
    else:
        progress = max(progress, prev)
    _max_progress[video_id] = progress

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE videos
               SET wizard_step = ?, analyze_progress = ?, progress_detail = ?
               WHERE id = ?""",
            (wizard_step, round(progress, 2), detail, video_id),
        )
        await db.commit()
```

Also add cleanup when analysis starts (at the top of `run_analysis`):
```python
_max_progress.pop(video_id, None)  # Reset for fresh analysis
```

And cleanup when analysis completes/errors:
```python
_max_progress.pop(video_id, None)  # Cleanup memory
```

**Step 2: Add SSE event emitter**

Add an in-memory event queue system to `batch_pipeline.py`:

```python
import queue
from collections import defaultdict

# Per-video event queues for SSE streaming
_event_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

def subscribe_events(video_id: str) -> asyncio.Queue:
    """Subscribe to analysis events for a video. Returns an asyncio.Queue."""
    q: asyncio.Queue = asyncio.Queue()
    _event_queues[video_id].append(q)
    return q

def unsubscribe_events(video_id: str, q: asyncio.Queue):
    """Remove a subscriber queue."""
    if video_id in _event_queues:
        _event_queues[video_id] = [x for x in _event_queues[video_id] if x is not q]
        if not _event_queues[video_id]:
            del _event_queues[video_id]

async def _emit_event(video_id: str, event: dict):
    """Push an event to all subscribers for this video."""
    for q in _event_queues.get(video_id, []):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if subscriber is too slow
```

**Step 3: Emit events during analysis**

Add `_emit_event` calls at key points in the `run_analysis` function:

- After participant detection: `await _emit_event(video_id, {"type": "stage_complete", "stage": "participants", "data": {"names": participant_names}})`
- During frame analysis progress callback: `await _emit_event(video_id, {"type": "result_preview", "stage": "frames", "data": {"participant": name, "emotion": emotion, "engagement": eng}})`
- When audio phase starts/completes: `await _emit_event(video_id, {"type": "stage_start/stage_complete", "stage": "audio", ...})`
- When text phase completes: `await _emit_event(video_id, {"type": "stage_complete", "stage": "text", ...})`
- When summary is ready: `await _emit_event(video_id, {"type": "result_preview", "stage": "summary", "data": {"preview": summary[:150]}})`
- On completion: `await _emit_event(video_id, {"type": "complete"})`

**Step 4: Add SSE endpoint to main.py**

```python
from starlette.responses import StreamingResponse
from backend.batch_pipeline import subscribe_events, unsubscribe_events

@app.get("/api/videos/{video_id}/analysis-stream")
async def analysis_stream(video_id: str):
    """SSE endpoint for real-time analysis events."""
    q = subscribe_events(video_id)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "complete" or event.get("type") == "error":
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            unsubscribe_events(video_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Step 5: Commit**

```bash
git add backend/batch_pipeline.py backend/main.py
git commit -m "feat: monotonic progress + SSE event stream for analysis"
```

---

### Task 5: Frontend — Streaming Stage Cards

**Files:**
- Modify: `frontend/src/components/wizard/AnalysisStep.tsx`
- Modify: `frontend/src/lib/api.ts` (add SSE helper if needed)

**Step 1: Replace AnalysisStep.tsx with streaming stage cards**

Rewrite the component to:
1. Connect to `/api/videos/{vid}/analysis-stream` via `EventSource`
2. Maintain state per stage: `{ status: 'waiting'|'active'|'done', results: string[] }`
3. Track a monotonic `maxProgress` that never decreases
4. Still poll `getVideoStatus` as a fallback (SSE can disconnect)
5. On `complete` event → call `onComplete()`

Key state shape:
```typescript
interface StageState {
  status: 'waiting' | 'active' | 'done';
  detail: string;
  results: string[];  // streaming result previews
}

const [stages, setStages] = useState<Record<string, StageState>>({
  participants: { status: 'waiting', detail: '', results: [] },
  frames: { status: 'waiting', detail: '', results: [] },
  audio: { status: 'waiting', detail: '', results: [] },
  text: { status: 'waiting', detail: '', results: [] },
  profiles: { status: 'waiting', detail: '', results: [] },
  summary: { status: 'waiting', detail: '', results: [] },
});
```

Each stage card renders as a glass card:
- **Waiting**: dimmed, collapsed (1 line)
- **Active**: blue glow border, expanded, shows streaming `results` list items animating in
- **Done**: green border, collapsed with checkmark, expandable on click

Top progress bar uses `maxProgress` (never decreases).

Multiple stages can be `active` simultaneously (frames, audio, text run concurrently).

Streaming results show as an animated feed inside the active card:
```
  ┌ Anuj Sharma: engaged (78%), leaning forward
  ├ Robin Mills: attentive (65%), upright posture
  └ Mahrous: neutral (52%), steady gaze
```

Keep polling `getVideoStatus` every 3s as fallback — if SSE disconnects, stages still update from progress thresholds (current behavior as backup).

**Step 2: Verify the full experience**

Run: `open http://localhost:5174`
Trigger analysis, observe:
- Deep ocean glass background
- Stage cards expand/collapse with streaming results
- Progress bar only goes forward
- Multiple stages active simultaneously
- Completion transitions to dashboard

**Step 3: Commit**

```bash
git add frontend/src/components/wizard/AnalysisStep.tsx frontend/src/lib/api.ts
git commit -m "feat: streaming stage cards with live analysis results"
```

---

### Task 6: Integration Test + Final Polish

**Files:**
- All modified files

**Step 1: End-to-end test**

1. Start both servers (`python -m uvicorn backend.main:app --port 8001 --reload` + `npm run dev`)
2. Upload a video or use existing one
3. Trigger analysis
4. Verify:
   - Glass UI renders correctly (navy background, blue cards)
   - Progress never goes backward
   - Stage cards stream live results
   - Summary generates (GPT-5 → GPT-4.1 fallback)
   - PDF export works
   - CSV export works

**Step 2: Test cascade fallback**

Temporarily set `CHAT_MODEL_STRONG=nonexistent-model` in .env to force fallback.
Verify that summary still generates using the next model in the cascade.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: glass UI + streaming progress + smart cascade — integration verified"
```
