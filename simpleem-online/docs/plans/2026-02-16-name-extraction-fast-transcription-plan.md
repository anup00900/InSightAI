# Name Extraction + Fast Transcription + Per-Person Insights — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract real participant names from meeting video frames, make transcription 3x faster with parallel workers, and deliver per-person named insights.

**Architecture:** On WebSocket connect, GPT-4o OCR reads name labels from frame 0. A name map (`Person 1` → `Anup Roy`) is built and propagated to all subsequent signals, transcript, coaching, and summary. Transcription uses 2 parallel workers with smaller chunks (2s) and lower trigger threshold (1.0s). Speaker assignment maps Whisper output to real names via GPT-4.1 batch calls every ~10s.

**Tech Stack:** Python 3.9+ (backend), GPT-4o vision (name OCR), GPT-4.1 (speaker assignment), Whisper (transcription), React/TypeScript (frontend), WebSocket (real-time communication)

---

### Task 1: Add `extract_names_from_frame()` to core42_client.py

**Files:**
- Modify: `backend/core42_client.py`

**Step 1: Add the name extraction function**

Add this function after the existing `analyze_frame_all_signals` function (around line 318):

```python
async def extract_names_from_frame(frame_path: str) -> dict[str, str]:
    """Extract participant display names from meeting UI overlays using GPT-4o vision.

    Reads name tags visible in Zoom/Teams/Meet/WebEx screenshots.
    Returns a mapping like {"Person 1": "Anup Roy", "Person 2": "John Smith"}.
    If no names are detectable, returns empty dict.
    """
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    fallback: dict[str, str] = {}

    async def _call():
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at reading text overlays in video conference screenshots. "
                        "Your job is to identify participant name labels visible in the meeting UI. "
                        "These are the name tags shown by Zoom, Microsoft Teams, Google Meet, WebEx, etc. "
                        "They typically appear at the bottom of each participant's video tile."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this meeting screenshot and read ALL visible participant name labels. "
                                "These are the text overlays shown by the video conferencing software "
                                "(usually at the bottom of each person's video tile).\n\n"
                                "For each person visible (numbered left-to-right, top-to-bottom as Person 1, Person 2, etc.), "
                                "read their displayed name from the UI overlay.\n\n"
                                "If a person's name label is not visible or unreadable, omit them.\n\n"
                                "Return ONLY valid JSON:\n"
                                '{"names": {"Person 1": "Anup Roy", "Person 2": "John Smith"}}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.1,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        parsed = json.loads(content)
        return parsed.get("names", {})

    return await safe_api_call(
        _call,
        timeout=30,
        fallback=fallback,
        label="extract_names_from_frame",
    )
```

**Step 2: Add `assign_speakers_to_names()` function**

Add this after `extract_names_from_frame`:

```python
async def assign_speakers_to_names(
    transcript_segments: list[dict],
    known_names: list[str],
) -> list[dict]:
    """Assign real speaker names to transcript segments using GPT-4.1.

    Takes raw transcript segments (from Whisper, no speaker info) and a list of
    known participant names. GPT infers who said what based on context, turn-taking,
    and content.

    Returns the same segments with 'speaker' field updated to real names.
    """
    if not known_names or not transcript_segments:
        return transcript_segments

    # Format segments for the prompt
    seg_text = "\n".join(
        f"[{s.get('start', 0):.1f}s-{s.get('end', 0):.1f}s]: {s.get('text', '')}"
        for s in transcript_segments
    )

    fallback = transcript_segments  # Return unmodified on failure

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at speaker identification in meeting transcripts. "
                        "Given a list of transcript segments and the known participants, "
                        "assign each segment to the most likely speaker based on: "
                        "conversational flow, topic changes, response patterns, and turn-taking cues."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Known participants: {', '.join(known_names)}\n\n"
                        f"Transcript segments:\n{seg_text}\n\n"
                        "For each segment, assign the most likely speaker. "
                        "If unsure, use your best guess based on conversation flow.\n\n"
                        "Return ONLY a valid JSON array of speaker names in order, "
                        "one per segment:\n"
                        '["Anup Roy", "John Smith", "Anup Roy", ...]'
                    ),
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        speakers = json.loads(content)

        if isinstance(speakers, list) and len(speakers) == len(transcript_segments):
            for i, seg in enumerate(transcript_segments):
                if speakers[i] in known_names:
                    seg["speaker"] = speakers[i]
        return transcript_segments

    return await safe_api_call(
        _call,
        timeout=20,
        fallback=fallback,
        label="assign_speakers_to_names",
    )
```

**Step 3: Verify no syntax errors**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.core42_client import extract_names_from_frame, assign_speakers_to_names; print('OK')"`
Expected: `OK`

---

### Task 2: Wire name extraction into realtime_pipeline.py initialize()

**Files:**
- Modify: `backend/realtime_pipeline.py`

**Step 1: Add import for new functions**

At the top of the file, add to the import from `core42_client` (line 22-31):

```python
from .core42_client import (
    analyze_frame_all_signals,
    analyze_voice_signal,
    analyze_words_signal,
    analyze_personality,
    build_correlations,
    transcribe_audio,
    generate_coaching,
    generate_summary_and_flags,
    extract_names_from_frame,
    assign_speakers_to_names,
)
```

**Step 2: Add name_map and speaker assignment state to `__init__`**

Add these fields to `__init__` (after `self.participant_labels` around line 89):

```python
        # Name mapping: "Person 1" -> "Anup Roy"
        self._name_map: dict[str, str] = {}
        self._speaker_assignment_segments: list[dict] = []  # Buffer for batch speaker assignment
        self._last_speaker_assignment_ts: float = 0.0
        self._speaker_assignment_interval: float = 10.0  # Assign speakers every 10s of audio
```

**Step 3: Update `initialize()` to extract names**

Replace the existing `initialize()` method with:

```python
    async def initialize(self):
        """Called on WebSocket connect. Start detection + audio extraction + name OCR immediately
        so results are ready by the time user hits play (~1s head start)."""
        logger.info(f"[{self.video_id[:8]}] Session initialized — pre-loading...")
        await self._send_status("Loading analysis...")
        # Start all three in parallel: detection, audio extraction, name OCR
        self._audio_task = asyncio.create_task(self._extract_audio_background())
        self._detect_task = asyncio.create_task(self._detect_participants_and_names())
```

**Step 4: Create `_detect_participants_and_names()` method**

Replace the existing `_detect_participants()` with:

```python
    async def _detect_participants_and_names(self):
        """Analyze frame 0 to detect participants AND extract real names from UI overlays.
        Runs both GPT-4o calls in parallel for speed."""
        try:
            frame_path = await extract_single_frame(self.video_path, 0.0, self.output_dir)

            # Run both analyses in parallel on the same frame
            signals_task = analyze_frame_all_signals(frame_path, 0.0)
            names_task = extract_names_from_frame(frame_path)
            result, name_map = await asyncio.gather(signals_task, names_task, return_exceptions=True)

            # Handle name extraction result
            if isinstance(name_map, dict) and name_map:
                self._name_map = name_map
                logger.info(f"[{self.video_id[:8]}] Extracted names: {name_map}")
                # Send name map to frontend
                await self._send_json({"type": "name_map", "data": name_map})
            elif isinstance(name_map, Exception):
                logger.warning(f"[{self.video_id[:8]}] Name extraction failed: {name_map}")

            # Handle signals result
            if isinstance(result, Exception):
                logger.error(f"[{self.video_id[:8]}] Participant detection failed: {result}")
                await self._send_status("Ready — press play")
                return

            result["timestamp"] = 0.0

            # Post-process: apply name map and ensure gestures/reactions
            for p in result.get("participants", []):
                label = p.get("label", "Person 1")
                # Apply real name if available
                if label in self._name_map:
                    p["label"] = self._name_map[label]
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
            logger.error(f"[{self.video_id[:8]}] Detection failed: {e}")
            await self._send_status("Ready — press play")
```

**Step 5: Update `_analyze_frame_live()` to apply name map**

In `_analyze_frame_live()`, after the line `result["timestamp"] = timestamp` (around line 285), update the participant post-processing to apply name map:

Replace the participant loop (lines 288-293) with:

```python
                for p in result.get("participants", []):
                    raw_label = p.get("label", "Person 1")
                    # Apply real name if available
                    if raw_label in self._name_map:
                        p["label"] = self._name_map[raw_label]
                    self.participant_labels.add(p.get("label", raw_label))
                    if not p.get("gestures"):
                        p["gestures"] = ["still posture", "resting hands"]
                    if not p.get("reactions"):
                        p["reactions"] = ["neutral gaze", "steady expression"]
```

**Step 6: Verify**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.realtime_pipeline import RealtimeSession; print('OK')"`
Expected: `OK`

---

### Task 3: Parallel transcription workers

**Files:**
- Modify: `backend/realtime_pipeline.py`

**Step 1: Update constants**

Change the existing constants in the class:

```python
    AUDIO_CHUNK_SECONDS = 2.0   # Smaller chunks = faster individual responses
    AUDIO_MIN_CHUNK = 1.0       # Trigger threshold — start worker every 1.0s of new audio
    MAX_PARALLEL_TRANSCRIBE = 2  # Number of parallel transcription workers
```

**Step 2: Replace serial transcription state with parallel state**

In `__init__`, replace the transcription state block (lines 74-80) with:

```python
        # Audio streaming state
        self._audio_path: str | None = None
        self._audio_ready = False
        self._audio_extracted = asyncio.Event()
        self._audio_transcribed_up_to = 0.0
        self._transcribe_lock = asyncio.Lock()  # Coordinate parallel workers
        self._active_transcribe_count = 0       # Number of workers currently running
        self._transcribe_queue: float = 0.0     # Queued target timestamp (latest wins)
        self._transcript_segments: list[dict] = []
        self._next_segment_idx = 0
        self._audio_task: asyncio.Task | None = None
        self._transcribe_tasks: list[asyncio.Task] = []
```

**Step 3: Update `handle_tick()` transcription trigger**

Replace the transcription section in `handle_tick()` (the block at lines 228-232) with:

```python
        # 3. Transcription — parallel workers, never drops audio
        if timestamp - self._audio_transcribed_up_to >= self.AUDIO_MIN_CHUNK:
            self._transcribe_queue = timestamp  # Always update target
            if self._active_transcribe_count < self.MAX_PARALLEL_TRANSCRIBE:
                task = asyncio.create_task(self._transcription_worker())
                self._transcribe_tasks.append(task)
                task.add_done_callback(lambda t: self._transcribe_tasks.remove(t) if t in self._transcribe_tasks else None)
```

**Step 4: Rewrite `_transcription_worker()` with lock-based range claiming**

Replace the entire `_transcription_worker` method with:

```python
    async def _transcription_worker(self, force: bool = False):
        """Parallel transcription worker. Claims a range under lock, transcribes, releases.

        Multiple workers can run concurrently on non-overlapping ranges.
        """
        self._active_transcribe_count += 1
        min_chunk = 0.5 if force else self.AUDIO_WORKER_MIN
        try:
            while not self._cancelled:
                # Claim a range under lock
                async with self._transcribe_lock:
                    target = self._transcribe_queue
                    start = self._audio_transcribed_up_to

                    if target - start < min_chunk:
                        break  # Nothing worth transcribing

                    # Claim up to AUDIO_CHUNK_SECONDS from the start
                    end = min(start + self.AUDIO_CHUNK_SECONDS, target)
                    self._audio_transcribed_up_to = end  # Advance immediately so next worker takes next range

                logger.info(f"[{self.video_id[:8]}] Transcribing [{start:.1f}-{end:.1f}]...")

                try:
                    if self._audio_ready and self._audio_path:
                        chunk_path = await extract_audio_chunk(
                            self._audio_path, start, end, self.output_dir
                        )
                    else:
                        chunk_path = await extract_audio_chunk_from_video(
                            self.video_path, start, end, self.output_dir
                        )

                    # Run transcription and audio analysis in parallel
                    transcript_task = transcribe_audio(chunk_path)
                    audio_task = self._analyze_audio_features(chunk_path, start, end)
                    transcript_data, _ = await asyncio.gather(
                        transcript_task, audio_task, return_exceptions=True
                    )

                    if isinstance(transcript_data, Exception):
                        logger.error(f"[{self.video_id[:8]}] Transcription failed [{start:.1f}-{end:.1f}]: {transcript_data}")
                        continue

                    segments = transcript_data.get("segments", [])
                    for seg in segments:
                        text = seg.get("text", "").strip()
                        if not text:
                            continue
                        seg["start"] = seg.get("start", 0) + start
                        seg["end"] = seg.get("end", 0) + start
                        self._transcript_segments.append(seg)
                        self._speaker_assignment_segments.append(seg)

                    logger.info(f"[{self.video_id[:8]}] Transcribed [{start:.1f}-{end:.1f}]: {len(segments)} segments")

                    # Immediately deliver
                    await self._send_pending_transcript(float("inf"))

                    # Voice/word signal analysis
                    chunk_text = " ".join(s.get("text", "") for s in segments)
                    if chunk_text.strip():
                        task = asyncio.create_task(self._analyze_audio_signals(chunk_text, (start + end) / 2))
                        task.add_done_callback(self._log_task_exception)

                    # Batch speaker assignment every ~10s
                    if (end - self._last_speaker_assignment_ts >= self._speaker_assignment_interval
                            and self._speaker_assignment_segments
                            and self._name_map):
                        asyncio.create_task(self._assign_speakers_batch())

                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

                except Exception as e:
                    logger.error(f"[{self.video_id[:8]}] Transcription [{start:.1f}-{end:.1f}] failed: {e}")

                # Check if more audio was queued
                if self._transcribe_queue <= self._audio_transcribed_up_to:
                    break

        finally:
            self._active_transcribe_count -= 1
```

**Step 5: Add `_analyze_audio_features()` helper**

Add this method after `_transcription_worker`:

```python
    async def _analyze_audio_features(self, chunk_path: str, start: float, end: float):
        """Analyze audio features from a transcription chunk."""
        try:
            features = self.audio_analyzer.analyze_file(chunk_path)
            audio_data = {
                "timestamp": (start + end) / 2,
                "energy": features.engagement_score,
                "pitch_mean": features.pitch_mean,
                "pitch_std": features.pitch_std,
                "volume_energy": features.volume_energy,
                "speaking_rate": features.speaking_rate,
                "pause_count": features.pause_count,
            }
            self.audio_features.append(audio_data)
            if not self._cancelled and not self._finalizing:
                await self._send_json({"type": "audio_features", "data": audio_data})
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] Audio analysis failed: {e}")
```

**Step 6: Add `_assign_speakers_batch()` method**

Add after `_analyze_audio_features`:

```python
    async def _assign_speakers_batch(self):
        """Batch-assign real speaker names to accumulated transcript segments."""
        if not self._name_map or not self._speaker_assignment_segments:
            return

        segments_to_assign = self._speaker_assignment_segments.copy()
        self._speaker_assignment_segments.clear()
        self._last_speaker_assignment_ts = segments_to_assign[-1].get("end", 0)

        known_names = list(self._name_map.values())
        try:
            assigned = await assign_speakers_to_names(segments_to_assign, known_names)
            # Send updated transcript segments with real speaker names
            for seg in assigned:
                if seg.get("speaker"):
                    await self._send_json({
                        "type": "speaker_update",
                        "data": {
                            "start_time": seg.get("start", 0),
                            "speaker": seg["speaker"],
                        },
                    })
            logger.info(f"[{self.video_id[:8]}] Assigned speakers to {len(assigned)} segments")
        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Speaker assignment failed: {e}")
```

**Step 7: Update `cleanup()` to cancel parallel tasks**

In `cleanup()`, replace the `_transcribe_task` cancellation with:

```python
        for task in self._transcribe_tasks:
            if not task.done():
                task.cancel()
```

**Step 8: Update `finalize()` transcription section**

In `finalize()`, replace the serial transcription wait (lines 584-592) with:

```python
        # Transcribe ALL remaining audio
        duration = await get_video_duration(self.video_path)
        if self._audio_ready and self._audio_transcribed_up_to < duration:
            self._transcribe_queue = duration
            # Wait for any running workers to finish
            if self._transcribe_tasks:
                await asyncio.gather(*self._transcribe_tasks, return_exceptions=True)
            # Force-transcribe any remaining audio
            if self._audio_transcribed_up_to < duration:
                await self._transcription_worker(force=True)

        # Final speaker assignment for any remaining segments
        if self._speaker_assignment_segments and self._name_map:
            await self._assign_speakers_batch()
```

**Step 9: Verify**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.realtime_pipeline import RealtimeSession; print('OK')"`
Expected: `OK`

---

### Task 4: Frontend — handle name_map and speaker_update WebSocket messages

**Files:**
- Modify: `frontend/src/lib/websocket.ts`
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts`

**Step 1: Add new message types to websocket.ts**

In `websocket.ts`, add `'name_map'` and `'speaker_update'` to the `MessageType` union (line 6-23):

```typescript
export type MessageType =
  | 'signals'
  | 'transcript'
  | 'voice'
  | 'words'
  | 'personality'
  | 'correlation'
  | 'flag'
  | 'summary'
  | 'coaching'
  | 'status'
  | 'error'
  | 'no_audio'
  | 'detecting'
  | 'video_ended_ack'
  | 'complete'
  | 'audio_features'
  | 'engagement_alert'
  | 'name_map'
  | 'speaker_update';
```

**Step 2: Add state and actions to useRealtimeAnalysis.ts**

Add `nameMap` to `RealtimeState`:

```typescript
export interface RealtimeState {
  // ... existing fields ...
  nameMap: Record<string, string>;  // "Person 1" -> "Anup Roy"
}
```

Initialize it in `initialState`:

```typescript
  nameMap: {},
```

Add new action types:

```typescript
type Action =
  // ... existing actions ...
  | { type: 'NAME_MAP'; data: Record<string, string> }
  | { type: 'SPEAKER_UPDATE'; data: { start_time: number; speaker: string } };
```

**Step 3: Add reducer cases**

Add these cases to the reducer:

```typescript
    case 'NAME_MAP': {
      const nameMap = action.data;
      // Rename existing participants
      const newParticipants = new Map<string, Partial<Participant>>();
      for (const [key, pData] of state.participants) {
        const newName = nameMap[key] || key;
        newParticipants.set(newName, { ...pData, id: newName, name: newName });
      }
      return { ...state, nameMap, participants: newParticipants };
    }

    case 'SPEAKER_UPDATE': {
      // Update speaker name on matching transcript segment
      const { start_time, speaker } = action.data;
      const newTranscript = state.transcript.map((seg) =>
        Math.abs(seg.start_time - start_time) < 0.5 ? { ...seg, speaker } : seg
      );
      return { ...state, transcript: newTranscript };
    }
```

**Step 4: Wire up WebSocket handlers in the hook**

In the `useEffect` where handlers are registered, add:

```typescript
    ws.on('name_map', (data) => dispatch({ type: 'NAME_MAP', data: data as Record<string, string> }));
    ws.on('speaker_update', (data) => dispatch({ type: 'SPEAKER_UPDATE', data: data as { start_time: number; speaker: string } }));
```

**Step 5: Update SIGNALS reducer to apply name map**

In the `SIGNALS` case, update the participant loop to apply the name map:

```typescript
      for (const p of snap.participants) {
        // Apply name map: "Person 1" -> "Anup Roy"
        const participantId = state.nameMap[p.label] || p.label;
```

This replaces the existing `const participantId = p.label;` on line 173.

**Step 6: TypeScript check**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc/frontend" && npx tsc --noEmit`
Expected: No errors

---

### Task 5: Pre-buffer first audio chunk during initialize()

**Files:**
- Modify: `backend/realtime_pipeline.py`

**Step 1: Add pre-transcription to initialize()**

Update `initialize()` to also pre-transcribe the first 2s of audio. Add after the detection task:

```python
    async def initialize(self):
        """Called on WebSocket connect. Start detection + audio extraction + name OCR immediately
        so results are ready by the time user hits play (~1s head start)."""
        logger.info(f"[{self.video_id[:8]}] Session initialized — pre-loading...")
        await self._send_status("Loading analysis...")
        # Start detection, audio extraction, AND first transcription chunk in parallel
        self._audio_task = asyncio.create_task(self._extract_audio_background())
        self._detect_task = asyncio.create_task(self._detect_participants_and_names())
        # Pre-transcribe first 2s so transcript is ready instantly on play
        asyncio.create_task(self._pre_transcribe_first_chunk())
```

**Step 2: Add `_pre_transcribe_first_chunk()` method**

```python
    async def _pre_transcribe_first_chunk(self):
        """Pre-transcribe the first 2s of audio during initialization.
        This way transcript appears instantly when user presses play."""
        try:
            # Wait a moment for audio extraction to start (but don't wait for completion)
            await asyncio.sleep(0.5)
            # Extract directly from video (don't wait for full audio extraction)
            chunk_path = await extract_audio_chunk_from_video(
                self.video_path, 0.0, 2.0, self.output_dir
            )
            transcript_data = await transcribe_audio(chunk_path)
            segments = transcript_data.get("segments", [])
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                seg["start"] = seg.get("start", 0)
                seg["end"] = seg.get("end", 0)
                self._transcript_segments.append(seg)
                self._speaker_assignment_segments.append(seg)

            self._audio_transcribed_up_to = 2.0
            logger.info(f"[{self.video_id[:8]}] Pre-transcribed first 2s: {len(segments)} segments")

            try:
                os.remove(chunk_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] Pre-transcription failed: {e}")
```

**Step 3: Verify**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.realtime_pipeline import RealtimeSession; print('OK')"`
Expected: `OK`

---

### Task 6: Update finalization and DB persistence with real names

**Files:**
- Modify: `backend/realtime_pipeline.py`

**Step 1: Update `finalize()` to use real names**

In the `finalize()` method, replace `participant_names = sorted(self.participant_labels) or ["Person 1"]` with:

```python
            participant_names = sorted(self.participant_labels) or ["Person 1"]
            # If we have a name map, use real names
            if self._name_map:
                participant_names = [
                    self._name_map.get(name, name)
                    for name in participant_names
                ]
                # Deduplicate (in case mapped names overlap)
                participant_names = sorted(set(participant_names))
```

**Step 2: Verify and restart**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.realtime_pipeline import RealtimeSession; print('OK')"`
Expected: `OK`

Restart backend:
```bash
lsof -ti:8000 | xargs kill -9; sleep 1
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
```

---

### Task 7: TypeScript check and E2E test

**Files:**
- No new files

**Step 1: TypeScript compilation**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc/frontend" && npx tsc --noEmit`
Expected: 0 errors

**Step 2: Backend import check**

Run: `cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-poc" && /usr/bin/python3 -c "from backend.realtime_pipeline import RealtimeSession; from backend.core42_client import extract_names_from_frame, assign_speakers_to_names; print('ALL OK')"`
Expected: `ALL OK`

**Step 3: E2E API test**

Run:
```bash
VIDEO_ID=$(curl -s http://localhost:8000/api/videos | python3 -c "import sys,json; vids=json.load(sys.stdin); print(vids[0]['id'] if vids else '')")
echo "Video: $VIDEO_ID"
curl -s http://localhost:8000/api/videos/$VIDEO_ID | python3 -c "import sys,json; v=json.load(sys.stdin); print(f'Status: {v[\"status\"]}')"
```

**Step 4: Verify frontend loads**

Open: `http://localhost:5173`
Expected: App loads, no console errors. Upload a meeting video and verify:
- Names appear (not "Person 1") within first 2-3 seconds
- Transcript starts appearing within 1-2 seconds of pressing play
- Speaker names in transcript are real names
- Coaching uses real names
