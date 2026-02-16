# Simpleem Offline V4 — Interactive Pipeline & Transcript Upload

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the pipeline interactive — live transcription display, transcript upload, more parallelization, better transcription quality.

**Architecture:** FastAPI + aiosqlite backend, React 19 + Vite + Tailwind frontend. Core42 API (Whisper-1). Incremental DB saves + frontend polling for progressive display.

**Tech Stack:** Python 3.13, FastAPI, aiosqlite, React 19, Tailwind CSS, VTT/SRT parsing.

---

### Task 1: Incremental Transcript Saving During Transcription

**Goal:** Save transcript segments to DB as each chunk batch completes (not all at the end), so frontend can poll live results.

**Files:**
- Modify: `backend/transcription_engine.py`

**Implementation:**

In `_transcribe_via_api`, after each batch of chunks is transcribed, save the segments to DB immediately (instead of collecting all and saving at the end in `run_transcription`).

Move the DB save logic into the chunk loop:

```python
# After results = await asyncio.gather(*transcribe_tasks, return_exceptions=True)
# ... parse results into chunk_segments ...

# Save incrementally to DB
if chunk_segments:
    async with aiosqlite.connect(DB_PATH) as db:
        for seg in chunk_segments:
            await db.execute(
                """INSERT INTO transcript_segments
                   (video_id, start_time, end_time, text, speaker)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, seg.get("start", 0), seg.get("end", 0),
                 seg.get("text", ""), seg.get("speaker", "")),
            )
        await db.commit()
```

Also update progress detail to include segment count:
```python
f"Transcribing chunks {batch_start+1}-{min(...)}/{total} ({len(segments)} segments so far)"
```

In `run_transcription`, clear old segments before starting, and skip the bulk insert at the end (segments already saved incrementally).

**Verify:** `python3.13 -m py_compile backend/transcription_engine.py`

---

### Task 2: Live Transcript Display in Frontend

**Goal:** Show transcript segments streaming in real-time during transcription with auto-scroll.

**Files:**
- Modify: `frontend/src/components/wizard/TranscriptStep.tsx`
- Modify: `frontend/src/lib/api.ts`

**Implementation:**

In `api.ts`, add a `getLiveTranscript` function that fetches the existing transcript endpoint:
```typescript
export async function getLiveTranscript(videoId: string): Promise<TranscriptSegment[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/transcript`);
  if (!res.ok) return [];
  return res.json();
}
```

In `TranscriptStep.tsx`:
- Add a `liveSegments` state that polls `getLiveTranscript` every 2s while transcribing
- Show segments streaming in below the progress ring with auto-scroll
- New segments slide in with animation
- Keep ProgressRing at top, transcript list below
- When done, switch to the full "Review Transcript" view (existing behavior)

**Verify:** `cd frontend && npx tsc --noEmit`

---

### Task 3: Transcript Upload — Backend API

**Goal:** Accept VTT/SRT file or plain text as transcript, parse and store, skip Whisper.

**Files:**
- Modify: `backend/main.py`
- Create: `backend/transcript_parser.py`

**Implementation:**

Create `backend/transcript_parser.py` with VTT/SRT/text parsing:

```python
def parse_vtt(content: str) -> list[dict]:
    """Parse WebVTT format into segment dicts."""
    # Skip WEBVTT header, parse timestamps like 00:01:23.456 --> 00:01:25.789
    # Extract speaker from text prefix (e.g., "Speaker 1: Hello")

def parse_srt(content: str) -> list[dict]:
    """Parse SRT format into segment dicts."""
    # Parse numbered blocks with HH:MM:SS,mmm --> HH:MM:SS,mmm

def parse_plain_text(content: str) -> list[dict]:
    """Parse plain text, optionally with speaker labels and timestamps."""
    # Handle formats like:
    # [00:01] Speaker: text
    # Speaker: text
    # Just text (no timestamps)

def parse_transcript(content: str, filename: str = "") -> list[dict]:
    """Auto-detect format and parse."""
    if filename.endswith('.vtt') or content.startswith('WEBVTT'):
        return parse_vtt(content)
    elif filename.endswith('.srt') or re.match(r'^\d+\s*\n\d{2}:', content):
        return parse_srt(content)
    else:
        return parse_plain_text(content)
```

In `main.py`, add endpoint:

```python
@app.post("/api/videos/{video_id}/upload-transcript")
async def upload_transcript(video_id: str, file: UploadFile = None, body: dict = None):
    """Upload an existing transcript (VTT/SRT file or plain text)."""
    # Parse content based on format
    # Store as transcript_segments
    # Update wizard_step to 'transcribed'
```

**Verify:** `python3.13 -m py_compile backend/main.py && python3.13 -m py_compile backend/transcript_parser.py`

---

### Task 4: Transcript Upload — Frontend UI

**Goal:** Add "Upload Transcript" option to TranscriptStep with file dropzone and text paste area.

**Files:**
- Modify: `frontend/src/components/wizard/TranscriptStep.tsx`
- Modify: `frontend/src/lib/api.ts`

**Implementation:**

In `api.ts`, add:
```typescript
export async function uploadTranscript(videoId: string, file?: File, text?: string): Promise<{ status: string }> {
  // If file, use FormData; if text, use JSON body
}
```

In `TranscriptStep.tsx`:
- Add a mode toggle at the top: "Transcribe Audio" | "Upload Transcript"
- Upload mode shows:
  - File dropzone accepting .vtt, .srt, .txt
  - OR a textarea for pasting text
  - A "Use Transcript" button that uploads and skips to next step
- Transcribe mode shows existing behavior + live transcript from Task 2

**Verify:** `cd frontend && npx tsc --noEmit`

---

### Task 5: Parallelization — Name Extraction + Text Analysis

**Goal:** Speed up the pipeline by parallelizing sequential loops.

**Files:**
- Modify: `backend/batch_pipeline.py`

**Implementation:**

1. **Name extraction** — In `_extract_names`, run all frame extractions + OCR calls concurrently:
```python
# Replace sequential loop with concurrent gather
tasks = []
for ts in sample_times[:NAME_EXTRACT_COUNT]:
    if ts > duration:
        continue
    tasks.append(_extract_name_from_single_frame(video_path, ts, output_dir))
results = await asyncio.gather(*tasks, return_exceptions=True)
```

2. **Text signal analysis** — In `_analyze_text_signals`, batch all chunks and run with semaphore:
```python
SEM = asyncio.Semaphore(5)

async def _analyze_chunk(ts, text):
    async with SEM:
        voice_res, word_res = await asyncio.gather(
            _analyze_voice(text, ts),
            _analyze_words(text, ts),
        )
        return voice_res, word_res

results = await asyncio.gather(*[_analyze_chunk(ts, text) for ts, text in chunks])
```

**Verify:** `python3.13 -m py_compile backend/batch_pipeline.py`

---

### Task 6: Better Whisper Transcription Quality

**Goal:** Improve transcription accuracy with better API parameters and post-processing.

**Files:**
- Modify: `backend/core42_client.py`
- Modify: `backend/transcription_engine.py`

**Implementation:**

In `core42_client.py`, update `transcribe_audio`:
- Add `language` parameter (default "en")
- Add `prompt` parameter for context priming (e.g., "This is a business meeting recording.")
- Ensure `timestamp_granularity` is set

In `transcription_engine.py`, add post-processing after transcription:
```python
def _post_process_segments(segments: list[dict]) -> list[dict]:
    """Clean up transcription artifacts."""
    # 1. Merge very short segments (< 1s) into adjacent ones
    # 2. Remove repeated words/phrases (stuttering artifacts)
    # 3. Strip leading/trailing whitespace
    # 4. Normalize speaker labels
```

**Verify:** `python3.13 -m py_compile backend/core42_client.py && python3.13 -m py_compile backend/transcription_engine.py`

---

### Task 7: Final Verification

**Goal:** Verify everything compiles and works end-to-end.

**Steps:**
1. `python3.13 -m py_compile backend/main.py`
2. `python3.13 -m py_compile backend/batch_pipeline.py`
3. `python3.13 -m py_compile backend/transcription_engine.py`
4. `python3.13 -m py_compile backend/core42_client.py`
5. `python3.13 -m py_compile backend/transcript_parser.py`
6. `cd frontend && npx tsc --noEmit`
