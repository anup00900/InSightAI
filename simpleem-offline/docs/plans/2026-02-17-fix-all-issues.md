# Fix PDF, Speaking Distribution, Summary, Voice Analysis, Signal Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 critical issues — PDF export crashes, speaking distribution miscalculation, broken summary, hardcoded voice analysis, and last-snapshot-only signal components — across backend and frontend.

**Architecture:** Backend-first approach. Fix data quality issues in `batch_pipeline.py` and `core42_client.py` first (speaking %, summary, voice analysis), then fix `export.py` for PDF, then update frontend components to consume the improved data. New DB tables for meeting notes and per-speaker audio features.

**Tech Stack:** Python/FastAPI (backend), React/TypeScript/Vite (frontend), SQLite via aiosqlite, librosa, fpdf2, Core42/Azure OpenAI API.

---

## Task 1: Fix PDF Export — Backend Crash Guards

**Files:**
- Modify: `backend/export.py:317-323` (None guards)
- Modify: `backend/export.py:447-464` (transcript truncation)
- Modify: `backend/export.py:453-460` (multi_cell width fix)

**Step 1: Fix None guards in `_build_pdf` metadata line**

In `backend/export.py`, line 317-323, the `round(video["overall_score"])` crashes when `overall_score` is None. Fix:

```python
# Line 317-323 — replace the meta line block
duration_min = round((video["duration"] or 0) / 60, 1)
overall = round(video["overall_score"]) if video["overall_score"] else 0
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(*GRAY)
pdf.cell(
    0, 6,
    f'{video["name"]}  |  {duration_min} min  |  {len(participants)} participants  |  Score: {overall}%',
    new_x="LMARGIN", new_y="NEXT",
)
```

**Step 2: Fix summary None guard**

Line 329 — `summary["summary"]` crashes when summary is None but the code already checks. No change needed, but ensure the `summary_text` assignment is safe:

```python
summary_text = summary["summary"] if summary else "No summary available."
```

Already correct. Move on.

**Step 3: Fix transcript rendering — actually truncate at 50 + fix width**

Replace lines 447-465 in `_build_pdf`:

```python
_section_heading(pdf, "Transcript", DARK)
pdf.set_font("Helvetica", "", 9)
max_segments = 50
for seg in transcript[:max_segments]:
    ts = _format_time(seg["start_time"])
    speaker = seg["speaker"] or ""
    text = seg["text"] or ""
    pdf.set_text_color(*GRAY)
    pdf.cell(16, 5, ts)
    speaker_w = 0
    if speaker:
        pdf.set_text_color(*BLUE)
        # Truncate long speaker names to prevent negative width
        display_speaker = speaker[:20] + "..." if len(speaker) > 20 else speaker
        speaker_w = 28
        pdf.cell(speaker_w, 5, display_speaker)
    pdf.set_text_color(*DARK)
    remaining_w = max(40, pdf.w - pdf.l_margin - pdf.r_margin - 16 - speaker_w)
    pdf.multi_cell(remaining_w, 5, text)
if len(transcript) > max_segments:
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, f"Showing first {max_segments} of {len(transcript)} segments...", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
```

**Step 4: Run backend to verify PDF generates without crash**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
python -c "
import asyncio
from backend.export import generate_pdf_report
# This will fail if no video exists, but we verify import works
print('PDF export module loads OK')
"
```

**Step 5: Commit**

```bash
git add backend/export.py
git commit -m "fix: PDF export crash guards — None safety, transcript truncation, speaker width"
```

---

## Task 2: Fix Speaking Distribution — Backend Normalization

**Files:**
- Modify: `backend/batch_pipeline.py:942-954` (speaking pct calculation)

**Step 1: Replace speaking percentage calculation**

In `batch_pipeline.py`, replace lines 942-954 (the `# --- Compute ACTUAL speaking percentages ---` block) with:

```python
# --- Compute ACTUAL speaking percentages from transcript ---
speaker_word_counts: dict[str, int] = {}
unmatched_words = 0
for seg in transcript_segments:
    speaker = seg.get("speaker", "")
    words = len(seg.get("text", "").split())
    if speaker:
        matched_name = name_lookup.get(_normalize_name(speaker)) or _fuzzy_match_name(speaker, participant_names)
        if matched_name:
            speaker_word_counts[matched_name] = speaker_word_counts.get(matched_name, 0) + words
        else:
            unmatched_words += words
    else:
        unmatched_words += words

# Distribute unmatched words proportionally among matched speakers
if unmatched_words > 0 and speaker_word_counts:
    total_matched = sum(speaker_word_counts.values())
    for name in speaker_word_counts:
        share = speaker_word_counts[name] / max(total_matched, 1)
        speaker_word_counts[name] += int(unmatched_words * share)
elif unmatched_words > 0 and not speaker_word_counts:
    # No speaker matched at all — distribute equally
    per_person = max(1, unmatched_words // max(len(participant_names), 1))
    for name in participant_names:
        speaker_word_counts[name] = per_person

logger.info(f"[{video_id[:8]}] Speaker word counts: {speaker_word_counts} (unmatched: {unmatched_words})")
total_words = max(sum(speaker_word_counts.values()), 1)

# Normalize to exactly 100%
raw_pcts = {name: (count / total_words) * 100 for name, count in speaker_word_counts.items()}
# Ensure all participants are included (even those who didn't speak)
for name in participant_names:
    if name not in raw_pcts:
        raw_pcts[name] = 0

# Round and adjust to sum to exactly 100
rounded_pcts = {name: round(pct) for name, pct in raw_pcts.items()}
total_rounded = sum(rounded_pcts.values())
if total_rounded != 100 and total_rounded > 0:
    # Add/subtract difference to the largest speaker
    diff = 100 - total_rounded
    largest = max(rounded_pcts, key=lambda k: rounded_pcts[k])
    rounded_pcts[largest] = max(0, rounded_pcts[largest] + diff)

actual_speaking_pct = rounded_pcts
logger.info(f"[{video_id[:8]}] Speaking distribution (normalized): {actual_speaking_pct}")
```

**Step 2: Commit**

```bash
git add backend/batch_pipeline.py
git commit -m "fix: speaking distribution normalization — handle unmatched speakers, sum to 100%"
```

---

## Task 3: Fix Speaking Distribution — Frontend Donut Normalization

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx:156-162` (speaking data normalization)

**Step 1: Replace the speaking data normalization block**

Replace lines 156-162:

```typescript
// Speaking distribution - normalize to always sum to 100
const speakingData = participants.map(p => ({ name: p.name, pct: p.speaking_pct }));
const totalSpk = speakingData.reduce((s, d) => s + d.pct, 0);
const normalizedSpk = totalSpk > 0
  ? speakingData.map(d => ({ ...d, pct: Math.round((d.pct / totalSpk) * 100) }))
  : speakingData.map(d => ({ ...d, pct: Math.round(100 / Math.max(participants.length, 1)) }));
```

**Step 2: Commit**

```bash
git add frontend/src/components/AnalysisDashboard.tsx
git commit -m "fix: donut chart always normalizes speaking distribution to 100%"
```

---

## Task 4: Summary — New Dedicated GPT Functions

**Files:**
- Modify: `backend/core42_client.py` (add `generate_meeting_summary`, `generate_meeting_notes`, rename old function)

**Step 1: Add `generate_meeting_summary` function**

Add after line 723 (after `build_correlations`):

```python
async def generate_meeting_summary(transcript: str) -> dict:
    """Generate a dedicated meeting summary using GPT-5 — focused only on summarization."""
    fallback = {
        "summary": "Meeting analysis completed. Summary generation encountered an issue.",
        "key_topics": ["meeting"],
        "overall_sentiment": "neutral",
    }

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL_STRONG,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting analyst and technical writer. "
                        "Your task is to write a clear, detailed meeting summary.\n\n"
                        "RULES:\n"
                        "- Write 3-4 substantive paragraphs covering: context/purpose, key discussions, "
                        "decisions reached, and overall dynamics/outcomes.\n"
                        "- Be specific — reference actual topics, names, and decisions from the transcript.\n"
                        "- Identify 4-8 key topics discussed.\n"
                        "- Determine overall sentiment (positive/neutral/negative) based on tone and outcomes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write a comprehensive summary of this meeting.\n\n"
                        f"FULL TRANSCRIPT:\n{_fit_context(transcript, 80000)}\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"summary": "3-4 detailed paragraphs...", '
                        '"key_topics": ["topic1", "topic2", ...], '
                        '"overall_sentiment": "positive|neutral|negative"}'
                    ),
                },
            ],
            max_completion_tokens=3000,
            temperature=0.4,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=90,
        required_keys=["summary"],
        fallback=fallback,
        label="generate_meeting_summary",
    )


async def generate_meeting_notes(transcript: str) -> dict:
    """Generate structured meeting notes — action items, decisions, follow-ups."""
    fallback = {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL_STRONG,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting secretary. Extract structured, actionable notes "
                        "from the meeting transcript. Be precise and specific.\n\n"
                        "For each action item: include WHO is responsible and WHAT they need to do.\n"
                        "For each decision: include WHAT was decided and any conditions.\n"
                        "For follow-ups: include WHAT needs follow-up and by WHEN if mentioned.\n"
                        "For key questions: include questions that were raised but NOT answered."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract structured meeting notes from this transcript:\n\n"
                        f"{_fit_context(transcript, 80000)}\n\n"
                        "Return ONLY valid JSON:\n"
                        "{\n"
                        '  "action_items": [{"owner": "Name", "task": "What to do", "deadline": "if mentioned or null"}],\n'
                        '  "decisions": [{"decision": "What was decided", "context": "Brief context"}],\n'
                        '  "follow_ups": [{"topic": "What needs follow-up", "owner": "Name or null", "timeline": "if mentioned or null"}],\n'
                        '  "key_questions": [{"question": "Unanswered question", "raised_by": "Name or null"}]\n'
                        "}"
                    ),
                },
            ],
            max_completion_tokens=2000,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=60,
        required_keys=["action_items"],
        fallback=fallback,
        label="generate_meeting_notes",
    )
```

**Step 2: Rename `generate_summary_and_flags` to `generate_flags_and_scores`**

Keep the existing function but rename it and remove the summary generation from its prompt. Change its system prompt to focus ONLY on flags + participant scores. Remove the "summary" and "key_topics" from its output format. Update `required_keys` to `["flags", "participant_scores"]`.

In lines 725-798, rename the function and update the prompt:

```python
async def generate_flags_and_scores(transcript: str, emotions_summary: str, participant_names: list[str]) -> dict:
    """Generate critical moment flags and per-participant scores."""
    fallback = {
        "flags": [],
        "participant_scores": [
            {
                "name": name,
                "engagement": 70, "sentiment": 0.3,
                "speaking_pct": round(100 / max(len(participant_names), 1)),
                "clarity": 70, "rapport": 70, "energy": 70,
            }
            for name in participant_names
        ],
    }

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting analyst. Identify critical moments (flags) and score each participant.\n\n"
                        "CRITICAL RULES:\n"
                        "- flags: Identify 3-8 critical moments with ACCURATE timestamps from the transcript.\n"
                        "- engagement: Active speakers with positive emotions = 70-90, silent/passive = 30-50, camera-off = 20-40.\n"
                        "- sentiment: Positive speakers = 0.5 to 1.0, negative = -0.5 to -1.0.\n"
                        "- clarity/rapport/energy: Score based on actual communication patterns. DO NOT use same score for everyone."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze this meeting with {len(participant_names)} participants.\n\n"
                        f"PARTICIPANTS: {', '.join(participant_names)}\n\n"
                        f"TRANSCRIPT:\n{_fit_context(transcript, 50000)}\n\n"
                        f"EMOTION/ENGAGEMENT DATA:\n{_fit_context(emotions_summary, 30000)}\n\n"
                        "Return ONLY valid JSON:\n"
                        "{\n"
                        '  "flags": [{"timestamp": 123.0, "type": "positive|negative", "description": "what happened", "severity": "low|medium|high"}],\n'
                        '  "participant_scores": [\n'
                        '    {"name": "exact name", "engagement": 0-100, "sentiment": -1.0 to 1.0, '
                        '"clarity": 0-100, "rapport": 0-100, "energy": 0-100}\n'
                        "  ]\n"
                        "}\n\n"
                        "EVERY participant must appear with DIFFERENT scores based on ACTUAL behavior."
                    ),
                },
            ],
            max_completion_tokens=3000,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=90,
        required_keys=["participant_scores"],
        fallback=fallback,
        label="generate_flags_and_scores",
    )
```

**Step 3: Commit**

```bash
git add backend/core42_client.py
git commit -m "feat: separate summary, meeting notes, and flags into dedicated GPT calls"
```

---

## Task 5: Summary — DB Schema + API Endpoints

**Files:**
- Modify: `backend/models.py` (add `meeting_notes` table)
- Modify: `backend/schemas.py` (add `MeetingNotesOut` schema)
- Modify: `backend/main.py` (add meeting notes endpoint)

**Step 1: Add `meeting_notes` table to `models.py`**

Add after line 98 (after summaries table), before the `signal_snapshots` table:

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS meeting_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL UNIQUE,
        action_items TEXT DEFAULT '[]',
        decisions TEXT DEFAULT '[]',
        follow_ups TEXT DEFAULT '[]',
        key_questions TEXT DEFAULT '[]',
        FOREIGN KEY (video_id) REFERENCES videos(id)
    )
""")
```

Add index at end:

```python
await db.execute("CREATE INDEX IF NOT EXISTS idx_meeting_notes_video ON meeting_notes(video_id)")
```

**Step 2: Add schema to `schemas.py`**

Add after `SummaryOut` (after line 67):

```python
class MeetingNotesOut(BaseModel):
    action_items: list[dict] = []
    decisions: list[dict] = []
    follow_ups: list[dict] = []
    key_questions: list[dict] = []
```

Update the `AnalysisResults` class to include meeting_notes:

```python
class AnalysisResults(BaseModel):
    video: VideoOut
    participants: list[ParticipantOut]
    emotions: list[EmotionPoint]
    transcript: list[TranscriptSegment]
    flags: list[Flag]
    summary: Optional[SummaryOut] = None
    meeting_notes: Optional[MeetingNotesOut] = None
```

**Step 3: Add API endpoint in `main.py`**

Add after the `get_summary` endpoint (after line 887):

```python
@app.get("/api/videos/{video_id}/meeting-notes")
async def get_meeting_notes(video_id: str):
    """Get structured meeting notes (action items, decisions, follow-ups)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM meeting_notes WHERE video_id = ?", (video_id,)
        )
        r = await cursor.fetchone()
        if not r:
            return {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}
        return {
            "action_items": json.loads(r["action_items"]),
            "decisions": json.loads(r["decisions"]),
            "follow_ups": json.loads(r["follow_ups"]),
            "key_questions": json.loads(r["key_questions"]),
        }
```

Update `get_results` to include meeting_notes:

```python
@app.get("/api/videos/{video_id}/results")
async def get_results(video_id: str) -> AnalysisResults:
    """Get full analysis results for a video."""
    video = await get_video(video_id)
    participants = await get_participants(video_id)
    emotions = await get_emotions(video_id)
    transcript = await get_transcript(video_id)
    flags = await get_flags(video_id)
    try:
        summary = await get_summary(video_id)
    except HTTPException:
        summary = None
    notes = await get_meeting_notes(video_id)

    return AnalysisResults(
        video=video,
        participants=participants,
        emotions=emotions,
        transcript=transcript,
        flags=flags,
        summary=summary,
        meeting_notes=MeetingNotesOut(**notes) if notes else None,
    )
```

Add the import at top of main.py:

```python
from .schemas import (
    VideoOut, ParticipantOut, EmotionPoint, TranscriptSegment,
    Flag, CoachingItem, SummaryOut, AnalysisResults, ImportUrlRequest,
    MeetingAnalytics, RenameParticipantRequest, VideoStatus, MeetingNotesOut,
)
```

**Step 4: Commit**

```bash
git add backend/models.py backend/schemas.py backend/main.py
git commit -m "feat: add meeting_notes DB table, schema, and API endpoint"
```

---

## Task 6: Summary — Wire Pipeline to Use New Functions

**Files:**
- Modify: `backend/batch_pipeline.py:27-37` (update imports)
- Modify: `backend/batch_pipeline.py:907-926` (Phase 4 — use new functions)
- Modify: `backend/batch_pipeline.py:1059-1070` (store summary + notes)

**Step 1: Update imports in `batch_pipeline.py`**

Replace the import block (lines 27-37):

```python
from .core42_client import (
    analyze_frame_all_signals,
    analyze_voice_signal,
    analyze_words_signal,
    analyze_personality,
    build_correlations,
    generate_coaching,
    generate_meeting_summary,
    generate_meeting_notes,
    generate_flags_and_scores,
    extract_names_from_frame,
    assign_speakers_to_names,
)
```

**Step 2: Update Phase 4 to run 4-way concurrent gather**

Replace lines 907-914 (the 3-way gather):

```python
# Run personality + correlations + summary + notes + flags ALL concurrently (5-way gather)
await _update_progress(video_id, "analyzing", 70, "Running personality + correlations + summary + notes in parallel...")

personality_result, correlations_result, summary_result, notes_result, scores_result = await asyncio.gather(
    analyze_personality(signals_summary, transcript_text),
    build_correlations(all_signals_text, transcript_text),
    generate_meeting_summary(transcript_text),
    generate_meeting_notes(transcript_text),
    generate_flags_and_scores(transcript_text, emotions_summary, participant_names),
)

# Defensive: ensure all results are dicts
if not isinstance(personality_result, dict):
    logger.warning(f"[{video_id[:8]}] personality_result is {type(personality_result)}, using fallback")
    personality_result = {"participants": []}
if not isinstance(correlations_result, dict):
    logger.warning(f"[{video_id[:8]}] correlations_result is {type(correlations_result)}, using fallback")
    correlations_result = {"correlations": []}
if not isinstance(summary_result, dict):
    logger.warning(f"[{video_id[:8]}] summary_result is {type(summary_result)}, using fallback")
    summary_result = {"summary": "", "key_topics": [], "overall_sentiment": "neutral"}
if not isinstance(notes_result, dict):
    logger.warning(f"[{video_id[:8]}] notes_result is {type(notes_result)}, using fallback")
    notes_result = {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}
if not isinstance(scores_result, dict):
    logger.warning(f"[{video_id[:8]}] scores_result is {type(scores_result)}, using fallback")
    scores_result = {"flags": [], "participant_scores": []}
```

**Step 3: Update participant score references**

Where the old code used `analysis.get(...)`, change to `scores_result.get(...)`:

- Line ~957: `p_scores = analysis.get("participant_scores", [])` → `p_scores = scores_result.get("participant_scores", [])`

**Step 4: Update summary + flags storage**

Replace the summary storage block (around line 1059-1070):

```python
# --- Store summary (from dedicated call) ---
await db.execute(
    """INSERT OR REPLACE INTO summaries
       (video_id, summary, key_topics, overall_sentiment)
       VALUES (?, ?, ?, ?)""",
    (
        video_id,
        summary_result.get("summary", ""),
        json.dumps(summary_result.get("key_topics", [])),
        summary_result.get("overall_sentiment", "neutral"),
    ),
)

# --- Store meeting notes ---
await db.execute(
    """INSERT OR REPLACE INTO meeting_notes
       (video_id, action_items, decisions, follow_ups, key_questions)
       VALUES (?, ?, ?, ?, ?)""",
    (
        video_id,
        json.dumps(notes_result.get("action_items", [])),
        json.dumps(notes_result.get("decisions", [])),
        json.dumps(notes_result.get("follow_ups", [])),
        json.dumps(notes_result.get("key_questions", [])),
    ),
)

# --- Store flags (from scores call) ---
raw_flags = scores_result.get("flags", [])
```

Also update the `meeting_analytics` sentiment source:

```python
# Line ~1102 area
"overall_sentiment": summary_result.get("overall_sentiment", "neutral"),
```

Also add `"meeting_notes"` to the tables cleared at re-analysis (line ~936-940):

```python
for table in [
    "flags", "coaching", "summaries", "meeting_notes",
    "personality_signals", "correlations", "meeting_analytics",
]:
```

**Step 5: Commit**

```bash
git add backend/batch_pipeline.py
git commit -m "feat: wire pipeline to separate summary, notes, and flags GPT calls"
```

---

## Task 7: Voice Analysis — Per-Speaker Audio Features

**Files:**
- Modify: `backend/models.py` (add `speaker_audio_features` table)
- Modify: `backend/batch_pipeline.py:350-400` (per-speaker audio analysis)
- Modify: `backend/main.py` (new detailed audio endpoint)

**Step 1: Add `speaker_audio_features` table**

In `models.py`, add after the `voice_signals` table:

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS speaker_audio_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        pitch_mean REAL DEFAULT 0,
        pitch_std REAL DEFAULT 0,
        volume_energy REAL DEFAULT 0,
        speaking_rate REAL DEFAULT 0,
        pause_ratio REAL DEFAULT 0,
        spectral_centroid REAL DEFAULT 0,
        engagement_score REAL DEFAULT 0,
        FOREIGN KEY (video_id) REFERENCES videos(id)
    )
""")
```

Add index:

```python
await db.execute("CREATE INDEX IF NOT EXISTS idx_speaker_audio_video ON speaker_audio_features(video_id)")
```

**Step 2: Add per-speaker audio analysis in `batch_pipeline.py`**

Add a new function after `_analyze_audio_features` (after line 400):

```python
async def _analyze_per_speaker_audio(
    audio_path: str,
    transcript_segments: list[dict],
    participant_names: list[str],
    output_dir: str,
    video_id: str,
) -> dict[str, list[dict]]:
    """Extract audio features per speaker using transcript timestamps.

    Maps each speaker's transcript segments to audio chunks and runs librosa
    on those chunks. Falls back to global audio if no speaker-specific chunks.

    Returns {participant_name: [feature_dicts]}.
    """
    if not audio_path or not os.path.exists(audio_path):
        return {}

    analyzer = AudioAnalyzer()
    results: dict[str, list[dict]] = {name: [] for name in participant_names}
    name_lookup = _build_name_lookup(participant_names)

    # Group transcript segments by speaker
    speaker_segments: dict[str, list[dict]] = {}
    for seg in transcript_segments:
        spk = seg.get("speaker", "")
        matched = name_lookup.get(_normalize_name(spk)) or _fuzzy_match_name(spk, participant_names)
        if matched:
            if matched not in speaker_segments:
                speaker_segments[matched] = []
            speaker_segments[matched].append(seg)

    # Merge adjacent segments for each speaker (reduce ffmpeg calls)
    MERGE_GAP = 2.0  # Merge segments within 2 seconds of each other
    for name, segs in speaker_segments.items():
        if not segs:
            continue
        segs.sort(key=lambda s: s["start_time"])
        merged_ranges: list[tuple[float, float]] = []
        cur_start = segs[0]["start_time"]
        cur_end = segs[0]["end_time"]
        for seg in segs[1:]:
            if seg["start_time"] - cur_end <= MERGE_GAP:
                cur_end = max(cur_end, seg["end_time"])
            else:
                if cur_end - cur_start >= 1.0:  # Skip very short chunks
                    merged_ranges.append((cur_start, cur_end))
                cur_start = seg["start_time"]
                cur_end = seg["end_time"]
        if cur_end - cur_start >= 1.0:
            merged_ranges.append((cur_start, cur_end))

        # Analyze each merged range
        loop = asyncio.get_event_loop()
        for start_t, end_t in merged_ranges[:20]:  # Cap at 20 chunks per speaker
            try:
                chunk_path = await extract_audio_chunk(audio_path, start_t, end_t, output_dir)
                features = await loop.run_in_executor(None, analyzer.analyze_file, chunk_path)
                results[name].append({
                    "timestamp": start_t,
                    "pitch_mean": features.pitch_mean,
                    "pitch_std": features.pitch_std,
                    "volume_energy": features.volume_energy,
                    "speaking_rate": features.speaking_rate,
                    "pause_ratio": features.pause_ratio,
                    "spectral_centroid": features.spectral_centroid,
                    "engagement_score": features.engagement_score,
                })
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass
            except Exception as e:
                logger.warning(f"Per-speaker audio failed for {name} at {start_t:.1f}s: {e}")

    return results
```

**Step 3: Wire per-speaker audio into main pipeline**

In `run_analysis`, after the 3-way gather for frame/audio/text (around line 724-726), add per-speaker audio as a follow-up:

```python
# Per-speaker audio analysis (after global audio completes — needs transcript speakers resolved)
await _update_progress(video_id, "analyzing", 58, "Analyzing per-speaker audio...")
per_speaker_audio = await _analyze_per_speaker_audio(
    audio_path, transcript_segments, participant_names, output_dir, video_id
)
logger.info(f"[{video_id[:8]}] Per-speaker audio: {', '.join(f'{k}: {len(v)} chunks' for k, v in per_speaker_audio.items())}")
```

Save to DB after frame results (add after `_save_text_results_to_db`):

```python
# Save per-speaker audio features to DB
async with aiosqlite.connect(DB_PATH) as db:
    await db.execute("DELETE FROM speaker_audio_features WHERE video_id = ?", (video_id,))
    for name, features_list in per_speaker_audio.items():
        pid = participant_id_map.get(name, "unknown")
        for feat in features_list:
            await db.execute(
                """INSERT INTO speaker_audio_features
                   (video_id, participant_id, timestamp, pitch_mean, pitch_std,
                    volume_energy, speaking_rate, pause_ratio, spectral_centroid, engagement_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video_id, pid, feat["timestamp"],
                    feat["pitch_mean"], feat["pitch_std"],
                    feat["volume_energy"], feat["speaking_rate"],
                    feat["pause_ratio"], feat["spectral_centroid"],
                    feat["engagement_score"],
                ),
            )
    await db.commit()
```

**Step 4: Add API endpoint for per-speaker audio**

In `main.py`, add after the existing `get_audio_features` endpoint:

```python
@app.get("/api/videos/{video_id}/signals/speaker-audio")
async def get_speaker_audio_features(video_id: str):
    """Per-speaker audio features from librosa analysis."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT saf.*, p.name as participant_name
               FROM speaker_audio_features saf
               JOIN participants p ON saf.participant_id = p.id
               WHERE saf.video_id = ? ORDER BY p.name, saf.timestamp""",
            (video_id,),
        )
        rows = await cursor.fetchall()

    # Group by participant
    from collections import defaultdict
    by_participant = defaultdict(list)
    for r in rows:
        by_participant[r["participant_name"]].append({
            "timestamp": r["timestamp"],
            "pitch_mean": r["pitch_mean"],
            "pitch_std": r["pitch_std"],
            "volume_energy": r["volume_energy"],
            "speaking_rate": r["speaking_rate"],
            "pause_ratio": r["pause_ratio"],
            "spectral_centroid": r["spectral_centroid"],
            "engagement_score": r["engagement_score"],
        })

    return {
        "participants": [
            {"name": name, "features": feats}
            for name, feats in by_participant.items()
        ]
    }
```

**Step 5: Commit**

```bash
git add backend/models.py backend/batch_pipeline.py backend/main.py
git commit -m "feat: per-speaker librosa audio analysis with DB storage and API endpoint"
```

---

## Task 8: Voice Analysis — Frontend API + VoiceSignal Redesign

**Files:**
- Modify: `frontend/src/lib/api.ts` (add types + API functions)
- Modify: `frontend/src/components/signals/VoiceSignal.tsx` (full redesign)
- Modify: `frontend/src/components/AnalysisDashboard.tsx` (fetch + pass new data)

**Step 1: Add types and API functions to `api.ts`**

Add after the `AudioFeatureData` interface (after line 129):

```typescript
export interface SpeakerAudioData {
  participants: Array<{
    name: string;
    features: Array<{
      timestamp: number;
      pitch_mean: number;
      pitch_std: number;
      volume_energy: number;
      speaking_rate: number;
      pause_ratio: number;
      spectral_centroid: number;
      engagement_score: number;
    }>;
  }>;
}

export interface MeetingNotes {
  action_items: Array<{ owner: string; task: string; deadline: string | null }>;
  decisions: Array<{ decision: string; context: string }>;
  follow_ups: Array<{ topic: string; owner: string | null; timeline: string | null }>;
  key_questions: Array<{ question: string; raised_by: string | null }>;
}
```

Add API functions at end of file (before `formatTime`):

```typescript
export async function getSpeakerAudio(videoId: string): Promise<SpeakerAudioData | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/speaker-audio`);
  if (!res.ok) return null;
  return res.json();
}

export async function getMeetingNotes(videoId: string): Promise<MeetingNotes | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/meeting-notes`);
  if (!res.ok) return null;
  return res.json();
}
```

**Step 2: Redesign VoiceSignal component**

Replace entire `frontend/src/components/signals/VoiceSignal.tsx`:

```tsx
import { type VoiceSignalData, type SpeakerAudioData, formatTime } from '../../lib/api';
import { Mic, Activity, Gauge, Timer } from 'lucide-react';

interface Props {
  signals: VoiceSignalData[];
  speakerAudio: SpeakerAudioData | null;
}

export default function VoiceSignal({ signals, speakerAudio }: Props) {
  const hasRealAudio = speakerAudio && speakerAudio.participants.length > 0;
  const hasGptVoice = signals.length > 0;

  if (!hasRealAudio && !hasGptVoice) {
    return (
      <div className="p-4">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Mic className="w-3.5 h-3.5" /> Voice Analysis
        </h4>
        <p className="text-slate-600 text-xs">No voice data available</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
        <Mic className="w-3.5 h-3.5" /> Voice Analysis
      </h4>

      {/* Per-speaker audio metrics (from librosa) */}
      {hasRealAudio && (
        <div className="space-y-3">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider">Per-Speaker Audio (librosa)</p>
          {speakerAudio!.participants.map((p) => {
            if (p.features.length === 0) return null;
            const avgPitch = Math.round(p.features.reduce((s, f) => s + f.pitch_mean, 0) / p.features.length);
            const avgEnergy = Math.round(p.features.reduce((s, f) => s + f.volume_energy, 0) / p.features.length);
            const avgRate = (p.features.reduce((s, f) => s + f.speaking_rate, 0) / p.features.length).toFixed(1);
            const avgPause = Math.round(p.features.reduce((s, f) => s + f.pause_ratio, 0) / p.features.length * 100);
            const avgEngagement = Math.round(p.features.reduce((s, f) => s + f.engagement_score, 0) / p.features.length);

            return (
              <div key={p.name} className="p-3 rounded-lg bg-white/[0.03] border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-300">{p.name}</span>
                  <span className={`text-xs font-bold ${avgEngagement >= 60 ? 'text-emerald-400' : avgEngagement >= 40 ? 'text-amber-400' : 'text-red-400'}`}>
                    {avgEngagement}% energy
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  <div className="text-center">
                    <Activity className="w-3 h-3 text-violet-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Pitch</p>
                    <p className="text-xs font-bold text-slate-300">{avgPitch} Hz</p>
                  </div>
                  <div className="text-center">
                    <Gauge className="w-3 h-3 text-blue-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Volume</p>
                    <p className="text-xs font-bold text-slate-300">{avgEnergy}%</p>
                  </div>
                  <div className="text-center">
                    <Mic className="w-3 h-3 text-cyan-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Rate</p>
                    <p className="text-xs font-bold text-slate-300">{avgRate}/s</p>
                  </div>
                  <div className="text-center">
                    <Timer className="w-3 h-3 text-amber-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Pauses</p>
                    <p className="text-xs font-bold text-slate-300">{avgPause}%</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Energy timeline bar chart (from GPT voice signals or librosa) */}
      {hasGptVoice && (
        <div>
          <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Energy Timeline</p>
          <div className="flex items-end gap-px h-16">
            {signals.map((s, i) => (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-violet-500/40 to-violet-400/80 rounded-t-sm transition-all hover:from-violet-400/60 hover:to-violet-300"
                style={{ height: `${Math.max(4, s.energy)}%` }}
                title={`${formatTime(s.timestamp)} — ${s.tone}, ${s.pace} pace, energy: ${s.energy}%`}
              />
            ))}
          </div>
        </div>
      )}

      {/* GPT qualitative voice analysis (last 3) */}
      {hasGptVoice && (
        <div className="space-y-1.5">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider">Tone & Dynamics</p>
          {signals.slice(-3).map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="text-slate-500 font-mono w-10">{formatTime(s.timestamp)}</span>
              <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-300 text-[10px]">{s.tone}</span>
              <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-300 text-[10px]">{s.pace}</span>
              <span className="text-slate-400 flex-1 truncate text-[10px]">{s.dynamics}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

**Step 3: Update AnalysisDashboard to fetch and pass speaker audio**

In `AnalysisDashboard.tsx`, add import:

```typescript
import {
  type Video, type AnalysisResults, type Participant, type SignalSnapshot,
  type VoiceSignalData, type WordSignalData, type PersonalitySignalData,
  type CorrelationPattern, type SpeakerAudioData, type MeetingNotes,
  formatTime, getSignalSnapshots, getVoiceSignals,
  getWordSignals, getPersonalitySignals, getCorrelations, getSpeakerAudio, getMeetingNotes,
} from '../lib/api';
```

Add state:

```typescript
const [speakerAudio, setSpeakerAudio] = useState<SpeakerAudioData | null>(null);
const [meetingNotes, setMeetingNotes] = useState<MeetingNotes | null>(null);
```

Update the `Promise.all` in useEffect (lines 62-74):

```typescript
Promise.all([
  getSignalSnapshots(videoId),
  getVoiceSignals(videoId),
  getWordSignals(videoId),
  getPersonalitySignals(videoId),
  getCorrelations(videoId),
  getSpeakerAudio(videoId),
  getMeetingNotes(videoId),
]).then(([snap, voice, words, pers, corr, spkAudio, notes]) => {
  setSnapshots(snap);
  setVoiceSignals(voice);
  setWordSignals(words);
  setPersonality(pers);
  setCorrelations(corr);
  setSpeakerAudio(spkAudio);
  setMeetingNotes(notes);
});
```

Update VoiceSignal usage (line 596):

```tsx
<VoiceSignal signals={voiceSignals} speakerAudio={speakerAudio} />
```

**Step 4: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/components/signals/VoiceSignal.tsx frontend/src/components/AnalysisDashboard.tsx
git commit -m "feat: redesigned voice analysis with per-speaker librosa data + GPT overlay"
```

---

## Task 9: Meeting Notes — Frontend Display

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx` (add Meeting Notes section in Overview tab)

**Step 1: Add Meeting Notes section after Summary in Overview tab**

After the summary section closing `</section>` (around line 442), add:

```tsx
{/* Meeting Notes Section */}
{meetingNotes && (meetingNotes.action_items.length > 0 || meetingNotes.decisions.length > 0 || meetingNotes.follow_ups.length > 0) && (
  <section className="pb-6 border-b border-white/5">
    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
      <CheckCircle2 className="w-3.5 h-3.5" /> Meeting Notes
    </h4>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Action Items */}
      {meetingNotes.action_items.length > 0 && (
        <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
          <h5 className="text-xs font-semibold text-blue-400 mb-3">Action Items</h5>
          <div className="space-y-2">
            {meetingNotes.action_items.map((item, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5 text-xs">&#9679;</span>
                <div>
                  <p className="text-sm text-slate-200">{item.task}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {item.owner && <span className="text-blue-300">{item.owner}</span>}
                    {item.deadline && <span> &middot; {item.deadline}</span>}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Decisions */}
      {meetingNotes.decisions.length > 0 && (
        <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
          <h5 className="text-xs font-semibold text-emerald-400 mb-3">Decisions Made</h5>
          <div className="space-y-2">
            {meetingNotes.decisions.map((d, i) => (
              <div key={i} className="flex items-start gap-2">
                <CheckCircle2 className="w-3 h-3 text-emerald-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm text-slate-200">{d.decision}</p>
                  {d.context && <p className="text-[10px] text-slate-500 mt-0.5">{d.context}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-ups */}
      {meetingNotes.follow_ups.length > 0 && (
        <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/10">
          <h5 className="text-xs font-semibold text-amber-400 mb-3">Follow-ups</h5>
          <div className="space-y-2">
            {meetingNotes.follow_ups.map((f, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-amber-400 mt-0.5 text-xs">&#9679;</span>
                <div>
                  <p className="text-sm text-slate-200">{f.topic}</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {f.owner && <span className="text-amber-300">{f.owner}</span>}
                    {f.timeline && <span> &middot; {f.timeline}</span>}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Questions */}
      {meetingNotes.key_questions.length > 0 && (
        <div className="p-4 rounded-xl bg-purple-500/5 border border-purple-500/10">
          <h5 className="text-xs font-semibold text-purple-400 mb-3">Open Questions</h5>
          <div className="space-y-2">
            {meetingNotes.key_questions.map((q, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5 text-xs">?</span>
                <div>
                  <p className="text-sm text-slate-200">{q.question}</p>
                  {q.raised_by && <p className="text-[10px] text-slate-500 mt-0.5">Raised by: <span className="text-purple-300">{q.raised_by}</span></p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  </section>
)}
```

**Step 2: Commit**

```bash
git add frontend/src/components/AnalysisDashboard.tsx
git commit -m "feat: add Meeting Notes section with action items, decisions, follow-ups"
```

---

## Task 10: Signal Components — Full Timeline for Emotions & Body Language

**Files:**
- Modify: `frontend/src/components/signals/EmotionsSignal.tsx` (show aggregated emotion distribution)
- Modify: `frontend/src/components/signals/BodyLanguageSignal.tsx` (show posture trends)
- Modify: `frontend/src/components/signals/GesturesSignal.tsx` (aggregate across all snapshots)
- Modify: `frontend/src/components/signals/ReactionsSignal.tsx` (aggregate across all snapshots)

**Step 1: Redesign EmotionsSignal to show emotion distribution + latest**

Replace entire `EmotionsSignal.tsx`:

```tsx
import { type SignalSnapshot, getEmotionColor } from '../../lib/api';
import { Sparkles } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

export default function EmotionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Sparkles className="w-3.5 h-3.5" /> Emotions</h4><p className="text-slate-600 text-xs">No emotion data</p></div>;

  // Aggregate emotion distribution across ALL snapshots
  const emotionCounts: Record<string, number> = {};
  let totalEntries = 0;
  snapshots.forEach(s => s.participants.forEach(p => {
    const e = p.emotions.primary;
    emotionCounts[e] = (emotionCounts[e] || 0) + 1;
    totalEntries++;
  }));
  const sortedEmotions = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);

  const latest = snapshots[snapshots.length - 1];

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Sparkles className="w-3.5 h-3.5" /> Emotions
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>

      {/* Emotion distribution bars */}
      <div className="space-y-1.5 mb-4">
        {sortedEmotions.map(([emotion, count]) => {
          const pct = Math.round((count / totalEntries) * 100);
          return (
            <div key={emotion} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 w-20 capitalize">{emotion}</span>
              <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: getEmotionColor(emotion) }} />
              </div>
              <span className="text-[10px] font-bold text-slate-400 w-8 text-right">{pct}%</span>
            </div>
          );
        })}
      </div>

      {/* Latest snapshot */}
      <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Current</p>
      {latest.participants.map((p, i) => (
        <div key={i} className="flex items-center justify-between py-1">
          <span className="text-xs text-slate-300">{p.label}</span>
          <div className="flex items-center gap-2">
            <span className="text-xs capitalize" style={{ color: getEmotionColor(p.emotions.primary) }}>{p.emotions.primary}</span>
            <span className="text-[10px] text-slate-500">{Math.round(p.emotions.engagement)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}
```

**Step 2: Redesign GesturesSignal to aggregate across ALL snapshots**

Replace entire `GesturesSignal.tsx`:

```tsx
import { type SignalSnapshot } from '../../lib/api';
import { Hand } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

const USELESS = new Set(['still posture', 'not visible', 'no gesture', 'none', 'n/a', 'unknown']);

export default function GesturesSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Hand className="w-3.5 h-3.5" /> Gestures</h4><p className="text-slate-600 text-xs">No gesture data</p></div>;

  // Aggregate gesture frequency across ALL snapshots, per participant
  const participantGestures: Record<string, Record<string, number>> = {};
  snapshots.forEach(s => s.participants.forEach(p => {
    if (!participantGestures[p.label]) participantGestures[p.label] = {};
    (p.gestures || []).forEach(g => {
      const gl = g.toLowerCase().trim();
      if (!USELESS.has(gl)) {
        participantGestures[p.label][gl] = (participantGestures[p.label][gl] || 0) + 1;
      }
    });
  }));

  const hasData = Object.values(participantGestures).some(g => Object.keys(g).length > 0);

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Hand className="w-3.5 h-3.5" /> Gestures
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>
      {!hasData ? (
        <p className="text-slate-600 text-xs">No visible gesture data</p>
      ) : (
        <div className="space-y-3">
          {Object.entries(participantGestures).map(([name, gestures]) => {
            const sorted = Object.entries(gestures).sort((a, b) => b[1] - a[1]).slice(0, 5);
            if (sorted.length === 0) return null;
            return (
              <div key={name}>
                <p className="text-xs text-slate-300 mb-1.5">{name}</p>
                <div className="flex flex-wrap gap-1">
                  {sorted.map(([gesture, count]) => (
                    <span key={gesture} className="px-2 py-0.5 text-[10px] rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
                      {gesture} <span className="text-amber-500/60">x{count}</span>
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
```

**Step 3: Redesign ReactionsSignal to aggregate across ALL snapshots**

Replace entire `ReactionsSignal.tsx`:

```tsx
import { type SignalSnapshot } from '../../lib/api';
import { Zap } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

const USELESS = new Set(['not visible', 'none', 'no reaction', 'n/a', 'unknown', 'neutral']);

export default function ReactionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Zap className="w-3.5 h-3.5" /> Reactions</h4><p className="text-slate-600 text-xs">No reaction data</p></div>;

  // Aggregate reaction frequency across ALL snapshots, per participant
  const participantReactions: Record<string, Record<string, number>> = {};
  snapshots.forEach(s => s.participants.forEach(p => {
    if (!participantReactions[p.label]) participantReactions[p.label] = {};
    (p.reactions || []).forEach(r => {
      const rl = r.toLowerCase().trim();
      if (!USELESS.has(rl)) {
        participantReactions[p.label][rl] = (participantReactions[p.label][rl] || 0) + 1;
      }
    });
  }));

  const hasData = Object.values(participantReactions).some(r => Object.keys(r).length > 0);

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Zap className="w-3.5 h-3.5" /> Reactions
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>
      {!hasData ? (
        <p className="text-slate-600 text-xs">No reaction data</p>
      ) : (
        <div className="space-y-3">
          {Object.entries(participantReactions).map(([name, reactions]) => {
            const sorted = Object.entries(reactions).sort((a, b) => b[1] - a[1]).slice(0, 5);
            if (sorted.length === 0) return null;
            return (
              <div key={name}>
                <p className="text-xs text-slate-300 mb-1.5">{name}</p>
                <div className="flex flex-wrap gap-1">
                  {sorted.map(([reaction, count]) => (
                    <span key={reaction} className="px-2 py-0.5 text-[10px] rounded-full bg-rose-500/10 text-rose-300 border border-rose-500/20">
                      {reaction} <span className="text-rose-500/60">x{count}</span>
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
```

**Step 4: Commit**

```bash
git add frontend/src/components/signals/EmotionsSignal.tsx frontend/src/components/signals/GesturesSignal.tsx frontend/src/components/signals/ReactionsSignal.tsx
git commit -m "feat: signal components show full timeline aggregation instead of last snapshot only"
```

---

## Task 11: PDF Export — Add Meeting Notes Section

**Files:**
- Modify: `backend/export.py` (add meeting notes to PDF)
- Modify: `backend/export.py:234-290` (query meeting notes + pass to builder)

**Step 1: Query meeting notes in `generate_pdf_report`**

In `generate_pdf_report`, after the correlations query (around line 283), add:

```python
cursor = await db.execute(
    "SELECT * FROM meeting_notes WHERE video_id = ?", (video_id,)
)
meeting_notes = await cursor.fetchone()
```

Update the `_build_pdf` call to include meeting_notes:

```python
return _build_pdf(
    video, participants, summary, flags, transcript,
    coaching, correlations, chart_path, meeting_notes,
)
```

**Step 2: Update `_build_pdf` signature and add meeting notes rendering**

Update function signature:

```python
def _build_pdf(
    video, participants, summary, flags, transcript,
    coaching, correlations, chart_path, meeting_notes=None,
) -> bytes:
```

Add meeting notes section after the summary section (after the key topics rendering, around line 343):

```python
# Meeting Notes section
if meeting_notes:
    action_items = json.loads(meeting_notes["action_items"]) if meeting_notes["action_items"] else []
    decisions = json.loads(meeting_notes["decisions"]) if meeting_notes["decisions"] else []
    follow_ups = json.loads(meeting_notes["follow_ups"]) if meeting_notes["follow_ups"] else []

    if action_items or decisions or follow_ups:
        _section_heading(pdf, "Meeting Notes", DARK)

        if action_items:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*BLUE)
            pdf.cell(0, 6, "Action Items", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            for item in action_items:
                task = item.get("task", "")
                owner = item.get("owner", "")
                pdf.cell(6, 5, chr(8226))
                label = f"{task}"
                if owner:
                    label += f" ({owner})"
                pdf.multi_cell(0, 5, label)
            pdf.ln(2)

        if decisions:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*BLUE)
            pdf.cell(0, 6, "Decisions", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            for d in decisions:
                pdf.cell(6, 5, chr(8226))
                pdf.multi_cell(0, 5, d.get("decision", ""))
            pdf.ln(2)

        if follow_ups:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*BLUE)
            pdf.cell(0, 6, "Follow-ups", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            for f in follow_ups:
                pdf.cell(6, 5, chr(8226))
                pdf.multi_cell(0, 5, f.get("topic", ""))
            pdf.ln(2)

        pdf.ln(4)
```

**Step 3: Commit**

```bash
git add backend/export.py
git commit -m "feat: add meeting notes section to PDF export"
```

---

## Task 12: Delete Dead Code + Final Cleanup

**Files:**
- Delete: `frontend/src/components/signals/SignalPanel.tsx` (unused)
- Modify: `backend/batch_pipeline.py` (remove old `generate_summary_and_flags` references if any remain)

**Step 1: Remove unused SignalPanel.tsx**

```bash
rm frontend/src/components/signals/SignalPanel.tsx
```

Verify no imports reference it:

```bash
grep -r "SignalPanel" frontend/src/ --include="*.tsx" --include="*.ts"
```

**Step 2: Remove old import of `generate_summary_and_flags` from core42_client if still referenced**

Verify the old function is fully replaced. The renamed function `generate_flags_and_scores` should be the only one used.

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: remove dead code (unused SignalPanel), final cleanup"
```

---

## Task 13: Integration Test — Run Full Stack

**Step 1: Start backend**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

Verify: `http://localhost:8001/docs` loads the Swagger UI.

**Step 2: Start frontend**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline/frontend"
npm run dev
```

Verify: `http://localhost:5174` loads without errors.

**Step 3: Manual verification checklist**

- [ ] Upload a video → starts without crash
- [ ] Run transcription → completes
- [ ] Run analysis → completes with progress updates showing all phases
- [ ] Dashboard loads → KPI row shows scores
- [ ] Speaking distribution donut → sums to 100%
- [ ] Summary section → shows 3-4 paragraph summary
- [ ] Meeting Notes → shows action items, decisions, follow-ups
- [ ] Signals tab → Emotions shows distribution bars
- [ ] Signals tab → Voice shows per-speaker audio metrics
- [ ] Signals tab → Gestures shows aggregated counts
- [ ] Signals tab → Reactions shows aggregated counts
- [ ] PDF export → downloads without crash
- [ ] CSV export → downloads without crash

---

## Summary of All Changes

| Task | Area | Files Modified |
|------|------|---------------|
| 1 | PDF crash fix | `backend/export.py` |
| 2 | Speaking % backend | `backend/batch_pipeline.py` |
| 3 | Speaking % frontend | `frontend/src/components/AnalysisDashboard.tsx` |
| 4 | Summary GPT functions | `backend/core42_client.py` |
| 5 | Summary DB + API | `backend/models.py`, `backend/schemas.py`, `backend/main.py` |
| 6 | Summary pipeline wiring | `backend/batch_pipeline.py` |
| 7 | Per-speaker audio | `backend/models.py`, `backend/batch_pipeline.py`, `backend/main.py` |
| 8 | Voice frontend redesign | `frontend/src/lib/api.ts`, `frontend/src/components/signals/VoiceSignal.tsx`, `frontend/src/components/AnalysisDashboard.tsx` |
| 9 | Meeting notes frontend | `frontend/src/components/AnalysisDashboard.tsx` |
| 10 | Signal components timeline | `EmotionsSignal.tsx`, `GesturesSignal.tsx`, `ReactionsSignal.tsx` |
| 11 | PDF meeting notes | `backend/export.py` |
| 12 | Cleanup | `SignalPanel.tsx` (deleted) |
| 13 | Integration test | Manual verification |
