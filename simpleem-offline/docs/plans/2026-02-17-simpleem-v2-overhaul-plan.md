# Simpleem Offline v2 Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul simpleem-offline for speed (3-4x), progressive results display, CEO-grade UI, enhanced exports, and improved analysis accuracy.

**Architecture:** Parallelize the 3 independent analysis phases (frames, audio, text) with asyncio.gather, save partial results to DB after each phase, add a partial-results API endpoint, redesign the frontend with executive-grade KPIs and progressive loading, enhance PDF/CSV exports with charts and more data.

**Tech Stack:** FastAPI, React 19, Tailwind CSS, Core42 API (GPT-4.1, GPT-4o, whisper-1), librosa, matplotlib, fpdf2

---

### Task 1: Parallelize Backend Analysis Pipeline

**Context:** The batch_pipeline.py currently runs phases sequentially: names → frames → audio → text → personality → correlations → summary → coaching. The 3 heavy phases (frames, audio, text) are independent and should run concurrently.

**Files:**
- Modify: `simpleem-offline/backend/batch_pipeline.py`
- Modify: `simpleem-offline/backend/transcription_engine.py`

**Changes to batch_pipeline.py `run_analysis()`:**

1. **Replace sequential Phase 2-4 with concurrent execution:**
```python
# ---- Phase 2: Run frame, audio, and text analysis IN PARALLEL ----
await _update_progress(video_id, "analyzing", 10, "Running visual, audio, and text analysis concurrently...")

audio_path = await extract_full_audio(video_path, output_dir)

# Launch all 3 independent analyses concurrently
frame_task = asyncio.create_task(
    _analyze_frames_batch(video_path, duration, output_dir, video_id)
)
audio_task = asyncio.create_task(
    _analyze_audio_features(audio_path, duration, output_dir, video_id)
)
text_task = asyncio.create_task(
    _analyze_text_signals(video_id, transcript_segments)
)

# Save partial results as each completes
done_tasks = set()
pending = {frame_task, audio_task, text_task}
task_names = {frame_task: "frames", audio_task: "audio", text_task: "text"}

while pending:
    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        name = task_names[task]
        done_tasks.add(name)
        result = task.result()
        # Store partial results immediately
        if name == "frames":
            frame_results = result
            await _save_frame_results(video_id, frame_results, name_mapping)
        elif name == "audio":
            audio_features = result
            await _save_audio_results(video_id, audio_features)
        elif name == "text":
            voice_signals, word_signals = result
            await _save_text_results(video_id, voice_signals, word_signals)

        pct = 10 + len(done_tasks) * 17  # 27%, 44%, 61%
        await _update_progress(video_id, "analyzing", pct, f"Completed {name} analysis ({len(done_tasks)}/3)...")
```

2. **Add helper functions to save partial results:**
```python
async def _save_frame_results(video_id, frame_results, name_mapping):
    """Save frame analysis results (emotions, signal_snapshots) to DB immediately."""
    # (move the emotions + signal_snapshots INSERT logic from the big block to here)

async def _save_audio_results(video_id, audio_features):
    """Save audio features — currently only used for engagement calculation, no DB table."""
    pass  # Audio features are kept in memory for engagement calc

async def _save_text_results(video_id, voice_signals, word_signals):
    """Save voice and word signals to DB immediately."""
    # (move the voice_signals + word_signals INSERT logic here)
```

3. **Increase FRAME_INTERVAL from 3.0 to 5.0** and **FRAME_BATCH_SIZE from 5 to 10:**
```python
FRAME_INTERVAL = 5.0      # Analyze a frame every 5 seconds (was 3)
FRAME_BATCH_SIZE = 10      # Parallel frame analysis batch size (was 5)
```

4. **Concurrent transcription chunks in transcription_engine.py:**
Change `_transcribe_via_api` to send up to 3 chunks concurrently:
```python
MAX_CONCURRENT_CHUNKS = 3

# Process chunks in groups of MAX_CONCURRENT_CHUNKS
for batch_start in range(0, len(chunk_starts), MAX_CONCURRENT_CHUNKS):
    batch = chunk_starts[batch_start:batch_start + MAX_CONCURRENT_CHUNKS]
    tasks = []
    for i_offset, start in enumerate(batch):
        end = min(start + CHUNK_DURATION_SECONDS, duration)
        chunk_path = await extract_audio_chunk(audio_path, start, end, output_dir)
        tasks.append((start, chunk_path, transcribe_audio(chunk_path)))

    results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
    for (start, chunk_path, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error(f"Chunk at {start}s failed: {result}")
        else:
            segments.extend(_parse_whisper_response(result, offset=start))
        try:
            os.remove(chunk_path)
        except OSError:
            pass
```

5. **Improve name extraction — sample first 5 seconds:**
```python
NAME_EXTRACT_COUNT = 5  # was 3

async def _extract_names(video_path, duration, output_dir):
    # Sample 2 frames from first 5 seconds (names most visible at start)
    sample_times = [1.0, 3.0]
    # Plus 3 samples from body of video
    for frac in [0.15, 0.35, 0.55]:
        t = duration * frac
        if t > 5.0:
            sample_times.append(t)
    # ... rest same
```

6. **Upgrade frame analysis image detail from "low" to "auto"** in core42_client.py:
```python
# In analyze_frame_all_signals:
"detail": "auto",  # was "low"
```

**Commit:** `perf: parallelize analysis pipeline, concurrent transcription, better name extraction`

---

### Task 2: Add Partial Results API Endpoint

**Context:** Frontend needs to show results progressively. Add an endpoint that returns whatever data is available so far.

**Files:**
- Modify: `simpleem-offline/backend/main.py`

**Add endpoint:**
```python
@app.get("/api/videos/{video_id}/partial-results")
async def get_partial_results(video_id: str):
    """Return whatever analysis data is available so far for progressive display."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Check video exists
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = await cursor.fetchone()
        if not video:
            raise HTTPException(404, "Video not found")

        ready_sections = []
        result = {"video": _build_video_out(video), "ready_sections": ready_sections}

        # Transcript
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM transcript_segments WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("transcript")
            cursor = await db.execute(
                "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time", (video_id,)
            )
            rows = await cursor.fetchall()
            result["transcript"] = [
                {"start_time": r["start_time"], "end_time": r["end_time"],
                 "text": r["text"], "speaker": r["speaker"]}
                for r in rows
            ]

        # Participants
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM participants WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("participants")
            cursor = await db.execute(
                "SELECT * FROM participants WHERE video_id = ?", (video_id,)
            )
            rows = await cursor.fetchall()
            result["participants"] = [dict(r) for r in rows]

        # Emotions (from frame analysis)
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM emotions WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("emotions")
            cursor = await db.execute(
                "SELECT * FROM emotions WHERE video_id = ? ORDER BY timestamp", (video_id,)
            )
            rows = await cursor.fetchall()
            result["emotions"] = [dict(r) for r in rows]

        # Voice signals
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM voice_signals WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("voice_signals")

        # Word signals
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM word_signals WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("word_signals")

        # Summary
        cursor = await db.execute(
            "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
        )
        row = await cursor.fetchone()
        if row:
            ready_sections.append("summary")
            result["summary"] = {
                "summary": row["summary"],
                "key_topics": json.loads(row["key_topics"]),
                "overall_sentiment": row["overall_sentiment"],
            }

        # Flags
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM flags WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("flags")
            cursor = await db.execute(
                "SELECT * FROM flags WHERE video_id = ? ORDER BY timestamp", (video_id,)
            )
            rows = await cursor.fetchall()
            result["flags"] = [dict(r) for r in rows]

        # Coaching
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM coaching WHERE video_id = ?", (video_id,)
        )
        cnt = (await cursor.fetchone())["cnt"]
        if cnt > 0:
            ready_sections.append("coaching")

        return result
```

**Also add to api.ts frontend:**
```typescript
export async function getPartialResults(videoId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/partial-results`);
  if (!res.ok) throw new Error('Partial results not available');
  return res.json();
}
```

**Commit:** `feat: add partial-results API endpoint for progressive display`

---

### Task 3: Redesign Empty State & Library View

**Context:** The library view with no videos shows a sad empty icon. With videos, the cards are basic. Redesign both for CEO-grade presentation.

**Files:**
- Modify: `simpleem-offline/frontend/src/components/VideoLibrary.tsx`
- Modify: `simpleem-offline/frontend/src/App.tsx`
- Modify: `simpleem-offline/frontend/src/index.css`

**Empty State redesign (VideoLibrary when videos.length === 0):**
```tsx
// Hero section with animated background
<div className="relative">
  {/* Animated gradient background */}
  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 via-purple-500/5 to-cyan-500/10 rounded-3xl" />
  <div className="relative text-center py-20 px-8">
    {/* Animated icon cluster */}
    <div className="flex items-center justify-center gap-4 mb-8">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-glow-purple animate-glow-pulse">
        <Brain className="w-8 h-8 text-white" />
      </div>
    </div>

    <h2 className="text-3xl font-bold gradient-text mb-3">
      AI-Powered Meeting Intelligence
    </h2>
    <p className="text-slate-400 text-lg max-w-xl mx-auto mb-10">
      Upload a recorded meeting to get instant insights on engagement,
      communication, and coaching opportunities.
    </p>

    {/* Feature cards */}
    <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto mb-10">
      {[
        { icon: BarChart3, title: "Engagement Analysis", desc: "Visual + Audio + Verbal scoring" },
        { icon: Users, title: "Participant Insights", desc: "Per-person coaching & personality" },
        { icon: Zap, title: "Real-time Results", desc: "See insights as they're generated" },
      ].map(f => (
        <div className="glass-card p-4 text-center">
          <f.icon className="w-6 h-6 text-purple-400 mx-auto mb-2" />
          <p className="text-sm font-semibold text-slate-200">{f.title}</p>
          <p className="text-xs text-slate-500 mt-1">{f.desc}</p>
        </div>
      ))}
    </div>

    {/* CTA */}
    <button onClick={onNewAnalysis} className="px-8 py-3.5 text-base font-semibold text-white rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-400 hover:to-purple-400 shadow-lg shadow-purple-500/25 transition-all hover:shadow-purple-500/40 hover:scale-105">
      Analyze Your First Meeting
    </button>
  </div>
</div>
```

**Library cards with thumbnails:**
- Add video thumbnail URL to Video type (backend extracts frame at 25%)
- Show mini engagement sparkline for completed videos
- Animated status badges
- Hover actions row

**Commit:** `ui: redesign empty state hero + enhanced library cards`

---

### Task 4: Redesign Dashboard — Executive KPI Header

**Context:** Dashboard needs a strong executive summary at the top with big KPI numbers before diving into details.

**Files:**
- Modify: `simpleem-offline/frontend/src/components/AnalysisDashboard.tsx`
- Modify: `simpleem-offline/frontend/src/components/ui/AnimatedCounter.tsx`

**Add KPI header row at top of dashboard:**
```tsx
{/* Executive KPI Row */}
<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
  {[
    { label: "Overall Score", value: vid.overall_score, suffix: "%", icon: TrendingUp, color: "purple" },
    { label: "Engagement", value: avgEngagement, suffix: "%", icon: Activity, color: "blue" },
    { label: "Sentiment", value: sentimentLabel, icon: Heart, color: sentimentColor },
    { label: "Participants", value: participants.length, icon: Users, color: "cyan" },
  ].map((kpi) => (
    <GlassCard key={kpi.label} className="text-center py-6">
      <kpi.icon className={`w-6 h-6 text-${kpi.color}-400 mx-auto mb-2`} />
      <div className="text-3xl font-black gradient-text">
        <AnimatedCounter target={kpi.value} />{kpi.suffix}
      </div>
      <p className="text-xs text-slate-500 mt-1">{kpi.label}</p>
    </GlassCard>
  ))}
</div>
```

**Also add progressive loading states:**
```tsx
{/* If section not ready yet, show skeleton */}
{!ready.includes('emotions') && (
  <div className="glass-card p-6 animate-pulse">
    <div className="h-4 bg-white/10 rounded w-1/3 mb-4" />
    <div className="h-32 bg-white/5 rounded" />
    <span className="text-xs text-purple-400 animate-glow-pulse">Analyzing video frames...</span>
  </div>
)}
```

**Commit:** `ui: executive KPI header + progressive loading skeletons`

---

### Task 5: Enhanced Export — PDF with Charts & Coaching

**Context:** PDF report is basic. Add engagement timeline chart, coaching recommendations, correlations, better branding.

**Files:**
- Modify: `simpleem-offline/backend/export.py`

**Add to PDF:**
1. **Engagement timeline chart** — use matplotlib to generate a PNG, embed in PDF:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile

def _generate_engagement_chart(emotions) -> bytes:
    """Generate engagement timeline chart as PNG bytes."""
    timestamps = [e["timestamp"] / 60 for e in emotions]  # minutes
    engagement = [e["engagement"] for e in emotions]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.fill_between(timestamps, engagement, alpha=0.3, color='#8b5cf6')
    ax.plot(timestamps, engagement, color='#8b5cf6', linewidth=2)
    ax.set_xlabel('Time (min)', fontsize=8, color='#64748b')
    ax.set_ylabel('Engagement', fontsize=8, color='#64748b')
    ax.set_ylim(0, 100)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#64748b', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(buf.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf.name
```

2. **Coaching section in PDF:**
```python
# After Participants table
_section_heading(pdf, "Coaching Recommendations", DARK)
for p in participants:
    coaching = await db.execute(
        "SELECT * FROM coaching WHERE video_id = ? AND participant_id = ? ORDER BY priority",
        (video_id, p["id"]),
    )
    items = await coaching.fetchall()
    if items:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, p["name"], new_x="LMARGIN", new_y="NEXT")
        for item in items:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, f"  • {item['recommendation']}")
```

3. **Correlations section in PDF**

4. **Add coaching.csv, correlations.csv, personality.csv, signal_snapshots.csv to CSV export**

**Commit:** `feat: enhanced PDF with charts + coaching, expanded CSV export`

---

### Task 6: Video Thumbnail Extraction & Backend Polish

**Context:** Library cards should show video thumbnails. Backend needs a thumbnail extraction endpoint.

**Files:**
- Modify: `simpleem-offline/backend/main.py`
- Modify: `simpleem-offline/backend/frame_extractor.py`

**Add thumbnail endpoint:**
```python
@app.get("/api/videos/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Return a JPEG thumbnail from 25% into the video."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT file_path, duration FROM videos WHERE id = ?", (video_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Video not found")

    file_path, duration = row
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(404, "Video file not found")

    ts = max(1.0, (duration or 10) * 0.25)
    thumb_dir = os.path.join(PROCESSED_DIR, video_id)
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, "thumb.jpg")

    if not os.path.exists(thumb_path):
        await extract_single_frame(file_path, ts, thumb_dir, output_name="thumb.jpg")

    return FileResponse(thumb_path, media_type="image/jpeg")
```

**Commit:** `feat: video thumbnail extraction endpoint`

---

### Task 7: Frontend Dashboard — Progressive Results & Polish

**Context:** Wire up progressive results display. Dashboard should poll /partial-results and show sections as they become available during analysis.

**Files:**
- Modify: `simpleem-offline/frontend/src/components/AnalysisDashboard.tsx`
- Modify: `simpleem-offline/frontend/src/components/wizard/AnalysisStep.tsx`
- Modify: `simpleem-offline/frontend/src/lib/api.ts`

**AnalysisStep.tsx — show partial results while analyzing:**
- Poll `/api/videos/{id}/partial-results` every 2 seconds while analyzing
- When transcript is ready, show transcript preview
- When emotions are ready, show emotion timeline preview
- When summary is ready, show summary preview
- All with subtle "Still analyzing..." indicator

**AnalysisDashboard.tsx — skeleton loading per section:**
- Check `ready_sections` array
- Show content for ready sections, skeletons for pending
- Auto-poll while video status is not "complete"
- Smooth fade-in transitions as sections appear

**Commit:** `feat: progressive results display during analysis`

---

### Task 8: Final UI Polish & Testing

**Context:** Final pass to ensure everything works end-to-end, fix any visual issues, test the full pipeline.

**Files:**
- All frontend components
- Backend tests

**Tasks:**
1. Test full pipeline: upload → transcribe → analyze → dashboard
2. Verify parallel analysis actually saves time
3. Verify progressive results display works
4. Test PDF export with charts
5. Test CSV export with new files
6. Fix any visual regressions
7. Ensure TypeScript compiles clean
8. Ensure Python imports clean

**Commit:** `chore: final polish and integration testing`
