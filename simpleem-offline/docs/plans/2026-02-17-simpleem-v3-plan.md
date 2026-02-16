# Simpleem Offline V3 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Production-grade meeting intelligence — accurate per-participant analysis, full signal panels, professional analytics UI.

**Architecture:** FastAPI + aiosqlite backend, React 19 + Vite + Tailwind frontend. Core42 API (GPT-4o, GPT-4.1, Whisper-1). Librosa for real audio features.

**Tech Stack:** Python 3.13, FastAPI, aiosqlite, librosa, React 19, Tailwind CSS, Recharts (for charts), Lucide React icons.

---

### Task 1: Signal API Endpoints

**Goal:** Add 6 new API endpoints for signal panel data.

**Files:**
- Modify: `backend/main.py`

**Implementation:**

Add these endpoints after the existing `/api/videos/{id}/partial-results` endpoint:

```python
@app.get("/api/videos/{video_id}/signals/snapshots")
async def get_signal_snapshots(video_id: str):
    """Frame-by-frame visual signals — emotions, body language, gestures, reactions per participant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT timestamp, participant_id, emotion, emotion_confidence, engagement,
                      posture, openness, leaning, mirroring, body_language_desc,
                      gestures, reactions
               FROM signal_snapshots WHERE video_id = ?
               ORDER BY timestamp""",
            (video_id,),
        )
        rows = await cursor.fetchall()

    # Group by timestamp into SignalSnapshot format
    from collections import defaultdict
    by_ts = defaultdict(list)
    # Get participant names
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        p_cursor = await db.execute("SELECT id, name FROM participants WHERE video_id = ?", (video_id,))
        p_rows = await p_cursor.fetchall()
    pid_to_name = {r["id"]: r["name"] for r in p_rows}

    for r in rows:
        by_ts[r["timestamp"]].append({
            "label": pid_to_name.get(r["participant_id"], r["participant_id"]),
            "emotions": {
                "primary": r["emotion"],
                "confidence": r["emotion_confidence"],
                "engagement": r["engagement"],
            },
            "body_language": {
                "posture": r["posture"],
                "openness": r["openness"],
                "leaning": r["leaning"],
                "mirroring": bool(r["mirroring"]),
                "description": r["body_language_desc"] or "",
            },
            "gestures": json.loads(r["gestures"]) if r["gestures"] else [],
            "reactions": json.loads(r["reactions"]) if r["reactions"] else [],
        })

    return [{"timestamp": ts, "participants": parts} for ts, parts in sorted(by_ts.items())]


@app.get("/api/videos/{video_id}/signals/voice")
async def get_voice_signals(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, tone, pace, energy, dynamics FROM voice_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        return [dict(r) for r in await cursor.fetchall()]


@app.get("/api/videos/{video_id}/signals/words")
async def get_word_signals(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, sentiment, sentiment_score, keywords, key_phrases FROM word_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [{
        "timestamp": r["timestamp"],
        "sentiment": r["sentiment"],
        "sentiment_score": r["sentiment_score"],
        "keywords": json.loads(r["keywords"]) if r["keywords"] else [],
        "key_phrases": json.loads(r["key_phrases"]) if r["key_phrases"] else [],
    } for r in rows]


@app.get("/api/videos/{video_id}/signals/personality")
async def get_personality_signals(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT ps.*, p.name FROM personality_signals ps JOIN participants p ON ps.participant_id = p.id WHERE ps.video_id = ?",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return {
        "participants": [{
            "label": r["name"],
            "traits": json.loads(r["traits"]) if r["traits"] else [],
            "communication_style": r["communication_style"] or "",
            "confidence_level": r["confidence_level"] or "",
        } for r in rows]
    }


@app.get("/api/videos/{video_id}/signals/correlations")
async def get_correlations(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT signal_pair, pattern, confidence, implication FROM correlations WHERE video_id = ?",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [{
        "signal_pair": json.loads(r["signal_pair"]) if r["signal_pair"] else [],
        "pattern": r["pattern"],
        "confidence": r["confidence"],
        "implication": r["implication"],
    } for r in rows]


@app.get("/api/videos/{video_id}/signals/audio-features")
async def get_audio_features(video_id: str):
    """Return stored audio features from the pre_analysis_cache or computed during analysis."""
    # Audio features are stored per-chunk in the analysis pipeline
    # Return voice_signals as audio feature proxy + any librosa data
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, tone, pace, energy, dynamics FROM voice_signals WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]
```

**Verify:** `python3.13 -m py_compile backend/main.py`

---

### Task 2: Better Frame Analysis Prompts

**Goal:** Calibrate engagement scoring with reference anchors for accuracy.

**Files:**
- Modify: `backend/core42_client.py` — `analyze_frame_all_signals` function

**Implementation:**

In the system prompt of `analyze_frame_all_signals`, add calibration anchors to the engagement instructions:

Add after "emotions, body language, gestures, and micro-reactions.":
```
ENGAGEMENT CALIBRATION (use these anchors):
- 85-100: Actively engaged — nodding, smiling, leaning forward, animated gestures, steady eye contact
- 65-84: Attentive — upright posture, looking at camera/screen, occasional nods
- 45-64: Passive attention — still, minimal expression, neutral posture, not obviously disengaged
- 25-44: Low engagement — looking away, slouched, minimal interaction, distracted
- 0-24: Disengaged — looking at phone, turned away, clearly not paying attention
- Camera-OFF participants: ALWAYS engagement=30, emotion='passive'
```

Also increase `max_tokens` from 2000 to 2500.

**Verify:** `python3.13 -m py_compile backend/core42_client.py`

---

### Task 3: Engagement Alerts Integration

**Goal:** Wire EngagementAlertDetector into the pipeline and save alerts as flags.

**Files:**
- Modify: `backend/batch_pipeline.py`

**Implementation:**

In `run_analysis`, after the engagement score computation loop (Phase 3), add:

```python
# Run engagement alert detection
from .engagement_engine import EngagementAlertDetector
alert_detector = EngagementAlertDetector(threshold=40, min_duration=30)
engagement_alerts = []

# Re-process engagement scores chronologically for alert detection
all_scores_chronological = []
for fr in frame_results:
    ts = fr.get("timestamp", 0)
    for p in fr.get("participants", []):
        label = p.get("label", "Person 1")
        name = name_lookup.get(_normalize_name(label)) or _fuzzy_match_name(label, participant_names)
        if name and name in participant_engagement:
            scores = participant_engagement[name]
            if scores:
                from .engagement_engine import EngagementScore
                score = EngagementScore(
                    participant_id=name, timestamp=ts,
                    holistic=scores[-1] if scores else 50,
                    visual=p.get("emotions", {}).get("engagement", 50),
                    audio=0, verbal=0,
                )
                alert = alert_detector.check(score)
                if alert:
                    engagement_alerts.append(alert)
```

Then in Phase 6 where flags are stored, add engagement alerts as flags:

```python
# Add engagement alerts as flags
for alert in engagement_alerts:
    await db.execute(
        """INSERT INTO flags (video_id, timestamp, type, description, severity)
           VALUES (?, ?, ?, ?, ?)""",
        (video_id, alert.timestamp, "engagement_alert",
         alert.suggested_action, "high"),
    )
```

**Verify:** `python3.13 -m py_compile backend/batch_pipeline.py`

---

### Task 4: Parallelization — 3-way gather for Phase 4+5

**Goal:** Run personality, correlations, AND summary/flags concurrently.

**Files:**
- Modify: `backend/batch_pipeline.py`

**Implementation:**

Replace the sequential Phase 4 + Phase 5 with a single concurrent gather:

```python
# ---- Phase 4+5: Personality + correlations + summary/flags CONCURRENT ----
await _update_progress(video_id, "analyzing", 65, "Generating personality, correlations, and summary...")

personality_result, correlations_result, analysis = await asyncio.gather(
    analyze_personality(signals_summary, transcript_text[:3000]),
    build_correlations(all_signals_text, transcript_text[:2000]),
    generate_summary_and_flags(transcript_text[:8000], emotions_summary[:4000], participant_names),
)
```

Remove the separate Phase 5 section since it's now merged into Phase 4.

**Verify:** `python3.13 -m py_compile backend/batch_pipeline.py`

---

### Task 5: Dashboard — Per-Channel Scores + Engagement Timeline + Speaking Chart

**Goal:** Add per-channel score breakdown, engagement timeline chart, speaking distribution, and engagement alerts to the dashboard.

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx`
- Modify: `frontend/src/lib/api.ts` — add new API functions

**Implementation:**

In `api.ts`, add functions to fetch signal data:

```typescript
export async function getSignalSnapshots(videoId: string): Promise<SignalSnapshot[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/snapshots`);
  if (!res.ok) return [];
  return res.json();
}

export async function getVoiceSignals(videoId: string): Promise<VoiceSignalData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/voice`);
  if (!res.ok) return [];
  return res.json();
}

export async function getWordSignals(videoId: string): Promise<WordSignalData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/words`);
  if (!res.ok) return [];
  return res.json();
}

export async function getPersonalitySignals(videoId: string): Promise<PersonalitySignalData | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/personality`);
  if (!res.ok) return null;
  return res.json();
}

export async function getCorrelations(videoId: string): Promise<CorrelationPattern[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/correlations`);
  if (!res.ok) return [];
  return res.json();
}

export async function getAudioFeatures(videoId: string): Promise<AudioFeatureData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/audio-features`);
  if (!res.ok) return [];
  return res.json();
}
```

In `AnalysisDashboard.tsx`:
- Add engagement timeline chart using inline SVG (no extra dependency) showing engagement points over time
- Add per-channel score badges under the Overall Score KPI card
- Add speaking distribution horizontal bars
- Add engagement alerts section showing any flags with type="engagement_alert"

**Verify:** `cd frontend && npx tsc --noEmit`

---

### Task 6: Wire Signal Panels into Dashboard

**Goal:** Connect all 7 signal panels to API data in the dashboard.

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx`

**Implementation:**

In the dashboard, after the tabs section, add a new "Signals" tab or section that renders all 7 signal panels in a responsive grid:

```tsx
// Add to imports
import EmotionsSignal from './signals/EmotionsSignal';
import BodyLanguageSignal from './signals/BodyLanguageSignal';
import GesturesSignal from './signals/GesturesSignal';
import ReactionsSignal from './signals/ReactionsSignal';
import VoiceSignal from './signals/VoiceSignal';
import WordsSignal from './signals/WordsSignal';
import PersonalitySignal from './signals/PersonalitySignal';
import CorrelationPanel from './signals/CorrelationPanel';

// Fetch signal data on mount
const [snapshots, setSnapshots] = useState<SignalSnapshot[]>([]);
const [voiceSignals, setVoiceSignals] = useState<VoiceSignalData[]>([]);
const [wordSignals, setWordSignals] = useState<WordSignalData[]>([]);
const [personality, setPersonality] = useState<PersonalitySignalData | null>(null);
const [correlations, setCorrelations] = useState<CorrelationPattern[]>([]);

useEffect(() => {
  if (!results) return;
  Promise.all([
    getSignalSnapshots(videoId),
    getVoiceSignals(videoId),
    getWordSignals(videoId),
    getPersonalitySignals(videoId),
    getCorrelations(videoId),
  ]).then(([snap, voice, words, pers, corr]) => {
    setSnapshots(snap);
    setVoiceSignals(voice);
    setWordSignals(words);
    setPersonality(pers);
    setCorrelations(corr);
  });
}, [results, videoId]);
```

Add "Signals" tab and render panels in a responsive grid.

**Verify:** `cd frontend && npx tsc --noEmit`

---

### Task 7: Landing Page Stats + Sparklines

**Goal:** Add aggregate stats bar and engagement sparklines to the library view.

**Files:**
- Modify: `frontend/src/components/VideoLibrary.tsx`
- Modify: `frontend/src/lib/api.ts`

**Implementation:**

Add a stats bar above the video grid when videos exist:
- Total meetings analyzed (videos.length)
- Average engagement (from video.overall_score)
- Total participants (sum from analytics or estimate)

Add a mini SVG sparkline in each video card showing overall score visually.

**Verify:** `cd frontend && npx tsc --noEmit`

---

### Task 8: Final Verification

**Goal:** Verify all TypeScript + Python compiles, test manually.

**Files:** All modified files

**Steps:**
1. `cd frontend && npx tsc --noEmit`
2. `python3.13 -m py_compile backend/main.py`
3. `python3.13 -m py_compile backend/batch_pipeline.py`
4. `python3.13 -m py_compile backend/core42_client.py`
5. Start both servers, verify the app loads
6. Upload a video and verify the full pipeline works
