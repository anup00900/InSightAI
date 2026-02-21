# Critical 4-Bug Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 critical bugs + add pyannote speaker diarization: preloaded participants, broken speaking distribution, missing speaker names in transcript, incomplete PDF export, and proper audio-based speaker identification.

**Architecture:** Backend changes to frame analysis (dynamic participants, better visual detection), pyannote diarization integration for audio-based speaker detection, frontend reducer rewrite (merge-based participant tracking, proper speaking%), and PDF export port from offline template.

**Tech Stack:** Python/FastAPI backend, React/TypeScript frontend, pyannote.audio (speaker diarization), fpdf2+matplotlib for PDF

---

## Task 1: Backend — Dynamic Participants (only send who's visible in frame)

**Files:**
- Modify: `backend/realtime_pipeline.py:584-608`

**Step 1: Replace the "ALL canonical names every frame" logic with "only matched participants"**

In `_analyze_frame_live`, replace lines 584-608 (the entire "Build final list" section):

```python
                # Build final list: ONLY participants detected in THIS frame
                # (not all canonical names — participants appear dynamically)
                final_participants: list[dict] = []
                if matched:
                    final_participants = list(matched.values())
                else:
                    final_participants = raw_participants
```

This replaces the previous logic that iterated ALL canonical names and padded with carry-forward defaults. Now only participants the vision model actually detected in the current frame are sent.

**Step 2: Remove the per-participant carry-forward cache writes (they're no longer needed)**

Delete lines 610-612:
```python
                # Save per-participant data for carry-forward
                for p in final_participants:
                    self._last_participant_data[p["label"]] = p
```

And remove `self._last_participant_data` from `__init__` (line 135):
```python
        # Per-participant last known data — for stable carry-forward
        self._last_participant_data: dict[str, dict] = {}
```

**Step 3: Verify** — restart backend, play video. Participants should appear one by one as they become visible.

**Step 4: Commit**
```bash
git add backend/realtime_pipeline.py
git commit -m "fix: dynamic participants — only send who's visible in current frame"
```

---

## Task 2: Frontend — SIGNALS handler: merge instead of replace, stop wiping speaking_pct

**Files:**
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts:193-231`

**Step 1: Rewrite the SIGNALS full-path to MERGE into existing Map**

Replace lines 193-231 with:

```typescript
      // FULL PATH: real frame analysis — MERGE new data into existing participants
      // Only add NEW participants; update existing ones. Never wipe the Map.
      const newParticipants = new Map(state.participants);

      for (const p of snap.participants) {
        const participantId = state.nameMap[p.label] || p.label;
        if (newParticipants.has(participantId)) {
          // UPDATE existing participant — preserve speaking_pct from TRANSCRIPT handler
          const existing = newParticipants.get(participantId)!;
          existing.engagement_score = ema(existing.engagement_score || 0, p.emotions.engagement);

          const bl = p.body_language;
          const opennessScore = bl.openness === 'open' ? 85 : bl.openness === 'mixed' ? 55 : 25;
          const mirrorBonus = bl.mirroring ? 15 : 0;
          existing.rapport_score = ema(existing.rapport_score || 0, Math.min(100, opennessScore + mirrorBonus));

          const postureScore = bl.posture === 'upright' ? 80 : bl.posture === 'leaning' ? 60 : 40;
          existing.clarity_score = ema(existing.clarity_score || 0, Math.min(100, (p.emotions.confidence * 60) + (postureScore * 0.4)));

          const leanScore = bl.leaning === 'forward' ? 85 : bl.leaning === 'neutral' ? 60 : 35;
          existing.energy_score = ema(existing.energy_score || 0, Math.min(100, (p.emotions.engagement * 0.6) + (leanScore * 0.4)));
          // NOTE: Do NOT touch existing.speaking_pct — only TRANSCRIPT handler sets it
        } else {
          // NEW participant first appearance — add with initial values
          newParticipants.set(participantId, {
            id: participantId,
            video_id: '',
            name: participantId,
            engagement_score: p.emotions.engagement,
            sentiment_score: 0,
            speaking_pct: 0, // starts at 0, TRANSCRIPT handler will set real value
            clarity_score: 0,
            rapport_score: 0,
            energy_score: 0,
          });
        }
      }
```

Key changes:
- `new Map(state.participants)` (clone existing) instead of `new Map()` (fresh empty)
- Existing participants: update scores but **never touch speaking_pct**
- New participants: start at `speaking_pct: 0`

**Step 2: Commit**
```bash
git add frontend/src/hooks/useRealtimeAnalysis.ts
git commit -m "fix: SIGNALS handler merges participants, preserves speaking_pct"
```

---

## Task 3: Frontend — TRANSCRIPT handler: use ALL segments for speaking distribution

**Files:**
- Modify: `frontend/src/hooks/useRealtimeAnalysis.ts:275-329`

**Step 1: Change transcript cap from 100 to 500 (need more data for accurate distribution)**

Replace line 269:
```typescript
      if (state.transcript.length >= 100) {
        newTranscript = [...state.transcript.slice(-99), seg];
```
With:
```typescript
      if (state.transcript.length >= 500) {
        newTranscript = [...state.transcript.slice(-499), seg];
```

**Step 2: Use ALL transcript segments (not last 30) for speaking distribution**

Replace line 277:
```typescript
      const recentSegments = newTranscript.slice(-30);
```
With:
```typescript
      const recentSegments = newTranscript; // Use ALL segments for accurate distribution
```

**Step 3: Reset unmatched participants to 0% instead of leaving stale values**

After the matching loop (after line 320), add:

```typescript
        // Zero out participants that had no transcript match
        for (const [pid, pData] of newParticipants) {
          if (!matched.has(pid)) {
            pData.speaking_pct = 0;
          }
        }
```

**Step 4: Commit**
```bash
git add frontend/src/hooks/useRealtimeAnalysis.ts
git commit -m "fix: speaking distribution uses all transcript segments"
```

---

## Task 4: Backend — Upgrade vision detail to 'auto' for better speaker detection

**Files:**
- Modify: `backend/core42_client.py:449`

**Step 1: Change image detail from "low" to "auto"**

Replace line 449:
```python
                                "detail": "low",
```
With:
```python
                                "detail": "auto",
```

**Step 2: Commit**
```bash
git add backend/core42_client.py
git commit -m "fix: upgrade vision detail to auto for better speaker border detection"
```

---

## Task 5: Backend — Add relaxed fallback for speaker attribution

**Files:**
- Modify: `backend/core42_client.py:144-189`

**Step 1: Add a relaxed fallback after consensus voting fails**

Replace the entire `attribute_speaker_from_context` function (lines 144-189) with:

```python
async def attribute_speaker_from_context(
    _transcript_text: str,
    speaking_timeline: list[dict],
    _known_names: list[str],
    timestamp: float,
) -> str:
    """Attribute a transcript segment to the speaker detected visually.

    Two-tier approach:
    1. CONSENSUS VOTING within tight ±1.5s window (high confidence)
    2. FALLBACK: most recent speaker within ±3s (lower confidence, better than nothing)
    Returns empty string if no match at all.
    """
    if not speaking_timeline:
        return ""

    # Tier 1: Consensus voting within tight window
    TIGHT_WINDOW = 1.5
    votes: dict[str, float] = {}
    for entry in speaking_timeline:
        dist = abs(entry.get("timestamp", 0) - timestamp)
        if dist <= TIGHT_WINDOW:
            speaker = entry.get("speaker", "")
            if speaker:
                weight = 1.0 - (dist / TIGHT_WINDOW)
                votes[speaker] = votes.get(speaker, 0) + weight

    if votes:
        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_speaker, top_weight = ranked[0]
        total_weight = sum(w for _, w in ranked)

        # Confident match: enough weight and clear majority
        if top_weight >= 0.3 and (len(ranked) == 1 or (top_weight / total_weight) >= 0.6):
            return top_speaker

    # Tier 2: Relaxed fallback — most recent speaker within wider window
    WIDE_WINDOW = 3.0
    best_speaker = ""
    best_distance = float("inf")
    for entry in speaking_timeline:
        dist = abs(entry.get("timestamp", 0) - timestamp)
        if dist < best_distance and dist <= WIDE_WINDOW:
            best_distance = dist
            best_speaker = entry.get("speaker", "")

    return best_speaker
```

**Step 2: Commit**
```bash
git add backend/core42_client.py
git commit -m "fix: two-tier speaker attribution — consensus + relaxed fallback"
```

---

## Task 6: Backend — Integrate pyannote speaker diarization into realtime pipeline

**Files:**
- Modify: `backend/realtime_pipeline.py` (add diarization integration)
- Existing: `backend/pipelines/diarization.py` (already built, never wired in)

The diarization module exists at `backend/pipelines/diarization.py` with:
- `diarize_audio(audio_path)` → returns `[DiarizationSegment(start, end, speaker)]`
- `assign_speakers_to_transcript(segments, diarization)` → assigns "Speaker N" by timestamp overlap
- Uses pyannote/speaker-diarization-3.1 via HuggingFace, runs in thread pool

**How it works:**
1. pyannote analyzes the AUDIO WAVEFORM to detect distinct voices → "SPEAKER_00 spoke 10.5s-15.3s"
2. Transcript segments get matched to diarization by timestamp overlap → "this text was said by SPEAKER_00"
3. Cross-reference with visual `_speaking_timeline` → "SPEAKER_00 = Anup Roy" (because visual detection saw Anup speaking during the same time range)

**Step 1: Add import and state to RealtimeSession.__init__**

Add import at top of `realtime_pipeline.py`:
```python
from .pipelines.diarization import diarize_audio, assign_speakers_to_transcript, DiarizationSegment
```

Add state variables in `__init__`:
```python
        # Speaker diarization state (pyannote)
        self._diarization_segments: list = []  # DiarizationSegment list
        self._diarization_ready = asyncio.Event()
        self._diarization_speaker_map: dict[str, str] = {}  # "SPEAKER_00" -> "Anup Roy"
```

**Step 2: Run diarization after audio extraction completes**

In `_extract_audio_background()`, after audio extraction succeeds, kick off diarization:

```python
            if self._audio_path:
                self._audio_ready = True
                self._audio_extracted.set()
                logger.info(f"[{self.video_id}] Audio extraction complete")
                # Start speaker diarization in background
                self._create_tracked_task(self._run_diarization())
```

**Step 3: Add _run_diarization method**

```python
    async def _run_diarization(self):
        """Run pyannote speaker diarization on the full audio track.
        Maps diarization speaker labels to real names via visual cross-reference."""
        try:
            if not self._audio_path:
                self._diarization_ready.set()
                return

            logger.info(f"[{self.video_id}] Running speaker diarization...")
            await self._send_status("Identifying speakers...")

            segments = await diarize_audio(self._audio_path)
            if not segments:
                logger.warning(f"[{self.video_id}] Diarization returned no segments (pyannote may not be available)")
                self._diarization_ready.set()
                return

            self._diarization_segments = segments
            unique_speakers = sorted(set(s.speaker for s in segments))
            logger.info(f"[{self.video_id}] Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers")

            # Map diarization labels to real names using visual speaking timeline
            await self._map_diarization_to_names()
            self._diarization_ready.set()

        except Exception as e:
            logger.error(f"[{self.video_id}] Diarization failed: {e}")
            self._diarization_ready.set()
```

**Step 4: Add _map_diarization_to_names method**

```python
    async def _map_diarization_to_names(self):
        """Cross-reference diarization speaker labels with visual speaking detections.

        For each diarized speaker (SPEAKER_00, SPEAKER_01...), find which real name
        was visually detected as speaking during the same time ranges.
        """
        if not self._diarization_segments or not self._speaking_timeline:
            return

        # For each unique diarization speaker, collect their time ranges
        speaker_ranges: dict[str, list[tuple[float, float]]] = {}
        for seg in self._diarization_segments:
            speaker_ranges.setdefault(seg.speaker, []).append((seg.start, seg.end))

        # For each diarization speaker, find which visual name overlaps most
        for dia_speaker, ranges in speaker_ranges.items():
            name_votes: dict[str, float] = {}
            for start, end in ranges:
                for vt in self._speaking_timeline:
                    ts = vt.get("timestamp", 0)
                    if start <= ts <= end:
                        name = vt.get("speaker", "")
                        if name:
                            name_votes[name] = name_votes.get(name, 0) + 1

            if name_votes:
                best_name = max(name_votes, key=name_votes.get)
                self._diarization_speaker_map[dia_speaker] = best_name

        # Also build reverse: "Speaker 1" → real name (for generic labels)
        unique_speakers = sorted(set(s.speaker for s in self._diarization_segments))
        for i, sp in enumerate(unique_speakers):
            generic_label = f"Speaker {i + 1}"
            if sp in self._diarization_speaker_map:
                self._diarization_speaker_map[generic_label] = self._diarization_speaker_map[sp]

        logger.info(f"[{self.video_id}] Diarization speaker map: {self._diarization_speaker_map}")
```

**Step 5: Use diarization in _transcription_worker for speaker assignment**

In `_transcription_worker`, replace the visual-only attribution with a diarization-first approach.
After `valid_segments` are built, replace the speaker attribution block (lines ~765-774):

```python
                    for seg in valid_segments:
                        seg_ts = seg.get("start", 0) + start
                        seg["start"] = seg_ts
                        seg["end"] = seg.get("end", 0) + start

                        # PRIORITY 1: Audio diarization (most reliable)
                        if self._diarization_segments:
                            assigned = assign_speakers_to_transcript([seg], self._diarization_segments)
                            raw_speaker = assigned[0].get("speaker", "") if assigned else ""
                            # Map "Speaker N" or "SPEAKER_XX" to real name
                            if raw_speaker:
                                real_name = self._diarization_speaker_map.get(raw_speaker, "")
                                if real_name:
                                    seg["speaker"] = real_name
                                else:
                                    seg["speaker"] = raw_speaker  # Keep "Speaker 1" if no visual match

                        # PRIORITY 2: Visual speaker detection (fallback)
                        if not seg.get("speaker") and self._speaking_timeline and self._name_map:
                            speaker = await attribute_speaker_from_context(
                                seg.get("text", ""),
                                self._speaking_timeline,
                                list(self._name_map.values()),
                                seg_ts,
                            )
                            if speaker:
                                seg["speaker"] = speaker

                        self._transcript_segments.append(seg)
                        accepted += 1
```

**Step 6: Re-map diarization speakers periodically as more visual data arrives**

In `handle_tick`, after frame analysis, periodically re-run the name mapping:

```python
        # Re-map diarization speakers every 30s as visual data grows
        if (self._diarization_segments
                and self._speaking_timeline
                and timestamp - getattr(self, '_last_diarization_remap_ts', 0) >= 30.0):
            self._last_diarization_remap_ts = timestamp
            await self._map_diarization_to_names()
```

**Step 7: Commit**
```bash
git add backend/realtime_pipeline.py
git commit -m "feat: integrate pyannote speaker diarization — audio-based speaker identification"
```

---

## Task 7: Backend — Port offline PDF template to online

**Files:**
- Replace: `backend/export.py` (322 lines → ~900 lines from offline template)
- Reference: `../simpleem-offline/backend/export.py` (the complete template)

**Step 1: Copy the offline export.py and adapt imports/DB schema**

The offline version uses the same DB schema (same table names, same columns). Key adaptations needed:
- Import path: `from .models import DB_PATH` (same)
- The offline version has `meeting_notes` table — add a fallback if it doesn't exist in the online DB
- The offline version queries `coaching` table — same schema in online
- The offline version queries `personality_signals` — same schema in online
- The offline version queries `correlations` — same schema in online

Copy the offline file, then:
1. Keep the online `generate_csv_export` function (it already works)
2. Replace only the PDF-related functions (`generate_pdf_report`, `_build_pdf`, helpers)
3. Add fallback for `meeting_notes` table (may not exist in online DB)

**Step 2: Verify the PDF export works**

Run:
```bash
curl -o test_report.pdf http://localhost:8000/api/videos/<video_id>/export/pdf
open test_report.pdf
```

**Step 3: Commit**
```bash
git add backend/export.py
git commit -m "fix: port complete PDF template from offline — charts, coaching, personality, full transcript"
```

---

## Task 8: Restart and verify all fixes

**Step 1: Kill and restart backend**
```bash
lsof -ti :8000 | xargs kill -9; sleep 1
cd simpleem-online && python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
```

**Step 2: Verify in browser**
- Open http://localhost:5173
- Play video
- Check: participants appear dynamically (not all at once)
- Check: speaking distribution changes as transcript flows
- Check: speaker names appear in transcript (from diarization)
- Check: PDF export includes all sections
- Check backend logs for: "Diarization complete: X segments, Y speakers"

---

## Execution Order

| Task | Bug | Priority | What |
|------|-----|----------|------|
| 1 | Bug 1 (participants) | HIGH | Backend: remove carry-forward, dynamic only |
| 2 | Bug 2 (speaking%) | CRITICAL | Frontend: SIGNALS merge, preserve speaking_pct |
| 3 | Bug 2 (speaking%) | CRITICAL | Frontend: use all transcript segments |
| 4 | Bug 3 (speaker names) | HIGH | Backend: detail:low → detail:auto |
| 5 | Bug 3 (speaker names) | HIGH | Backend: two-tier speaker attribution fallback |
| 6 | Bug 3 (speaker names) | CRITICAL | Backend: pyannote diarization integration |
| 7 | Bug 4 (PDF) | HIGH | Backend: port offline PDF template |
| 8 | All | — | Restart & verify |

Tasks 1-5 can be done in parallel (independent files). Task 6 depends on Tasks 4-5. Task 7 is independent. Task 8 requires all others complete.
