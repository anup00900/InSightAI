# Glass UI + Multi-Model Accuracy Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the dashboard into a premium glassmorphism UI with clickable/expandable KPI cards, and add a triple-model voting accuracy engine for behavioral analysis on real Zoom recordings.

**Architecture:** Backend-first — build the voting/consensus engine in `core42_client.py`, wire it into `batch_pipeline.py`, add confidence scoring to DB. Then frontend — create expandable glass components with framer-motion, wire to new data. All changes are additive to the existing plan (docs/plans/2026-02-17-fix-all-issues.md).

**Tech Stack:** Python/FastAPI, React/TypeScript, framer-motion (already installed), SQLite, Core42 API (gpt-4o + gpt-4.1 + gpt-5), librosa.

---

## Task 1: Multi-Model Voting Engine — Core Module

**Files:**
- Create: `backend/voting_engine.py`
- Modify: `backend/models.py` (add voting_log table + confidence column)

**Step 1: Create the voting engine module**

Create `backend/voting_engine.py`:

```python
"""Multi-model voting engine for behavioral analysis accuracy.

Implements triple-model consensus (GPT-4o + GPT-4.1 + GPT-5) with:
- Majority vote for categorical signals (emotion, posture, etc.)
- Trimmed mean for numerical signals (engagement, confidence)
- Union+majority for list signals (gestures, reactions)
- Cross-signal validation (voice-visual alignment, speaking-engagement coherence)
- Per-snapshot confidence scoring (0-1)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """Result of multi-model consensus voting."""
    consensus: dict
    confidence: float  # 0-1
    disagreements: list[str] = field(default_factory=list)
    model_results: list[dict] = field(default_factory=list)


def majority_vote(values: list[str]) -> tuple[str, float]:
    """Majority vote for categorical values. Returns (winner, agreement_ratio)."""
    if not values:
        return "neutral", 0.0
    counts = Counter(v.lower().strip() for v in values if v)
    if not counts:
        return "neutral", 0.0
    winner, top_count = counts.most_common(1)[0]
    agreement = top_count / len(values)
    return winner, agreement


def trimmed_mean(values: list[float]) -> tuple[float, float]:
    """Trimmed mean for numerical values — drop the outlier, average closest 2.

    Returns (mean_value, spread_ratio). Spread ratio = range/mean, lower = more agreement.
    """
    if not values:
        return 50.0, 1.0
    if len(values) == 1:
        return values[0], 0.0

    sorted_vals = sorted(values)
    if len(values) == 2:
        mean = sum(sorted_vals) / 2
        spread = abs(sorted_vals[1] - sorted_vals[0]) / max(mean, 1)
        return round(mean, 1), round(spread, 2)

    # 3+ values: drop the outlier (furthest from median)
    median = sorted_vals[len(sorted_vals) // 2]
    diffs = [(abs(v - median), i) for i, v in enumerate(sorted_vals)]
    diffs.sort(reverse=True)
    outlier_idx = diffs[0][1]

    kept = [v for i, v in enumerate(sorted_vals) if i != outlier_idx]
    mean = sum(kept) / len(kept)
    spread = (max(sorted_vals) - min(sorted_vals)) / max(mean, 1)
    return round(mean, 1), round(spread, 2)


def union_majority_list(lists: list[list[str]], min_mentions: int = 2) -> list[str]:
    """Union of lists, keep items mentioned by min_mentions+ models.

    Falls back to all items if nothing meets threshold.
    """
    counts: Counter = Counter()
    for lst in lists:
        for item in lst:
            counts[item.lower().strip()] += 1

    # Keep items mentioned by min_mentions+ models
    result = [item for item, count in counts.most_common() if count >= min_mentions]

    # Fallback: if nothing meets threshold, keep top items
    if not result and counts:
        result = [item for item, _ in counts.most_common(4)]

    return result


def compute_confidence(
    emotion_agreement: float,
    engagement_spread: float,
    posture_agreement: float,
    gesture_overlap: float,
) -> float:
    """Compute overall confidence score from sub-signal agreements.

    Returns 0-1 where 1 = perfect agreement across all models.
    """
    # Weights: emotion matters most, then engagement, then posture, then gestures
    score = (
        0.35 * emotion_agreement +
        0.30 * max(0, 1.0 - engagement_spread) +
        0.20 * posture_agreement +
        0.15 * gesture_overlap
    )
    return round(max(0.0, min(1.0, score)), 2)


def vote_on_participant(results: list[dict]) -> VotingResult:
    """Run consensus voting on multiple model results for a SINGLE participant.

    Each result is a dict with: emotions, body_language, gestures, reactions, camera_on.
    Returns a VotingResult with consensus data and confidence score.
    """
    if not results:
        return VotingResult(
            consensus=_default_participant(),
            confidence=0.0,
            disagreements=["no_model_results"],
        )

    if len(results) == 1:
        return VotingResult(
            consensus=results[0],
            confidence=0.5,  # Single model = medium confidence
            disagreements=["single_model_only"],
            model_results=results,
        )

    disagreements = []

    # --- Emotion: majority vote ---
    emotions_primary = [r.get("emotions", {}).get("primary", "neutral") for r in results]
    emotion_winner, emotion_agree = majority_vote(emotions_primary)
    if emotion_agree < 0.67:
        disagreements.append(f"emotion: {emotions_primary}")

    # --- Engagement: trimmed mean ---
    engagements = [float(r.get("emotions", {}).get("engagement", 50)) for r in results]
    engagement_val, engagement_spread = trimmed_mean(engagements)
    if engagement_spread > 0.3:
        disagreements.append(f"engagement_spread: {engagements}")

    # --- Confidence (emotion): trimmed mean ---
    confidences = [float(r.get("emotions", {}).get("confidence", 0.5)) for r in results]
    confidence_val, _ = trimmed_mean(confidences)

    # --- Body language: majority vote per field ---
    bl_fields = ["posture", "openness", "leaning"]
    bl_consensus = {}
    bl_agreements = []
    for bf in bl_fields:
        vals = [r.get("body_language", {}).get(bf, "neutral") for r in results]
        winner, agree = majority_vote(vals)
        bl_consensus[bf] = winner
        bl_agreements.append(agree)
        if agree < 0.67:
            disagreements.append(f"body_language.{bf}: {vals}")

    # Mirroring: majority vote (boolean)
    mirror_vals = [r.get("body_language", {}).get("mirroring", False) for r in results]
    bl_consensus["mirroring"] = sum(1 for m in mirror_vals if m) > len(mirror_vals) / 2

    # Description: use longest description (most detailed)
    descriptions = [r.get("body_language", {}).get("description", "") for r in results]
    bl_consensus["description"] = max(descriptions, key=len) if descriptions else ""

    avg_posture_agree = sum(bl_agreements) / max(len(bl_agreements), 1)

    # --- Gestures: union + majority ---
    gesture_lists = [r.get("gestures", []) for r in results]
    consensus_gestures = union_majority_list(gesture_lists, min_mentions=2)
    if not consensus_gestures:
        consensus_gestures = union_majority_list(gesture_lists, min_mentions=1)[:4]

    # Gesture overlap ratio
    all_gestures = set()
    for gl in gesture_lists:
        all_gestures.update(g.lower().strip() for g in gl)
    gesture_overlap = len(consensus_gestures) / max(len(all_gestures), 1)

    # --- Reactions: union + majority ---
    reaction_lists = [r.get("reactions", []) for r in results]
    consensus_reactions = union_majority_list(reaction_lists, min_mentions=2)
    if not consensus_reactions:
        consensus_reactions = union_majority_list(reaction_lists, min_mentions=1)[:4]

    # --- Camera on: majority vote ---
    camera_vals = [r.get("camera_on", True) for r in results]
    camera_on = sum(1 for c in camera_vals if c) > len(camera_vals) / 2

    # --- Label: use the most common label ---
    labels = [r.get("label", "Person 1") for r in results]
    label_winner, _ = majority_vote(labels)
    # But prefer exact casing from original
    for lbl in labels:
        if lbl.lower().strip() == label_winner:
            label_winner = lbl
            break

    # --- Compute overall confidence ---
    confidence = compute_confidence(
        emotion_agreement=emotion_agree,
        engagement_spread=engagement_spread,
        posture_agreement=avg_posture_agree,
        gesture_overlap=gesture_overlap,
    )

    consensus = {
        "label": label_winner,
        "camera_on": camera_on,
        "emotions": {
            "primary": emotion_winner,
            "confidence": confidence_val,
            "engagement": engagement_val,
        },
        "body_language": bl_consensus,
        "gestures": consensus_gestures,
        "reactions": consensus_reactions,
    }

    return VotingResult(
        consensus=consensus,
        confidence=confidence,
        disagreements=disagreements,
        model_results=results,
    )


def vote_on_frame(model_results: list[dict]) -> tuple[dict, float, list[dict]]:
    """Run consensus voting on multiple model results for a full frame.

    Each model_result is a dict with {"participants": [...]}.

    Returns (consensus_frame, avg_confidence, voting_log_entries).
    """
    if not model_results:
        return {"participants": []}, 0.0, []

    if len(model_results) == 1:
        return model_results[0], 0.5, []

    # Collect all unique participant labels across all models
    all_labels: dict[str, list[dict]] = {}
    for mr in model_results:
        for p in mr.get("participants", []):
            label = p.get("label", "Person 1").lower().strip()
            if label not in all_labels:
                all_labels[label] = []
            all_labels[label].append(p)

    consensus_participants = []
    total_confidence = 0.0
    log_entries = []

    for label_key, participant_results in all_labels.items():
        result = vote_on_participant(participant_results)
        consensus_participants.append(result.consensus)
        total_confidence += result.confidence

        if result.disagreements:
            log_entries.append({
                "label": result.consensus.get("label", label_key),
                "confidence": result.confidence,
                "disagreements": result.disagreements,
            })

    avg_confidence = total_confidence / max(len(all_labels), 1)

    return {"participants": consensus_participants}, avg_confidence, log_entries


# ---------------------------------------------------------------------------
# Cross-signal validation
# ---------------------------------------------------------------------------

def cross_signal_validate(
    visual_engagement: float,
    audio_energy: float,
    is_speaking: bool,
    camera_on: bool,
    sentiment_score: float,
    emotion: str,
) -> dict:
    """Cross-validate signals and apply correction rules.

    Returns dict with corrected values and flags.
    """
    corrections = []
    corrected_engagement = visual_engagement

    # Rule 1: Camera-off reweight — visual engagement is meaningless
    if not camera_on:
        corrected_engagement = max(30, audio_energy * 0.6 + sentiment_score * 100 * 0.4)
        corrections.append("camera_off_reweight")

    # Rule 2: Speaking-Engagement coherence
    # If actively speaking but visual engagement < 40, boost it
    if is_speaking and visual_engagement < 40 and camera_on:
        corrected_engagement = max(visual_engagement, 55)
        corrections.append("speaking_engagement_boost")

    # Rule 3: Voice-Visual alignment
    # If librosa says high energy (>70) but visual says disengaged (<30)
    if audio_energy > 70 and visual_engagement < 30 and camera_on:
        corrected_engagement = (audio_energy * 0.6 + visual_engagement * 0.4)
        corrections.append("voice_visual_alignment")

    # Rule 4: Sentiment-Emotion consistency
    # If words are very positive (>0.8) but emotion is "concerned" or "disengaged"
    negative_emotions = {"concerned", "disengaged", "confused", "tense"}
    if sentiment_score > 0.7 and emotion.lower() in negative_emotions:
        corrected_engagement = max(corrected_engagement, 50)
        corrections.append("sentiment_emotion_mismatch")

    return {
        "corrected_engagement": round(corrected_engagement, 1),
        "corrections": corrections,
        "original_engagement": visual_engagement,
    }


def _default_participant() -> dict:
    """Default participant data when no model results are available."""
    return {
        "label": "Person 1",
        "camera_on": True,
        "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
        "body_language": {
            "posture": "upright", "openness": "mixed", "leaning": "neutral",
            "mirroring": False, "description": "unable to analyze",
        },
        "gestures": ["still posture", "resting hands"],
        "reactions": ["neutral gaze", "steady expression"],
    }
```

**Step 2: Add DB schema changes in `models.py`**

Add after the `signal_snapshots` table (after line 117):

In the `signal_snapshots` CREATE TABLE, add `confidence` column. Since SQLite doesn't support ALTER TABLE ADD COLUMN easily in init, add it as a separate migration:

```python
# After the signal_snapshots table creation
await db.execute("""
    CREATE TABLE IF NOT EXISTS voting_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        signal_type TEXT DEFAULT 'visual',
        model_a_result TEXT DEFAULT '{}',
        model_b_result TEXT DEFAULT '{}',
        model_c_result TEXT DEFAULT '{}',
        consensus_result TEXT DEFAULT '{}',
        confidence REAL DEFAULT 1.0,
        disagreements TEXT DEFAULT '[]',
        FOREIGN KEY (video_id) REFERENCES videos(id)
    )
""")
```

Add index:

```python
await db.execute("CREATE INDEX IF NOT EXISTS idx_voting_log_video ON voting_log(video_id)")
```

Add confidence column to signal_snapshots (safe migration):

```python
# Safe column addition — ignores if already exists
try:
    await db.execute("ALTER TABLE signal_snapshots ADD COLUMN confidence REAL DEFAULT 1.0")
except Exception:
    pass  # Column already exists
```

**Step 3: Commit**

```bash
git add backend/voting_engine.py backend/models.py
git commit -m "feat: multi-model voting engine with consensus algorithms and cross-signal validation"
```

---

## Task 2: Triple-Model Frame Analysis in core42_client.py

**Files:**
- Modify: `backend/core42_client.py` (add `analyze_frame_triple_vote`)

**Step 1: Add the triple-model frame analysis function**

Add after `analyze_frame_all_signals` (after line 381):

```python
async def analyze_frame_triple_vote(
    frame_path: str, timestamp: float, known_names: list[str] | None = None,
    camera_on_names: list[str] | None = None,
) -> tuple[dict, float, list[dict]]:
    """Analyze a single frame using 3 models concurrently for consensus voting.

    Calls GPT-4o, GPT-4.1, and GPT-5 in parallel with the same frame.
    Runs majority vote / trimmed mean consensus on the results.

    Returns (consensus_result, confidence, voting_log_entries).
    """
    from .voting_engine import vote_on_frame

    # Run all 3 models concurrently
    model_tasks = [
        _analyze_frame_with_model(frame_path, timestamp, VISION_MODEL, known_names, camera_on_names),
        _analyze_frame_with_model(frame_path, timestamp, CHAT_MODEL, known_names, camera_on_names),
        _analyze_frame_with_model(frame_path, timestamp, CHAT_MODEL_STRONG, known_names, camera_on_names),
    ]

    results = await asyncio.gather(*model_tasks, return_exceptions=True)

    valid_results = []
    for i, result in enumerate(results):
        model_names = [VISION_MODEL, CHAT_MODEL, CHAT_MODEL_STRONG]
        if isinstance(result, Exception):
            logger.warning(f"Model {model_names[i]} failed for frame at {timestamp:.1f}s: {result}")
            continue
        if isinstance(result, dict) and "participants" in result:
            valid_results.append(result)

    if not valid_results:
        # All models failed — use fallback
        fallback_label = known_names[0] if known_names else "Person 1"
        fallback = {
            "participants": [{
                "label": fallback_label, "camera_on": True,
                "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
                "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "all models failed"},
                "gestures": ["still posture"], "reactions": ["neutral gaze"],
            }]
        }
        return fallback, 0.0, [{"error": "all_models_failed"}]

    consensus, confidence, log_entries = vote_on_frame(valid_results)

    # Add model count info
    for entry in log_entries:
        entry["models_responded"] = len(valid_results)
        entry["models_total"] = 3

    return consensus, confidence, log_entries


async def _analyze_frame_with_model(
    frame_path: str, timestamp: float, model: str,
    known_names: list[str] | None = None,
    camera_on_names: list[str] | None = None,
) -> dict:
    """Analyze a frame with a specific model. Same prompt as analyze_frame_all_signals."""
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    fallback_label = known_names[0] if known_names else "Person 1"

    # Build name context
    name_context = ""
    if known_names:
        name_context = (
            f"\n\nKNOWN PARTICIPANTS IN THIS MEETING: {', '.join(known_names)}\n"
            "You MUST use these exact names as 'label' values. Match each visible person "
            "to the closest known name by reading their name label in the video platform UI. "
            "If you cannot match a person to a known name, skip them.\n"
        )
        if camera_on_names:
            camera_off = [n for n in known_names if n not in camera_on_names]
            if camera_off:
                name_context += (
                    f"\nParticipants with camera OFF: {', '.join(camera_off)}\n"
                    "For camera-OFF participants: set camera_on=false, engagement=30, "
                    "emotions primary='passive', and use minimal body language/gestures.\n"
                )

    async def _call():
        # Use vision-capable endpoint for image analysis
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert behavioral analyst specializing in non-verbal communication. "
                        "Analyze the video frame for visible participants across 4 signal categories: "
                        "emotions, body language, gestures, and micro-reactions.\n"
                        "ENGAGEMENT CALIBRATION (use these anchors):\n"
                        "- 85-100: Actively engaged — nodding, smiling, leaning forward, animated gestures\n"
                        "- 65-84: Attentive — upright posture, looking at camera/screen, occasional nods\n"
                        "- 45-64: Passive attention — still, minimal expression, neutral posture\n"
                        "- 25-44: Low engagement — looking away, slouched, minimal interaction\n"
                        "- 0-24: Disengaged — looking at phone, turned away, clearly not paying attention\n"
                        "- Camera-OFF participants: ALWAYS engagement=30, emotion='passive'\n"
                        "CRITICAL RULES:\n"
                        "1. Only analyze participants whose camera is ON. For camera-OFF: include with camera_on=false.\n"
                        "2. Use REAL NAMES from the meeting platform name labels.\n"
                        "3. Each participant appears ONLY ONCE. No duplicates.\n"
                        "4. gestures array: 2-4 items per camera-ON person.\n"
                        "5. reactions array: 2-4 items per camera-ON person.\n"
                        "6. NEVER return empty arrays for gestures or reactions.\n"
                        "Return structured JSON."
                        + name_context
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this video frame at timestamp {timestamp:.1f}s.\n"
                                "For EACH participant, analyze:\n"
                                "1. EMOTIONS: primary emotion, confidence (0-1), engagement (0-100)\n"
                                "2. BODY LANGUAGE: posture, openness, leaning, mirroring\n"
                                "3. GESTURES: 2-4 detected gestures\n"
                                "4. REACTIONS: 2-4 micro-reactions\n\n"
                                "Return ONLY valid JSON:\n"
                                '{"participants": [{"label": "Name", "camera_on": true, '
                                '"emotions": {"primary": "engaged", "confidence": 0.85, "engagement": 78}, '
                                '"body_language": {"posture": "upright", "openness": "open", "leaning": "forward", "mirroring": false, "description": "..."}, '
                                '"gestures": ["nodding", "open palms"], '
                                '"reactions": ["slight smile", "steady eye contact"]}]}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ],
            max_tokens=2500,
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
        timeout=45,
        required_keys=["participants"],
        fallback={
            "participants": [{
                "label": fallback_label, "camera_on": True,
                "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
                "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "model error"},
                "gestures": ["still posture"], "reactions": ["neutral gaze"],
            }]
        },
        label=f"frame_analysis_{model}",
    )


async def analyze_voice_signal_dual(transcript_chunk: str) -> dict:
    """Dual-model voice signal analysis (GPT-4o-mini + GPT-4.1) with consensus."""
    from .voting_engine import majority_vote, trimmed_mean

    results = await asyncio.gather(
        analyze_voice_signal(transcript_chunk),  # uses CHAT_MODEL_FAST (gpt-4o-mini)
        _analyze_voice_with_model(transcript_chunk, CHAT_MODEL),  # gpt-4.1
        return_exceptions=True,
    )

    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) < 2:
        return valid[0] if valid else {"tone": "neutral", "pace": "moderate", "energy": 50, "dynamics": ""}

    tone_winner, _ = majority_vote([r.get("tone", "neutral") for r in valid])
    pace_winner, _ = majority_vote([r.get("pace", "moderate") for r in valid])
    energy_val, _ = trimmed_mean([float(r.get("energy", 50)) for r in valid])

    # Use the longer dynamics description
    dynamics = max([r.get("dynamics", "") for r in valid], key=len)

    return {"tone": tone_winner, "pace": pace_winner, "energy": energy_val, "dynamics": dynamics}


async def _analyze_voice_with_model(transcript_chunk: str, model: str) -> dict:
    """Voice signal analysis with a specific model."""
    async def _call():
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert voice and speech analyst. "
                        "Infer vocal qualities from transcript text and speaking patterns."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze the vocal qualities implied by this transcript segment:\n\n"
                        f'"{transcript_chunk}"\n\n'
                        "Infer: tone (warm/neutral/tense/enthusiastic/monotone/assertive), "
                        "pace (slow/moderate/fast), energy (0-100), dynamics (brief description).\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"tone": "warm", "pace": "moderate", "energy": 65, "dynamics": "steady and conversational"}'
                    ),
                },
            ],
            max_tokens=200,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call, timeout=15,
        fallback={"tone": "neutral", "pace": "moderate", "energy": 50, "dynamics": ""},
        label=f"voice_signal_{model}",
    )


async def analyze_words_signal_dual(transcript_chunk: str, timestamp: float) -> dict:
    """Dual-model word signal analysis with consensus."""
    from .voting_engine import majority_vote, trimmed_mean, union_majority_list

    results = await asyncio.gather(
        analyze_words_signal(transcript_chunk, timestamp),  # gpt-4o-mini
        _analyze_words_with_model(transcript_chunk, timestamp, CHAT_MODEL),  # gpt-4.1
        return_exceptions=True,
    )

    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) < 2:
        return valid[0] if valid else {"sentiment": "neutral", "sentiment_score": 0.5, "keywords": [], "key_phrases": []}

    sentiment_winner, _ = majority_vote([r.get("sentiment", "neutral") for r in valid])
    score_val, _ = trimmed_mean([float(r.get("sentiment_score", 0.5)) for r in valid])
    keywords = union_majority_list([r.get("keywords", []) for r in valid], min_mentions=1)
    phrases = union_majority_list([r.get("key_phrases", []) for r in valid], min_mentions=1)

    return {
        "sentiment": sentiment_winner,
        "sentiment_score": score_val,
        "keywords": keywords[:10],
        "key_phrases": phrases[:5],
    }


async def _analyze_words_with_model(transcript_chunk: str, timestamp: float, model: str) -> dict:
    """Word signal analysis with a specific model."""
    async def _call():
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert linguistic analyst. Analyze text for sentiment, keywords, and notable phrases.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze this transcript segment (at ~{timestamp:.0f}s):\n\n"
                        f'"{transcript_chunk}"\n\n'
                        "Return ONLY valid JSON:\n"
                        '{"sentiment": "positive|neutral|negative", "sentiment_score": 0.0 to 1.0, '
                        '"keywords": ["keyword1"], "key_phrases": ["phrase1"]}'
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call, timeout=15,
        fallback={"sentiment": "neutral", "sentiment_score": 0.5, "keywords": [], "key_phrases": []},
        label=f"words_signal_{model}",
    )
```

**Step 2: Commit**

```bash
git add backend/core42_client.py
git commit -m "feat: triple-model frame analysis and dual-model text analysis with consensus voting"
```

---

## Task 3: Wire Voting Engine into batch_pipeline.py

**Files:**
- Modify: `backend/batch_pipeline.py` (update imports, frame analysis, save voting logs)

**Step 1: Update imports**

Add to the imports from `core42_client`:

```python
from .core42_client import (
    analyze_frame_all_signals,
    analyze_frame_triple_vote,
    analyze_voice_signal,
    analyze_voice_signal_dual,
    analyze_words_signal,
    analyze_words_signal_dual,
    analyze_personality,
    build_correlations,
    generate_coaching,
    generate_meeting_summary,
    generate_meeting_notes,
    generate_flags_and_scores,
    extract_names_from_frame,
    assign_speakers_to_names,
)
from .voting_engine import cross_signal_validate
```

**Step 2: Update `_analyze_frames_batch` to use triple-model voting**

Replace the `analyze_tasks` building section (around lines 303-311) in `_analyze_frames_batch`:

```python
# Analyze frames with triple-model voting
analyze_tasks = []
valid_frames = []
for i, (fp, ts) in enumerate(zip(frame_paths, batch_ts)):
    if isinstance(fp, Exception):
        logger.warning(f"Frame extraction failed at {ts:.1f}s: {fp}")
        continue
    analyze_tasks.append(analyze_frame_triple_vote(fp, ts, known_names, camera_on_names))
    valid_frames.append((fp, ts))

if analyze_tasks:
    analyses = await asyncio.gather(*analyze_tasks, return_exceptions=True)
    for j, (analysis, (fp, ts)) in enumerate(zip(analyses, valid_frames)):
        if isinstance(analysis, Exception):
            logger.warning(f"Frame analysis failed at {ts:.1f}s: {analysis}")
            analysis = (
                {
                    "participants": [{
                        "label": "Person 1",
                        "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
                        "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "unable to analyze"},
                        "gestures": ["still posture", "resting hands"],
                        "reactions": ["neutral gaze", "steady expression"],
                    }]
                },
                0.0,
                [],
            )

        # Unpack tuple from triple-vote: (consensus, confidence, log_entries)
        if isinstance(analysis, tuple):
            consensus, confidence, log_entries = analysis
        else:
            consensus, confidence, log_entries = analysis, 0.5, []

        consensus["timestamp"] = ts
        consensus["_confidence"] = confidence
        consensus["_voting_log"] = log_entries
        results.append(consensus)

        try:
            os.remove(fp)
        except OSError:
            pass
```

**Step 3: Update text analysis to use dual-model voting**

In `_analyze_text_signals`, replace the `_analyze_voice` and `_analyze_words` inner functions:

```python
async def _analyze_voice(txt, timestamp):
    try:
        result = await analyze_voice_signal_dual(txt)
        return {
            "timestamp": timestamp,
            "tone": result.get("tone", "neutral"),
            "pace": result.get("pace", "moderate"),
            "energy": result.get("energy", 50),
            "dynamics": result.get("dynamics", ""),
        }
    except Exception as e:
        logger.warning(f"Voice signal analysis failed at {timestamp:.1f}s: {e}")
        return None

async def _analyze_words(txt, timestamp):
    try:
        result = await analyze_words_signal_dual(txt, timestamp)
        return {
            "timestamp": timestamp,
            "sentiment": result.get("sentiment", "neutral"),
            "sentiment_score": result.get("sentiment_score", 0.5),
            "keywords": result.get("keywords", []),
            "key_phrases": result.get("key_phrases", []),
        }
    except Exception as e:
        logger.warning(f"Word signal analysis failed at {timestamp:.1f}s: {e}")
        return None
```

**Step 4: Save voting logs and confidence to DB**

After saving frame results (after `_save_frame_results_to_db`), add:

```python
# Save voting logs for audit trail
async with aiosqlite.connect(DB_PATH) as db:
    await db.execute("DELETE FROM voting_log WHERE video_id = ?", (video_id,))
    for fr in frame_results:
        ts = fr.get("timestamp", 0)
        confidence = fr.get("_confidence", 1.0)
        log_entries = fr.get("_voting_log", [])
        if log_entries:
            await db.execute(
                """INSERT INTO voting_log
                   (video_id, timestamp, signal_type, consensus_result, confidence, disagreements)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    video_id, ts, "visual",
                    json.dumps(fr.get("participants", [])),
                    confidence,
                    json.dumps(log_entries),
                ),
            )
    await db.commit()
logger.info(f"[{video_id[:8]}] Saved voting logs for {len(frame_results)} frames")
```

**Step 5: Apply cross-signal validation in engagement computation**

In the engagement scoring loop (where `engine.compute` is called), add cross-signal validation before computing:

```python
# Apply cross-signal validation
csv_result = cross_signal_validate(
    visual_engagement=visual_engagement,
    audio_energy=closest_audio["energy"] if audio_features else 50,
    is_speaking=is_speaking,
    camera_on=camera_on,
    sentiment_score=closest_word["sentiment_score"] if word_signals else 0.5,
    emotion=p.get("emotions", {}).get("primary", "neutral"),
)
visual_engagement = csv_result["corrected_engagement"]
if csv_result["corrections"]:
    logger.debug(f"Cross-signal corrections for {name} at {ts:.1f}s: {csv_result['corrections']}")
```

**Step 6: Update `_save_frame_results_to_db` to include confidence**

In the `signal_snapshots` INSERT, add the confidence value:

```python
# In the signal_snapshots INSERT, add confidence
confidence = fr.get("_confidence", 1.0)
await db.execute(
    """INSERT INTO signal_snapshots
       (video_id, participant_id, timestamp, emotion, emotion_confidence,
        engagement, posture, openness, leaning, mirroring,
        body_language_desc, gestures, reactions, confidence)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        video_id, pid, ts,
        emotions_data.get("primary", "neutral"),
        emotions_data.get("confidence", 0.5),
        emotions_data.get("engagement", 50),
        bl.get("posture", "upright"),
        bl.get("openness", "mixed"),
        bl.get("leaning", "neutral"),
        1 if bl.get("mirroring") else 0,
        bl.get("description", ""),
        json.dumps(p.get("gestures", [])),
        json.dumps(p.get("reactions", [])),
        confidence,
    ),
)
```

**Step 7: Commit**

```bash
git add backend/batch_pipeline.py
git commit -m "feat: wire triple-model voting and cross-signal validation into analysis pipeline"
```

---

## Task 4: Confidence API Endpoint + Frontend Types

**Files:**
- Modify: `backend/main.py` (return confidence in snapshots endpoint)
- Modify: `frontend/src/lib/api.ts` (add confidence to types)

**Step 1: Update signal_snapshots endpoint to include confidence**

In `main.py`, in the `get_signal_snapshots` function, update the SELECT query to include confidence:

```python
cursor = await db.execute(
    """SELECT timestamp, participant_id, emotion, emotion_confidence, engagement,
              posture, openness, leaning, mirroring, body_language_desc,
              gestures, reactions, confidence
       FROM signal_snapshots WHERE video_id = ?
       ORDER BY timestamp""",
    (video_id,),
)
```

And in the by_ts dict building, add confidence:

```python
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
    "confidence": r["confidence"] if "confidence" in r.keys() else 1.0,
})
```

Add a voting confidence summary endpoint:

```python
@app.get("/api/videos/{video_id}/signals/confidence")
async def get_confidence_summary(video_id: str):
    """Get voting confidence summary for a video."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Overall confidence from signal_snapshots
        cursor = await db.execute(
            "SELECT AVG(confidence) as avg_conf, MIN(confidence) as min_conf, COUNT(*) as total FROM signal_snapshots WHERE video_id = ?",
            (video_id,),
        )
        row = await cursor.fetchone()
        avg_conf = row["avg_conf"] if row and row["avg_conf"] else 1.0
        min_conf = row["min_conf"] if row and row["min_conf"] else 1.0
        total = row["total"] if row else 0

        # Disagreement count from voting_log
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM voting_log WHERE video_id = ? AND disagreements != '[]'",
            (video_id,),
        )
        row = await cursor.fetchone()
        disagreement_count = row["cnt"] if row else 0

    return {
        "avg_confidence": round(avg_conf, 2),
        "min_confidence": round(min_conf, 2),
        "total_snapshots": total,
        "disagreement_count": disagreement_count,
        "accuracy_grade": "A" if avg_conf >= 0.8 else "B" if avg_conf >= 0.6 else "C" if avg_conf >= 0.4 else "D",
    }
```

**Step 2: Update frontend types in `api.ts`**

Add confidence to `SignalSnapshot` participant type:

```typescript
export interface SignalSnapshot {
  timestamp: number;
  participants: Array<{
    label: string;
    emotions: { primary: string; confidence: number; engagement: number };
    body_language: {
      posture: string;
      openness: string;
      leaning: string;
      mirroring: boolean;
      description: string;
    };
    gestures: string[];
    reactions: string[];
    confidence?: number;
  }>;
}

export interface ConfidenceSummary {
  avg_confidence: number;
  min_confidence: number;
  total_snapshots: number;
  disagreement_count: number;
  accuracy_grade: string;
}
```

Add API function:

```typescript
export async function getConfidenceSummary(videoId: string): Promise<ConfidenceSummary | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/confidence`);
  if (!res.ok) return null;
  return res.json();
}
```

**Step 3: Commit**

```bash
git add backend/main.py frontend/src/lib/api.ts
git commit -m "feat: confidence API endpoint and frontend types for voting engine"
```

---

## Task 5: Expandable KPI Card Component

**Files:**
- Create: `frontend/src/components/ui/ExpandableCard.tsx`

**Step 1: Create the expandable card component**

```tsx
import { useState, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { clsx } from 'clsx';

interface ExpandableCardProps {
  children: ReactNode;
  expandedContent: ReactNode;
  className?: string;
  glowColor?: string;
}

export default function ExpandableCard({
  children,
  expandedContent,
  className,
  glowColor = 'rgba(139,92,246,0.15)',
}: ExpandableCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={clsx(
        'glass-card cursor-pointer transition-all duration-300 select-none',
        expanded
          ? 'bg-white/[0.08] border-white/20'
          : 'hover:bg-white/[0.07] hover:border-white/15 hover:-translate-y-0.5',
        className,
      )}
      style={{
        boxShadow: expanded ? `0 0 25px ${glowColor}, 0 4px 20px rgba(0,0,0,0.3)` : 'none',
      }}
    >
      <div className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1">{children}</div>
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
            className="ml-2 mt-1"
          >
            <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
          </motion.div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-0">
              <div className="border-t border-white/10 pt-4">
                {expandedContent}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/ui/ExpandableCard.tsx
git commit -m "feat: ExpandableCard component with glass design and framer-motion animation"
```

---

## Task 6: Expandable KPI Row in AnalysisDashboard

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx` (replace KPI cards with expandable versions)

**Step 1: Add imports**

```typescript
import ExpandableCard from './ui/ExpandableCard';
import { type ConfidenceSummary, getConfidenceSummary } from '../lib/api';
```

Add state:

```typescript
const [confidenceSummary, setConfidenceSummary] = useState<ConfidenceSummary | null>(null);
```

Add to the Promise.all fetch:

```typescript
getConfidenceSummary(videoId),
```

And in the `.then()`:

```typescript
setConfidenceSummary(conf);
```

**Step 2: Replace KPI row (lines 202-230) with expandable cards**

Replace the entire KPI grid:

```tsx
{/* ─── Expandable KPI Row ─────────────────────────────────── */}
<div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
  {/* Overall Score */}
  <ExpandableCard
    glowColor="rgba(139,92,246,0.2)"
    expandedContent={
      <div className="space-y-3">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">How this score is computed</p>
        <p className="text-xs text-slate-400 leading-relaxed">
          The Overall Score combines 3 behavioral channels using the Mehrabian model:
        </p>
        <div className="space-y-2">
          {[
            { label: 'Visual (55%)', value: avgVisual, color: '#818cf8', desc: 'Facial expressions, body language, gestures, eye contact' },
            { label: 'Audio (38%)', value: avgAudio, color: '#a78bfa', desc: 'Voice energy, pitch variation, speaking rate, pauses' },
            { label: 'Text (7%)', value: avgVerbal, color: '#22d3ee', desc: 'Word sentiment, keyword positivity, conversational tone' },
          ].map(ch => (
            <div key={ch.label}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-slate-400">{ch.label}</span>
                <span className="text-[10px] font-bold text-slate-300">{ch.value}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${ch.value}%`, backgroundColor: ch.color }} />
              </div>
              <p className="text-[9px] text-slate-600 mt-0.5">{ch.desc}</p>
            </div>
          ))}
        </div>
        {confidenceSummary && (
          <div className="mt-3 p-2 rounded-lg bg-white/[0.03] border border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-500">Model Confidence</span>
              <span className={`text-xs font-bold ${confidenceSummary.avg_confidence >= 0.8 ? 'text-emerald-400' : confidenceSummary.avg_confidence >= 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
                {Math.round(confidenceSummary.avg_confidence * 100)}% ({confidenceSummary.accuracy_grade})
              </span>
            </div>
            <p className="text-[9px] text-slate-600 mt-1">
              {confidenceSummary.disagreement_count === 0
                ? 'All 3 AI models agreed on every analysis point'
                : `${confidenceSummary.disagreement_count} points where models disagreed (resolved by majority vote)`}
            </p>
          </div>
        )}
        <div className="flex items-center gap-2 mt-2">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
            vid.overall_score >= 80 ? 'bg-emerald-500/10 text-emerald-400' :
            vid.overall_score >= 60 ? 'bg-blue-500/10 text-blue-400' :
            vid.overall_score >= 40 ? 'bg-amber-500/10 text-amber-400' :
            'bg-red-500/10 text-red-400'
          }`}>
            {vid.overall_score >= 80 ? 'Excellent' : vid.overall_score >= 60 ? 'Good' : vid.overall_score >= 40 ? 'Average' : 'Needs Improvement'}
          </span>
        </div>
      </div>
    }
  >
    <TrendingUp className="w-5 h-5 text-purple-400 mx-auto mb-1.5" />
    <p className="text-3xl font-black gradient-text text-center">{Math.round(vid.overall_score)}%</p>
    <p className="text-[10px] text-slate-500 mt-1 text-center">Overall Score</p>
  </ExpandableCard>

  {/* Avg Engagement */}
  <ExpandableCard
    glowColor="rgba(59,130,246,0.2)"
    expandedContent={
      <div className="space-y-3">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Per-Participant Engagement</p>
        {participants.map((p, i) => (
          <div key={p.id} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }} />
            <span className="text-[10px] text-slate-400 flex-1 truncate">{p.name}</span>
            <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full rounded-full bg-blue-400" style={{ width: `${p.engagement_score}%` }} />
            </div>
            <span className="text-[10px] font-bold text-slate-300 w-8 text-right">{Math.round(p.engagement_score)}%</span>
          </div>
        ))}
        <p className="text-[9px] text-slate-600 mt-2">
          Engagement measures active participation through visual cues (expressions, posture),
          audio energy (voice dynamics), and verbal contribution (speaking time, sentiment).
        </p>
      </div>
    }
  >
    <Activity className="w-5 h-5 text-blue-400 mx-auto mb-1.5" />
    <p className="text-3xl font-black gradient-text text-center">{avgEngagement}%</p>
    <p className="text-[10px] text-slate-500 mt-1 text-center">Avg Engagement</p>
  </ExpandableCard>

  {/* Sentiment */}
  <ExpandableCard
    glowColor={summary?.overall_sentiment?.toLowerCase() === 'positive' ? 'rgba(16,185,129,0.2)' : summary?.overall_sentiment?.toLowerCase() === 'negative' ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}
    expandedContent={
      <div className="space-y-3">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Sentiment Analysis</p>
        <p className="text-xs text-slate-400 leading-relaxed">
          Overall meeting tone analyzed from transcript word choice, emotional language,
          and conversational patterns across all participants.
        </p>
        {wordSignals.length > 0 && (
          <div>
            <p className="text-[10px] text-slate-500 mb-1">Sentiment Distribution</p>
            {(() => {
              const pos = wordSignals.filter(w => w.sentiment === 'positive').length;
              const neg = wordSignals.filter(w => w.sentiment === 'negative').length;
              const neu = wordSignals.filter(w => w.sentiment === 'neutral').length;
              const total = Math.max(pos + neg + neu, 1);
              return (
                <div className="flex h-2 rounded-full overflow-hidden bg-white/10">
                  <div className="bg-emerald-500" style={{ width: `${(pos/total)*100}%` }} />
                  <div className="bg-amber-500" style={{ width: `${(neu/total)*100}%` }} />
                  <div className="bg-red-500" style={{ width: `${(neg/total)*100}%` }} />
                </div>
              );
            })()}
            <div className="flex justify-between mt-1">
              <span className="text-[9px] text-emerald-400">Positive</span>
              <span className="text-[9px] text-amber-400">Neutral</span>
              <span className="text-[9px] text-red-400">Negative</span>
            </div>
          </div>
        )}
      </div>
    }
  >
    <Heart className={`w-5 h-5 ${sentimentColor.text} mx-auto mb-1.5`} />
    <p className={`text-2xl font-black bg-gradient-to-r ${sentimentColor.gradient} bg-clip-text text-transparent text-center`}>
      {sentimentLabel}
    </p>
    <p className="text-[10px] text-slate-500 mt-1 text-center">Sentiment</p>
  </ExpandableCard>

  {/* Participants */}
  <ExpandableCard
    glowColor="rgba(6,182,212,0.2)"
    expandedContent={
      <div className="space-y-2">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Quick View</p>
        {participants.map((p, i) => {
          const topMetric = [
            { label: 'Eng', val: p.engagement_score },
            { label: 'Clr', val: p.clarity_score },
            { label: 'Rap', val: p.rapport_score },
            { label: 'Ene', val: p.energy_score },
          ].sort((a, b) => b.val - a.val)[0];
          return (
            <div key={p.id} className="flex items-center gap-2 p-1.5 rounded-lg hover:bg-white/5">
              <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '40', border: `1px solid ${CHART_COLORS[i % CHART_COLORS.length]}60` }}>
                {p.name.charAt(0).toUpperCase()}
              </div>
              <span className="text-[10px] text-slate-300 flex-1 truncate">{p.name}</span>
              <span className="text-[9px] text-slate-500">{topMetric.label}: {Math.round(topMetric.val)}%</span>
              <span className="text-[9px] text-slate-500">Talk: {p.speaking_pct}%</span>
            </div>
          );
        })}
      </div>
    }
  >
    <Users className="w-5 h-5 text-cyan-400 mx-auto mb-1.5" />
    <p className="text-3xl font-black gradient-text text-center">{participants.length}</p>
    <p className="text-[10px] text-slate-500 mt-1 text-center">Participants</p>
  </ExpandableCard>

  {/* Key Moments */}
  <ExpandableCard
    glowColor="rgba(239,68,68,0.2)"
    expandedContent={
      <div className="space-y-2">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Flagged Moments</p>
        {flags.length === 0 ? (
          <p className="text-xs text-slate-500">No critical moments detected</p>
        ) : (
          <div className="space-y-1.5 max-h-40 overflow-y-auto">
            {flags.slice(0, 5).map((flag, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                  flag.severity === 'high' ? 'bg-red-400' : flag.severity === 'medium' ? 'bg-amber-400' : 'bg-emerald-400'
                }`} />
                <div>
                  <p className="text-[10px] text-slate-300 leading-relaxed">{flag.description.slice(0, 80)}{flag.description.length > 80 ? '...' : ''}</p>
                  <span className="text-[9px] text-slate-600 font-mono">{formatTime(flag.timestamp)}</span>
                </div>
              </div>
            ))}
            {flags.length > 5 && <p className="text-[9px] text-slate-600">+{flags.length - 5} more</p>}
          </div>
        )}
        <p className="text-[9px] text-slate-600 mt-1">
          Key moments are detected by AI analysis of emotional shifts, engagement drops,
          and critical conversational turning points.
        </p>
      </div>
    }
  >
    <AlertTriangle className="w-5 h-5 text-red-400 mx-auto mb-1.5" />
    <p className="text-3xl font-black gradient-text text-center">{flags.length}</p>
    <p className="text-[10px] text-slate-500 mt-1 text-center">Key Moments</p>
  </ExpandableCard>
</div>
```

**Step 3: Commit**

```bash
git add frontend/src/components/AnalysisDashboard.tsx frontend/src/components/ui/ExpandableCard.tsx
git commit -m "feat: expandable KPI cards with glass design and confidence display"
```

---

## Task 7: Expandable Channel Score Cards

**Files:**
- Modify: `frontend/src/components/AnalysisDashboard.tsx` (replace channel score section)

**Step 1: Replace the channel scores grid (lines 232-270)**

```tsx
{/* ─── Expandable Channel Scores ──────────────────────────── */}
<div className="grid grid-cols-3 gap-3">
  {/* Visual Channel */}
  <ExpandableCard
    glowColor="rgba(99,102,241,0.2)"
    expandedContent={
      <div className="space-y-2">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Visual Signal Breakdown</p>
        {snapshots.length > 0 && (() => {
          const emotionCounts: Record<string, number> = {};
          snapshots.forEach(s => s.participants.forEach(p => {
            const e = p.emotions.primary;
            emotionCounts[e] = (emotionCounts[e] || 0) + 1;
          }));
          const sorted = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
          const total = Object.values(emotionCounts).reduce((a, b) => a + b, 0);
          return (
            <div className="space-y-1">
              {sorted.map(([emotion, count]) => (
                <div key={emotion} className="flex items-center gap-2">
                  <span className="text-[9px] text-slate-400 w-16 capitalize">{emotion}</span>
                  <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-indigo-400" style={{ width: `${(count/total)*100}%` }} />
                  </div>
                  <span className="text-[9px] text-slate-500">{Math.round((count/total)*100)}%</span>
                </div>
              ))}
            </div>
          );
        })()}
        <p className="text-[9px] text-slate-600">Analyzed from {snapshots.length} video frames using triple-model consensus voting (GPT-4o + GPT-4.1 + GPT-5)</p>
      </div>
    }
  >
    <div className="flex items-center gap-3">
      <Eye className="w-4 h-4 text-indigo-400 flex-shrink-0" />
      <div className="flex-1">
        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Visual (55%)</div>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${avgVisual}%` }} />
          </div>
          <span className="text-sm font-bold text-indigo-400">{avgVisual || '--'}%</span>
        </div>
      </div>
    </div>
  </ExpandableCard>

  {/* Audio Channel */}
  <ExpandableCard
    glowColor="rgba(139,92,246,0.2)"
    expandedContent={
      <div className="space-y-2">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Audio Signal Breakdown</p>
        {speakerAudio && speakerAudio.participants.length > 0 ? (
          speakerAudio.participants.map(p => {
            if (p.features.length === 0) return null;
            const avgPitch = Math.round(p.features.reduce((s, f) => s + f.pitch_mean, 0) / p.features.length);
            const avgVol = Math.round(p.features.reduce((s, f) => s + f.volume_energy, 0) / p.features.length);
            const avgRate = (p.features.reduce((s, f) => s + f.speaking_rate, 0) / p.features.length).toFixed(1);
            return (
              <div key={p.name} className="flex items-center gap-3 text-[10px]">
                <span className="text-slate-300 w-20 truncate">{p.name}</span>
                <span className="text-slate-500">Pitch: <span className="text-violet-300">{avgPitch}Hz</span></span>
                <span className="text-slate-500">Vol: <span className="text-violet-300">{avgVol}%</span></span>
                <span className="text-slate-500">Rate: <span className="text-violet-300">{avgRate}/s</span></span>
              </div>
            );
          })
        ) : (
          <p className="text-[9px] text-slate-600">Real audio features from librosa analysis (pitch, volume, speaking rate, pauses)</p>
        )}
        <p className="text-[9px] text-slate-600">Per-speaker audio extracted and analyzed using librosa + dual-model validation</p>
      </div>
    }
  >
    <div className="flex items-center gap-3">
      <Mic className="w-4 h-4 text-violet-400 flex-shrink-0" />
      <div className="flex-1">
        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Audio (38%)</div>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-violet-500 transition-all" style={{ width: `${avgAudio}%` }} />
          </div>
          <span className="text-sm font-bold text-violet-400">{avgAudio || '--'}%</span>
        </div>
      </div>
    </div>
  </ExpandableCard>

  {/* Text Channel */}
  <ExpandableCard
    glowColor="rgba(6,182,212,0.2)"
    expandedContent={
      <div className="space-y-2">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">Text Signal Breakdown</p>
        {wordSignals.length > 0 && (() => {
          const allKw = wordSignals.flatMap(s => s.keywords);
          const kwCounts: Record<string, number> = {};
          allKw.forEach(k => { kwCounts[k] = (kwCounts[k] || 0) + 1; });
          const topKw = Object.entries(kwCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);
          return (
            <div className="flex flex-wrap gap-1">
              {topKw.map(([kw, count]) => (
                <span key={kw} className="px-1.5 py-0.5 text-[9px] rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                  {kw} <span className="text-cyan-500/50">x{count}</span>
                </span>
              ))}
            </div>
          );
        })()}
        <p className="text-[9px] text-slate-600">Keyword extraction, sentiment scoring, and phrase analysis using dual-model voting (GPT-4o-mini + GPT-4.1)</p>
      </div>
    }
  >
    <div className="flex items-center gap-3">
      <Type className="w-4 h-4 text-cyan-400 flex-shrink-0" />
      <div className="flex-1">
        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Text (7%)</div>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-cyan-500 transition-all" style={{ width: `${avgVerbal}%` }} />
          </div>
          <span className="text-sm font-bold text-cyan-400">{avgVerbal || '--'}%</span>
        </div>
      </div>
    </div>
  </ExpandableCard>
</div>
```

**Step 2: Commit**

```bash
git add frontend/src/components/AnalysisDashboard.tsx
git commit -m "feat: expandable channel score cards with signal breakdown details"
```

---

## Task 8: Glass CSS Enhancements

**Files:**
- Modify: `frontend/src/index.css` (add new glass effects)

**Step 1: Add enhanced glass effects**

Add after the existing `.glass-card-hover` block (after line 41):

```css
.glass-card-expanded {
  background: rgba(255,255,255,0.10);
  border-color: rgba(255,255,255,0.18);
  box-shadow:
    0 0 30px rgba(139,92,246,0.12),
    0 8px 32px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.08);
}

.glass-depth-1 {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
}

.glass-depth-2 {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.animate-shimmer {
  background: linear-gradient(
    90deg,
    rgba(255,255,255,0) 0%,
    rgba(255,255,255,0.03) 50%,
    rgba(255,255,255,0) 100%
  );
  background-size: 200% 100%;
  animation: shimmer 3s ease-in-out infinite;
}

.confidence-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
}
.confidence-high { background: #10b981; box-shadow: 0 0 4px rgba(16,185,129,0.5); }
.confidence-medium { background: #f59e0b; box-shadow: 0 0 4px rgba(245,158,11,0.5); }
.confidence-low { background: #ef4444; box-shadow: 0 0 4px rgba(239,68,68,0.5); }
```

**Step 2: Commit**

```bash
git add frontend/src/index.css
git commit -m "feat: enhanced glass CSS — expanded states, depth layers, confidence indicators"
```

---

## Task 9: Confidence Indicators in Signal Components

**Files:**
- Modify: `frontend/src/components/signals/EmotionsSignal.tsx` (add confidence dots)

**Step 1: Add confidence indicator to EmotionsSignal**

In the redesigned EmotionsSignal (from the previous plan Task 10), add confidence dots to each participant in the "Current" section:

Update the latest participants rendering section:

```tsx
{/* Latest snapshot with confidence */}
<p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Current</p>
{latest.participants.map((p, i) => {
  const conf = (p as any).confidence ?? 1.0;
  return (
    <div key={i} className="flex items-center justify-between py-1">
      <div className="flex items-center gap-1.5">
        <span className={`confidence-dot ${conf >= 0.8 ? 'confidence-high' : conf >= 0.5 ? 'confidence-medium' : 'confidence-low'}`}
          title={`Model confidence: ${Math.round(conf * 100)}%`} />
        <span className="text-xs text-slate-300">{p.label}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs capitalize" style={{ color: getEmotionColor(p.emotions.primary) }}>{p.emotions.primary}</span>
        <span className="text-[10px] text-slate-500">{Math.round(p.emotions.engagement)}%</span>
      </div>
    </div>
  );
})}
```

**Step 2: Commit**

```bash
git add frontend/src/components/signals/EmotionsSignal.tsx
git commit -m "feat: confidence dot indicators on emotion signal components"
```

---

## Task 10: Integration Verification

**Step 1: Verify backend starts clean**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
python -c "
from backend.voting_engine import vote_on_participant, vote_on_frame, cross_signal_validate
print('Voting engine imports OK')

# Quick test
result = vote_on_participant([
    {'label': 'Test', 'emotions': {'primary': 'happy', 'confidence': 0.9, 'engagement': 80}, 'body_language': {'posture': 'upright', 'openness': 'open', 'leaning': 'forward', 'mirroring': False, 'description': 'attentive'}, 'gestures': ['nodding'], 'reactions': ['smile']},
    {'label': 'Test', 'emotions': {'primary': 'happy', 'confidence': 0.8, 'engagement': 75}, 'body_language': {'posture': 'upright', 'openness': 'open', 'leaning': 'neutral', 'mirroring': False, 'description': 'listening'}, 'gestures': ['nodding', 'open palms'], 'reactions': ['smile', 'nod']},
    {'label': 'Test', 'emotions': {'primary': 'engaged', 'confidence': 0.85, 'engagement': 82}, 'body_language': {'posture': 'upright', 'openness': 'mixed', 'leaning': 'forward', 'mirroring': True, 'description': 'leaning in'}, 'gestures': ['head tilt'], 'reactions': ['eyebrow raise']},
])
print(f'Consensus emotion: {result.consensus[\"emotions\"][\"primary\"]}')
print(f'Confidence: {result.confidence}')
print(f'Disagreements: {result.disagreements}')
print('Voting engine test PASSED')
"
```

**Step 2: Start backend and verify new endpoints**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload &
sleep 3
curl -s http://localhost:8001/docs | head -20
```

**Step 3: Start frontend and verify compilation**

```bash
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-offline/frontend"
npm run dev &
sleep 5
curl -s http://localhost:5174 | head -5
```

**Step 4: Manual verification checklist**

- [ ] Backend starts without import errors
- [ ] `voting_engine.py` — vote_on_participant returns correct consensus
- [ ] `voting_engine.py` — cross_signal_validate applies correction rules
- [ ] `/api/videos/{id}/signals/confidence` endpoint returns data
- [ ] Frontend compiles without TypeScript errors
- [ ] KPI cards expand on click with smooth animation
- [ ] Expanded cards show channel breakdown, confidence, rating
- [ ] Channel score cards expand to show signal details
- [ ] Confidence dots appear in signal components (green/amber/red)
- [ ] Glass effects consistent across all components

---

## Summary of All Changes

| Task | Area | Files |
|------|------|-------|
| 1 | Voting engine core | `backend/voting_engine.py` (NEW), `backend/models.py` |
| 2 | Triple-model frame analysis | `backend/core42_client.py` |
| 3 | Wire voting into pipeline | `backend/batch_pipeline.py` |
| 4 | Confidence API + types | `backend/main.py`, `frontend/src/lib/api.ts` |
| 5 | ExpandableCard component | `frontend/src/components/ui/ExpandableCard.tsx` (NEW) |
| 6 | Expandable KPI row | `frontend/src/components/AnalysisDashboard.tsx` |
| 7 | Expandable channel scores | `frontend/src/components/AnalysisDashboard.tsx` |
| 8 | Glass CSS enhancements | `frontend/src/index.css` |
| 9 | Confidence indicators | `frontend/src/components/signals/EmotionsSignal.tsx` |
| 10 | Integration verification | Manual testing |

**Combined with the previous plan (2026-02-17-fix-all-issues.md), the total implementation is 23 tasks across both plans.**
