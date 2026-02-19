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
            confidence=0.5,
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
    if is_speaking and visual_engagement < 40 and camera_on:
        corrected_engagement = max(visual_engagement, 55)
        corrections.append("speaking_engagement_boost")

    # Rule 3: Voice-Visual alignment
    if audio_energy > 70 and visual_engagement < 30 and camera_on:
        corrected_engagement = (audio_energy * 0.6 + visual_engagement * 0.4)
        corrections.append("voice_visual_alignment")

    # Rule 4: Sentiment-Emotion consistency
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
