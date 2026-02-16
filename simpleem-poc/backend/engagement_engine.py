"""
EngagementEngine — computes holistic per-participant engagement scores
using Simpleem-style 55% visual / 38% audio / 7% verbal weighting.

Also provides EngagementAlertDetector which monitors for sustained low
engagement and emits EngagementAlert objects when a participant stays
below a configurable threshold for a configurable duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

from backend.signal_bus import SignalEvent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EngagementScore:
    """A computed engagement score for a single participant at a point in time."""
    participant_id: str
    timestamp: float
    holistic: float  # 0-100 weighted composite
    visual: float    # 0-100 visual channel score
    audio: float     # 0-100 audio channel score
    verbal: float    # 0-100 verbal/text channel score


@dataclass
class EngagementAlert:
    """Alert emitted when a participant's engagement stays below threshold."""
    participant_id: str
    timestamp: float
    score: float
    duration_below: float
    suggested_action: str


# ---------------------------------------------------------------------------
# EngagementEngine
# ---------------------------------------------------------------------------

class EngagementEngine:
    """Computes holistic engagement scores from visual, audio, and text signals.

    Weighting follows the Simpleem model:
        - Visual  : 55%
        - Audio   : 38%
        - Verbal  :  7%

    When one or more signal channels are missing, the remaining weights are
    redistributed proportionally so they still sum to 1.0.
    """

    VISUAL_WEIGHT: float = 0.55
    AUDIO_WEIGHT: float = 0.38
    VERBAL_WEIGHT: float = 0.07

    def compute(
        self,
        participant_id: str,
        timestamp: float,
        visual_signal: Optional[SignalEvent] = None,
        audio_signal: Optional[SignalEvent] = None,
        text_signal: Optional[SignalEvent] = None,
    ) -> EngagementScore:
        """Compute a weighted engagement score from available signals.

        Parameters
        ----------
        participant_id : str
            Identifier of the meeting participant.
        timestamp : float
            The point in time this score refers to.
        visual_signal : SignalEvent | None
            Expected ``data["engagement"]`` in 0-100.
        audio_signal : SignalEvent | None
            Expected ``data["energy"]`` in 0-100.
        text_signal : SignalEvent | None
            Expected ``data["sentiment_score"]`` in 0-1 (scaled to 0-100).

        Returns
        -------
        EngagementScore
        """
        # Extract per-channel scores (0-100).
        visual_score: float = 0.0
        audio_score: float = 0.0
        verbal_score: float = 0.0

        # Build list of (weight, score) for available channels.
        available: list[tuple[float, float]] = []

        if visual_signal is not None:
            visual_score = float(visual_signal.data["engagement"])
            available.append((self.VISUAL_WEIGHT, visual_score))

        if audio_signal is not None:
            audio_score = float(audio_signal.data["energy"])
            available.append((self.AUDIO_WEIGHT, audio_score))

        if text_signal is not None:
            verbal_score = float(text_signal.data["sentiment_score"]) * 100.0
            available.append((self.VERBAL_WEIGHT, verbal_score))

        # Compute holistic score.
        if len(available) == 0:
            holistic = 0.0
        elif len(available) == 1:
            # Single signal: use it directly.
            holistic = available[0][1]
        else:
            # Redistribute weights proportionally among available signals.
            total_weight = sum(w for w, _ in available)
            holistic = sum((w / total_weight) * s for w, s in available)

        return EngagementScore(
            participant_id=participant_id,
            timestamp=timestamp,
            holistic=holistic,
            visual=visual_score,
            audio=audio_score,
            verbal=verbal_score,
        )


# ---------------------------------------------------------------------------
# EngagementAlertDetector
# ---------------------------------------------------------------------------

class EngagementAlertDetector:
    """Monitors engagement scores and emits alerts for sustained low engagement.

    An alert fires when a participant's holistic score stays below *threshold*
    for at least *min_duration* seconds.  Only one alert is emitted per
    continuous drop episode; the participant must go above the threshold and
    then drop again to trigger a new alert.

    Parameters
    ----------
    threshold : float
        Score below which engagement is considered "low" (default 40).
    min_duration : float
        Seconds the score must remain below *threshold* before an alert
        is generated (default 30).
    """

    def __init__(self, threshold: float = 40, min_duration: float = 30) -> None:
        self.threshold = threshold
        self.min_duration = min_duration

        # participant_id -> timestamp when they first dropped below threshold
        self._drop_start: Dict[str, float] = {}

        # participant_ids for which an alert has already been sent in the
        # current drop episode (prevents duplicate alerts).
        self._alerted: Set[str] = set()

    def check(self, score: EngagementScore) -> Optional[EngagementAlert]:
        """Evaluate an engagement score and return an alert if warranted.

        Parameters
        ----------
        score : EngagementScore
            The latest computed engagement score for a participant.

        Returns
        -------
        EngagementAlert | None
            An alert object if the participant has been below threshold for
            at least *min_duration* seconds (and hasn't already been alerted
            for this episode), otherwise ``None``.
        """
        pid = score.participant_id

        if score.holistic >= self.threshold:
            # Above threshold: reset tracking for this participant.
            self._drop_start.pop(pid, None)
            self._alerted.discard(pid)
            return None

        # Below threshold.
        if pid not in self._drop_start:
            # First time below: start tracking.
            self._drop_start[pid] = score.timestamp
            return None

        # Already tracking — check duration.
        duration = score.timestamp - self._drop_start[pid]

        if duration >= self.min_duration and pid not in self._alerted:
            self._alerted.add(pid)
            return EngagementAlert(
                participant_id=pid,
                timestamp=score.timestamp,
                score=score.holistic,
                duration_below=duration,
                suggested_action=(
                    f"Participant {pid} engagement has been below "
                    f"{self.threshold} for {duration:.0f}s. "
                    "Consider prompting interaction or checking in."
                ),
            )

        # Either duration too short or already alerted.
        return None

    def reset(self) -> None:
        """Clear all tracking state."""
        self._drop_start.clear()
        self._alerted.clear()
