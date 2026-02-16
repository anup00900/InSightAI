"""Tests for the EngagementEngine and EngagementAlertDetector."""

import pytest
from backend.signal_bus import SignalEvent, SignalType
from backend.engagement_engine import (
    EngagementEngine,
    EngagementScore,
    EngagementAlertDetector,
    EngagementAlert,
)


@pytest.fixture
def engine():
    """Return a fresh EngagementEngine for each test."""
    return EngagementEngine()


@pytest.fixture
def detector():
    """Return an EngagementAlertDetector with threshold=40, min_duration=30."""
    return EngagementAlertDetector(threshold=40, min_duration=30)


# ---------------------------------------------------------------------------
# EngagementEngine tests
# ---------------------------------------------------------------------------


class TestEngagementEngine:
    def test_compute_with_all_signals(self, engine):
        """visual=80, audio=70, verbal=80 -> holistic ~ 76.2"""
        visual_signal = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="p1",
            timestamp=1.0,
            data={"engagement": 80},
        )
        audio_signal = SignalEvent(
            signal_type=SignalType.AUDIO,
            participant_id="p1",
            timestamp=1.0,
            data={"energy": 70},
        )
        text_signal = SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="p1",
            timestamp=1.0,
            data={"sentiment_score": 0.8},  # 0.8 * 100 = 80
        )

        score = engine.compute(
            participant_id="p1",
            timestamp=1.0,
            visual_signal=visual_signal,
            audio_signal=audio_signal,
            text_signal=text_signal,
        )

        assert isinstance(score, EngagementScore)
        assert score.participant_id == "p1"
        assert score.timestamp == 1.0
        assert score.visual == 80.0
        assert score.audio == 70.0
        assert score.verbal == 80.0
        # holistic = 0.55*80 + 0.38*70 + 0.07*80 = 44 + 26.6 + 5.6 = 76.2
        assert abs(score.holistic - 76.2) < 0.1

    def test_compute_without_audio(self, engine):
        """Only visual + text -> weights redistributed proportionally."""
        visual_signal = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="p1",
            timestamp=2.0,
            data={"engagement": 80},
        )
        text_signal = SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="p1",
            timestamp=2.0,
            data={"sentiment_score": 0.6},  # 0.6 * 100 = 60
        )

        score = engine.compute(
            participant_id="p1",
            timestamp=2.0,
            visual_signal=visual_signal,
            text_signal=text_signal,
        )

        assert score.visual == 80.0
        assert score.audio == 0.0
        assert score.verbal == 60.0
        # Redistributed weights: visual_w = 0.55/(0.55+0.07), verbal_w = 0.07/(0.55+0.07)
        # visual_w ≈ 0.887, verbal_w ≈ 0.113
        # holistic = 0.887*80 + 0.113*60 ≈ 70.97 + 6.77 ≈ 77.74
        expected = (0.55 / (0.55 + 0.07)) * 80 + (0.07 / (0.55 + 0.07)) * 60
        assert abs(score.holistic - expected) < 0.1

    def test_compute_visual_only(self, engine):
        """Only visual signal -> holistic equals visual score directly."""
        visual_signal = SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id="p1",
            timestamp=3.0,
            data={"engagement": 65},
        )

        score = engine.compute(
            participant_id="p1",
            timestamp=3.0,
            visual_signal=visual_signal,
        )

        assert score.visual == 65.0
        assert score.audio == 0.0
        assert score.verbal == 0.0
        assert score.holistic == 65.0

    def test_compute_no_signals(self, engine):
        """No signals at all -> all zeros."""
        score = engine.compute(
            participant_id="p1",
            timestamp=4.0,
        )

        assert score.visual == 0.0
        assert score.audio == 0.0
        assert score.verbal == 0.0
        assert score.holistic == 0.0


# ---------------------------------------------------------------------------
# EngagementAlertDetector tests
# ---------------------------------------------------------------------------


class TestEngagementAlertDetector:
    def test_alert_fires_after_duration(self, detector):
        """Below threshold for > min_duration -> alert fires."""
        # First check: score below threshold at t=0
        score1 = EngagementScore(
            participant_id="p1", timestamp=0.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        result1 = detector.check(score1)
        assert result1 is None  # first time below, just starts tracking

        # Second check: still below threshold at t=31 (> 30s min_duration)
        score2 = EngagementScore(
            participant_id="p1", timestamp=31.0,
            holistic=25.0, visual=25.0, audio=25.0, verbal=25.0,
        )
        result2 = detector.check(score2)
        assert result2 is not None
        assert isinstance(result2, EngagementAlert)
        assert result2.participant_id == "p1"
        assert result2.duration_below >= 30.0
        assert result2.score == 25.0
        assert isinstance(result2.suggested_action, str)
        assert len(result2.suggested_action) > 0

        # Third check: still below, but should NOT re-alert (same episode)
        score3 = EngagementScore(
            participant_id="p1", timestamp=45.0,
            holistic=20.0, visual=20.0, audio=20.0, verbal=20.0,
        )
        result3 = detector.check(score3)
        assert result3 is None  # already alerted for this drop

    def test_no_alert_when_above_threshold(self, detector):
        """Score goes above threshold -> resets tracking, no alert."""
        # Below threshold at t=0
        score1 = EngagementScore(
            participant_id="p1", timestamp=0.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        detector.check(score1)

        # Above threshold at t=10 -> resets
        score2 = EngagementScore(
            participant_id="p1", timestamp=10.0,
            holistic=50.0, visual=50.0, audio=50.0, verbal=50.0,
        )
        result2 = detector.check(score2)
        assert result2 is None

        # Below threshold again at t=20 (fresh start)
        score3 = EngagementScore(
            participant_id="p1", timestamp=20.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        detector.check(score3)

        # Still below at t=40, but only 20s into new drop (< 30s min_duration)
        score4 = EngagementScore(
            participant_id="p1", timestamp=40.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        result4 = detector.check(score4)
        assert result4 is None  # not long enough yet

    def test_no_alert_when_duration_too_short(self, detector):
        """Below threshold but not long enough -> no alert."""
        score1 = EngagementScore(
            participant_id="p1", timestamp=0.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        result1 = detector.check(score1)
        assert result1 is None

        # Only 20 seconds later (< 30s min_duration)
        score2 = EngagementScore(
            participant_id="p1", timestamp=20.0,
            holistic=35.0, visual=35.0, audio=35.0, verbal=35.0,
        )
        result2 = detector.check(score2)
        assert result2 is None

    def test_reset_clears_state(self, detector):
        """reset() clears all tracking state."""
        score1 = EngagementScore(
            participant_id="p1", timestamp=0.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        detector.check(score1)

        detector.reset()

        # After reset, this should be treated as fresh start
        score2 = EngagementScore(
            participant_id="p1", timestamp=31.0,
            holistic=30.0, visual=30.0, audio=30.0, verbal=30.0,
        )
        result = detector.check(score2)
        assert result is None  # first time below after reset, no alert
