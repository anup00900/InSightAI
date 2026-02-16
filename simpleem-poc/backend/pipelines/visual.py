"""Visual signal pipeline — GPT-4o Vision frame analysis.

Analyzes video frames for emotions, body language, gestures, and reactions.
Contributes 55% to the holistic engagement score.
"""

from backend.core42_client import analyze_frame_all_signals, analyze_frame_emotions
from backend.frame_extractor import extract_single_frame
from backend.signal_bus import SignalBus, SignalEvent, SignalType


async def analyze_frame_and_publish(
    frame_path: str,
    timestamp: float,
    signal_bus: SignalBus,
) -> dict:
    """Analyze a frame and publish visual signals to the bus."""
    result = await analyze_frame_all_signals(frame_path, timestamp)
    result["timestamp"] = timestamp

    # Publish per-participant visual signals
    for p in result.get("participants", []):
        label = p.get("label", "Person 1")
        emotions = p.get("emotions", {})

        await signal_bus.publish(SignalEvent(
            signal_type=SignalType.VISUAL,
            participant_id=label,
            timestamp=timestamp,
            data={
                "engagement": emotions.get("engagement", 50),
                "emotion": emotions.get("primary", "neutral"),
                "confidence": emotions.get("confidence", 0.5),
                "body_language": p.get("body_language", {}),
                "gestures": p.get("gestures", []),
                "reactions": p.get("reactions", []),
            },
        ))

    return result
