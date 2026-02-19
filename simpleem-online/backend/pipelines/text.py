"""Text signal pipeline — Whisper transcription + GPT NLP analysis.

Transcribes audio and analyzes sentiment/keywords.
Contributes 7% to the holistic engagement score.
"""

from backend.core42_client import transcribe_audio, analyze_words_signal, analyze_voice_signal
from backend.signal_bus import SignalBus, SignalEvent, SignalType


async def transcribe_and_publish(
    audio_chunk_path: str,
    chunk_start: float,
    signal_bus: SignalBus,
) -> dict:
    """Transcribe an audio chunk and publish text signals to the bus."""
    transcript_data = await transcribe_audio(audio_chunk_path)
    segments = transcript_data.get("segments", [])

    # Analyze words for this chunk
    chunk_text = " ".join(s.get("text", "") for s in segments)
    if chunk_text.strip():
        words_result = await analyze_words_signal(chunk_text, chunk_start)

        await signal_bus.publish(SignalEvent(
            signal_type=SignalType.TEXT,
            participant_id="all",  # Text signal applies to all for now
            timestamp=chunk_start,
            data={
                "sentiment": words_result.get("sentiment", "neutral"),
                "sentiment_score": words_result.get("sentiment_score", 0.5),
                "keywords": words_result.get("keywords", []),
                "key_phrases": words_result.get("key_phrases", []),
                "text": chunk_text,
            },
        ))

    return transcript_data
