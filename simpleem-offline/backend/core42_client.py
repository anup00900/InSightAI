from __future__ import annotations

import os
import base64
import json
import re
import asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv

from .api_utils import safe_api_call

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _strip_json_markdown(text: str) -> str:
    """Remove markdown code block wrappers from JSON responses."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


# Azure-format client for chat completions and vision
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.core42.ai/"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)

# OpenAI-compatible client for audio transcription (Azure endpoint doesn't support whisper)
audio_client = AsyncOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.core42.ai/").rstrip("/") + "/v1",
)

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")


async def transcribe_audio(
    audio_path: str,
    language: str = "en",
    prompt: str = "This is a business meeting recording with multiple speakers discussing work topics.",
) -> dict:
    """Transcribe audio using Core42 transcription model via OpenAI-compatible endpoint."""
    async def _call():
        with open(audio_path, "rb") as f:
            try:
                response = await audio_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=language,
                    prompt=prompt,
                )
            except Exception:
                # Fallback: some models don't support timestamp_granularities
                f.seek(0)
                response = await audio_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                    language=language,
                    prompt=prompt,
                )
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    return await safe_api_call(
        _call,
        timeout=60,
        fallback={"segments": [], "text": ""},
        label="transcribe_audio",
    )


async def analyze_frame_emotions(frame_path: str, timestamp: float) -> dict:
    """Analyze a video frame for participant emotions using GPT-4o Vision."""
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    fallback = {
        "participants": [{
            "label": "Person 1", "emotion": "neutral",
            "confidence": 0.5, "engagement": 50,
            "body_language": "unable to analyze",
        }]
    }

    async def _call():
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing facial expressions and body language in video frames. "
                        "You MUST carefully count and analyze EVERY single visible person in the frame, "
                        "even if they are partially visible, in the background, small, or at the edges. "
                        "Do NOT skip anyone. Return JSON with their emotions and engagement level."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this video frame at timestamp {timestamp}s. "
                                "IMPORTANT: First, carefully count ALL visible people in the frame - "
                                "check the center, left side, right side, background, and edges. "
                                "Include everyone even if partially visible or blurry.\n\n"
                                "For EACH visible person, identify:\n"
                                "1. Their label (Person 1, Person 2, etc.) - number them left to right\n"
                                "2. Primary emotion (happy, neutral, surprised, confused, engaged, disengaged, concerned, excited)\n"
                                "3. Confidence level (0-1)\n"
                                "4. Engagement level (0-100)\n"
                                "5. Brief description of body language\n\n"
                                "Return ONLY valid JSON in this format:\n"
                                '{"participants": [{"label": "Person 1", "emotion": "engaged", '
                                '"confidence": 0.85, "engagement": 78, "body_language": "leaning forward, nodding"}]}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=30,
        required_keys=["participants"],
        fallback=fallback,
        label="analyze_frame_emotions",
    )


async def analyze_batch_frames(frame_paths: list[tuple[str, float]], batch_size: int = 5) -> list[dict]:
    """Analyze multiple frames in parallel batches."""
    results = []
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[analyze_frame_emotions(path, ts) for path, ts in batch],
            return_exceptions=True,
        )
        for j, result in enumerate(batch_results):
            ts = batch[j][1]
            if isinstance(result, Exception):
                results.append(
                    {
                        "timestamp": ts,
                        "participants": [
                            {
                                "label": "Person 1",
                                "emotion": "neutral",
                                "confidence": 0.5,
                                "engagement": 50,
                                "body_language": "error in analysis",
                            }
                        ],
                    }
                )
            else:
                result["timestamp"] = ts
                results.append(result)
    return results


async def generate_coaching(transcript: str, emotions_summary: str, participant_name: str) -> list[dict]:
    """Generate coaching recommendations for a participant."""
    fallback = [{"recommendation": "Maintain consistent engagement throughout the meeting", "category": "engagement", "priority": 3}]

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert communication coach analyzing meeting interactions. "
                        "Provide specific, actionable coaching recommendations based on the data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on the following data for participant '{participant_name}', "
                        "provide 3-5 specific coaching recommendations.\n\n"
                        f"TRANSCRIPT EXCERPTS:\n{transcript}\n\n"
                        f"EMOTION/ENGAGEMENT SUMMARY:\n{emotions_summary}\n\n"
                        "Return ONLY valid JSON array:\n"
                        '[{"recommendation": "...", "category": "communication|engagement|leadership|listening", "priority": 1-5}]'
                    ),
                },
            ],
            max_tokens=800,
            temperature=0.5,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=30,
        fallback=fallback,
        label="generate_coaching",
    )


async def analyze_frame_all_signals(
    frame_path: str, timestamp: float, known_names: list[str] | None = None,
    camera_on_names: list[str] | None = None,
) -> dict:
    """Analyze a single frame for ALL visual signals: emotions, body language, gestures, reactions.

    If known_names is provided, the model matches participants to those names instead
    of using generic "Person N" labels.  camera_on_names controls which participants
    have cameras on (and thus can be visually analyzed).

    Returns a combined JSON with all 4 visual signal categories in one GPT-4o call.
    """
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    fallback_label = known_names[0] if known_names else "Person 1"
    fallback = {
        "participants": [{
            "label": fallback_label,
            "camera_on": True,
            "emotions": {"primary": "neutral", "confidence": 0.5, "engagement": 50},
            "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "unable to analyze"},
            "gestures": ["still posture", "resting hands"],
            "reactions": ["neutral gaze", "steady expression"],
        }]
    }

    # Build name context for the prompt
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
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert behavioral analyst specializing in non-verbal communication. "
                        "Analyze the video frame for visible participants across 4 signal categories: "
                        "emotions, body language, gestures, and micro-reactions.\n"
                        "ENGAGEMENT CALIBRATION (use these anchors):\n"
                        "- 85-100: Actively engaged — nodding, smiling, leaning forward, animated gestures, steady eye contact\n"
                        "- 65-84: Attentive — upright posture, looking at camera/screen, occasional nods\n"
                        "- 45-64: Passive attention — still, minimal expression, neutral posture, not obviously disengaged\n"
                        "- 25-44: Low engagement — looking away, slouched, minimal interaction, distracted\n"
                        "- 0-24: Disengaged — looking at phone, turned away, clearly not paying attention\n"
                        "- Camera-OFF participants: ALWAYS engagement=30, emotion='passive'\n"
                        "CRITICAL RULES:\n"
                        "1. Only analyze participants whose camera is ON (face/video visible). "
                        "For camera-OFF participants (initials circle, avatar, black tile): "
                        "include them with camera_on=false, engagement=30, emotions='passive', "
                        "minimal body language.\n"
                        "2. Use REAL NAMES from the meeting platform name labels as 'label'. "
                        "Do NOT use 'Person N' numbering.\n"
                        "3. Each participant appears ONLY ONCE. No duplicates.\n"
                        "4. gestures array: 2-4 items per camera-ON person. "
                        "Examples: nodding, hand wave, pointing, arms crossed, head tilt, "
                        "resting hands, still posture, chin rest, open palms, clasped hands.\n"
                        "5. reactions array: 2-4 items per camera-ON person. "
                        "Examples: eyebrow raise, smile flash, lip press, neutral gaze, "
                        "steady eye contact, slight nod, soft smile, focused expression.\n"
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
                                "For EACH participant with camera ON, analyze:\n\n"
                                "1. EMOTIONS: primary emotion, confidence (0-1), engagement (0-100)\n"
                                "2. BODY LANGUAGE: posture (upright/slouched/leaning), "
                                "openness (open/closed/mixed), leaning (forward/back/neutral), "
                                "mirroring (true/false)\n"
                                "3. GESTURES: 2-4 detected gestures\n"
                                "4. REACTIONS: 2-4 micro-reactions\n\n"
                                "For camera-OFF participants: include with camera_on=false, "
                                "engagement=30, emotion='passive'.\n\n"
                                "Return ONLY valid JSON:\n"
                                '{"participants": [{'
                                '"label": "Anup Roy", "camera_on": true, '
                                '"emotions": {"primary": "engaged", "confidence": 0.85, "engagement": 78}, '
                                '"body_language": {"posture": "upright", "openness": "open", '
                                '"leaning": "forward", "mirroring": false, "description": "leaning forward attentively"}, '
                                '"gestures": ["nodding", "open palms"], '
                                '"reactions": ["slight smile", "steady eye contact"]'
                                "}, {"
                                '"label": "Jane Doe", "camera_on": false, '
                                '"emotions": {"primary": "passive", "confidence": 0.5, "engagement": 30}, '
                                '"body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": false, "description": "camera off"}, '
                                '"gestures": ["still posture"], '
                                '"reactions": ["not visible"]'
                                "}]}"
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
        # Post-process
        for p in parsed.get("participants", []):
            if not p.get("gestures"):
                p["gestures"] = ["still posture", "resting hands"]
            if not p.get("reactions"):
                p["reactions"] = ["neutral gaze", "steady expression"]
        return parsed

    return await safe_api_call(
        _call,
        timeout=30,
        required_keys=["participants"],
        fallback=fallback,
        label="analyze_frame_all_signals",
    )


async def extract_names_from_frame(frame_path: str) -> list[dict]:
    """Extract participant names AND camera status from meeting UI (Zoom/Teams/Meet).

    Uses GPT-4o vision with high detail to OCR name tags visible in the frame.
    Returns a list like:
      [{"name": "Anup Roy", "camera_on": true}, {"name": "Jane Doe", "camera_on": false}]
    """
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    async def _call():
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at reading text overlays in video conferencing screenshots. "
                        "Your task is to read ALL name labels/tags displayed by the meeting platform "
                        "(Zoom, Microsoft Teams, Google Meet, etc.) for every participant.\n\n"
                        "CRITICAL RULES:\n"
                        "1. Read the ACTUAL NAME TEXT shown by the platform for each participant tile.\n"
                        "2. Determine if each participant's camera is ON (face/video visible) or OFF "
                        "(colored circle with initials, avatar, or black tile).\n"
                        "3. In Microsoft Teams, camera-off participants show as colored circles with "
                        "1-2 letter initials AND their full name as a text label. READ THE FULL NAME.\n"
                        "4. Do NOT number them. Use their REAL NAMES from the platform labels.\n"
                        "5. Do NOT include duplicate names — each person appears once."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this meeting screenshot and list ALL participants.\n\n"
                                "For EACH participant visible in the meeting grid:\n"
                                "- Read their EXACT name from the platform's name label\n"
                                "- Determine if their camera is ON (face/video visible) or OFF "
                                "(initials circle, avatar, black tile)\n\n"
                                "Return ONLY valid JSON array — one entry per unique participant:\n"
                                '[{"name": "John Smith", "camera_on": true}, '
                                '{"name": "Jane Doe", "camera_on": false}]\n\n'
                                "IMPORTANT:\n"
                                "- Use the EXACT name shown in the platform label\n"
                                "- Each person only once — no duplicates\n"
                                "- If a name is truncated, include what you can read\n"
                                "- Return [] ONLY if this is a splash/title screen with no participants"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.2,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        result = json.loads(content)
        # Handle both old dict format and new list format
        if isinstance(result, dict):
            return [{"name": name, "camera_on": True} for name in result.values() if name]
        return result

    return await safe_api_call(
        _call,
        timeout=30,
        fallback=[],
        label="extract_names_from_frame",
    )


async def assign_speakers_to_names(
    transcript_segments: list[dict], known_names: list[str]
) -> list[dict]:
    """Assign real speaker names to transcript segments using GPT-4.1.

    Takes transcript segments (each with 'start', 'end', 'text', 'speaker' keys)
    and a list of known participant names. Uses the chat model to intelligently
    match speakers to names based on conversational context.

    Returns the segments list with updated 'speaker' fields.
    """
    if not known_names or not transcript_segments:
        return transcript_segments

    # Format segments for the prompt
    formatted_lines = []
    for seg in transcript_segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "")
        formatted_lines.append(f"[{start:.1f}s-{end:.1f}s]: {text}")
    segments_text = "\n".join(formatted_lines)

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at speaker diarization and identification. "
                        "Given transcript segments and a list of known participant names, "
                        "assign the most likely speaker name to each segment based on "
                        "conversational context, speaking patterns, and content clues."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Known participants: {', '.join(known_names)}\n\n"
                        f"Transcript segments:\n{segments_text}\n\n"
                        "For each segment in order, assign the most likely speaker from "
                        "the known participants list. Return ONLY a valid JSON array of "
                        "speaker names, one per segment, in the same order:\n"
                        '["Name1", "Name2", "Name1", ...]'
                    ),
                },
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        speaker_names = json.loads(content)

        # Update each segment with the assigned speaker name
        for i, seg in enumerate(transcript_segments):
            if i < len(speaker_names):
                seg["speaker"] = speaker_names[i]
        return transcript_segments

    return await safe_api_call(
        _call,
        timeout=30,
        fallback=transcript_segments,
        label="assign_speakers_to_names",
    )


async def analyze_voice_signal(transcript_chunk: str) -> dict:
    """Infer voice signal (tone, pace, dynamics) from a transcript chunk using GPT-4.1."""
    fallback = {"tone": "neutral", "pace": "moderate", "energy": 50, "dynamics": "unable to analyze"}

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
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
                        "Infer:\n"
                        "1. tone: (warm/neutral/tense/enthusiastic/monotone/assertive)\n"
                        "2. pace: (slow/moderate/fast)\n"
                        "3. energy: 0-100 scale\n"
                        "4. dynamics: brief description of vocal pattern\n\n"
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
        _call,
        timeout=15,
        fallback=fallback,
        label="analyze_voice_signal",
    )


async def analyze_words_signal(transcript_chunk: str, timestamp: float) -> dict:
    """Analyze word-level signals: sentiment, keywords, key phrases using GPT-4.1."""
    fallback = {"sentiment": "neutral", "sentiment_score": 0.5, "keywords": [], "key_phrases": []}

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert linguistic analyst. "
                        "Analyze the text for sentiment, key words, and notable phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze this transcript segment (at ~{timestamp:.0f}s):\n\n"
                        f'"{transcript_chunk}"\n\n'
                        "Return ONLY valid JSON:\n"
                        '{"sentiment": "positive|neutral|negative", '
                        '"sentiment_score": 0.0 to 1.0, '
                        '"keywords": ["keyword1", "keyword2"], '
                        '"key_phrases": ["notable phrase 1"]}'
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=15,
        fallback=fallback,
        label="analyze_words_signal",
    )


async def analyze_personality(signals_summary: str, transcript: str) -> dict:
    """Analyze personality traits and communication style from accumulated data using GPT-4.1."""
    fallback = {
        "participants": [{"label": "Person 1", "traits": ["neutral"], "communication_style": "conversational", "confidence_level": "moderate"}]
    }

    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert behavioral psychologist. "
                        "Analyze accumulated behavioral signals and speech to identify personality traits "
                        "and communication styles."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Based on the following accumulated behavioral data and transcript, "
                        "identify personality traits and communication style for each participant.\n\n"
                        f"BEHAVIORAL SIGNALS:\n{signals_summary[:3000]}\n\n"
                        f"TRANSCRIPT:\n{transcript[:3000]}\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"participants": [{"label": "Person 1", '
                        '"traits": ["analytical", "collaborative", "assertive"], '
                        '"communication_style": "direct and structured", '
                        '"confidence_level": "high"}]}'
                    ),
                },
            ],
            max_tokens=600,
            temperature=0.4,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=30,
        required_keys=["participants"],
        fallback=fallback,
        label="analyze_personality",
    )


async def build_correlations(all_signals: str, transcript: str) -> dict:
    """Detect cross-signal patterns and correlations using GPT-4.1."""
    async def _call():
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert behavioral analyst specializing in multi-modal signal correlation. "
                        "Identify patterns where different behavioral signals reinforce or contradict each other."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze these behavioral signals for cross-signal patterns and correlations:\n\n"
                        f"SIGNALS:\n{all_signals[:4000]}\n\n"
                        f"TRANSCRIPT:\n{transcript[:2000]}\n\n"
                        "Identify 2-5 notable correlations. Return ONLY valid JSON:\n"
                        '{"correlations": [{'
                        '"signal_pair": ["emotions", "body_language"], '
                        '"pattern": "Engagement spikes correlate with forward leaning posture", '
                        '"confidence": 0.85, '
                        '"implication": "Genuine interest indicated by aligned verbal and non-verbal cues"'
                        "}]}"
                    ),
                },
            ],
            max_tokens=800,
            temperature=0.4,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=30,
        required_keys=["correlations"],
        fallback={"correlations": []},
        label="build_correlations",
    )


async def generate_summary_and_flags(transcript: str, emotions_summary: str, participant_names: list[str]) -> dict:
    """Generate meeting summary and critical moment flags."""
    fallback = {
        "summary": "Meeting analysis completed. Unable to parse detailed results.",
        "key_topics": ["meeting"],
        "overall_sentiment": "neutral",
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
                        "You are an expert meeting analyst. Analyze the meeting data and provide "
                        "a comprehensive summary, identify critical moments, and assess overall dynamics."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze this meeting and provide:\n"
                        "1. A 2-3 paragraph summary\n"
                        "2. Key topics discussed\n"
                        "3. Critical moments (positive and negative flags with timestamps)\n"
                        "4. Overall sentiment\n"
                        "5. Engagement and rapport scores per participant\n\n"
                        f"PARTICIPANTS: {', '.join(participant_names)}\n\n"
                        f"TRANSCRIPT:\n{transcript}\n\n"
                        f"EMOTION DATA:\n{emotions_summary}\n\n"
                        "Return ONLY valid JSON:\n"
                        "{\n"
                        '  "summary": "...",\n'
                        '  "key_topics": ["topic1", "topic2"],\n'
                        '  "overall_sentiment": "positive|neutral|negative",\n'
                        '  "flags": [{"timestamp": 0.0, "type": "positive|negative", "description": "...", "severity": "low|medium|high"}],\n'
                        '  "participant_scores": [{"name": "...", "engagement": 0-100, "sentiment": -1.0 to 1.0, '
                        '"speaking_pct": 0-100, "clarity": 0-100, "rapport": 0-100, "energy": 0-100}]\n'
                        "}"
                    ),
                },
            ],
            max_tokens=2000,
            temperature=0.4,
        )
        content = _strip_json_markdown(response.choices[0].message.content)
        return json.loads(content)

    return await safe_api_call(
        _call,
        timeout=60,
        required_keys=["summary", "participant_scores"],
        fallback=fallback,
        label="generate_summary_and_flags",
    )
