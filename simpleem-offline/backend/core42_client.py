from __future__ import annotations

import os
import base64
import json
import logging
import re
import asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv

from .api_utils import safe_api_call, cascade_api_call

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _strip_json_markdown(text: str) -> str:
    """Remove markdown code block wrappers and sanitize control chars from JSON responses."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    # Replace invalid JSON control characters inside strings (raw newlines, tabs, etc.)
    # These cause json.loads() to fail with "Invalid control character" errors
    text = text.replace('\r\n', '\\n').replace('\r', '\\n')
    # Replace raw newlines inside JSON string values (between quotes) but keep structural newlines
    # Simpler approach: replace all control chars except structural whitespace
    sanitized = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            sanitized.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            sanitized.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            sanitized.append(ch)
            continue
        if in_string and ch in ('\n', '\t'):
            sanitized.append('\\n' if ch == '\n' else '\\t')
            continue
        sanitized.append(ch)
    return ''.join(sanitized)


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
CHAT_MODEL_STRONG = os.getenv("CHAT_MODEL_STRONG", "gpt-5")  # For complex analysis (summary, flags)
CHAT_MODEL_FAST = os.getenv("CHAT_MODEL_FAST", "gpt-4o-mini")  # For simple tasks (voice/word signals, coaching)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

# Smart cascade: ordered fallback models per task category
MODEL_CASCADE = {
    "vision":        [VISION_MODEL, CHAT_MODEL, CHAT_MODEL_FAST],
    "summary":       [CHAT_MODEL_STRONG, CHAT_MODEL, VISION_MODEL],
    "analysis":      [CHAT_MODEL, VISION_MODEL, CHAT_MODEL_FAST],
    "fast":          [CHAT_MODEL_FAST, CHAT_MODEL, VISION_MODEL],
    "transcription": [WHISPER_MODEL],
}

# Rough token estimation: ~4 chars per token for English text
# GPT-4.1 on Core42 context is typically 128K tokens; leave room for system prompt + output
MAX_INPUT_CHARS = 100_000  # ~25K tokens — safe for any context window

# GPT-5 on Core42 requires max_completion_tokens (not max_tokens) and no temperature param
_GPT5_MODELS = {"gpt-5", "gpt5"}


def _model_kwargs(model: str, max_tokens: int = 2500, temperature: float = 0.3) -> dict:
    """Build model-compatible API kwargs — handles GPT-5 parameter differences."""
    if model.lower() in _GPT5_MODELS:
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": temperature}


def _fit_context(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Trim text to fit within context window, keeping start + end for best quality."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... [middle trimmed for context limit] ...\n\n" + text[-half:]


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

    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
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
                            f"TRANSCRIPT EXCERPTS:\n{_fit_context(transcript, 30000)}\n\n"
                            f"EMOTION/ENGAGEMENT SUMMARY:\n{_fit_context(emotions_summary, 15000)}\n\n"
                            "Return ONLY valid JSON array:\n"
                            '[{"recommendation": "...", "category": "communication|engagement|leadership|listening", "priority": 1-5}]'
                        ),
                    },
                ],
                **_model_kwargs(model, max_tokens=800, temperature=0.5),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["fast"],
        timeout=45,
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

    For large transcripts (200+ segments), batches requests in groups of 50
    to avoid token limits and timeouts.

    Returns the segments list with updated 'speaker' fields.
    """
    if not known_names or not transcript_segments:
        return transcript_segments

    BATCH_SIZE = 50
    all_speaker_names = []

    for batch_start in range(0, len(transcript_segments), BATCH_SIZE):
        batch = transcript_segments[batch_start:batch_start + BATCH_SIZE]

        formatted_lines = []
        for i, seg in enumerate(batch):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
            speaker = seg.get("speaker", "")
            formatted_lines.append(f"[{batch_start+i}] [{start:.1f}s-{end:.1f}s] {speaker}: {text}")
        segments_text = "\n".join(formatted_lines)

        # Context from previous batch for continuity
        context = ""
        if all_speaker_names:
            last_few = all_speaker_names[-3:]
            context = f"\nPrevious speakers in order: {last_few}\n"

        # Build a numbered reference list of names for the prompt
        numbered_names = "\n".join(f"  {i+1}. \"{n}\"" for i, n in enumerate(known_names))

        async def _call(st=segments_text, ctx=context):
            response = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at speaker diarization and identification. "
                            "Given transcript segments and a list of known participant names, "
                            "assign the most likely speaker name to each segment.\n\n"
                            "CRITICAL RULES:\n"
                            "1. You MUST use EXACTLY the full names from the known participants list — "
                            "copy-paste them character-for-character. Do NOT use first names only, "
                            "nicknames, or abbreviations.\n"
                            "2. For 'Last, First' format names (e.g. 'Chong, Sam'), output the FULL "
                            "string 'Chong, Sam', NOT just 'Sam'.\n"
                            "3. For names with parenthetical notes (e.g. 'Giorgio Manenti (E)'), "
                            "include the full string exactly.\n"
                            "4. NEVER invent names not in the known participants list.\n"
                            "5. In large meetings, many people take turns speaking briefly. "
                            "Use conversational cues (greetings, being addressed by name, topic "
                            "context) to identify speakers. If a speaker says 'Thanks Sam' the "
                            "NEXT speaker is likely 'Chong, Sam'.\n"
                            "6. If you truly cannot determine the speaker, use the most recent "
                            "identified speaker rather than guessing a wrong name."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Known participants (use EXACTLY these strings):\n{numbered_names}\n{ctx}\n"
                            f"Transcript segments:\n{st}\n\n"
                            f"For each of these {len(batch)} segments, assign the most likely speaker "
                            "from the known participants list above. "
                            "Return ONLY a valid JSON array of speaker names, one per segment, in exact order. "
                            "Each name MUST be copied exactly from the known participants list:\n"
                            '["Full Name 1", "Full Name 2", ...]'
                        ),
                    },
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)

        batch_names = await safe_api_call(
            _call,
            timeout=60,
            fallback=[seg.get("speaker", "") for seg in batch],
            label=f"assign_speakers_batch_{batch_start}",
        )
        all_speaker_names.extend(batch_names)

    # Post-process: fuzzy-match any returned names that don't exactly match known_names
    known_set = set(known_names)
    known_lower_map = {n.lower().strip(): n for n in known_names}
    # Build partial name lookup (first name, last name, etc.)
    import re as _re
    partial_lookup: dict[str, str] = {}
    for n in known_names:
        # Strip parenthetical suffixes and split on commas/spaces
        cleaned = _re.sub(r'\s*\([^)]*\)', '', n)
        parts = _re.split(r'[,\s]+', cleaned.lower().strip())
        for part in parts:
            if len(part) >= 3 and part not in partial_lookup:
                partial_lookup[part] = n

    for i, speaker in enumerate(all_speaker_names):
        if not speaker or speaker in known_set:
            continue
        # Try case-insensitive exact match
        matched = known_lower_map.get(speaker.lower().strip())
        if matched:
            all_speaker_names[i] = matched
            continue
        # Try partial match (first name only, last name only)
        speaker_lower = speaker.lower().strip()
        matched = partial_lookup.get(speaker_lower)
        if matched:
            all_speaker_names[i] = matched
            continue
        # Try substring match
        for kn in known_names:
            if speaker_lower in kn.lower() or kn.lower() in speaker_lower:
                all_speaker_names[i] = kn
                break

    # Apply speaker names to segments
    for i, seg in enumerate(transcript_segments):
        if i < len(all_speaker_names) and all_speaker_names[i]:
            seg["speaker"] = all_speaker_names[i]

    return transcript_segments


async def analyze_voice_signal(transcript_chunk: str) -> dict:
    """Infer voice signal (tone, pace, dynamics) from a transcript chunk using GPT-4.1."""
    fallback = {"tone": "neutral", "pace": "moderate", "energy": 50, "dynamics": "unable to analyze"}

    def _make_call(model):
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
                **_model_kwargs(model, max_tokens=200, temperature=0.3),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["fast"],
        timeout=15,
        fallback=fallback,
        label="analyze_voice_signal",
    )


async def analyze_words_signal(transcript_chunk: str, timestamp: float) -> dict:
    """Analyze word-level signals: sentiment, keywords, key phrases using GPT-4.1."""
    fallback = {"sentiment": "neutral", "sentiment_score": 0.5, "keywords": [], "key_phrases": []}

    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
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
                **_model_kwargs(model, max_tokens=300, temperature=0.3),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["fast"],
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
                        f"BEHAVIORAL SIGNALS:\n{_fit_context(signals_summary, 40000)}\n\n"
                        f"TRANSCRIPT:\n{_fit_context(transcript, 40000)}\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"participants": [{"label": "Person 1", '
                        '"traits": ["analytical", "collaborative", "assertive"], '
                        '"communication_style": "direct and structured", '
                        '"confidence_level": "high"}]}'
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
                        f"SIGNALS:\n{_fit_context(all_signals, 40000)}\n\n"
                        f"TRANSCRIPT:\n{_fit_context(transcript, 30000)}\n\n"
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
        timeout=60,
        required_keys=["correlations"],
        fallback={"correlations": []},
        label="build_correlations",
    )


async def generate_meeting_summary(transcript: str, participant_names: list[str] | None = None) -> dict:
    """Generate a dedicated meeting summary using smart cascade fallback."""
    fallback = {
        "summary": "Meeting analysis completed. Summary generation encountered an issue.",
        "key_topics": ["meeting"],
        "overall_sentiment": "neutral",
    }

    names_context = ""
    if participant_names:
        names_context = f"\nPARTICIPANTS: {', '.join(participant_names)}\n"

    system_msg = (
        "You are an expert meeting analyst and technical writer. "
        "Your task is to write a clear, detailed meeting summary.\n\n"
        "RULES:\n"
        "- Write 3-4 substantive paragraphs covering: context/purpose, key discussions, "
        "decisions reached, and overall dynamics/outcomes.\n"
        "- Be specific — reference actual topics, names, and decisions from the transcript.\n"
        "- Mention participants by name and what they contributed or discussed.\n"
        "- Identify 4-8 key topics discussed.\n"
        "- Determine overall sentiment (positive/neutral/negative) based on tone and outcomes.\n"
        "- If speaker labels are generic (e.g. 'SPEAKER_00'), infer identity from context."
    )
    user_msg = (
        f"Write a comprehensive summary of this meeting.\n"
        f"{names_context}\n"
        f"FULL TRANSCRIPT:\n{_fit_context(transcript, 80000)}\n\n"
        "Return ONLY valid JSON:\n"
        '{"summary": "3-4 detailed paragraphs...", '
        '"key_topics": ["topic1", "topic2", ...], '
        '"overall_sentiment": "positive|neutral|negative"}'
    )

    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                **_model_kwargs(model, max_tokens=3000, temperature=0.4),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["summary"],
        timeout=120,
        required_keys=["summary"],
        fallback=fallback,
        label="generate_meeting_summary",
    )


async def generate_meeting_notes(transcript: str) -> dict:
    """Generate structured meeting notes using smart cascade fallback."""
    fallback = {"action_items": [], "decisions": [], "follow_ups": [], "key_questions": []}

    system_msg = (
        "You are an expert meeting secretary. Extract structured, actionable notes "
        "from the meeting transcript. Be precise and specific.\n\n"
        "For each action item: include WHO is responsible and WHAT they need to do.\n"
        "For each decision: include WHAT was decided and any conditions.\n"
        "For follow-ups: include WHAT needs follow-up and by WHEN if mentioned.\n"
        "For key questions: include questions that were raised but NOT answered."
    )
    user_msg = (
        f"Extract structured meeting notes from this transcript:\n\n"
        f"{_fit_context(transcript, 80000)}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "action_items": [{"owner": "Name", "task": "What to do", "deadline": "if mentioned or null"}],\n'
        '  "decisions": [{"decision": "What was decided", "context": "Brief context"}],\n'
        '  "follow_ups": [{"topic": "What needs follow-up", "owner": "Name or null", "timeline": "if mentioned or null"}],\n'
        '  "key_questions": [{"question": "Unanswered question", "raised_by": "Name or null"}]\n'
        "}"
    )

    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                **_model_kwargs(model, max_tokens=2000, temperature=0.3),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["summary"],
        timeout=90,
        required_keys=["action_items"],
        fallback=fallback,
        label="generate_meeting_notes",
    )


async def generate_flags_and_scores(transcript: str, emotions_summary: str, participant_names: list[str]) -> dict:
    """Generate critical moment flags and per-participant scores."""
    fallback = {
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

    def _make_call(model):
        async def _call():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert meeting analyst. Identify critical moments (flags) and score each participant.\n\n"
                            "CRITICAL RULES:\n"
                            "- flags: Identify 3-8 critical moments with ACCURATE timestamps from the transcript.\n"
                            "- engagement: Active speakers with positive emotions = 70-90, silent/passive = 30-50, camera-off = 20-40.\n"
                            "- sentiment: Positive speakers = 0.5 to 1.0, negative = -0.5 to -1.0.\n"
                            "- clarity/rapport/energy: Score based on actual communication patterns. DO NOT use same score for everyone."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Analyze this meeting with {len(participant_names)} participants.\n\n"
                            f"PARTICIPANTS: {', '.join(participant_names)}\n\n"
                            f"TRANSCRIPT:\n{_fit_context(transcript, 50000)}\n\n"
                            f"EMOTION/ENGAGEMENT DATA:\n{_fit_context(emotions_summary, 30000)}\n\n"
                            "Return ONLY valid JSON:\n"
                            "{\n"
                            '  "flags": [{"timestamp": 123.0, "type": "positive|negative", "description": "what happened", "severity": "low|medium|high"}],\n'
                            '  "participant_scores": [\n'
                            '    {"name": "exact name", "engagement": 0-100, "sentiment": -1.0 to 1.0, '
                            '"clarity": 0-100, "rapport": 0-100, "energy": 0-100}\n'
                            "  ]\n"
                            "}\n\n"
                            "EVERY participant must appear with DIFFERENT scores based on ACTUAL behavior."
                        ),
                    },
                ],
                **_model_kwargs(model, max_tokens=3000, temperature=0.3),
            )
            content = _strip_json_markdown(response.choices[0].message.content)
            return json.loads(content)
        return _call

    return await cascade_api_call(
        _make_call,
        models=MODEL_CASCADE["analysis"],
        timeout=90,
        required_keys=["participant_scores"],
        fallback=fallback,
        label="generate_flags_and_scores",
    )


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
    vision_models = MODEL_CASCADE["vision"][:3]  # gpt-4o, gpt-4.1, gpt-4o-mini
    model_tasks = [
        _analyze_frame_with_model(frame_path, timestamp, m, known_names, camera_on_names)
        for m in vision_models
    ]

    results = await asyncio.gather(*model_tasks, return_exceptions=True)

    valid_results = []
    for i, result in enumerate(results):
        model_names = vision_models
        if isinstance(result, Exception):
            continue
        if isinstance(result, dict) and "participants" in result:
            valid_results.append(result)

    if not valid_results:
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
            **_model_kwargs(model, max_tokens=2500, temperature=0.3),
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
                "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "No visual data available"},
                "gestures": ["still posture"], "reactions": ["neutral gaze"],
            }]
        },
        label=f"frame_analysis_{model}",
    )


async def analyze_voice_signal_dual(transcript_chunk: str) -> dict:
    """Dual-model voice signal analysis (GPT-4o-mini + GPT-4.1) with consensus."""
    from .voting_engine import majority_vote, trimmed_mean

    results = await asyncio.gather(
        analyze_voice_signal(transcript_chunk),
        _analyze_voice_with_model(transcript_chunk, CHAT_MODEL),
        return_exceptions=True,
    )

    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) < 2:
        return valid[0] if valid else {"tone": "neutral", "pace": "moderate", "energy": 50, "dynamics": ""}

    tone_winner, _ = majority_vote([r.get("tone", "neutral") for r in valid])
    pace_winner, _ = majority_vote([r.get("pace", "moderate") for r in valid])
    energy_val, _ = trimmed_mean([float(r.get("energy", 50)) for r in valid])

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
        analyze_words_signal(transcript_chunk, timestamp),
        _analyze_words_with_model(transcript_chunk, timestamp, CHAT_MODEL),
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
