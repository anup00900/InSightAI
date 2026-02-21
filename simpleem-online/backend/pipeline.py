import os
import uuid
import json
import asyncio
import aiosqlite
from .models import DB_PATH
from .video_processor import extract_audio, extract_frames, get_video_duration, is_video_file
from .core42_client import (
    transcribe_audio,
    analyze_batch_frames,
    generate_coaching,
    generate_summary_and_flags,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


async def run_analysis_pipeline(video_id: str, file_path: str):
    """Run the full analysis pipeline for an uploaded video/audio file."""
    output_dir = os.path.join(PROCESSED_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET status = ? WHERE id = ?", ("processing", video_id)
        )
        await db.commit()

    try:
        # Step 1: Get duration
        duration = await get_video_duration(file_path)

        # Step 2: Extract audio (may be None if video has no audio track)
        audio_path = await extract_audio(file_path, output_dir)

        # Step 3: Extract frames (if video, not audio-only)
        frames = []
        if is_video_file(os.path.basename(file_path)):
            frames = await extract_frames(file_path, output_dir, fps=1.0)

        # Step 4: Transcribe audio (skip if no audio)
        segments = []
        if audio_path:
            transcript_data = await transcribe_audio(audio_path)
            segments = transcript_data.get("segments", [])

        # Step 5: Analyze frames for emotions (if we have frames)
        emotion_results = []
        if frames:
            emotion_results = await analyze_batch_frames(
                frames, batch_size=5
            )

        # Step 6: Build transcript text and emotions summary for GPT
        transcript_text = "\n".join(
            f"[{s.get('start', 0):.1f}s - {s.get('end', 0):.1f}s]: {s.get('text', '')}"
            for s in segments
        )

        # Collect participant labels and count appearances across frames
        from collections import Counter
        label_counts = Counter()
        total_frames = len(emotion_results)
        for er in emotion_results:
            for p in er.get("participants", []):
                label_counts[p.get("label", "Person 1")] += 1

        # Only keep participants that appear in at least 20% of frames
        min_appearances = max(1, int(total_frames * 0.2))
        participant_labels = {
            label for label, count in label_counts.items()
            if count >= min_appearances
        }
        if not participant_labels:
            participant_labels = {label_counts.most_common(1)[0][0]} if label_counts else {"Person 1"}
        participant_names = sorted(participant_labels)

        emotions_summary = json.dumps(
            [
                {
                    "timestamp": er.get("timestamp", 0),
                    "participants": er.get("participants", []),
                }
                for er in emotion_results[:30]  # Limit to avoid token overflow
            ],
            indent=1,
        )

        # Step 7: Generate summary, flags, and participant scores
        analysis = await generate_summary_and_flags(
            transcript_text[:8000],  # Limit transcript length
            emotions_summary[:4000],
            participant_names,
        )

        # Step 8: Store everything in database
        async with aiosqlite.connect(DB_PATH) as db:
            # Update video duration and overall score
            avg_engagement = 0
            p_scores = analysis.get("participant_scores", [])
            if p_scores:
                avg_engagement = sum(p.get("engagement", 70) for p in p_scores) / len(p_scores)

            await db.execute(
                "UPDATE videos SET duration = ?, overall_score = ?, status = ? WHERE id = ?",
                (duration, avg_engagement, "complete", video_id),
            )

            # Store participants
            participant_id_map = {}
            for ps in p_scores:
                pid = str(uuid.uuid4())
                name = ps.get("name", "Unknown")
                participant_id_map[name] = pid
                speak = ps.get("speaking_pct", 25)
                raw_eng = ps.get("engagement", 70)
                raw_sent = max(0, min(100, (ps.get("sentiment", 0.3) + 1) * 50))
                raw_clar = ps.get("clarity", 70)
                raw_rap = ps.get("rapport", 70)
                raw_ener = ps.get("energy", 70)

                if speak < 1.0:
                    raw_eng = min(raw_eng, 15)
                    raw_sent = 50
                    raw_clar = min(raw_clar, 15)
                    raw_rap = min(raw_rap, 15)
                    raw_ener = min(raw_ener, 15)
                elif speak < 5.0:
                    cap = 30 + speak * 4
                    raw_eng = min(raw_eng, cap)
                    raw_clar = min(raw_clar, cap)
                    raw_rap = min(raw_rap, cap)
                    raw_ener = min(raw_ener, cap)

                await db.execute(
                    """INSERT INTO participants (id, video_id, name, engagement_score,
                       sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid, video_id, name,
                        raw_eng, raw_sent, speak,
                        raw_clar, raw_rap, raw_ener,
                    ),
                )

            # Store emotions timeline
            for er in emotion_results:
                ts = er.get("timestamp", 0)
                for p in er.get("participants", []):
                    label = p.get("label", "Person 1")
                    pid = participant_id_map.get(label)
                    if not pid:
                        # Map to closest matching participant
                        pid = list(participant_id_map.values())[0] if participant_id_map else "unknown"
                    await db.execute(
                        """INSERT INTO emotions (video_id, participant_id, timestamp, emotion, confidence, engagement)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            video_id, pid, ts,
                            p.get("emotion", "neutral"),
                            p.get("confidence", 0.5),
                            p.get("engagement", 50),
                        ),
                    )

            # Store transcript
            for seg in segments:
                await db.execute(
                    """INSERT INTO transcript_segments (video_id, start_time, end_time, text, speaker)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        video_id,
                        seg.get("start", 0),
                        seg.get("end", 0),
                        seg.get("text", ""),
                        seg.get("speaker", ""),
                    ),
                )

            # Store flags
            for flag in analysis.get("flags", []):
                await db.execute(
                    """INSERT INTO flags (video_id, timestamp, type, description, severity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        video_id,
                        flag.get("timestamp", 0),
                        flag.get("type", "neutral"),
                        flag.get("description", ""),
                        flag.get("severity", "medium"),
                    ),
                )

            # Store summary
            await db.execute(
                """INSERT OR REPLACE INTO summaries (video_id, summary, key_topics, overall_sentiment)
                   VALUES (?, ?, ?, ?)""",
                (
                    video_id,
                    analysis.get("summary", ""),
                    json.dumps(analysis.get("key_topics", [])),
                    analysis.get("overall_sentiment", "neutral"),
                ),
            )

            await db.commit()

            # Step 9: Generate coaching for each participant
            for name, pid in participant_id_map.items():
                # Get participant-specific emotion summary
                p_emotions = [
                    er for er in emotion_results
                    if any(p.get("label") == name for p in er.get("participants", []))
                ]
                p_emo_summary = json.dumps(p_emotions[:15], indent=1)

                coaching_items = await generate_coaching(
                    transcript_text[:4000], p_emo_summary[:2000], name
                )
                for item in coaching_items:
                    await db.execute(
                        """INSERT INTO coaching (video_id, participant_id, recommendation, category, priority)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            video_id, pid,
                            item.get("recommendation", ""),
                            item.get("category", "general"),
                            item.get("priority", 3),
                        ),
                    )
            await db.commit()

    except Exception as e:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE videos SET status = ? WHERE id = ?",
                (f"error: {str(e)[:200]}", video_id),
            )
            await db.commit()
        raise
