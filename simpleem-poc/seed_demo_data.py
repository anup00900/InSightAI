"""
Seed the database with realistic demo data so the dashboard can be demonstrated
without requiring a Core42 API key or actual video processing.
"""
import asyncio
import json
import os
import uuid
import random
import aiosqlite

DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")

PARTICIPANTS = [
    {"name": "Sarah Chen", "role": "Product Lead"},
    {"name": "Marcus Johnson", "role": "Engineering Manager"},
    {"name": "Priya Patel", "role": "Data Scientist"},
    {"name": "James Wilson", "role": "Sales Director"},
    {"name": "Emily Rodriguez", "role": "UX Designer"},
]

EMOTIONS = ["happy", "engaged", "neutral", "surprised", "confused", "concerned", "excited", "disengaged"]

COACHING_TEMPLATES = {
    "communication": [
        "Consider pausing more between key points to let your message sink in",
        "Your pace was strong during the introduction but sped up in the Q&A - try to maintain consistency",
        "Use more concrete examples when explaining complex concepts",
        "Mirror the language of other participants to build rapport",
    ],
    "engagement": [
        "Increase eye contact during others' presentations to signal active listening",
        "Ask follow-up questions when others share ideas to boost team engagement",
        "You showed high energy in the first half but it dropped - consider taking short mental breaks",
        "Your reactions were subtle - more visible nodding and acknowledgment helps speakers",
    ],
    "leadership": [
        "Set clearer agendas at the start so participants know what to expect",
        "When facilitating, ensure all voices are heard before moving to the next topic",
        "Summarize action items before transitioning to the next discussion point",
        "Your confidence was evident - leverage that by mentoring quieter participants",
    ],
    "listening": [
        "You interrupted twice during the budget discussion - practice the 3-second rule before responding",
        "Great active listening during the technical deep-dive - maintain that throughout",
        "Take notes when others speak to demonstrate engagement and retain key points",
        "Paraphrase what others say before responding to show understanding",
    ],
}


async def seed():
    # Clean existing data
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Init DB
    from backend.models import init_db
    await init_db()

    video_id = str(uuid.uuid4())
    duration = 1842.0  # ~30 minutes

    async with aiosqlite.connect(DB_PATH) as db:
        # Create video
        await db.execute(
            "INSERT INTO videos (id, name, filename, file_path, duration, status, overall_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, "Q4 Board Strategy Review.mp4", "demo_meeting.mp4", "/demo/path.mp4", duration, "complete", 81.4),
        )

        # Create participants with varied scores
        participant_ids = []
        scores = [
            {"engagement": 92, "sentiment": 0.7, "speaking": 28, "clarity": 88, "rapport": 85, "energy": 90},
            {"engagement": 78, "sentiment": 0.4, "speaking": 22, "clarity": 82, "rapport": 75, "energy": 72},
            {"engagement": 85, "sentiment": 0.6, "speaking": 18, "clarity": 90, "rapport": 80, "energy": 78},
            {"engagement": 64, "sentiment": 0.2, "speaking": 20, "clarity": 70, "rapport": 65, "energy": 60},
            {"engagement": 88, "sentiment": 0.5, "speaking": 12, "clarity": 85, "rapport": 82, "energy": 84},
        ]

        for i, p in enumerate(PARTICIPANTS):
            pid = str(uuid.uuid4())
            participant_ids.append(pid)
            s = scores[i]
            await db.execute(
                """INSERT INTO participants (id, video_id, name, engagement_score, sentiment_score,
                   speaking_pct, clarity_score, rapport_score, energy_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, video_id, p["name"], s["engagement"], s["sentiment"],
                 s["speaking"], s["clarity"], s["rapport"], s["energy"]),
            )

        # Generate emotion timeline (every 3 seconds for ~30 min = ~600 points)
        for t in range(0, int(duration), 3):
            for i, pid in enumerate(participant_ids):
                base_engagement = scores[i]["engagement"]
                # Add natural variation
                variation = random.gauss(0, 8)
                engagement = max(20, min(100, base_engagement + variation))

                # Simulate some interesting patterns
                if 200 < t < 240:  # tension moment
                    engagement = max(20, engagement - 15) if i == 3 else engagement
                if 450 < t < 500:  # excitement spike
                    engagement = min(100, engagement + 12)
                if 1200 < t < 1260:  # disagreement
                    engagement = max(25, engagement - 10) if i in [1, 3] else engagement

                # Pick emotion based on engagement level
                if engagement > 80:
                    emotion = random.choice(["happy", "engaged", "excited"])
                elif engagement > 60:
                    emotion = random.choice(["engaged", "neutral", "surprised"])
                elif engagement > 40:
                    emotion = random.choice(["neutral", "confused", "concerned"])
                else:
                    emotion = random.choice(["disengaged", "confused", "concerned"])

                await db.execute(
                    "INSERT INTO emotions (video_id, participant_id, timestamp, emotion, confidence, engagement) VALUES (?, ?, ?, ?, ?, ?)",
                    (video_id, pid, float(t), emotion, round(random.uniform(0.6, 0.95), 2), round(engagement, 1)),
                )

        # Generate transcript segments
        transcript_texts = [
            (0, 15, "Sarah Chen", "Good morning everyone. Let's kick off our Q4 board strategy review. We have a packed agenda today."),
            (16, 35, "Sarah Chen", "First, I'd like to go through our Q3 performance metrics, then move into our Q4 strategic priorities."),
            (36, 60, "Marcus Johnson", "Before we start, I want to flag that the engineering team has made significant progress on the AI pipeline."),
            (61, 90, "Marcus Johnson", "We've reduced processing time by 40% and our model accuracy is now at 94.2%, which is ahead of our Q3 target."),
            (91, 120, "Priya Patel", "The data team has been instrumental in that improvement. We retrained on 2 million additional samples."),
            (121, 155, "Sarah Chen", "Excellent work. Let's look at the revenue numbers. James, can you walk us through the sales pipeline?"),
            (156, 200, "James Wilson", "Sure. We're tracking at 85% of our annual target. The enterprise segment is strong but SMB has been challenging."),
            (201, 235, "James Wilson", "The main issue is the competitive landscape in the SMB space. Competitors are undercutting on price."),
            (236, 270, "Emily Rodriguez", "From a product perspective, our user research shows that SMB customers want simpler onboarding."),
            (271, 310, "Emily Rodriguez", "I have some mockups for a streamlined onboarding flow that could address this. Can I share my screen?"),
            (311, 350, "Sarah Chen", "Please do. This ties directly into our Q4 product roadmap discussion."),
            (351, 400, "Emily Rodriguez", "So here's the proposed flow. We reduce the onboarding steps from 12 to 5, with AI-assisted setup."),
            (401, 440, "Marcus Johnson", "That's ambitious from an engineering perspective, but I think we can deliver it with the current team."),
            (441, 480, "Priya Patel", "The AI-assisted setup could leverage our existing models. We wouldn't need to build from scratch."),
            (481, 520, "James Wilson", "If we can get this live by mid-Q4, I'm confident we can recover the SMB pipeline gap."),
            (521, 560, "Sarah Chen", "Let's make that a priority. Marcus, can you put together a sprint plan for this?"),
            (561, 600, "Marcus Johnson", "I'll have it ready by end of week. We should also discuss the board's questions about our AI ethics framework."),
            (601, 650, "Sarah Chen", "Good point. Priya, your team has been leading the responsible AI initiative. Where do we stand?"),
            (651, 720, "Priya Patel", "We've completed our bias audit across all models. Results are positive, but I want to highlight some areas we need to monitor."),
            (721, 780, "Priya Patel", "Specifically, our emotion detection models show slightly lower accuracy for underrepresented demographics."),
            (781, 830, "Sarah Chen", "That's critical. What's the remediation plan?"),
            (831, 900, "Priya Patel", "We're collecting more diverse training data and implementing fairness constraints in our model training pipeline."),
            (901, 950, "Marcus Johnson", "The engineering team is also building monitoring dashboards to track fairness metrics in production."),
            (951, 1000, "James Wilson", "From a sales perspective, our enterprise clients have been asking about our AI governance. This work will help."),
            (1001, 1060, "Emily Rodriguez", "I've designed a transparency dashboard for customers. It shows how our AI makes decisions."),
            (1061, 1120, "Sarah Chen", "This is exactly what the board wants to hear. Let's package all of this into a comprehensive update."),
            (1121, 1200, "Sarah Chen", "Moving to Q4 budget. We're requesting a 15% increase for the AI research team. Marcus, can you justify this?"),
            (1201, 1260, "Marcus Johnson", "Absolutely. The ROI on our AI investment has been 3.2x this year. The additional budget goes toward GPU infrastructure and two senior hires."),
            (1261, 1320, "James Wilson", "I'd like to also request additional budget for sales enablement tools. The team needs better demo capabilities."),
            (1321, 1380, "Sarah Chen", "Both are valid requests. Let's present them as a unified technology investment package to the board."),
            (1381, 1440, "Priya Patel", "One more thing - our research team has a breakthrough on real-time video analysis that could be a game changer."),
            (1441, 1500, "Sarah Chen", "Tell us more. This could be a strong differentiator in the market."),
            (1501, 1580, "Priya Patel", "We can now process live video streams with under 200ms latency, while maintaining 92% accuracy on emotion detection."),
            (1581, 1650, "Marcus Johnson", "That's a significant technical achievement. We should consider filing a patent."),
            (1651, 1720, "Emily Rodriguez", "From a UX perspective, real-time feedback during meetings could transform how teams interact."),
            (1721, 1780, "James Wilson", "Enterprise clients would pay premium for this. I already have three accounts that expressed interest."),
            (1781, 1830, "Sarah Chen", "Let's fast-track this. Marcus, add it to the Q4 roadmap. Priya, prepare a technical brief for the board."),
            (1831, 1842, "Sarah Chen", "Great meeting everyone. Let's reconvene next week with the sprint plans and budget proposals ready."),
        ]

        for start, end, speaker, text in transcript_texts:
            # Find participant ID for speaker
            pid = None
            for j, p in enumerate(PARTICIPANTS):
                if p["name"] == speaker:
                    pid = participant_ids[j]
                    break
            await db.execute(
                "INSERT INTO transcript_segments (video_id, participant_id, start_time, end_time, text, speaker) VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, pid, float(start), float(end), text, speaker),
            )

        # Generate flags
        flags = [
            (200, "negative", "Tension detected: James appears disengaged when discussing competitive challenges", "medium"),
            (270, "positive", "Strong collaboration: Emily builds on James's feedback with user research insights", "medium"),
            (440, "positive", "Team alignment: Multiple participants enthusiastically support the onboarding redesign", "high"),
            (721, "negative", "Concern raised: Bias issues in emotion detection models flagged by Priya", "high"),
            (900, "positive", "Resolution: Clear remediation plan proposed with team consensus", "medium"),
            (1200, "negative", "Tension: Budget discussion shows some disagreement between engineering and sales priorities", "medium"),
            (1380, "positive", "Breakthrough announcement: Real-time video analysis capability generates excitement", "high"),
            (1580, "positive", "Strong agreement: All participants align on fast-tracking the real-time feature", "high"),
        ]

        for ts, flag_type, desc, severity in flags:
            await db.execute(
                "INSERT INTO flags (video_id, timestamp, type, description, severity) VALUES (?, ?, ?, ?, ?)",
                (video_id, float(ts), flag_type, desc, severity),
            )

        # Generate coaching
        for i, pid in enumerate(participant_ids):
            categories = list(COACHING_TEMPLATES.keys())
            random.shuffle(categories)
            for j, cat in enumerate(categories[:3]):
                rec = random.choice(COACHING_TEMPLATES[cat])
                await db.execute(
                    "INSERT INTO coaching (video_id, participant_id, recommendation, category, priority) VALUES (?, ?, ?, ?, ?)",
                    (video_id, pid, rec, cat, j + 1),
                )

        # Generate summary
        summary = (
            "The Q4 Board Strategy Review meeting demonstrated strong team alignment on key strategic priorities. "
            "The team reviewed Q3 performance metrics, with engineering reporting a 40% improvement in AI pipeline processing "
            "and 94.2% model accuracy. Sales is tracking at 85% of annual target, with enterprise performing well but SMB "
            "facing competitive pricing pressure.\n\n"
            "A major highlight was the proposed streamlined onboarding flow that could address SMB challenges, reducing steps "
            "from 12 to 5 with AI-assisted setup. The team unanimously supported fast-tracking this initiative. The responsible "
            "AI discussion revealed important bias considerations in emotion detection models, with a clear remediation plan "
            "involving diverse training data and fairness constraints.\n\n"
            "The meeting concluded with an exciting breakthrough announcement on real-time video analysis with sub-200ms latency, "
            "which all participants agreed should be prioritized in Q4. Budget discussions for a 15% AI research increase and "
            "sales enablement tools were consolidated into a unified technology investment proposal for the board."
        )

        await db.execute(
            "INSERT INTO summaries (video_id, summary, key_topics, overall_sentiment) VALUES (?, ?, ?, ?)",
            (
                video_id,
                summary,
                json.dumps(["Q4 Strategy", "AI Pipeline", "SMB Growth", "Onboarding Redesign", "AI Ethics", "Budget Planning", "Real-time Video Analysis"]),
                "positive",
            ),
        )

        await db.commit()

    print(f"Demo data seeded successfully!")
    print(f"Video ID: {video_id}")
    print(f"Participants: {len(PARTICIPANTS)}")
    print(f"Emotion data points: ~{int(duration/3) * len(PARTICIPANTS)}")
    print(f"Transcript segments: {len(transcript_texts)}")
    print(f"Critical flags: {len(flags)}")


if __name__ == "__main__":
    asyncio.run(seed())
