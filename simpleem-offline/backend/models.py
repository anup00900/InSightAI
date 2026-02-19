import aiosqlite
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                duration REAL DEFAULT 0,
                status TEXT DEFAULT 'uploaded',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                overall_score REAL DEFAULT 0,
                source_url TEXT DEFAULT '',
                wizard_step TEXT DEFAULT 'upload',
                transcribe_progress REAL DEFAULT 0,
                analyze_progress REAL DEFAULT 0,
                progress_detail TEXT DEFAULT ''
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                name TEXT NOT NULL,
                engagement_score REAL DEFAULT 0,
                sentiment_score REAL DEFAULT 0,
                speaking_pct REAL DEFAULT 0,
                clarity_score REAL DEFAULT 0,
                rapport_score REAL DEFAULT 0,
                energy_score REAL DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                engagement REAL DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos(id),
                FOREIGN KEY (participant_id) REFERENCES participants(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                type TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'medium',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS coaching (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                priority INTEGER DEFAULT 1,
                FOREIGN KEY (video_id) REFERENCES videos(id),
                FOREIGN KEY (participant_id) REFERENCES participants(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                key_topics TEXT DEFAULT '[]',
                overall_sentiment TEXT DEFAULT 'neutral',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS meeting_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                action_items TEXT DEFAULT '[]',
                decisions TEXT DEFAULT '[]',
                follow_ups TEXT DEFAULT '[]',
                key_questions TEXT DEFAULT '[]',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signal_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                emotion TEXT DEFAULT 'neutral',
                emotion_confidence REAL DEFAULT 0,
                engagement REAL DEFAULT 0,
                posture TEXT DEFAULT 'upright',
                openness TEXT DEFAULT 'mixed',
                leaning TEXT DEFAULT 'neutral',
                mirroring INTEGER DEFAULT 0,
                body_language_desc TEXT DEFAULT '',
                gestures TEXT DEFAULT '[]',
                reactions TEXT DEFAULT '[]',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS voice_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tone TEXT DEFAULT 'neutral',
                pace TEXT DEFAULT 'moderate',
                energy REAL DEFAULT 50,
                dynamics TEXT DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS word_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                sentiment TEXT DEFAULT 'neutral',
                sentiment_score REAL DEFAULT 0.5,
                keywords TEXT DEFAULT '[]',
                key_phrases TEXT DEFAULT '[]',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS personality_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                traits TEXT DEFAULT '[]',
                communication_style TEXT DEFAULT '',
                confidence_level TEXT DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pre_analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                signal_pair TEXT DEFAULT '[]',
                pattern TEXT DEFAULT '',
                confidence REAL DEFAULT 0,
                implication TEXT DEFAULT '',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS meeting_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                avg_engagement REAL DEFAULT 0,
                avg_visual_engagement REAL DEFAULT 0,
                avg_audio_engagement REAL DEFAULT 0,
                avg_verbal_engagement REAL DEFAULT 0,
                participant_count INTEGER DEFAULT 0,
                duration REAL DEFAULT 0,
                overall_sentiment TEXT DEFAULT 'neutral',
                meeting_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS face_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_embedding TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS face_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_label TEXT NOT NULL,
                profile_id TEXT,
                custom_name TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
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
        await db.execute("""
            CREATE TABLE IF NOT EXISTS speaker_audio_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                pitch_mean REAL DEFAULT 0,
                pitch_std REAL DEFAULT 0,
                volume_energy REAL DEFAULT 0,
                speaking_rate REAL DEFAULT 0,
                pause_ratio REAL DEFAULT 0,
                spectral_centroid REAL DEFAULT 0,
                engagement_score REAL DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        # Indexes for query performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_emotions_video ON emotions(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_emotions_ts ON emotions(video_id, timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_transcript_video ON transcript_segments(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_signal_snap_video ON signal_snapshots(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_voice_video ON voice_signals(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_word_video ON word_signals(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_participants_video ON participants(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_coaching_video ON coaching(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_flags_video ON flags(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_pre_cache_video ON pre_analysis_cache(video_id, data_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_analytics_video ON meeting_analytics(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_meeting_notes_video ON meeting_notes(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_voting_log_video ON voting_log(video_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_speaker_audio_video ON speaker_audio_features(video_id)")
        # Safe column addition — ignores if already exists
        try:
            await db.execute("ALTER TABLE signal_snapshots ADD COLUMN confidence REAL DEFAULT 1.0")
        except Exception:
            pass  # Column already exists
        await db.commit()


async def get_db():
    return aiosqlite.connect(DB_PATH)
