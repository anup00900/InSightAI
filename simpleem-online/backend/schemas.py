from pydantic import BaseModel
from typing import Optional


class VideoOut(BaseModel):
    id: str
    name: str
    filename: str
    duration: float
    status: str
    created_at: str
    overall_score: float = 0.0
    source_url: str = ""


class ParticipantOut(BaseModel):
    id: str
    video_id: str
    name: str
    engagement_score: float
    sentiment_score: float
    speaking_pct: float
    clarity_score: float
    rapport_score: float
    energy_score: float


class EmotionPoint(BaseModel):
    timestamp: float
    emotion: str
    confidence: float
    engagement: float
    participant_id: str


class TranscriptSegment(BaseModel):
    start_time: float
    end_time: float
    text: str
    speaker: str
    participant_id: Optional[str] = None


class Flag(BaseModel):
    id: int
    timestamp: float
    type: str
    description: str
    severity: str


class CoachingItem(BaseModel):
    id: int
    recommendation: str
    category: str
    priority: int


class SummaryOut(BaseModel):
    summary: str
    key_topics: list[str]
    overall_sentiment: str


class SignalSnapshot(BaseModel):
    timestamp: float
    participant_id: str
    emotion: str
    emotion_confidence: float
    engagement: float
    posture: str
    openness: str
    leaning: str
    mirroring: bool
    body_language_desc: str
    gestures: list[str]
    reactions: list[str]


class VoiceSignal(BaseModel):
    timestamp: float
    tone: str
    pace: str
    energy: float
    dynamics: str


class WordSignal(BaseModel):
    timestamp: float
    sentiment: str
    sentiment_score: float
    keywords: list[str]
    key_phrases: list[str]


class PersonalitySignal(BaseModel):
    participant_id: str
    traits: list[str]
    communication_style: str
    confidence_level: str


class CorrelationPattern(BaseModel):
    signal_pair: list[str]
    pattern: str
    confidence: float
    implication: str


class AnalysisResults(BaseModel):
    video: VideoOut
    participants: list[ParticipantOut]
    emotions: list[EmotionPoint]
    transcript: list[TranscriptSegment]
    flags: list[Flag]
    summary: Optional[SummaryOut] = None


class ImportUrlRequest(BaseModel):
    url: str


class MeetingAnalytics(BaseModel):
    video_id: str
    video_name: str
    avg_engagement: float
    avg_visual_engagement: float
    avg_audio_engagement: float
    avg_verbal_engagement: float
    participant_count: int
    duration: float
    overall_sentiment: str
    meeting_date: str


class JoinMeetingRequest(BaseModel):
    meeting_url: str
    bot_name: str = "InsightAI Bot"


class RenameParticipantRequest(BaseModel):
    name: str
