"""Signal processing pipelines for InsightAI v2.

- visual.py: Frame analysis via GPT-4o Vision (55% weight)
- audio.py: Audio feature extraction via librosa (38% weight)
- text.py: Transcription + NLP via Whisper + GPT (7% weight)
- diarization.py: Speaker diarization via pyannote (Phase 3)
"""
