import pytest
from backend.pipelines.diarization import (
    DiarizationSegment,
    assign_speakers_to_transcript,
)


class TestAssignSpeakers:
    def test_basic_assignment(self):
        diarization = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ]
        transcript = [
            {"start": 1.0, "end": 3.0, "text": "Hello", "speaker": ""},
            {"start": 6.0, "end": 8.0, "text": "Hi there", "speaker": ""},
        ]

        result = assign_speakers_to_transcript(transcript, diarization)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_overlap_picks_majority(self):
        diarization = [
            DiarizationSegment(start=0.0, end=3.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=3.0, end=6.0, speaker="SPEAKER_01"),
        ]
        transcript = [
            {"start": 2.0, "end": 5.0, "text": "Some overlap", "speaker": ""},
        ]

        result = assign_speakers_to_transcript(transcript, diarization)
        # 2-3s with SPEAKER_00 (1s), 3-5s with SPEAKER_01 (2s) -> SPEAKER_01 wins
        assert result[0]["speaker"] == "Speaker 2"

    def test_empty_diarization(self):
        transcript = [
            {"start": 1.0, "end": 3.0, "text": "Hello", "speaker": ""},
        ]
        result = assign_speakers_to_transcript(transcript, [])
        assert result[0]["speaker"] == ""  # Unchanged
