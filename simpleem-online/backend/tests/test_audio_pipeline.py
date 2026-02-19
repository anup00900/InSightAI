import numpy as np
import pytest
from backend.pipelines.audio import AudioAnalyzer, AudioFeatures


class TestAudioAnalyzer:
    def setup_method(self):
        self.analyzer = AudioAnalyzer()

    def test_analyze_silence(self):
        """Silent audio should produce low engagement."""
        sr = 16000
        duration = 2.0
        silence = np.zeros(int(sr * duration))

        features = self.analyzer.analyze_array(silence, sr)
        assert isinstance(features, AudioFeatures)
        assert features.volume_energy < 10  # Very low for silence
        assert features.speaking_rate == 0  # No speech in silence

    def test_analyze_tone(self):
        """A pure tone should have consistent pitch."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        tone = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200 Hz tone

        features = self.analyzer.analyze_array(tone, sr)
        assert features.pitch_mean > 0  # Should detect pitch
        assert features.volume_energy > 20  # Audible

    def test_engagement_score_range(self):
        """Engagement score should always be 0-100."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        tone = 0.3 * np.sin(2 * np.pi * 300 * t)

        features = self.analyzer.analyze_array(tone, sr)
        score = features.engagement_score
        assert 0 <= score <= 100

    def test_features_from_file(self, tmp_path):
        """Test loading from a WAV file."""
        import soundfile as sf

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.4 * np.sin(2 * np.pi * 250 * t)

        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, sr)

        features = self.analyzer.analyze_file(wav_path)
        assert isinstance(features, AudioFeatures)
        assert features.duration > 1.5
