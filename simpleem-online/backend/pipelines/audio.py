"""Audio signal pipeline — librosa-based audio feature extraction.

Extracts real acoustic features from audio waveform: pitch, volume, speaking rate,
pauses, spectral brightness. Contributes 38% to holistic engagement score.
"""

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from backend.signal_bus import SignalBus, SignalEvent, SignalType

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Extracted audio features for a chunk."""
    duration: float
    pitch_mean: float          # Average F0 in Hz (0 if no pitch detected)
    pitch_std: float           # Pitch variation (std dev of F0)
    volume_energy: float       # RMS energy normalized to 0-100
    volume_dynamics: float     # Dynamic range of volume (0-100)
    speaking_rate: float       # Estimated syllables per second
    pause_count: int           # Number of silence gaps > 0.3s
    pause_ratio: float         # Fraction of time in silence (0-1)
    spectral_centroid: float   # Average spectral centroid (Hz)
    zcr_mean: float            # Mean zero-crossing rate

    @property
    def pitch_variation_normalized(self) -> float:
        """Pitch variation normalized to 0-100. Higher variation = more engaged."""
        if self.pitch_std <= 0:
            return 0
        # Typical speech pitch std is 20-60 Hz
        return min(100, (self.pitch_std / 60.0) * 100)

    @property
    def volume_energy_normalized(self) -> float:
        """Already 0-100."""
        return self.volume_energy

    @property
    def speaking_rate_normalized(self) -> float:
        """Natural speaking rate scores highest (3-5 syl/s). Too fast or slow penalized."""
        if self.speaking_rate <= 0:
            return 0
        # Bell curve centered around 4 syllables/second
        optimal = 4.0
        deviation = abs(self.speaking_rate - optimal)
        return max(0, min(100, 100 - (deviation / optimal) * 60))

    @property
    def pause_pattern_normalized(self) -> float:
        """Natural pauses (10-30% of speech) score high. Too many/few penalized."""
        if self.pause_ratio < 0.05:
            return 70  # Almost no pauses — still ok
        if self.pause_ratio < 0.15:
            return 90
        if self.pause_ratio < 0.30:
            return 80
        if self.pause_ratio < 0.50:
            return 50
        return 20  # Mostly silence — disengaged

    @property
    def spectral_brightness_normalized(self) -> float:
        """Spectral centroid normalized to 0-100. Higher = more alert/aroused."""
        if self.spectral_centroid <= 0:
            return 0
        # Typical speech centroid: 500-3000 Hz
        return min(100, (self.spectral_centroid / 3000.0) * 100)

    @property
    def engagement_score(self) -> float:
        """Weighted audio engagement sub-score (0-100)."""
        return (
            0.30 * self.volume_energy_normalized +
            0.25 * self.pitch_variation_normalized +
            0.20 * self.speaking_rate_normalized +
            0.15 * self.pause_pattern_normalized +
            0.10 * self.spectral_brightness_normalized
        )


class AudioAnalyzer:
    """Extract acoustic features from audio using librosa."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze_file(self, audio_path: str) -> AudioFeatures:
        """Load a WAV file and extract features."""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.analyze_array(y, sr)

    def analyze_array(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract features from a numpy audio array."""
        duration = len(y) / sr
        if duration < 0.1 or np.max(np.abs(y)) < 1e-6:
            return AudioFeatures(
                duration=duration, pitch_mean=0, pitch_std=0,
                volume_energy=0, volume_dynamics=0,
                speaking_rate=0, pause_count=0, pause_ratio=1.0,
                spectral_centroid=0, zcr_mean=0,
            )

        # Pitch (F0) via pyin
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        if len(voiced_f0) == 0:
            voiced_f0 = np.array([0.0])

        pitch_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0
        pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 1 else 0

        # Volume (RMS energy)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms))
        rms_max = float(np.max(rms))
        # Normalize to 0-100 (typical speech RMS is 0.01-0.3)
        volume_energy = min(100, (rms_mean / 0.15) * 100)
        volume_dynamics = min(100, ((rms_max - float(np.min(rms))) / max(rms_max, 1e-6)) * 100)

        # Speaking rate (syllable estimation using onset detection)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        speaking_rate = len(onsets) / max(duration, 0.1)

        # Pause detection (silence segments > 0.3s)
        frame_duration = 512 / sr  # hop_length / sr
        silence_threshold = 0.01
        is_silent = rms < silence_threshold
        pause_count = 0
        pause_frames = 0
        in_pause = False
        current_pause_len = 0
        min_pause_frames = int(0.3 / frame_duration)

        for silent in is_silent:
            if silent:
                current_pause_len += 1
                if not in_pause and current_pause_len >= min_pause_frames:
                    in_pause = True
                    pause_count += 1
                if in_pause:
                    pause_frames += 1
            else:
                in_pause = False
                current_pause_len = 0

        total_frames = len(rms)
        pause_ratio = pause_frames / max(total_frames, 1)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = float(np.mean(centroid))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        return AudioFeatures(
            duration=duration,
            pitch_mean=round(pitch_mean, 1),
            pitch_std=round(pitch_std, 1),
            volume_energy=round(volume_energy, 1),
            volume_dynamics=round(volume_dynamics, 1),
            speaking_rate=round(speaking_rate, 2),
            pause_count=pause_count,
            pause_ratio=round(pause_ratio, 3),
            spectral_centroid=round(spectral_centroid, 1),
            zcr_mean=round(zcr_mean, 4),
        )


async def analyze_audio_and_publish(
    audio_chunk_path: str,
    timestamp: float,
    signal_bus: SignalBus,
    participant_id: str = "all",
) -> AudioFeatures:
    """Analyze an audio chunk and publish audio signals to the bus."""
    analyzer = AudioAnalyzer()
    features = analyzer.analyze_file(audio_chunk_path)

    await signal_bus.publish(SignalEvent(
        signal_type=SignalType.AUDIO,
        participant_id=participant_id,
        timestamp=timestamp,
        data={
            "energy": features.engagement_score,
            "pitch_mean": features.pitch_mean,
            "pitch_std": features.pitch_std,
            "volume_energy": features.volume_energy,
            "volume_dynamics": features.volume_dynamics,
            "speaking_rate": features.speaking_rate,
            "pause_count": features.pause_count,
            "pause_ratio": features.pause_ratio,
            "spectral_centroid": features.spectral_centroid,
            "zcr_mean": features.zcr_mean,
        },
    ))

    return features
