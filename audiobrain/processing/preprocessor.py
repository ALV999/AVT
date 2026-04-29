"""
Audio preprocessing module.
Normalizes, cleans, and prepares audio for feature extraction.
"""

import numpy as np
import librosa
from typing import Tuple


class AudioPreprocessor:
    """Preprocesses audio files for consistent processing."""
    
    def __init__(self, target_sr: int = 22050):
        """
        Initialize preprocessor.
        
        Args:
            target_sr: Target sample rate for all audio.
        """
        self.target_sr = target_sr
    
    def preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess an audio file.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Tuple of (audio waveform, sample rate).
        """
        # Load with target sample rate
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Normalize to [-1, 1] range
        audio = self.normalize(audio)
        
        # Remove extreme silence at start/end
        audio = self.trim_silence(audio)
        
        return audio, self.target_sr
    
    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dBFS."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio
        
        target_rms = 10 ** (target_db / 20.0)
        scale = target_rms / rms
        normalized = audio * scale
        
        # Clip to prevent overflow
        return np.clip(normalized, -1.0, 1.0)
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from start and end of audio."""
        # Find non-silent regions
        mask = np.abs(audio) > threshold
        
        if not mask.any():
            return audio
        
        # Get first and last non-zero indices
        indices = np.where(mask)[0]
        start = indices[0]
        end = indices[-1] + 1
        
        return audio[start:end]
