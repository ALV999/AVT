"""
Audio preprocessing utilities for cleaning and preparing input files.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


class AudioPreprocessor:
    """Preprocesses audio files to ensure consistent quality and format."""
    
    @staticmethod
    def remove_silence(
        audio: np.ndarray, 
        sr: int, 
        top_db: float = 30.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Remove silent portions from audio.
        
        Args:
            audio: Input audio waveform
            sr: Sample rate
            top_db: Threshold in dB below peak to consider as silence
            frame_length: Window length for energy calculation
            hop_length: Hop size for energy calculation
            
        Returns:
            Audio with silence removed
        """
        # Trim leading and trailing silence
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=top_db, 
            frame_length=frame_length, 
            hop_length=hop_length
        )
        return audio_trimmed
    
    @staticmethod
    def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Normalize audio to a target peak amplitude.
        
        Args:
            audio: Input audio waveform
            target_peak: Target peak amplitude (0.0 to 1.0)
            
        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono."""
        if len(audio.shape) == 1:
            return audio
        return librosa.to_mono(audio)
    
    @staticmethod
    def preprocess(
        filepath: str,
        target_sr: int = 22050,
        remove_silent: bool = True,
        normalize_audio: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline for a single file.
        
        Args:
            filepath: Path to audio file
            target_sr: Target sample rate
            remove_silent: Whether to remove silence
            normalize_audio: Whether to normalize
            
        Returns:
            (processed_audio, sample_rate)
        """
        # Load file
        audio, sr = librosa.load(filepath, sr=None, mono=False)
        
        # Convert to mono
        audio = AudioPreprocessor.to_mono(audio)
        
        # Resample if needed
        if sr != target_sr:
            audio = AudioPreprocessor.resample(audio, sr, target_sr)
            sr = target_sr
        
        # Remove silence
        if remove_silent:
            audio = AudioPreprocessor.remove_silence(audio, sr)
        
        # Normalize
        if normalize_audio:
            audio = AudioPreprocessor.normalize(audio)
        
        return audio, sr
