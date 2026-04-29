"""Audio preprocessing: normalization, resampling, silence removal."""
import numpy as np
import librosa
from typing import Tuple

class AudioPreprocessor:
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
    
    def preprocess(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Normalize only if too quiet or clipping
        peak = np.max(np.abs(audio))
        if peak > 0.99:
            # Prevent clipping
            audio = audio * (0.95 / peak)
        elif peak < 0.1:
            # Boost if very quiet
            audio = audio * (0.5 / peak) if peak > 0 else audio
            
        return audio.astype(np.float32), self.target_sr
