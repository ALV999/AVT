"""Audio preprocessing: resampling, gain, EQ character, normalization, stereo, silence trim."""
import numpy as np
import torch
import torchaudio.functional as F
from typing import Tuple, Optional

from audiobrain.processing.config import PreprocessingConfig


class AudioPreprocessor:
    """
    Flexible audio preprocessor with abstract controls.
    
    Applies a chain: gain -> character filter -> resample -> normalize -> stereo -> trim.
    All destructive operations are guarded -- the system won't crash on extreme values.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
    
    def preprocess(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline.
        """
        cfg = self.config
        audio = audio.astype(np.float32).copy()
        
        # 1. Stereo handling
        audio = self._apply_stereo(audio)
        
        # 2. Gain (linear from dB)
        if abs(cfg.gain_db) > 0.01:
            gain_linear = 10 ** (cfg.gain_db / 20.0)
            audio = audio * gain_linear
        
        # 3. Character filter (EQ preset)
        audio = self._apply_character(audio, sr)
        
        # 4. Resample
        if sr != cfg.target_sr:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            resampled = F.resample(waveform, orig_freq=sr, new_freq=cfg.target_sr)
            audio = resampled.squeeze(0).numpy()
            sr = cfg.target_sr
        
        # 5. Normalize
        audio = self._apply_normalize(audio)
        
        # 6. Silence trim
        if cfg.trim_silence:
            audio = self._trim_silence(audio, sr)
        
        # Final safety: clip to [-1, 1] and ensure float32
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32), sr
    
    def _apply_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Handle stereo/multi-channel based on config."""
        mode = self.config.stereo_mode
        
        if audio.ndim == 1:
            return audio
        
        if mode == 'mono':
            return audio.mean(axis=0) if audio.ndim == 2 else audio
        elif mode == 'left':
            return audio[0] if audio.ndim == 2 else audio
        elif mode == 'right':
            return audio[1] if audio.ndim == 2 and audio.shape[0] > 1 else audio[-1]
        else:
            return audio
    
    def _apply_character(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply abstract EQ character using biquad filters."""
        char = self.config.character
        if char == 'raw':
            return audio
        
        wf = torch.from_numpy(audio.copy()).float().unsqueeze(0)
        
        if char == 'warm':
            bass = F.lowpass_biquad(wf, sr, cutoff_freq=300, Q=0.7)
            mids = F.highpass_biquad(F.lowpass_biquad(wf, sr, cutoff_freq=6000, Q=0.5), sr, cutoff_freq=300, Q=0.7)
            return (bass * 1.3 + mids * 0.7).squeeze(0).numpy()
        elif char == 'bright':
            return F.highpass_biquad(wf, sr, cutoff_freq=8000, Q=0.5).squeeze(0).numpy() * 0.5 + wf.squeeze(0).numpy()
        elif char == 'dark':
            return F.lowpass_biquad(wf, sr, cutoff_freq=4000, Q=1.0).squeeze(0).numpy()
        elif char == 'airy':
            hpf = F.highpass_biquad(wf, sr, cutoff_freq=200, Q=0.5)
            return hpf.squeeze(0).numpy()
        
        return audio
    
    def _apply_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Apply normalization based on config."""
        mode = self.config.norm_mode
        
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio
        
        if mode == 'peak':
            target_peak = 10 ** (-1.0 / 20.0)
            return audio * (target_peak / peak)
        elif mode == 'rms':
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 1e-10:
                return audio
            target_rms = 10 ** (-16.0 / 20.0)
            return audio * (target_rms / rms)
        else:
            if peak > 1.0:
                return audio * (0.99 / peak)
            return audio
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove leading and trailing silence below threshold."""
        threshold = 10 ** (self.config.trim_threshold_db / 20.0)
        rms_window = max(1, int(sr * 0.01))
        
        energy = np.array([
            np.sqrt(np.mean(audio[max(0, i-rms_window//2): min(len(audio), i+rms_window//2)] ** 2))
            for i in range(0, len(audio), rms_window)
        ])
        
        above = energy > threshold
        if not above.any():
            return audio
        
        start_idx = np.argmax(above) * rms_window
        end_idx = (len(above) - np.argmax(above[::-1])) * rms_window
        
        return audio[start_idx:min(end_idx, len(audio))]
