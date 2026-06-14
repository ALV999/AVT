"""
Configuration for audio generation and preprocessing parameters.
Controls creativity, density, generation mode, and preprocessing.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class GenerationConfig:
    """
    Configuration for controlling the audio generation process.
    """
    
    temperature: float = 0.5
    density: float = 1.0
    mode: Literal['fluid', 'glitch', 'evolving'] = 'fluid'
    seed: int | None = None
    segment_duration: float = 1.0
    min_segment_energy: float = 0.01
    
    def __post_init__(self):
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError("Density must be between 0.0 and 1.0")
        if not 0.5 <= self.segment_duration <= 2.0:
            raise ValueError("Segment duration must be between 0.5 and 2.0 seconds")
        if not 0.0 <= self.min_segment_energy <= 1.0:
            raise ValueError("Min segment energy must be between 0.0 and 1.0")
        
        if self.seed is not None:
            import random
            import numpy as np
            import torch
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)


@dataclass
class PreprocessingConfig:
    """
    Audio preprocessing configuration with abstract controls.
    All parameters have hard safety limits -- values outside the valid
    range are silently clamped to prevent system breakage.
    
    Attributes:
        target_sr: Target sample rate (8000-96000 Hz).
        gain_db: Input gain in dB (-24 to +12).
        character: EQ profile: raw, warm, bright, dark, airy.
        norm_mode: Normalization: peak, rms, none.
        stereo_mode: Channel handling: mono, left, right, stereo.
        trim_silence: Remove leading/trailing silence.
        trim_threshold_db: Silence threshold in dB (-96 to -6).
    """
    
    target_sr: int = 32000
    gain_db: float = 0.0
    character: Literal['raw', 'warm', 'bright', 'dark', 'airy'] = 'raw'
    norm_mode: Literal['peak', 'rms', 'none'] = 'peak'
    stereo_mode: Literal['mono', 'left', 'right', 'stereo'] = 'mono'
    trim_silence: bool = False
    trim_threshold_db: float = -60.0
    
    def __post_init__(self):
        """Safety clamp all parameters to prevent system breakage."""
        self.target_sr = max(8000, min(96000, int(self.target_sr)))
        self.gain_db = max(-24.0, min(12.0, float(self.gain_db)))
        self.trim_threshold_db = max(-96.0, min(-6.0, float(self.trim_threshold_db)))
