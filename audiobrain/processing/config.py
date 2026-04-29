"""
Configuration for audio generation parameters.
Controls creativity, density, generation mode, and reproducibility.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class GenerationConfig:
    """
    Configuration for controlling the audio generation process.
    
    Attributes:
        temperature: Controls randomness in segment selection (0.0-1.0).
                    0.0 = always pick best match (deterministic)
                    1.0 = pick randomly from top k neighbors (creative)
        density: Controls how many segments to use per time unit (0.0-1.0).
                 0.0 = sparse, glitchy output with gaps
                 1.0 = dense, continuous output
        mode: Generation strategy.
              'fluid' = smooth transitions, prioritize similarity
              'glitch' = abrupt changes, allow discontinuities
              'evolving' = progressive changes over time
        seed: Random seed for reproducibility. None = random each time.
        segment_duration: Duration of each segment in seconds (0.5-2.0).
        min_segment_energy: Minimum RMS energy to include a segment (0.0-1.0).
    """
    
    temperature: float = 0.5
    density: float = 1.0
    mode: Literal['fluid', 'glitch', 'evolving'] = 'fluid'
    seed: int | None = None
    segment_duration: float = 1.0
    min_segment_energy: float = 0.01
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError("Density must be between 0.0 and 1.0")
        if not 0.5 <= self.segment_duration <= 2.0:
            raise ValueError("Segment duration must be between 0.5 and 2.0 seconds")
        if not 0.0 <= self.min_segment_energy <= 1.0:
            raise ValueError("Min segment energy must be between 0.0 and 1.0")
        
        # Set seed for reproducibility
        if self.seed is not None:
            import random
            import numpy as np
            import torch
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
