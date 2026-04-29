"""
Configuration for audio generation parameters.
Controls creativity, structure, and reproducibility.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import random
import numpy as np
import torch


@dataclass
class GenerationConfig:
    """
    Configuration for the audio generation pipeline.
    
    Parameters:
        temperature: Controls randomness in neighbor selection (0.0 = deterministic, 1.0 = chaotic).
        density: Probability of keeping a segment (0.0 to 1.0). Lower = more sparse/glitchy.
        mode: Generation strategy ('fluid', 'glitch', 'evolving').
        seed: Random seed for reproducibility (None = random).
        segment_duration: Duration of each segment in seconds (0.5 to 2.0).
        crossfade_duration: Duration of crossfade between segments.
        min_input_duration: Minimum total duration required from input files.
        max_output_duration: Maximum duration of generated output (None = unlimited).
    """
    
    temperature: float = 0.5
    density: float = 1.0
    mode: Literal['fluid', 'glitch', 'evolving'] = 'fluid'
    seed: Optional[int] = None
    segment_duration: float = 1.0
    crossfade_duration: float = 0.1
    min_input_duration: float = 30.0
    max_output_duration: Optional[float] = 60.0
    
    def __post_init__(self):
        """Validate and apply configuration."""
        # Validate ranges
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError("Density must be between 0.0 and 1.0")
        if not 0.5 <= self.segment_duration <= 2.0:
            raise ValueError("Segment duration must be between 0.5s and 2.0s")
        
        # Set seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        
        # Adjust crossfade based on mode
        if self.mode == 'glitch':
            self.crossfade_duration = 0.01  # Almost no crossfade
        elif self.mode == 'fluid':
            self.crossfade_duration = max(0.1, self.segment_duration * 0.2)
        
        print(f"⚙️  Generation Config: Mode={self.mode}, Temp={self.temperature}, Density={self.density}")
        if self.seed is not None:
            print(f"   Seed set to: {self.seed}")
