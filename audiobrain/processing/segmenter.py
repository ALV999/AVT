"""Intelligent audio segmentation with RMS energy calculation."""
import numpy as np
from typing import List, Tuple

class AudioSegmenter:
    def __init__(self, segment_duration: float = 1.0, sample_rate: int = 22050):
        self.segment_samples = int(segment_duration * sample_rate)
        self.sr = sample_rate
    
    def segment(self, audio: np.ndarray, min_energy: float = 0.005) -> List[Tuple[np.ndarray, float]]:
        """
        Segments audio and filters by energy.
        min_energy: Absolute RMS threshold (0.005 works well for normalized audio).
        """
        segments = []
        num_segments = len(audio) // self.segment_samples
        
        print(f"  Segmenter: Input length {len(audio)}, expecting ~{num_segments} segments")
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = audio[start:end]
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(segment ** 2))
            
            # Debug first few segments
            if i < 3:
                print(f"    Seg {i}: RMS={energy:.4f}, Max={np.max(np.abs(segment)):.4f}")
            
            if energy >= min_energy:
                segments.append((segment, energy))
            else:
                if i < 3:
                    print(f"    -> Skipped (below threshold {min_energy})")
        
        print(f"  Segmenter: Kept {len(segments)}/{num_segments} valid segments")
        return segments
    
    def get_segment_count(self, audio: np.ndarray) -> int:
        return len(audio) // self.segment_samples
