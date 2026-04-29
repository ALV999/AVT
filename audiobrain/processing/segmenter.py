"""Intelligent audio segmentation with RMS energy calculation."""
import numpy as np
from typing import List, Tuple

class AudioSegmenter:
    def __init__(self, segment_duration: float = 1.0, sample_rate: int = 22050):
        self.segment_samples = int(segment_duration * sample_rate)
        self.sr = sample_rate
    
    def segment(self, audio: np.ndarray, min_energy: float = 0.001) -> List[Tuple[np.ndarray, float]]:
        """
        Divide el audio en segmentos.
        min_energy bajado a 0.001 para aceptar señales más suaves.
        """
        segments = []
        num_segments = len(audio) // self.segment_samples
        
        print(f"  [Segmenter] Audio length: {len(audio)/self.sr:.2f}s, Expected segments: {num_segments}")
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = audio[start:end]
            
            # Calcular energía RMS
            energy = np.sqrt(np.mean(segment ** 2))
            
            # Debug: mostrar energía de los primeros 5 segmentos
            if i < 5:
                print(f"    Segment {i}: Energy={energy:.6f} (Threshold={min_energy})")
            
            if energy >= min_energy:
                segments.append((segment, energy))
            else:
                print(f"    Segment {i}: Skipped (Too quiet)")
        
        print(f"  [Segmenter] Valid segments extracted: {len(segments)}/{num_segments}")
        return segments
    
    def get_segment_count(self, audio: np.ndarray) -> int:
        return len(audio) // self.segment_samples
