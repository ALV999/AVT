"""Intelligent audio segmentation with RMS energy calculation."""
import numpy as np
from typing import List, Tuple

class AudioSegmenter:
    def __init__(self, segment_duration: float = 1.0, sample_rate: int = 22050):
        self.segment_samples = int(segment_duration * sample_rate)
        self.sr = sample_rate
        # Umbral muy bajo para aceptar casi cualquier sonido no-silencioso
        self.min_energy = 0.0001 
    
    def segment(self, audio: np.ndarray, min_energy: float = None) -> List[Tuple[np.ndarray, float]]:
        if min_energy is None:
            min_energy = self.min_energy
            
        segments = []
        num_segments = len(audio) // self.segment_samples
        
        print(f"  [Segmenter] Audio length: {len(audio)}, Expected segments: {num_segments}, Min Energy: {min_energy}")
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = audio[start:end]
            
            # Calcular energía RMS
            energy = np.sqrt(np.mean(segment ** 2))
            
            if energy >= min_energy:
                segments.append((segment, energy))
            else:
                print(f"    - Segment {i} rejected (Energy: {energy:.6f} < {min_energy})")
                
        if not segments:
            max_eng = np.max([np.sqrt(np.mean(audio[i*self.segment_samples:(i+1)*self.segment_samples]**2)) 
                              for i in range(num_segments)]) if num_segments > 0 else 0
            print(f"  [Segmenter] WARNING: No segments passed. Max energy found: {max_eng:.6f}")
            
        return segments
    
    def get_segment_count(self, audio: np.ndarray) -> int:
        return len(audio) // self.segment_samples
