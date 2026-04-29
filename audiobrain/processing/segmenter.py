"""
Audio segmentation module.
Splits audio into segments with energy analysis for intelligent filtering.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""
    audio: np.ndarray
    energy: float
    start_sample: int
    end_sample: int


class AudioSegmenter:
    """Segments audio files and calculates energy metrics."""
    
    def __init__(self, segment_duration: float = 1.0, sample_rate: int = 22050):
        """
        Initialize segmenter.
        
        Args:
            segment_duration: Duration of each segment in seconds.
            sample_rate: Sample rate of audio.
        """
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.samples_per_segment = int(segment_duration * sample_rate)
    
    def segment(self, audio: np.ndarray, min_energy: float = 0.01) -> List[AudioSegment]:
        """
        Split audio into segments, filtering by energy.
        
        Args:
            audio: Audio waveform.
            min_energy: Minimum RMS energy to include segment.
            
        Returns:
            List of AudioSegment objects.
        """
        segments = []
        num_segments = len(audio) // self.samples_per_segment
        
        for i in range(num_segments):
            start = i * self.samples_per_segment
            end = start + self.samples_per_segment
            
            segment_audio = audio[start:end]
            energy = self.calculate_energy(segment_audio)
            
            # Skip low-energy segments (silence)
            if energy >= min_energy:
                segments.append(AudioSegment(
                    audio=segment_audio,
                    energy=energy,
                    start_sample=start,
                    end_sample=end
                ))
        
        return segments
    
    @staticmethod
    def calculate_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy of audio segment."""
        return np.sqrt(np.mean(audio ** 2))
    
    def get_total_segments(self, audio_length: int) -> int:
        """Get total number of segments for given audio length."""
        return audio_length // self.samples_per_segment
