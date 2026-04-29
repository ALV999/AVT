"""
Audio segmenter for dividing audio into fixed-duration segments.
"""

import numpy as np
from typing import List, Tuple


class AudioSegmenter:
    """Segments audio into fixed-duration chunks with optional overlap."""
    
    @staticmethod
    def segment(
        audio: np.ndarray,
        sr: int,
        segment_duration: float = 1.0,
        hop_duration: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Split audio into segments of fixed duration.
        
        Args:
            audio: Input audio waveform
            sr: Sample rate
            segment_duration: Duration of each segment in seconds
            hop_duration: Hop size in seconds (if None, equals segment_duration = no overlap)
            
        Returns:
            List of audio segments
        """
        if hop_duration is None:
            hop_duration = segment_duration
        
        samples_per_segment = int(segment_duration * sr)
        hop_samples = int(hop_duration * sr)
        
        segments = []
        
        # Slide window across audio
        for start in range(0, len(audio) - samples_per_segment + 1, hop_samples):
            end = start + samples_per_segment
            segment = audio[start:end]
            
            # Only add if we have a full segment
            if len(segment) == samples_per_segment:
                segments.append(segment)
        
        return segments
    
    @staticmethod
    def segment_with_features(
        audio: np.ndarray,
        sr: int,
        segment_duration: float = 1.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segment audio and calculate RMS energy for each segment.
        
        Args:
            audio: Input audio waveform
            sr: Sample rate
            segment_duration: Duration of each segment
            
        Returns:
            (segments, energies) where energies can be used for 'evolving' mode
        """
        segments = AudioSegmenter.segment(audio, sr, segment_duration)
        
        # Calculate RMS energy for each segment
        energies = []
        for seg in segments:
            rms = np.sqrt(np.mean(seg ** 2))
            energies.append(rms)
        
        return segments, energies
    
    @staticmethod
    def get_total_segments(audio_length_samples: int, sr: int, segment_duration: float) -> int:
        """Calculate how many segments can be extracted from audio of given length."""
        samples_per_segment = int(segment_duration * sr)
        return audio_length_samples // samples_per_segment
