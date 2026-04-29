"""Audio file validation before processing."""
import os
import librosa
from typing import List, Tuple

class AudioValidator:
    @staticmethod
    def validate_files(file_paths: List[str], min_duration: float = 1.0) -> Tuple[bool, str]:
        """
        Validates a list of audio files.
        
        Args:
            file_paths: List of paths to audio files.
            min_duration: Minimum duration in seconds required for each file.
            
        Returns:
            Tuple of (is_valid, message).
        """
        if not file_paths:
            return False, "No files provided"
        
        valid_count = 0
        for path in file_paths:
            if not os.path.exists(path):
                return False, f"File not found: {path}"
            
            if not path.lower().endswith('.wav'):
                return False, f"Invalid format (need .wav): {path}"
            
            try:
                # Load just 1 second to check readability and estimate SR
                audio, sr = librosa.load(path, sr=None, mono=True, duration=1.0)
                
                # If the file is shorter than 1s, librosa returns less samples
                duration_sampled = len(audio) / sr
                
                # If we got less than requested, the file is short
                if duration_sampled < 1.0:
                    if duration_sampled < min_duration:
                        return False, f"File too short ({duration_sampled:.2f}s < {min_duration}s): {path}"
                
                valid_count += 1
                
            except Exception as e:
                return False, f"Cannot read {path}: {str(e)}"
        
        return True, f"Validated {valid_count}/{len(file_paths)} files successfully"
