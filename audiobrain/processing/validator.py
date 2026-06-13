"""Audio file validation before processing."""
import os
import soundfile as sf
from typing import List, Tuple

class AudioValidator:
    @staticmethod
    def validate_files(file_paths: List[str], min_duration: float = 1.0) -> Tuple[bool, str]:
        if not file_paths:
            return False, "No files provided"
        
        valid_count = 0
        for path in file_paths:
            if not os.path.exists(path):
                return False, f"File not found: {path}"
            if not path.lower().endswith('.wav'):
                return False, f"Invalid format (need .wav): {path}"
            try:
                info = sf.info(path)
                duration = info.duration
                if duration < min_duration:
                    return False, f"File too short ({duration:.2f}s < {min_duration}s): {path}"
                valid_count += 1
            except Exception as e:
                return False, f"Cannot read {path}: {str(e)}"
        
        return True, f"Validated {valid_count}/{len(file_paths)} files successfully"
