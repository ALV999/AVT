"""
Audio validation utilities to ensure processing will succeed.
"""

import os
import librosa
from pathlib import Path
from typing import List, Tuple


class AudioValidator:
    """Validates audio files before processing to prevent pipeline failures."""
    
    @staticmethod
    def validate_file(filepath: str) -> Tuple[bool, str]:
        """
        Validate a single audio file.
        
        Returns:
            (is_valid, error_message)
        """
        path = Path(filepath)
        
        # Check file exists
        if not path.exists():
            return False, f"File not found: {filepath}"
        
        # Check extension
        if path.suffix.lower() not in ['.wav', '.flac', '.mp3', '.ogg']:
            return False, f"Unsupported format: {path.suffix}"
        
        # Try to load and check properties
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=True, duration=1.0)
            
            if len(audio) == 0:
                return False, "File is empty or silent"
            
            # Check for corruption (NaN or Inf values)
            if not all(librosa.util.valid_audio(audio, mono=False)):
                return False, "Audio contains invalid values (NaN/Inf)"
                
        except Exception as e:
            return False, f"Failed to load file: {str(e)}"
        
        return True, ""
    
    @staticmethod
    def validate_files(filepaths: List[str], min_total_duration: float = 30.0) -> Tuple[List[str], List[str]]:
        """
        Validate a list of audio files and check total duration.
        
        Returns:
            (valid_files, errors)
        """
        valid_files = []
        errors = []
        total_duration = 0.0
        
        print(f"\n🔍 Validating {len(filepaths)} input files...")
        
        for filepath in filepaths:
            is_valid, error = AudioValidator.validate_file(filepath)
            
            if is_valid:
                # Get full duration
                try:
                    audio, sr = librosa.load(filepath, sr=None, mono=True)
                    duration = len(audio) / sr
                    total_duration += duration
                    valid_files.append(filepath)
                    print(f"  ✅ {Path(filepath).name}: {duration:.2f}s")
                except Exception as e:
                    errors.append(f"{filepath}: {str(e)}")
            else:
                errors.append(f"{filepath}: {error}")
                print(f"  ❌ {Path(filepath).name}: {error}")
        
        # Check total duration
        if total_duration < min_total_duration:
            errors.append(f"Total duration ({total_duration:.2f}s) is less than minimum required ({min_total_duration:.2f}s)")
            print(f"\n⚠️  Warning: Total duration {total_duration:.2f}s < {min_total_duration:.2f}s required")
        
        print(f"\n✓ Validation complete: {len(valid_files)}/{len(filepaths)} files valid")
        print(f"  Total duration: {total_duration:.2f}s")
        
        return valid_files, errors
