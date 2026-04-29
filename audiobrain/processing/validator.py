"""
Audio file validation module.
Ensures input files are valid before processing to prevent runtime errors.
"""

import os
from pathlib import Path
from typing import List
import soundfile as sf


class AudioValidator:
    """Validates audio files before processing."""
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.mp3'}
    MIN_DURATION = 1.0  # Minimum duration in seconds
    
    @staticmethod
    def validate_files(file_paths: List[str]) -> List[str]:
        """
        Validate a list of audio file paths.
        
        Args:
            file_paths: List of file paths to validate.
            
        Returns:
            List of valid file paths.
            
        Raises:
            ValueError: If any file is invalid.
        """
        valid_files = []
        errors = []
        
        for path in file_paths:
            try:
                AudioValidator._validate_file(path)
                valid_files.append(path)
            except Exception as e:
                errors.append(f"{path}: {str(e)}")
        
        if errors:
            error_msg = "Invalid files found:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        return valid_files
    
    @staticmethod
    def _validate_file(file_path: str):
        """Validate a single audio file."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError("File does not exist")
        
        # Check extension
        if path.suffix.lower() not in AudioValidator.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{path.suffix}'. Supported: {AudioValidator.SUPPORTED_FORMATS}")
        
        # Check if file is readable and get duration
        try:
            info = sf.info(file_path)
            if info.duration < AudioValidator.MIN_DURATION:
                raise ValueError(f"Duration too short ({info.duration:.2f}s). Minimum: {AudioValidator.MIN_DURATION}s")
        except Exception as e:
            raise ValueError(f"Cannot read audio file: {str(e)}")
