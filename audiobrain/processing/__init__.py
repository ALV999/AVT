"""
Audio processing module.
Provides validation, preprocessing, segmentation, and configuration for audio generation.
"""

from audiobrain.processing.config import GenerationConfig
from audiobrain.processing.validator import AudioValidator
from audiobrain.processing.preprocessor import AudioPreprocessor
from audiobrain.processing.segmenter import AudioSegmenter

__all__ = [
    'GenerationConfig',
    'AudioValidator',
    'AudioPreprocessor',
    'AudioSegmenter',
    'AudioSegment',
]
