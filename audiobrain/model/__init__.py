"""
AudioBrain model module — Transformer, feature extraction, synthesis, and visualization.
"""

from audiobrain.model.config import BrainConfig
from audiobrain.model.core import AudioBrainCore
from audiobrain.model.feature_extractor import PANNsFeatureExtractor
from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioMosaicSynthesizer
from audiobrain.model.generation_pipeline import AudioGenerationPipeline
from audiobrain.model.visualizer import (
    AudioBrainVisualizer,
    LatentSpectrogram,
    ChladniOscilloscope,
    visualize_latents,
)

__all__ = [
    "BrainConfig",
    "AudioBrainCore",
    "PANNsFeatureExtractor",
    "AudioProcessingPipeline",
    "AudioMosaicSynthesizer",
    "AudioGenerationPipeline",
    "AudioBrainVisualizer",
    "LatentSpectrogram",
    "ChladniOscilloscope",
    "visualize_latents",
]
