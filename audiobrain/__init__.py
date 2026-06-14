"""
AudioBrain — Generative Soundscape System.

A/VT is a generative system that creates original soundscape compositions
from user-provided audio samples. It extracts audio features via pretrained
models, processes them through a Transformer encoder, synthesizes new audio
via k-NN mosaicing, and visualizes the latent structure as artistic 128x128
spectrograms and Chladni-plate oscilloscopic patterns.

Modules:
    1. Feature Extraction (PANNs/AST)   — Complete
    2. Transformer Core Model           — Complete
    3. Mosaicing Synthesis (k-NN)       — Complete
    4. Artistic 128x128 Visualization   — Complete
    5. CLI & Integration                — Complete
    + Audio Effects (post-processing)   — Complete

Usage:
    # CLI
    audiobrain generate song.wav --database samples/*.wav --viz html
    audiobrain render song.wav --viz both
    audiobrain train --epochs 50
    audiobrain info

    # Python
    from audiobrain import AudioBrainCore, BrainConfig, AudioBrainVisualizer
    from audiobrain import AudioGenerationPipeline
    pipeline = AudioGenerationPipeline(database_files=["samples/*.wav"])
    pipeline.generate_and_save("source.wav", "output.wav")
"""

from audiobrain.model import (
    AudioBrainCore,
    BrainConfig,
    PANNsFeatureExtractor,
    AudioProcessingPipeline,
    AudioMosaicSynthesizer,
    AudioGenerationPipeline,
    AudioBrainVisualizer,
    LatentSpectrogram,
    ChladniOscilloscope,
    visualize_latents,
)

from audiobrain.processing import (
    GenerationConfig,
    AudioValidator,
    AudioPreprocessor,
    AudioSegmenter,
)

from audiobrain.effects import (
    EffectChain,
    bitcrush,
    pitch_down,
    pitch_up,
    flange,
    glitch,
    distort,
    delay,
)

__all__ = [
    # Core model
    "AudioBrainCore",
    "BrainConfig",
    # Feature extraction & processing
    "PANNsFeatureExtractor",
    "AudioProcessingPipeline",
    "AudioGenerationPipeline",
    # Synthesis
    "AudioMosaicSynthesizer",
    # Visualization
    "AudioBrainVisualizer",
    "LatentSpectrogram",
    "ChladniOscilloscope",
    "visualize_latents",
    # Processing pipeline components
    "GenerationConfig",
    "AudioValidator",
    "AudioPreprocessor",
    "AudioSegmenter",
    # Audio effects
    "EffectChain",
    "bitcrush",
    "pitch_down",
    "pitch_up",
    "flange",
    "glitch",
    "distort",
    "delay",
]

__version__ = "0.2.0"
