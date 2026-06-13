# A/VT — Audio/Visual Transformer

**Generative Soundscape System**

A/VT is a thesis project that creates original soundscape compositions from user-provided audio samples. It extracts audio features using pretrained models (PANNs/AST), processes them through a Transformer encoder to learn latent representations, synthesizes new audio via k-NN mosaicing with overlap-add crossfading, and visualizes the latent structure as 63x63 ASCII self-similarity matrices.

## Architecture

```
Audio Input  →  PANNs/AST  →  Projection  →  Transformer  →  k-NN Mosaicing  →  Output Audio
                  (2048)      (2048→512)    Encoder (512)     + Crossfade
                                                              ↓
                                                          ASCII Matrix
                                                          Visualization
```

## Modules

| Module | Description |
|--------|-------------|
| **Feature Extraction** | HuggingFace AST models (AudioSet) with manual CNN fallback, 2048-dim embeddings |
| **Transformer Core** | 2-layer encoder, 8 attention heads, 512-dim latent space, autoregressive generation |
| **Mosaicing Synthesis** | k-NN search in latent space, raised-cosine crossfade, temperature/density/mode controls |
| **ASCII Visualization** | 63x63 cosine-similarity matrix with ANSI color support and multiple character sets |
| **Audio Effects** | Bitcrush, pitch shift, flanger, glitch, distortion, delay via EffectChain |
| **CLI** | Full argparse interface with `generate`, `visualize`, `train`, `info` subcommands |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Display system and model info
python audiobrain/cli.py info

# Generate new audio from samples
python audiobrain/cli.py generate --source input.wav --database samples/*.wav --output output.wav

# Visualize latent structure
python audiobrain/cli.py visualize --source output.wav --stats

# Train the transformer on synthetic data (for testing)
python audiobrain/cli.py train --epochs 50
```

## Python API

```python
from audiobrain import AudioGenerationPipeline, AudioBrainCore, BrainConfig

# End-to-end generation pipeline
pipeline = AudioGenerationPipeline(database_files=["samples/*.wav"])
pipeline.generate_and_save("source.wav", "output.wav")

# Low-level model access
config = BrainConfig(device="mps", seed=42)
model = AudioBrainCore(config=config)
print(f"Parameters: {model.count_parameters():,}")
```

## Project Structure

```
audiobrain/
  __init__.py           # Package root
  cli.py                # CLI interface
  model/                # Core model
    core.py             # AudioBrainCore (main class)
    config.py           # BrainConfig dataclass
    projection.py       # ProjectionHead (2048->512)
    transformer.py      # SoundscapeTransformer (encoder)
    feature_extractor.py# PANNs/AST feature extraction
    pipeline.py         # AudioProcessingPipeline
    synthesizer.py      # AudioMosaicSynthesizer
    generation_pipeline.py # AudioGenerationPipeline
    visualizer.py       # AudioBrainVisualizer (ASCII matrices)
  processing/           # Audio processing
    validator.py        # AudioValidator
    preprocessor.py     # AudioPreprocessor
    segmenter.py        # AudioSegmenter
    config.py           # GenerationConfig
  effects/              # Post-processing effects
    chain.py            # EffectChain
    bitcrusher.py, pitch.py, flanger.py, glitch.py, distortion.py, delay.py
  tests/                # Unit tests
    test_projection.py
    test_transformer.py
    test_core.py
    run_tests.py        # Plain-Python test runner
src/                    # Supporting scripts
  experiments/train.py  # Training verification script
tests/                  # Integration tests
```

## Validation

- **32 unit tests** covering shape validation, dimension mismatch, determinism, gradient flow, checkpoint roundtrip, loss decrease
- Training verified: loss decreases ~0.183 → ~0.015 over 50 epochs
- Checkpoint save/load roundtrip verified (weight-perfect match)
- Deterministic generation verified at temperature=0

```bash
python audiobrain/tests/run_tests.py
```

## License

MIT
