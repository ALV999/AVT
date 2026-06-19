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

| Module                  | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **Feature Extraction**  | HuggingFace AST models (AudioSet) with manual CNN fallback, 2048-dim embeddings                    |
| **Transformer Core**    | 2-layer encoder, 8 attention heads, 512-dim latent space, autoregressive generation                |
| **Mosaicing Synthesis** | k-NN search in latent space, raised-cosine crossfade, temperature/density/mode controls            |
| **ASCII Visualization** | 63x63 cosine-similarity matrix with ANSI color support and multiple character sets                 |
| **Audio Effects**       | Bitcrush, pitch shift, flanger, glitch, distortion, delay via EffectChain                          |
| **CLI**                 | Full argparse interface with `generate`, `visualize`, `info` subcommands                           |
| **TUI**                 | Terminal User Interface (Textual) with file browser, effects sliders, progress bar, audio playback |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the TUI (recommended)
avt
# or: python -m audiobrain

# Display system and model info
python audiobrain/cli.py info

# Generate new audio from samples
python audiobrain/cli.py generate --source input.wav --database samples/*.wav --output output.wav

# Visualize latent structure
python audiobrain/cli.py visualize --source output.wav --stats
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
  __main__.py           # Module entry point
  cli.py                # CLI interface
  tui/                  # Terminal User Interface
    app.py              # Textual app (Home, Workspace, About screens)
    app_base.py         # Base TUI utilities
    app_header.py       # Header rendering
    widgets.py          # Custom TUI widgets
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
```

## License

MIT
