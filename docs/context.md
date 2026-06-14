# A/VT — Architecture Reference

## Project Overview

A/VT is a generative soundscape system that creates original audio compositions from user-provided samples. The system uses a Transformer model to learn latent representations of audio segments, then synthesizes new audio via k-NN mosaicing with crossfade blending.

**Status**: All five core phases are complete, plus a post-processing audio effects module.

---

## Architecture

### Data Flow

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│  Raw Audio  │ →  │ Feature Ext.  │ →  │  Transformer   │ →  │  k-NN Mosaicing  │ →  Output WAV
│  (.wav)     │    │  (PANNs/AST)  │    │   Encoder      │    │  + Crossfade     │
└─────────────┘    └───────────────┘    └────────────────┘    └──────────────────┘
                       2048-dim            512-dim latents         │
                                                                   ↓
                                                          ┌──────────────────┐
                                                          │  ASCII Matrix    │
                                                          │  Visualization   │
                                                          └──────────────────┘
```

---

## Module 1: Feature Extraction (PANNs/AST)

**Files**: `audiobrain/model/feature_extractor.py`, `audiobrain/model/pipeline.py`

Extracts 2048-dimensional embeddings from raw audio using pretrained models.

**Model loading priority** (first successful load wins):
1. `MIT/ast-finetuned-audioset-10-10-0.4593` — AST, AudioSet (768-dim)
2. `ntu-spml/distil-ast` — Distil-AST, AudioSet (768-dim)
3. Manual CNN fallback (untrained, produces target_embedding_dim)

If the loaded model's output dimension differs from the target (default: 2048), a linear projection layer is added automatically.

**Key classes**:
- `PANNsFeatureExtractor` — Extracts audio features from files or waveforms
- `AudioProcessingPipeline` — Connects PANNs → AudioBrainCore in a single pipeline

**Dimensions**:
| Stage | Shape |
|-------|-------|
| Input audio | Raw waveform, variable length |
| Per-chunk embedding | `[768]` or `[2048]` |
| Sequence output | `[1, seq_len, 2048]` |
| After transformer | `[1, seq_len, 512]` |

---

## Module 2: Transformer Core Model

**Files**: `audiobrain/model/core.py`, `audiobrain/model/projection.py`, `audiobrain/model/transformer.py`, `audiobrain/model/config.py`

### BrainConfig

Configuration dataclass with auto device detection (CUDA → MPS → CPU).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Latent space dimension |
| `nhead` | 8 | Attention heads |
| `num_layers` | 2 | Encoder depth |
| `dim_feedforward` | 2048 | FFN dimension |
| `dropout` | 0.1 | Dropout rate |
| `learning_rate` | 0.001 | Adam LR |
| `embedding_dim_input` | 2048 | PANNs output dim |
| `embedding_dim_latent` | 512 | Latent space dim |
| `max_seq_len` | 512 | Max sequence length |

### ProjectionHead

Adapts PANNs embeddings to the Transformer latent space.

```
Linear(2048, 512) → LayerNorm → Dropout(0.1)
Input:  [batch, seq_len, 2048]
Output: [batch, seq_len, 512]
```

### SoundscapeTransformer

PyTorch `nn.TransformerEncoder` with 2 layers, 8 attention heads.

```
Input:  [batch, seq_len, 512]
Output: [batch, seq_len, 512]
```

### AudioBrainCore

Main class combining projection + transformer with training and generation methods.

| Method | Description |
|--------|-------------|
| `forward(x)` | Projection → Transformer |
| `encode(x)` | Alias for forward() |
| `train_step(batch, optim, criterion)` | Single step with gradient clipping |
| `generate(start_emb, length, temp)` | Autoregressive generation |
| `save_checkpoint(path)` | Persist weights + config |
| `load_checkpoint(path)` | Restore from checkpoint |

**Generation**: Temperature-controlled autoregressive inference. At each step, the transformer sees the accumulated sequence and predicts the next latent vector.

**Parameters**: 7,355,904 total (all trainable).

---

## Module 3: Mosaicing Synthesis (k-NN)

**Files**: `audiobrain/model/synthesizer.py`, `audiobrain/model/generation_pipeline.py`

### AudioMosaicSynthesizer

Builds a database of audio segments with their latent vectors, then matches target latents to the nearest database segments via k-NN search.

**Synthesis modes**:
| Mode | Behavior |
|------|----------|
| `fluid` | Random selection from top-k neighbors |
| `evolving` | Always picks the nearest neighbor |
| `glitch` | Shortened crossfade for abrupt transitions |

**Parameters**: temperature (creativity), density (energy threshold for segment selection), crossfade duration.

**Crossfade**: Raised-cosine window (Hann) for smooth segment transitions.

### AudioGenerationPipeline

Convenience wrapper combining feature extraction, transformer processing, and synthesis into a single interface.

---

## Module 4: ASCII Visualization

**Files**: `audiobrain/model/visualizer.py`

### AudioBrainVisualizer

Renders latent vectors `[1, 63, 512]` as a 63×63 cosine self-similarity matrix using ANSI-colored ASCII characters.

**Features**:
- Multiple character sets (dots, blocks, braille)
- Color schemes (viridis, plasma, inferno, grayscale, thermal)
- Modes: `similarity`, `distance`, `binary`
- Methods: `render()`, `render_with_stats()`, `to_grid()`
- Cross-platform color support via `colorama` with auto-fallback

---

## Module 5: CLI

**Files**: `audiobrain/cli.py`

| Command | Description |
|---------|-------------|
| `generate` | Generate audio from source using database |
| `visualize` | Render ASCII matrix of audio latent space |
| `train` | Train AudioBrainCore on synthetic data |
| `info` | Display system and model metadata |

---

## Audio Effects (Bonus Module)

**Files**: `audiobrain/effects/`

Post-processing effects applied to generated audio:

| Effect | Description |
|--------|-------------|
| `EffectChain` | Ordered pipeline of effects |
| `bitcrush` | Bit-depth reduction |
| `pitch_up` / `pitch_down` | Pitch shifting |
| `flanger` | Flanger modulation |
| `glitch` | Glitch/stutter effects |
| `distort` | Waveform distortion |
| `delay` | Echo/delay |

---

## Input/Output Summary

| Stage | Input | Output |
|-------|-------|--------|
| Feature Extractor | Raw `.wav` audio | `[seq_len, 2048]` |
| ProjectionHead | `[batch, seq, 2048]` | `[batch, seq, 512]` |
| SoundscapeTransformer | `[batch, seq, 512]` | `[batch, seq, 512]` |
| k-NN Synthesis | `[1, seq, 512]` + audio DB | Raw waveform |
| Visualization | `[1, 63, 512]` | 63×63 ASCII string |

## Testing

- **32 unit tests** in `audiobrain/tests/` (pytest-compatible + raw Python runner)
- Training verification: `src/experiments/train.py`
- Integration tests: `tests/test_audio_generation.py`

```bash
python audiobrain/tests/run_tests.py
pytest audiobrain/tests/
```
