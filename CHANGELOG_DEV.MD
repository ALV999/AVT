# Development Changelog

All notable changes to the A/VT project during development will be documented in this file.

## [0.4.0] - 2026-06-13

### TUI Redesign — Planned

**Comprehensive analysis completed** (`docs/tui-analysis.md`) identifying critical issues and improvement opportunities.

#### Identified Issues

- **Architecture**
  - Unused `widgets.py` module (VisualizationPanel, MiniStrip, EffectChainWidget not integrated)
  - Monolithic `app.py` (562 lines, no separation of concerns)
  - Duplicated pipeline logic instead of reusing `cli.py`
  - No error recovery or cancel mechanism
  - No state persistence

- **User Experience**
  - No file type filtering (user can select non-audio files)
  - No audio preview capability
  - Confusing database workflow ("Add to DB" button unclear)
  - No parameter validation or help tooltips
  - Progress bar with no time estimates
  - Cannot cancel long-running operations
  - Effects cannot be removed individually, no parameter editing

- **Visual Design**
  - Cramped 3-column layout wastes space
  - No visual hierarchy
  - ASCII visualization panel too small (48-grid vs 128-grid CLI)
  - No theme support

- **Missing Features**
  - No file filtering (.wav, .mp3, .flac only)
  - No multi-select or drag-and-drop
  - No batch processing
  - No preset system
  - No recent files history

#### New TUI Architecture (Planned)

**Two-screen flow:**

1. **Home Screen**
   - ASCII art logo
   - Menu: Start, Docs, About, Quit
   - Clean, minimal entry point

2. **Main Workspace**
   - **Left Column (Commands)**
     - Load Files button
     - Quick actions (Clear DB, Load Preset, Save Preset)
   - **Right Column (File List)**
     - Added source files
     - Database files count
     - Remove individual files
   - **Below: Processing Options**
     - Preprocessing section (sample rate, gain, EQ, normalization, stereo)
     - Synthesis section (mode, temperature, density, segment duration, crossfade, k-NN, seed)
     - Effects section with:
       - Checkboxes to enable/disable each effect
       - Sliders for intensity/mix parameters
       - Remove button per effect
   - **Bottom: Generate**
     - Large, prominent Generate button
     - Integrated progress bar with status messages
     - Cancel button during processing

#### Implementation Plan

**Phase 1: Critical Fixes** (1-2 days)

- [ ] Remove unused `widgets.py` or integrate properly
- [ ] Add file type filtering to DirectoryTree
- [ ] Add cancel button for long operations
- [ ] Add parameter validation (range checks)
- [ ] Extract pipeline logic to reusable controller

**Phase 2: Core Redesign** (3-5 days)

- [ ] Implement Home Screen with ASCII logo
- [ ] Build new 2-column + bottom panel layout
- [ ] Replace visualization panel with file list
- [ ] Add effect checkboxes and sliders
- [ ] Implement large Generate button with progress bar
- [ ] Add parameter help tooltips

**Phase 3: Polish** (2-3 days)

- [ ] Add audio preview widget
- [ ] Add preset save/load system
- [ ] Add keyboard shortcuts
- [ ] Add theme support
- [ ] Add recent files history
- [ ] End-to-end testing

#### Files to Modify

| File                          | Action    | Description                               |
| ----------------------------- | --------- | ----------------------------------------- |
| `audiobrain/tui/app.py`       | Rewrite   | New home screen + workspace layout        |
| `audiobrain/tui/widgets.py`   | Integrate | Use custom widgets or remove              |
| `audiobrain/tui/controllers/` | Create    | Pipeline controller, effect controller    |
| `audiobrain/tui/views/`       | Create    | Home screen, file browser, control panels |
| `audiobrain/cli.py`           | Refactor  | Extract reusable pipeline function        |

---

## [0.3.0] - 2026-06-13

### Added

- **Effects module** (`audiobrain/effects/`) — post-processing audio effects chain
  - `bitcrusher.py` — bit-depth reduction with optional downsampling (4 params: bits, downsample, mix, seed)
  - `pitch.py` — pitch shift up/down via resampling (semitones + mix)
  - `flanger.py` — comb-filter modulation (depth_ms, rate_hz, mix)
  - `glitch.py` — stochastic segment reversal/repetition (intensity, seed)
  - `distortion.py` — tanh waveshaping drive (drive, mix)
  - `delay.py` — feedback delay line (time_ms, feedback, mix)
  - `chain.py` — `EffectChain` class: `add()`, `remove()`, `clear()`, `apply()` for composing effects
  - All effects include safety clamping and dry/wet mix blending
- **Visualizer v2** (`audiobrain/model/visualizer.py`) — complete rewrite (641 → 985 lines)
  - `LatentSpectrogram` class — 128×128 spectrogram field from latent dimension activations
  - `ChladniOscilloscope` class — physical resonance simulation with nodal line detection + thickness
  - `WaveformBar` class — compact horizontal waveform energy bar
  - `WaveformLine` class — traditional oscilloscope line trace with SVG export
  - `RealSpectrogram` class — STFT spectrogram computed from raw audio waveform
  - `compute_views()` — unified spectrogram + Chladni nodal line computation
  - `render_terminal()` — ANSI terminal output with spectrogram overlay modes (`overlay`, `spectrogram`, `chladni`, `both`)
  - `build_html()` — self-contained HTML artifact with:
    - Latent spectrogram rendered as base64 PPM pixel image
    - Chladni nodal lines overlaid as SVG rect elements
    - Oscilloscope waveform line (SVG polyline)
    - Waveform energy bar + real audio spectrogram (bottom strips)
    - Audio player (base64-embedded WAV)
    - Boxed panel layout with labels and legend
  - `save_html()` — saves HTML artifact with waveform and sample rate
  - `_field_to_data_url()` — converts 2D numpy field to base64-encoded PPM (P6) image
  - `_lines_to_svg()` — converts nodal line mask + thickness to SVG rect elements
  - `_field_to_ansi()` / `_field_to_ansi_overlay()` — ANSI terminal rendering with character ramps
  - New color scheme: `"aurora"` (purple, cyan, green gradient)
  - New character ramps: `CHAR_RAMPS` replaces `CHAR_SETS` with extended ranges
  - Removed: `similarity_to_chars`, `similarity_to_ansi`, `to_grid`, `render`, `render_with_stats`,
    `save_ansi`, `latent_to_ascii_art`, `_compute_pca_data`, `save_artifact`
- **Effects integration** (`audiobrain/model/generation_pipeline.py`)
  - `generate()` and `generate_and_save()` now accept optional `effect_chain` parameter
  - Effects applied post-synthesis before audio output
- **Effect test suite** (`tests/test_effects.py`)
  - 26 tests covering: bitcrusher (shape, peak, mix=0 dry, downsample blend),
    pitch (up/down length, semitones=0 identity, mix=0 dry),
    flanger (shape, peak, mix=0 dry), glitch (shape, peak, reproducibility),
    distortion (shape, peak, mix=0 dry), delay (shape, peak, mix=0 dry),
    EffectChain (add/remove/clear/apply, shape, peak, length),
    safety clamps (out-of-range params)
- **External test run** (`tests/run_external_test.py`)
  - 5 setups with distinct effect chains: crushed, drift, swirl, annihilated, pure
  - Each setup: generates WAV + HTML artifact with unique color scheme
  - Results: `tests/test_data/results/` (5 WAV + 5 HTML)

### Changed

- **Visualizer API** — `AudioBrainVisualizer.__init__` now takes `chars` (not `charset`),
  adds `chladni_chars` and `chladni_color` parameters
- **Visualizer HTML format** — `build_html()` signature changed: `audio_data` (bytes) replaces
  `audio_base64` (str); adds `waveform` (np.ndarray) and `sample_rate` (int)
- **Save method** — `save_artifact()` replaced by `save_html()`

### Fixed

- **AST sample rate mismatch** (`audiobrain/model/feature_extractor.py`)
  - `_try_load_models()` now adopts HF feature extractor's native `sampling_rate` after load
  - Fixes "was trained using a sampling rate of 16000 and not 32000" error on MPS/CUDA
  - Pipeline resamples input audio to match loaded model's expected rate

### Known Issues

- `tests/run_external_test.py` calls deprecated `save_artifact()` and uses old `charset` kwarg
  — needs update to `save_html()` API with `waveform` and `sample_rate` parameters

### Verification

- Effect tests: 26/26 passed (all effects + chain + safety clamps)
- External test: 5/5 setups generate WAV + HTML artifacts successfully
- HTML artifacts: spectrogram + Chladni lines + waveform strips render correctly

---

## [0.2.0] - 2026-05-26

### Added

- **Sample rate unification**: All modules now default to 32kHz (PANNs native)
  - `synthesizer.py`, `preprocessor.py`, `segmenter.py` updated from 22.05kHz
  - Test scripts (`generate_*.py`, `test_audio_generation.py`) updated accordingly
- **Unit test suite** (`audiobrain/tests/`)
  - `run_tests.py`: 32 tests covering ProjectionHead, SoundscapeTransformer, AudioBrainCore
  - `test_projection.py`, `test_transformer.py`, `test_core.py`: pytest-compatible test classes
  - Tests: shape validation, dimension mismatch, determinism, gradient flow, checkpoint roundtrip, loss decrease
- **PANNs model loading** (`audiobrain/model/feature_extractor.py`)
  - Priority-ordered model loading: tries MIT AST (AudioSet) → Distil-AST → manual CNN fallback
  - Dynamic output dimension detection with automatic linear projection to target dimension
- **Fase 4: ASCII Visualization** (`audiobrain/model/visualizer.py`)
  - `AudioBrainVisualizer`: Renders latent vectors as 63×63 self-similarity matrix
  - Cosine similarity mapping to 10-character gradient ramp
  - `render()`, `render_with_stats()`, `to_grid()` methods
  - `visualize_latents()` convenience function
- **Fase 5: CLI** (`audiobrain/cli.py`)
  - Subcommands: `generate`, `visualize`, `train`, `info`
  - Full argparse interface with all generation parameters
  - Glob pattern expansion for source/database files
  - Checkpoint save after training
- **Module exports**: Visualizer and CLI exposed in package `__init__.py`

### Changed

- MPS device auto-detection improved in `config.py`
- Test scripts use uniform 32kHz sample rate
- `feature_extractor.py` now uses `target_embedding_dim` instead of `embedding_dim`

### Verification

- Unit tests: 32/32 passed
- CLI `info` command: displays system and model metadata
- Visualizer: correct 63×63 ASCII grid with cosine similarity matrix

---

## [0.1.5] - 2026-05-26

### Added

- **Training loop** (`audiobrain/model/core.py`)
  - `train_step(batch, optimizer, criterion)`: Self-supervised reconstruction training with gradient clipping
  - `generate(start_emb, length, temperature)`: Autoregressive latent sequence generation
  - `save_checkpoint(path)`: Model weights + config persistence
  - `load_checkpoint(path)`: Checkpoint restoration with device mapping
- **Training verification script** (`src/experiments/train.py`)
  - 50-epoch training loop on dummy PANNs embeddings
  - Checkpoint roundtrip validation
  - Autoregressive generation shape and determinism tests
- **Dependency manifest** (`requirements.txt`)
  - Pinned versions for torch, torchaudio, transformers, librosa, soundfile, numpy, scikit-learn
- **BrainConfig**: Added `max_seq_len: int = 512` field

### Changed

- **`src/config.py`**: Replaced with deprecation notice pointing to canonical `audiobrain/model/config.py`

### Verification

- Model: 7,355,904 parameters initialized successfully
- Training: Loss decreased 0.183 → 0.015 over 50 epochs (CPU)
- Checkpoint: Save/load roundtrip verified (weight-perfect match)
- Generation: Correct output shape `[1, 60, 512]`, deterministic at temperature=0

---

## [0.1.4] - 2026-04-29

### Added

- **Processing Module** - Complete audio processing pipeline (`audiobrain/processing/`)
  - `config.py`: `GenerationConfig` dataclass
    - Temperature, density, mode, and seed controls
    - Validation and reproducibility seeding
  - `validator.py`: `AudioValidator` class
    - File format checking
    - Corruption detection
    - Duration validation with configurable `min_duration`
  - `preprocessor.py`: `AudioPreprocessor` class
    - Silence removal
    - Normalization
    - Resampling
    - Mono conversion
  - `segmenter.py`: `AudioSegmenter` class
    - Fixed-duration audio segmentation
    - RMS energy filtering for intelligent segmentation
    - Feature-based segment extraction
  - `__init__.py`: Exports all processing components

### Changed

- **Synthesizer Integration** (`audiobrain/model/synthesizer.py`)
  - Updated to use new processing modules
  - Added `sample_rate` parameter support
  - Refactored for improved readability and maintainability
  - Better integration with validation, preprocessing, and segmentation workflows

- **Code Quality Improvements**
  - Refactored synthesizer code structure
  - Improved module documentation
  - Enhanced error handling throughout processing pipeline

### Fixed

- Normalization logic in audio processing
- Energy threshold lowered to 0.0001 for better segmentation
- Argument parsing and validation logic in test scripts
- Added debug logs for segmentation troubleshooting
- `min_duration` argument handling in `AudioValidator.validate_files`

### Technical Details

- **New Directory Structure**: `audiobrain/processing/` for audio manipulation logic
- **User Inference Controls**: GenerationConfig enables fine-tuned generation parameters
- **Pre-processing Validation**: AudioValidator prevents runtime errors by validating inputs early
- **Intelligent Segmentation**: RMS energy filtering ensures quality segment selection

### Files Created/Modified

| File                                    | Changes | Description                              |
| --------------------------------------- | ------- | ---------------------------------------- |
| `audiobrain/processing/__init__.py`     | +15     | Package exports for processing module    |
| `audiobrain/processing/config.py`       | +64     | GenerationConfig dataclass               |
| `audiobrain/processing/preprocessor.py` | +135    | AudioPreprocessor implementation         |
| `audiobrain/processing/segmenter.py`    | +107    | AudioSegmenter with RMS filtering        |
| `audiobrain/processing/validator.py`    | +110    | AudioValidator with corruption detection |
| `audiobrain/model/synthesizer.py`       | ~183    | Updated to use processing modules        |

### Testing

- Random audio sample generation (10 diverse samples, 5-20s duration)
- 30s sample generator for extended testing
- Full pipeline integration tests with auto-extension

---

## [0.1.3] - 2024-01-XX

### Added

- **Phase 3: Audio Synthesis by Mosaicing** ✅ COMPLETED
  - `audiobrain/model/synthesizer.py`: `AudioSynthesizer` class
  - k-NN search in latent space for audio segment matching
  - Overlap-add crossfading for smooth transitions
  - `synthesize_from_latent()` method for end-to-end generation
  - `synthesize_from_file()` convenience method
- **Testing Infrastructure**
  - `tests/` directory created
  - `tests/test_audio_generation.py`: Full pipeline integration test
    - Generates synthetic source audio (sweeping sine wave)
    - Runs complete pipeline: PANNs → Transformer → k-NN Synthesis
    - Saves output WAV file for verification

### Changed

- Updated `audiobrain/model/__init__.py` to export `AudioSynthesizer`
- Updated `CHANGELOG_DEV.MD` with Phase 3 completion details

### Technical Details

- **Input**: 512-dim latent vectors from `AudioBrainCore`
- **Output**: Raw audio waveform (numpy array)
- **Parameters**: Configurable `n_neighbors` and `crossfade_samples`
- **Dependencies**: `librosa`, `scikit-learn`, `soundfile`

### How to Test

```bash
python tests/test_audio_generation.py
# Output: tests/test_data/generated_output.wav
```

## [0.1.2] - 2024-01-XX

### Added

- **Phase 2: PANNs Integration** ✅ COMPLETED
  - `audiobrain/model/feature_extractor.py`: `PANNsFeatureExtractor` class
  - `audiobrain/model/pipeline.py`: `AudioProcessingPipeline` class
  - Automatic download of PANNs CNN14 pretrained weights
  - End-to-end processing: Raw Audio → PANNs → Transformer → Latent Vectors

### Changed

- Updated `audiobrain/model/__init__.py` to export new classes
- Updated documentation with data flow diagram

### Technical Details

- **Model**: PANNs CNN14 (pretrained on AudioSet)
- **Input**: Raw audio waveform (any sample rate, converted to 22050Hz)
- **Output**: `[batch, seq_len, 512]` latent vectors
- **Embedding Dim**: 2048 (PANNs) → 512 (Transformer)

## [0.1.1] - 2024-01-XX

### Added

- **Phase 1: Core Transformer Model** ✅ COMPLETED
  - `audiobrain/model/core.py`: `AudioBrainCore` class
  - `audiobrain/model/config.py`: `BrainConfig` dataclass
  - `audiobrain/model/transformer.py`: `SoundscapeTransformer` and `ProjectionHead`
  - Complete transformer architecture with 8 attention heads, 2 layers

### Technical Details

- **Framework**: PyTorch 2.0+
- **Input**: `[batch, seq_len, 2048]` PANNs embeddings
- **Output**: `[batch, seq_len, 512]` contextualized latent vectors
- **Config**: d_model=512, nhead=8, num_layers=2, dim_feedforward=2048

## [0.1.0] - 2024-01-XX

### Added

- Initial project structure
- Repository setup with MIT License
- Basic README.md

---

**Legend:**

- ✅ = Completed
- 🚧 = In Progress
- ⏳ = Pending
