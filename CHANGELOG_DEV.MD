# Development Changelog

All notable changes to the A/VT project during development will be documented in this file.

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
