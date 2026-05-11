# Development Changelog - AudioBrain

This file is a short English development summary. For the detailed (Spanish) session log, see `docs/CHANGELOG_DEV.MD`.

## [Unreleased] - 2026-05-11

### Docs

- Reconciled changelog facts with the current code (component names, defaults, license).

### Known Gaps

- No dependency manifest (`pyproject.toml`/`requirements.txt`) or CI workflow is present yet; setup is not reproducible.

## [0.1.3] - 2026-04-26

### Added

- Phase 3: Audio synthesis by mosaicing (completed)
  - `audiobrain/model/synthesizer.py`: `AudioMosaicSynthesizer`
  - `audiobrain/model/generation_pipeline.py`: `AudioGenerationPipeline`
  - k-NN search in latent space for audio segment matching
  - Overlap-add crossfading for smooth transitions

- Test scripts
  - `tests/test_audio_generation.py`: end-to-end generation script writing `tests/test_data/generated_output.wav`

### Technical Notes

- Synthesizer defaults: `segment_duration=1.0`, `sample_rate=22050`, `crossfade_duration=0.1`
- Generation pipeline defaults: `segment_duration=0.1`, `sample_rate=32000`, `crossfade_duration=0.02`
- Dependencies implied by code: `torch`, `torchaudio`, `transformers`, `librosa`, `scikit-learn`, `soundfile`

### How to Test

```bash
python tests/test_audio_generation.py
```

## [0.1.2] - 2026-03-31

### Added

- Phase 2: PANNs feature extraction + processing pipeline (completed)
  - `audiobrain/model/feature_extractor.py`: `PANNsFeatureExtractor`
  - `audiobrain/model/pipeline.py`: `AudioProcessingPipeline`

## [0.1.1] - 2026-03-31

### Added

- Phase 1: Transformer model (completed)
  - `audiobrain/model/core.py`: `AudioBrainCore`
  - `audiobrain/model/config.py`: `BrainConfig`
  - `audiobrain/model/transformer.py`: `SoundscapeTransformer`
  - `audiobrain/model/projection.py`: `ProjectionHead`

## [0.1.0] - 2026-03-31

### Added

- Initial `audiobrain/` package structure
- Repository setup with Apache-2.0 License

---

**Legend:**
- ✅ = Completed
- 🚧 = In Progress
- ⏳ = Pending
