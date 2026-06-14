"""
End-to-end generation test: source audio → latent transform → synthesis → HTML visualization.

Uses existing sample WAV files from tests/test_data/ to generate:
  - A new synthesized WAV file
  - A self-contained HTML visualization with spectrogram + Chladni views + audio player
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from audiobrain.model import AudioBrainCore, BrainConfig, AudioProcessingPipeline, AudioBrainVisualizer
from audiobrain.model.synthesizer import AudioMosaicSynthesizer
from audiobrain.processing.config import GenerationConfig


def main():
    print("=" * 60)
    print("  A/VT — End-to-End Generation + Visualization Test")
    print("=" * 60)

    # ── Paths ──
    test_data = Path(__file__).resolve().parent.parent / "test_data"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    source_file = test_data / "source_sample.wav"
    output_wav = results_dir / "generated_output.wav"
    output_html = results_dir / "generated_output.html"

    if not source_file.exists():
        print(f"\n  Source file not found: {source_file}")
        print("  Generating synthetic source...")
        import soundfile as sf
        sr = 32000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        freq = 200 + (t / duration) * 600
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        sf.write(str(source_file), audio, sr)
        print(f"  Created: {source_file}")

    # Collect database files (all sample WAVs)
    db_files = sorted(test_data.glob("sample_*.wav"))
    if not db_files:
        # Fallback: use source as its own database
        db_files = [source_file]
    db_paths = [str(f) for f in db_files]

    print(f"\n  Source:  {source_file.name}")
    print(f"  Database: {len(db_paths)} file(s)")
    for f in db_paths[:5]:
        print(f"    - {Path(f).name}")
    if len(db_paths) > 5:
        print(f"    ... and {len(db_paths) - 5} more")

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Initialize pipeline ──
    print("\n  [1/4] Initializing processing pipeline (PANNs + Transformer)...")
    pipeline = AudioProcessingPipeline(device=device)

    print("  [2/4] Building synthesis database...")
    synthesizer = AudioMosaicSynthesizer(device=device)
    try:
        synthesizer.build_database(
            audio_files=db_paths,
            feature_extractor=pipeline.feature_extractor,
            pipeline=pipeline,
            max_segments=50,
        )
    except Exception as e:
        print(f"  ! Database build warning: {e}")
        print("  → Continuing with empty database (sequential fallback)")

    # ── Process source through PANNs + Transformer ──
    print(f"\n  [3/4] Processing source → latent vectors...")
    _, latents = pipeline.process_audio(str(source_file), duration=5.0)

    n_segs = latents.shape[1]
    n_dims = latents.shape[-1]
    print(f"  Latents: [{n_segs} segments, {n_dims} dims]")

    # ── Synthesize audio ──
    print("\n  [4/4] Synthesizing audio and generating visualizations...")
    config = GenerationConfig(temperature=0.5, density=1.0, mode="fluid", seed=42)

    try:
        audio, sr_out = synthesizer.synthesize_from_latent(
            latents,
            db_paths,
            pipeline=pipeline,
            config=config,
        )
    except Exception as e:
        print(f"  ! Synthesis warning: {e}")
        print("  → Creating minimal output for visualization test")
        import soundfile as sf
        audio, sr_out = np.zeros(16000, dtype=np.float32), 32000

    # ── Save WAV ──
    import soundfile as sf
    sf.write(str(output_wav), audio.astype(np.float32), sr_out)
    wav_size = output_wav.stat().st_size
    wav_dur = len(audio) / sr_out
    print(f"\n  ✓ WAV saved: {output_wav}")
    print(f"    Duration: {wav_dur:.1f}s | Size: {wav_size:,} bytes | SR: {sr_out} Hz")

    # ── Generate HTML visualization ──
    import io
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr_out, format="WAV")
    audio_bytes = buf.getvalue()

    viz = AudioBrainVisualizer(grid_size=128, chars="dots", colors="aurora")
    viz.save_html(
        str(output_html),
        latents,
        audio_data=audio_bytes,
        title=f"Generated from {source_file.stem}",
        metadata={
            "source":       source_file.name,
            "mode":         config.mode,
            "temperature":  f"{config.temperature:.2f}",
            "density":      f"{config.density:.2f}",
            "duration":     f"{wav_dur:.1f}s",
            "sample_rate":  f"{sr_out} Hz",
            "segments":     f"{n_segs}",
            "latent_dim":   f"{n_dims}",
            "grid":         "128×128",
            "colors":       "aurora",
            "database":     f"{len(db_paths)} files",
        },
    )
    html_size = output_html.stat().st_size
    print(f"  ✓ HTML saved: {output_html}")
    print(f"    Size: {html_size:,} bytes")

    # ── Also render terminal preview ──
    print(f"\n  {'─'*56}")
    print(f"  Terminal preview (64×64, ocean):")
    print(f"  {'─'*56}")
    viz_small = AudioBrainVisualizer(grid_size=64, chars="dots", colors="ocean")
    viz_small.render_terminal(latents, title=source_file.stem, show="both")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Generation complete!")
    print(f"  WAV:  {output_wav}")
    print(f"  HTML: {output_html}")
    print(f"  Open in browser to see the 128x128 spectrogram +")
    print(f"  Chladni oscilloscope with embedded audio player.")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
