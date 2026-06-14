#!/usr/bin/env python3
"""
External test run -- generate using only external_test files with
different effect chains and visualization setups for each.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from audiobrain.model import AudioGenerationPipeline, AudioBrainVisualizer
import base64
from audiobrain.effects import EffectChain, bitcrush, pitch_down, flange, glitch, distort, delay


def main():
    base = Path(__file__).resolve().parent / "test_data"
    ext_dir = base / "external_test"
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(ext_dir.glob("*.wav"))
    source_paths = [str(f) for f in sources]

    print("=" * 70)
    print("EXTERNAL TEST RUN -- 5 setups, 5 effect chains, 5 visualizations")
    print("=" * 70)
    for s in sources:
        info = sf.info(str(s))
        print(f"  {s.name:20s}  {info.duration:5.1f}s  {info.samplerate}Hz")

    setups = [
        {
            "name": "07065086_crushed",
            "source": source_paths[0],
            "db": [source_paths[1], source_paths[2], source_paths[3], source_paths[4]],
            "chain": EffectChain().add(bitcrush, bits=6, downsample=2, mix=0.8)
                                 .add(distort, drive=4, mix=0.5),
            "viz_charset": "blocks",
            "viz_colors": "ocean",
            "description": "bitcrushed 6-bit + overdrive | ocean blocks",
        },
        {
            "name": "07070075_drift",
            "source": source_paths[1],
            "db": [source_paths[0], source_paths[2], source_paths[3], source_paths[4]],
            "chain": EffectChain().add(pitch_down, semitones=5, mix=0.7)
                                 .add(delay, time_ms=250, feedback=0.3, mix=0.5),
            "viz_charset": "ascii",
            "viz_colors": "heat",
            "description": "pitched down -5st + delay | heat ascii",
        },
        {
            "name": "07070097_swirl",
            "source": source_paths[2],
            "db": [source_paths[0], source_paths[1], source_paths[3], source_paths[4]],
            "chain": EffectChain().add(flange, depth_ms=4, rate_hz=0.4, mix=0.6)
                                 .add(glitch, intensity=0.15, seed=42),
            "viz_charset": "braille",
            "viz_colors": "forest",
            "description": "flanger + gentle glitch | forest braille",
        },
        {
            "name": "07071147_annihilated",
            "source": source_paths[3],
            "db": [source_paths[0], source_paths[1], source_paths[2], source_paths[4]],
            "chain": EffectChain().add(bitcrush, bits=4, downsample=4, mix=0.9)
                                 .add(pitch_down, semitones=8, mix=0.8)
                                 .add(flange, depth_ms=2, rate_hz=0.6, mix=0.4)
                                 .add(distort, drive=8, mix=0.6)
                                 .add(delay, time_ms=120, feedback=0.5, mix=0.3),
            "viz_charset": "shades",
            "viz_colors": "sunset",
            "description": "full destruction chain (5 effects) | sunset shades",
        },
        {
            "name": "NHU05104236_pure",
            "source": source_paths[4],
            "db": [source_paths[0], source_paths[1], source_paths[2], source_paths[3]],
            "chain": None,
            "viz_charset": "lines",
            "viz_colors": "matrix",
            "description": "no effects -- pure | matrix lines",
        },
    ]

    for s in setups:
        print("\n" + "=" * 70)
        print(f"SETUP: {s['name']}")
        print(f"  {s['description']}")
        print("=" * 70)

        out_wav = results_dir / f"{s['name']}_generated.wav"
        out_html = results_dir / f"{s['name']}_generated.html"

        try:
            pipeline = AudioGenerationPipeline(
                database_files=s["db"],
                max_database_segments=200,
                segment_duration=1.0,
                crossfade_duration=0.02,
            )

            print(f"  Chain: {s['chain'] or 'none'}")
            audio, latents = pipeline.generate(
                s["source"],
                k_neighbors=5,
                effect_chain=s["chain"],
            )

            pipeline.synthesizer.save_audio(audio, str(out_wav))

            # Encode audio as base64 for embedding
            with open(out_wav, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

            viz = AudioBrainVisualizer(
                charset=s["viz_charset"],
                colors=s["viz_colors"],
                grid_size=128,
                force_color=True,
            )

            metadata = {
                "source": Path(s["source"]).name,
                "duration": "{:.1f}s".format(len(audio) / 32000),
                "sample_rate": "32000 Hz",
                "charset": s["viz_charset"],
                "colors": s["viz_colors"],
                "chain": str(s["chain"]) if s["chain"] else "none",
                "description": s["description"],
            }

            viz.save_artifact(
                str(out_html),
                latents,
                audio_base64=audio_b64,
                title=s["name"],
                metadata=metadata,
            )

            duration = len(audio) / 32000
            print(f"  Output: {out_wav.name}  ({duration:.1f}s)")
            print(f"  HTML:   {out_html.name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for f in sorted(results_dir.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name:50s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
