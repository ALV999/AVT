#!/usr/bin/env python3
"""
AudioBrain CLI — Generative Soundscape System.

Commands:
    generate    Generate new audio from source samples
    render      Visualize existing audio/latents (no synthesis)
    train       Train the AudioBrainCore model
    info        Display system and model information
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import sys
from pathlib import Path


# ── Project root ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from audiobrain.model import (
    AudioBrainCore,
    BrainConfig,
    AudioGenerationPipeline,
    AudioProcessingPipeline,
    AudioBrainVisualizer,
)
from audiobrain.processing.config import GenerationConfig, PreprocessingConfig
from audiobrain.processing.validator import AudioValidator
from audiobrain.effects import (
    EffectChain,
    bitcrush,
    delay,
    distort,
    flange,
    glitch,
    pitch_down,
    pitch_up,
)


# ═══════════════════════════════════════════════════════════════
# Effect registry — maps name → (function, param_specs)
# Each param_spec: (name, type_fn, default, help)
# ═══════════════════════════════════════════════════════════════

EFFECT_REGISTRY: dict[str, tuple[callable, list[tuple[str, type, object, str]]]] = {
    "bitcrush": (bitcrush, [
        ("bits",       int,   8,   "Bit depth (1-24)"),
        ("downsample", int,   1,   "Decimation factor (1-16)"),
        ("mix",        float, 1.0, "Dry/wet blend (0-1)"),
    ]),
    "delay": (delay, [
        ("time",       float, 300, "Delay time in ms (20-2000)"),
        ("feedback",   float, 0.4, "Feedback amount (0-0.95)"),
        ("mix",        float, 0.5, "Dry/wet blend (0-1)"),
    ]),
    "distort": (distort, [
        ("drive",      float, 3.0, "Drive gain (1-20)"),
        ("mix",        float, 0.5, "Dry/wet blend (0-1)"),
    ]),
    "flanger": (flange, [
        ("depth",      float, 3.0, "Depth in ms (0.1-10)"),
        ("rate",       float, 0.3, "LFO rate in Hz (0.05-5)"),
        ("feedback",   float, 0.5, "Feedback (0-0.95)"),
        ("mix",        float, 0.5, "Dry/wet blend (0-1)"),
    ]),
    "glitch": (glitch, [
        ("intensity",  float, 0.2, "Glitch intensity (0.05-0.8)"),
        ("seed",       int,   None, "Random seed"),
    ]),
    "pitch-down": (pitch_down, [
        ("semitones",  float, 7.0, "Semitones down (0-24)"),
        ("mix",        float, 1.0, "Dry/wet blend (0-1)"),
    ]),
    "pitch-up": (pitch_up, [
        ("semitones",  float, 7.0, "Semitones up (0-24)"),
        ("mix",        float, 1.0, "Dry/wet blend (0-1)"),
    ]),
}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _expand_globs(patterns: list[str]) -> list[str]:
    """Expand glob patterns, keeping literal paths that don't match."""
    files: list[str] = []
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            files.extend(sorted(matches))
        else:
            files.append(p)
    return files


def _parse_effect_spec(raw: str) -> tuple[str, dict]:
    """
    Parse "--effect name:key=val,key=val" into (name, {key: val}).

    Examples:
        "bitcrush:bits=6,mix=0.7"
        "delay:time=400,feedback=0.3"
    """
    if ":" in raw:
        name, params_str = raw.split(":", 1)
    else:
        name = raw
        params_str = ""

    name = name.strip().lower()
    params: dict = {}

    if params_str:
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try int, then float, then string
                try:
                    params[k] = int(v)
                except ValueError:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v
            else:
                # Bare value: treat as activating the effect with defaults
                pass

    return name, params


def _build_effect_chain(effect_specs: list[str]) -> EffectChain | None:
    """Parse effect specs and build an EffectChain."""
    if not effect_specs:
        return None

    chain = EffectChain()
    for spec in effect_specs:
        name, params = _parse_effect_spec(spec)
        if name not in EFFECT_REGISTRY:
            print(f"Warning: unknown effect '{name}', skipping. Available: {', '.join(EFFECT_REGISTRY)}",
                  file=sys.stderr)
            continue
        fn, _ = EFFECT_REGISTRY[name]
        chain.add(fn, **params)
    return chain


# ═══════════════════════════════════════════════════════════════
# Command: generate
# ═══════════════════════════════════════════════════════════════

def cmd_generate(args: argparse.Namespace) -> int:
    """Generate new audio and visualizations from source samples."""
    # ── Resolve input files ──
    source_files = _expand_globs(args.source)
    db_files = _expand_globs(args.database) if args.database else source_files

    is_valid, msg = AudioValidator.validate_files(db_files, min_duration=args.segment_duration)
    if not is_valid:
        print(f"Error: {msg}", file=sys.stderr)
        return 1

    # ── Build effect chain ──
    effect_chain = _build_effect_chain(args.effect or [])

    # ── Build generation config ──
    gen_config = GenerationConfig(
        temperature=args.temperature,
        density=args.density,
        mode=args.mode,
        seed=args.seed,
        segment_duration=args.segment_duration,
        min_segment_energy=args.min_energy,
    )

    # ── Build preprocessing config ──
    pp_config = PreprocessingConfig(
        target_sr=args.sample_rate,
        gain_db=args.gain,
        character=args.character,
        norm_mode=args.norm,
        stereo_mode=args.stereo,
        trim_silence=args.trim_silence,
        trim_threshold_db=args.trim_threshold,
    )

    print(f"\n{'='*60}")
    print(f"  AudioBrain — Generate")
    print(f"{'='*60}")
    print(f"  Source:       {len(source_files)} file(s)")
    print(f"  Database:     {len(db_files)} file(s)")
    print(f"  Mode:         {gen_config.mode}")
    print(f"  Temperature:  {gen_config.temperature}")
    print(f"  Density:      {gen_config.density}")
    print(f"  Sample rate:  {pp_config.target_sr} Hz")
    print(f"  Character:    {pp_config.character}")
    if effect_chain and len(effect_chain) > 0:
        print(f"  Effects:      {effect_chain}")
    print(f"{'='*60}\n")

    # ── Initialize pipeline ──
    pipeline = AudioGenerationPipeline(
        database_files=db_files,
        max_database_segments=args.max_segments,
        segment_duration=gen_config.segment_duration,
        crossfade_duration=args.crossfade,
        sample_rate=pp_config.target_sr,
    )

    # ── Initialize visualizer ──
    viz = AudioBrainVisualizer(
        grid_size=args.grid_size,
        chars=args.chars,
        colors=args.colors,
        force_color=args.force_color,
    )

    exit_code = 0

    for i, src in enumerate(source_files):
        stem = Path(src).stem
        out_wav = args.output
        if len(source_files) > 1 or args.output_dir:
            out_dir = args.output_dir or Path(args.output).parent
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_wav = str(Path(out_dir) / f"{stem}_generated.wav")

        print(f"[{i+1}/{len(source_files)}] {stem}")

        try:
            # ── Process audio through PANNs + Transformer ──
            _, latents = pipeline.processing_pipeline.process_audio(
                src,
                duration=args.duration,
                target_seq_len=63,
            )

            # ── Synthesize audio ──
            from audiobrain.processing.config import GenerationConfig as GC
            synth_config = GC(
                temperature=gen_config.temperature,
                density=gen_config.density,
                mode=gen_config.mode,
                seed=gen_config.seed,
            )
            audio, _ = pipeline.synthesizer.synthesize_from_latent(
                latents,
                [str(src)],
                pipeline=pipeline.processing_pipeline,
                config=synth_config,
            )

            # ── Apply effects ──
            if effect_chain and len(effect_chain) > 0:
                print(f"  → Applying effects: {effect_chain}")
                audio = effect_chain.apply(audio, pp_config.target_sr)

            # ── Save audio ──
            import soundfile as sf
            sf.write(out_wav, audio.astype(np.float32), pp_config.target_sr)
            print(f"  → Saved: {out_wav}")

            # ── Encode audio as WAV bytes for HTML ──
            audio_bytes = None
            if args.viz in ("html", "both"):
                buf = io.BytesIO()
                sf.write(buf, audio.astype(np.float32), pp_config.target_sr, format="WAV")
                audio_bytes = buf.getvalue()

            # ── Build metadata ──
            metadata = {
                "source":        Path(src).name,
                "mode":          gen_config.mode,
                "temperature":   f"{gen_config.temperature:.2f}",
                "density":       f"{gen_config.density:.2f}",
                "duration":      f"{len(audio) / pp_config.target_sr:.1f}s",
                "sample_rate":   f"{pp_config.target_sr} Hz",
                "segments":      f"{latents.shape[1]}",
                "latent_dim":    f"{latents.shape[-1]}",
                "grid":          f"{args.grid_size}×{args.grid_size}",
                "colors":        args.colors,
                "character":     pp_config.character,
            }

            # ── Visualize ──
            if args.viz in ("terminal", "both"):
                viz.render_terminal(
                    latents,
                    title=Path(src).stem,
                    show=args.show,
                )

            if args.viz in ("html", "both"):
                html_path = str(Path(out_wav).with_suffix(".html"))
                viz.save_html(
                    html_path,
                    latents,
                    audio_data=audio_bytes,
                    title=Path(src).stem,
                    metadata=metadata,
                )
                print(f"  → HTML: {html_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


# ═══════════════════════════════════════════════════════════════
# Command: render
# ═══════════════════════════════════════════════════════════════

def cmd_render(args: argparse.Namespace) -> int:
    """Visualize audio without synthesis."""
    source_files = _expand_globs(args.source)

    pipeline = AudioProcessingPipeline(
        sample_rate=args.sample_rate,
    )

    viz = AudioBrainVisualizer(
        grid_size=args.grid_size,
        chars=args.chars,
        colors=args.colors,
        force_color=args.force_color,
    )

    exit_code = 0

    for src in source_files:
        if not Path(src).exists():
            print(f"Error: file not found: {src}", file=sys.stderr)
            return 1

        print(f"Processing: {Path(src).name}")
        try:
            _, latents = pipeline.process_audio(
                src,
                duration=args.duration,
            )

            if args.viz in ("terminal", "both"):
                viz.render_terminal(
                    latents,
                    title=Path(src).stem,
                    show=args.show,
                )

            if args.viz in ("html", "both"):
                html_path = args.output or f"{Path(src).stem}_visual.html"
                viz.save_html(
                    html_path,
                    latents,
                    title=Path(src).stem,
                    metadata={
                        "source":     Path(src).name,
                        "segments":   f"{latents.shape[1]}",
                        "latent_dim": f"{latents.shape[-1]}",
                        "grid":       f"{args.grid_size}×{args.grid_size}",
                    },
                )
                print(f"  → HTML: {html_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


# ═══════════════════════════════════════════════════════════════
# Command: train
# ═══════════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> int:
    """Train the AudioBrainCore model."""
    from torch import nn

    config = BrainConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        seed=args.seed,
    )

    model = AudioBrainCore(config=config)
    print(f"Model: {model.count_parameters():,} parameters on {config.device}")
    print(f"Training for {args.epochs} epochs (batch={args.batch_size}, seq={args.seq_len})...")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        batch = torch.randn(args.batch_size, args.seq_len, 2048)
        loss = model.train_step(batch, optimizer, criterion)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  loss={loss:.6f}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "audiobrain.pt"
    model.save_checkpoint(str(ckpt_path))
    print(f"Checkpoint saved: {ckpt_path}")
    return 0


# ═══════════════════════════════════════════════════════════════
# Command: info
# ═══════════════════════════════════════════════════════════════

def cmd_info(args: argparse.Namespace) -> int:
    """Display system and model metadata."""
    config = BrainConfig()
    model = AudioBrainCore(config=config)

    print("\n  AudioBrain — System Information")
    print(f"  {'='*50}")
    print(f"  Device:          {config.device}")
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  CUDA:            {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"  MPS:             {torch.backends.mps.is_available()}")
    print()
    print(f"  Model Architecture:")
    print(f"    d_model:        {config.d_model}")
    print(f"    nhead:          {config.nhead}")
    print(f"    num_layers:     {config.num_layers}")
    print(f"    dim_feedforward:{config.dim_feedforward}")
    print(f"    dropout:        {config.dropout}")
    print(f"    parameters:     {model.count_parameters():,}")
    print(f"    max_seq_len:    {config.max_seq_len}")
    print()
    print(f"  I/O Dimensions:")
    print(f"    PANNs → [batch, seq, 2048]")
    print(f"    Core  → [batch, seq, 512]")
    print()
    print(f"  Available Effects: {', '.join(EFFECT_REGISTRY)}")
    from audiobrain.model.visualizer import COLOR_SCHEMES as _CS
    print(f"  Color Schemes:     {", ".join(_CS.keys())}")
    print()
    return 0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
    gen = sub.add_parser(
        "generate",
        help="Generate new audio from source samples with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audiobrain generate song.wav --database samples/*.wav
  audiobrain generate song.wav --database samples/ --mode glitch --temperature 0.8
  audiobrain generate song.wav --effect bitcrush:bits=6,mix=0.7 --effect delay:time=400
  audiobrain generate song.wav --viz html --colors aurora --grid-size 128
        """,
    )

    # ── Input / Output ──
    io_group = gen.add_argument_group("Input / Output")
    io_group.add_argument("source", nargs="+", help="Source audio file(s)")
    io_group.add_argument("--database", "-d", nargs="+",
                          help="Database audio files for synthesis (default: same as source)")
    io_group.add_argument("--output", "-o", default="output.wav",
                          help="Output audio file path (default: output.wav)")
    io_group.add_argument("--output-dir", default=None,
                          help="Directory for batch output (default: same as --output)")
    io_group.add_argument("--duration", type=float, default=None,
                          help="Duration to process in seconds (default: full file)")

    # ── Processing ──
    proc_group = gen.add_argument_group("Audio Processing")
    proc_group.add_argument("--sample-rate", type=int, default=32000,
                            help="Target sample rate Hz (8000-96000, default: 32000)")
    proc_group.add_argument("--gain", type=float, default=0.0,
                            help="Input gain dB (-24 to +12, default: 0)")
    proc_group.add_argument("--character", choices=["raw", "warm", "bright", "dark", "airy"],
                            default="raw", help="EQ character profile (default: raw)")
    proc_group.add_argument("--norm", choices=["peak", "rms", "none"],
                            default="peak", help="Normalization mode (default: peak)")
    proc_group.add_argument("--stereo", choices=["mono", "left", "right", "stereo"],
                            default="mono", help="Channel handling (default: mono)")
    proc_group.add_argument("--trim-silence", action="store_true",
                            help="Remove leading/trailing silence")
    proc_group.add_argument("--trim-threshold", type=float, default=-60.0,
                            help="Silence threshold dB (-96 to -6, default: -60)")

    # ── Synthesis ──
    synth_group = gen.add_argument_group("Synthesis")
    synth_group.add_argument("--mode", "-m", choices=["fluid", "glitch", "evolving"],
                             default="fluid", help="Synthesis mode (default: fluid)")
    synth_group.add_argument("--temperature", "-t", type=float, default=0.5,
                             help="Creativity level 0-1 (default: 0.5)")
    synth_group.add_argument("--density", type=float, default=1.0,
                             help="Segment density 0-1 (default: 1.0)")
    synth_group.add_argument("--seed", type=int, default=None,
                             help="Random seed for reproducibility")
    synth_group.add_argument("--segment-duration", type=float, default=1.0,
                             help="Segment duration in seconds (0.5-2.0, default: 1.0)")
    synth_group.add_argument("--min-energy", type=float, default=0.01,
                             help="Minimum segment energy 0-1 (default: 0.01)")
    synth_group.add_argument("--crossfade", type=float, default=0.1,
                             help="Crossfade duration in seconds (default: 0.1)")
    synth_group.add_argument("--k-neighbors", type=int, default=5,
                             help="k-NN neighbors for synthesis (default: 5)")
    synth_group.add_argument("--max-segments", type=int, default=1000,
                             help="Max database segments (default: 1000)")

    # ── Effects ──
    fx_group = gen.add_argument_group("Audio Effects")
    fx_group.add_argument("--effect", "-fx", action="append", dest="effect", default=[],
                          metavar="NAME:KEY=VAL,...",
                          help="Apply an audio effect. Repeatable. "
                               "Format: name:param=val,param=val. "
                               "Names: bitcrush, delay, distort, flanger, glitch, pitch-down, pitch-up")

    # ── Visualization ──
    viz_group = gen.add_argument_group("Visualization")
    viz_group.add_argument("--viz", choices=["terminal", "html", "both", "none"],
                           default="html", help="Visualization output (default: html)")
    viz_group.add_argument("--grid-size", type=int, default=128,
                           help="Grid size N×N (default: 128)")
    viz_group.add_argument("--show", choices=["overlay", "spectrogram", "chladni", "both"],
                           default="overlay", help="Which view to show (default: both)")
    viz_group.add_argument("--colors", choices=["heat", "ocean", "forest", "sunset", "aurora", "mono"],
                           default="heat", help="Color scheme (default: heat)")
    viz_group.add_argument("--chars", choices=["dots", "ascii", "blocks", "braille", "binary", "shades"],
                           default="dots", help="Character set for terminal (default: dots)")
    viz_group.add_argument("--force-color", action="store_true",
                           help="Force ANSI colors even if terminal doesn't support them")


def _add_render_parser(sub: argparse._SubParsersAction) -> None:
    render = sub.add_parser(
        "render",
        help="Visualize audio latents without synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audiobrain render song.wav --viz both
  audiobrain render song.wav --viz html --colors ocean --grid-size 128
        """,
    )
    render.add_argument("source", nargs="+", help="Audio file(s) to visualize")
    render.add_argument("--output", "-o", default=None, help="HTML output path")
    render.add_argument("--duration", type=float, default=None)
    render.add_argument("--sample-rate", type=int, default=32000)
    render.add_argument("--viz", choices=["terminal", "html", "both", "none"],
                        default="terminal", help="Visualization output (default: terminal)")
    render.add_argument("--grid-size", type=int, default=128)
    render.add_argument("--show", choices=["overlay", "spectrogram", "chladni", "both"],
                        default="both")
    render.add_argument("--colors", choices=["heat", "ocean", "forest", "sunset", "aurora", "mono"],
                        default="heat")
    render.add_argument("--chars", choices=["dots", "ascii", "blocks", "braille", "binary", "shades"],
                        default="dots")
    render.add_argument("--force-color", action="store_true")


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    tr = sub.add_parser("train", help="Train the AudioBrainCore model")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--seq-len", type=int, default=60)
    tr.add_argument("--learning-rate", type=float, default=0.001)
    tr.add_argument("--d-model", type=int, default=512)
    tr.add_argument("--nhead", type=int, default=8)
    tr.add_argument("--num-layers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--checkpoint-dir", default="checkpoints")


def _add_info_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("info", help="Display system and model information")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="audiobrain",
        description="AudioBrain — Generative Soundscape System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Start:
  audiobrain info
  audiobrain generate song.wav --database samples/*.wav --viz both
  audiobrain render song.wav --viz html
  audiobrain train --epochs 50
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    _add_generate_parser(sub)
    _add_render_parser(sub)
    _add_train_parser(sub)
    _add_info_parser(sub)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "generate": cmd_generate,
        "render":   cmd_render,
        "train":    cmd_train,
        "info":     cmd_info,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
