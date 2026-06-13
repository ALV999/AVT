"""AudioBrain TUI — terminal user interface for generative soundscapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

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
from audiobrain.model.visualizer import CHAR_RAMPS, COLOR_SCHEMES
from audiobrain.processing.config import GenerationConfig, PreprocessingConfig

# ── Registry ──────────────────────────────────────────────────

EFFECT_MAP = {
    "bitcrush": bitcrush,
    "delay": delay,
    "distort": distort,
    "flanger": flange,
    "glitch": glitch,
    "pitch-down": pitch_down,
    "pitch-up": pitch_up,
}

EQ = [
    ("Raw", "raw"),
    ("Warm", "warm"),
    ("Bright", "bright"),
    ("Dark", "dark"),
    ("Airy", "airy"),
]
NORM = [("Peak", "peak"), ("RMS", "rms"), ("None", "none")]
STEREO = [("Mono", "mono"), ("Left", "left"), ("Right", "right"), ("Stereo", "stereo")]
MODE = [("Fluid", "fluid"), ("Glitch", "glitch"), ("Evolving", "evolving")]
COLORS_SEL = [(k.title(), k) for k in COLOR_SCHEMES]
CHARS_SEL = [(k.title(), k) for k in CHAR_RAMPS]


# ═══════════════════════════════════════════════════════════════


class AudioBrainTUI(App):
    """AudioBrain generative soundscape TUI."""

    CSS = """
    #left   { width: 28; border: solid $primary-background; padding: 0 1; }
    #center { width: 1fr; border: solid $primary-background; }
    #right  { width: 30; border: solid $primary-background; padding: 0 1; overflow-y: auto; }
    #fx-bar { height: 5; border: solid $primary-background; padding: 0 1; }
    #fx-btns { width: 20; }
    #actions { height: 3; padding: 0 1; }
    #strips { height: 9; }
    #viz-panel { height: 1fr; padding: 0 1; }
    #effects-label { height: 4; padding: 0 1; }
    #strip-1, #strip-2, #strip-3 { height: 3; padding: 0 1; }
    .section-title { text-style: bold; color: $text-muted; padding: 1 0 0 0; }
    Select { width: 100%; }
    Input { width: 100%; }
    Button { margin: 0 1 0 0; }
    """

    BINDINGS = [
        Binding("g", "generate", "Generate"),
        Binding("r", "render", "Render"),
        Binding("c", "cycle_colors", "Colors"),
        Binding("h", "cycle_chars", "Chars"),
        Binding("x", "clear_fx", "Clear FX"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self._source_files: list[str] = []
        self._db_files: list[str] = []
        self._latents = None
        self._waveform = None
        self._effects_list: list[str] = []
        self._color_idx = 0
        self._char_idx = 0

    # ── Compose ───────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # ── LEFT: Files ──
            with Vertical(id="left"):
                yield Label("SOURCE", classes="section-title")
                yield DirectoryTree(Path.cwd(), id="file-tree")
                yield Label("No source selected", id="source-label")
                yield Label("DB: 0 files", id="db-count")
                yield Button("Add to DB", id="btn-add-db")
                yield Button("Clear DB", id="btn-clear-db")

            # ── CENTER: Visualization ──
            with Vertical(id="center"):
                yield Label("Latent Spectrogram + Chladni", classes="section-title")
                yield Static(id="viz-panel")

            # ── RIGHT: Controls ──
            with Vertical(id="right"):
                # Preprocessing
                yield Label("PREPROCESSING", classes="section-title")
                yield Label("Sample Rate")
                yield Input(value="32000", id="sr")
                yield Label("Gain (dB)")
                yield Input(value="0.0", id="gain")
                yield Label("EQ")
                yield Select(EQ, value="raw", id="eq")
                yield Label("Normalization")
                yield Select(NORM, value="peak", id="norm")
                yield Label("Stereo")
                yield Select(STEREO, value="mono", id="stereo")

                # Synthesis
                yield Label("SYNTHESIS", classes="section-title")
                yield Label("Mode")
                yield Select(MODE, value="fluid", id="mode")
                yield Label("Temperature")
                yield Input(value="0.5", id="temp")
                yield Label("Density")
                yield Input(value="1.0", id="dens")
                yield Label("Segment (s)")
                yield Input(value="1.0", id="seg")
                yield Label("Crossfade (s)")
                yield Input(value="0.1", id="xfade")
                yield Label("k-NN")
                yield Input(value="5", id="knn")
                yield Label("Max DB segs")
                yield Input(value="500", id="maxseg")
                yield Label("Seed")
                yield Input(value="42", id="seed")

                # Output
                yield Label("OUTPUT", classes="section-title")
                yield Label("Output path")
                yield Input(value="output.wav", id="outpath")

                # Visualization
                yield Label("VISUALIZATION", classes="section-title")
                yield Label("Colors")
                yield Select(COLORS_SEL, value="ocean", id="colors")
                yield Label("Chars")
                yield Select(CHARS_SEL, value="ascii", id="chars")

        # ── Effects bar ──
        with Horizontal(id="fx-bar"):
            yield Static("Effects: none", id="effects-label")
            with Vertical(id="fx-btns"):
                for ef_name in EFFECT_MAP:
                    yield Button(f"+{ef_name}", id=f"fx-{ef_name}")

        # ── Actions ──
        with Horizontal(id="actions"):
            yield Button("▶ Generate", id="btn-generate", variant="success")
            yield Button("👁 Render", id="btn-render", variant="primary")
            yield Button("■ Stop", id="btn-stop", variant="error")
            yield ProgressBar(total=100, show_eta=False, id="progress")

        # ── Footer strips ──
        with Vertical(id="strips"):
            yield Static(" Oscilloscope: [dim]no data[/dim]", id="strip-1")
            yield Static(" Waveform: [dim]no data[/dim]", id="strip-2")
            yield Static(" Spectrogram: [dim]no data[/dim]", id="strip-3")

        yield Footer()

    # ── Actions ───────────────────────────────────────────────

    def action_generate(self):
        self._run_pipeline(generate=True)

    def action_render(self):
        self._run_pipeline(generate=False)

    def action_clear_fx(self):
        self._effects_list.clear()
        self.query_one("#effects-label", Static).update("Effects: [dim]none[/dim]")

    def action_cycle_colors(self):
        keys = list(COLOR_SCHEMES.keys())
        self._color_idx = (self._color_idx + 1) % len(keys)
        self.query_one("#colors", Select).value = keys[self._color_idx]
        self._refresh_viz()

    def action_cycle_chars(self):
        keys = list(CHAR_RAMPS.keys())
        self._char_idx = (self._char_idx + 1) % len(keys)
        self.query_one("#chars", Select).value = keys[self._char_idx]
        self._refresh_viz()

    # ── Events ────────────────────────────────────────────────

    @on(DirectoryTree.FileSelected, "#file-tree")
    def on_file_select(self, event: DirectoryTree.FileSelected):
        path = str(event.path)
        if path.lower().endswith(".wav"):
            self._source_files = [path]
            self.query_one("#source-label", Label).update(
                f"Source: [bold]{Path(path).name}[/bold]"
            )
            self.action_render()

    @on(Button.Pressed, "#btn-add-db")
    def on_add_db(self):
        if self._source_files:
            self._db_files.extend(self._source_files)
            self._db_files = list(set(self._db_files))
        self.query_one("#db-count", Label).update(
            f"DB: [bold]{len(self._db_files)} files[/bold]"
        )

    @on(Button.Pressed, "#btn-clear-db")
    def on_clear_db(self):
        self._db_files.clear()
        self.query_one("#db-count", Label).update("DB: [dim]cleared[/dim]")

    @on(Button.Pressed, "#btn-generate")
    def on_gen(self):
        self.action_generate()

    @on(Button.Pressed, "#btn-render")
    def on_ren(self):
        self.action_render()

    @on(Button.Pressed, "#btn-stop")
    def on_stop(self):
        self.query_one("#progress", ProgressBar).update(progress=0)

    @on(Button.Pressed)
    def on_fx_button(self, event: Button.Pressed):
        bid = event.button.id or ""
        if bid.startswith("fx-"):
            name = bid.replace("fx-", "")
            if name in EFFECT_MAP:
                self._effects_list.append(name)
                display = (
                    " → ".join(f"{i + 1}.{n}" for i, n in enumerate(self._effects_list))
                    if self._effects_list
                    else "none"
                )
                self.query_one("#effects-label", Static).update(
                    f"Effects: [bold]{display}[/bold]"
                )

    @on(Select.Changed)
    def on_select_change(self, event: Select.Changed):
        if event.select.id in ("colors", "chars"):
            self._refresh_viz()

    # ── Helpers ───────────────────────────────────────────────

    def _read_float(self, wid: str, default: float) -> float:
        try:
            return float(self.query_one(f"#{wid}", Input).value)
        except (ValueError, AttributeError):
            return default

    def _read_int(self, wid: str, default: int) -> int:
        try:
            return int(self.query_one(f"#{wid}", Input).value)
        except (ValueError, AttributeError):
            return default

    def _read_str(self, wid: str, default: str) -> str:
        try:
            return self.query_one(f"#{wid}", Input).value
        except AttributeError:
            return default

    def _read_select(self, wid: str, default: str) -> str:
        try:
            return self.query_one(f"#{wid}", Select).value
        except AttributeError:
            return default

    def _build_effect_chain(self) -> EffectChain | None:
        if not self._effects_list:
            return None
        chain = EffectChain()
        for name in self._effects_list:
            fn = EFFECT_MAP.get(name)
            if fn:
                chain.add(fn)
        return chain

    # ── Pipeline ──────────────────────────────────────────────

    @work(thread=True)
    def _run_pipeline(self, generate: bool = False):
        prog = self.query_one("#progress", ProgressBar)
        colors = self._read_select("colors", "ocean")
        chars = self._read_select("chars", "ascii")
        viz_panel = self.query_one("#viz-panel", Static)
        strip1 = self.query_one("#strip-1", Static)
        strip2 = self.query_one("#strip-2", Static)
        strip3 = self.query_one("#strip-3", Static)

        # Read all user params
        pp_config = PreprocessingConfig(
            target_sr=self._read_int("sr", 32000),
            gain_db=self._read_float("gain", 0.0),
            character=self._read_select("eq", "raw"),
            norm_mode=self._read_select("norm", "peak"),
            stereo_mode=self._read_select("stereo", "mono"),
        )
        gen_config = GenerationConfig(
            temperature=self._read_float("temp", 0.5),
            density=self._read_float("dens", 1.0),
            mode=self._read_select("mode", "fluid"),
            seed=self._read_int("seed", 42) or None,
            segment_duration=self._read_float("seg", 1.0),
        )
        out_path = self._read_str("outpath", "output.wav")
        max_segs = self._read_int("maxseg", 500)
        xfade = self._read_float("xfade", 0.1)
        knn = self._read_int("knn", 5)

        # Build effect chain
        effect_chain = self._build_effect_chain()

        prog.update(progress=5)

        try:
            from audiobrain.model.pipeline import AudioProcessingPipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            pipeline = AudioProcessingPipeline(
                device=device,
                sample_rate=pp_config.target_sr,
            )
            prog.update(progress=25)

            src = self._source_files[0] if self._source_files else None
            if src is None:
                self.call_from_thread(
                    viz_panel.update, " [yellow]No source file selected[/yellow]"
                )
                return

            # Extract features → latents
            self.call_from_thread(
                strip1.update,
                f" Oscilloscope: [bold yellow]Extracting features...[/bold yellow]",
            )
            _, latents = pipeline.process_audio(src, duration=5.0)
            self._latents = latents
            prog.update(progress=50)

            # Render preview
            preview = self._build_preview_text(latents, colors, chars)
            self.call_from_thread(viz_panel.update, preview)
            prog.update(progress=65)

            if generate and self._db_files:
                from audiobrain.model.synthesizer import AudioMosaicSynthesizer

                self.call_from_thread(
                    strip1.update,
                    " Oscilloscope: [bold yellow]Synthesizing...[/bold yellow]",
                )

                synth = AudioMosaicSynthesizer(
                    device=device,
                    segment_duration=gen_config.segment_duration,
                    sample_rate=pp_config.target_sr,
                    crossfade_duration=xfade,
                )
                synth.build_database(
                    audio_files=self._db_files,
                    feature_extractor=pipeline.feature_extractor,
                    pipeline=pipeline,
                    max_segments=max_segs,
                )
                audio, sr = synth.synthesize_from_latent(
                    latents,
                    self._db_files,
                    pipeline=pipeline,
                    config=gen_config,
                )

                # Apply effects
                if effect_chain and len(effect_chain) > 0:
                    self.call_from_thread(
                        strip1.update,
                        f" Oscilloscope: [bold yellow]Applying effects: {effect_chain}...[/bold yellow]",
                    )
                    audio = effect_chain.apply(audio.astype(np.float32), sr)

                self._waveform = audio.astype(np.float32)

                # Save
                import soundfile as sf

                sf.write(out_path, audio.astype(np.float32), sr)
                prog.update(progress=90)
                dur = len(audio) / sr

                self.call_from_thread(
                    strip1.update,
                    f" Oscilloscope: [bold green]Saved {out_path} ({dur:.1f}s, {sr}Hz)[/bold green]",
                )

                # Update waveform + spectrogram strips
                wf_text = self._build_waveform_text(audio)
                self.call_from_thread(strip2.update, wf_text)

                # Real spectrogram from audio
                from audiobrain.model.visualizer import RealSpectrogram

                rs = RealSpectrogram(grid_size=64)
                rs_field = rs.compute(audio.astype(np.float32), sr)
                rs_text = self._build_spec_strip_text(rs_field)
                self.call_from_thread(strip3.update, rs_text)

                # Also save HTML
                try:
                    import base64
                    import io

                    from audiobrain.model.visualizer import AudioBrainVisualizer

                    buf = io.BytesIO()
                    sf.write(buf, audio.astype(np.float32), sr, format="WAV")
                    html_path = Path(out_path).with_suffix(".html")
                    viz = AudioBrainVisualizer(
                        grid_size=128, colors=colors, chars=chars
                    )
                    viz.save_html(
                        str(html_path),
                        latents,
                        audio_data=buf.getvalue(),
                        title=Path(src).stem,
                        metadata={
                            "source": Path(src).name,
                            "mode": gen_config.mode,
                            "temperature": f"{gen_config.temperature:.2f}",
                            "duration": f"{dur:.1f}s",
                            "colors": colors,
                        },
                        waveform=audio.astype(np.float32),
                        sample_rate=sr,
                    )
                except Exception:
                    pass

            else:
                dur = latents.shape[1] * 0.5  # approximate
                self.call_from_thread(
                    strip1.update,
                    f" Oscilloscope: [bold green]Render complete ({latents.shape[1]} segments)[/bold green]",
                )

            prog.update(progress=100)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.call_from_thread(viz_panel.update, f" [red]Error: {e}[/red]")
            prog.update(progress=0)

    # ── ASCII builders ────────────────────────────────────────

    def _build_preview_text(self, latents, colors: str, chars: str) -> str:
        try:
            from audiobrain.model.visualizer import CHAR_RAMPS, AudioBrainVisualizer

            gs = 48
            charset = CHAR_RAMPS.get(chars, CHAR_RAMPS["ascii"])
            viz = AudioBrainVisualizer(grid_size=gs, chars=chars, colors=colors)
            views = viz.compute_views(latents)
            spec = views["spectrogram"]
            lm = views["chladni_lines"]
            n_ch = len(charset)
            sep = "─" * gs
            lines = ["  " + sep]
            for y in range(gs):
                row = ""
                for x in range(gs):
                    if lm[y, x]:
                        hh = (x > 0 and lm[y, x - 1]) or (x < gs - 1 and lm[y, x + 1])
                        hv = (y > 0 and lm[y - 1, x]) or (y < gs - 1 and lm[y + 1, x])
                        if hh and hv:
                            row += "┼"
                        elif hh:
                            row += "─"
                        elif hv:
                            row += "│"
                        else:
                            row += "·"
                    else:
                        v = float(spec[y, x])
                        row += charset[min(int(v * (n_ch - 1)), n_ch - 1)]
                lines.append("  " + row)
            lines.append("  " + sep)
            return "\n".join(lines)
        except Exception as e:
            return f" [red]Preview error: {e}[/red]"

    def _build_waveform_text(self, waveform) -> str:
        from audiobrain.model.visualizer import WaveformBar

        wb = WaveformBar(width=100, height=3)
        field = wb.compute(waveform)
        chs = " ·∘●"
        rows = []
        for y in range(3):
            r = "".join(chs[min(int(float(field[y, x]) * 3), 3)] for x in range(100))
            rows.append(r)
        return f" Waveform: {rows[0]}\n  {rows[1]}\n  {rows[2]}"

    def _build_spec_strip_text(self, field) -> str:
        """Compact spectrogram strip (64→40 cols downsampled)."""
        gs = field.shape[0]
        chs = " ·∘●"
        rows = []
        # Downsample to 40 cols for display
        stride = max(1, gs // 40)
        for y in range(0, gs, max(1, gs // 4)):
            r = "".join(
                chs[min(int(float(field[y, x]) * 3), 3)] for x in range(0, gs, stride)
            )
            rows.append(r)
        return " Spectrogram:\n  " + "\n  ".join(rows)

    def _refresh_viz(self):
        if self._latents is not None:
            colors = self._read_select("colors", "ocean")
            chars = self._read_select("chars", "ascii")
            text = self._build_preview_text(self._latents, colors, chars)
            self.query_one("#viz-panel", Static).update(text)


def run():
    AudioBrainTUI().run()


if __name__ == "__main__":
    run()
