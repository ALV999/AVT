"""Custom Textual widgets for AudioBrain TUI."""

from __future__ import annotations

import numpy as np
from textual.widgets import Static


class VisualizationPanel(Static):
    """Renders the N×N latent spectrogram + Chladni overlay as ASCII art."""

    DEFAULT_CSS = """
    VisualizationPanel {
        height: 1fr;
        padding: 0 1;
        color: $text;
        background: $surface;
    }
    """

    def __init__(self, grid_size: int = 48, **kwargs):
        super().__init__("", **kwargs)
        self.grid_size = grid_size
        self._latents = None
        self._colors_name = "ocean"
        self._chars_name = "ascii"

    def on_mount(self):
        self._build_preview()

    def set_data(self, latents, colors="ocean", chars="ascii"):
        self._latents = latents
        self._colors_name = colors
        self._chars_name = chars
        self._build_preview()

    def _build_preview(self):
        gs = self.grid_size
        sep = "─" * gs
        blank = "·" * gs

        if self._latents is None:
            lines = ["  AudioBrain", "", "  " + sep]
            for _ in range(gs):
                lines.append("  " + blank)
            lines.append("  " + sep)
            self.update("\n".join(lines))
            return

        try:
            from audiobrain.model.visualizer import (
                AudioBrainVisualizer, CHAR_RAMPS, COLOR_SCHEMES,
            )
            chars = CHAR_RAMPS.get(self._chars_name, CHAR_RAMPS["ascii"])
            viz = AudioBrainVisualizer(
                grid_size=gs,
                chars=self._chars_name,
                colors=self._colors_name,
            )
            views = viz.compute_views(self._latents)
            spec = views["spectrogram"]
            lines_mask = views["chladni_lines"]

            n_chars = len(chars)
            output = ["  " + sep]
            for y in range(gs):
                row = ""
                for x in range(gs):
                    if lines_mask[y, x]:
                        has_h = (x > 0 and lines_mask[y, x-1]) or (x < gs-1 and lines_mask[y, x+1])
                        has_v = (y > 0 and lines_mask[y-1, x]) or (y < gs-1 and lines_mask[y+1, x])
                        if has_h and has_v:
                            ch = "┼"
                        elif has_h:
                            ch = "─"
                        elif has_v:
                            ch = "│"
                        else:
                            ch = "·"
                    else:
                        v = float(spec[y, x])
                        idx = min(int(v * (n_chars - 1)), n_chars - 1)
                        ch = chars[idx]
                    row += ch
                output.append("  " + row)
            output.append("  " + sep)
            self.update("\n".join(output))
        except Exception as e:
            self.update(f"  Render error: {e}")


class MiniStrip(Static):
    """Compact horizontal strip for waveforms."""

    DEFAULT_CSS = """
    MiniStrip {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(self, label: str = "", **kwargs):
        super().__init__("", **kwargs)
        self.label = label

    def on_mount(self):
        self.update(f" {self.label}: [dim]no data[/dim]")

    def set_waveform(self, waveform: np.ndarray | None):
        if waveform is None or len(waveform) == 0:
            self.update(f" {self.label}: [dim]no data[/dim]")
            return
        try:
            from audiobrain.model.visualizer import WaveformBar
            wb = WaveformBar(width=100, height=3)
            field = wb.compute(waveform)
            chars = " ·∘●"
            rows = []
            for y in range(field.shape[0]):
                row = ""
                for x in range(field.shape[1]):
                    v = float(field[y, x])
                    idx = min(int(v * 3), 3)
                    row += chars[idx]
                rows.append(row)
            display = f" {self.label}: " + rows[0] + "\n   " + rows[1] + "\n   " + rows[2]
            self.update(display)
        except Exception as e:
            self.update(f" {self.label}: [dim]{e}[/dim]")


class EffectChainWidget(Static):
    """Displays the audio effect chain."""

    DEFAULT_CSS = """
    EffectChainWidget {
        height: 4;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._effects: list[tuple[str, dict]] = []

    def on_mount(self):
        self.update(" Effects: [dim]none[/dim]")

    def add(self, name: str, params: dict | None = None):
        self._effects.append((name, params or {}))
        self._render_effects()

    def remove(self, index: int):
        if 0 <= index < len(self._effects):
            self._effects.pop(index)
            self._render_effects()

    def clear(self):
        self._effects.clear()
        self._render_effects()

    def _render_effects(self):
        if not self._effects:
            self.update(" Effects: [dim]none[/dim]")
            return
        parts = [" Effects: "]
        for i, (name, params) in enumerate(self._effects):
            param_str = " ".join(f"{k}={v}" for k, v in params.items())
            parts.append(f"{i+1}. [bold]{name}[/bold] {param_str}")
        self.update("  ".join(parts))
