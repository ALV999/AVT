"""
Artistic Visualization Module — 128x128 latent spectrogram with Chladni overlay.

Two complementary views derived entirely from the ML model's latent vectors:
  1. Latent Spectrogram: 128x128 grid where (t, dim) = activation of learned feature
  2. Chladni Nodal Lines: circular plate interference patterns from latent mode
     amplitudes, rendered as organic line art overlaid on the spectrogram.

The 512-dimensional latent space is the product of:
  PANNs/AST feature extraction → ProjectionHead(2048→512) → SoundscapeTransformer(512)

Each latent dimension has learned to represent some aspect of the audio. We harvest
that representation to create artistic visualizations that ARE the ML product.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════
# Color support
# ═══════════════════════════════════════════════════════════════

_colorama_available = False
try:
    import colorama
    colorama.init()
    _colorama_available = True
except ImportError:
    pass


def _supports_color() -> bool:
    if not _colorama_available:
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                mode = ctypes.c_ulong()
                handle = kernel32.GetStdHandle(-11)
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                return bool(mode.value & 0x0004)
            except Exception:
                return False
        if not sys.stdout.isatty():
            return False
        term = os.environ.get('TERM', '')
        if term in ('dumb', ''):
            return False
        return True
    return True


def _ansi_rgb(r: int, g: int, b: int, char: str) -> str:
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"


def _plain_char(r, g, b, char):
    return char


_color_fn = _ansi_rgb if _supports_color() else _plain_char

# ═══════════════════════════════════════════════════════════════
# Character ramps
# ═══════════════════════════════════════════════════════════════

CHAR_RAMPS: dict[str, str] = {
    "dots":     " ·∘●",
    "ascii":    " .:-=+*#%@",
    "blocks":   " ░▒▓█",
    "braille":  " ⠁⠃⠇⠏⠟⠿",
    "binary":   " 01",
    "shades":   " ░░▒▒▓▓██",
}

# ═══════════════════════════════════════════════════════════════
# Color schemes
# ═══════════════════════════════════════════════════════════════

def _interp_heat(v: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, v))
    if v < 0.33:
        t = v / 0.33
        return (int(255 * t), 0, 0)
    elif v < 0.66:
        t = (v - 0.33) / 0.33
        return (255, int(255 * t), 0)
    else:
        t = (v - 0.66) / 0.34
        return (255, 255, int(255 * t))


def _interp_ocean(v: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, v))
    if v < 0.33:
        t = v / 0.33
        return (0, 0, int(128 * t))
    elif v < 0.66:
        t = (v - 0.33) / 0.33
        return (0, int(200 * t), 128 + int(127 * t))
    else:
        t = (v - 0.66) / 0.34
        return (int(200 * t), 200 + int(55 * t), 255)


def _interp_forest(v: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        t = v / 0.5
        return (0, int(180 * t), 0)
    else:
        t = (v - 0.5) / 0.5
        return (int(255 * t), 180 + int(75 * t), int(255 * t))


def _interp_sunset(v: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, v))
    if v < 0.33:
        t = v / 0.33
        return (int(128 * (1 - t) + 200 * t), 0, int(128 * (1 - t)))
    elif v < 0.66:
        t = (v - 0.33) / 0.33
        return (200 + int(55 * t), int(100 * t), 0)
    else:
        t = (v - 0.66) / 0.34
        return (255, 100 + int(155 * t), int(200 * t))


def _interp_aurora(v: float) -> tuple[int, int, int]:
    """Teal → purple → gold gradient."""
    v = max(0.0, min(1.0, v))
    if v < 0.33:
        t = v / 0.33
        return (0, int(128 + 127 * t), int(128 + 127 * (1 - t)))
    elif v < 0.66:
        t = (v - 0.33) / 0.33
        return (int(128 * t), int(255 * (1 - t)), int(128 * t))
    else:
        t = (v - 0.66) / 0.34
        return (128 + int(127 * t), int(128 * t), int(128 + 127 * (1 - t)))


def _interp_mono(v: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, v))
    g = int(v * 255)
    return (g, g, g)


COLOR_SCHEMES: dict[str, callable] = {
    "heat":   _interp_heat,
    "ocean":  _interp_ocean,
    "forest": _interp_forest,
    "sunset": _interp_sunset,
    "aurora": _interp_aurora,
    "mono":   _interp_mono,
}


# ═══════════════════════════════════════════════════════════════
# Latent Spectrogram — 128x128 view from ML latent vectors
# ═══════════════════════════════════════════════════════════════

class LatentSpectrogram:
    """
    128x128 spectrogram-like field derived entirely from latent vectors.

    The ML pipeline outputs [1, N, 512] vectors. We interpret:
      - Time axis: N segments → interpolated to 128 steps
      - Frequency axis: first 128 of the 512 latent dimensions

    Each latent dimension has learned (through the Transformer's self-supervised
    training) to represent some audio characteristic. By treating the raw dimension
    activations as "frequency bins", we create a spectrogram that IS the model's
    understanding of the sound.
    """

    def __init__(self, grid_size: int = 128):
        self.grid_size = grid_size

    def compute(self, latents: torch.Tensor) -> np.ndarray:
        """Map latent vectors to a [grid_size, grid_size] spectrogram grid."""
        if latents.dim() == 3:
            latents = latents[0]
        latents = latents.cpu().numpy().astype(np.float32)

        n_steps, n_dims = latents.shape

        # Time axis: interpolate N steps → grid_size steps
        if n_steps != self.grid_size:
            x_old = np.linspace(0, 1, n_steps)
            x_new = np.linspace(0, 1, self.grid_size)
            latents_interp = np.zeros((self.grid_size, n_dims), dtype=np.float32)
            for d in range(n_dims):
                latents_interp[:, d] = np.interp(x_new, x_old, latents[:, d])
            latents = latents_interp

        # Frequency axis: take first grid_size dimensions (learned features)
        freq_dims = min(n_dims, self.grid_size)
        spectrogram = latents[:, :freq_dims]

        if freq_dims < self.grid_size:
            pad = np.zeros((self.grid_size, self.grid_size - freq_dims), dtype=np.float32)
            spectrogram = np.concatenate([spectrogram, pad], axis=1)

        # Per-column percentile normalization
        result = np.zeros_like(spectrogram)
        for col in range(self.grid_size):
            col_data = spectrogram[:, col]
            lo = np.percentile(col_data, 5)
            hi = np.percentile(col_data, 95)
            rng = hi - lo
            if rng > 1e-8:
                result[:, col] = np.clip((col_data - lo) / rng, 0.0, 1.0)
            else:
                result[:, col] = 0.5

        return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Chladni Oscilloscope — circular plate, nodal line art
# ═══════════════════════════════════════════════════════════════


class WaveformBar:
    """Compact horizontal waveform bar."""
    def __init__(self, width=128, height=12):
        self.width = width
        self.height = height

    def compute(self, waveform):
        if len(waveform) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        n = len(waveform)
        chunk = max(1, n // self.width)
        result = np.zeros((self.height, self.width), dtype=np.float32)
        for c in range(self.width):
            start = c * chunk
            end = min(start + chunk, n)
            if start >= n: break
            seg = waveform[start:end]
            if len(seg) == 0: continue
            peak = float(np.abs(seg).max())
            peak = min(peak, 1.0)
            center = self.height // 2
            bar_height = int(peak * center)
            for h in range(bar_height):
                if center + h < self.height:
                    result[center + h, c] = (h + 1) / bar_height if bar_height > 0 else 0.0
                if center - h - 1 >= 0:
                    result[center - h - 1, c] = (h + 1) / bar_height if bar_height > 0 else 0.0
        return result.astype(np.float32)


class WaveformLine:
    """Traditional line oscilloscope trace."""
    def __init__(self, width=256, height=40):
        self.width = width
        self.height = height

    def compute_points(self, waveform):
        if len(waveform) == 0:
            return np.zeros((self.width, 2), dtype=np.float32)
        n = len(waveform)
        chunk = max(1, n // self.width)
        mid = self.height / 2.0
        points = np.zeros((self.width, 2), dtype=np.float32)
        for i in range(self.width):
            start = i * chunk
            end = min(start + chunk, n)
            if start >= n:
                points[i, 1] = mid
                points[i, 0] = float(i)
                continue
            seg = waveform[start:end]
            if len(seg) == 0:
                points[i, 1] = mid
            else:
                peak = float(np.abs(seg).max())
                peak = min(peak, 1.0)
                points[i, 1] = mid - peak * (mid - 1)
            points[i, 0] = float(i)
        return points

    def to_svg(self, waveform, width_px=640, height_px=60):
        pts = self.compute_points(waveform)
        scale_x = width_px / self.width
        scale_y = height_px / self.height * 0.8
        offset_y = height_px * 0.5
        svg_pts = []
        for i in range(len(pts)):
            x = pts[i, 0] * scale_x
            y = offset_y + (pts[i, 1] - self.height / 2) * scale_y
            svg_pts.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(svg_pts)
        return (
            f'<polyline points="{polyline}" fill="none" '
            f'stroke="rgba(255,255,255,0.5)" stroke-width="1" '
            f'vector-effect="non-scaling-stroke"/>'
        )


class RealSpectrogram:
    """Traditional STFT spectrogram."""
    def __init__(self, grid_size=64):
        self.grid_size = grid_size

    def compute(self, waveform, sample_rate):
        gs = self.grid_size
        if len(waveform) == 0:
            return np.zeros((gs, gs), dtype=np.float32)
        nperseg = gs * 2
        hop = max(1, len(waveform) // gs)
        n_frames = min(gs, max(1, (len(waveform) - nperseg) // hop + 1))
        if n_frames < 2:
            return np.zeros((gs, gs), dtype=np.float32)
        window = np.hanning(nperseg)
        spec = np.zeros((gs // 2 + 1, gs), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            end = min(start + nperseg, len(waveform))
            if end - start < nperseg // 2: continue
            frame = np.zeros(nperseg, dtype=np.float32)
            seg = waveform[start:end]
            frame[:len(seg)] = seg * window[:len(seg)]
            fft = np.abs(np.fft.rfft(frame))
            col = int(i * gs / n_frames)
            if col >= gs: col = gs - 1
            bins = min(len(fft), gs // 2 + 1)
            spec[:bins, col] += fft[:bins]
        spec = np.flipud(spec[:gs // 2 + 1, :gs])
        if spec.shape[0] > gs: spec = spec[:gs, :]
        elif spec.shape[0] < gs:
            pad = np.zeros((gs - spec.shape[0], gs), dtype=np.float32)
            spec = np.concatenate([pad, spec], axis=0)
        if spec.shape[1] > gs: spec = spec[:, :gs]
        elif spec.shape[1] < gs:
            pad = np.zeros((gs, gs - spec.shape[1]), dtype=np.float32)
            spec = np.concatenate([spec, pad], axis=1)
        spec = np.log1p(spec)
        vmin = spec.min(); vmax = spec.max()
        if vmax - vmin > 1e-8:
            spec = (spec - vmin) / (vmax - vmin)
        else: spec.fill(0.5)
        return spec.astype(np.float32)


class ChladniOscilloscope:
    """
    128x128 nodal line patterns from latent vectors — circular Chladni plate.

    Real Chladni patterns form when a circular plate vibrates at resonant
    frequencies. Sand collects at the nodes (still points) creating organic
    concentric rings, radial spokes, and interference patterns.

    We aggregate latent mode amplitudes across all time steps, build the FULL
    2D circular Chladni field, detect zero-crossings (nodal lines), and
    compute variable line thickness from the local gradient magnitude.
    """

    def __init__(self, grid_size: int = 128, num_modes: int = 128):
        self.grid_size = grid_size
        self.num_modes = min(num_modes, 128)

        # Build mode list: (angular_order, radial_order)
        self._modes: list[tuple[int, int]] = []
        for i in range(self.num_modes):
            m = i % 16           # 0=rings, 1=diameter, 2=cross, ...
            n = (i // 16) + 1    # 1=one ring, 2=two rings, ...
            self._modes.append((m, n))

        # Precompute spatial coordinates centered in the grid
        gs = grid_size
        y_idx, x_idx = np.indices((gs, gs))
        cx = (gs - 1) / 2.0
        cy = (gs - 1) / 2.0
        # Normalize radius so corners reach 1.0 (fills full square grid)
        self._radius = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2)
        self._radius = self._radius / np.sqrt(2.0)  # corners = 1.0
        self._radius = np.clip(self._radius, 0.0, 1.0)
        self._theta = np.arctan2(y_idx - cy, x_idx - cx)

    def compute(self, latents: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FULL 2D circular Chladni field from aggregated latent data.

        Instead of sampling row-by-row, we average mode amplitudes across all
        time steps and build the complete 128x128 circular interference pattern.

        Args:
            latents: [1, N, 512] or [N, 512] from AudioBrainCore.

        Returns:
            (line_mask, thickness):
                line_mask: [128, 128] bool — nodal lines
                thickness: [128, 128] float32 — line width (0=no line, 1=thick)
        """
        if latents.dim() == 3:
            latents = latents[0]
        latents = latents.cpu().numpy().astype(np.float32)

        n_time, n_dims = latents.shape
        gs = self.grid_size

        # ── Aggregate latent amplitudes across time ──
        # Each latent dimension pair = (amplitude, phase) for one mode
        num_mode_pairs = min(self.num_modes, n_dims // 4)
        # Take the RMS amplitude across time for each mode
        avg_amps = np.zeros(num_mode_pairs, dtype=np.float32)
        avg_phases = np.zeros(num_mode_pairs, dtype=np.float32)
        for i in range(num_mode_pairs):
            amp_dim = 4 * i
            phase_dim = 4 * i + 1
            avg_amps[i] = np.sqrt(np.mean(latents[:, amp_dim] ** 2))
            avg_phases[i] = np.mean(latents[:, phase_dim]) * np.pi

        # ── Build full 2D Chladni field ──
        field = np.zeros((gs, gs), dtype=np.float32)
        for i in range(num_mode_pairs):
            amp = avg_amps[i]
            if amp < 1e-4:
                continue
            phase = avg_phases[i]
            m, n = self._modes[i]

            # Radial: Bessel-like radial oscillation
            radial = np.sin(np.pi * n * self._radius)
            # Angular: cos(m·θ + φ)
            angular = np.cos(m * self._theta + phase)
            frame = amp * radial * angular
            field += frame

        # ── Detect nodal lines with thickness ──
        line_mask, thickness = self._detect_nodal_lines_with_thickness(field)

        return line_mask, thickness

    def _detect_nodal_lines_with_thickness(
        self, field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect zero-crossings and compute line thickness from gradient.

        Where the field changes sign between adjacent cells, we mark a nodal
        line. The thickness is inversely proportional to the gradient magnitude:
          - Steep crossing → thin, crisp line
          - Gentle crossing → thicker, sandy line

        Returns:
            line_mask: [gs, gs] bool
            thickness: [gs, gs] float32 in [0, 1]
        """
        gs = self.grid_size
        sign_field = np.sign(field)

        # Compute gradient magnitude (Sobel-like)
        gy = np.zeros_like(field)
        gx = np.zeros_like(field)
        gy[1:-1] = field[2:] - field[:-2]
        gx[:, 1:-1] = field[:, 2:] - field[:, :-2]
        grad_mag = np.sqrt(gy ** 2 + gx ** 2)

        # Normalize gradient to [0, 1] per column (for even distribution)
        grad_norm = np.zeros_like(grad_mag)
        for col in range(gs):
            col_grad = grad_mag[:, col]
            hi = np.percentile(col_grad, 95)
            if hi > 1e-8:
                grad_norm[:, col] = np.clip(col_grad / hi, 0.0, 1.0)
            else:
                grad_norm[:, col] = 0.5

        # Find sign changes (all 4 directions)
        lines = np.zeros((gs, gs), dtype=bool)
        h_edges = sign_field[:, :-1] != sign_field[:, 1:]
        lines[:, :-1] |= h_edges
        lines[:, 1:] |= h_edges
        v_edges = sign_field[:-1, :] != sign_field[1:, :]
        lines[:-1, :] |= v_edges
        lines[1:, :] |= v_edges
        d1_edges = sign_field[:-1, :-1] != sign_field[1:, 1:]
        lines[:-1, :-1] |= d1_edges
        lines[1:, 1:] |= d1_edges
        d2_edges = sign_field[:-1, 1:] != sign_field[1:, :-1]
        lines[:-1, 1:] |= d2_edges
        lines[1:, :-1] |= d2_edges

        # Thickness: inverse of gradient at line positions
        # High gradient → thin (0.0), low gradient → thick (1.0)
        thickness = np.zeros((gs, gs), dtype=np.float32)
        thickness[lines] = np.clip(1.0 - grad_norm[lines], 0.05, 1.0)

        return lines, thickness.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Unified Visualizer — with overlay support
# ═══════════════════════════════════════════════════════════════

class AudioBrainVisualizer:
    """
    Artistic visualizer for AudioBrain latent space.

    Two complementary 128x128 views derived entirely from the ML product:
      - Latent spectrogram: (t, dim) activation grid — colored background
      - Chladni nodal lines: circular plate interference line art — overlay

    Output modes:
      - Terminal: ANSI-colored ASCII art with character-based overlay
      - HTML: self-contained dark-themed page with CSS grid + SVG overlay
    """

    def __init__(
        self,
        grid_size: int = 128,
        chars: str = "dots",
        colors: str = "heat",
        chladni_chars: str = "lines",
        chladni_color: str = "white",
        force_color: bool = False,
    ):
        self.grid_size = grid_size
        self.chars = CHAR_RAMPS.get(chars, CHAR_RAMPS["ascii"])
        self.colors = COLOR_SCHEMES.get(colors, COLOR_SCHEMES["heat"])
        self.chladni_chars = chladni_chars
        self.chladni_color = chladni_color
        self._use_color = force_color or _supports_color()

        self.spectrogram = LatentSpectrogram(grid_size=grid_size)
        self.oscilloscope = ChladniOscilloscope(grid_size=grid_size)
        self.waveform = WaveformBar(width=grid_size, height=12)
        self.waveform_line = WaveformLine(width=256, height=40)
        self.real_spec = RealSpectrogram(grid_size=grid_size)

    # ── Core computation ──────────────────────────────────────

    def compute_views(self, latents: torch.Tensor) -> dict:
        """Compute both views. Chladni returns (line_mask, thickness)."""
        lines, thickness = self.oscilloscope.compute(latents)
        return {
            "spectrogram": self.spectrogram.compute(latents),
            "chladni_lines": lines,
            "chladni_thickness": thickness,
        }

    # ── ANSI terminal with overlay ────────────────────────────

    def _field_to_ansi(self, field: np.ndarray) -> list[str]:
        """Render a 2D field as ANSI-colored ASCII rows."""
        n_chars = len(self.chars)
        rows = []
        for y in range(self.grid_size):
            parts = []
            for x in range(self.grid_size):
                v = float(field[y, x])
                idx = min(int(v * (n_chars - 1)), n_chars - 1)
                ch = self.chars[idx]
                if self._use_color:
                    r, g, b = self.colors(v)
                    parts.append(_color_fn(r, g, b, ch))
                else:
                    parts.append(ch)
            rows.append("".join(parts))
        return rows

    def _field_to_ansi_overlay(
        self, field: np.ndarray, line_mask: np.ndarray
    ) -> list[str]:
        """
        Render spectrogram with Chladni lines overlaid.

        Spectrogram cells get their normal color character.
        Where a Chladni nodal line crosses, the character is replaced
        with a bright line-drawing character.
        """
        n_chars = len(self.chars)
        rows = []
        for y in range(self.grid_size):
            parts = []
            for x in range(self.grid_size):
                v = float(field[y, x])
                if line_mask[y, x]:
                    # Nodal line: use bright line character
                    # Pick direction-appropriate glyph
                    has_h = (x > 0 and line_mask[y, x - 1]) or (x < self.grid_size - 1 and line_mask[y, x + 1])
                    has_v = (y > 0 and line_mask[y - 1, x]) or (y < self.grid_size - 1 and line_mask[y + 1, x])
                    if has_h and has_v:
                        ch = "┼"
                    elif has_h:
                        ch = "─"
                    elif has_v:
                        ch = "│"
                    else:
                        ch = "·"
                    if self._use_color:
                        # Bright white/contrast for lines
                        r, g, b = self.colors(0.9)
                        parts.append(_color_fn(255, 255, 255, ch))
                    else:
                        parts.append(ch)
                else:
                    idx = min(int(v * (n_chars - 1)), n_chars - 1)
                    ch = self.chars[idx]
                    if self._use_color:
                        r, g, b = self.colors(v)
                        parts.append(_color_fn(r, g, b, ch))
                    else:
                        parts.append(ch)
            rows.append("".join(parts))
        return rows

    def render_terminal(
        self,
        latents: torch.Tensor,
        title: str = "AudioBrain",
        show: str = "overlay",
    ) -> str:
        """
        Render to terminal.

        Args:
            latents: [1, N, 512] latent vectors.
            title: Display title.
            show: "overlay" (spectrogram + Chladni lines),
                  "spectrogram", "chladni", or "both" (separate panels).

        Returns:
            Terminal string with ANSI escapes.
        """
        views = self.compute_views(latents)
        sep = "  " + "─" * self.grid_size
        output_lines = []
        output_lines.append(f"\n  {title} — {self.grid_size}×{self.grid_size}")
        output_lines.append(f"  Chars: {self.chars}  Colors: {list(COLOR_SCHEMES.keys())}")
        output_lines.append("")

        if show == "overlay":
            output_lines.append("  [ Spectrogram + Chladni Nodal Lines ]")
            output_lines.append(sep)
            for row in self._field_to_ansi_overlay(
                views["spectrogram"], views["chladni_lines"]
            ):
                output_lines.append(f"  {row}")
            output_lines.append(sep)
            output_lines.append("")
        elif show in ("spectrogram", "both"):
            output_lines.append("  [ Latent Spectrogram ]")
            output_lines.append(sep)
            for row in self._field_to_ansi(views["spectrogram"]):
                output_lines.append(f"  {row}")
            output_lines.append(sep)
            output_lines.append("")
        if show in ("chladni", "both"):
            output_lines.append("  [ Chladni Nodal Lines ]")
            output_lines.append(sep)
            for row in self._field_to_ansi_overlay(
                np.zeros_like(views["spectrogram"]), views["chladni_lines"]
            ):
                output_lines.append(f"  {row}")
            output_lines.append(sep)
            output_lines.append("")

        result = "\n".join(output_lines)
        print(result)
        return result

    # ── HTML rendering — spectrogram with Chladni SVG overlay ──

    def build_html(
        self,
        latents: torch.Tensor,
        audio_data: bytes | None = None,
        title: str = "AudioBrain",
        metadata: dict | None = None,
        waveform: np.ndarray | None = None,
        sample_rate: int = 32000,
    ) -> str:
        """
        Build self-contained HTML with waveform bar, real spectrogram,
        latent spectrogram, and Chladni nodal line overlay.
        """
        views = self.compute_views(latents)
        gs = self.grid_size

        spec = views["spectrogram"]
        lines = views["chladni_lines"]

        # ── Spectrogram as pixel-perfect PPM image ──
        spec_img = self._field_to_data_url(spec)

        # ── Chladni nodal lines as SVG ──
        svg_paths = self._lines_to_svg(lines, gs, thickness=views.get("chladni_thickness"))

        # ── Bottom strip: oscilloscope line + waveform bar + spectrogram ──
        bottom_html = ""
        if waveform is not None and len(waveform) > 0:
            line_svg = self.waveform_line.to_svg(waveform, width_px=640, height_px=60)
            wf_field = self.waveform.compute(waveform)
            wf_img = self._field_to_data_url(wf_field)
            wf_rows = self.waveform.height
            wf_cols = self.waveform.width
            real_field = self.real_spec.compute(waveform, sample_rate)
            real_img = self._field_to_data_url(real_field)
            bottom_html = (
                '<div class="strip-box">\n'
                '  <span class="box-label">Oscilloscope</span>\n'
                '  <div class="strip">\n'
                f'    <svg viewBox="0 0 640 60" preserveAspectRatio="xMidYMid meet">\n      {line_svg}\n    </svg>\n'
                '  </div>\n'
                '</div>\n'
                '<div class="strip-box">\n'
                '  <span class="box-label">Waveform</span>\n'
                '  <div class="strip">\n'
                f'    <img src="{wf_img}" style="width:100%;height:100%;display:block;image-rendering:pixelated;" alt="">\n'
                '  </div>\n'
                '</div>\n'
                '<div class="strip-box">\n'
                '  <span class="box-label">Spectrogram</span>\n'
                '  <div class="strip">\n'
                f'    <img src="{real_img}" style="width:100%;height:100%;display:block;image-rendering:pixelated;" alt="">\n'
                '  </div>\n'
                '</div>\n'
            )


        # ── Audio player ──
        audio_html = ""
        if audio_data:
            b64 = base64.b64encode(audio_data).decode("utf-8")
            audio_html = (
                '<div class="player">\n'
                f'  <audio controls src="data:audio/wav;base64,{b64}"></audio>\n'
                '</div>\n'
            )

        # ── Metadata ──
        meta_html = ""
        if metadata:
            for k, v in metadata.items():
                meta_html += (
                    f'<div class="meta"><span class="key">{k}</span> '
                    f'<span class="val">{v}</span></div>\n'
                )

        n_segs = latents.shape[1] if latents.dim() == 3 else latents.shape[0]
        n_dims = latents.shape[-1]

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AudioBrain — {title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #08080a;
    color: #888;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    padding: 2em;
    min-height: 100vh;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{
    font-size: 11px; font-weight: 400; color: #555;
    letter-spacing: 0.4em; text-transform: uppercase;
    margin-bottom: 0.3em; padding: 0 1em;
  }}
  h2 {{
    font-size: 18px; font-weight: 400; color: #ccc;
    margin-bottom: 0.15em; padding: 0 1em;
  }}
  .subtitle {{
    color: #444; margin-bottom: 2em; font-size: 10px; padding: 0 1em;
  }}
  /* ── Box: outlined container for each panel ── */
  .box {{
    border: 1px solid #1a1a1a;
    padding: 1.2em;
    margin: 0 0 1.5em 0;
    position: relative;
  }}
  .box-label {{
    position: absolute; top: -8px; left: 12px;
    background: #08080a; padding: 0 8px;
    color: #333; font-size: 9px;
    letter-spacing: 0.2em; text-transform: uppercase;
  }}
  /* ── Main canvas ── */
  .canvas-wrap {{
    position: relative;
    width: 100%;
    aspect-ratio: 1/1;
    margin: 0 auto;
    image-rendering: pixelated;
  }}
  .spec-layer {{
    display: grid;
    grid-template-columns: repeat({gs}, 1fr);
    grid-template-rows: repeat({gs}, 1fr);
    position: absolute; inset: 0;
    width: 100%; height: 100%; gap: 0;
  }}
  .spec-layer .cell {{
    width: 100%; height: 100%; min-width: 0; min-height: 0;
    border: 0; outline: 0;
  }}
  .chladni-layer {{
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 2;
  }}
  .chladni-layer rect {{ fill: #ffffff; }}
  /* ── Bottom strips ── */
  .strip-box {{
    border: 1px solid #1a1a1a;
    padding: 1em;
    margin: 0 0 1em 0;
    position: relative;
  }}
  .strip {{
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .strip svg {{ width: 100%; height: 100%; }}
  .strip-row {{
    display: flex; gap: 1em; flex-wrap: wrap;
  }}
  .strip-panel {{
    flex: 1 1 100%;
    border: 1px solid #1a1a1a;
    padding: 0.8em; position: relative;
  }}
  .meta {{ margin: 0.3em 0; font-size: 10px; }}
  .key {{ color: #444; }}
  .val {{ color: #777; }}
  .player {{ margin: 0 0 2em 0; }}
  audio {{ width: 100%; filter: invert(0.85); opacity: 0.7; }}
  audio:hover {{ opacity: 1; }}
  .section-label {{
    color: #333; font-size: 9px; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 0.8em;
  }}
  .legend {{
    display: flex; justify-content: center; gap: 2em;
    margin: 1em 0 2em; font-size: 9px; color: #333;
  }}
  .legend span {{ display: flex; align-items: center; gap: 5px; }}
  .legend .swatch {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  }}
  footer {{
    margin-top: 3em; font-size: 9px; color: #222; text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>AudioBrain</h1>
  <h2>{title}</h2>
  <div class="subtitle">
    {n_segs} segments &middot; {n_dims}-dim latent space &middot; {gs}x{gs} grid
  </div>

  {audio_html}

  <div class="box">
    <span class="box-label">Latent Spectrogram + Chladni Nodal Lines</span>
    <div class="canvas-wrap">
    <img src="{spec_img}" style="width:100%;height:100%;display:block;image-rendering:pixelated;" alt="">
    <svg class="chladni-layer" viewBox="0 0 {gs} {gs}" preserveAspectRatio="none">
      {svg_paths}
    </svg>
    </div>
  </div>

  {bottom_html}

  <div class="legend">
    <span><span class="swatch" style="background:rgb(0,0,0);"></span> latent dim activation</span>
    <span><span class="swatch" style="background:rgb(255,255,200);"></span> chladni nodal lines</span>
  </div>

  <div class="box">
    <span class="box-label">Metadata</span>
    {meta_html}
  </div>

  <footer>AudioBrain v0.2.0 &middot; spectrogram + chladni oscilloscope &middot; {n_segs} segments</footer>
</div>
</body>
</html>'''

        return html

    def _lines_to_svg(self, line_mask, gs, thickness=None):
        """Convert nodal lines to SVG rects with variable thickness."""
        rects = []
        for y in range(gs):
            for x in range(gs):
                if not line_mask[y, x]:
                    continue
                if thickness is not None:
                    t = float(thickness[y, x])
                    sz = max(1, int(1 + t * 1.5))
                else:
                    sz = 1
                rects.append(
                    f'<rect x="{x}" y="{y}" width="{sz}" height="{sz}" '
                    f'fill="#ffffff"/>'
                )
        return "\n      ".join(rects)

    def _field_to_data_url(self, field):
        """Convert a 2D field [H,W] in [0,1] to a base64 PPM image — zero gaps."""
        import base64
        h, w = field.shape
        # Scale to 0-255 RGB bytes
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                r, g, b = self.colors(float(field[y, x]))
                rgb[y, x] = [r, g, b]
        # PPM P6 header
        header = f'P6\n{w} {h}\n255\n'.encode('ascii')
        data = header + rgb.tobytes()
        b64 = base64.b64encode(data).decode('ascii')
        return f'data:image/x-portable-pixmap;base64,{b64}'

    def _build_css_grid(self, field, prefix):
        """Build CSS grid cells for a square field."""
        gs = field.shape[0]
        cells = []
        for y in range(gs):
            for x in range(gs):
                v = float(field[y, x])
                r, g, b = self.colors(v)
                cells.append(
                    f'<div class="cell" '
                    f'style="background:rgb({r},{g},{b});" '
                    f'data-{prefix}-y="{y}" data-{prefix}-x="{x}">'
                    f'</div>'
                )
        return "\n    ".join(cells)

    def _build_css_grid_2d(self, field, prefix, h, w):
        """Build CSS grid cells with arbitrary dimensions."""
        cells = []
        for y in range(h):
            for x in range(w):
                v = float(field[y, x])
                r, g, b = self.colors(v)
                cells.append(
                    f'<div class="cell" '
                    f'style="background:rgb({r},{g},{b});" '
                    f'data-{prefix}-y="{y}" data-{prefix}-x="{x}">'
                    f'</div>'
                )
        return "\n    ".join(cells)

    def save_html(self, path, latents, audio_data=None, title="AudioBrain",
                  metadata=None, waveform=None, sample_rate=32000):
        """Save HTML artifact to file and return path."""
        html = self.build_html(latents, audio_data, title, metadata, waveform, sample_rate)
        Path(path).write_text(html, encoding="utf-8")
        return str(path)


def visualize_latents(latents, grid_size=128, chars="dots", colors="heat",
                      title="AudioBrain", show="overlay"):
    """Quick one-shot terminal visualization."""
    viz = AudioBrainVisualizer(grid_size=grid_size, chars=chars, colors=colors)
    viz.render_terminal(latents, title=title, show=show)
