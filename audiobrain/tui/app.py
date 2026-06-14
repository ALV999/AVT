#!/usr/bin/env python3
"""AudioBrain TUI — Terminal User Interface for Generative Soundscapes."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import io, os, sys, subprocess
import numpy as np
import torch
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, Footer, Header, Input,
    Label, ListItem, ListView, Markdown, ProgressBar, Rule, Select, Static,
)
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from audiobrain.processing.config import GenerationConfig, PreprocessingConfig
from audiobrain.effects import EffectChain
from audiobrain.model.visualizer import CHAR_RAMPS, COLOR_SCHEMES

# ═══════════════════════════════════════════════════════════════
# Effect Registry
# ═══════════════════════════════════════════════════════════════
EFFECT_REGISTRY = {
    "bitcrush": {"label": "Bitcrush", "description": "Lo-fi bit-depth reduction"},
    "delay": {"label": "Delay", "description": "Echo with feedback"},
    "distort": {"label": "Distortion", "description": "Tanh overdrive"},
    "flanger": {"label": "Flanger", "description": "Sweeping comb filter"},
    "glitch": {"label": "Glitch", "description": "Stochastic stutter"},
    "pitch-down": {"label": "Pitch ↓", "description": "Lower pitch"},
    "pitch-up": {"label": "Pitch ↑", "description": "Raise pitch"},
}
EFFECT_PARAMS = {
    "bitcrush":   {"bits": (2, 16, 8), "mix": (0.0, 1.0, 0.5)},
    "delay":      {"time_ms": (20, 1000, 200), "feedback": (0.0, 1.0, 0.3), "mix": (0.0, 1.0, 0.4)},
    "distort":    {"drive": (1, 20, 4), "mix": (0.0, 1.0, 0.5)},
    "flanger":    {"depth_ms": (1, 10, 3), "rate_hz": (0.1, 2.0, 0.3), "mix": (0.0, 1.0, 0.5)},
    "glitch":     {"intensity": (0.0, 1.0, 0.3), "mix": (0.0, 1.0, 0.6)},
    "pitch-down": {"semitones": (1, 24, 7), "mix": (0.0, 1.0, 0.7)},
    "pitch-up":   {"semitones": (1, 24, 7), "mix": (0.0, 1.0, 0.7)},
}

PARAM_LABELS = {
    "bits":"Bits","mix":"Mix","time_ms":"Time ms","feedback":"Feedback",
    "drive":"Drive","depth_ms":"Depth ms","rate_hz":"Rate Hz",
    "intensity":"Intensity","semitones":"Semitones",
}

EQ = [("Raw","raw"),("Warm","warm"),("Bright","bright"),("Dark","dark"),("Airy","airy")]
NORM = [("Peak","peak"),("RMS","rms"),("None","none")]
STEREO = [("Mono","mono"),("Left","left"),("Right","right"),("Stereo","stereo")]
MODE = [("Fluid","fluid"),("Glitch","glitch"),("Evolving","evolving")]

# ═══════════════════════════════════════════════════════════════
# Home Screen
# ═══════════════════════════════════════════════════════════════
class Logo(Static):
    """Rich-rendered A/VT logo with Panel + gradient — █-only for perfect alignment."""

    LOGO_LINES = [
        " ██████       ██    ██     ████████ ",
        " ██  ██       ██    ██        ██    ",
        " ██████       ██    ██        ██    ",
        " ██  ██       █ █  █ █        ██    ",
        " ██  ██        ██  ██         ██    ",
        " ██  ██         ████          ██    ",
    ]

    @staticmethod
    def _gradient(start_rgb, end_rgb, steps):
        sr, sg, sb = start_rgb
        er, eg, eb = end_rgb
        colors = []
        for i in range(steps):
            t = i / max(1, steps - 1)
            r = int(sr + (er - sr) * t)
            g = int(sg + (eg - sg) * t)
            b = int(sb + (eb - sb) * t)
            colors.append(f"bold rgb({r},{g},{b})")
        return colors

    def render(self):
        n = len(self.LOGO_LINES)
        gradient = self._gradient((60, 70, 180), (100, 210, 255), n)

        inner = Text()
        for i, line in enumerate(self.LOGO_LINES):
            inner.append(line + "\n", style=gradient[i])

        inner.append(
            "\nAudio / Visual Transformer — generative soundscape engine",
            style="dim italic"
        )

        return Panel(
            Align.center(inner, vertical="middle"),
            border_style="dim cyan",
            padding=(1, 6),
            expand=False,
            title="[bold]A/VT[/bold]",
            subtitle="[dim]v1.0[/dim]",
        )


class HomeScreen(Screen):
    """Home screen with ASCII logo and main menu."""
    BINDINGS = [
        Binding("1", "start", "Start"),
        Binding("2", "docs", "Docs"),
        Binding("3", "about", "About"),
        Binding("4", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="home-container"):
            with Vertical(id="home-content"):
                yield Logo(id="logo")
                with Vertical(id="menu"):
                    yield Button("▶ Start", id="btn-start", variant="primary")
                    yield Button(" Docs", id="btn-docs", variant="default")
                    yield Button("ℹ About", id="btn-about", variant="default")
                    yield Button("✕ Quit", id="btn-quit", variant="error")
        yield Footer()

    def action_start(self):
        self.app.push_screen("workspace")

    def action_docs(self):
        self.app.push_screen("docs")

    def action_about(self):
        self.app.push_screen("about")

    def action_quit(self):
        self.app.exit()

    @on(Button.Pressed, "#btn-start")
    def on_start(self):
        self.action_start()

    @on(Button.Pressed, "#btn-docs")
    def on_docs(self):
        self.action_docs()

    @on(Button.Pressed, "#btn-about")
    def on_about(self):
        self.action_about()

    @on(Button.Pressed, "#btn-quit")
    def on_quit(self):
        self.action_quit()


# ═══════════════════════════════════════════════════════════════
# Docs Screen
# ═══════════════════════════════════════════════════════════════
class DocsScreen(Screen):
    """Documentation screen."""
    BINDINGS = [Binding("escape", "back", "Back")]

    DOCS_MD = """\
# AudioBrain — Complete Documentation

## Architecture & Pipeline

AudioBrain uses a generative audio pipeline that maps real-world sounds through a
learned latent space and back into new synthesized audio:

```
Source Audio → PANNs Feature Extractor (2048-dim) → Projection (2048→512)
  → SoundscapeTransformer (512-dim, 64 segments) → Latent Representation
  → k-NN Mosaic Synthesizer → Output Audio
```

The **latent space** (512 dimensions × 64 time segments) encodes the learned
characteristics of the source audio. Each dimension captures some acoustic feature
— pitch, timbre, rhythm, texture — discovered during the model's self-supervised learning.
The visualization is a direct window into this space.

---

## Synthesis Parameters

### Mode
The generation behavior that controls how the synthesizer moves through the
latent space of the source audio database.
- **Fluid** — smooth, continuous transitions between segments. Latent vectors
  are interpolated, creating seamless morphing between timbres. Best for
  ambient pads, drones, and evolving textures.
- **Glitch** — abrupt, fragmented jumps. The synthesizer skips randomly through
  the latent space, producing sharp cuts between dissimilar audio fragments.
  Good for stutter effects, rhythmic glitch, and IDM textures.
- **Evolving** — gradual drift between states. The latent path wanders slowly
  through the space, producing slow transformations over multiple segments.
  Ideal for long-form generative pieces and organic soundscapes.

### Temperature
Controls the randomness of the k-NN neighbor selection during synthesis.
- `0.0` — **Deterministic.** Always picks the nearest latent neighbor. Same
  input + same seed = identical output every time. Use for reproducibility.
- `0.1–0.3` — **Controlled.** Mostly picks close neighbors with occasional
  variation. Good balance between coherence and surprise.
- `0.5–0.7` — **Creative.** Significant randomness in neighbor choice. Each
  segment may jump to unexpected audio fragments.
- `0.8–1.0` — **Chaotic.** Nearly uniform random selection. The output bears
  minimal relationship to the source. Exploratory/experimental.

### Density
How many of the available segment slots are filled with audio.
- `0.0` — Empty output (silent). Useful as a baseline.
- `0.2–0.4` — Sparse. Gaps of silence between audio segments. Creates
  breathing room, good for ambient with negative space.
- `0.5–0.7` — Medium. Some silence, mostly audio. Natural feel.
- `1.0` — Full coverage. Every segment slot is filled. Maximum density,
  no gaps. Best for continuous soundscapes.

### Segment Duration
Length of each individual audio fragment in **seconds**.
- `0.1–0.5s` — Micro-granular. Hundreds of tiny grains per second of output.
  Creates textural, glitchy, or granular-synthesis effects.
- `0.5–2.0s` — Medium grain. Each fragment is a recognizable sound snippet.
  Good balance for most music production.
- `2.0–5.0s` — Long segments. Fewer grains preserve the character of the
  source audio. Best for long-form ambient or drone.

### Crossfade
Overlap between adjacent segments, in **seconds**.
- `0.0` — Hard cuts. Abrupt transitions, no blending.
- `0.05–0.2s` — Tight overlap. Minimal smoothing, preserves attack transients.
- `0.3–0.6s` — Smooth overlap. Natural-sounding joins between segments.
- `0.7–2.0s` — Long crossfade. Heavy blending can wash out detail but
  produces extremely smooth, pad-like textures.

### Seed
Integer seed for the pseudorandom number generator.
- Used to make results **reproducible**.
- Same seed + same audio file + `temp=0.0` = bit-identical output.
- Different seeds produce different latent paths even with `temp=0.0`.
- Set to `-1` (or any non-integer) for truly random initialization each run.

---

## Preprocessing

### Sample Rate (Hz)
The audio sample rate for processing and synthesis.
- `32000` — PANNs model's native rate. **Recommended.** Produces the best
  quality because the feature extractor was trained at this rate.
- `16000` — Half-rate. Smaller files, faster processing, 8 kHz bandwidth.
  Sufficient for lo-fi or experimental work.
- `22050, 44100, 48000` — Other common rates. The system will resample
  to a supported rate, but quality may degrade slightly.

### Gain (dB)
Input gain applied before feature extraction.
- `0.0` — Unity. No change to input level.
- `-6.0` — Quieter input, less activation in the latent space.
- `+6.0` — Louder input, pushes features into higher activation ranges.
- Values beyond ±12 dB are clamped to avoid clipping.

### EQ Presets
Spectral shaping applied before feature extraction:
- **Raw** — No EQ. The source audio is processed as-is.
- **Warm** — Boosts low-mids (200–500 Hz), cuts highs. Vintage sound.
- **Bright** — Boosts highs (2–10 kHz). Adds air and presence.
- **Dark** — Cuts highs, boosts lows. Muffled, underwater tone.
- **Airy** — Gentle high-shelf boost above 8 kHz. Subtle sparkle.

### Normalization
Level normalization applied to the output:
- **Peak** — Normalizes so the loudest sample hits 0 dBFS. Max loudness.
- **RMS** — Normalizes average (RMS) level. More natural perceived loudness.
- **None** — No normalization. Preserves original dynamics.

### Stereo Mode
How multi-channel audio is handled:
- **Mono** — Sums to mono. Both channels averaged.
- **Left** — Uses only the left channel.
- **Right** — Uses only the right channel.
- **Stereo** — Processes both channels independently. Preserves stereo field
  but doubles computation.

### Color Scheme
The HTML visualization's color palette. 10 presets are available:
- **Heat** — Red → yellow → white (intensity ramp)
- **Ocean** — Deep blue → teal → white (depth)
- **Forest** — Dark green → lime → mint
- **Sunset** — Purple → orange → yellow
- **Aurora** — Teal → magenta → cyan
- **Mono** — Grayscale gradient
- **Neon** — Magenta → cyan → green → yellow
- **Dusk** — Deep purple → magenta → orange
- **Ember** — Dark red → orange → bright
- **Glacier** — Pale blue → white

### Character Ramp
The ASCII character set used to render the visualization. 15 sets:
- **Dots** — `··∘∘●●●•◦◆◇◈◉◊○◎●` — Rounded, organic feel
- **Ascii** — `.:;-=+*#%@★★★★` — Classic terminal look
- **Blocks** — `░░▒▒▓▓█████` — Solid block gradients
- **Braille** — `⠁⠃⠇⠏⠟⠿⣿` — Tiny dot patterns
- **Binary** — `0101010101` — Abstract digital
- **Pixels** — `░░▒▒▓▓██████` — Pixel-art style

---

## Audio Effects

Effects are applied to the synthesized audio **after** synthesis, in the order
they appear in the chain.

### Bitcrush
Lo-fi bit-depth and sample-rate reduction for a crunchy, retro-digital sound.
- **Bits** (2–16): Bit depth. 8 = 8-bit console, 4 = harsh aliasing, 2 = extreme.
- **Mix** (0.0–1.0): Dry/wet blend. 0 = clean, 1 = fully crushed.

### Delay
Echo with adjustable feedback and time.
- **Time ms** (20–1000): Delay time in milliseconds. 20–50 = slapback, 100–300 = echo, 500+ = long trails.
- **Feedback** (0.0–1.0): How much signal feeds back. 0 = single repeat, 0.5 = moderate, 0.9+ = near-infinite.
- **Mix** (0.0–1.0): Dry/wet balance.

### Distortion
Tanh-based soft-clipping overdrive for warmth or aggression.
- **Drive** (1–20): Gain before clipping. 1–3 = subtle warmth, 5–10 = overdrive, 15–20 = heavy distortion.
- **Mix** (0.0–1.0): Dry/wet blend.

### Flanger
Sweeping comb-filter modulation for jet-plane whoosh effects.
- **Depth ms** (1–10): Modulation depth. Higher = more dramatic sweep.
- **Rate Hz** (0.1–2.0): LFO frequency. 0.1–0.3 = slow sweep, 0.5–1.0 = classic, 1.5+ = fast warble.
- **Mix** (0.0–1.0): Dry/wet blend.

### Glitch
Stochastic stutter and repetition for IDM/experimental textures.
- **Intensity** (0.0–1.0): Probability and depth of glitch events. 0 = none, 0.3 = occasional, 0.7+ = heavy.
- **Mix** (0.0–1.0): Dry/wet blend.

### Pitch ↓ / Pitch ↑
Frequency shifting up or down in semitones.
- **Semitones** (1–24): Shift amount. 12 = one octave, 7 = perfect fifth, 24 = two octaves.
- **Mix** (0.0–1.0): Dry/wet blend.

---

## Visualization

The HTML output contains three synchronized views:

### Latent Spectrogram
A **{gs}×{gs}** grid where each column maps to one latent dimension and each
row is one time segment. Cell color intensity = activation strength of that
feature at that moment. Character density adds a second dimension of information.

### Chladni Nodal Lines
Derived from the latent vectors' resonant modes. The system interprets the
latent amplitudes as vibrating plate harmonics and draws the sign-change
boundaries (nodal lines) as bright characters overlaid on the spectrogram.
These are the "fingerprint" of the sound — different audio sources produce
visibly different Chladni patterns.

### Real Spectrogram
A classic STFT (Short-Time Fourier Transform) of the final audio output.
**64 frequency bins × 64 time frames**, rendered with the same ASCII character
grid. This shows what frequencies are actually present in the output, allowing
direct comparison between the learned latent structure and physical sound.

---

## Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| `1` | Home | Start (go to workspace) |
| `2` | Home | Documentation |
| `3` | Home | About |
| `4`, `q` | Home | Quit |
| `h` | Workspace | Go home |
| `g` | Workspace | Generate audio |
| `c` | Workspace | Clear all effects |
| `Esc` | Any | Cancel generation / go back |

---

## File Management

- **Load Source** — Opens a native OS file dialog to select a `.wav`, `.mp3`,
  or `.flac` audio file. The file is added to the database.
- **Database** — Holds all loaded audio files. The synthesizer draws from
  these files when building segments. Multiple files = richer synthesis.
- **Clear DB** — Removes all files from the database.
- **Output Path** — Where the generated `.wav` and `.html` files are saved.
- **View** — Opens the HTML visualization in your default browser.
- **Play** — Plays the generated audio through your system's default player.
- **Stop** — Cancels an in-progress generation or stops playback.

---

## Tips

1. **Start with a single source file** in Fluid mode, temp=0.3, density=1.0.
   This gives you a clean, listenable result to tune further.
2. **Add more files to the database** for richer, more varied output. The
   k-NN synthesizer can jump between different source files.
3. **Use low temperature (0.1–0.3) for ambient**, higher (0.5–0.8) for
   experimental/glitch work.
4. **Short segments (0.2–0.5s) + Glitch mode** = granular IDM textures.
5. **Long segments (3–5s) + Fluid mode** = cinematic drones and pads.
6. **Effects chain order matters** — try Distortion → Delay → Reverb-like
   combinations for sound design.
7. **The visualization IS the model's output** — the Chladni patterns and
   spectrogram are direct renderings of the learned latent features, not
   just decorative elements.
"""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with ScrollableContainer(id="docs-scroll"):
            yield Markdown(self.DOCS_MD)
        with Horizontal(id="docs-footer"):
            yield Button("← Back", id="btn-back")
        yield Footer()

    def action_back(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#btn-back")
    def on_back(self):
        self.action_back()


# ═══════════════════════════════════════════════════════════════
# About Screen
# ═══════════════════════════════════════════════════════════════
class AboutScreen(Screen):
    """About screen."""
    BINDINGS = [Binding("escape", "back", "Back")]

    ABOUT_MD = """\
# About AudioBrain TUI

**Version:** 0.5.0
**Date:** 2026-06-14

AudioBrain TUI is the terminal frontend for the **A/VT** (Audio/Visual Transformer)
generative soundscape system. It processes real audio through a deep learning latent
space and synthesizes entirely new soundscapes — with synchronized ASCII-art
visualizations that ARE the outputs of the model, not decorative add-ons.

---

## Architecture & Pipeline

The full processing chain:

```
┌──────────────────────────────────────────────────────────────┐
│ Source Audio (.wav / .mp3 / .flac)                          │
│   → Resample & preprocess (gain, EQ, normalization)         │
│   → Segment into overlapping windows                        │
│   → PANNs CNN14 feature extractor (2048-dim embeddings)     │
│   → ProjectionHead (2048 → 512) — dimensionality reduction  │
│   → SoundscapeTransformer (6-layer, 512-dim, 64-segment)    │
│   → Latent representation: (64 × 512) tensor                │
│                                                              │
│ SYNTHESIS PHASE                                              │
│   → k-NN search across audio database's latent segment map  │
│   → Temperature-weighted neighbor selection                 │
│   → Mode-driven segment sequencing (fluid/glitch/evolving)  │
│   → Crossfade overlap assembly                              │
│   → Audio effects chain (bitcrush, delay, distort, etc.)    │
│                                                              │
│ OUTPUT                                                       │
│   → Generated audio (.wav)                                  │
│   → Interactive HTML visualization (.html)                  │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

**Feature Extractor (PANNs)**
The CNN14 model from the PANNs (Pretrained Audio Neural Networks) family,
pretrained on AudioSet. It produces 2048-dimensional embeddings that capture
semantic audio features — instrument type, pitch, timbre, spatial characteristics.
This is the "ear" of the system.

**ProjectionHead**
A learned linear projection from the 2048-dim feature space down to the
512-dim latent space. Trained via contrastive learning to preserve the most
salient acoustic differences while discarding noise.

**SoundscapeTransformer**
6-layer autoregressive transformer operating on 64-segment sequences of
512-dim latent vectors. Each segment represents ~0.5–1.0s of audio. The
transformer learns temporal relationships between segments — what sounds
typically follow, how textures evolve, rhythm and pacing patterns.

**AudioMosaicSynthesizer**
The k-NN-based mosaic synthesis engine. For each output segment position:
1. Queries the latent space for the k nearest neighbor segments
2. Selects one via temperature-weighted probability
3. Extracts the corresponding raw audio from the database
4. Applies crossfade overlap with the previous segment

---

## Visualization System

The HTML output contains **three synchronized, real-time ASCII visualization views**,
all rendered directly from the model's internal state:

### 1. Latent Spectrogram
A **{gs}×{gs}** grid of colored characters. Each of the {gs} columns represents one
dimension of the 512-dim latent space (the 64 most "active" dimensions are selected).
Each row is a time step. Cell color intensity maps to the activation strength of
that feature at that moment. Character density provides a second information
channel — denser characters = stronger activations.

### 2. Chladni Nodal Lines
Named after Ernst Chladni's 18th-century vibrating plate experiments, these are
the **sign-change boundaries** computed from the latent vectors' resonant modes.
The system treats the latent amplitudes as vibrating plate harmonics and draws
the nodes (lines where the sign changes) as bright characters overlaid on the
spectrogram. Different audio sources produce recognizably different Chladni
patterns — these are the "fingerprint" of your sound.

The Chladni computation uses:
- Circular plate geometry with 128 harmonic modes
- Amplitude-weighted modal superposition
- Sign-change edge detection (8-directional)
- Thickness modulation based on local gradient magnitude

### 3. Real Spectrogram
A classic Short-Time Fourier Transform (STFT) of the final audio output,
rendered in the same ASCII grid format for direct visual comparison with
the latent views.

Parameters: 64 frequency bins (log-spaced), 64 time frames, Hanning window.

### 4. Live Waveform
An interactive oscilloscope driven by the Web Audio API's AnalyserNode.
80 columns × 12 rows of ASCII characters, updated at 60fps from the
time-domain waveform data.

---

## Color Schemes (10 presets)

| Scheme   | Gradient                         | Character |
|----------|----------------------------------|-----------|
| Heat     | Black → red → yellow → white     | Intense   |
| Ocean    | Deep blue → teal → bright cyan   | Flowing   |
| Forest   | Dark green → lime → mint         | Organic   |
| Sunset   | Deep purple → orange → yellow    | Warm      |
| Aurora   | Teal → magenta → cyan            | Electric  |
| Mono     | Black → gray → white             | Clean     |
| Neon     | Magenta → cyan → lime → yellow   | Synthetic |
| Dusk     | Dark purple → magenta → orange   | Moody     |
| Ember    | Dark maroon → orange → bright    | Hot       |
| Glacier  | Pale blue → ice white            | Cool      |

---

## Character Ramps (15 sets)

| Ramp       | Characters                   | Style            |
|------------|------------------------------|------------------|
| Dots       | ` ··∘∘●●●•◦◆◇◈◉◊○◎●`        | Rounded, organic |
| Ascii      | ` .:;-=+*#%@★★★★`           | Classic terminal |
| Blocks     | ` ░░▒▒▓▓█████`              | Solid gradients  |
| Braille    | ` ⠁⠃⠇⠏⠟⠿⣿`           | Dot patterns     |
| Binary     | ` 0101010101`                | Abstract digital |
| Minimal    | ` .·:·::▪▪▪`                | Sparse & clean   |
| Hexdots    | ` ⠄⠆⠇⠏⠿⣿⣿`         | Dense dot field  |
| Cubes      | ` ░░▒▒▓▓████▓▓`             | 3D cube illusion |
| Arrows     | ` ·›»►▸▶▼▲◆`                | Directional      |
| Sparks     | ` ·˙∴⋯⋰⋱⚡✨`               | Electric feel    |
| Pixels     | ` ░░▒▒▓▓██████`             | Pixel art        |
| Cistercian | ` ⠀⠁⠃⠇⠏⠟⠿⣿⣿`   | Medieval numeral |
| Morse      | ` ·-▪▬▬▪`                   | Telegraph code   |
| LCD        | ` ▓███▓`                     | Segmented readout|
| Shades     | ` ░░▒░▒▓▒▓██▓█`             | Half-tone dither |

---

## Technology Stack

| Layer          | Technology                      |
|----------------|---------------------------------|
| **TUI**        | Textual (0.x), Rich markup      |
| **ML Backend** | PyTorch 2.x, Transformers 4.x   |
| **Audio**      | Librosa, SoundFile, NumPy       |
| **Feature Ext**| PANNs CNN14 (AudioSet pretrained)|
| **Effects**    | Custom DSP: bitcrush, delay, distortion, flanger, glitch, pitch shift |
| **File I/O**   | Native OS dialogs (AppleScript, PowerShell, Zenity) |
| **Playback**   | afplay / wmplayer / xdg-open    |
| **Visualization** | HTML/CSS/JS with Web Audio API, HTML2Canvas |

---

## License

**MIT** — AudioBrain is free and open source. Use it, modify it, share it.
"""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with ScrollableContainer(id="about-scroll"):
            yield Markdown(self.ABOUT_MD)
        with Horizontal(id="about-footer"):
            yield Button("← Back", id="btn-back")
        yield Footer()

    def action_back(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#btn-back")
    def on_back(self):
        self.action_back()


# ═══════════════════════════════════════════════════════════════
# File Browser Modal
# ═══════════════════════════════════════════════════════════════
def _pick_audio_file(start_dir: str | None = None) -> str | None:
    """Open native system file dialog for selecting an audio file."""
    start = start_dir or str(Path.home())
    if sys.platform == "darwin":
        script = (
            'set theFile to POSIX path of (choose file of type {"public.audio"} '
            f'default location (POSIX file "{start}"))\n'
            'return theFile'
        )
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    elif sys.platform == "win32":
        ps = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$f = New-Object System.Windows.Forms.OpenFileDialog; '
            '$f.Filter = "Audio files (*.wav;*.mp3;*.flac)|*.wav;*.mp3;*.flac"; '
            f'$f.InitialDirectory = "{start}"; '
            'if ($f.ShowDialog() -eq "OK") { $f.FileName }'
        )
        r = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    else:
        r = subprocess.run(
            ["zenity", "--file-selection", f"--filename={start}/",
             "--file-filter=Audio files (*.wav *.mp3 *.flac) | *.wav *.mp3 *.flac"],
            capture_output=True, text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else None


def _pick_save_file(start_path: str = "output.wav") -> str | None:
    """Open native save dialog for output path."""
    start = str(Path(start_path).absolute())
    if sys.platform == "darwin":
        folder = str(Path(start).parent)
        name = Path(start).name
        script = (
            f'set theFile to POSIX path of (choose file name '
            f'default name "{name}" default location (POSIX file "{folder}"))\n'
            'return theFile'
        )
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    elif sys.platform == "win32":
        ps = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$f = New-Object System.Windows.Forms.SaveFileDialog; '
            '$f.Filter = "WAV files (*.wav)|*.wav"; '
            f'$f.FileName = "{start}"; '
            'if ($f.ShowDialog() -eq "OK") { $f.FileName }'
        )
        r = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    else:
        r = subprocess.run(
            ["zenity", "--file-selection", "--save", f"--filename={start}",
             "--file-filter=WAV files (*.wav) | *.wav"],
            capture_output=True, text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else None


# ═══════════════════════════════════════════════════════════════
# Workspace Screen
# ═══════════════════════════════════════════════════════════════
class WorkspaceScreen(Screen):
    """Main workspace."""
    BINDINGS = [
        Binding("h", "home", "Home"),
        Binding("g", "generate", "Generate"),
        Binding("c", "clear_fx", "Clear FX"),
        Binding("escape", "escape_handler", "Cancel/Back"),
    ]

    def __init__(self):
        super().__init__()
        self._source_files: list[str] = []
        self._db_files: list[str] = []
        self._enabled_effects: dict[str, bool] = {}
        self._effect_params: dict[str, dict] = {}
        self._is_generating = False
        self._cancel_requested = False
        self._audio_buffer: np.ndarray | None = None
        self._sample_rate: int = 32000
        self._latents = None
        self._play_process = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # ── TOP ROW: Commands | Files | Generate ──
        with Horizontal(id="workspace-top"):
            # LEFT: Commands
            with Vertical(id="left-panel"):
                yield Label("COMMANDS", classes="section-title")
                yield Button("Load Source", id="btn-load", variant="primary")
                yield Button("Clear DB", id="btn-clear-db", variant="error")
                yield Button("Docs", id="btn-docs-ws", variant="default")
                yield Static("", id="source-info")
                yield Static("", id="db-info")

            # MIDDLE: File list (ListView)
            with Vertical(id="right-panel"):
                yield Label("FILES", classes="section-title")
                yield ListView(id="file-list")

            # RIGHT: Output + Generate in 2 columns
            with Vertical(id="gen-panel"):
                yield Label("GENERATE", classes="section-title")
                with Horizontal(id="gen-grid"):
                    with Vertical(classes="gen-col"):
                        yield Label("Output:")
                        with Horizontal(id="outpath-row"):
                            yield Input(value="output.wav", id="outpath", placeholder="output.wav")
                            yield Button("...", id="btn-browse-out", variant="default")
                        yield Button("▶ Generate", id="btn-generate", variant="success")
                    with Vertical(classes="gen-col"):
                        yield Label("View Output:")
                        yield Button("View", id="btn-view", variant="primary")
                        with Vertical(id="post-gen"):
                            yield Button("■ Stop", id="btn-stop", variant="error", disabled=True)
                            yield Button("▶ Play", id="btn-play", variant="primary", disabled=True)
                yield ProgressBar(total=100, show_eta=False, id="progress")

        # ── BOTTOM ROW: Prep+Synth | Effects ─
        with Horizontal(id="controls-row"):
            # LEFT: Preprocessing + Synthesis
            with ScrollableContainer(id="prep-synth-col"):
                with Vertical(id="prep-section"):
                    yield Label("PREPROCESSING", classes="section-title")
                    with Horizontal(classes="prep-grid"):
                        with Vertical(classes="prep-col"):
                            with Horizontal(classes="prep-row"):
                                yield Label("Sample Rate:")
                                yield Input(value="32000", id="sr", placeholder="Hz")
                            with Horizontal(classes="prep-row"):
                                yield Label("Gain (dB):")
                                yield Input(value="0.0", id="gain", placeholder="dB")
                            with Horizontal(classes="prep-row"):
                                yield Label("EQ:")
                                yield Select(EQ, value="raw", id="eq")
                            with Horizontal(classes="prep-row"):
                                yield Label("Colors:")
                                yield Select(
                                    [(k.title(), k) for k in COLOR_SCHEMES.keys()],
                                    value="ocean", id="colors", prompt="Colors"
                                )
                        with Vertical(classes="prep-col"):
                            with Horizontal(classes="prep-row"):
                                yield Label("Norm:")
                                yield Select(NORM, value="peak", id="norm")
                            with Horizontal(classes="prep-row"):
                                yield Label("Stereo:")
                                yield Select(STEREO, value="mono", id="stereo")
                            with Horizontal(classes="prep-row"):
                                yield Label("Chars:")
                                yield Select(
                                    [(k.title(), k) for k in CHAR_RAMPS.keys()],
                                    value="ascii", id="chars", prompt="Chars"
                                )

                with Vertical(id="synth-section"):
                    yield Label("SYNTHESIS", classes="section-title")
                    with Horizontal(classes="synth-grid"):
                        with Vertical(classes="synth-col"):
                            with Horizontal(classes="synth-row"):
                                yield Label("Mode:")
                                yield Select(
                                    [("Fluid", "fluid"), ("Glitch", "glitch"), ("Evolving", "evolving")],
                                    value="fluid", id="mode"
                                )
                            with Horizontal(classes="synth-row"):
                                yield Label("Temp:")
                                yield Input(value="0.5", id="temp", placeholder="0.0-1.0")
                            with Horizontal(classes="synth-row"):
                                yield Label("Segment:")
                                yield Input(value="1.0", id="seg", placeholder="s")
                        with Vertical(classes="synth-col"):
                            with Horizontal(classes="synth-row"):
                                yield Label("Density:")
                                yield Input(value="1.0", id="dens", placeholder="0.0-1.0")
                            with Horizontal(classes="synth-row"):
                                yield Label("Xfade:")
                                yield Input(value="0.1", id="xfade", placeholder="s")
                            with Horizontal(classes="synth-row"):
                                yield Label("Seed:")
                                yield Input(value="42", id="seed", placeholder="Seed")

            # RIGHT: Effects (checkbox + params in bordered box)
            with ScrollableContainer(id="fx-col"):
                with Vertical(id="fx-section"):
                    yield Label("EFFECTS", classes="section-title")
                    for fx_name, fx_info in EFFECT_REGISTRY.items():
                        with Vertical(classes="fx-item"):
                            yield Checkbox(fx_info["label"], id=f"fx-{fx_name}")
                            params = EFFECT_PARAMS.get(fx_name, {})
                            if params:
                                with Horizontal(classes="fx-params"):
                                    for pname, (pmin, pmax, pdef) in params.items():
                                        plabel = PARAM_LABELS.get(pname, pname)
                                        yield Label(plabel, classes="fx-param-label")
                                        yield Input(
                                            value=str(pdef),
                                            id=f"fxp-{fx_name}-{pname}",
                                            placeholder=f"{pmin}–{pmax}",
                                            disabled=True,
                                        )
                    yield Static("", id="fx-status")

        yield Footer()

    def action_home(self):
        self.app.pop_screen()

    def action_generate(self):
        if not self._is_generating:
            self._run_pipeline()

    def action_escape_handler(self):
        if self._is_generating:
            self._cancel_requested = True
            self.notify("Cancelling...", severity="warning")
        else:
            self.app.pop_screen()

    def action_clear_fx(self):
        self._enabled_effects.clear()
        self._effect_params.clear()
        for fx_name in EFFECT_REGISTRY:
            cb = self.query_one(f"#fx-{fx_name}", Checkbox)
            cb.value = False
            for pname in EFFECT_PARAMS.get(fx_name, {}):
                inp = self.query_one(f"#fxp-{fx_name}-{pname}", Input)
                inp.disabled = True
        self._update_fx_status()

    # ── File loading ─────────────────────────────────────────────
    @on(Button.Pressed, "#btn-load")
    def on_load(self):
        path = _pick_audio_file()
        if path:
            self._source_files = [path]
            self._db_files = list(set(self._db_files + [path]))
            self._update_info()
            self.notify(f"Loaded: {Path(path).name}")

    @on(Button.Pressed, "#btn-docs-ws")
    def on_docs_ws(self):
        self.app.push_screen("docs")

    @on(Button.Pressed, "#btn-browse-out")
    def on_browse_out(self):
        current = self.query_one("#outpath", Input).value or "output.wav"
        path = _pick_save_file(current)
        if path:
            self.query_one("#outpath", Input).value = path

    # ── Database ─────────────────────────────────────────────────
    @on(Button.Pressed, "#btn-clear-db")
    def on_clear_db(self):
        self._db_files.clear()
        self._update_info()
        self.notify("Database cleared")

    # ── Generate / Stop / Play ───────────────────────────────────
    @on(Button.Pressed, "#btn-generate")
    def on_gen(self):
        self.action_generate()

    @on(Button.Pressed, "#btn-stop")
    def on_stop(self):
        if self._is_generating:
            self._cancel_requested = True
            self._is_generating = False
            self.query_one("#btn-generate", Button).disabled = False
            self.query_one("#btn-stop", Button).disabled = True
            self.notify("Cancelled", severity="warning")
        elif hasattr(self, '_play_process') and self._play_process is not None:
            self._play_process.kill()
            self._play_process = None
            self.query_one("#btn-stop", Button).disabled = True
            self.notify("Playback stopped")

    @on(Button.Pressed, "#btn-view")
    def on_view(self):
        """Open the generated HTML file in the default browser."""
        outpath = self.query_one("#outpath", Input).value or "output.wav"
        html_path = str(Path(outpath).with_suffix(".html"))
        if not Path(html_path).exists():
            self.notify("No output to view — generate first", severity="warning")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", html_path])
        elif sys.platform == "win32":
            os.startfile(html_path)
        else:
            subprocess.Popen(["xdg-open", html_path])
        self.notify("Opening visualization...")

    @on(Button.Pressed, "#btn-play")
    def on_play(self):
        if self._audio_buffer is None:
            self.notify("Nothing to play — generate first", severity="warning")
            return
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, self._audio_buffer.astype(np.float32), self._sample_rate)
            self._play_file = f.name
        if sys.platform == "darwin":
            self._play_process = subprocess.Popen(["afplay", self._play_file])
        elif sys.platform == "win32":
            self._play_process = subprocess.Popen(
                ["start", "/min", "wmplayer", self._play_file], shell=True
            )
        else:
            self._play_process = subprocess.Popen(["xdg-open", self._play_file])
        self.notify("Playing...")
        self.query_one("#btn-stop", Button).disabled = False

    # ── Effects ──────────────────────────────────────────────────
    @on(Checkbox.Changed)
    def on_fx_change(self, event: Checkbox.Changed):
        cid = event.checkbox.id or ""
        if cid.startswith("fx-"):
            name = cid.replace("fx-", "")
            self._enabled_effects[name] = event.value
            for pname in EFFECT_PARAMS.get(name, {}):
                inp = self.query_one(f"#fxp-{name}-{pname}", Input)
                inp.disabled = not event.value
            self._update_fx_status()

    def _update_info(self):
        src_info = self.query_one("#source-info", Static)
        db_info = self.query_one("#db-info", Static)
        flist = self.query_one("#file-list", ListView)
        if self._source_files:
            src_info.update(f"Source: [bold]{Path(self._source_files[0]).name}[/bold]")
        else:
            src_info.update("Source: [dim]none[/dim]")
        db_info.update(f"DB: [bold]{len(self._db_files)}[/bold] files")
        # Update ListView
        flist.clear()
        if self._db_files:
            for f in self._db_files:
                flist.append(ListItem(Label(f"• {Path(f).name}")))
        else:
            flist.append(ListItem(Label("[dim]No files in database[/dim]")))

    def _update_fx_status(self):
        active = [n for n, v in self._enabled_effects.items() if v]
        status = self.query_one("#fx-status", Static)
        if active:
            status.update(f"Active: [bold]{' → '.join(active)}[/bold]")
        else:
            status.update("[dim]No effects enabled[/dim]")

    def _read_select(self, wid: str, default: str) -> str:
        try:
            val = self.query_one(f"#{wid}", Select).value
            return str(val) if val is not None else default
        except AttributeError:
            return default

    def _cancel_check(self):
        return self._cancel_requested

    def _btn(self, btn, disabled):
        self.app.call_from_thread(setattr, btn, 'disabled', disabled)

    def _progress(self, prog, n):
        self.app.call_from_thread(prog.update, progress=n)

    def _msg(self, text, error=False):
        self.app.call_from_thread(self.notify, text, severity="error" if error else "information")

    @work(exclusive=True, thread=True)
    def _run_pipeline(self):
        prog = self.query_one("#progress", ProgressBar)
        gen_btn = self.query_one("#btn-generate", Button)
        stop_btn = self.query_one("#btn-stop", Button)
        play_btn = self.query_one("#btn-play", Button)
        self._is_generating = True
        self._cancel_requested = False
        self._audio_buffer = None
        self._btn(gen_btn, True)
        self._btn(stop_btn, False)
        self._btn(play_btn, True)

        # Read settings
        try:
            sr = int(self.query_one("#sr", Input).value or "32000")
            mode = self.query_one("#mode", Select).value
            temp = float(self.query_one("#temp", Input).value or "0.5")
            dens = float(self.query_one("#dens", Input).value or "1.0")
            seg = float(self.query_one("#seg", Input).value or "1.0")
            xfade = float(self.query_one("#xfade", Input).value or "0.1")
            seed = int(self.query_one("#seed", Input).value or "42")
            outpath = self.query_one("#outpath", Input).value or "output.wav"
            colors = self.query_one("#colors", Select).value
            chars = self.query_one("#chars", Select).value
            self._sample_rate = sr
        except ValueError as e:
            self._msg(f"Invalid parameter: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        self._progress(prog, 5)
        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if not self._db_files:
            self._progress(prog, 0)
            self._msg("No files in database!", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        from audiobrain.model.pipeline import AudioProcessingPipeline
        from audiobrain.model.synthesizer import AudioMosaicSynthesizer
        from audiobrain.model.visualizer import AudioBrainVisualizer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        src = self._db_files[0]

        # Load pipeline
        try:
            self._progress(prog, 15)
            pipeline = AudioProcessingPipeline(device=device, sample_rate=sr)
        except Exception as e:
            self._msg(f"Pipeline: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # Process
        try:
            self._progress(prog, 25)
            _, latents = pipeline.process_audio(src, duration=5.0)
        except Exception as e:
            self._msg(f"Process: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # Database
        try:
            self._progress(prog, 40)
            synth = AudioMosaicSynthesizer(
                device=device, segment_duration=seg,
                sample_rate=sr, crossfade_duration=xfade
            )
            synth.build_database(
                audio_files=self._db_files,
                feature_extractor=pipeline.feature_extractor,
                pipeline=pipeline, max_segments=500
            )
        except Exception as e:
            self._msg(f"Database: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # Synthesize
        try:
            self._progress(prog, 60)
            gen_config = GenerationConfig(temperature=temp, density=dens, mode=mode, seed=seed)
            audio, _ = synth.synthesize_from_latent(
                latents, self._db_files, pipeline=pipeline, config=gen_config
            )
        except Exception as e:
            self._msg(f"Synthesis: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # Effects
        active_fx = [n for n, v in self._enabled_effects.items() if v]
        if active_fx:
            try:
                self._progress(prog, 75)
                chain = EffectChain()
                fx_map = {
                    "bitcrush": ("audiobrain.effects", "bitcrush"),
                    "delay": ("audiobrain.effects", "delay"),
                    "distort": ("audiobrain.effects", "distort"),
                    "flanger": ("audiobrain.effects", "flange"),
                    "glitch": ("audiobrain.effects", "glitch"),
                    "pitch-down": ("audiobrain.effects", "pitch_down"),
                    "pitch-up": ("audiobrain.effects", "pitch_up"),
                }
                import importlib
                for fx_name in active_fx:
                    kw = {}
                    for pname, (pmin, pmax, pdef) in EFFECT_PARAMS.get(fx_name, {}).items():
                        try:
                            val = float(self.query_one(f"#fxp-{fx_name}-{pname}", Input).value)
                            kw[pname] = max(pmin, min(pmax, val))
                        except (ValueError, AttributeError):
                            kw[pname] = pdef
                    mod_name, fn_name = fx_map.get(fx_name, (None, None))
                    if mod_name:
                        mod = importlib.import_module(mod_name)
                        chain.add(getattr(mod, fn_name), **kw)
                audio = chain.apply(audio.astype(np.float32), sr)
            except Exception as e:
                self._msg(f"Effects: {e}", error=True)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # Save WAV
        try:
            self._progress(prog, 85)
            import soundfile as sf
            sf.write(outpath, audio.astype(np.float32), sr)
        except Exception as e:
            self._msg(f"Save: {e}", error=True)
            return self._cleanup(gen_btn, stop_btn, play_btn)

        if self._cancel_check():
            return self._cleanup(gen_btn, stop_btn, play_btn)

        # HTML
        try:
            self._progress(prog, 92)
            html_path = Path(outpath).with_suffix(".html")
            viz = AudioBrainVisualizer(grid_size=128, colors=colors, chars=chars)
            import soundfile as sf
            audio_buf = io.BytesIO()
            sf.write(audio_buf, audio.astype(np.float32), sr, format="WAV")
            viz.save_html(
                str(html_path), latents, audio_data=audio_buf.getvalue(),
                title=Path(src).stem,
                metadata={"source": Path(src).name, "mode": mode},
                waveform=audio.astype(np.float32), sample_rate=sr
            )
        except Exception as e:
            self._msg(f"HTML: {e}", error=True)

        self._audio_buffer = audio
        self._progress(prog, 100)
        self._btn(play_btn, False)
        dur = len(audio) / sr
        self._msg(f"Done! {outpath} ({dur:.1f}s)")

    def _cleanup(self, gen_btn, stop_btn, play_btn):
        self._is_generating = False
        self._btn(gen_btn, False)
        self._btn(stop_btn, True)
        self._btn(play_btn, self._audio_buffer is None and True or False)
        self._msg("Cancelled", error=True)


# ═══════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════
class AudioBrainTUI(App):
    """AudioBrain TUI Application."""

    CSS = """
    /* ── Global ── */
    Screen {
        text-style: none;
    }

    Button {
        min-height: 1;
        border: none;
        text-style: none;
    }

    Input {
        height: 1;
        min-height: 1;
        border: none;
    }

    Select {
        min-height: 1;
        border: none;
    }

    Select > .select-list-viewport {
        max-height: 10;
    }

    Label {
        height: 1;
        min-height: 1;
    }

    Checkbox {
        min-height: 1;
        border: none;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        height: 1;
        margin-bottom: 1;
    }

    /* ── Home ─ */
    #home-container {
        align: center middle;
        height: 1fr;
    }

    #home-content {
        height: auto;
        width: 54;
        align: center middle;
    }

    #logo {
        content-align: center middle;
        text-style: bold;
        color: $primary;
        height: auto;
        width: 100%;
    }

    #menu {
        height: auto;
        width: 100%;
        align: center middle;
        margin-top: 2;
        content-align: center middle;
    }

    #menu Button {
        width: 100%;
        height: 3;
        margin: 1 0;
        content-align: center middle;
        min-width: 30;
    }

    #version {
        text-align: center;
        color: $text-muted;
        height: 1;
        width: 100%;
    }

    /* ── Workspace top row ── */
    #workspace-top {
        height: 20;
        min-height: 20;
    }

    #left-panel {
        width: 1fr;
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }

    #left-panel Button {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    #left-panel Static {
        margin-top: 1;
    }

    #right-panel {
        width: 2fr;
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }

    #file-list {
        height: 1fr;
    }

    ListView {
        height: 1fr;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem Label {
        height: 1;
    }

    #gen-panel {
        width: 3fr;
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }

    #gen-grid {
        height: auto;
        margin-bottom: 1;
    }

    .gen-col {
        width: 1fr;
        height: auto;
        margin-right: 2;
    }

    .gen-col:last-child {
        margin-right: 0;
    }

    #outpath-row {
        height: 3;
        margin-bottom: 1;
    }

    #outpath-row Input {
        height: 3;
    }

    #outpath-row Button {
        height: 3;
    }

    #outpath-row Input {
        width: 1fr;
    }

    #outpath-row Button {
        width: 5;
        min-width: 5;
    }

    #btn-generate {
        width: 100%;
        height: 3;
    }

    #btn-view {
        width: 100%;
        height: 3;
    }

    #progress {
        width: 100%;
    }

    #post-gen {
        height: auto;
        align: center middle;
    }

    #post-gen Button {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    /* ── Rule separators ── */
    Rule {
        margin: 1 0;
    }

    /* ── Workspace middle row ── */
    #controls-row {
        height: 1fr;
    }

    #prep-synth-col {
        width: 1fr;
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }

    #prep-section {
        height: auto;
    }

    #synth-section {
        height: auto;
    }

    #fx-col {
        width: 1fr;
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }

    #fx-section {
        height: auto;
    }

    #controls-row Input {
        width: 1fr;
    }

    #controls-row Select {
        width: 1fr;
    }

    .field-row {
        height: 1;
    }

    .field-row Input {
        width: 1fr;
        margin-right: 1;
    }

    .field-row Input:last-child {
        margin-right: 0;
    }

    /* ─ Preprocessing 2-column grid ── */
    .prep-grid {
        height: auto;
        margin-bottom: 1;
    }

    .prep-col {
        width: 1fr;
        height: auto;
        margin-right: 2;
    }

    .prep-col:last-child {
        margin-right: 0;
    }

    .prep-row {
        height: auto;
        margin-bottom: 1;
    }

    .prep-row Label {
        width: 12;
        color: $text-muted;
    }

    .prep-row Input, .prep-row Select {
        width: 1fr;
    }

    /* ── Synthesis 2-column grid ── */
    .synth-grid {
        height: auto;
        margin-bottom: 1;
    }

    .synth-col {
        width: 1fr;
        height: auto;
        margin-right: 2;
    }

    .synth-col:last-child {
        margin-right: 0;
    }

    .synth-row {
        height: auto;
        margin-bottom: 1;
    }

    .synth-row Label {
        width: 8;
        color: $text-muted;
    }

    .synth-row Input, .synth-row Select {
        width: 1fr;
    }

    /* ─ Effects compact layout ── */
    .fx-item {
        height: auto;
        margin-bottom: 1;
        border: solid $primary-background;
        padding: 0 1;
    }

    .fx-params {
        height: 1;
        align: left middle;
        margin-top: 1;
    }

    .fx-param-label {
        text-style: bold;
        color: $text-muted;
        height: 1;
        margin-right: 1;
    }

    .fx-params Input {
        width: 1fr;
        margin-right: 1;
    }

    .fx-params Input:last-child {
        margin-right: 0;
    }

    #fx-status {
        margin-top: 1;
    }

    /* ── Docs & About ── */
    #docs-scroll, #about-scroll {
        height: 1fr;
    }

    #docs-footer, #about-footer {
        height: 1;
        align: center middle;
    }

    Markdown {
        padding: 1 2;
    }
    """

    SCREENS = {
        "home": HomeScreen,
        "workspace": WorkspaceScreen,
        "docs": DocsScreen,
        "about": AboutScreen,
    }

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def on_mount(self):
        self.push_screen("home")


def run():
    AudioBrainTUI().run()


if __name__ == "__main__":
    run()