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
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

from audiobrain.processing.config import GenerationConfig, PreprocessingConfig
from audiobrain.effects import EffectChain


# ═══════════════════════════════════════════════════════════════
# ASCII Logo
# ═══════════════════════════════════════════════════════════════

AUDIOBRAIN_LOGO = r"""
  █████╗ ██╗   ██╗██████╗  ██████╗ ██╗   ██╗██╗  ████████╗
 ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██║   ██║██║  ╚══██╔══╝
 ███████║██║   ██║██████╔╝██║   ██║██║   ██║██║     ██║
 ██╔══██║██║   ██║██╔══██╗██║   ██║██║   ██║██║     ██║
 ██║  ██║╚██████╔╝██████╔╝╚██████╔╝╚██████╔╝██║     ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝     ╚═╝

      ██████╗ ██╗   ██╗███████╗
     ██╔═══██╗██║   ██║██╔══██╗
     ██║   ██║██║   ██║███████║
     ██║▄▄ ██║██║   ██║╚════██║
     ╚██████╔╝╚██████╔╝███████║
      ╚══▀▀═╝  ╚═════╝ ╚══════╝

           Generative Soundscape System
"""


# ═══════════════════════════════════════════════════════════════
# Effect Registry
# ═══════════════════════════════════════════════════════════════

EFFECT_REGISTRY = {
    "bitcrush": {"label": "Bitcrush", "description": "Lo-fi bit-depth reduction"},
    "delay": {"label": "Delay", "description": "Echo with feedback"},
    "distort": {"label": "Distortion", "description": "Tanh overdrive"},
    "flanger": {"label": "Flanger", "description": "Sweeping comb filter"},
    "glitch": {"label": "Glitch", "description": "Stochastic stutter"},
    "pitch-down": {"label": "Pitch \u2193", "description": "Lower pitch"},
    "pitch-up": {"label": "Pitch \u2191", "description": "Raise pitch"},
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
