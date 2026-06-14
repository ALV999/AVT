"""Pitch shift: transpose audio up or down via resampling."""

import numpy as np
import torch
import torchaudio.functional as F


def _shift(waveform: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Core pitch shift via resampling."""
    semitones = max(-24.0, min(24.0, float(semitones)))
    if abs(semitones) < 0.01:
        return waveform

    factor = 2 ** (semitones / 12.0)
    wf = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
    shifted = F.resample(wf, orig_freq=sample_rate, new_freq=int(sample_rate * factor))
    return shifted.squeeze(0).numpy().astype(np.float32)


def pitch_down(
    waveform: np.ndarray,
    sample_rate: int,
    semitones: float = 7.0,
    mix: float = 1.0,
) -> np.ndarray:
    """
    Lower pitch by N semitones (resample down, then stretch to original length).
    
    Args:
        waveform: Input audio.
        sample_rate: Sample rate in Hz.
        semitones: Pitch shift down (0-24). Clamped to [0, 24].
        mix: Dry/wet blend (0-1).
    """
    semitones = max(0.0, min(24.0, float(semitones)))
    mix = max(0.0, min(1.0, float(mix)))
    if semitones < 0.01 or mix <= 0:
        return waveform

    shifted = _shift(waveform, sample_rate, -semitones)
    if mix < 1.0:
        return ((1 - mix) * waveform[:len(shifted)] + mix * shifted).astype(np.float32)
    return shifted.astype(np.float32)


def pitch_up(
    waveform: np.ndarray,
    sample_rate: int,
    semitones: float = 7.0,
    mix: float = 1.0,
) -> np.ndarray:
    """
    Raise pitch by N semitones.
    
    Args:
        waveform: Input audio.
        sample_rate: Sample rate in Hz.
        semitones: Pitch shift up (0-24). Clamped to [0, 24].
        mix: Dry/wet blend (0-1).
    """
    semitones = max(0.0, min(24.0, float(semitones)))
    mix = max(0.0, min(1.0, float(mix)))
    if semitones < 0.01 or mix <= 0:
        return waveform

    shifted = _shift(waveform, sample_rate, semitones)
    if mix < 1.0:
        return ((1 - mix) * waveform[:len(shifted)] + mix * shifted).astype(np.float32)
    return shifted.astype(np.float32)
