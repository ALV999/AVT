"""Distortion: soft-clip overdrive and hard clip."""

import numpy as np


def distort(
    waveform: np.ndarray,
    sample_rate: int,
    drive: float = 3.0,
    mix: float = 0.5,
) -> np.ndarray:
    """
    Soft-clip distortion (tanh overdrive).
    
    Args:
        waveform: Input audio (float32).
        sample_rate: Sample rate in Hz (unused).
        drive: Gain before clipping (1-20). Clamped.
        mix: Dry/wet blend (0-1).

    Returns:
        Distorted waveform.
    """
    drive = max(1.0, min(20.0, float(drive)))
    mix = max(0.0, min(1.0, float(mix)))

    if mix <= 0:
        return waveform

    # Overdrive + tanh soft-clip
    driven = waveform * drive
    distorted = np.tanh(driven)

    # Normalize
    peak = np.max(np.abs(distorted))
    if peak > 0:
        distorted = distorted / peak * 0.95

    if mix < 1.0:
        return ((1 - mix) * waveform + mix * distorted).astype(np.float32)
    return distorted.astype(np.float32)
