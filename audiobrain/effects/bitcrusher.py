"""Bitcrusher: amplitude quantization + sample rate decimation."""

import numpy as np


def bitcrush(
    waveform: np.ndarray,
    sample_rate: int,
    bits: int = 8,
    downsample: int = 1,
    mix: float = 1.0,
) -> np.ndarray:
    """
    Reduce bit depth and/or sample rate for lo-fi crunch.

    Args:
        waveform: Input audio (float32, [-1, 1]).
        sample_rate: Sample rate in Hz (unused but kept for API consistency).
        bits: Bit depth (1-24). Lower = more crushed. Clamped to [1, 24].
        downsample: Decimation factor (1-16). 1 = off. Clamped to [1, 16].
        mix: Dry/wet blend (0-1). 0 = clean, 1 = fully crushed.

    Returns:
        Crushed waveform.
    """
    bits = max(1, min(24, int(bits)))
    downsample = max(1, min(16, int(downsample)))
    mix = max(0.0, min(1.0, float(mix)))

    if mix <= 0:
        return waveform

    # Quantize amplitude to N bits
    levels = 2 ** (bits - 1)
    crushed = np.round(waveform * levels) / levels

    # Decimate (sample-and-hold)
    if downsample > 1:
        decimated = np.zeros_like(crushed)
        for i in range(0, len(crushed), downsample):
            end = min(i + downsample, len(crushed))
            decimated[i:end] = crushed[i]
        crushed = decimated

    # Blend
    if mix < 1.0:
        return (1 - mix) * waveform + mix * crushed
    return crushed.astype(np.float32)
