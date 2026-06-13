"""Delay: feedback delay with wet/dry mix."""

import numpy as np


def delay(
    waveform: np.ndarray,
    sample_rate: int,
    time_ms: float = 300.0,
    feedback: float = 0.4,
    mix: float = 0.5,
) -> np.ndarray:
    """
    Feedback delay / echo effect.
    
    Args:
        waveform: Input audio (float32).
        sample_rate: Sample rate in Hz.
        time_ms: Delay time in ms (20-2000). Clamped.
        feedback: Feedback amount (0-0.95). Clamped.
        mix: Dry/wet blend (0-1).

    Returns:
        Delayed waveform.
    """
    time_ms = max(20.0, min(2000.0, float(time_ms)))
    feedback = max(0.0, min(0.95, float(feedback)))
    mix = max(0.0, min(1.0, float(mix)))

    if mix <= 0:
        return waveform

    delay_samples = int(time_ms / 1000.0 * sample_rate)
    if delay_samples < 1:
        delay_samples = 1

    output = waveform.copy()
    for i in range(delay_samples, len(waveform)):
        output[i] += feedback * output[i - delay_samples]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.95

    if mix < 1.0:
        return ((1 - mix) * waveform + mix * output).astype(np.float32)
    return output.astype(np.float32)
