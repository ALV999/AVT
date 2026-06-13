"""Flanger: comb filter with LFO-modulated delay for swooshing effects."""

import numpy as np


def flange(
    waveform: np.ndarray,
    sample_rate: int,
    depth_ms: float = 3.0,
    rate_hz: float = 0.3,
    feedback: float = 0.5,
    mix: float = 0.5,
) -> np.ndarray:
    """
    Classic flanger effect via modulated comb filter.
    
    Args:
        waveform: Input audio (float32).
        sample_rate: Sample rate in Hz.
        depth_ms: Modulation depth in ms (0.1-10). Clamped.
        rate_hz: LFO rate in Hz (0.05-5). Clamped.
        feedback: Feedback amount (0-0.95). Clamped.
        mix: Dry/wet blend (0-1).

    Returns:
        Flanged waveform.
    """
    depth_ms = max(0.1, min(10.0, float(depth_ms)))
    rate_hz = max(0.05, min(5.0, float(rate_hz)))
    feedback = max(0.0, min(0.95, float(feedback)))
    mix = max(0.0, min(1.0, float(mix)))

    if mix <= 0:
        return waveform

    max_delay = int(depth_ms / 1000.0 * sample_rate) + 1
    buffer = np.zeros(len(waveform) + max_delay)
    buffer[:len(waveform)] = waveform

    output = np.zeros_like(waveform)
    lfo_phase = 0.0

    for i in range(len(waveform)):
        lfo = np.sin(2 * np.pi * lfo_phase)
        delay_samples = int((depth_ms / 1000.0 * sample_rate) * (1.0 + lfo) / 2.0)
        delay_samples = max(1, min(max_delay - 1, delay_samples))

        delayed = buffer[i - delay_samples] if i >= delay_samples else 0.0
        output[i] = waveform[i] + feedback * delayed
        buffer[i] = waveform[i] + feedback * delayed

        lfo_phase += rate_hz / sample_rate
        if lfo_phase > 1.0:
            lfo_phase -= 1.0

    # Normalize to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak

    if mix < 1.0:
        return ((1 - mix) * waveform + mix * output).astype(np.float32)
    return output.astype(np.float32)
