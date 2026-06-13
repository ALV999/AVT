"""Glitch: stochastic audio destruction — stutter, reverse, gaps, jitter."""

import numpy as np


def glitch(
    waveform: np.ndarray,
    sample_rate: int,
    intensity: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply random glitch effects: stutter repeats, chunk reversal, silence gaps.
    
    The glitch pattern is deterministic for a given seed.
    Higher intensity = more frequent and aggressive glitching.
    
    Args:
        waveform: Input audio (float32).
        sample_rate: Sample rate in Hz.
        intensity: How much glitch (0.05-0.8). Clamped.
        seed: Random seed for reproducible glitching.

    Returns:
        Glitched waveform.
    """
    intensity = max(0.05, min(0.8, float(intensity)))
    rng = np.random.RandomState(seed)

    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    if chunk_size < 100:
        chunk_size = 100

    num_chunks = len(waveform) // chunk_size
    if num_chunks < 2:
        return waveform

    output = waveform.copy()
    i = 0
    while i < num_chunks:
        start = i * chunk_size
        end = min(start + chunk_size, len(waveform))

        if rng.random() < intensity:
            action = rng.choice(['stutter', 'reverse', 'gap', 'jitter'])

            if action == 'stutter' and i > 0:
                # Repeat previous chunk
                p_start = max(0, start - chunk_size)
                p_end = start
                repeat_len = min(chunk_size, end - start)
                output[start:start + repeat_len] = output[p_start:p_start + repeat_len]

            elif action == 'reverse':
                output[start:end] = output[start:end][::-1]

            elif action == 'gap':
                # Silence this chunk
                output[start:end] = 0.0

            elif action == 'jitter':
                # Random micro-cuts within chunk
                for j in range(start, end, max(1, chunk_size // 8)):
                    if rng.random() < 0.3:
                        je = min(j + rng.randint(10, chunk_size // 16), end)
                        output[j:je] = 0.0

        i += 1

    return output.astype(np.float32)
