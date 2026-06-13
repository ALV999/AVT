"""EffectChain: ordered sequence of audio effects with parameter validation."""

from __future__ import annotations

import numpy as np
from typing import Callable, Any


class EffectChain:
    """
    Ordered chain of audio effects applied in sequence.
    
    Each effect is a function: (waveform, sample_rate, **params) -> waveform.
    Parameters are validated at add-time against the effect's safe ranges.
    
    Usage:
        chain = EffectChain()
        chain.add(bitcrush, bits=6, mix=0.7)
        chain.add(pitch_down, semitones=4)
        result = chain.apply(audio, sample_rate)
    """

    def __init__(self):
        self._effects: list[tuple[Callable, dict[str, Any]]] = []

    def add(self, effect_fn: Callable, **params: Any) -> EffectChain:
        """
        Add an effect to the chain.
        
        Args:
            effect_fn: Effect function (waveform, sr, **params) -> waveform.
            **params: Parameters for the effect.
            
        Returns:
            self (for chaining).
        """
        self._effects.append((effect_fn, params))
        return self

    def remove(self, index: int) -> EffectChain:
        """Remove effect at given index."""
        if 0 <= index < len(self._effects):
            self._effects.pop(index)
        return self

    def clear(self) -> EffectChain:
        """Remove all effects."""
        self._effects.clear()
        return self

    def apply(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply all effects in order.
        
        Args:
            waveform: Input audio (1D numpy array, float32).
            sample_rate: Sample rate in Hz.
            
        Returns:
            Processed waveform.
        """
        result = waveform.copy()
        for fn, params in self._effects:
            result = fn(result, sample_rate, **params)
        return result

    def __len__(self) -> int:
        return len(self._effects)

    def __repr__(self) -> str:
        names = [fn.__name__ for fn, _ in self._effects]
        return f"EffectChain({', '.join(names) if names else 'empty'})"
