"""Audio effects module — post-processing for generated soundscapes."""

from audiobrain.effects.chain import EffectChain
from audiobrain.effects.bitcrusher import bitcrush
from audiobrain.effects.pitch import pitch_down, pitch_up
from audiobrain.effects.flanger import flange
from audiobrain.effects.glitch import glitch
from audiobrain.effects.distortion import distort
from audiobrain.effects.delay import delay

__all__ = [
    "EffectChain",
    "bitcrush",
    "pitch_down",
    "pitch_up",
    "flange",
    "glitch",
    "distort",
    "delay",
]
