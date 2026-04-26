"""
AudioBrain - Sistema generativo de paisajes sonoros.

Modulos:
1. Extraccion de Features (PANNs) - Pendiente
2. Modelo Transformer (CORE) - En progreso
3. Sintesis por Mosaicing (k-NN) - Pendiente
4. Visualizacion ASCII 63x63 - Pendiente
"""

from audiobrain.model import AudioBrainCore, BrainConfig

__all__ = ["AudioBrainCore", "BrainConfig"]
__version__ = "0.1.0"
