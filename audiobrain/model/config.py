"""
Configuración para AudioBrainCore.

Ver Sección 3.2 del Capítulo 3 de la tesis sobre hiperparámetros del Transformer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class BrainConfig:
    """
    Configuración completa para el modelo AudioBrainCore.

    Ver Sección 3.3 (arquitectura del Transformer) y Sección 4.1 (entrenamiento).

    Atributos:
        d_model: Dimensión del espacio latente (512 en el paper).
        nhead: Número de cabezas de atención (8 en el paper).
        num_layers: Profundidad del encoder (2 en el paper).
        dim_feedforward: Dimensión capa FFN (2048 en el paper).
        dropout: Tasa de dropout (0.1 en el paper).
        batch_first: Poner batch como primera dimensión (True para compatibilidad).
        learning_rate: Tasa de aprendizaje para Adam.
        weight_decay: Regularización L2.
        device: Dispositivo a usar (cuda/cpu).
        seed: Semilla para reproducibilidad (opcional).
        embedding_dim_input: Dimensión de salida de PANNs (2048).
        embedding_dim_latent: Dimensión del espacio latente (512).
    """
    # Arquitectura del Transformer (Sección 3.3)
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 2048
    dropout: float = 0.1
    batch_first: bool = True

    # Parámetros de entrenamiento (Sección 4.1)
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_norm: float = 1.0  # Para gradient clipping
    num_epochs: int = 50  # few-shot learning

    # Configuración del dispositivo
    device: Optional[str] = None  # 'cuda'/'cpu' o None para auto-detectar

    # Dimensiones de embedding
    embedding_dim_input: int = 2048  # Salida de PANNs
    embedding_dim_latent: int = 512  # Espacio latente del Transformer

    # Reproducibilidad
    seed: Optional[int] = None

    # Longitud máxima de secuencia
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        """Inicializar device y verificar consistencia de hiperparámetros."""
        # Auto-detectar device si no se especifica
        if self.device is None:
            import torch
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            )

        # Verificar consistencia: nhead debe dividir a d_model
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"nhead ({self.nhead}) debe dividir exactamente a d_model ({self.d_model})"
            )

        # Verificar dropout en rango válido
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout debe estar en [0, 1), actual: {self.dropout}")

    def setup(self) -> None:
        """Configurar semilla para reproducibilidad si se especificó."""
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
