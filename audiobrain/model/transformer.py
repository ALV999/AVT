"""
Modelo Transformer para secuenciación de embeddings de paisajes sonoros.

Ver Sección 3.3 del documento de especificación (context.md).

Arquitectura:
    nn.TransformerEncoder con:
    - d_model: 512
    - nhead: 8
    - num_layers: 2
    - dim_feedforward: 2048
    - dropout: 0.1
    - batch_first: True

Input:  [batch, seq_len, 512]
Output: [batch, seq_len, 512]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder


class SoundscapeTransformer(nn.Module):
    """
    Encoder Transformer para aprender contextos secuenciales en embeddings.

    Ver Sección 3.3 de context.md para arquitectura detallada.

    Atributos:
        encoder: nn.TransformerEncoder configurado con hiperparámetros del paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        """
        Inicializar el modelo Transformer.

        Ver Sección 3.3 de context.md para especificación de inputs/outputs.

        Args:
            d_model: Dimensión del modelo (512).
            nhead: Número de cabezas de atención (8).
            num_layers: Número de capas del encoder (2).
            dim_feedforward: Dimensión de la capa FFN (2048).
            dropout: Tasa de dropout (0.1).
            batch_first: Orden de dimensiones (True para [batch, seq, feat]).
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Verificar consistencia de hiperparámetros
        if d_model % nhead != 0:
            raise ValueError(
                f"nhead ({nhead}) debe dividir exactamente a d_model ({d_model})"
            )

        # Crear el encoder Transformer
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass para contextualizar embeddings secuenciales.

        Ver Sección 3.3 de context.md para especificación de inputs/outputs.

        Args:
            x: Tensor de embeddings con shape [batch, seq_len, d_model].
            src_key_padding_mask: Mascaras de padding (opcional).

        Returns:
            Tensor contextualizado con shape [batch, seq_len, d_model].
        """
        # Validación de dimensiones
        assert x.shape[-1] == self.d_model, (
            f"Expected model dimension {self.d_model}, "
            f"got {x.shape[-1]}"
        )
        assert x.shape[0] == x.shape[1] or self.batch_first, (
            "Transformer expects batch_first=True"
        )

        # Aplicar el encoder
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return out
