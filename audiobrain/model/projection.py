"""
Capa de proyección para adaptar embeddings de PANNs al espacio latente del Transformer.

Ver Sección 3.3 del documento de especificación (context.md).

Arquitectura:
    Linear(2048, 512) -> LayerNorm -> Dropout(0.1)

Input:  [batch, 2048]
Output: [batch, 512]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """
    Capa de proyección para embeddings de PANNs.

    Reduce la dimensionalidad de los embeddings extraídos por PANNs
    (2048 dimensiones) al espacio latente del Transformer (512 dimensiones).

    Ver Sección 3.3 de context.md para arquitectura detallada.

    Atributos:
        embedding_dim_input: Dimensión original de los embeddings (2048).
        embedding_dim_latent: Dimensión del espacio latente (512).
        dropout: Tasa de dropout (0.1).
    """

    def __init__(
        self,
        embedding_dim_input: int = 2048,
        embedding_dim_latent: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """
        Inicializar la capa de proyección.

        Args:
            embedding_dim_input: Dimensión de entrada (salida de PANNs).
            embedding_dim_latent: Dimensión de salida (espacio latente).
            dropout: Tasa de dropout para regularización.
        """
        super().__init__()

        self.embedding_dim_input = embedding_dim_input
        self.embedding_dim_latent = embedding_dim_latent
        self.dropout = dropout

        # Capa de proyección lineal
        self.linear = nn.Linear(embedding_dim_input, embedding_dim_latent)

        # Normalización y dropout
        self.norm = nn.LayerNorm(embedding_dim_latent)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para proyección de embeddings.

        Ver Sección 3.3 de context.md para especificación de inputs/outputs.

        Args:
            x: Tensor de embeddings con shape [batch, seq_len, embedding_dim_input].

        Returns:
            Tensor con shape [batch, seq_len, embedding_dim_latent].
        """
        # Validación de dimensiones
        assert x.shape[-1] == self.embedding_dim_input, (
            f"Expected input dimension {self.embedding_dim_input}, "
            f"got {x.shape[-1]}"
        )

        # Proyección lineal
        x = self.linear(x)

        # Normalización y dropout
        x = self.norm(x)
        x = self.dropout_layer(x)

        return x
