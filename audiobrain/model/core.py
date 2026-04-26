"""
AudioBrainCore: Modelo principal para generacion de paisajes sonoros.

Ver Sección 3.3 del documento de especificación (context.md).

Arquitectura:
    ProjectionHead -> SoundscapeTransformer
    
Input:  [batch, seq_len, 2048] (embeddings de PANNs)
Output: [batch, seq_len, 512] (espacio latente contextualizado)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from audiobrain.model.config import BrainConfig
from audiobrain.model.projection import ProjectionHead
from audiobrain.model.transformer import SoundscapeTransformer


class AudioBrainCore(nn.Module):
    """
    Modelo principal que integra proyección y transformer para paisajes sonoros.

    Combina la capa de proyección (para adaptar embeddings de PANNs) con el
    encoder Transformer (para aprender contextos secuenciales).

    Ver Sección 3.3 de context.md para arquitectura detallada.

    Atributos:
        config: Configuración completa del modelo (BrainConfig).
        projection: Capa de proyección de 2048 -> 512 dimensiones.
        transformer: Encoder Transformer para secuenciación.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        """
        Inicializar el modelo AudioBrainCore.

        Args:
            config: Configuración del modelo. Si es None, se usa BrainConfig() por defecto.
        """
        super().__init__()

        # Configurar hiperparámetros
        self.config = config if config is not None else BrainConfig()
        
        # Aplicar configuración de reproducibilidad si hay seed
        self.config.setup()

        # Capa de proyección: PANNs embeddings (2048) -> espacio latente (512)
        self.projection = ProjectionHead(
            embedding_dim_input=self.config.embedding_dim_input,
            embedding_dim_latent=self.config.embedding_dim_latent,
            dropout=self.config.dropout,
        )

        # Transformer encoder para secuenciación contextual
        self.transformer = SoundscapeTransformer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )

        # Mover al dispositivo configurado
        self.to(self.config.device)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass completo: proyección + transformer.

        Procesa embeddings de PANNs a través de la capa de proyección
        y luego los contextualiza con el transformer encoder.

        Ver Sección 3.3 de context.md para especificación de inputs/outputs.

        Args:
            x: Tensor de embeddings de PANNs con shape [batch, seq_len, 2048].
            src_key_padding_mask: Mascaras de padding para el transformer (opcional).

        Returns:
            Tensor contextualizado en espacio latente con shape [batch, seq_len, 512].
        """
        # Validación de dimensiones de entrada
        assert x.shape[-1] == self.config.embedding_dim_input, (
            f"Expected input dimension {self.config.embedding_dim_input}, "
            f"got {x.shape[-1]}"
        )

        # Proyección: [batch, seq_len, 2048] -> [batch, seq_len, 512]
        x = self.projection(x)

        # Transformer: [batch, seq_len, 512] -> [batch, seq_len, 512]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return x

    def encode(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Codificar embeddings de audio a espacio latente contextualizado.

        Método convenience que llama a forward() para mayor claridad semántica.

        Args:
            audio_embeddings: Embeddings de PANNs con shape [batch, seq_len, 2048].

        Returns:
            Representación latente contextualizada con shape [batch, seq_len, 512].
        """
        return self.forward(audio_embeddings)

    def get_config(self) -> BrainConfig:
        """
        Obtener la configuración actual del modelo.

        Returns:
            Instancia de BrainConfig con todos los hiperparámetros.
        """
        return self.config

    def count_parameters(self) -> int:
        """
        Contar el número total de parámetros entrenables del modelo.

        Returns:
            Número total de parámetros entrenables.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Representación string del modelo con resumen de arquitectura."""
        return (
            f"AudioBrainCore(\n"
            f"  d_model={self.config.d_model},\n"
            f"  nhead={self.config.nhead},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  dim_feedforward={self.config.dim_feedforward},\n"
            f"  dropout={self.config.dropout},\n"
            f"  device={self.config.device},\n"
            f"  total_parameters={self.count_parameters():,}\n"
            f")"
        )
