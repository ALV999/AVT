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

    def __init__(self, config: BrainConfig | None = None, device: str | None = None) -> None:
        """
        Inicializar el modelo AudioBrainCore.

        Args:
            config: Configuración del modelo. Si es None, se usa BrainConfig() por defecto.
            device: Dispositivo donde cargar el modelo ('cuda', 'mps', 'cpu'). 
                    Si es None, se usa el definido en config o se auto-detecta.
        """
        super().__init__()

        # Configurar hiperparámetros
        self.config = config if config is not None else BrainConfig()
        
        # Determinar dispositivo
        target_device = device if device is not None else self.config.device
        
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

        # Mover al dispositivo determinado
        self.to(target_device)
        # Guardar el dispositivo usado en la config para referencia
        self.config.device = target_device

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

    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Un paso de entrenamiento con gradient clipping.

        Ver Sección 5 de context.md para especificación de entrenamiento.

        La función de pérdida es reconstrucción auto-supervisada:
        el transformer aprende a preservar la estructura del embedding
        proyectado a través del espacio latente.

        Args:
            batch: Embeddings de PANNs con shape [batch, seq_len, 2048].
            optimizer: Optimizador (Adam recomendado).
            criterion: Función de pérdida (MSELoss recomendado).

        Returns:
            Valor de pérdida (float) para logging.
        """
        self.train()
        optimizer.zero_grad()

        # Mover batch al dispositivo del modelo
        x = batch.to(self.config.device)

        # Forward pass completo: proyección + transformer
        projected = self.projection(x)  # [batch, seq_len, 512]
        output = self.transformer(projected)  # [batch, seq_len, 512]

        # Pérdida de reconstrucción
        loss = criterion(output, projected)

        # Backprop con gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_norm)
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        start_emb: torch.Tensor,
        length: int = 60,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Generación autoregresiva de secuencias en el espacio latente.

        Ver Sección 3 (Etapa 3) de context.md para especificación.

        Partiendo de un embedding inicial, genera una secuencia completa
        aplicando iterativamente el transformer y añadiendo ruido
        controlado por temperatura.

        Args:
            start_emb: Embedding inicial con shape [1, 1, 512].
            length: Longitud total de la secuencia a generar.
            temperature: Control de creatividad (0 = determinístico,
                         1 = máxima variabilidad).

        Returns:
            Secuencia generada con shape [1, length, 512].
        """
        self.eval()

        # Validar dimensiones del embedding inicial
        assert start_emb.shape[-1] == self.config.d_model, (
            f"start_emb debe tener d_model={self.config.d_model}, "
            f"recibido shape={start_emb.shape}"
        )

        generated: list[torch.Tensor] = [start_emb.to(self.config.device)]

        for _ in range(length - 1):
            # Construir secuencia acumulada
            seq = torch.cat(generated, dim=1)  # [1, current_len, 512]

            # Pasar por el transformer
            output = self.transformer(seq)  # [1, current_len, 512]

            # Tomar el último embedding como predicción del siguiente
            next_emb = output[:, -1:, :]  # [1, 1, 512]

            # Añadir ruido controlado por temperatura
            if temperature > 0:
                noise = torch.randn_like(next_emb) * temperature * 0.1
                next_emb = next_emb + noise

            generated.append(next_emb)

        return torch.cat(generated, dim=1)  # [1, length, 512]

    def save_checkpoint(self, path: str) -> None:
        """
        Guardar pesos del modelo a disco.

        Ver Sección 4 de context.md para especificación.

        Args:
            path: Ruta donde guardar el checkpoint (ej. 'checkpoints/model.pt').
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Cargar pesos del modelo desde disco.

        Ver Sección 4 de context.md para especificación.

        Args:
            path: Ruta del checkpoint a cargar.
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])