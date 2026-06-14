"""
PANNs Feature Extractor Module
Extracts audio embeddings using pretrained Audio Neural Networks.

Model priority (first successful load wins):
  1. MIT/ast-finetuned-audioset-10-10-0.4593  — AST, AudioSet, 768-dim
  2. ntu-spml/distil-ast                        — Distil-AST, AudioSet, 768-dim
  3. Manual CNN fallback (untrained, 2048-dim)

If the loaded model outputs a dimension different from the target (2048),
a linear projection layer is added automatically.
"""

import torch
import torchaudio
import numpy as np
from typing import Union, Optional
from pathlib import Path
from transformers import AutoModel, AutoFeatureExtractor


class PANNsFeatureExtractor:
    """
    Extracts audio features using pretrained audio models.

    Uses models trained on AudioSet (or similar large-scale audio datasets)
    to generate embedding vectors from raw audio.

    Input: Raw audio waveform
    Output: Sequence of embedding vectors (default 2048-dim, configurable)
    """

    # Priority-ordered list of pretrained AudioSet models to try.
    # Each entry: (model_name, output_dim, description)
    _MODEL_CANDIDATES = [
        ("MIT/ast-finetuned-audioset-10-10-0.4593", 768, "AST (AudioSet, verified)"),
        ("ntu-spml/distil-ast", 768, "Distil-AST (AudioSet)"),
    ]

    def __init__(self,
                 model_name: str | None = None,
                 device: Optional[str] = None,
                 sample_rate: int = 32000,
                 embedding_dim: int = 2048,
                 hop_length: int = 320):  # ~10ms hop at 32kHz

        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available()
                                 else 'cpu')

        self.sample_rate = sample_rate
        self.target_embedding_dim = embedding_dim
        self.hop_length = hop_length
        self.model_output_dim = embedding_dim  # may be updated by loader
        self.feature_extractor = None

        # Load model — try candidates in priority order
        loaded = self._try_load_models(model_name)
        if not loaded:
            print("All pretrained models failed. Falling back to manual CNN (untrained).")
            self._load_manual_panns()
            self.model_output_dim = embedding_dim

        self.model.eval()

        # If the model output doesn't match the target embedding dim,
        # add a linear projection layer.
        if self.model_output_dim != self.target_embedding_dim:
            from torch import nn
            self._output_proj = nn.Linear(
                self.model_output_dim, self.target_embedding_dim
            ).to(self.device)
            print(f"  Added output projection: {self.model_output_dim} -> {self.target_embedding_dim}")
        else:
            self._output_proj = None

    def _try_load_models(self, override_model: str | None) -> bool:
        """Try loading a pretrained model. Returns True if successful."""
        candidates = list(self._MODEL_CANDIDATES)
        if override_model:
            candidates.insert(0, (override_model, None, "user-specified"))

        for model_id, known_dim, desc in candidates:
            print(f"  Trying {model_id} ({desc})...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                ).to(self.device)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                # Adopt the model's native sample rate so preprocess_audio
                # resamples correctly and extract_features passes the right rate.
                if hasattr(self.feature_extractor, 'sampling_rate'):
                    self.sample_rate = self.feature_extractor.sampling_rate
                if known_dim is not None:
                    self.model_output_dim = known_dim
                elif hasattr(self.model.config, 'hidden_size'):
                    self.model_output_dim = self.model.config.hidden_size
                else:
                    self.model_output_dim = self.target_embedding_dim
                print(f"  Loaded {model_id} (output dim: {self.model_output_dim})")
                return True
            except Exception as e:
                print(f"    Failed: {e}")
        return False

    def _load_manual_panns(self):
        """Manual CNN encoder fallback (untrained placeholder)."""
        from torch import nn

        class CNNEncoder(nn.Module):
            def __init__(self, embed_dim=2048):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.fc = nn.Linear(1024, embed_dim)

            def forward(self, x):
                features = self.conv_layers(x)
                features = features.squeeze(-1).squeeze(-1)
                return self.fc(features)

        self.model = CNNEncoder(self.target_embedding_dim).to(self.device)
        self.feature_extractor = None
        print("Manual CNN encoder initialized (placeholder — not pretrained)")

    def preprocess_audio(self,
                        audio_path: Union[str, Path],
                        duration: Optional[float] = None) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max()

        # Handle duration
        target_samples = int(self.sample_rate * (duration or waveform.shape[1] / self.sample_rate))

        if waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]

        return waveform

    def extract_features(self,
                        audio_input: Union[str, Path, torch.Tensor],
                        duration: Optional[float] = None,
                        chunk_duration: float = 1.0) -> torch.Tensor:
        """
        Extract embeddings from audio.

        Args:
            audio_input: Path to audio file or preloaded waveform tensor
            duration: Total duration to process (None = full audio)
            chunk_duration: Duration of each chunk for embedding extraction

        Returns:
            Tensor of shape [seq_len, target_embedding_dim]
        """
        if isinstance(audio_input, (str, Path)):
            waveform = self.preprocess_audio(audio_input, duration)
        else:
            waveform = audio_input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[1]
        chunk_samples = int(self.sample_rate * chunk_duration)
        num_chunks = max(1, total_samples // chunk_samples)

        embeddings = []

        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples, total_samples)
                chunk = waveform[:, start_idx:end_idx]

                if chunk.shape[1] < 1000:
                    continue

                if self.feature_extractor is not None:
                    # HuggingFace model path
                    inputs = self.feature_extractor(
                        chunk.squeeze().numpy(),
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    )
                    # Move all tensor values to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    # Pool to single vector per chunk
                    if hasattr(outputs, 'last_hidden_state'):
                        embedding = outputs.last_hidden_state.mean(dim=1)
                    elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output
                    else:
                        embedding = outputs[0].mean(dim=1)
                else:
                    # Manual CNN path — convert to spectrogram
                    spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_mels=128,
                        hop_length=self.hop_length
                    ).to(self.device)
                    spec = spec_transform(chunk.to(self.device))
                    spec = spec.unsqueeze(1)  # [batch, 1, freq, time]
                    embedding = self.model(spec)

                # Apply output projection if needed
                if self._output_proj is not None:
                    embedding = self._output_proj(embedding)

                embeddings.append(embedding.squeeze(0))

        if len(embeddings) == 0:
            raise ValueError("No embeddings extracted — audio may be too short")

        return torch.stack(embeddings)

    def extract_for_transformer(self,
                               audio_input: Union[str, Path, torch.Tensor],
                               target_seq_len: int = 63) -> torch.Tensor:
        """
        Extract features formatted for AudioBrainCore transformer input.

        Args:
            audio_input: Path to audio file or waveform
            target_seq_len: Target sequence length (default 63 for 63x63 visualization)

        Returns:
            Tensor of shape [1, target_seq_len, target_embedding_dim]
        """
        embeddings = self.extract_features(audio_input)
        current_len = embeddings.shape[0]

        if current_len == target_seq_len:
            result = embeddings.unsqueeze(0)
        elif current_len > target_seq_len:
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
            result = torch.nn.functional.adaptive_avg_pool1d(
                embeddings.permute(0, 2, 1),
                target_seq_len
            ).permute(0, 2, 1).squeeze(0)
        else:
            padding = target_seq_len - current_len
            pad_tensor = torch.zeros(padding, self.target_embedding_dim, device=embeddings.device)
            result = torch.cat([embeddings, pad_tensor], dim=0).unsqueeze(0)

        return result  # [1, target_seq_len, target_embedding_dim]

    def __repr__(self):
        dim = self.target_embedding_dim
        return (f"PANNsFeatureExtractor(device={self.device}, "
                f"sample_rate={self.sample_rate}, "
                f"embedding_dim={dim})")
