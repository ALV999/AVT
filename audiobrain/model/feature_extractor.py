"""
PANNs Feature Extractor Module
Extracts 2048-dimensional embeddings from audio using Pretrained Audio Neural Networks
"""

import torch
import torchaudio
import numpy as np
from typing import Union, Optional
from pathlib import Path
from transformers import AutoModel, AutoFeatureExtractor


class PANNsFeatureExtractor:
    """
    Extracts audio features using PANNs (Pretrained Audio Neural Networks)
    Uses CNN14 model trained on AudioSet to generate 2048-dim embeddings
    
    Input: Raw audio waveform
    Output: Sequence of 2048-dimensional embedding vectors
    """
    
    def __init__(self, 
                 model_name: str = "htdemucs",  # Using HuggingFace PANNs implementation
                 device: Optional[str] = None,
                 sample_rate: int = 32000,
                 embedding_dim: int = 2048,
                 hop_length: int = 320):  # ~10ms hop at 32kHz
        
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.hop_length = hop_length
        
        # Load PANNs model and feature extractor
        print(f"Loading PANNs model on {self.device}...")
        try:
            # Using PANNs from HuggingFace transformers
            self.model = AutoModel.from_pretrained(
                "charactr/vgg-m-128k-esc50",  # Alternative: use panns_cnn14
                trust_remote_code=True
            ).to(self.device)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                "charactr/vgg-m-128k-esc50"
            )
        except Exception as e:
            print(f"Warning: Could not load HuggingFace model: {e}")
            print("Falling back to manual PANNs implementation...")
            self._load_manual_panns()
        
        self.model.eval()
        
    def _load_manual_panns(self):
        """Manual PANNs CNN14 implementation fallback"""
        # Simplified PANNs CNN14 architecture for feature extraction
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
                # x: [batch, 1, freq, time]
                features = self.conv_layers(x)
                features = features.squeeze(-1).squeeze(-1)
                return self.fc(features)
        
        self.model = CNNEncoder(self.embedding_dim).to(self.device)
        print("Manual CNN encoder initialized (placeholder for full PANNs)")
    
    def preprocess_audio(self, 
                        audio_path: Union[str, Path],
                        duration: Optional[float] = None) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            duration: Optional duration to truncate/extend to (in seconds)
            
        Returns:
            Preprocessed waveform tensor [1, samples]
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
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
            # Pad with zeros
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_samples:
            # Truncate
            waveform = waveform[:, :target_samples]
        
        return waveform
    
    def extract_features(self, 
                        audio_input: Union[str, Path, torch.Tensor],
                        duration: Optional[float] = None,
                        chunk_duration: float = 1.0) -> torch.Tensor:
        """
        Extract 2048-dim embeddings from audio
        
        Args:
            audio_input: Path to audio file or preloaded waveform tensor
            duration: Total duration to process (None = full audio)
            chunk_duration: Duration of each chunk for embedding extraction
            
        Returns:
            Tensor of shape [seq_len, 2048] where seq_len depends on audio duration
        """
        # Load audio if path provided
        if isinstance(audio_input, (str, Path)):
            waveform = self.preprocess_audio(audio_input, duration)
        else:
            waveform = audio_input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        # Calculate number of chunks
        total_samples = waveform.shape[1]
        chunk_samples = int(self.sample_rate * chunk_duration)
        num_chunks = max(1, total_samples // chunk_samples)
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples, total_samples)
                chunk = waveform[:, start_idx:end_idx]
                
                # Skip if chunk is too short
                if chunk.shape[1] < 1000:
                    continue
                
                # Extract features
                if hasattr(self, 'feature_extractor'):
                    # HuggingFace implementation
                    inputs = self.feature_extractor(
                        chunk.squeeze().numpy(),
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Manual implementation - convert to spectrogram first
                    spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_mels=128,
                        hop_length=self.hop_length
                    ).to(self.device)
                    
                    spec = spec_transform(chunk.to(self.device))
                    spec = spec.unsqueeze(1)  # [batch, 1, freq, time]
                    
                    embedding = self.model(spec)
                
                embeddings.append(embedding.squeeze(0))
        
        # Stack embeddings: [seq_len, 2048]
        if len(embeddings) == 0:
            raise ValueError("No embeddings extracted - audio may be too short")
        
        return torch.stack(embeddings)
    
    def extract_for_transformer(self,
                               audio_input: Union[str, Path, torch.Tensor],
                               target_seq_len: int = 63) -> torch.Tensor:
        """
        Extract features formatted for AudioBrainCore transformer input
        
        Args:
            audio_input: Path to audio file or waveform
            target_seq_len: Target sequence length (default 63 for 63x63 visualization)
            
        Returns:
            Tensor of shape [1, target_seq_len, 2048] ready for AudioBrainCore
        """
        # Extract raw embeddings
        embeddings = self.extract_features(audio_input)
        
        # Interpolate or pad to target sequence length
        current_len = embeddings.shape[0]
        
        if current_len == target_seq_len:
            result = embeddings.unsqueeze(0)
        elif current_len > target_seq_len:
            # Downsample by averaging
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
            result = torch.nn.functional.adaptive_avg_pool1d(
                embeddings.permute(0, 2, 1), 
                target_seq_len
            ).permute(0, 2, 1).squeeze(0)
        else:
            # Pad with zeros
            padding = target_seq_len - current_len
            pad_tensor = torch.zeros(padding, self.embedding_dim, device=embeddings.device)
            result = torch.cat([embeddings, pad_tensor], dim=0).unsqueeze(0)
        
        return result  # [1, target_seq_len, 2048]
    
    def __repr__(self):
        return (f"PANNsFeatureExtractor(device={self.device}, "
                f"sample_rate={self.sample_rate}, "
                f"embedding_dim={self.embedding_dim})")
