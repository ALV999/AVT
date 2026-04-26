"""
Pipeline Module - Connects PANNs Feature Extraction with AudioBrainCore Transformer

This module provides the complete audio processing pipeline:
1. Load raw audio
2. Extract features using PANNs (2048-dim embeddings)
3. Process through AudioBrainCore Transformer
4. Output contextualized 512-dim latent vectors
"""

import torch
from typing import Union, Optional, Tuple
from pathlib import Path

from audiobrain.model.feature_extractor import PANNsFeatureExtractor
from audiobrain.model.core import AudioBrainCore
from audiobrain.model.config import BrainConfig


class AudioProcessingPipeline:
    """
    Complete audio processing pipeline connecting PANNs feature extraction
    with AudioBrainCore transformer processing.
    
    Usage:
        pipeline = AudioProcessingPipeline()
        latent_vectors = pipeline.process_audio("input.wav")
    """
    
    def __init__(self,
                 config: Optional[BrainConfig] = None,
                 panns_model: Optional[str] = None,
                 sample_rate: int = 32000,
                 device: Optional[str] = None):
        """
        Initialize the processing pipeline
        
        Args:
            config: BrainConfig for AudioBrainCore (uses defaults if None)
            panns_model: PANNs model name (uses default if None)
            sample_rate: Audio sample rate (default: 32000)
            device: Device to run on (auto-detects if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        print(f"Initializing Audio Processing Pipeline on {self.device}...")
        
        # Initialize PANNs feature extractor
        self.feature_extractor = PANNsFeatureExtractor(
            model_name=panns_model,
            device=self.device,
            sample_rate=sample_rate
        )
        
        self.brain = AudioBrainCore(config=config)
        self.brain.to(self.device)
                
        print(f"Pipeline ready: PANNs -> AudioBrainCore")
        print(f"  Input: Raw audio waveform")
        print(f"  Output: 512-dim contextualized latent vectors")
    
    def process_audio(self,
                     audio_path: Union[str, Path],
                     duration: Optional[float] = None,
                     target_seq_len: int = 63) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio file through the complete pipeline
        
        Args:
            audio_path: Path to input audio file
            duration: Optional duration to process (None = full audio)
            target_seq_len: Target sequence length for transformer (default: 63)
            
        Returns:
            Tuple of (raw_embeddings, latent_vectors):
                - raw_embeddings: [1, seq_len, 2048] from PANNs
                - latent_vectors: [1, seq_len, 512] from AudioBrainCore
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Processing: {audio_path.name}")
        
        # Step 1: Extract features using PANNs
        print("  [1/2] Extracting features with PANNs...")
        raw_embeddings = self.feature_extractor.extract_for_transformer(
            audio_path,
            target_seq_len=target_seq_len
        )
        
        print(f"    PANNs output shape: {raw_embeddings.shape}")
        
        # Step 2: Process through AudioBrainCore Transformer
        print("  [2/2] Processing through AudioBrainCore...")
        with torch.no_grad():
            latent_vectors = self.brain.encode(raw_embeddings)
        
        print(f"    AudioBrainCore output shape: {latent_vectors.shape}")
        print("  ✓ Processing complete!")
        
        return raw_embeddings, latent_vectors
    
    def process_batch(self,
                     audio_paths: list,
                     target_seq_len: int = 63) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process multiple audio files in batch
        
        Args:
            audio_paths: List of paths to audio files
            target_seq_len: Target sequence length
            
        Returns:
            Tuple of (all_raw_embeddings, all_latent_vectors):
                - all_raw_embeddings: [batch, seq_len, 2048]
                - all_latent_vectors: [batch, seq_len, 512]
        """
        all_raw = []
        all_latent = []
        
        for i, path in enumerate(audio_paths):
            print(f"\nProcessing [{i+1}/{len(audio_paths)}]: {Path(path).name}")
            raw, latent = self.process_audio(path, target_seq_len=target_seq_len)
            all_raw.append(raw)
            all_latent.append(latent)
        
        # Stack into batches
        all_raw_embeddings = torch.cat(all_raw, dim=0)
        all_latent_vectors = torch.cat(all_latent, dim=0)
        
        print(f"\n✓ Batch processing complete!")
        print(f"  Total samples: {len(audio_paths)}")
        print(f"  Batch shapes:")
        print(f"    Raw embeddings: {all_raw_embeddings.shape}")
        print(f"    Latent vectors: {all_latent_vectors.shape}")
        
        return all_raw_embeddings, all_latent_vectors
    
    def get_model_info(self) -> dict:
        """Get information about the pipeline components"""
        return {
            'device': self.device,
            'feature_extractor': str(self.feature_extractor),
            'brain_config': self.brain.get_config(),
            'brain_params': self.brain.count_parameters(),
            'input_dim': 2048,
            'output_dim': 512,
            'default_seq_len': 63
        }
    
    def __repr__(self):
        return (f"AudioProcessingPipeline(device={self.device}, "
                f"params={self.brain.count_parameters():,})")


# Convenience function for quick testing
def test_pipeline(audio_file: Optional[str] = None):
    """
    Test the pipeline with a sample audio file
    
    Args:
        audio_file: Path to test audio file (optional)
    """
    print("=" * 60)
    print("AUDIO PROCESSING PIPELINE TEST")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AudioProcessingPipeline()
    
    # Show model info
    info = pipeline.get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with audio file if provided
    if audio_file:
        print(f"\nTesting with: {audio_file}")
        try:
            raw, latent = pipeline.process_audio(audio_file)
            print(f"\n✓ Test successful!")
            print(f"  Raw embeddings: {raw.shape}")
            print(f"  Latent vectors: {latent.shape}")
            return True
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            return False
    else:
        print("\nNo audio file provided for testing.")
        print("Usage: test_pipeline('path/to/audio.wav')")
        return True


if __name__ == "__main__":
    # Run test if executed directly
    test_pipeline()
