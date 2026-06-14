"""
Complete Audio Generation Pipeline - From input audio to synthesized output

This module provides a unified interface for the complete audio generation workflow:
1. Extract features from source audio using PANNs
2. Process through AudioBrainCore Transformer
3. Synthesize new audio using mosaicing from a database
"""

import torch
from typing import List, Optional, Union
from pathlib import Path

from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioMosaicSynthesizer
from audiobrain.model.config import BrainConfig


class AudioGenerationPipeline:
    """
    Complete pipeline for generating new audio from source audio.
    
    This combines feature extraction, transformer processing, and synthesis
    into a single easy-to-use interface.
    
    Usage:
        # Initialize with database of audio samples
        pipeline = AudioGenerationPipeline(database_files=['samples/*.wav'])
        
        # Generate new audio from source
        output_audio = pipeline.generate('source_audio.wav')
        
        # Or save directly to file
        pipeline.generate_and_save('source_audio.wav', 'output.wav')
    """
    
    def __init__(self,
                 database_files: Optional[List[str]] = None,
                 config: Optional[BrainConfig] = None,
                 segment_duration: float = 1.0,
                 sample_rate: int = 32000,
                 crossfade_duration: float = 0.02,
                 max_database_segments: int = 1000,
                 device: Optional[str] = None):
        """
        Initialize the complete generation pipeline
        
        Args:
            database_files: List of audio files to build the synthesis database
            config: BrainConfig for the transformer model
            segment_duration: Duration of audio segments for mosaicing
            sample_rate: Audio sample rate
            crossfade_duration: Crossfade duration between segments
            max_database_segments: Maximum segments to extract for database
            device: Device to run on (auto-detects if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        print("=" * 60)
        print("INITIALIZING AUDIO GENERATION PIPELINE")
        print("=" * 60)
        
        # Initialize processing pipeline (PANNs + Transformer)
        self.processing_pipeline = AudioProcessingPipeline(
            config=config,
            sample_rate=sample_rate,
            device=self.device
        )
        
        # Initialize synthesizer
        self.synthesizer = AudioMosaicSynthesizer(
            segment_duration=segment_duration,
            sample_rate=sample_rate,
            crossfade_duration=crossfade_duration,
            device=self.device
        )
        
        # Build database if files provided
        if database_files:
            self.build_database(database_files, max_database_segments)
        else:
            print("\n⚠️  No database files provided.")
            print("   Call build_database() before generating audio.")
        
        print("\n" + "=" * 60)
        print("PIPELINE READY")
        print("=" * 60)
    
    def build_database(self,
                      audio_files: List[str],
                      max_segments: int = 1000):
        """
        Build the synthesis database from audio files
        
        Args:
            audio_files: List of paths to audio files
            max_segments: Maximum number of segments to extract
        """
        self.synthesizer.build_database(
            audio_files=audio_files,
            feature_extractor=self.processing_pipeline.feature_extractor,
            pipeline=self.processing_pipeline,
            max_segments=max_segments
        )
    
    def generate(self,
                source_audio: Union[str, Path],
                duration: Optional[float] = None,
                k_neighbors: int = 5,
                effect_chain = None) -> tuple:
        """
        Generate new audio from a source audio file
        
        Args:
            source_audio: Path to source audio file
            duration: Optional duration to process (None = full audio)
            k_neighbors: Number of nearest neighbors for synthesis
            effect_chain: Optional EffectChain to apply after synthesis
            
        Returns:
            Tuple of (synthesized_audio, latent_vectors):
                - synthesized_audio: numpy array of audio waveform
                - latent_vectors: transformer output latents
        """
        source_path = Path(source_audio)
        if not source_path.exists():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        
        if self.synthesizer.latent_database is None:
            raise ValueError(
                "Database not built! Call build_database() first or provide "
                "database_files during initialization."
            )
        
        print(f"\nGenerating audio from: {source_path.name}")
        
        # Step 1: Extract features and process through transformer
        _, latent_vectors = self.processing_pipeline.process_audio(source_path, duration)
        
        # Step 2: Synthesize audio from latent vectors
        from audiobrain.processing.config import GenerationConfig
        config = GenerationConfig(temperature=0.6, density=1.0, mode='fluid')
        synthesized_audio, out_sr = self.synthesizer.synthesize_from_latent(
            latent_vectors,
            [str(source_path)],
            pipeline=self.processing_pipeline,
            config=config,
        )
        
        # Step 3: Apply effects chain if provided
        if effect_chain is not None:
            print(f"  Applying effects: {effect_chain}")
            synthesized_audio = effect_chain.apply(synthesized_audio, out_sr)
        
        return synthesized_audio, latent_vectors
    
    def generate_and_save(self,
                         source_audio: Union[str, Path],
                         output_path: Union[str, Path],
                         duration: Optional[float] = None,
                         k_neighbors: int = 5,
                         effect_chain=None):
        """
        Generate audio from source and save to file
        
        Args:
            source_audio: Path to source audio file
            output_path: Path to save generated audio
            duration: Optional duration to process
            k_neighbors: Number of nearest neighbors for synthesis
        """
        synthesized_audio, _ = self.generate(
            source_audio,
            duration=duration,
            k_neighbors=k_neighbors,
            effect_chain=effect_chain,
        )
        
        self.synthesizer.save_audio(synthesized_audio, output_path)
    
    def generate_batch(self,
                      source_audios: List[str],
                      output_dir: Optional[str] = None,
                      k_neighbors: int = 5) -> List[tuple]:
        """
        Generate audio from multiple source files
        
        Args:
            source_audios: List of paths to source audio files
            output_dir: Optional directory to save outputs
            k_neighbors: Number of nearest neighbors for synthesis
            
        Returns:
            List of tuples (output_path, synthesized_audio, latent_vectors)
        """
        results = []
        
        for i, source in enumerate(source_audios):
            print(f"\n{'='*60}")
            print(f"Processing [{i+1}/{len(source_audios)}]: {Path(source).name}")
            print('='*60)
            
            synthesized_audio, latents = self.generate(source, k_neighbors=k_neighbors)
            
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"generated_{Path(source).stem}.wav"
                self.synthesizer.save_audio(synthesized_audio, output_path)
            
            results.append((output_path, synthesized_audio, latents))
        
        return results
    
    def get_info(self) -> dict:
        """Get information about the pipeline configuration"""
        return {
            'device': self.device,
            'processing_pipeline': self.processing_pipeline.get_model_info(),
            'synthesizer': self.synthesizer.get_database_info()
        }
    
    def __repr__(self):
        db_size = len(self.synthesizer.audio_database)
        params = self.processing_pipeline.brain.count_parameters()
        return (f"AudioGenerationPipeline(database={db_size} segments, "
                f"transformer_params={params:,}, device={self.device})")


# Convenience function for quick testing
def test_generation_pipeline():
    """Test the complete generation pipeline"""
    print("=" * 60)
    print("AUDIO GENERATION PIPELINE TEST")
    print("=" * 60)
    
    print("\nThis test requires:")
    print("  1. A database of audio files for the synthesizer")
    print("  2. A source audio file to transform")
    print("\nExample usage:")
    print("""
    from audiobrain import AudioGenerationPipeline
    
    # Initialize with database
    pipeline = AudioGenerationPipeline(
        database_files=['samples/*.wav'],
        max_database_segments=500
    )
    
    # Generate new audio
    audio, latents = pipeline.generate('source.wav')
    
    # Save to file
    pipeline.generate_and_save('source.wav', 'output.wav')
    """)
    
    return True


if __name__ == "__main__":
    test_generation_pipeline()
