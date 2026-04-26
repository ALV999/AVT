"""
Audio Synthesizer Module - Generates audio from latent vectors using mosaicing
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import librosa
import soundfile as sf


class AudioMosaicSynthesizer:
    """
    Synthesizes audio by mosaicing (concatenating) audio segments from a database
    based on similarity to target latent vectors.
    
    Uses k-NN search to find the best matching audio segments for each position
    in the sequence, then crossfades them together for smooth transitions.
    """
    
    def __init__(self,
                 segment_duration: float = 0.1,
                 sample_rate: int = 32000,
                 crossfade_duration: float = 0.02,
                 device: Optional[str] = None):
        """
        Initialize the synthesizer
        
        Args:
            segment_duration: Duration of each audio segment in seconds
            sample_rate: Audio sample rate
            crossfade_duration: Duration of crossfade between segments
            device: Device for computation (auto-detects if None)
        """
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.crossfade_duration = crossfade_duration
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        # Calculate samples per segment
        self.samples_per_segment = int(segment_duration * sample_rate)
        self.crossfade_samples = int(crossfade_duration * sample_rate)
        
        # Database storage
        self.audio_database: List[np.ndarray] = []
        self.latent_database: List[torch.Tensor] = []
        self.segment_latents: torch.Tensor = None  # [num_segments, 512]
        
        print(f"AudioMosaicSynthesizer initialized on {self.device}")
        print(f"  Segment duration: {segment_duration}s ({self.samples_per_segment} samples)")
        print(f"  Crossfade: {crossfade_duration}s ({self.crossfade_samples} samples)")
    
    def build_database(self,
                      audio_files: List[str],
                      feature_extractor,
                      pipeline,
                      max_segments: int = 1000):
        """
        Build the audio segment database from a collection of audio files
        
        Args:
            audio_files: List of paths to audio files
            feature_extractor: PANNsFeatureExtractor instance
            pipeline: AudioProcessingPipeline instance
            max_segments: Maximum number of segments to extract
        """
        print(f"\nBuilding audio database from {len(audio_files)} files...")
        
        self.audio_database = []
        self.latent_database = []
        
        segments_extracted = 0
        
        for i, audio_path in enumerate(audio_files):
            if segments_extracted >= max_segments:
                break
                
            print(f"  Processing [{i+1}/{len(audio_files)}]: {Path(audio_path).name}")
            
            try:
                # Load audio
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                
                # Extract segments
                num_segments = len(audio) // self.samples_per_segment
                
                for j in range(num_segments):
                    if segments_extracted >= max_segments:
                        break
                    
                    start_idx = j * self.samples_per_segment
                    end_idx = start_idx + self.samples_per_segment
                    
                    # Extract audio segment
                    segment = audio[start_idx:end_idx]
                    
                    # Skip silent segments
                    if np.max(np.abs(segment)) < 0.01:
                        continue
                    
                    # Get latent vector for this segment
                    # Create temporary file for segment processing
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, segment, self.sample_rate)
                        tmp_path = tmp.name
                    
                    try:
                        _, latent = pipeline.process_audio(tmp_path)
                        latent_vector = latent[0, 0, :]  # Get first frame's latent
                        
                        self.audio_database.append(segment)
                        self.latent_database.append(latent_vector.cpu())
                        segments_extracted += 1
                        
                    finally:
                        import os
                        os.unlink(tmp_path)
                        
            except Exception as e:
                print(f"    Error processing {audio_path}: {e}")
                continue
        
        # Convert to tensors
        if len(self.latent_database) > 0:
            self.segment_latents = torch.stack(self.latent_database).to(self.device)
            print(f"\n✓ Database built: {segments_extracted} segments")
            print(f"  Latent database shape: {self.segment_latents.shape}")
        else:
            print("\n✗ No segments extracted!")
    
    def synthesize(self,
                  target_latents: torch.Tensor,
                  k: int = 5) -> np.ndarray:
        """
        Synthesize audio from target latent vectors
        
        Args:
            target_latents: Target latent vectors [batch, seq_len, 512]
            k: Number of nearest neighbors to consider
            
        Returns:
            Synthesized audio waveform
        """
        if self.segment_latents is None:
            raise ValueError("Database not built. Call build_database() first.")
        
        print(f"\nSynthesizing audio from {target_latents.shape} latents...")
        
        # Ensure target is on correct device
        target_latents = target_latents.to(self.device)
        
        # Handle batch dimension
        if target_latents.dim() == 3:
            target_latents = target_latents[0]  # Take first batch item
        
        # target_latents shape: [seq_len, 512]
        seq_len = target_latents.shape[0]
        
        synthesized_segments = []
        
        with torch.no_grad():
            for i in range(seq_len):
                target = target_latents[i:i+1, :]  # [1, 512]
                
                # Find k-nearest neighbors in latent space
                distances = torch.norm(self.segment_latents - target, dim=1)
                _, indices = torch.topk(distances, k, largest=False)
                
                # Randomly select one of the k best matches for variety
                selected_idx = indices[torch.randint(0, k, (1,))].item()
                
                # Get corresponding audio segment
                segment = self.audio_database[selected_idx]
                synthesized_segments.append(segment)
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{seq_len} segments")
        
        # Concatenate segments with crossfading
        print("  Applying crossfades...")
        output_audio = self._crossfade_concatenate(synthesized_segments)
        
        print(f"✓ Synthesis complete! Output length: {len(output_audio)/self.sample_rate:.2f}s")
        
        return output_audio
    
    def _crossfade_concatenate(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate audio segments with crossfading
        
        Args:
            segments: List of audio segments
            
        Returns:
            Concatenated audio with smooth transitions
        """
        if len(segments) == 0:
            return np.array([], dtype=np.float32)
        
        if len(segments) == 1:
            return segments[0]
        
        # Calculate output length
        total_length = len(segments[0]) + (len(segments) - 1) * (
            self.samples_per_segment - self.crossfade_samples
        )
        
        output = np.zeros(total_length, dtype=np.float32)
        window = self._create_crossfade_window()
        
        pos = 0
        for i, segment in enumerate(segments):
            if i == 0:
                # First segment: just copy
                output[pos:pos+len(segment)] = segment
                pos += self.samples_per_segment - self.crossfade_samples
            else:
                # Apply crossfade
                overlap_start = pos - self.crossfade_samples
                
                # Fade out previous segment
                output[overlap_start:overlap_start+self.crossfade_samples] *= (1 - window)
                
                # Fade in current segment
                fade_in = segment[:self.crossfade_samples] * window
                output[overlap_start:overlap_start+self.crossfade_samples] += fade_in
                
                # Copy rest of segment
                remaining = segment[self.crossfade_samples:]
                output[pos:pos+len(remaining)] = remaining
                
                pos += self.samples_per_segment - self.crossfade_samples
        
        return output
    
    def _create_crossfade_window(self) -> np.ndarray:
        """Create a smooth crossfade window (raised cosine)"""
        t = np.linspace(0, 1, self.crossfade_samples)
        return 0.5 * (1 - np.cos(np.pi * t))
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save synthesized audio to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        print(f"✓ Saved audio to: {output_path}")
    
    def get_database_info(self) -> dict:
        """Get information about the audio database"""
        return {
            'num_segments': len(self.audio_database),
            'segment_duration': self.segment_duration,
            'sample_rate': self.sample_rate,
            'latent_dim': 512 if self.segment_latents is not None else None,
            'device': self.device
        }
    
    def __repr__(self):
        db_size = len(self.audio_database)
        return f"AudioMosaicSynthesizer(database={db_size} segments, device={self.device})"


# Convenience function for testing
def test_synthesizer():
    """Test the synthesizer module"""
    print("=" * 60)
    print("AUDIO MOSAIC SYNTHESIZER TEST")
    print("=" * 60)
    
    synth = AudioMosaicSynthesizer()
    print(f"\n{synth}")
    print(f"Database info: {synth.get_database_info()}")
    
    print("\nNote: To test synthesis, you need to:")
    print("  1. Build a database with build_database()")
    print("  2. Provide target latent vectors")
    print("  3. Call synthesize()")
    
    return True


if __name__ == "__main__":
    test_synthesizer()
