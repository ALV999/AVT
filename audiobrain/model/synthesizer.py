"""
Audio Synthesizer Module - Generates audio from latent vectors using k-NN mosaicing.
Now integrated with the processing pipeline for validation, preprocessing, and segmentation.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import librosa
import soundfile as sf
import tempfile
import os

from audiobrain.processing.config import GenerationConfig
from audiobrain.processing.validator import AudioValidator
from audiobrain.processing.preprocessor import AudioPreprocessor
from audiobrain.processing.segmenter import AudioSegmenter


class AudioMosaicSynthesizer:
    """
    Synthesizes audio by mosaicing audio segments from a database based on 
    similarity to target latent vectors using k-NN search and crossfading.
    """
    
    def __init__(self,
                 config: Optional[GenerationConfig] = None,
                 device: Optional[str] = None):
        
        self.config = config or GenerationConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        # Calculate samples per segment from config
        self.samples_per_segment = int(self.config.segment_duration * 22050)
        self.crossfade_samples = int(self.config.crossfade_duration * 22050)
        
        # Database storage
        self.audio_database: List[np.ndarray] = []
        self.latent_database: Optional[torch.Tensor] = None
        self.segment_energies: List[float] = []  # For 'evolving' mode
        
        print(f"AudioMosaicSynthesizer initialized on {self.device}")
        print(f"  Segment: {self.config.segment_duration}s ({self.samples_per_segment} samples)")
        print(f"  Crossfade: {self.config.crossfade_duration}s ({self.crossfade_samples} samples)")
        print(f"  Mode: {self.config.mode}, Temperature: {self.config.temperature}")
    
    def build_database(self,
                      audio_files: List[str],
                      feature_extractor,
                      pipeline,
                      max_segments: int = 100):
        """Builds the audio segment database with corresponding latent vectors."""
        print(f"\n📚 Building database from {len(audio_files)} files...")
        
        self.audio_database = []
        self.segment_energies = []
        latent_list = []
        segments_extracted = 0
        
        for i, audio_path in enumerate(audio_files):
            if segments_extracted >= max_segments:
                break
                
            print(f"  Processing [{i+1}/{len(audio_files)}]: {Path(audio_path).name}")
            
            try:
                # Preprocess audio
                audio, sr = AudioPreprocessor.preprocess(
                    audio_path, 
                    target_sr=22050,
                    remove_silent=True,
                    normalize_audio=True
                )
                
                # Segment audio with energy calculation
                segments, energies = AudioSegmenter.segment_with_features(
                    audio, sr, self.config.segment_duration
                )
                
                for j, (segment, energy) in enumerate(zip(segments, energies)):
                    if segments_extracted >= max_segments:
                        break
                    
                    # Skip very silent segments
                    if np.max(np.abs(segment)) < 0.01:
                        continue
                    
                    # Get latent vector for this segment
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, segment, 22050)
                        tmp_path = tmp.name
                    
                    try:
                        _, latent = pipeline.process_audio(tmp_path)
                        latent_vector = latent[0, 0, :]
                        
                        self.audio_database.append(segment)
                        self.segment_energies.append(energy)
                        latent_list.append(latent_vector.cpu())
                        segments_extracted += 1
                    finally:
                        os.unlink(tmp_path)
                        
            except Exception as e:
                print(f"    Error processing {audio_path}: {e}")
                continue
        
        if latent_list:
            self.latent_database = torch.stack(latent_list).to(self.device)
            print(f"\n✓ Database built: {segments_extracted} segments")
            print(f"  Latent shape: {self.latent_database.shape}")
        else:
            raise ValueError("Failed to build database. No segments extracted.")

    def synthesize_from_latent(self,
                               target_latents: torch.Tensor,
                               source_audio_paths: List[str],
                               pipeline=None,
                               n_neighbors: int = 5) -> Tuple[np.ndarray, int]:
        """
        Main entry point: Validates, builds DB from source audio, and synthesizes using k-NN.
        """
        print(f"\n🎼 Starting Synthesis Pipeline...")
        print(f"  Input files: {len(source_audio_paths)}")
        print(f"  Target shape: {target_latents.shape}")
        print(f"  Neighbors (k): {n_neighbors}")

        # Validate input files
        valid_files, errors = AudioValidator.validate_files(
            source_audio_paths, 
            min_total_duration=self.config.min_input_duration
        )
        
        if not valid_files:
            raise ValueError(f"No valid input files. Errors: {errors}")
        
        if errors:
            print(f"\n⚠️  Warnings: {len(errors)} files had issues but continuing with valid ones")

        # Build database if needed
        if pipeline is not None and self.latent_database is None:
            print("\n  Building database from validated sources...")
            self.build_database(valid_files, pipeline.feature_extractor, pipeline)

        if self.latent_database is not None:
            return self._synthesize_knn(target_latents, n_neighbors)
        else:
            print("  ⚠️  No DB found. Falling back to sequential mapping.")
            # Fallback: load first file and map sequentially
            audio_full, _ = librosa.load(valid_files[0], sr=22050, mono=True)
            return self._synthesize_sequential(target_latents, audio_full)

    def _synthesize_knn(self, target_latents: torch.Tensor, k: int) -> Tuple[np.ndarray, int]:
        """Performs true k-NN synthesis with temperature and density control."""
        target_latents = target_latents.to(self.device)
        if target_latents.dim() == 3:
            target_latents = target_latents[0]
            
        seq_len = target_latents.shape[0]
        synthesized_segments = []
        db_size = self.latent_database.shape[0]
        actual_k = min(k, db_size)
        
        print(f"  Searching {db_size} segments with mode='{self.config.mode}'...")
        
        with torch.no_grad():
            for i in range(seq_len):
                # Apply density filter
                if self.config.density < 1.0 and np.random.random() > self.config.density:
                    # Insert silence or skip segment based on mode
                    if self.config.mode != 'glitch':
                        synthesized_segments.append(np.zeros(self.samples_per_segment, dtype=np.float32))
                    continue
                
                target = target_latents[i:i+1, :]
                distances = torch.norm(self.latent_database - target, dim=1)
                
                # Apply temperature to distances
                if self.config.temperature > 0:
                    # Softmax-like selection with temperature
                    exp_distances = torch.exp(-distances / (self.config.temperature + 0.1))
                    probs = exp_distances / exp_distances.sum()
                    
                    # Sample from distribution
                    indices = torch.multinomial(probs, min(actual_k, len(probs)), replacement=False)
                    selected_idx = indices[torch.randint(0, len(indices), (1,))].item()
                else:
                    # Deterministic: pick closest
                    _, indices = torch.topk(distances, actual_k, largest=False)
                    selected_idx = indices[0].item()
                
                synthesized_segments.append(self.audio_database[selected_idx])
                
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i+1}/{seq_len}")
        
        output = self._crossfade_concatenate(synthesized_segments)
        return output, 22050

    def _synthesize_sequential(self, target_latents: torch.Tensor, audio_full: np.ndarray) -> Tuple[np.ndarray, int]:
        """Fallback: Sequential mapping."""
        seq_len = target_latents.shape[0]
        synthesized_segments = []
        num_segs = len(audio_full) // self.samples_per_segment
        
        for j in range(min(seq_len, num_segs)):
            start = j * self.samples_per_segment
            synthesized_segments.append(audio_full[start:start+self.samples_per_segment])
                
        return self._crossfade_concatenate(synthesized_segments), 22050
    
    def _crossfade_concatenate(self, segments: List[np.ndarray]) -> np.ndarray:
        """Concatenates segments with raised-cosine crossfades."""
        if not segments:
            return np.array([], dtype=np.float32)
        if len(segments) == 1:
            return segments[0]
        
        total_length = len(segments[0]) + (len(segments) - 1) * (self.samples_per_segment - self.crossfade_samples)
        output = np.zeros(total_length, dtype=np.float32)
        window = self._create_crossfade_window()
        
        pos = 0
        for i, segment in enumerate(segments):
            if i == 0:
                output[pos:pos+len(segment)] = segment
                pos += self.samples_per_segment - self.crossfade_samples
            else:
                overlap_start = pos - self.crossfade_samples
                output[overlap_start:overlap_start+self.crossfade_samples] *= (1 - window)
                fade_in = segment[:self.crossfade_samples] * window
                output[overlap_start:overlap_start+self.crossfade_samples] += fade_in
                remaining = segment[self.crossfade_samples:]
                output[pos:pos+len(remaining)] = remaining
                pos += self.samples_per_segment - self.crossfade_samples
        
        return output
    
    def _create_crossfade_window(self) -> np.ndarray:
        t = np.linspace(0, 1, max(2, self.crossfade_samples))
        return 0.5 * (1 - np.cos(np.pi * t))

    def save_audio(self, audio: np.ndarray, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, 22050)
        print(f"✓ Saved: {output_path}")

    def get_database_info(self) -> dict:
        return {
            'num_segments': len(self.audio_database),
            'sample_rate': 22050,
            'latent_dim': self.latent_database.shape[1] if self.latent_database is not None else None,
            'device': self.device,
            'mode': self.config.mode
        }

    def __repr__(self):
        return f"AudioMosaicSynthesizer(db={len(self.audio_database)}, mode={self.config.mode}, device={self.device})"
