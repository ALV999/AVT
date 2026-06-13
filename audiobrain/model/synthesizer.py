"""
Audio Synthesizer Module - Generates audio from latent vectors using k-NN mosaicing.
Integrated with the new processing module for validation and segmentation.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import soundfile as sf
import tempfile
import os

# Import new processing modules
from audiobrain.processing.config import GenerationConfig, PreprocessingConfig
from audiobrain.processing.validator import AudioValidator
from audiobrain.processing.preprocessor import AudioPreprocessor
from audiobrain.processing.segmenter import AudioSegmenter


class AudioMosaicSynthesizer:
    """
    Synthesizes audio by mosaicing audio segments from a database based on 
    similarity to target latent vectors using k-NN search and crossfading.
    """
    
    def __init__(self,
                 segment_duration: float = 1.0,
                 sample_rate: int = 32000,
                 crossfade_duration: float = 0.1,
                 device: Optional[str] = None):
        
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.crossfade_duration = crossfade_duration
        self.device = device or ('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        self.samples_per_segment = int(segment_duration * sample_rate)
        self.crossfade_samples = int(crossfade_duration * sample_rate)
        
        # Database storage
        self.audio_database: List[np.ndarray] = []
        self.latent_database: Optional[torch.Tensor] = None
        self.energy_database: List[float] = []
        
        # Initialize processing tools
        self.preprocessor = AudioPreprocessor(PreprocessingConfig(target_sr=sample_rate))
        self.segmenter = AudioSegmenter(segment_duration=segment_duration, sample_rate=sample_rate)
        
        print(f"AudioMosaicSynthesizer initialized on {self.device}")
        print(f"  Segment: {segment_duration}s ({self.samples_per_segment} samples)")
        print(f"  Crossfade: {crossfade_duration}s ({self.crossfade_samples} samples)")
    
    def build_database(self,
                      audio_files: List[str],
                      feature_extractor,
                      pipeline,
                      max_segments: int = 100):
        """Builds the audio segment database with corresponding latent vectors."""
        print(f"\nBuilding database from {len(audio_files)} files...")
        
        # Validate inputs first
        is_valid, msg = AudioValidator.validate_files(audio_files, min_duration=self.segment_duration)
        if not is_valid:
            raise ValueError(f"Validation failed: {msg}")
        print(f"  [OK] {msg}")
        
        self.audio_database = []
        self.energy_database = []
        latent_list = []
        segments_extracted = 0
        
        for i, audio_path in enumerate(audio_files):
            if segments_extracted >= max_segments:
                break
                
            print(f"  Processing [{i+1}/{len(audio_files)}]: {Path(audio_path).name}")
            
            try:
                # Load and preprocess (use soundfile instead of librosa)
                audio_raw, sr = sf.read(audio_path)
                if audio_raw.ndim > 1:
                    audio_raw = audio_raw.mean(axis=1)  # mono
                audio_proc, _ = self.preprocessor.preprocess(audio_raw, sr)
                
                # Segment with energy calculation
                segments = self.segmenter.segment(audio_proc, min_energy=0.01)
                
                for segment, energy in segments:
                    if segments_extracted >= max_segments:
                        break
                    
                    # Skip very silent segments
                    if energy < 0.01:
                        continue
                    
                    # Get latent vector for this segment
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, segment, self.sample_rate)
                        tmp_path = tmp.name
                    
                    try:
                        _, latent = pipeline.process_audio(tmp_path)
                        latent_vector = latent[0, 0, :]
                        
                        self.audio_database.append(segment)
                        self.energy_database.append(energy)
                        latent_list.append(latent_vector.cpu())
                        segments_extracted += 1
                    finally:
                        os.unlink(tmp_path)
                        
            except Exception as e:
                print(f"    Error processing {audio_path}: {e}")
                continue
        
        if latent_list:
            self.latent_database = torch.stack(latent_list).to(self.device)
            print(f"\n[OK] Database built: {segments_extracted} segments")
            print(f"  Latent shape: {self.latent_database.shape}")
        else:
            raise ValueError("Failed to build database. No valid segments extracted.")

    def synthesize_from_latent(self,
                               target_latents: torch.Tensor,
                               source_audio_paths: List[str],
                               pipeline=None,
                               config: Optional[GenerationConfig] = None) -> Tuple[np.ndarray, int]:
        """
        Main entry point: Builds DB from source audio and synthesizes using k-NN.
        """
        if config is None:
            config = GenerationConfig()
        
        if config.mode == 'glitch':
            self.crossfade_samples = int(self.crossfade_samples * 0.2)

        print(f"\n[Generating] Starting Synthesis (Mode: {config.mode}, Temp: {config.temperature})...")
        print(f"  Target shape: {target_latents.shape}")

        if pipeline is not None and self.latent_database is None:
            print("  Building database from sources...")
            self.build_database(source_audio_paths, pipeline.feature_extractor, pipeline)

        if self.latent_database is not None:
            return self._synthesize_knn(target_latents, config)
        else:
            print("  [WARNING] No DB found. Falling back to sequential mapping.")
            audio_raw, sr = sf.read(source_audio_paths[0])
            if audio_raw.ndim > 1:
                audio_raw = audio_raw.mean(axis=1)
            audio_proc, _ = self.preprocessor.preprocess(audio_raw, sr)
            return self._synthesize_sequential(target_latents, audio_proc)

    def _synthesize_knn(self, target_latents: torch.Tensor, config: GenerationConfig) -> Tuple[np.ndarray, int]:
        """Performs k-NN synthesis with temperature and mode control."""
        target_latents = target_latents.to(self.device)
        if target_latents.dim() == 3:
            target_latents = target_latents[0]
            
        seq_len = target_latents.shape[0]
        synthesized_segments = []
        db_size = self.latent_database.shape[0]
        
        energy_threshold = 0.01 * (1.0 - config.density)
        
        print(f"  Searching {db_size} segments (Temp={config.temperature}, Density={config.density})...")
        
        with torch.no_grad():
            for i in range(seq_len):
                target = target_latents[i:i+1, :]
                distances = torch.norm(self.latent_database - target, dim=1)
                
                valid_indices = [idx for idx, e in enumerate(self.energy_database) if e >= energy_threshold]
                if not valid_indices:
                    valid_indices = list(range(db_size))
                
                valid_distances = distances[valid_indices]
                
                if config.temperature > 0.5:
                    noise_scale = (config.temperature - 0.5) * 2.0
                    noise = torch.randn_like(valid_distances) * noise_scale
                    valid_distances = valid_distances + noise
                
                k_to_select = min(5, len(valid_indices))
                _, indices_local = torch.topk(valid_distances, k_to_select, largest=False)
                
                if config.mode == 'evolving':
                    selected_idx_local = indices_local[0].item()
                else:
                    selected_idx_local = indices_local[torch.randint(0, len(indices_local), (1,))].item()
                
                global_idx = valid_indices[selected_idx_local]
                synthesized_segments.append(self.audio_database[global_idx])
                
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i+1}/{seq_len}")
        
        return self._crossfade_concatenate(synthesized_segments), self.sample_rate

    def _synthesize_sequential(self, target_latents: torch.Tensor, audio_full: np.ndarray) -> Tuple[np.ndarray, int]:
        """Fallback: Sequential mapping."""
        seq_len = target_latents.shape[0]
        synthesized_segments = []
        num_segs = len(audio_full) // self.samples_per_segment
        
        for j in range(min(seq_len, num_segs)):
            start = j * self.samples_per_segment
            synthesized_segments.append(audio_full[start:start+self.samples_per_segment])
                
        return self._crossfade_concatenate(synthesized_segments), self.sample_rate
    
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
        t = np.linspace(0, 1, self.crossfade_samples)
        return 0.5 * (1 - np.cos(np.pi * t))

    def save_audio(self, audio: np.ndarray, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        print(f"[OK] Saved: {output_path}")

    def get_database_info(self) -> dict:
        return {
            'num_segments': len(self.audio_database),
            'sample_rate': self.sample_rate,
            'latent_dim': self.latent_database.shape[1] if self.latent_database is not None else None,
            'device': self.device
        }

    def __repr__(self):
        return f"AudioMosaicSynthesizer(db={len(self.audio_database)}, device={self.device})"
