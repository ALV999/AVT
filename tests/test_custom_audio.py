"""
Test script for custom audio files with automatic duration extension.
Ensures output is at least 30 seconds long.

Usage:
    python test_custom_audio.py /path/to/your/audio.wav
    python test_custom_audio.py  # Uses default 30s sample
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioMosaicSynthesizer

MIN_DURATION = 30.0  # seconds
TARGET_DURATION = 35.0  # Generate slightly more to be safe

def extend_audio(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Extends audio by looping if it's shorter than target duration."""
    current_duration = len(audio) / sr
    if current_duration >= target_duration:
        return audio
    
    print(f"  ⚠️  Input too short ({current_duration:.2f}s). Extending to {target_duration}s...")
    repetitions = int(np.ceil(target_duration / current_duration))
    extended = np.tile(audio, repetitions)[:int(target_duration * sr)]
    
    # Apply quick crossfade at loop points to avoid clicks
    crossfade_samples = int(0.05 * sr)  # 50ms crossfade
    window = np.linspace(0, 1, crossfade_samples)
    
    for i in range(1, repetitions):
        start_idx = int(i * len(audio)) - crossfade_samples
        if start_idx < len(extended) - crossfade_samples:
            extended[start_idx:start_idx+crossfade_samples] *= (1 - window)
            extended[start_idx+crossfade_samples:start_idx+2*crossfade_samples] *= window
    
    return extended

def run_custom_pipeline(audio_path: str):
    """Runs the complete pipeline on a custom audio file."""
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return False
    
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Load and validate input
    print(f"\n📥 Loading: {audio_path}")
    audio_input, sr = sf.read(audio_path)
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)  # Convert to mono
        print("  Converted to mono")
    
    duration = len(audio_input) / sr
    print(f"  Duration: {duration:.2f}s | Sample Rate: {sr}Hz")
    
    # Extend if necessary
    if duration < MIN_DURATION:
        audio_input = extend_audio(audio_input, sr, TARGET_DURATION)
        duration = len(audio_input) / sr
        print(f"  Extended duration: {duration:.2f}s")
    
    # Save extended input for reference
    extended_input_path = os.path.join(test_dir, "input_extended.wav")
    sf.write(extended_input_path, audio_input, sr)
    print(f"  Saved extended input: {extended_input_path}")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🚀 Device: {device}")
    
    try:
        print("\n⚙️  Initializing components...")
        processor = AudioProcessingPipeline(device=device)
        synthesizer = AudioMosaicSynthesizer(sample_rate=sr, device=device)
        
        # Process through PANNs + Transformer
        print(f"\n🔄 Processing audio ({duration:.2f}s)...")
        latent_vectors = processor.process_file(extended_input_path)
        print(f"✅ Latent shape: {latent_vectors.shape}")
        
        # Synthesize with k-NN
        print("\n🎼 Synthesizing (k-NN Mosaicing)...")
        output_audio, output_sr = synthesizer.synthesize_from_latent(
            latent_vectors, 
            extended_input_path, 
            pipeline=processor,
            n_neighbors=5, 
            crossfade_samples=int(0.05 * sr)  # 50ms crossfade
        )
        
        # Save output
        output_path = os.path.join(test_dir, "generated_output.wav")
        sf.write(output_path, output_audio, output_sr)
        
        out_duration = len(output_audio) / output_sr
        print(f"\n🎉 Success!")
        print(f"   Output: {output_path}")
        print(f"   Duration: {out_duration:.2f}s")
        print(f"   Sample Rate: {output_sr}Hz")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default to 30s sample if it exists
        default_path = os.path.join(os.path.dirname(__file__), "test_data", "30s_sample.wav")
        if os.path.exists(default_path):
            audio_file = default_path
        else:
            print("❌ No audio file specified and default sample not found.")
            print("Usage: python test_custom_audio.py /path/to/audio.wav")
            print("Or run: python generate_30s_sample.py first")
            sys.exit(1)
    
    success = run_custom_pipeline(audio_file)
    sys.exit(0 if success else 1)
