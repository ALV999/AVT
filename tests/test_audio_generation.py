"""
Test script for the full AudioBrain pipeline:
1. Generates a synthetic source audio file
2. Extracts features using PANNs
3. Processes through the Transformer
4. Synthesizes new audio via k-NN mosaicing
5. Saves the result
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioSynthesizer

def generate_sample_audio(output_path: str, duration: float = 5.0, sr: int = 22050):
    """Generate a simple sweeping sine wave for testing."""
    print(f"🎵 Generating sample audio ({duration}s at {sr}Hz)...")
    
    t = np.linspace(0, duration, int(sr * duration))
    # Frequency sweeps from 200Hz to 800Hz
    freq = 200 + (t / duration) * 600 
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    sf.write(output_path, audio, sr)
    print(f"✅ Sample saved: {output_path}")
    return output_path

def run_full_pipeline():
    """Run the complete audio generation pipeline."""
    # Setup paths
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    source_path = os.path.join(test_dir, "source_sample.wav")
    output_path = os.path.join(test_dir, "generated_output.wav")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    try:
        # Step 1: Generate sample audio if it doesn't exist
        if not os.path.exists(source_path):
            generate_sample_audio(source_path)
        
        # Step 2: Initialize components
        print("\n⚙️  Initializing components...")
        processor = AudioProcessingPipeline(device=device)
        synthesizer = AudioSynthesizer(device=device)
        
        # Step 3: Process audio (Extract -> Transform)
        print(f"\n📥 Processing: {source_path}")
        latent_vectors = processor.process_file(source_path)
        print(f"✅ Latent vectors shape: {latent_vectors.shape}")
        
        # Step 4: Synthesize new audio
        print("\n🎼 Synthesizing new audio (k-NN Mosaicing)...")
        output_audio, output_sr = synthesizer.synthesize_from_latent(
            latent_vectors, 
            source_path, 
            n_neighbors=5, 
            crossfade_samples=4410  # 0.2s crossfade at 22050Hz
        )
        
        # Step 5: Save result
        sf.write(output_path, output_audio, output_sr)
        print(f"\n🎉 Success! Output saved to: {output_path}")
        print(f"   Duration: {len(output_audio)/output_sr:.2f}s")
        print(f"   Sample Rate: {output_sr}Hz")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
