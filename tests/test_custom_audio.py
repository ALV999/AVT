"""
Test script for the full AudioBrain pipeline with user inference controls.
Accepts multiple input files and generation parameters.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioMosaicSynthesizer
from audiobrain.processing.config import GenerationConfig
from audiobrain.processing.validator import AudioValidator

def run_custom_pipeline(input_files, args):
    """Runs the complete audio generation pipeline with custom settings."""
    
    # 1. Validate inputs
    print("🔍 Validating input files...")
    is_valid, msg = AudioValidator.validate_files(input_files, min_duration=2.0)
    if not is_valid:
        print(f"❌ Validation failed: {msg}")
        return False
    print(f"✅ {msg}")
    
    # 2. Setup paths
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    output_path = os.path.join(test_dir, "generated_output.wav")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    try:
        # 3. Initialize components
        print("\n⚙️  Initializing components...")
        processor = AudioProcessingPipeline(device=device)
        synthesizer = AudioMosaicSynthesizer(device=device)
        
        # 4. Create Generation Config
        config = GenerationConfig(
            temperature=args.temperature,
            density=args.density,
            mode=args.mode,
            seed=args.seed,
            segment_duration=args.segment_duration
        )
        print(f"⚙️  Config: Temp={config.temperature}, Mode={config.mode}, Density={config.density}")
        
        # 5. Process inputs (Extract features from all files)
        print(f"\n📥 Processing {len(input_files)} file(s)...")
        all_latents = []
        
        for f in input_files:
            print(f"  Processing: {os.path.basename(f)}")
            latents = processor.process_file(f)
            all_latents.append(latents)
        
        # Concatenate latents from all sources
        combined_latents = torch.cat(all_latents, dim=1)  # [1, total_seq, 512]
        print(f"✅ Combined latent shape: {combined_latents.shape}")
        
        # 6. Synthesize
        print("\n🎼 Synthesizing (k-NN Mosaicing)...")
        output_audio, output_sr = synthesizer.synthesize_from_latent(
            combined_latents, 
            input_files[0],  # Use first file as reference for DB building if needed
            pipeline=processor,
            config=config,
            crossfade_samples=int(0.05 * 22050)  # 50ms crossfade
        )
        
        # 7. Save result
        sf.write(output_path, output_audio, output_sr)
        print(f"\n🎉 Success! Output: {output_path}")
        print(f"   Duration: {len(output_audio)/output_sr:.2f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioBrain Custom Generation")
    parser.add_argument("inputs", nargs="+", help="Input WAV files (1 to 10)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Creativity (0.0-1.0)")
    parser.add_argument("--density", type=float, default=0.5, help="Segment density (0.0-1.0)")
    parser.add_argument("--mode", type=str, default="fluid", choices=["fluid", "glitch", "evolving"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--segment-duration", type=float, default=1.0, help="Segment duration in seconds")
    
    args = parser.parse_args()
    
    success = run_custom_pipeline(args.inputs, args)
    sys.exit(0 if success else 1)
