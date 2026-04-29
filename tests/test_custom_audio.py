"""
Test script for the full AudioBrain pipeline with user inference controls.
Supports multiple input files, validation, and generation parameters.
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

def run_custom_pipeline(input_files: list, config: GenerationConfig):
    """Runs the complete audio generation pipeline with validation and config."""
    
    # 1. Validate Inputs
    print("🛡️  Validating inputs...")
    is_valid, message = AudioValidator.validate_files(input_files, min_duration=2.0)
    if not is_valid:
        print(f"❌ Validation Failed: {message}")
        return False
    print(f"✅ {message}")

    # 2. Setup Paths & Device
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    output_path = os.path.join(test_dir, "generated_output.wav")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    print(f"⚙️  Config: Temp={config.temperature}, Mode={config.mode}, Density={config.density}")

    try:
        # 3. Initialize Components
        print("\n⚙️  Initializing components...")
        # Note: Pipeline handles its own sample rate internally usually, or we pass target_sr
        processor = AudioProcessingPipeline(device=device)
        
        # Synthesizer now takes sample_rate from config or defaults
        synthesizer = AudioMosaicSynthesizer(
            segment_duration=config.segment_duration,
            sample_rate=22050, # Match pipeline default
            device=device
        )

        # 4. Process Inputs (Extract Features)
        print(f"\n📥 Processing {len(input_files)} file(s)...")
        all_latents = []
        
        # Simple concatenation of latents for multiple inputs for now
        # In a more advanced version, we might weight them differently
        for f in input_files:
            print(f"  Processing: {os.path.basename(f)}")
            latents = processor.process_file(f)
            all_latents.append(latents)
        
        # Concatenate along sequence dimension (dim 1)
        if len(all_latents) == 1:
            latent_vectors = all_latents[0]
        else:
            latent_vectors = torch.cat(all_latents, dim=1)
            
        print(f"✅ Total latent shape: {latent_vectors.shape}")

        # 5. Synthesize
        print("\n🎼 Synthesizing (k-NN Mosaicing)...")
        # Pass the first file as reference for database building if needed, 
        # but ideally we build DB from all inputs. 
        # For this test, we build DB from the combined inputs implicitly or just the first one.
        # To make it robust, let's pass all inputs to a hypothetical build step or rely on internal logic.
        # Current synthesizer expects one source to build DB if none exists. 
        # Let's pass the first file as the primary source for DB construction for simplicity in this test.
        source_for_db = input_files[0] 

        output_audio, output_sr = synthesizer.synthesize_from_latent(
            latent_vectors, 
            source_for_db, 
            pipeline=processor, 
            config=config # Pass the new config object
        )

        # 6. Save Result
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
    parser = argparse.ArgumentParser(description="Generate audio with k-NN mosaicing.")
    parser.add_argument("input_files", nargs="+", help="Input .wav files (1 to 10)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Creativity (0.0-1.0)")
    parser.add_argument("--density", type=float, default=0.5, help="Segment density/glitch (0.0-1.0)")
    parser.add_argument("--mode", type=str, default="fluid", choices=["fluid", "glitch", "evolving"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--segment-duration", type=float, default=1.0, help="Segment duration in seconds")

    args = parser.parse_args()

    config = GenerationConfig(
        temperature=args.temperature,
        density=args.density,
        mode=args.mode,
        seed=args.seed,
        segment_duration=args.segment_duration
    )

    success = run_custom_pipeline(args.input_files, config)
    sys.exit(0 if success else 1)