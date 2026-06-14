"""
Test script for custom audio generation with user inference controls.
Usage: python3 tests/test_custom_audio.py audio1.wav audio2.wav --temperature 0.7 --mode fluid
"""
import os, sys, argparse, torch, numpy as np, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobrain.model.pipeline import AudioProcessingPipeline
from audiobrain.model.synthesizer import AudioMosaicSynthesizer
from audiobrain.processing.config import GenerationConfig
from audiobrain.processing.validator import AudioValidator

def run_custom_pipeline(input_files, args):
    print(f"🔍 Validating {len(input_files)} input files...")
    
    # Debug: Print what we received
    print(f"   Received paths: {input_files}")
    
    is_valid, msg = AudioValidator.validate_files(input_files)
    if not is_valid:
        print(f"❌ Error: Validation failed: {msg}")
        return False
    print(f"✅ {msg}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    try:
        print("\n⚙️  Initializing components...")
        processor = AudioProcessingPipeline(device=device)
        
        # Use segment duration from config or default
        seg_dur = getattr(args, 'segment_duration', 1.0)
        synthesizer = AudioMosaicSynthesizer(segment_duration=seg_dur, device=device)

        # Build configuration
        config = GenerationConfig(
            temperature=args.temperature,
            density=args.density,
            mode=args.mode,
            seed=args.seed,
            segment_duration=seg_dur
        )
        print(f"   Config: Temp={config.temperature}, Mode={config.mode}, Density={config.density}, Seed={config.seed}")

        print(f"\n📥 Processing inputs...")
        # Process all input files and concatenate latents? 
        # For now, let's just process the first one to get a target sequence, 
        # but build the DB from ALL files.
        # Better approach: Process a dummy or combined signal for target length?
        # Simplest for now: Target length based on total input duration
        
        all_latents = []
        total_duration = 0
        
        for f in input_files:
            print(f"   Extracting from {os.path.basename(f)}...")
            latents = processor.process_file(f)
            all_latents.append(latents)
            
            # Get duration for total time estimation
            audio, sr = librosa.load(f, sr=None)
            total_duration += len(audio)/sr

        # Concatenate latents to form a long target sequence (simple approach)
        # Shape: [batch, seq, dim] -> we want [1, total_seq, dim]
        # Note: process_file returns [1, seq, 512] usually
        if len(all_latents) == 1:
            target_latents = all_latents[0]
        else:
            target_latents = torch.cat(all_latents, dim=1)
            
        print(f"✅ Total latent shape: {target_latents.shape} (approx {total_duration:.1f}s)")

        print("\n🎼 Synthesizing...")
        output_audio, output_sr = synthesizer.synthesize_from_latent(
            target_latents,
            input_files, # Pass list of files to build DB from all
            pipeline=processor,
            config=config
        )

        out_path = "tests/test_data/generated_output.wav"
        sf.write(out_path, output_audio, output_sr)
        print(f"\n🎉 Success! Output: {out_path}")
        print(f"   Duration: {len(output_audio)/output_sr:.2f}s")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio with k-NN mosaicing")
    parser.add_argument("inputs", nargs="+", help="Input .wav files (1 or more)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Creativity (0.0-1.0)")
    parser.add_argument("--density", type=float, default=0.5, help="Segment density/glitch (0.0-1.0)")
    parser.add_argument("--mode", choices=['fluid', 'glitch', 'evolving'], default='fluid')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--segment-duration", type=float, default=1.0, help="Segment size in seconds")
    
    args = parser.parse_args()
    
    # Expand wildcards if shell didn't (unlikely in bash but safe)
    import glob
    expanded_inputs = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        if matches:
            expanded_inputs.extend(matches)
        else:
            expanded_inputs.append(pattern)
    
    success = run_custom_pipeline(expanded_inputs, args)
    sys.exit(0 if success else 1)
