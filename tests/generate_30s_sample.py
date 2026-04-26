"""
Generates a 30-second synthetic audio sample for testing the pipeline.
Creates a complex soundscape with sweeping frequencies and harmonics.
"""

import numpy as np
import soundfile as sf
import os

def generate_complex_soundscape(duration: float = 30.0, sr: int = 22050):
    """Generates a rich 30s soundscape with multiple layers."""
    print(f"🎵 Generating {duration}s complex soundscape...")
    
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    # Layer 1: Base drone (sweeping low freq)
    freq1 = 100 + (t / duration) * 200
    audio += 0.3 * np.sin(2 * np.pi * freq1 * t)
    
    # Layer 2: Mid-range harmonic sweep
    freq2 = 400 + (t / duration) * 800
    audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
    
    # Layer 3: High frequency texture
    freq3 = 2000 + (t / duration) * 1000
    audio += 0.1 * np.sin(2 * np.pi * freq3 * t) * np.sin(2 * np.pi * 5 * t)
    
    # Layer 4: Rhythmic modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return audio, sr

def main():
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    output_path = os.path.join(test_dir, "30s_sample.wav")
    
    audio, sr = generate_complex_soundscape()
    sf.write(output_path, audio, sr)
    
    print(f"✅ Generated: {output_path}")
    print(f"   Duration: {len(audio)/sr:.2f}s")
    print(f"   Sample Rate: {sr}Hz")
    print(f"   Samples: {len(audio)}")

if __name__ == "__main__":
    main()
