"""
Generates 10 diverse random audio samples (5-20s each) for testing the k-NN mosaicing system.
Creates a variety of waveforms, frequencies, and modulations to build a rich database.
"""

import os
import numpy as np
import soundfile as sf
import random

def generate_random_sample(duration: float, sample_rate: int = 32000) -> np.ndarray:
    """Generates a random audio snippet with varying waveform and modulation."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Randomly choose waveform type
    wave_type = random.choice(['sine', 'square', 'saw', 'mixed'])
    
    # Random base frequency (100Hz - 800Hz)
    base_freq = random.uniform(100, 800)
    
    # Random modulation frequency (0.5Hz - 5Hz) for tremolo/vibrato effect
    mod_freq = random.uniform(0.5, 5.0)
    mod_depth = random.uniform(0.3, 0.8)
    
    if wave_type == 'sine':
        # Simple sine wave with frequency sweep
        freq_sweep = base_freq + (t / duration) * random.uniform(-200, 400)
        audio = np.sin(2 * np.pi * freq_sweep * t)
        
    elif wave_type == 'square':
        # Square wave with amplitude modulation
        carrier = np.sign(np.sin(2 * np.pi * base_freq * t))
        envelope = 1.0 - mod_depth * np.sin(2 * np.pi * mod_freq * t)
        audio = carrier * envelope
        
    elif wave_type == 'saw':
        # Sawtooth wave
        audio = 2 * (t * base_freq - np.floor(0.5 + t * base_freq))
        # Apply decay envelope
        envelope = np.linspace(1.0, 0.5, len(audio))
        audio *= envelope
        
    else:  # mixed
        # Combination of two sine waves with different frequencies
        freq2 = base_freq * random.uniform(1.5, 2.5)  # Harmonic or dissonant
        audio = 0.5 * np.sin(2 * np.pi * base_freq * t) + \
                0.3 * np.sin(2 * np.pi * freq2 * t)
        # Add some noise
        noise = np.random.normal(0, 0.1, len(audio))
        audio += noise
    
    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)

def main():
    """Generates 10 random samples and saves them to test_data/"""
    output_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎵 Generating 10 diverse random audio samples...")
    print("-" * 50)
    
    generated_files = []
    
    for i in range(1, 11):
        # Random duration between 5 and 20 seconds
        duration = random.uniform(5.0, 20.0)
        
        print(f"Generating sample {i}/10: {duration:.2f}s...")
        
        audio = generate_random_sample(duration)
        
        filename = f"sample_{i:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        sf.write(filepath, audio, 32000)
        generated_files.append(filepath)
        
        print(f"  ✅ Saved: {filename} ({duration:.2f}s)")
    
    print("-" * 50)
    print(f"🎉 Success! Generated {len(generated_files)} files in {output_dir}")
    print("\nNext step: Run the synthesis pipeline with all files:")
    print(f"   python tests/test_custom_audio.py tests/test_data/sample_*.wav")

if __name__ == "__main__":
    main()
