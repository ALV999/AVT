"""
Test effects module: verify each effect runs, preserves shape, stays within bounds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from audiobrain.effects import (
    EffectChain, bitcrush, pitch_down, pitch_up,
    flange, glitch, distort, delay,
)


SR = 32000
DURATION = 2.0
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def make_wave():
    """2s sine sweep."""
    t = np.linspace(0, DURATION, int(SR * DURATION))
    return (0.5 * np.sin(2 * np.pi * (200 + t * 400) * t)).astype(np.float32)


audio = make_wave()
print(f"Test audio: {audio.shape}, peak={np.max(np.abs(audio)):.3f}")
print()

# --- Bitcrusher ---
print("[Bitcrusher]")
b = bitcrush(audio, SR, bits=4, mix=1.0)
check("shape preserved", b.shape == audio.shape)
check("peak in range", 0 < np.max(np.abs(b)) <= 1.0)

b = bitcrush(audio, SR, bits=999, mix=0.0)  # should clamp and return dry
check("mix=0 returns dry", np.allclose(b, audio))

b = bitcrush(audio, SR, bits=1, downsample=8, mix=0.5)
check("downsample+blend shape ok", b.shape == audio.shape)

# --- Pitch ---
print("[Pitch]")
pd = pitch_down(audio, SR, semitones=7)
check("pitch_down is shorter", len(pd) < len(audio))

pu = pitch_up(audio, SR, semitones=7)
check("pitch_up is longer (higher freq = more samples)", len(pu) > len(audio))

p0 = pitch_down(audio, SR, semitones=0.0)
check("semitones=0 unchanged", np.allclose(p0, audio))

p99 = pitch_down(audio, SR, semitones=99, mix=0.0)
check("mix=0 returns dry (pitch)", np.allclose(p99, audio))

# --- Flanger ---
print("[Flanger]")
f = flange(audio, SR, depth_ms=3, rate_hz=0.3, mix=0.5)
check("flange shape ok", f.shape == audio.shape)
check("flange peak ok", np.max(np.abs(f)) <= 1.0)

f = flange(audio, SR, mix=0.0)
check("flange mix=0 dry", np.allclose(f, audio))

# --- Glitch ---
print("[Glitch]")
g = glitch(audio, SR, intensity=0.3, seed=42)
check("glitch shape ok", g.shape == audio.shape)
check("glitch peak ok", np.max(np.abs(g)) <= 1.0)

g2 = glitch(audio, SR, intensity=0.3, seed=42)
check("glitch reproducible", np.allclose(g, g2))

# --- Distortion ---
print("[Distortion]")
d = distort(audio, SR, drive=5, mix=0.5)
check("distort shape ok", d.shape == audio.shape)
check("distort peak ok", np.max(np.abs(d)) <= 1.0)

d = distort(audio, SR, drive=1, mix=0.0)
check("distort mix=0 dry", np.allclose(d, audio))

# --- Delay ---
print("[Delay]")
dl = delay(audio, SR, time_ms=200, feedback=0.3, mix=0.4)
check("delay shape ok", dl.shape == audio.shape)
check("delay peak ok", np.max(np.abs(dl)) <= 1.0)

dl = delay(audio, SR, time_ms=50, feedback=0, mix=0.0)
check("delay mix=0 dry", np.allclose(dl, audio))

# --- EffectChain ---
print("[EffectChain]")
chain = EffectChain()
chain.add(bitcrush, bits=6, mix=0.3)
chain.add(distort, drive=2, mix=0.4)
chain.add(delay, time_ms=150, feedback=0.2, mix=0.3)
result = chain.apply(audio, SR)
check("chain shape ok", result.shape == audio.shape)
check("chain peak ok", np.max(np.abs(result)) <= 1.0)
check("chain len", len(chain) == 3)

chain.remove(1)
check("chain after remove", len(chain) == 2)

chain.clear()
check("chain cleared", len(chain) == 0)

# --- Safety clamps ---
print("[Safety]")
b = bitcrush(audio, SR, bits=-5, downsample=999, mix=3.0)
check("out-of-range bits clamped", b.shape == audio.shape)
check("out-of-range still within bounds", np.max(np.abs(b)) <= 1.0)

# --- Summary ---
print(f"\n{'='*60}")
total = passed + failed
print(f"  Results: {passed}/{total} passed")
if failed:
    print(f"  {failed} FAILURES")
    sys.exit(1)
else:
    print(f"  All tests passed!")
print(f"{'='*60}")
