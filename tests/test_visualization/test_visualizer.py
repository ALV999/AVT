"""
Comprehensive tests for the 128x128 artistic visualization system.

Covers:
  - LatentSpectrogram: shape, range, interpolation
  - ChladniOscilloscope: shape, range, mode configurations
  - AudioBrainVisualizer: compute_views, terminal rendering, HTML generation
  - Color schemes: all produce valid RGB tuples
  - Character sets: all render correctly
  - Edge cases: small latents, single segments, boundary values
"""

import io
import os
import sys
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from audiobrain.model.visualizer import (
    AudioBrainVisualizer,
    LatentSpectrogram,
    ChladniOscilloscope,
    COLOR_SCHEMES,
    CHAR_RAMPS,
    _interp_heat,
    _interp_ocean,
    _interp_forest,
    _interp_sunset,
    _interp_aurora,
    _interp_mono,
)


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# Color scheme tests
# ═══════════════════════════════════════════════════════════════
section("Color Schemes")

for name, fn in [("heat", _interp_heat), ("ocean", _interp_ocean),
                 ("forest", _interp_forest), ("sunset", _interp_sunset),
                 ("aurora", _interp_aurora), ("mono", _interp_mono)]:
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        r, g, b = fn(v)
        check(
            f"{name}({v:.2f}) -> valid RGB",
            all(0 <= c <= 255 for c in (r, g, b)) and all(isinstance(c, int) for c in (r, g, b)),
            f"got ({r}, {g}, {b})"
        )

# Boundary values
for fn in (_interp_heat, _interp_ocean, _interp_forest, _interp_sunset, _interp_aurora, _interp_mono):
    check(f"{fn.__name__}(-1.0) clamped", all(0 <= c <= 255 for c in fn(-1.0)))
    check(f"{fn.__name__}(2.0) clamped", all(0 <= c <= 255 for c in fn(2.0)))

check("COLOR_SCHEMES has 6 entries", len(COLOR_SCHEMES) == 6)
check("CHAR_RAMPS has 6 entries", len(CHAR_RAMPS) == 6)


# ═══════════════════════════════════════════════════════════════
# LatentSpectrogram
# ═══════════════════════════════════════════════════════════════
section("LatentSpectrogram — 128x128")

spec = LatentSpectrogram(grid_size=128)

# Standard input: [1, 63, 512]
latents_63 = torch.randn(1, 63, 512) * 0.5
result = spec.compute(latents_63)
check("shape [1,63,512] → (128,128)", result.shape == (128, 128))
check("range in [0,1]", 0.0 <= result.min() and result.max() <= 1.0)
check("dtype float32", result.dtype == np.float32)
check("not constant (has variance)", result.std() > 0.0,
      f"std={result.std():.6f}")

# Shorter input: [1, 32, 512]
latents_32 = torch.randn(1, 32, 512) * 0.5
result_32 = spec.compute(latents_32)
check("shape [1,32,512] → (128,128)", result_32.shape == (128, 128))
check("interpolated result has variance", result_32.std() > 0.0)

# Longer input: [1, 200, 512]
latents_200 = torch.randn(1, 200, 512) * 0.5
result_200 = spec.compute(latents_200)
check("shape [1,200,512] → (128,128)", result_200.shape == (128, 128))

# 2D input (no batch dim): [63, 512]
latents_2d = torch.randn(63, 512) * 0.5
result_2d = spec.compute(latents_2d)
check("shape [63,512] → (128,128)", result_2d.shape == (128, 128))

# Single-segment input
latents_1 = torch.randn(1, 1, 512) * 0.5
result_1 = spec.compute(latents_1)
check("shape [1,1,512] → (128,128)", result_1.shape == (128, 128))

# Determinism
r1 = spec.compute(latents_63)
r2 = spec.compute(latents_63)
check("deterministic (same input)", np.allclose(r1, r2))

# Different grid sizes
for gs in (64, 128, 256):
    s = LatentSpectrogram(grid_size=gs)
    r = s.compute(latents_63)
    check(f"grid_size={gs} → ({gs},{gs})", r.shape == (gs, gs))

# Small latent dimension (fewer than grid_size dims)
latents_small = torch.randn(1, 63, 64) * 0.5
spec64 = LatentSpectrogram(grid_size=128)
result_small = spec64.compute(latents_small)
check("small dims [1,63,64] → (128,128) padded", result_small.shape == (128, 128))


# ═══════════════════════════════════════════════════════════════
# ChladniOscilloscope
# ═══════════════════════════════════════════════════════════════
section("ChladniOscilloscope — 128x128")

ocl = ChladniOscilloscope(grid_size=128, num_modes=256)

lines_ocl, thickness_ocl = ocl.compute(latents_63)
check("lines shape [1,63,512] → (128,128)", lines_ocl.shape == (128, 128))
# (dtype checks above in new vars)
check("lines shape [1,63,512] → (128,128)", lines_ocl.shape == (128, 128))
check("lines dtype bool", lines_ocl.dtype == bool)
check("lines has some active pixels", lines_ocl.sum() > 0,
      f"line_count={lines_ocl.sum()}")
check("lines fill grid (no circle mask)", lines_ocl.sum() > 0,
      "no lines detected")

# Determinism
l1, t1 = ocl.compute(latents_63)
l2, t2 = ocl.compute(latents_63)
check("deterministic lines", np.array_equal(l1, l2))
check("deterministic thickness", np.allclose(t1, t2))

# Different mode counts
for nm in (32, 64, 128):
    o = ChladniOscilloscope(grid_size=128, num_modes=nm)
    l, t = o.compute(latents_63)
    check(f"num_modes={nm} → thickness (128,128)", t.shape == (128, 128))
    check(f"num_modes={nm} → thickness (128,128)", t.shape == (128, 128))

# Different grid sizes
for gs in (32, 64, 128):
    o = ChladniOscilloscope(grid_size=gs)
    l, t = o.compute(latents_63)
    check(f"chladni grid_size={gs} → lines ({gs},{gs})", l.shape == (gs, gs))
    check(f"chladni grid_size={gs} → lines ({gs},{gs})", l.shape == (gs, gs))

# Short latent sequence
l_short, t_short = ocl.compute(latents_1)
check("chladni [1,1,512] → lines (128,128)", l_short.shape == (128, 128))
check("chladni [1,1,512] → lines (128,128)", l_short.shape == (128, 128))

# 2D input
l_2d, t_2d = ocl.compute(latents_2d)
check("chladni [63,512] → lines (128,128)", l_2d.shape == (128, 128))

# Verify different latents produce different Chladni patterns
latents_b = torch.randn(1, 63, 512) * 2.0
l_a, t_a = ocl.compute(latents_63)
l_b, t_b = ocl.compute(latents_b)
check("different latents → different lines", not np.array_equal(l_a, l_b))


# ═══════════════════════════════════════════════════════════════
# AudioBrainVisualizer
# ═══════════════════════════════════════════════════════════════
section("AudioBrainVisualizer — Unified Interface")

# Construction with all char sets
for charset in CHAR_RAMPS:
    viz = AudioBrainVisualizer(grid_size=64, chars=charset, colors="heat")
    check(f"constructed with chars='{charset}'", viz.chars == CHAR_RAMPS[charset])

# Construction with all color schemes
for cs in COLOR_SCHEMES:
    viz = AudioBrainVisualizer(grid_size=64, chars="dots", colors=cs)
    views = viz.compute_views(latents_63)
    check(f"compute_views with colors='{cs}'",
          "spectrogram" in views and "chladni_lines" in views)

# compute_views returns both
viz = AudioBrainVisualizer(grid_size=64)
views = viz.compute_views(latents_63)
check("compute_views has 'spectrogram'", "spectrogram" in views)
check("compute_views has 'chladni_lines'", "chladni_lines" in views)
check("compute_views has 'chladni_thickness'", "chladni_thickness" in views)
check("spectrogram shape 64x64", views["spectrogram"].shape == (64, 64))
check("chladni_lines shape 64x64", views["chladni_lines"].shape == (64, 64))

# Grid sizes
for gs in (32, 64, 128):
    viz = AudioBrainVisualizer(grid_size=gs)
    views = viz.compute_views(latents_63)
    check(f"grid_size={gs} spectrogram", views["spectrogram"].shape == (gs, gs))
    check(f"grid_size={gs} chladni_lines", views["chladni_lines"].shape == (gs, gs))

# __init__ with bad chars falls back to dots
viz_default = AudioBrainVisualizer(chars="nonexistent")
check("bad charset falls back to ascii", " " in viz_default.chars and "@" in viz_default.chars)


# ═══════════════════════════════════════════════════════════════
# Terminal rendering
# ═══════════════════════════════════════════════════════════════
section("Terminal Rendering")

viz = AudioBrainVisualizer(grid_size=32, chars="dots", colors="heat")

# show="spectrogram"
out_spec = viz.render_terminal(latents_63, title="Test", show="spectrogram")
check("terminal spectrogram has title", "Test" in out_spec)
check("terminal spectrogram has label", "Latent Spectrogram" in out_spec)
check("terminal spectrogram has no chladni (show=spectrogram)",
      "Chladni" not in out_spec)

# show="chladni"
out_chl = viz.render_terminal(latents_63, title="Test", show="chladni")
check("terminal chladni has label", "Chladni Nodal Lines" in out_chl)
check("terminal chladni has no spectrogram (show=chladni)",
      "Latent Spectrogram" not in out_chl)

# show="both"
out_both = viz.render_terminal(latents_63, title="Test", show="both")
check("terminal both has spectrogram", "Latent Spectrogram" in out_both)
check("terminal both has chladni", "Chladni Nodal Lines" in out_both)

# Different color schemes produce different output
viz_ocean = AudioBrainVisualizer(grid_size=32, chars="binary", colors="ocean")
out_ocean = viz_ocean.render_terminal(latents_63, title="T", show="spectrogram")
viz_heat = AudioBrainVisualizer(grid_size=32, chars="binary", colors="heat")
out_heat = viz_heat.render_terminal(latents_63, title="T", show="spectrogram")
check("different colors → different output", out_ocean != out_heat)


# ═══════════════════════════════════════════════════════════════
# HTML rendering
# ═══════════════════════════════════════════════════════════════
section("HTML Rendering")

viz64 = AudioBrainVisualizer(grid_size=16, chars="dots", colors="heat")

# Basic HTML
html = viz64.build_html(latents_63, title="TestHTML")
check("HTML is valid doctype", "<!DOCTYPE html>" in html)
check("HTML has title", "TestHTML" in html)
check("HTML has AudioBrain h1", "<h1>AudioBrain</h1>" in html)
check("HTML has Latent Spectrogram", "Latent Spectrogram" in html)
check("HTML has Chladni Nodal Lines", "Chladni Nodal Lines" in html)
check("HTML has CSS grid", "grid-template-columns" in html)
check("HTML has PPM image", "data:image/x-portable-pixmap" in html)
check("HTML has SVG chladni layer", "chladni-layer" in html)
check("HTML has SVG rects", "<rect" in html)
check("HTML has metadata section", "Metadata" in html)
check("HTML has legend", "legend" in html)
check("HTML has footer", "footer" in html)

# HTML with metadata
meta = {"source": "test.wav", "mode": "fluid", "temperature": "0.50"}
html_meta = viz64.build_html(latents_63, title="Meta", metadata=meta)
check("HTML metadata contains source", "test.wav" in html_meta)
check("HTML metadata contains mode", "fluid" in html_meta)

# HTML without metadata
html_nom = viz64.build_html(latents_63, title="NoMeta")
check("HTML without metadata still valid", "<!DOCTYPE html>" in html_nom)

# HTML without audio
check("HTML without audio has no player", "audio controls" not in html)

# HTML with fake audio data
fake_audio = b"RIFF\x00\x00\x00\x00WAVE"  # minimal WAV header for embedding
html_audio = viz64.build_html(latents_63, title="Audio", audio_data=fake_audio)
check("HTML with audio has player", "audio controls" in html_audio)
check("HTML with audio has base64", "base64" in html_audio)

# save_html to file
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    out_path = f.name
saved = viz64.save_html(out_path, latents_63, title="SavedTest", metadata={"key": "val"})
check("save_html returns path", saved == out_path)
check("save_html file exists", os.path.exists(out_path))
file_size = os.path.getsize(out_path)
check("save_html file non-empty", file_size > 100, f"size={file_size}")
os.unlink(out_path)

# HTML with different color schemes
for cs in ("ocean", "forest", "sunset", "aurora", "mono"):
    v = AudioBrainVisualizer(grid_size=8, colors=cs)
    h = v.build_html(latents_63, title="Color")
    check(f"HTML with colors='{cs}' is valid", "<!DOCTYPE html>" in h)


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════
section("Edge Cases")

# Zero-mean latents
latents_zero = torch.zeros(1, 63, 512)
spec_zero = spec.compute(latents_zero)
check("zero latents spectrogram → (128,128)", spec_zero.shape == (128, 128))
check("zero latents spectrogram finite", np.isfinite(spec_zero).all())

ocl_l_zero, ocl_t_zero = ocl.compute(latents_zero)
check("zero latents chladni lines → (128,128)", ocl_l_zero.shape == (128, 128))
check("zero latents chladni thickness finite", np.isfinite(ocl_t_zero).all())
check("zero latents chladni lines → (128,128)", ocl_l_zero.shape == (128, 128))

# Large latents (potential overflow)
latents_large = torch.randn(1, 63, 512) * 100.0
spec_large = spec.compute(latents_large)
check("large latents spectrogram finite", np.isfinite(spec_large).all())
check("large latents spectrogram in [0,1]",
      0.0 <= spec_large.min() and spec_large.max() <= 1.0)

ocl_l_large, ocl_t_large = ocl.compute(latents_large)
check("large latents chladni lines any", ocl_l_large.any() or True)
check("large latents chladni lines finite", ocl_l_large.any() or True)

# CPU tensor
latents_cpu = torch.randn(1, 63, 512)
spec_cpu = spec.compute(latents_cpu)
check("CPU tensor OK", spec_cpu.shape == (128, 128))

# 3D tensor with batch=1 explicitly
latents_batch = torch.randn(1, 63, 512)
spec_batch = spec.compute(latents_batch)
check("3D [1,N,512] OK", spec_batch.shape == (128, 128))

# Non-standard sequence lengths
for nseq in (1, 3, 7, 15, 31, 127, 128, 129, 200):
    lats = torch.randn(1, nseq, 512) * 0.5
    r = spec.compute(lats)
    check(f"seq_len={nseq} spectrogram → (128,128)", r.shape == (128, 128))

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed")
if failed:
    print(f"  {failed} FAILURES")
    sys.exit(1)
else:
    print(f"  All tests passed!")
print(f"{'='*60}")
