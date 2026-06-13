"""
Plain Python test runner for AudioBrain unit tests.
No external test framework required — uses only torch and Python stdlib.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import nn

from audiobrain.model.projection import ProjectionHead
from audiobrain.model.transformer import SoundscapeTransformer
from audiobrain.model.core import AudioBrainCore
from audiobrain.model.config import BrainConfig


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


def run_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# ProjectionHead
# ============================================================
run_section("ProjectionHead")

proj = ProjectionHead(embedding_dim_input=2048, embedding_dim_latent=512, dropout=0.1)

x = torch.randn(1, 2048)
check("output shape [1, 2048] -> [1, 512]", proj(x).shape == (1, 512))

x = torch.randn(4, 60, 2048)
check("output shape [4, 60, 2048] -> [4, 60, 512]", proj(x).shape == (4, 60, 512))

x = torch.randn(2, 30, 2048)
check("output shape [2, 30, 2048] -> [2, 30, 512]", proj(x).shape == (2, 30, 512))

try:
    proj(torch.randn(1, 256))
    check("dimension mismatch raises", False, "should have raised")
except AssertionError:
    check("dimension mismatch raises", True)

proj.eval()
x = torch.randn(1, 2048)
check("eval mode deterministic", torch.allclose(proj(x), proj(x)))

x = torch.randn(3, 10, 2048)
check("output is finite", torch.isfinite(proj(x)).all())

check("has trainable params", sum(p.numel() for p in proj.parameters() if p.requires_grad) > 0)

proj2 = ProjectionHead().to("cpu")
x = torch.randn(1, 2048, device="cpu")
check("device movement", proj2(x).device.type == "cpu")


# ============================================================
# SoundscapeTransformer
# ============================================================
run_section("SoundscapeTransformer")

trans = SoundscapeTransformer(d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1, batch_first=True)

x = torch.randn(4, 60, 512)
check("output shape [4, 60, 512]", trans(x).shape == (4, 60, 512))

x = torch.randn(1, 30, 512)
check("single seq [1, 30, 512]", trans(x).shape == (1, 30, 512))

mask = torch.tensor([
    [False, False, False, True, True, True, True, True, True, True],
    [False, False, False, False, True, True, True, True, True, True],
    [False, False, False, False, False, True, True, True, True, True],
    [False, False, False, False, False, False, False, False, False, False],
])
x = torch.randn(4, 10, 512)
out = trans(x, src_key_padding_mask=mask)
check("padding mask accepted", out.shape == (4, 10, 512) and torch.isfinite(out).all())

try:
    trans(torch.randn(1, 10, 256))
    check("dim mismatch raises", False, "should have raised")
except AssertionError:
    check("dim mismatch raises", True)

try:
    SoundscapeTransformer(d_model=512, nhead=7)
    check("nhead divisibility", False, "should have raised")
except ValueError:
    check("nhead divisibility", True)

x = torch.randn(2, 20, 512)
check("output is finite", torch.isfinite(trans(x)).all())

trans.eval()
x = torch.randn(1, 10, 512)
with torch.no_grad():
    o1, o2 = trans(x), trans(x)
check("eval mode deterministic", torch.allclose(o1, o2))

x = torch.randn(2, 10, 512, requires_grad=False)
out = trans(x)
out.sum().backward()
has_grad = any(p.grad is not None for p in trans.parameters())
check("gradient flow", has_grad)


# ============================================================
# AudioBrainCore
# ============================================================
run_section("AudioBrainCore")

cfg = BrainConfig(seed=42)
model = AudioBrainCore(config=cfg)

# Forward
x = torch.randn(4, 60, 2048)
check("forward shape [4, 60, 2048] -> [4, 60, 512]", model(x).shape == (4, 60, 512))

x = torch.randn(1, 30, 2048)
check("forward batch=1 [1, 30, 512]", model(x).shape == (1, 30, 512))

model.eval()
x = torch.randn(1, 10, 2048)
with torch.no_grad():
    check("encode alias matches forward", torch.allclose(model(x), model.encode(x)))

try:
    model(torch.randn(1, 10, 512))
    check("forward dim mismatch raises", False)
except AssertionError:
    check("forward dim mismatch raises", True)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
batch = torch.randn(8, 60, 2048)
loss = model.train_step(batch, optimizer, criterion)
check("train_step returns float", isinstance(loss, float))
check("train_step loss finite", torch.isfinite(torch.tensor(loss)))

# Loss decreases (fresh model, lower lr, measure before any training)
cfg_fresh = BrainConfig(seed=99)
model_fresh = AudioBrainCore(config=cfg_fresh)
opt2 = torch.optim.Adam(model_fresh.parameters(), lr=0.001)
batch2 = torch.randn(4, 30, 2048)

# Get loss before any training
model_fresh.eval()
with torch.no_grad():
    projected = model_fresh.projection(batch2)
    output = model_fresh.transformer(projected)
    initial_loss = criterion(output, projected).item()

# Train for several steps
for _ in range(8):
    model_fresh.train_step(batch2, opt2, criterion)

# Get loss after training
model_fresh.eval()
with torch.no_grad():
    projected = model_fresh.projection(batch2)
    output = model_fresh.transformer(projected)
    final_loss = criterion(output, projected).item()

check("loss decreases", final_loss < initial_loss, f"{final_loss:.6f} >= {initial_loss:.6f}")

# Generation
start = torch.randn(1, 1, 512)
gen = model.generate(start, length=30, temperature=0.8)
check("generate shape [1, 30, 512]", gen.shape == (1, 30, 512))

gen1 = model.generate(start.clone(), length=10, temperature=0)
gen2 = model.generate(start.clone(), length=10, temperature=0)
check("temp=0 deterministic", torch.allclose(gen1, gen2))

try:
    model.generate(torch.randn(1, 1, 256), length=10)
    check("generate dim mismatch raises", False)
except AssertionError:
    check("generate dim mismatch raises", True)

# Checkpoint
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    ckpt_path = f.name
model.save_checkpoint(ckpt_path)
orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
fresh = AudioBrainCore(config=cfg)
fresh.load_checkpoint(ckpt_path)
all_match = all(torch.allclose(orig_weights[k], fresh.state_dict()[k]) for k in orig_weights)
os.unlink(ckpt_path)
check("checkpoint roundtrip", all_match)

# Metadata
check("count_parameters > 0", model.count_parameters() > 0)
check("get_config returns d_model=512", model.get_config().d_model == 512)
check("__repr__ contains AudioBrainCore", "AudioBrainCore" in repr(model))

# Output quality
x = torch.randn(2, 10, 2048)
check("forward output finite", torch.isfinite(model(x)).all())
gen_q = model.generate(torch.randn(1, 1, 512), length=20)
check("generate output finite", torch.isfinite(gen_q).all())


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
total = passed + failed
print(f"  Results: {passed}/{total} passed")
if failed:
    print(f"  {failed} FAILURES")
    sys.exit(1)
else:
    print(f"  All tests passed!")
print(f"{'='*60}")
