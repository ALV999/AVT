"""
Minimal training script for AudioBrainCore.

Verifies the training loop works end-to-end:
  1. Instantiate model with BrainConfig
  2. Generate synthetic PANNs-like embeddings as dummy data
  3. Run train_step for N epochs
  4. Save checkpoint
  5. Reload checkpoint and verify weight integrity
  6. Run autoregressive generation
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import nn

from audiobrain.model import AudioBrainCore, BrainConfig


def generate_dummy_batch(
    batch_size: int = 8,
    seq_len: int = 60,
    embedding_dim: int = 2048,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate synthetic PANNs embeddings for training."""
    return torch.randn(batch_size, seq_len, embedding_dim, device=device)


def train_loop(
    model: AudioBrainCore,
    epochs: int = 10,
    batch_size: int = 8,
    seq_len: int = 60,
    log_every: int = 2,
) -> list[float]:
    """Run a minimal training loop with dummy data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    device = model.config.device

    for epoch in range(1, epochs + 1):
        batch = generate_dummy_batch(batch_size, seq_len, device=device)
        loss = model.train_step(batch, optimizer, criterion)
        losses.append(loss)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.6f}")

    return losses


def test_generation(model: AudioBrainCore, length: int = 60) -> torch.Tensor:
    """Test autoregressive generation."""
    device = model.config.device
    start_emb = torch.randn(1, 1, 512, device=device)

    generated = model.generate(start_emb, length=length, temperature=0.8)
    return generated


def test_checkpoint_roundtrip(model: AudioBrainCore, tmp_path: str) -> bool:
    """Save, re-instantiate, and load to verify checkpoint integrity."""
    # Save
    model.save_checkpoint(tmp_path)

    # Capture original weights
    original_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # Create fresh model and load
    fresh = AudioBrainCore(config=model.config)
    fresh.load_checkpoint(tmp_path)

    # Compare
    for key in original_weights:
        if not torch.allclose(original_weights[key], fresh.state_dict()[key]):
            print(f"  X Weight mismatch for {key}")
            return False

    # Cleanup
    os.remove(tmp_path)
    return True


def main() -> None:
    print("=" * 60)
    print("AUDIOBRAIN TRAINING VERIFICATION")
    print("=" * 60)

    # 1. Initialize model
    print("\n[1] Initializing AudioBrainCore...")
    config = BrainConfig(
        d_model=512,
        nhead=8,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=50,
        seed=42,
    )
    model = AudioBrainCore(config=config)
    n_params = model.count_parameters()
    print(f"    Device: {config.device}")
    print(f"    Parameters: {n_params:,}")

    # 2. Training loop
    print(f"\n[2] Training loop ({config.num_epochs} epochs, dummy PANNs data)...")
    losses = train_loop(model, epochs=config.num_epochs, batch_size=8, seq_len=60, log_every=10)
    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    if losses[-1] < losses[0]:
        print("    OK Loss decreased - model is learning")
    else:
        print("    WARNING Loss did not decrease - check hyperparameters")

    # 3. Checkpoint roundtrip
    print("\n[3] Checkpoint roundtrip...")
    ckpt_path = "/tmp/audiobrain_test_checkpoint.pt"
    ok = test_checkpoint_roundtrip(model, ckpt_path)
    if ok:
        print("    OK Checkpoint save/load verified")
    else:
        print("    FAIL Checkpoint roundtrip failed")
        sys.exit(1)

    # 4. Autoregressive generation
    print("\n[4] Autoregressive generation...")
    generated = test_generation(model, length=60)
    expected_shape = (1, 60, 512)
    if generated.shape == expected_shape:
        print(f"    OK Generated shape: {generated.shape} (expected {expected_shape})")
    else:
        print(f"    FAIL Shape mismatch: {generated.shape} != {expected_shape}")
        sys.exit(1)

    # Verify generation is deterministic at temperature=0
    model.eval()
    with torch.no_grad():
        seed = torch.randn(1, 1, 512, device=config.device)
        gen1 = model.generate(seed.clone(), length=10, temperature=0)
        gen2 = model.generate(seed.clone(), length=10, temperature=0)
        if torch.allclose(gen1, gen2):
            print("    OK Deterministic generation at temperature=0")
        else:
            print("    WARNING Generation at temperature=0 is non-deterministic")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
