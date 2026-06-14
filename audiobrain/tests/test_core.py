"""
Unit tests for AudioBrainCore.
"""

import pytest
import torch
from torch import nn
from audiobrain.model.core import AudioBrainCore
from audiobrain.model.config import BrainConfig


class TestAudioBrainCore:
    """Test suite for the main AudioBrainCore model."""

    @pytest.fixture
    def config(self):
        return BrainConfig(seed=42)

    @pytest.fixture
    def model(self, config):
        return AudioBrainCore(config=config)

    # --- Forward pass tests ---

    def test_forward_shape(self, model):
        """Standard forward: [batch, seq, 2048] -> [batch, seq, 512]."""
        x = torch.randn(4, 60, 2048)
        out = model(x)
        assert out.shape == (4, 60, 512)

    def test_forward_batch1(self, model):
        """Batch size 1."""
        x = torch.randn(1, 30, 2048)
        out = model(x)
        assert out.shape == (1, 30, 512)

    def test_encode_alias(self, model):
        """encode() produces same output as forward()."""
        x = torch.randn(1, 10, 2048)
        model.eval()
        with torch.no_grad():
            fwd = model(x)
            enc = model.encode(x)
        assert torch.allclose(fwd, enc)

    def test_dimension_mismatch_forward(self, model):
        """Wrong input dim (not 2048) raises."""
        x = torch.randn(1, 10, 512)
        with pytest.raises(AssertionError):
            model(x)

    # --- Training tests ---

    def test_train_step_returns_float(self, model):
        """train_step returns a finite float."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        batch = torch.randn(8, 60, 2048)
        loss = model.train_step(batch, optimizer, criterion)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_train_step_decreases_loss(self, model):
        """Loss decreases after multiple steps on same batch."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        batch = torch.randn(4, 30, 2048)

        initial_loss = model.train_step(batch, optimizer, criterion)
        for _ in range(4):
            model.train_step(batch, optimizer, criterion)
        final_loss = model.train_step(batch, optimizer, criterion)

        assert final_loss < initial_loss, f"{final_loss} >= {initial_loss}"

    # --- Generation tests ---

    def test_generate_shape(self, model):
        """Correct output shape for autoregressive generation."""
        start_emb = torch.randn(1, 1, 512)
        gen = model.generate(start_emb, length=30, temperature=0.8)
        assert gen.shape == (1, 30, 512)

    def test_generate_no_temp_deterministic(self, model):
        """temperature=0 produces identical outputs."""
        start_emb = torch.randn(1, 1, 512)
        gen1 = model.generate(start_emb.clone(), length=10, temperature=0)
        gen2 = model.generate(start_emb.clone(), length=10, temperature=0)
        assert torch.allclose(gen1, gen2)

    def test_generate_start_dim_mismatch(self, model):
        """Wrong start embedding dimension raises."""
        start_emb = torch.randn(1, 1, 256)  # should be 512
        with pytest.raises(AssertionError):
            model.generate(start_emb, length=10)

    # --- Checkpoint tests ---

    def test_checkpoint_roundtrip(self, model, tmp_path):
        """Save and load preserves weights."""
        ckpt = tmp_path / "test.pt"
        model.save_checkpoint(str(ckpt))

        # Snapshot original weights
        orig = {k: v.clone() for k, v in model.state_dict().items()}

        # Load into fresh model
        fresh = AudioBrainCore(config=model.config)
        fresh.load_checkpoint(str(ckpt))

        for key in orig:
            assert torch.allclose(orig[key], fresh.state_dict()[key]), \
                f"Weight mismatch for {key}"

    # --- Metadata tests ---

    def test_count_parameters(self, model):
        """Parameter count is positive."""
        n = model.count_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_get_config(self, model):
        """get_config returns the config."""
        cfg = model.get_config()
        assert cfg.d_model == 512
        assert cfg.nhead == 8

    def test_repr(self, model):
        """__repr__ is a non-empty string."""
        r = repr(model)
        assert "AudioBrainCore" in r
        assert "d_model=512" in r

    # --- Output quality ---

    def test_forward_output_finite(self, model):
        """Forward output is all finite."""
        x = torch.randn(2, 10, 2048)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_generate_output_finite(self, model):
        """Generated output is all finite."""
        start = torch.randn(1, 1, 512)
        gen = model.generate(start, length=20)
        assert torch.isfinite(gen).all()
