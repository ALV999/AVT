"""
Unit tests for SoundscapeTransformer.
"""

import pytest
import torch
from audiobrain.model.transformer import SoundscapeTransformer


class TestSoundscapeTransformer:
    """Test suite for the Transformer encoder."""

    @pytest.fixture
    def transformer(self):
        return SoundscapeTransformer(
            d_model=512,
            nhead=8,
            num_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )

    def test_output_shape(self, transformer):
        """Standard input: [batch, seq_len, d_model] -> same shape."""
        x = torch.randn(4, 60, 512)
        out = transformer(x)
        assert out.shape == (4, 60, 512)

    def test_single_sequence(self, transformer):
        """Single sequence: [1, seq_len, 512]."""
        x = torch.randn(1, 30, 512)
        out = transformer(x)
        assert out.shape == (1, 30, 512)

    def test_padding_mask(self, transformer):
        """Padding mask is accepted and doesn't crash."""
        x = torch.randn(4, 10, 512)
        mask = torch.tensor([
            [False, False, False, True, True, True, True, True, True, True],
            [False, False, False, False, True, True, True, True, True, True],
            [False, False, False, False, False, True, True, True, True, True],
            [False, False, False, False, False, False, False, False, False, False],
        ])
        out = transformer(x, src_key_padding_mask=mask)
        assert out.shape == (4, 10, 512)
        assert torch.isfinite(out).all()

    def test_dimension_mismatch_raises(self, transformer):
        """Wrong d_model raises AssertionError."""
        x = torch.randn(1, 10, 256)  # 256 != 512
        with pytest.raises(AssertionError):
            transformer(x)

    def test_nhead_divisibility(self):
        """nhead must divide d_model."""
        with pytest.raises(ValueError):
            SoundscapeTransformer(d_model=512, nhead=7)

    def test_output_is_finite(self, transformer):
        """Output contains no NaN or Inf."""
        x = torch.randn(2, 20, 512)
        out = transformer(x)
        assert torch.isfinite(out).all()

    def test_eval_mode_deterministic(self, transformer):
        """Same input in eval mode gives same output."""
        transformer.eval()
        x = torch.randn(1, 10, 512)
        with torch.no_grad():
            out1 = transformer(x)
            out2 = transformer(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, transformer):
        """Gradients flow through the transformer."""
        x = torch.randn(2, 10, 512, requires_grad=False)
        out = transformer(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in transformer.parameters())
        assert has_grad
