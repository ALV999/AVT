"""
Unit tests for ProjectionHead.
"""

import pytest
import torch
from audiobrain.model.projection import ProjectionHead


class TestProjectionHead:
    """Test suite for the projection layer (2048 -> 512)."""

    @pytest.fixture
    def projection(self):
        return ProjectionHead(
            embedding_dim_input=2048,
            embedding_dim_latent=512,
            dropout=0.1,
        )

    def test_output_shape_single(self, projection):
        """Single vector: [batch=1, 2048] -> [1, 512]."""
        x = torch.randn(1, 2048)
        out = projection(x)
        assert out.shape == (1, 512)

    def test_output_shape_batched(self, projection):
        """Batched sequence: [4, 60, 2048] -> [4, 60, 512]."""
        x = torch.randn(4, 60, 2048)
        out = projection(x)
        assert out.shape == (4, 60, 512)

    def test_output_shape_variable_seq(self, projection):
        """Variable sequence length preserved."""
        x = torch.randn(2, 30, 2048)
        out = projection(x)
        assert out.shape == (2, 30, 512)

    def test_dimension_mismatch_raises(self, projection):
        """Wrong input dimension raises AssertionError."""
        x = torch.randn(1, 256)
        with pytest.raises(AssertionError):
            projection(x)

    def test_dropout_disabled_in_eval(self, projection):
        """Output is deterministic in eval mode."""
        projection.eval()
        x = torch.randn(1, 2048)
        out1 = projection(x)
        out2 = projection(x)
        assert torch.allclose(out1, out2)

    def test_output_is_finite(self, projection):
        """Output contains no NaN or Inf."""
        x = torch.randn(3, 10, 2048)
        out = projection(x)
        assert torch.isfinite(out).all()

    def test_weights_are_trainable(self, projection):
        """Layer has trainable parameters."""
        assert sum(p.numel() for p in projection.parameters() if p.requires_grad) > 0

    def test_device_movement(self):
        """Model can be moved between devices if available."""
        proj = ProjectionHead()
        proj = proj.to("cpu")
        x = torch.randn(1, 2048, device="cpu")
        out = proj(x)
        assert out.device.type == "cpu"
