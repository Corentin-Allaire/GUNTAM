import pytest
import torch

from GUNTAM.Seed.SeedTransformer import SeedTransformer


class TestSeedTransformerInitialization:
    """Test suite for initialization of SeedTransformer."""

    def test_valid_initialization(self):
        nb_layers_t, nb_heads, dim_embedding = 3, 2, 96
        model = SeedTransformer(nb_layers_t=nb_layers_t, nb_heads=nb_heads, dim_embedding=dim_embedding, dropout=0.0)

        assert model.nb_layers_t == nb_layers_t
        assert model.dim_embedding == dim_embedding
        assert isinstance(model.fourier_encoding, torch.nn.Module)
        assert isinstance(model.embedding_projection, torch.nn.Linear)
        assert len(model.transformer.layers) == nb_layers_t
        # matching_attention uses single head
        assert model.matching_attention.num_heads == 1

    def test_frequency_inference(self):
        # dim_embedding drives inferred nfreq = max(1,(dim_embedding -3)//6)
        dim_embedding = 99  # (99-3)//6 = 16
        model = SeedTransformer(dim_embedding=dim_embedding, nb_heads=3)
        expected_nfreq = max(1, (dim_embedding - 3) // 6)
        # output_dim of fourier_encoding = 3 * nfreq * 2 + 3
        assert model.fourier_encoding.num_frequencies == expected_nfreq
        assert model.fourier_encoding.output_dim == 3 * expected_nfreq * 2 + 3


class TestSeedTransformerForward:
    """Test suite for forward pass and core behaviors."""

    def _make_inputs(self, batch_size=2, seq_len=10):
        # hits layout: first 3 coords (x,y,z), then phi, then eta-like feature
        coords = torch.randn(batch_size, seq_len, 3)
        phi = torch.randn(batch_size, seq_len, 1)
        eta = torch.randn(batch_size, seq_len, 1)
        hits = torch.cat([coords, phi, eta], dim=-1)  # (B, S, 5)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # (B,S) unmasked
        return hits, mask

    def test_forward_shapes(self):
        model = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=64, dropout=0.0)
        hits, mask = self._make_inputs(batch_size=3, seq_len=7)
        output, attn_weights = model(hits, None)

        # Output shape (B, S, dim_embedding)
        assert output.shape == (3, 7, 64)
        # Attention weights from matching_attention single head -> (B, 1, S, S)
        assert attn_weights is not None
        assert attn_weights.shape == (3, 7, 7)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN values"

    def test_forward_with_mask(self):
        model = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=32, dropout=0.0)
        hits, mask = self._make_inputs(batch_size=2, seq_len=6)
        # Mask last 2 query positions (True = masked) in (B,S) form
        mask[:, -2:] = True
        output, attn_weights = model(hits, mask)

        assert output.shape == (2, 6, 32)
        assert attn_weights.shape == (2, 6, 6)
        # Masked rows in attn_weights should be -inf (matching manual attention behavior)
        # Only check if any inf appears in masked rows
        assert torch.isinf(attn_weights[:, :, -2:]).all(), "Masked query rows not set to -inf"

    def test_reproducibility(self):
        torch.manual_seed(123)
        model1 = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=48, dropout=0.0)
        hits, mask = self._make_inputs(batch_size=1, seq_len=5)
        out1, attn1 = model1(hits, mask)

        torch.manual_seed(123)
        model2 = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=48, dropout=0.0)
        out2, attn2 = model2(hits, mask)

        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(attn1, attn2, atol=1e-6)

    def test_gradient_flow(self):
        model = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=40, dropout=0.0)
        hits, mask = self._make_inputs(batch_size=2, seq_len=4)
        hits.requires_grad_(True)
        output, attn_weights = model(hits, mask)
        loss = output.sum()
        loss.backward()
        assert hits.grad is not None
        assert not torch.isnan(hits.grad).any()
        # All parameters that contribute to the loss should have gradients.
        # matching_attention output isn't used in the loss path, so its params may have no grad.
        for name, p in model.named_parameters():
            if name.startswith("matching_attention."):
                continue
            assert p.grad is not None, f"Parameter missing gradient: {name}"

    def test_dimension_mismatch_raises(self):
        model = SeedTransformer(dim_embedding=32)
        # hits must have last dim 5; provide wrong dim
        bad_hits = torch.randn(2, 6, 4)  # last dim 4 instead of 5
        mask = torch.zeros(2, 6, dtype=torch.bool)
        with pytest.raises(RuntimeError):  # Linear will complain or downstream shape mismatch
            _ = model(bad_hits, mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
