import pytest
import torch

from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.TransformerConfig import TransformerConfig


def make_config(
    nb_layers_t: int = 2,
    nb_heads: int = 2,
    dim_embedding: int = 64,
    dropout: float = 0.0,
    fourier_num_frequencies: list | None = None,
) -> TransformerConfig:
    """Build a minimal TransformerConfig for tests.

    Mirrors the auto-derivation logic from TransformerConfig.apply_args so that
    fourier_num_frequencies is always a valid list (never None) when the model is
    constructed.
    """
    cfg = TransformerConfig()
    cfg.nb_layers_t = nb_layers_t
    cfg.nb_heads = nb_heads
    cfg.dim_embedding = dim_embedding
    cfg.dropout = dropout
    if fourier_num_frequencies is not None:
        cfg.fourier_num_frequencies = fourier_num_frequencies
    else:
        # Replicate apply_args auto-derivation:
        # cosine-processed embedding features expand to cos+sin (+1 per such feature)
        embed_cosine_n = len(set(cfg.embedding_feature) & set(cfg.cosine_processing))
        coord_dim = len(cfg.embedding_feature) + embed_cosine_n
        n_high = len(cfg.high_level_features) + len(set(cfg.high_level_features) & set(cfg.cosine_processing))
        cfg.fourier_num_frequencies = [
            max(1, (dim_embedding - n_high) // (2 * coord_dim))
        ] * coord_dim
    return cfg


class TestSeedTransformerInitialization:
    """Test suite for initialization of SeedTransformer."""

    def test_valid_initialization(self):
        nb_layers_t, nb_heads, dim_embedding = 3, 2, 96
        cfg = make_config(nb_layers_t=nb_layers_t, nb_heads=nb_heads, dim_embedding=dim_embedding)
        model = SeedTransformer(transformer_config=cfg)

        assert model.cfg.nb_layers_t == nb_layers_t
        assert model.cfg.dim_embedding == dim_embedding
        assert isinstance(model.fourier_encoding, torch.nn.Module)
        assert isinstance(model.embedding_projection, torch.nn.Linear)
        assert len(model.transformer.layers) == nb_layers_t
        # matching_attention uses single head
        assert model.matching_attention.num_heads == 1

    def test_frequency_inference(self):
        # Default config: embedding=[0,1,2,3], cosine=[4], high_level=[4,5]
        # embed_cosine = {0,1,2,3}∩{4} = {} → coord_dim = 4
        # high_dim = 2 + |{4,5}∩{4}| = 3
        # n_freq = (dim_embedding - 3) // (4*2) = (dim_embedding - 3) // 8
        dim_embedding = 98  # (98-3)//8 = 11
        cfg = make_config(dim_embedding=dim_embedding, nb_heads=2)
        model = SeedTransformer(transformer_config=cfg)

        embed_cosine_n = len(set(cfg.embedding_feature) & set(cfg.cosine_processing))
        coord_dim = len(cfg.embedding_feature) + embed_cosine_n
        n_high = len(cfg.high_level_features) + len(set(cfg.high_level_features) & set(cfg.cosine_processing))
        expected_nfreq = max(1, (dim_embedding - n_high) // (2 * coord_dim))
        assert model.fourier_encoding.num_frequencies == [expected_nfreq] * coord_dim
        # output_dim = sum(num_frequencies)*2 + actual high_level tensor size
        assert model.fourier_encoding.output_dim == sum(model.fourier_encoding.num_frequencies) * 2 + n_high

    def test_variable_frequencies_per_dimension(self):
        # Test with different frequencies for each dimension (4 embedded features by default)
        num_frequencies_list = [4, 6, 8, 5]
        cfg = make_config(dim_embedding=64, nb_heads=2, fourier_num_frequencies=num_frequencies_list)
        model = SeedTransformer(transformer_config=cfg)

        assert model.cfg.fourier_num_frequencies == num_frequencies_list
        assert model.fourier_encoding.num_frequencies == num_frequencies_list
        # Default embedding=[0,1,2,3], cosine=[4]: embed_cosine={} → coord_dim=4
        # high_level=[4,5], cosine=[4]: high_dim = 2+1 = 3
        # output_dim = sum([4,6,8,5])*2 + 3 = 46 + 3 = 49
        n_high = len(cfg.high_level_features) + len(set(cfg.high_level_features) & set(cfg.cosine_processing))
        expected_output_dim = sum(num_frequencies_list) * 2 + n_high
        assert model.fourier_encoding.output_dim == expected_output_dim

        # Test forward pass works with 6 features: x, y, z, r, phi, eta
        hits = torch.randn(2, 5, 6)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        output, attn = model(hits, mask)
        assert output.shape == (2, 5, 64)
        assert attn.shape == (2, 5, 5)


class TestSeedTransformerForward:
    """Test suite for forward pass and core behaviors."""

    def _make_inputs(self, batch_size=2, seq_len=10):
        # hits layout: x, y, z, r, phi, eta (6 features)
        coords = torch.randn(batch_size, seq_len, 3)  # x, y, z
        r = torch.randn(batch_size, seq_len, 1)  # r
        phi = torch.randn(batch_size, seq_len, 1)  # phi
        eta = torch.randn(batch_size, seq_len, 1)  # eta
        hits = torch.cat([coords, r, phi, eta], dim=-1)  # (B, S, 6)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # (B,S) unmasked
        return hits, mask

    def test_forward_shapes(self):
        cfg = make_config(nb_layers_t=2, nb_heads=2, dim_embedding=64)
        model = SeedTransformer(transformer_config=cfg)
        hits, mask = self._make_inputs(batch_size=3, seq_len=7)
        output, attn_weights = model(hits, None)

        # Output shape (B, S, dim_embedding)
        assert output.shape == (3, 7, 64)
        # Attention weights from matching_attention single head -> (B, S, S)
        assert attn_weights is not None
        assert attn_weights.shape == (3, 7, 7)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN values"

    def test_forward_with_mask(self):
        cfg = make_config(nb_layers_t=2, nb_heads=2, dim_embedding=32)
        model = SeedTransformer(transformer_config=cfg)
        hits, mask = self._make_inputs(batch_size=2, seq_len=6)
        # Mask last 2 query positions (True = masked) in (B,S) form
        mask[:, -2:] = True
        output, attn_weights = model(hits, mask)

        assert output.shape == (2, 6, 32)
        assert attn_weights.shape == (2, 6, 6)
        # Masked rows in attn_weights should be -inf (matching manual attention behavior)
        assert torch.isinf(attn_weights[:, :, -2:]).all(), "Masked query rows not set to -inf"

    def test_reproducibility(self):
        cfg = make_config(nb_layers_t=2, nb_heads=2, dim_embedding=48)
        hits, mask = self._make_inputs(batch_size=1, seq_len=5)

        torch.manual_seed(123)
        model1 = SeedTransformer(transformer_config=cfg)
        out1, attn1 = model1(hits, mask)

        torch.manual_seed(123)
        model2 = SeedTransformer(transformer_config=cfg)
        out2, attn2 = model2(hits, mask)

        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(attn1, attn2, atol=1e-6)

    def test_gradient_flow(self):
        cfg = make_config(nb_layers_t=2, nb_heads=2, dim_embedding=40)
        model = SeedTransformer(transformer_config=cfg)
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
        # Default config expects 6 features (4 embedding + 2 high-level: x,y,z,r,phi,eta)
        cfg = make_config(dim_embedding=32)
        model = SeedTransformer(transformer_config=cfg)
        # Provide wrong last dim (4 instead of 6)
        bad_hits = torch.randn(2, 6, 4)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            _ = model(bad_hits, mask)


class TestSeedTransformerCheckpointing:
    """Test suite for checkpoint loading behavior."""

    def test_load_missing_checkpoint(self, tmp_path, capsys):
        cfg = make_config()
        model = SeedTransformer(transformer_config=cfg)
        missing_path = tmp_path / "non_existent_checkpoint.pt"

        # Ensure the path does not exist so load hits the FileNotFoundError branch
        assert not missing_path.exists()

        start_epoch = model.load(str(missing_path), device=torch.device("cpu"))
        captured = capsys.readouterr()

        assert start_epoch == 0
        assert f"No checkpoint found at {missing_path}" in captured.out

    def test_save_writes_checkpoint(self, tmp_path):
        cfg = make_config(nb_layers_t=1, nb_heads=2, dim_embedding=16)
        model = SeedTransformer(transformer_config=cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "seed_ckpt.pt"

        model.save(epoch=5, path=str(ckpt_path), optimizer=optimizer)

        assert ckpt_path.exists()
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        assert checkpoint["epoch"] == 5
        assert "model_state_dict" in checkpoint and checkpoint["model_state_dict"]
        assert "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]
        # save() stores the full TransformerConfig under "transformer_config"
        saved_cfg = checkpoint.get("transformer_config")
        assert saved_cfg is not None
        assert saved_cfg["nb_layers_t"] == 1
        assert saved_cfg["dim_embedding"] == 16
        assert saved_cfg["nb_heads"] == 2
        assert saved_cfg["dropout"] == 0.0
        assert saved_cfg["fourier_num_frequencies"] == model.cfg.fourier_num_frequencies

    def test_save_and_load_round_trip(self, tmp_path):
        torch.manual_seed(42)
        cfg = make_config(nb_layers_t=2, nb_heads=2, dim_embedding=24)
        model = SeedTransformer(transformer_config=cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "round_trip.pt"

        # Perform a training step to change parameters from initialization
        hits = torch.randn(1, 5, 6)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        out, _ = model(hits, mask)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save current state
        model.save(epoch=3, path=str(ckpt_path), optimizer=optimizer)

        # Load into a fresh model and optimizer (architecture will be rebuilt from checkpoint)
        torch.manual_seed(123)
        loaded_model = SeedTransformer(transformer_config=make_config(nb_layers_t=2, nb_heads=2, dim_embedding=24))
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
        start_epoch = loaded_model.load(str(ckpt_path), device=torch.device("cpu"), optimizer=loaded_optimizer)

        assert start_epoch == 4

        # Check that parameters match exactly after load
        for name, param in model.state_dict().items():
            assert torch.equal(param, loaded_model.state_dict()[name]), f"Mismatch in parameter {name}"

        # Verify optimizer state was restored
        assert loaded_optimizer.state_dict()["state"], "Optimizer state should not be empty after load"

        # Forward pass outputs should match
        torch.manual_seed(7)
        hits = torch.randn(1, 5, 6)
        ref_out, ref_attn = model(hits, None)
        test_out, test_attn = loaded_model(hits, None)

        assert torch.allclose(ref_out, test_out, atol=1e-6)
        assert torch.allclose(ref_attn, test_attn, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
