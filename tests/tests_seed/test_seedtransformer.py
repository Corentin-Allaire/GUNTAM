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
        # fourier_encoding.num_frequencies is now a list internally
        assert model.fourier_encoding.num_frequencies == [expected_nfreq, expected_nfreq, expected_nfreq]
        # output_dim of fourier_encoding = sum(num_frequencies) * 2 + 4
        assert model.fourier_encoding.output_dim == sum(model.fourier_encoding.num_frequencies) * 2 + 4

    def test_variable_frequencies_per_dimension(self):
        # Test with different frequencies for each dimension
        num_frequencies_list = [4, 6, 8]
        model = SeedTransformer(num_frequencies=num_frequencies_list, dim_embedding=64, nb_heads=2)
        
        assert model.fourier_num_frequencies == num_frequencies_list
        assert model.fourier_encoding.num_frequencies == num_frequencies_list
        # output_dim = sum([4, 6, 8]) * 2 + 4 = 18 * 2 + 4 = 40
        expected_output_dim = sum(num_frequencies_list) * 2 + 4
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
        # hits must have last dim 6 (x, y, z, r, phi, eta); provide wrong dim
        bad_hits = torch.randn(2, 6, 4)  # last dim 4 instead of 6
        mask = torch.zeros(2, 6, dtype=torch.bool)
        with pytest.raises(RuntimeError):  # Linear will complain or downstream shape mismatch
            _ = model(bad_hits, mask)


class TestSeedTransformerCheckpointing:
    """Test suite for checkpoint loading behavior."""

    def test_load_missing_checkpoint(self, tmp_path, capsys):
        model = SeedTransformer()
        missing_path = tmp_path / "non_existent_checkpoint.pt"

        # Ensure the path does not exist so load hits the FileNotFoundError branch
        assert not missing_path.exists()

        start_epoch = model.load(str(missing_path), device=torch.device("cpu"))
        captured = capsys.readouterr()

        assert start_epoch == 0
        assert f"No checkpoint found at {missing_path}" in captured.out

    def test_save_writes_checkpoint(self, tmp_path):
        model = SeedTransformer(nb_layers_t=1, nb_heads=2, dim_embedding=16, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "seed_ckpt.pt"

        model.save(epoch=5, path=str(ckpt_path), optimizer=optimizer)

        assert ckpt_path.exists()
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        assert checkpoint["epoch"] == 5
        assert "model_state_dict" in checkpoint and checkpoint["model_state_dict"]
        assert "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]
        config = checkpoint.get("model_config")
        assert config is not None
        assert config["nb_layers_t"] == 1
        assert config["dim_embedding"] == 16
        assert config["nb_heads"] == 2
        assert config["dropout"] == 0.0
        assert config["num_frequencies"] == model.fourier_num_frequencies

    def test_save_and_load_round_trip(self, tmp_path):
        torch.manual_seed(42)
        model = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=24, dropout=0.0)
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

        # Load into a fresh model and optimizer
        torch.manual_seed(123)
        loaded_model = SeedTransformer(nb_layers_t=2, nb_heads=2, dim_embedding=24, dropout=0.0)
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
