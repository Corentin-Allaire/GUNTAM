import math

import pytest
import torch

from GUNTAM.Transformer.Embeding import FourierPositionalEncoding


class TestFourierPositionalEncodingInitialization:
    """Test suite for initialization of FourierPositionalEncoding."""

    def test_valid_initialization_default(self):
        """Test proper initialization with default high_level_dim."""
        input_dim, num_frequencies, high_level_dim = 3, [6, 6, 6], 3
        fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=high_level_dim)

        assert fpe.input_dim == input_dim
        assert fpe.num_frequencies == num_frequencies
        expected_output_dim = sum(num_frequencies) * 2 + high_level_dim
        assert fpe.output_dim == expected_output_dim
        # Single shared frequency-level schedule: [2^0, 2^1, ..., 2^(max-1)]
        assert fpe.freq_levels.shape == (max(num_frequencies),)
        assert torch.allclose(fpe.freq_levels, 2.0 ** torch.arange(max(num_frequencies)).float())

    def test_high_level_dim_variations(self):
        """Changing high_level_dim should update output_dim accordingly."""
        input_dim, num_frequencies = 3, [4, 4, 4]
        for h in [1, 5, 7]:
            fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=h)
            assert fpe.output_dim == sum(fpe.num_frequencies) * 2 + h

    def test_valid_initialization_with_list(self):
        """Test proper initialization with list of num_frequencies."""
        input_dim, high_level_dim = 3, 3
        num_frequencies_list = [4, 6, 8]
        fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies_list, high_level_dim=high_level_dim)

        assert fpe.input_dim == input_dim
        assert fpe.num_frequencies == num_frequencies_list
        expected_output_dim = sum(num_frequencies_list) * 2 + high_level_dim
        assert fpe.output_dim == expected_output_dim
        # Frequency-level schedule covers the maximum per-dimension count
        assert fpe.freq_levels.shape == (max(num_frequencies_list),)
        assert torch.allclose(fpe.freq_levels, 2.0 ** torch.arange(max(num_frequencies_list)).float())

    def test_initialization_errors(self):
        """Test that invalid constructor arguments raise errors."""
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=0)
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=3, num_frequencies=[0, 6, 4])  # Non-positive value
        with pytest.raises(ValueError):
            FourierPositionalEncoding(num_frequencies=[6, 0, 4])  # Non-positive in list
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=3, num_frequencies=[4, 6])  # Length mismatch
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=3, dim_max=[1.0, 2.0])  # length mismatch
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=2, dim_max=[10.0, -5.0])  # non-positive dim_max


class TestFourierPositionalEncodingForward:
    """Test suite for forward pass of FourierPositionalEncoding."""

    def test_forward_shapes_and_ranges(self):
        """Test output shape and value ranges with a single high_level_dim."""
        batch_size, seq_len = 2, 5
        input_dim, num_frequencies, high_level_dim = 3, [4, 4, 4], 3
        fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=high_level_dim)

        x_sampled = torch.randn(batch_size, seq_len, input_dim)
        x_high_level = torch.randn(batch_size, seq_len, high_level_dim)

        output = fpe(x_sampled, x_high_level)

        expected_feature_dim = sum(fpe.num_frequencies) * 2 + high_level_dim
        assert output.shape == (batch_size, seq_len, expected_feature_dim)
        assert expected_feature_dim == fpe.output_dim

        fourier_part = output[..., : sum(fpe.num_frequencies) * 2]
        assert torch.all(fourier_part <= 1.0 + 1e-6)
        assert torch.all(fourier_part >= -1.0 - 1e-6)

        tail = output[..., -high_level_dim:]
        assert torch.allclose(tail, x_high_level, atol=1e-6)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_frequency_blocked_layout(self):
        """Test the frequency-blocked ordering.

        For each frequency level k = 0 .. max(n_d) - 1, the output block is
        [sin(2pi 2^k x_d) for all d with n_d > k,
         cos(2pi 2^k x_d) for all d with n_d > k]
        (dimensions in order). High-level features are appended at the end.
        """
        batch_size, seq_len = 2, 3
        input_dim, num_frequencies, high_level_dim = 3, [4, 2, 6], 2
        dim_max = [200.0, 200.0, 1000.0]
        shift = [10.0, -5.0, 0.0]
        fpe = FourierPositionalEncoding(
            input_dim=input_dim,
            high_level_dim=high_level_dim,
            num_frequencies=num_frequencies,
            dim_max=dim_max,
            shift=shift,
        )

        x_sampled = torch.randn(batch_size, seq_len, input_dim)
        x_high_level = torch.arange(batch_size * seq_len * high_level_dim, dtype=torch.float32).reshape(
            batch_size, seq_len, high_level_dim
        )
        output = fpe(x_sampled, x_high_level)

        # Reference: per-dimension features in the expected block order
        coord = (x_sampled + torch.tensor(shift, dtype=torch.float32)) / torch.tensor(dim_max, dtype=torch.float32)
        expected_blocks = []
        for k in range(max(num_frequencies)):
            freq = 2.0**k
            active_dims = [d for d, n in enumerate(num_frequencies) if n > k]
            proj = 2 * math.pi * coord[..., active_dims] * freq  # (B, S, n_active)
            expected_blocks.append(torch.sin(proj))
            expected_blocks.append(torch.cos(proj))
        expected_fourier = torch.cat(expected_blocks, dim=-1)

        assert output.shape == (batch_size, seq_len, sum(num_frequencies) * 2 + high_level_dim)
        assert torch.allclose(output[..., : sum(num_frequencies) * 2], expected_fourier, atol=1e-5)
        assert torch.allclose(output[..., -high_level_dim:], x_high_level, atol=1e-6)

    def test_uniform_layout_matches_manual_blocks(self):
        """With uniform num_frequencies, each block has exactly 2 * input_dim features."""
        input_dim, num_freq = 3, 3
        fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=[num_freq] * input_dim)
        x_sampled = torch.zeros(1, 1, input_dim)  # zero coords -> sin=0, cos=1
        out = fpe(x_sampled, torch.empty(1, 1, 0))

        for k in range(num_freq):
            # Each block is [sin of all dims, cos of all dims]; at zero coords sin=0, cos=1
            block_start = k * 2 * input_dim
            sin_block = out[0, 0, block_start : block_start + input_dim]
            cos_block = out[0, 0, block_start + input_dim : block_start + 2 * input_dim]
            assert torch.allclose(sin_block, torch.zeros(input_dim), atol=1e-6)
            assert torch.allclose(cos_block, torch.ones(input_dim), atol=1e-6)

    def test_no_high_level_features(self):
        """Passing no high-level features returns only the Fourier part."""
        input_dim, num_frequencies = 3, [2, 3, 4]
        fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies)
        x_sampled = torch.randn(1, 2, input_dim)
        out = fpe(x_sampled, None)
        assert out.shape == (1, 2, sum(num_frequencies) * 2)

    def test_forward_high_level_dim_variations(self):
        """Changing high_level_dim in forward input should yield correct concatenated size."""
        batch_size, seq_len = 1, 3
        input_dim, num_frequencies = 3, [2, 2, 2]
        for h in [1, 4, 6]:
            fpe = FourierPositionalEncoding(input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=h)
            x_sampled = torch.randn(batch_size, seq_len, input_dim)
            x_high_level = torch.randn(batch_size, seq_len, h)
            out = fpe(x_sampled, x_high_level)
            assert out.shape[-1] == sum(fpe.num_frequencies) * 2 + h
            assert out.shape[-1] == fpe.output_dim
            assert not torch.isnan(out).any()

    def test_forward_reproducibility(self):
        """Test deterministic output given same inputs and high_level_dim."""
        torch.manual_seed(42)
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=[5, 5, 5], high_level_dim=4)
        x_sampled = torch.randn(1, 4, 3)
        x_high_level = torch.randn(1, 4, 4)
        out1 = fpe(x_sampled, x_high_level)

        torch.manual_seed(42)
        out2 = fpe(x_sampled, x_high_level)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradient propagation with custom high_level_dim."""
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=[4, 4, 4], high_level_dim=2)
        x_sampled = torch.randn(2, 3, 3, requires_grad=True)
        x_high_level = torch.randn(2, 3, 2, requires_grad=True)
        out = fpe(x_sampled, x_high_level)
        loss = out.sum()
        loss.backward()
        assert x_sampled.grad is not None
        assert x_high_level.grad is not None
        assert not torch.isnan(x_sampled.grad).any()
        assert not torch.isnan(x_high_level.grad).any()

    def test_forward_seq_len_edge_cases(self):
        """Test seq_len edge case with high_level_dim variation."""
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=[2, 2, 2], high_level_dim=5)
        x_sampled = torch.randn(2, 1, 3)
        x_high_level = torch.randn(2, 1, 5)
        out = fpe(x_sampled, x_high_level)
        expected_dim = sum(fpe.num_frequencies) * 2 + 5
        assert out.shape == (2, 1, expected_dim)
        assert out.shape[-1] == fpe.output_dim
        assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
