import pytest
import torch

from GUNTAM.Transformer.Embeding import FourierPositionalEncoding


class TestFourierPositionalEncodingInitialization:
    """Test suite for initialization of FourierPositionalEncoding."""

    def test_valid_initialization_default(self):
        """Test proper initialization with default high_level_dim."""
        input_dim, num_frequencies, high_level_dim = 3, 6, 3
        fpe = FourierPositionalEncoding(
            input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=high_level_dim
        )

        assert fpe.input_dim == input_dim
        assert fpe.num_frequencies == num_frequencies
        expected_output_dim = input_dim * num_frequencies * 2 + high_level_dim
        assert fpe.output_dim == expected_output_dim
        assert hasattr(fpe, "freq_matrix")
        assert fpe.freq_matrix.shape == (input_dim, num_frequencies)
        assert torch.allclose(fpe.freq_matrix[0], 2.0 ** torch.arange(num_frequencies).float())

    def test_high_level_dim_variations(self):
        """Changing high_level_dim should update output_dim accordingly."""
        input_dim, num_frequencies = 3, 4
        for h in [1, 5, 7]:
            fpe = FourierPositionalEncoding(
                input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=h
            )
            assert fpe.output_dim == input_dim * num_frequencies * 2 + h

    def test_initialization_errors(self):
        """Test that invalid constructor arguments raise errors."""
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=0)
        with pytest.raises(ValueError):
            FourierPositionalEncoding(num_frequencies=0)
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=3, dim_max=[1.0, 2.0])  # length mismatch
        with pytest.raises(ValueError):
            FourierPositionalEncoding(input_dim=2, dim_max=[10.0, -5.0])  # non-positive dim_max


class TestFourierPositionalEncodingForward:
    """Test suite for forward pass of FourierPositionalEncoding."""

    def test_forward_shapes_and_ranges(self):
        """Test output shape and value ranges with a single high_level_dim."""
        batch_size, seq_len = 2, 5
        input_dim, num_frequencies, high_level_dim = 3, 4, 3
        fpe = FourierPositionalEncoding(
            input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=high_level_dim
        )

        x_sampled = torch.randn(batch_size, seq_len, input_dim)
        x_high_level = torch.randn(batch_size, seq_len, high_level_dim)

        output = fpe(x_sampled, x_high_level)

        expected_feature_dim = input_dim * num_frequencies * 2 + high_level_dim
        assert output.shape == (batch_size, seq_len, expected_feature_dim)
        assert expected_feature_dim == fpe.output_dim

        fourier_part = output[..., : input_dim * num_frequencies * 2]
        assert torch.all(fourier_part <= 1.0 + 1e-6)
        assert torch.all(fourier_part >= -1.0 - 1e-6)

        tail = output[..., -high_level_dim:]
        assert torch.allclose(tail, x_high_level, atol=1e-6)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_high_level_dim_variations(self):
        """Changing high_level_dim in forward input should yield correct concatenated size."""
        batch_size, seq_len = 1, 3
        input_dim, num_frequencies = 3, 2
        for h in [1, 4, 6]:
            fpe = FourierPositionalEncoding(
                input_dim=input_dim, num_frequencies=num_frequencies, high_level_dim=h
            )
            x_sampled = torch.randn(batch_size, seq_len, input_dim)
            x_high_level = torch.randn(batch_size, seq_len, h)
            out = fpe(x_sampled, x_high_level)
            assert out.shape[-1] == input_dim * num_frequencies * 2 + h
            assert out.shape[-1] == fpe.output_dim
            assert not torch.isnan(out).any()

    def test_forward_reproducibility(self):
        """Test deterministic output given same inputs and high_level_dim."""
        torch.manual_seed(42)
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=5, high_level_dim=4)
        x_sampled = torch.randn(1, 4, 3)
        x_high_level = torch.randn(1, 4, 4)
        out1 = fpe(x_sampled, x_high_level)

        torch.manual_seed(42)
        out2 = fpe(x_sampled, x_high_level)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradient propagation with custom high_level_dim."""
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=4, high_level_dim=2)
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
        fpe = FourierPositionalEncoding(input_dim=3, num_frequencies=2, high_level_dim=5)
        x_sampled = torch.randn(2, 1, 3)
        x_high_level = torch.randn(2, 1, 5)
        out = fpe(x_sampled, x_high_level)
        expected_dim = 3 * 2 * 2 + 5
        assert out.shape == (2, 1, expected_dim)
        assert out.shape[-1] == fpe.output_dim
        assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
