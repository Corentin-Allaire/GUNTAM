import torch
import torch.nn as nn
from torch import Tensor
import math


class FourierPositionalEncoding(nn.Module):
    """Fourier positional encoding module.

    Applies Fourier feature encoding to coordinates of the form::

        E(x) = [sin(2π B x), cos(2π B x)]

    where ``x`` is a vector of length ``input_dim`` and ``B`` uses powers
    of 2: ``[2^0, 2^1, 2^2, ..., 2^(num_frequencies-1)]``. Each input
    coordinate is first normalized by ``dim_max`` before the Fourier
    features are computed.

    The resulting Fourier features are then concatenated with an
    additional high-level feature tensor provided at call time.

    Args:
        input_dim: Dimension of the coordinate vector in ``x_sampled``
            (for example 3 for x, y, z).
        num_frequencies: Number of frequency components per coordinate.
        dim_max: Per-dimension maximum values used to normalize
            ``x_sampled`` (length must be ``input_dim``).
        device_acc: Device for computation ("cpu" or "cuda").

    Forward inputs:
        x_sampled: Tensor of shape ``(batch_size, seq_len, input_dim)``
            containing the coordinates to be encoded.
        x_high_level: Tensor of shape ``(batch_size, seq_len, H)``
            containing additional high-level features to concatenate.

    Forward output:
        Tensor of shape ``(batch_size, seq_len, input_dim *
        num_frequencies * 2 + H)``, where the last dimension is the
        concatenation of the Fourier features and ``x_high_level``.
    """

    def __init__(
        self,
        input_dim: int = 3,
        high_level_dim: int = 3,
        num_frequencies: int = 6,
        dim_max: list = [200.0, 200.0, 1000.0],
        device_acc: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension must be greater than 0")
        if num_frequencies <= 0:
            raise ValueError("Number of frequencies must be greater than 0")
        if len(dim_max) != input_dim:
            raise ValueError("dim_max length must match input_dim")
        if any(d <= 0 for d in dim_max):
            raise ValueError("All dim_max values must be greater than 0")

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * num_frequencies * 2 + high_level_dim  # Fourier + cos(phi) + sin(phi) + eta
        self.dim_max = torch.tensor(dim_max, device=device_acc)

        # Create frequency matrix B with powers of 2: [2^0, 2^1, 2^2, ..., 2^(num_frequencies-1)]
        frequencies = 2.0 ** torch.arange(num_frequencies, device=device_acc).float()
        self.freq_matrix = (
            frequencies.unsqueeze(0).expand(input_dim, -1).clone()
        )  # Expand and clone to avoid memory aliasing

    def forward(self, x_sampled: Tensor, x_high_level: Tensor) -> Tensor:
        """Apply Fourier positional encoding.
        Args:
            x_sampled: Input tensor of the features that will be sampled
               Shape: (batch_size, seq_len, input_dim)
            x_high_level: Input tensor of high-level features that will be concatenated
               Shape: (batch_size, seq_len, high_level_dim)
        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
            Output contains: [Fourier(x_sampled normalized), x_high_level]
        """

        if x_sampled.size(-1) != self.input_dim:
            raise ValueError(f"x_sampled last dim {x_sampled.size(-1)} does not match " f"input_dim={self.input_dim}")

        coord = x_sampled / self.dim_max  # Normalize coordinates

        # Reshape for per-coordinate Fourier transform
        # xyz_normalized: (batch, seq_len, 3) -> (batch, seq_len, 3, 1)
        coord = coord.unsqueeze(-1)  # (batch, seq_len, 3, 1)

        # B: (3, num_frequencies) -> broadcast to (batch, seq_len, 3, num_frequencies)
        B_expanded = self.freq_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, num_frequencies)

        # Element-wise multiplication: each coordinate gets its own frequency
        proj = 2 * math.pi * coord * B_expanded  # (batch, seq_len, 3, num_frequencies)

        # Compute sin and cos for each coordinate and frequency
        sin_features = torch.sin(proj)  # (batch, seq_len, 3, num_frequencies)
        cos_features = torch.cos(proj)  # (batch, seq_len, 3, num_frequencies)

        # Interleave sin and cos for each coordinate: [sin_x, cos_x, sin_y, cos_y, ...]
        # Stack: (batch, seq_len, 3, num_frequencies, 2)
        features = torch.stack([sin_features, cos_features], dim=-1)

        # Flatten: (batch, seq_len, 3 * num_frequencies * 2)
        fourier_features = features.flatten(start_dim=-3)

        # Concatenate: [Fourier features, cos(phi), sin(phi), eta]
        output = torch.cat([fourier_features, x_high_level], dim=-1)

        return output
