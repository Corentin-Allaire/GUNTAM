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
            Can be an int (same for all dimensions) or a list of ints
            (one per dimension, length must match ``input_dim``).
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
        num_frequencies: int | list[int] = 6,
        dim_max: list = [200.0, 200.0, 1000.0],
        device_acc: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension must be greater than 0")

        # Normalize num_frequencies to list
        if isinstance(num_frequencies, int):
            if num_frequencies <= 0:
                raise ValueError("Number of frequencies must be greater than 0")
            num_frequencies_list = [num_frequencies] * input_dim
        else:
            num_frequencies_list = list(num_frequencies)
            if len(num_frequencies_list) != input_dim:
                raise ValueError(
                    f"num_frequencies list length ({len(num_frequencies_list)}) must match input_dim ({input_dim})"
                )
            if any(n <= 0 for n in num_frequencies_list):
                raise ValueError("All num_frequencies values must be greater than 0")

        if len(dim_max) != input_dim:
            raise ValueError("dim_max length must match input_dim")
        if any(d <= 0 for d in dim_max):
            raise ValueError("All dim_max values must be greater than 0")

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies_list
        self.output_dim = sum(num_frequencies_list) * 2 + high_level_dim  # Fourier features + high_level
        self.dim_max = torch.tensor(dim_max, device=device_acc)
        self.device_acc = device_acc

        # Create frequency lists for each dimension with powers of 2: [2^0, 2^1, 2^2, ..., 2^(num_frequencies-1)]
        # Store as a list of tensors since each dimension can have different frequencies
        self.freq_tensors = []
        for num_freq in num_frequencies_list:
            frequencies = 2.0 ** torch.arange(num_freq, device=device_acc).float()
            self.freq_tensors.append(frequencies)

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

        # Process each dimension separately since they can have different number of frequencies
        fourier_features_list = []

        for dim_idx in range(self.input_dim):
            # Extract coordinate for this dimension: (batch, seq_len)
            coord_dim = coord[:, :, dim_idx]

            # Get frequency tensor for this dimension
            freq_tensor = self.freq_tensors[dim_idx]  # (num_freq_dim,)

            # Broadcast and compute projection: (batch, seq_len, num_freq_dim)
            proj = 2 * math.pi * coord_dim.unsqueeze(-1) * freq_tensor.unsqueeze(0).unsqueeze(0)

            # Compute sin and cos
            sin_features = torch.sin(proj)  # (batch, seq_len, num_freq_dim)
            cos_features = torch.cos(proj)  # (batch, seq_len, num_freq_dim)

            # Interleave sin and cos: [sin, cos, sin, cos, ...]
            # Stack and flatten: (batch, seq_len, num_freq_dim, 2) -> (batch, seq_len, num_freq_dim * 2)
            features_dim = torch.stack([sin_features, cos_features], dim=-1).flatten(start_dim=-2)

            fourier_features_list.append(features_dim)

        # Concatenate features from all dimensions: (batch, seq_len, sum(num_frequencies) * 2)
        fourier_features = torch.cat(fourier_features_list, dim=-1)

        # Concatenate: [Fourier features, cos(phi), sin(phi), eta]
        output = torch.cat([fourier_features, x_high_level], dim=-1)

        return output
