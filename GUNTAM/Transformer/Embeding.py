import torch
import torch.nn as nn
from torch import Tensor
import math


class FourierPositionalEncoding(nn.Module):
    """Fourier positional encoding module (frequency-blocked layout).

    Applies Fourier feature encoding to coordinates of the form::

        E(x) = [sin(2π 2^0 x_d), cos(2π 2^0 x_d),
                sin(2π 2^1 x_d), cos(2π 2^1 x_d), ...]

    where ``x`` is a vector of length ``input_dim`` and frequencies are
    powers of 2: ``[2^0, 2^1, 2^2, ..., 2^(num_frequencies-1)]``. Each
    input coordinate is first shifted by ``shift`` and normalized by
    ``dim_max`` (``(x + shift) / dim_max``) before the Fourier features
    are computed.

    The output is laid out in blocks, one per frequency level
    ``k = 0 .. max(num_frequencies) - 1``: for each level, the sin of
    all active dimensions is followed by the cos of all active
    dimensions. A level-``k`` block only contains dimension ``d`` when
    ``num_frequencies[d] > k``, so dimensions may have different
    numbers of frequencies. The block order is deterministic, so the
    output width is always ``sum(num_frequencies) * 2 + high_level_dim``.

    The resulting Fourier features are then concatenated with an
    additional high-level feature tensor provided at call time.

    Args:
        input_dim: Dimension of the coordinate vector in ``x_sampled``
        high_level_dim: Dimension of the ``x_high_level`` feature vector
            appended after the Fourier features.
        num_frequencies: Number of frequency components per coordinate.
            Must be a list of ints (one per dimension, length must match
            ``input_dim``); values may differ across dimensions.
        dim_max: Per-dimension maximum values used to normalize
            ``x_sampled`` (length must be ``input_dim``, all > 0).
        shift: Per-dimension offset added to ``x_sampled`` before
            normalization (length must be ``input_dim``).
        device_acc: Device for buffers``input_dim``).
        device_acc: Device for computation ("cpu" or "cuda").

    Forward inputs:
        x_sampled: Tensor of shape ``(batch_size, seq_len, input_dim)``
            containing the coordinates to be encoded.
        x_high_level: Tensor of shape ``(batch_size, seq_len, H)``
            containing additional high-level features to concatenate.

    Forward output:
        Tensor of shape ``(batch_size, seq_len,
        sum(num_frequencies) * 2 + H)``, where the last dimension is the
        concatenation of the Fourier features (in the frequency-blocked
        order described above) and ``x_high_level``.
    """

    freq_levels: Tensor

    def __init__(
        self,
        input_dim: int = 3,
        high_level_dim: int = 3,
        num_frequencies: list[int] = [6, 6, 6],
        dim_max: list = [200.0, 200.0, 1000.0],
        shift: list[float] = [0.0, 0.0, 0.0],
        device_acc: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = sum(num_frequencies) * 2 + high_level_dim  # Fourier features + high_level
        self.register_buffer("dim_max", torch.tensor(dim_max, dtype=torch.float32, device=device_acc), persistent=False)
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32, device=device_acc), persistent=False)
        self.device_acc = device_acc

        # One shared frequency-level tensor with powers of 2: [2^0, 2^1, ..., 2^(max-1)]
        # (all dimensions share the same frequency scale schedule; ragged per-dim counts are handled in forward)
        max_num_freq = max(num_frequencies)
        self.register_buffer(
            "freq_levels",
            2.0 ** torch.arange(max_num_freq, dtype=torch.float32, device=device_acc),
            persistent=False,
        )

        if input_dim <= 0:
            raise ValueError("Input dimension must be greater than 0")
        # Validate num_frequencies list
        if len(num_frequencies) != input_dim:
            raise ValueError(f"num_frequencies list length ({len(num_frequencies)}) must match input_dim ({input_dim})")
        if any(n <= 0 for n in num_frequencies):
            raise ValueError("All num_frequencies values must be greater than 0")
        if len(dim_max) != input_dim:
            raise ValueError("dim_max length must match input_dim")
        if any(d <= 0 for d in dim_max):
            raise ValueError("All dim_max values must be greater than 0")
        if len(shift) != input_dim:
            raise ValueError("shift length must match input_dim")

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

        coord = (x_sampled + self.shift) / self.dim_max  # type: ignore

        # Process by frequency level: for each level k, block = [sin(2π 2^k x_d), cos(2π 2^k x_d)]
        # over all dimensions d with num_frequencies[d] > k (in dimension order)
        fourier_features_list = []

        for k in range(len(self.freq_levels)):
            # Dimensions active at this frequency level
            active_dims = [d for d, n in enumerate(self.num_frequencies) if n > k]
            if not active_dims:
                continue

            # Coordinates of the active dimensions: (batch, seq_len, n_active)
            coord_active = coord[:, :, active_dims]

            # Projection with the single frequency of this level: (batch, seq_len, n_active)
            proj = 2 * math.pi * coord_active * self.freq_levels[k]

            cos_features = torch.cos(proj)  # (batch, seq_len, n_active)
            sin_features = torch.sin(proj)  # (batch, seq_len, n_active)

            # Block for this level: [sin of all active dims, cos of all active dims]
            fourier_features_list.append(torch.cat([sin_features, cos_features], dim=-1))

        # Concatenate blocks: (batch, seq_len, sum(num_frequencies) * 2)
        fourier_features = torch.cat(fourier_features_list, dim=-1)

        if x_high_level is None or x_high_level.size(-1) == 0:
            # If no high-level features, return only Fourier features
            output = fourier_features

        elif fourier_features.size(-1) == 0:
            # If no Fourier features, return only high-level features
            output = x_high_level

        else:
            # Concatenate: [Fourier features, x_high_level]
            output = torch.cat([fourier_features, x_high_level], dim=-1)

        return output
