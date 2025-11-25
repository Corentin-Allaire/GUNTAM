from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from GUNTAM.Transformer import MultiHeadAttention
from GUNTAM.Transformer import TransformerEncoder
from GUNTAM.Transformer import FourierPositionalEncoding


class SeedTransformer(nn.Module):
    """
    Transformer network for seed finding and track fitting.

    This module encodes a sequence of hits using Fourier positional
    encoding, projects them to a fixed embedding dimension, and applies
    a Transformer encoder followed by a matching attention layer.

    Attributes:
        - transformer (TransformerEncoder): Transformer encoder operating on embedded hits.
        - fourier_encoding (FourierPositionalEncoding): Fourier-based positional encoder for hit coordinates.
        - embedding_projection (nn.Linear): Linear layer projecting encoded features to `dim_embedding`.
        - matching_attention (MultiHeadAttention): Attention module producing matching scores and weights.
        - dim_embedding (int): Dimension of the internal embedding space.
        - device_acc (str): Device on which the model’s parameters are allocated.
        - nb_layers_t (int): Number of Transformer encoder layers.

    Args:
        - nb_layers_t (int, optional): Number of Transformer encoder layers. Defaults to 3.
        - nb_heads (int, optional): Number of attention heads in the Transformer encoder. Defaults to 2.
        - dim_embedding (int, optional): Dimension of the embedding (hit feature dimension after projection).
            Defaults to 96.
        - dropout (float, optional): Dropout rate used in Transformer and attention layers. Defaults to 0.1.
        - num_frequencies (int | None, optional): Number of Fourier frequencies for positional encoding.
            If None, it is chosen such that the encoded dimension is close to `dim_embedding`. Defaults to None.
        - device_acc (str, optional): Device to run the model on (e.g. "cpu" or "cuda"). Defaults to "cpu".
    """

    def __init__(
        self,
        nb_layers_t: int = 3,
        nb_heads: int = 2,
        dim_embedding: int = 96,
        dropout: float = 0.1,
        num_frequencies: int | None = None,
        device_acc: str = "cpu",
    ) -> None:
        super(SeedTransformer, self).__init__()

        self.dim_embedding = dim_embedding
        self.device_acc = device_acc
        self.nb_layers_t = nb_layers_t

        # Calculate number of frequencies to get close to dim_embedding if not provided
        # Output will be: 3 * nfreq * 2 + 3 (Fourier features + cos(phi) + sin(phi) + eta)
        nfreq = max(1, (dim_embedding - 3) // 6) if (num_frequencies is None) else int(num_frequencies)

        self.fourier_encoding = FourierPositionalEncoding(
            input_dim=3,
            num_frequencies=nfreq,
            dim_max=[200.0, 200.0, 1000.0],
            device_acc=device_acc,
        )

        # Set input dimension for projection
        embedding_input_dim = 3 * nfreq * 2 + 3

        self.embedding_projection = nn.Linear(
            embedding_input_dim,
            dim_embedding,
            device=device_acc
        )

        # Transformer model
        self.transformer = TransformerEncoder(
            n_layers=nb_layers_t,
            input_dim=dim_embedding,
            model_dim=dim_embedding,
            num_heads=nb_heads,  # Number of attention heads can be adjusted
            dropout=dropout,  # Dropout rate can be adjusted
            device=device_acc,
        )

        self.matching_attention = MultiHeadAttention(
            input_dim=dim_embedding,
            model_dim=dim_embedding,
            num_heads=1,
            dropout=dropout,
            device=device_acc,
            use_pytorch=False,
        )

    def encodeSpacePoint(self, hits: Tensor, mask: Tensor) -> Tensor:
        """
        Encode the input hit sequence.
        Args:
            - hits (Tensor): Input source sequence.
            - mask (Tensor): Source mask.
        Returns:
            - encoded (Tensor): Encoded memory.
        """

        coord = hits[..., :3]
        high_level = torch.cat([torch.cos(hits[..., 3:4]),
                                torch.sin(hits[..., 3:4]),
                                hits[..., 4:5]], dim=-1)
        # Use Fourier positional encoding
        encoded_hits = self.fourier_encoding(coord, high_level)
        # Apply generic projection if needed
        encoded_hits = self.embedding_projection(encoded_hits)

        transformer_output = self.transformer(x=encoded_hits, mask=mask)

        return transformer_output

    def forward(
        self,
        hits: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the transformer network.
        Args:
            - hits (Tensor): Input source sequence.
            - mask_hits (Tensor): Source mask.
        Returns:
            - encoded (Tensor): Encoded memory.
            - attention_weights (Tensor): Attention weights from all layers.
        """

        # Encode the input hit sequence
        transformer_output = self.encodeSpacePoint(hits, mask)
        _, attn_weights = self.matching_attention(transformer_output, mask)

        return transformer_output, attn_weights
