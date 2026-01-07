from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from GUNTAM.Transformer.Transformer import MultiHeadAttention
from GUNTAM.Transformer.Transformer import TransformerEncoder
from GUNTAM.Transformer.Transformer import load_state_dict_flex
from GUNTAM.Transformer.Embeding import FourierPositionalEncoding


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
        self.nb_heads = nb_heads
        self.dropout = dropout
        # Calculate number of frequencies to get close to dim_embedding if not provided
        # Output will be: 3 * nfreq * 2 + 3 (Fourier features + cos(phi) + sin(phi) + eta)
        self.fourier_num_frequencies = (
            max(1, (dim_embedding - 3) // 6) if (num_frequencies is None) else int(num_frequencies)
        )
        self._setup_modules()

    def _setup_modules(
        self,
    ) -> None:
        """
        Initialize or rebuild all submodules with the provided hyperparameters.
        """

        self.fourier_encoding = FourierPositionalEncoding(
            input_dim=3,
            num_frequencies=self.fourier_num_frequencies,
            dim_max=[200.0, 200.0, 1000.0],
            device_acc=self.device_acc,
        )

        # Set input dimension for projection
        embedding_input_dim = 3 * self.fourier_num_frequencies * 2 + 3

        self.embedding_projection = nn.Linear(embedding_input_dim, self.dim_embedding, device=self.device_acc)

        # Transformer model
        self.transformer = TransformerEncoder(
            n_layers=self.nb_layers_t,
            input_dim=self.dim_embedding,
            model_dim=self.dim_embedding,
            num_heads=self.nb_heads,  # Number of attention heads can be adjusted
            dropout=self.dropout,  # Dropout rate can be adjusted
            device=self.device_acc,
        )

        self.matching_attention = MultiHeadAttention(
            input_dim=self.dim_embedding,
            model_dim=self.dim_embedding,
            num_heads=1,
            dropout=self.dropout,
            device=self.device_acc,
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
        high_level = torch.cat(
            [torch.cos(hits[..., 3:4]), torch.sin(hits[..., 3:4]), hits[..., 4:5]],
            dim=-1,
        )
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

        # The number of heads is 1 for matching attention, so we can squeeze that dimension
        attn_weights = attn_weights.squeeze(1)

        return transformer_output, attn_weights

    def print_model_info(self) -> None:
        """
        Print model information including number of layers and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("SeedTransformer Model Info:")
        print(f"  - Number of Transformer layers: {self.nb_layers_t}")
        print(f"  - Total parameters: {total_params}")
        print(f"  - Trainable parameters: {trainable_params}")

    def save(
        self,
        epoch: int,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """
        Save the model state to a file.
        Args:
            - path (str): File path to save the model.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                # Save model architecture parameters from config
                "model_config": {
                    "nb_layers_t": self.nb_layers_t,
                    "dim_embedding": self.dim_embedding,
                    "nb_heads": self.nb_heads,
                    "dropout": self.dropout,
                    "num_frequencies": self.fourier_num_frequencies,
                },
            },
            path,
        )

    def load(
        self,
        path: str,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> int:
        """
        Load the model state from a file.
        Args:
            - path (str): File path to load the model from.
        Returns:
            - start_epoch (int): Epoch to resume training from.
        """
        start_epoch = 0
        try:
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                print("Checkpoint missing 'model_state_dict'; starting from scratch.")
            else:
                # Rebuild architecture to match the checkpoint if freq/embedding/layers differ
                self._rebuild_from_checkpoint_config(checkpoint.get("model_config"), device)
                load_state_dict_flex(self, state_dict, desc="resume")
                if "optimizer_state_dict" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                    print(f"Resumed training from epoch {start_epoch}")
        except FileNotFoundError:
            print(f"Error: No checkpoint found at {path}. Starting training from scratch.")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}); starting from scratch.")
        return start_epoch

    def _rebuild_from_checkpoint_config(self, model_cfg: dict | None, device: torch.device) -> None:
        """
        Recreate architecture modules to match a checkpoint config.
        Allows loading checkpoints with different architecture parameters.
        Args:
            - model_cfg (dict | None): Model configuration from checkpoint.
            - device (torch.device): Device to allocate rebuilt modules on.
        Returns:
            - None
        """
        if not model_cfg:
            return

        # Use checkpoint values, fall back to current ones
        nb_layers_t = int(model_cfg.get("nb_layers_t", self.nb_layers_t))
        dim_embedding = int(model_cfg.get("dim_embedding", self.dim_embedding))
        nb_heads = int(model_cfg.get("nb_heads", self.nb_heads))
        dropout = float(model_cfg.get("dropout", self.dropout))
        num_frequencies = model_cfg.get("num_frequencies", None)

        # If nothing differs, keep current modules
        if (
            nb_layers_t == self.nb_layers_t
            and dim_embedding == self.dim_embedding
            and nb_heads == self.nb_heads
            and dropout == self.dropout
            and num_frequencies == self.fourier_num_frequencies
        ):
            return

        print("Rebuilding SeedTransformer modules to match checkpoint configuration...")

        self.nb_layers_t = nb_layers_t
        self.nb_heads = nb_heads
        self.dim_embedding = dim_embedding
        self.dropout = dropout
        self.fourier_num_frequencies = num_frequencies
        self.device_acc = device

        self._setup_modules()

        self.to(device)
