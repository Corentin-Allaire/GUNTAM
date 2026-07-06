from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from GUNTAM.Seed.TransformerConfig_JUSTINE import TransformerConfig
from GUNTAM.Transformer.Transformer import MultiHeadAttention
from GUNTAM.Transformer.Transformer import TransformerEncoder
from GUNTAM.Transformer.Transformer import load_state_dict_flex
from GUNTAM.Transformer.Embeding import FourierPositionalEncoding
from GUNTAM.Seed.shuffle_features import shuffle_features, shuffle_features_per_i


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
        - cfg (TransformerConfig): Full architecture configuration.
        - device_acc (torch.device): Device on which the model's parameters are allocated.

    Args:
        - transformer_config (TransformerConfig): Architecture configuration object.
        - device_acc (torch.device, optional): Device to run the model on. Defaults to cpu.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig = TransformerConfig(),
        device_acc: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ) -> None:
        super(SeedTransformer, self).__init__()

        self.cfg = transformer_config
        self.device_acc = device_acc
        self.dtype = dtype
        self._setup_modules()
        self.to(dtype)

    def _setup_modules(
        self,
    ) -> None:
        """
        Initialize or rebuild all submodules with the provided hyperparameters.
        """

        coord_dim = len(self.cfg.embedding_feature) + len(set(self.cfg.embedding_feature) & set(self.cfg.cosine_processing))
        high_level_dim = len(self.cfg.high_level_features) + len(
            set(self.cfg.high_level_features) & set(self.cfg.cosine_processing)
        )
        self.fourier_encoding = FourierPositionalEncoding(
            input_dim=coord_dim,
            num_frequencies=self.cfg.fourier_num_frequencies,  # type: ignore[arg-type]
            high_level_dim=high_level_dim,
            dim_max=self.cfg.dim_max,
            shift=self.cfg.shift,
            device_acc=self.device_acc,
        )

        # Set input dimension for projection
        # fourier_encoding.output_dim already accounts for variable frequencies
        embedding_input_dim = self.fourier_encoding.output_dim
        self.embedding_projection = nn.Linear(embedding_input_dim, self.cfg.dim_embedding, device=self.device_acc)

        # Transformer model
        self.transformer = TransformerEncoder(
            n_layers=self.cfg.nb_layers_t,
            input_dim=self.cfg.dim_embedding,
            model_dim=self.cfg.feed_forward_ratio * self.cfg.dim_embedding,
            num_heads=self.cfg.nb_heads,  # Number of attention heads can be adjusted
            dropout=self.cfg.dropout,  # Dropout rate can be adjusted
            device=self.device_acc,
        )

        self.matching_attention = MultiHeadAttention(
            input_dim=self.cfg.dim_embedding,
            model_dim=self.cfg.dim_embedding,
            num_heads=1,
            dropout=self.cfg.dropout,
            device=self.device_acc,
            use_pytorch=False,
        )

        if self.cfg.regression:

            self.regression_MLP = nn.Sequential(
                nn.Linear(self.cfg.dim_embedding, self.cfg.dim_embedding * 2, device=self.device_acc),
                nn.ReLU(),
                nn.Linear(self.cfg.dim_embedding * 2, self.cfg.dim_embedding * 2, device=self.device_acc),
                nn.ReLU(),
            )
            self.hits_score_layer = nn.Sequential(nn.Linear(self.cfg.dim_embedding * 2, 1, device=self.device_acc), nn.Sigmoid())

    def encodeSpacePoint(self, hits: Tensor, mask: Tensor, *, shuffle_v: Optional[int] = None, situation: Optional[str] = None) -> Tensor:
        """
        Encode the input hit sequence.
        Args:
            - hits (Tensor): Input source sequence.
            - mask (Tensor): Source mask.
            - shuffle_v: Indice of the feature we shuffle. 
            - situation: name of the situation corresponding to which features we want to shuffle together.

        Returns:
            - encoded (Tensor): Encoded memory.
        """

        if situation is not None and shuffle_v is not None:
            raise ValueError("`situation` or `shuffle_v` are not well defined")

        if any(i in self.cfg.embedding_feature for i in self.cfg.cosine_processing):
            embedding_cosine = [i for i in self.cfg.embedding_feature if i in self.cfg.cosine_processing]
            embedding_no_cosine = [i for i in self.cfg.embedding_feature if i not in self.cfg.cosine_processing]
            coord = torch.cat(
                [
                    hits[..., embedding_no_cosine],
                    torch.cos(hits[..., embedding_cosine]),
                    torch.sin(hits[..., embedding_cosine]),
                ],
                dim=-1,
            )
        else:
            coord = hits[..., self.cfg.embedding_feature]  # Select features for embedding (e.g., x,y,z,r)

        if self.cfg.high_level_features:
            if any(i in self.cfg.high_level_features for i in self.cfg.cosine_processing):
                high_level_cosine = [i for i in self.cfg.high_level_features if i in self.cfg.cosine_processing]
                high_level_no_cosine = [i for i in self.cfg.high_level_features if i not in self.cfg.cosine_processing]
                high_level = torch.cat(
                    [
                        torch.cos(hits[..., high_level_cosine]),
                        torch.sin(hits[..., high_level_cosine]),
                        hits[..., high_level_no_cosine],
                    ],
                    dim=-1,
                )
            else:
                high_level = hits[..., self.cfg.high_level_features]  # Select high-level features (e.g., phi, eta)
        else:
            high_level = None

        # Use Fourier positional encoding
        encoded_hits = self.fourier_encoding(coord, high_level)

        if situation is not None and shuffle_v is None:
            encoded_hits = shuffle_features(encoded_hits, situation)
        if situation is None and shuffle_v is not None:
            encoded_hits = shuffle_features_per_i(encoded_hits, shuffle_v)
        if situation is None and shuffle_v is None:
            encoded_hits = encoded_hits

        # Apply generic projection if needed

        if self.cfg.embedding_mode == "MLP": 

            encoded_hits = self.embedding_projection(encoded_hits)
            

        elif self.cfg.embedding_mode == "padding":

            pad_size = self.cfg.dim_embedding - encoded_hits.shape[-1]
            encoded_hits = F.pad(encoded_hits, (0, pad_size))
        
        else:

            raise ValueError(f"Unknown embedding_mode: {self.cfg.embedding_mode}")


        transformer_output = self.transformer(x=encoded_hits, mask=mask)

        return transformer_output

    def forward(
        self,
        hits: Tensor,
        mask: Tensor,
        *,
        shuffle_v: Optional[int] = None,
        situation: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the transformer network.
        Args:
            - hits (Tensor): Input source sequence.
            - mask_hits (Tensor): Source mask.
            - shuffle_v: Indice of the feature we shuffle. 
            - situation: name of the situation corresponding to which features we want to shuffle together.

        Returns:
            - encoded (Tensor): Encoded memory.
            - attention_weights (Tensor): Attention weights from all layers.
        """

        # Encode the input hit sequence

        if situation is not None and shuffle_v is None:
            transformer_output = self.encodeSpacePoint(hits, mask, situation=situation)
        if situation is None and shuffle_v is not None:
            transformer_output = self.encodeSpacePoint(hits, mask, shuffle_v=shuffle_v)
        if situation is None and shuffle_v is None:
            transformer_output = self.encodeSpacePoint(hits, mask)

        _, attn_weights = self.matching_attention(transformer_output, mask)

        # The number of heads is 1 for matching attention, so we can squeeze that dimension
        attn_weights = attn_weights.squeeze(1)

        if self.cfg.regression:
            embedding = self.regression_MLP(transformer_output)
            hits_score = self.hits_score_layer(embedding)
            return hits_score, attn_weights

        return transformer_output, attn_weights

    def print_model_info(self) -> None:
        """
        Print model information including number of layers and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("SeedTransformer Model Info:")
        print(f"  - Number of Transformer layers: {self.cfg.nb_layers_t}")
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
                # Save full transformer architecture config
                "transformer_config": self.cfg.to_dict(),
                "dtype": str(self.dtype).replace("torch.", ""),
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
                self._rebuild_from_checkpoint_config(checkpoint.get("transformer_config"), device)
                load_state_dict_flex(self, state_dict, desc="resume")
                self.to(device)
                if "dtype" in checkpoint:
                    saved_dtype = getattr(torch, checkpoint["dtype"], None)
                    if saved_dtype is not None:
                        self.dtype = saved_dtype
                        self.to(saved_dtype)
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
        When a field differs between the CLI config and the checkpoint config, the CLI
        value takes precedence if it was explicitly changed from the default; otherwise
        the checkpoint value is used.
        Args:
            - model_cfg (dict | None): Model configuration from checkpoint.
            - device (torch.device): Device to allocate rebuilt modules on.
        Returns:
            - None
        """
        if not model_cfg:
            return

        default_cfg = TransformerConfig().to_dict()
        cli_cfg = self.cfg.to_dict()

        # Start from checkpoint config, then let non-default CLI values win
        new_cfg = TransformerConfig()
        new_cfg.from_dict(cli_cfg)  # start from current (CLI) config
        new_cfg.from_dict(model_cfg)  # overlay with checkpoint

        # For any field where CLI != checkpoint, prefer CLI if CLI != default
        for key, ckpt_val in model_cfg.items():
            cli_val = cli_cfg.get(key)
            default_val = default_cfg.get(key)
            if cli_val != ckpt_val and cli_val != default_val:
                print(
                    f"Warning: '{key}' mismatch — checkpoint={ckpt_val}, CLI={cli_val} (not default={default_val}). "
                    f"Using CLI value."
                )
                setattr(new_cfg, key, cli_val)

        if new_cfg.to_dict() == self.cfg.to_dict():
            return

        print("Rebuilding SeedTransformer modules to match checkpoint configuration...")
        self.cfg = new_cfg
        self.device_acc = device
        self._setup_modules()
