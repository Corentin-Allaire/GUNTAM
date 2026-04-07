#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#"""""""""""""""""""""""""""""""""" SEED TRANSFORMER """"""""""""""""""""""""""""""""""""""
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from GUNTAM.Seed.TransformerConfig import TransformerConfig
from GUNTAM.Transformer.Transformer import MultiHeadAttention
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Transformer.Transformer import TransformerEncoder
from GUNTAM.IO.DataLoader import DataLoader
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
        dtype: torch.dtype = torch.float32,
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
            num_frequencies=self.cfg.fourier_num_frequencies,
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

    def encodeSpacePoint(self, hits: Tensor, mask: Tensor) -> Tensor:
        """
        Encode the input hit sequence.
        Args:
            - hits (Tensor): Input source sequence.
            - mask (Tensor): Source mask.
        Returns:
            - encoded (Tensor): Encoded memory.
        """

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

        shuffle_hits = torch.randperm(encoded_hits.size(1)) #on shuffle les hits

        shuffle_variable0 = encoded_hits[:,shuffle_hits,0] #on shuffle la première variable

        encoded_hits[:,:,0] = shuffle_variable0 #on remplace dans encoded_hits

        # Apply generic projection if needed
        encoded_hits = self.embedding_projection(encoded_hits)
        
        #print(encoded_hits.shape)
        #transformer_output = self.transformer(x=encoded_hits, mask=mask)

        return encoded_hits.shape

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
        # print("SeedTransformer Model Info:")
        # print(f"  - Number of Transformer layers: {self.cfg.nb_layers_t}")
        # print(f"  - Total parameters: {total_params}")
        # print(f"  - Trainable parameters: {trainable_params}")

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
        Args:
            - model_cfg (dict | None): Model configuration from checkpoint.
            - device (torch.device): Device to allocate rebuilt modules on.
        Returns:
            - None
        """
        if not model_cfg:
            return

        new_cfg = TransformerConfig()
        new_cfg.from_dict(self.cfg.to_dict())  # start from current (CLI) config
        new_cfg.from_dict(model_cfg)  # overlay only fields present in checkpoint

        if new_cfg.to_dict() == self.cfg.to_dict():
            return

        print("Rebuilding SeedTransformer modules to match checkpoint configuration...")
        self.cfg = new_cfg
        self.device_acc = device
        self._setup_modules()


#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#"""""""""""""""""""""""""""" TRAIN """"""""""""""""""""""""""""""""""""""""""""""""
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import random
import sys
import os
import math
import torch
import time
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from GUNTAM.Seed.SeedTransformer import SeedTransformer
import GUNTAM.Seed.SeedLoss as Losses
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Transformer.Utils import ts_print
import GUNTAM.Transformer.Utils as Utils
import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor
from GUNTAM.IO.PrepareTensor import compute_barcode, prepare_tensor


#on veut print la loss des données classiques et celle des données mélangées pour les comparer :
#--> on peut enlever tout ce qui est optimizer


def initialize_loss_dictionary(active_components: list, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Initialize a loss dictionary with zero values for active loss components.

    Args:
        active_components: List of active loss component names.
        device: Torch device for tensor initialization.

    Returns:
        Initialized loss dictionary with zero values.
    """

    # Helper to add a key lazily
    def add_loss_key(key: str):
        if key not in event_losses:
            event_losses[key] = torch.tensor(0.0, device=device)

    # Initialize per-event losses dynamically based on active loss components
    event_losses = {"total": torch.tensor(0.0, device=device)}

    # Attention variants
    if "attention" in active_components:
        add_loss_key("attention")
    if "topk_attention" in active_components:
        add_loss_key("topk_attention")
    if "full_attention" in active_components:
        add_loss_key("full_attention")
    if "attention_next" in active_components:
        add_loss_key("attention_next")
    if "attention_back" in active_components:
        add_loss_key("attention_back")

    # Classification losses
    if "hit_BCE" in active_components:
        add_loss_key("hit_BCE")

    return event_losses


def train_model(
    model: SeedTransformer,
    train_file_indices: list,
    dataset: DataLoader,
    nb_events: int,
    batch_size: int,
    cfg: SeedConfig,
    writer: SummaryWriter,
    start_epoch: int = 0,
) -> SeedTransformer:
    """
    Train the transformer model for seed reconstruction.

    Args:
        model: The transformer model to be trained.
        train_file_indices: List of file indices for training data.
        dataset: The dataset object containing training data.
        nb_events: Number of events per file.
        batch_size: Batch size for training.
        cfg: Configuration object with training parameters.
        writer: TensorBoard writer for logging.
        optimiser: Optimizer for training.
        scheduler: Learning rate scheduler.
        start_epoch: Starting epoch number (default: 0).

    Returns:
        The trained transformer model.
    """
    epoch_nb = cfg.epoch_nb

    # Loop over the number of epoch starting from start_epoch
    ts_print("Starting the training of the transformer model for seed reconstruction")
    ts_print("Train from epoch ", start_epoch, " to ", start_epoch + epoch_nb)

    # Print active loss components
    active_losses = []
    for component, weight in cfg.loss_config.items():
        active_losses.append(f"{component} (weight: {weight})")

    ts_print("Active loss components: " + ", ".join(active_losses))

    # if optimiser and scheduler:
    #     scheduler.step()
    #     print(f"Initial learning rate: {scheduler.get_last_lr()}")

    for epoch in range(start_epoch, start_epoch + epoch_nb):
        ts_print("Epoch: ", epoch)
        entry = 0

        # Track epoch-level losses
        epoch_train_losses = []
        epoch_val_losses = []

        # Deterministic split of files into training/validation sets (no shuffling)
        files = list(train_file_indices)
        n_val_files = int(cfg.val_fraction * len(files)) if hasattr(cfg, "val_fraction") else 0
        val_files_set = set(files[-n_val_files:]) if n_val_files > 0 else set()

        for file_idx in files:
            # Decide status per file to keep train/val files separate
            status = "Validation" if file_idx in val_files_set else "Training"
            if status == "Validation":
                model.eval()
            else:
                model.train()

            model_dtype = model.dtype
            # Load the data
            batch_data = dataset.get_file(file_idx)
            # print(type(dataset))
            # print(dataset.file_paths)
            # print(file_idx)

            hits_tensor = batch_data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = batch_data["particles_tensor"].to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = batch_data["hit_to_particle_tensor"].to(cfg.device_acc)
            padding_mask = batch_data["padding_mask"].to(cfg.device_acc)
            good_pairs = batch_data["good_pairs"].to(cfg.device_acc)

            # Iterate through each event in this batch with a random order
            num_events_in_batch = hits_tensor.shape[0]
            event_indices = list(range(num_events_in_batch))
            random.shuffle(event_indices)

            for event_idx in event_indices:
                num_valid_bins = 0
                # Extract data for this specific event
                batch_hits_tensor = hits_tensor[event_idx]  # [num_bin, max_hit_input, num_hit_features]
                batch_good_pairs = good_pairs[event_idx]  # [num_bin, num_pairs, 3]
                batch_padding_mask = padding_mask[event_idx]  # [num_bin, max_hit_input, 1]

                batch_hit_to_particle_tensor = hit_to_particle_tensor[event_idx]  # [num_bin, max_hit_input, 1]
                batch_particles_tensor = particles_tensor[event_idx]  # [num_particles, num_particle_features]

                event_losses = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                # Process bins in batches of size batch_size
                grad_enabled = status == "Training"
                accumulated_loss = torch.tensor(0.0, device=cfg.device_acc, requires_grad=True)

                # Collect all bins with valid pairs (vectorized)
                valid_bins = torch.where(batch_good_pairs.sum(dim=(1, 2)) > 0)[0].tolist()

                with torch.set_grad_enabled(grad_enabled):

                    # Loop over the event batch
                    for batch_start in range(0, len(valid_bins), batch_size):

                        batch_bin_indices = valid_bins[batch_start : batch_start + batch_size]
                        batched_hits = batch_hits_tensor[batch_bin_indices]  # [batch_size, max_hit_input, num_hit_features]
                        batched_masks = batch_padding_mask[batch_bin_indices]  # [batch_size, max_hit_input]
                        batched_pairs = batch_good_pairs[batch_bin_indices]  # [batch_size, num_pairs, 3]
                        batched_hit_to_particle_indices = batch_hit_to_particle_tensor[batch_bin_indices].squeeze(
                            -1
                        )  # [batch_size, max_hit_input]
                        # Gather particles using the hit-to-particle mapping
                        batched_particles = batch_particles_tensor[
                            batched_hit_to_particle_indices
                        ]  # [batch_size, max_hit_input, num_particle_features]
                        # Skip if no valid bins in this batch
                        if len(batch_bin_indices) == 0:
                            continue

                        batch_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                        # Perform the hit embedding for all bins in one forward pass
                        encoded_space_points, attention_maps = model(
                            batched_hits, batched_masks
                        )  # encoded_space_points: [N, max_hit_input, dim_embedding]

                        # Compute reconstructed parameters if needed
                        if cfg.transformer_config.regression and cfg.has_loss_component("hit_BCE"):

                            hits_score = encoded_space_points  # [N, max_hit_input, 1]
                            if cfg.has_loss_component("hit_BCE"):
                                batch_loss["hit_BCE"] = Losses.hit_classification_loss(
                                    hits_score,
                                    batched_particles,
                                    batched_masks,
                                )

                        # Process each bin's results for pair-based losses
                        for idx_in_batch, bin_idx in enumerate(batch_bin_indices):
                            pairs1, pairs2, target = batched_pairs[idx_in_batch].unbind(
                                dim=1
                            )  # [num_pairs], [num_pairs], [num_pairs]

                            # Skip bins with no valid pairs (all zeros after orphan filtering)
                            if target.sum() == 0:
                                print(f"Skipping bin {bin_idx.item()} in event {entry} due to no valid pairs after filtering.")
                                continue

                            # Extract this bin's attention map and squeeze batch dim -> [seq_len, seq_len]
                            attention_map_bin = attention_maps[idx_in_batch].squeeze(0)

                            # Compute the attention loss
                            if cfg.has_loss_component("attention"):
                                if attention_map_bin is not None:
                                    batch_loss["attention"] += Losses.attention_loss(attention_map_bin, pairs1, pairs2, target)

                            # Compute the full attention loss (treat all non-positive pairs as negatives)
                            if cfg.has_loss_component("full_attention"):
                                if attention_map_bin is not None:
                                    batch_loss["full_attention"] += Losses.full_attention_loss(
                                        attention_map_bin, pairs1, pairs2, target
                                    )

                            # Compute the top-k attention loss
                            if cfg.has_loss_component("topk_attention"):
                                if attention_map_bin is not None:
                                    batch_loss["topk_attention"] += Losses.top_attention_loss(
                                        attention_map_bin, pairs1, pairs2, target
                                    )

                            # Compute the attention next loss (sequential pairs with cross-entropy)
                            if cfg.has_loss_component("attention_next"):
                                if attention_map_bin is not None:
                                    batch_loss["attention_next"] += Losses.attention_next_loss(
                                        attention_map_bin, pairs1, pairs2, target
                                    )
                            # Compute the attention backward loss (sequential pairs with cross-entropy)
                            if cfg.has_loss_component("attention_back"):
                                if attention_map_bin is not None:
                                    batch_loss["attention_back"] += Losses.attention_backward_loss(
                                        attention_map_bin, pairs1, pairs2, target
                                    )

                            num_valid_bins += 1

                        # Consolidate losses: sum into total, log components, and accumulate weighted grads
                        for key, value in batch_loss.items():
                            if key == "total":
                                continue
                            event_losses[key] += value.detach()
                            batch_loss["total"] += value
                            if status == "Training":
                                accumulated_loss = accumulated_loss + cfg.get_loss_weight(key) * value

                        # Check for NaN/Inf loss
                        if torch.isnan(accumulated_loss) or torch.isinf(accumulated_loss):
                            raise ValueError(
                                f"Loss became NaN/Inf during training at epoch {epoch}, event {entry}, "
                                f"file_idx {file_idx}, batch bins {batch_bin_indices}"
                            )

                        event_losses["total"] += batch_loss["total"].detach()

                        # Apply gradients once per batch of bins (for training)
                        if status == "Training":
                            #optimiser.zero_grad()
                            accumulated_loss.backward()
                            #optimiser.step()
                            accumulated_loss = torch.tensor(0.0, device=cfg.device_acc, requires_grad=True)

                # Average the losses across valid bins
                if num_valid_bins > 0:
                    for key in event_losses.keys():
                        event_losses[key] = event_losses[key] / num_valid_bins

                if writer:
                    # Log gradients once per event for training
                    # if status == "Training":
                    #     Utils.log_gradients(model, writer, epoch * nb_events + entry)

                    # write per-event losses
                    for key, value in event_losses.items():
                        writer.add_scalar(
                            f"loss_components/{key}/{status}",
                            value.item(),
                            epoch * nb_events + entry,
                        )

                    # # write learning rate
                    # if optimiser and scheduler:
                    #     writer.add_scalar(
                    #         "learning_rate/{}".format(status),
                    #         optimiser.param_groups[0]["lr"],
                    #         epoch * nb_events + entry,
                    #     )

                    entry += 1

                # Track event loss for epoch averaging
                if status == "Training":
                    epoch_train_losses.append(event_losses["total"].item())
                else:
                    epoch_val_losses.append(event_losses["total"].item())

        # Compute and log epoch-level average losses
        if epoch_train_losses:
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            ts_print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.6f} ({len(epoch_train_losses)} events)")
            if writer:
                writer.add_scalar("loss_epoch/Training", avg_train_loss, epoch)

        if epoch_val_losses:
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            ts_print(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss:.6f} ({len(epoch_val_losses)} events)")
            if writer:
                writer.add_scalar("loss_epoch/Validation", avg_val_loss, epoch)

        # Save backup checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            backup_path = cfg.model_path.replace(".pt", f"_backup_epoch_{epoch + 1}.pt")
            model.save(
                epoch=epoch,
                path=backup_path
            )
            print(f"Saved backup checkpoint to {backup_path}")
        # if optimiser and scheduler:
        #     scheduler.step()

    return model

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#"""""""""""""""""""""""""""""""""""""" RESULTATS"""""""""""""""""""""""""""""""""""""""""
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Dataset: #consrtuit un dataloader
    def __init__(self, data):
        self.data = data

    def get_file(self, idx):
        return self.data


data_1event = torch.load("tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)

data_transformer = torch.load("transformer.pt", weights_only=False, map_location=torch.device('cpu')) 

dataset = Dataset(data_1event)

#dataset = DataLoader(dataset_name="tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", device=torch.device('cpu'))

cfg = SeedConfig()
cfg.parse_args()
cfg.epoch_nb = 1  

model = SeedTransformer(
    transformer_config=cfg.transformer_config,
    device_acc=cfg.device_acc,
    dtype=torch.float32,
)
model.to(cfg.device_acc)
model.load(path="transformer.pt", device='cpu')

#on print la loss avec shuffle pour chacune des variables :
#permutation importances
#test de significativité

#idée 1 : modifier SeedTransformer en faisant une boucle
#idée 2 (mieux) : créer une fonction qui fait la boucle et l'appeler dans SeedTransformer

writer = None

train_model(
    model=model,
    train_file_indices=[0],   
    dataset=dataset,
    nb_events=1,
    batch_size=1,
    cfg=cfg,
    writer=writer
)


#sans shuffle : average loss = 437
# on shuffle la première variable après embedding projection (ligne 154) : average loss = 430 (256 features)
# on shuffle la première variable après fourier encoding (ligne 152) : average loss = 429 (207 features)


