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


# il faut que je return la loss

class Dataset: #consrtuit un dataloader
    def __init__(self, data):
        self.data = data

    def get_file(self, idx):
        return self.data

data_1event = torch.load("tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)
#loss sans data_transformer = 1170

data_transformer = torch.load("transformer.pt", weights_only=False, map_location=torch.device('cpu')) 
#c'est le modèle entraîné, il y a les poids du transformer dedans, on les prends dans notre modèle
#mais on test toujours data_1event
#loss avec data_transformer (avec les "bons poids") = 437
#objectif = diminuer loss

dataset = Dataset(data_1event)

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
