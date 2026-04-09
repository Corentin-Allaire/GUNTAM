#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#"""""""""""""""""""""""""""" Validation """"""""""""""""""""""""""""""""""""""""""""""""
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


def validate_model(
    model: SeedTransformer,
    file_indices: list,
    nb_events,
    dataset,
    batch_size: int,
    cfg: SeedConfig,
):
    """
    Validation uniquement sur toutes les données.
    Pas d'entraînement, pas de backward, pas de split train/val.
    """

    ts_print("Starting validation on all available data")

    model.eval()
    model_dtype = model.dtype

    global_losses = []
    nb_total_events = 0

    with torch.no_grad():
        for file_idx in file_indices:
            batch_data = dataset.get_file(file_idx)

            hits_tensor = batch_data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = batch_data["particles_tensor"].to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = batch_data["hit_to_particle_tensor"].to(cfg.device_acc)
            padding_mask = batch_data["padding_mask"].to(cfg.device_acc)
            good_pairs = batch_data["good_pairs"].to(cfg.device_acc)

            num_events_in_batch = hits_tensor.shape[0]

            for event_idx in range(num_events_in_batch):
                num_valid_bins = 0

                batch_hits_tensor = hits_tensor[event_idx]
                batch_good_pairs = good_pairs[event_idx]
                batch_padding_mask = padding_mask[event_idx]
                batch_hit_to_particle_tensor = hit_to_particle_tensor[event_idx]
                batch_particles_tensor = particles_tensor[event_idx]

                event_losses = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                valid_bins = torch.where(batch_good_pairs.sum(dim=(1, 2)) > 0)[0].tolist()

                for batch_start in range(0, len(valid_bins), batch_size):
                    batch_bin_indices = valid_bins[batch_start: batch_start + batch_size]

                    if len(batch_bin_indices) == 0:
                        continue

                    batched_hits = batch_hits_tensor[batch_bin_indices]
                    batched_masks = batch_padding_mask[batch_bin_indices]
                    batched_pairs = batch_good_pairs[batch_bin_indices]
                    batched_hit_to_particle_indices = batch_hit_to_particle_tensor[batch_bin_indices].squeeze(-1)
                    batched_particles = batch_particles_tensor[batched_hit_to_particle_indices]

                    batch_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                    encoded_space_points, attention_maps = model(batched_hits, batched_masks)

                    if cfg.transformer_config.regression and cfg.has_loss_component("hit_BCE"):
                        hits_score = encoded_space_points
                        batch_loss["hit_BCE"] = Losses.hit_classification_loss(
                            hits_score,
                            batched_particles,
                            batched_masks,
                        )

                    for idx_in_batch, bin_idx in enumerate(batch_bin_indices):
                        pairs1, pairs2, target = batched_pairs[idx_in_batch].unbind(dim=1)

                        if target.sum() == 0:
                            continue

                        attention_map_bin = attention_maps[idx_in_batch].squeeze(0)

                        if cfg.has_loss_component("attention") and attention_map_bin is not None:
                            batch_loss["attention"] += Losses.attention_loss(
                                attention_map_bin, pairs1, pairs2, target
                            )

                        if cfg.has_loss_component("full_attention") and attention_map_bin is not None:
                            batch_loss["full_attention"] += Losses.full_attention_loss(
                                attention_map_bin, pairs1, pairs2, target
                            )

                        if cfg.has_loss_component("topk_attention") and attention_map_bin is not None:
                            batch_loss["topk_attention"] += Losses.top_attention_loss(
                                attention_map_bin, pairs1, pairs2, target
                            )

                        if cfg.has_loss_component("attention_next") and attention_map_bin is not None:
                            batch_loss["attention_next"] += Losses.attention_next_loss(
                                attention_map_bin, pairs1, pairs2, target
                            )

                        if cfg.has_loss_component("attention_back") and attention_map_bin is not None:
                            batch_loss["attention_back"] += Losses.attention_backward_loss(
                                attention_map_bin, pairs1, pairs2, target
                            )

                        num_valid_bins += 1

                    for key, value in batch_loss.items():
                        if key == "total":
                            continue
                        event_losses[key] += value.detach()
                        batch_loss["total"] += value

                    event_losses["total"] += batch_loss["total"].detach()

                if num_valid_bins > 0:
                    for key in event_losses.keys():
                        event_losses[key] = event_losses[key] / num_valid_bins

                global_losses.append(event_losses["total"].item())
                nb_total_events += 1

                print(f"[Validation] Event {event_idx} - loss totale = {event_losses['total'].item():.6f}")

    avg_loss = sum(global_losses) / len(global_losses) if global_losses else float("nan")

    ts_print(f"Validation finished on {nb_total_events} events")
    ts_print(f"Average validation loss on all data: {avg_loss:.6f}")

    return avg_loss


