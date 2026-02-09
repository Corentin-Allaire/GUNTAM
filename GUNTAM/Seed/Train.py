import random
import sys
import os
import math
import torch
import time
import numpy as np
import gc
from typing import List, Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from GUNTAM.Seed.SeedTransformer import SeedTransformer
import GUNTAM.Seed.SeedLoss as Losses
from GUNTAM.Seed.Config import config
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Transformer.Utils import ts_print
import GUNTAM.Transformer.Utils as Utils
import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor


def compute_parameter_loss_norms(dataset) -> Dict[str, float]:
    """
    Compute normalization factors for parameter loss based on the truth distribution.
    Args:
        dataset (DataLoader): The dataset object containing training data.
    Returns:
        dict: Normalization factors for parameters z0, eta, phi, and pt.
    """
    ts_print("Computing normalization factors from truth distribution...")
    valid_particles_list = []

    # Use first training file (in original order, before shuffling) for consistent normalization
    batch_data = dataset.get_file(0)

    tensor_particles = batch_data["tensor_particles"]
    padding_mask_hit = batch_data["padding_mask_hit"]

    # Create mask for valid hits
    valid_mask = ~padding_mask_hit.bool()
    valid_particles = tensor_particles[valid_mask]
    nonzero_pt_mask = valid_particles[:, 3] > 0
    valid_particles = valid_particles[nonzero_pt_mask]

    if len(valid_particles) > 0:
        valid_particles_list.append(valid_particles.cpu())

    # Compute normalization factors
    if valid_particles_list:
        all_valid_particles = torch.cat(valid_particles_list, dim=0)
        stds = torch.std(all_valid_particles, dim=0)
        std_pt = torch.std(1 / all_valid_particles[:, 3], dim=0)
        norm_factors = {
            "z0": float(stds[0]),
            "eta": float(stds[1]),
            "phi": float(stds[2]),
            "pt": float(std_pt),
        }
        ts_print(
            "Normalization factors: "
            f"z0: {norm_factors['z0']:.3f}, "
            f"eta: {norm_factors['eta']:.3f}, "
            f"phi: {norm_factors['phi']:.3f}, "
            f"pt: {norm_factors['pt']:.3f}"
        )
    else:
        ts_print("Warning: No valid particles found for normalization computation!")
        norm_factors = {"z0": 1.0, "eta": 1.0, "phi": 1.0, "pt": 1.0}

    return norm_factors


def initialize_loss_dictionary(active_components: list, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Initialize a loss dictionary with zero values for active loss components.
    Args:
        active_components (list): List of active loss component names.
    Returns:
        dict: Initialized loss dictionary with zero values.
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

    # Reconstruction losses and sub-components
    if "MSE" in active_components or "L1" in active_components:
        add_loss_key("reco")
        add_loss_key("reco_z0")
        add_loss_key("reco_eta")
        add_loss_key("reco_phi")
        add_loss_key("reco_pt")
        if "MSE" in active_components and "L1" in active_components:
            raise ValueError("Cannot have both MSE and L1 reconstruction losses active simultaneously.")

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
    cfg: config,
    writer: SummaryWriter,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int = 0,
) -> SeedTransformer:
    """
    Train the transformer model for seed reconstruction.
    Args:
        model (SeedTransformer): The transformer model to be trained.
        train_file_indices (list): List of file indices for training data.
        dataset (Seeding_Dataset): The dataset object containing training data.
        nb_events (int): Number of events per file.
        cfg (config): Configuration object with training parameters.
        writer (SummaryWriter): TensorBoard writer for logging.
        optimiser (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        start_epoch (int, optional): Starting epoch number. Defaults to 0.
    Returns:
        SeedTransformer: The trained transformer model.
    """
    epoch_nb = cfg.epoch_nb

    # Extract normalization factors only if reconstruction loss is used.
    norm_factors = {"z0": 1.0, "eta": 1.0, "phi": 1.0, "pt": 1.0}
    if cfg.has_loss_component("MSE") or cfg.has_loss_component("L1"):
        norm_factors = compute_parameter_loss_norms(dataset)

    # Loop over the number of epoch starting from start_epoch
    ts_print("Starting the training of the transformer model for seed reconstruction")
    ts_print("Train from epoch ", start_epoch, " to ", start_epoch + epoch_nb)

    # Print active loss components
    active_losses = []
    for component, weight in cfg.loss_config.items():
        active_losses.append(f"{component} (weight: {weight})")

    ts_print("Active loss components: " + ", ".join(active_losses))

    if optimiser and scheduler:
        scheduler.step()

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

            # Load the data
            batch_data = dataset.get_file(file_idx)
            tensor_hits = batch_data["tensor_hits"].to(cfg.device_acc)
            tensor_particles = batch_data["tensor_particles"].to(cfg.device_acc)
            padding_mask_hit = batch_data["padding_mask_hit"].to(cfg.device_acc)
            all_pairs = batch_data["all_pairs"]

            # Iterate through each event in this batch with a random order
            num_events_in_batch = tensor_hits.shape[0]
            event_indices = list(range(num_events_in_batch))
            random.shuffle(event_indices)

            for event_idx in event_indices:
                num_valid_bins = 0
                # Extract data for this specific event
                batch_tensor_hits = tensor_hits[event_idx]  # Shape: [bins, hits, features]
                batch_tensor_particles = tensor_particles[event_idx]
                batch_padding_hit = padding_mask_hit[event_idx]
                event_pairs = all_pairs[event_idx]  # Pairs for this specific event

                event_losses = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                # Process bins in batches of size batch_size
                grad_enabled = status == "Training"
                accumulated_loss = torch.tensor(0.0, device=cfg.device_acc, requires_grad=True)

                # Collect all bins with valid pairs
                valid_bins = [
                    (bin_idx, bin_data)
                    for bin_idx, bin_data in event_pairs.items()
                    if len(bin_data[0]) > 0 and len(bin_data[1]) > 0
                ]

                with torch.set_grad_enabled(grad_enabled):

                    # Loop over the event batch
                    for batch_start in range(0, len(valid_bins), batch_size):

                        batch_end = min(batch_start + batch_size, len(valid_bins))
                        bins_in_batch = valid_bins[batch_start:batch_end]

                        # Collect bin indices and pair data
                        batch_bin_indices = []
                        batch_pairs_list = []  # List of (pairs1, pairs2, target, bin_mask)

                        # TODO: Maybe the we we store bin can be optimised
                        for bin_idx, bin_data in bins_in_batch:
                            pairs1, pairs2, target = bin_data
                            if len(pairs1) == 0 or len(pairs2) == 0:
                                ts_print(f"Skipping bin {bin_idx} for event {entry} due to empty pairs")
                                continue

                            # Ensure pairs are on the correct device
                            pairs1 = pairs1.to(cfg.device_acc)
                            pairs2 = pairs2.to(cfg.device_acc)
                            target = target.to(cfg.device_acc)

                            # Get valid hits in this bin
                            bin_mask = ~batch_padding_hit[bin_idx].bool()
                            if not torch.any(bin_mask):
                                ts_print(f"Skipping bin {bin_idx} for event {entry} due to no valid hits")
                                continue

                            # Store only indices and pairs - use tensor views for actual data
                            batch_bin_indices.append(bin_idx)
                            batch_pairs_list.append((pairs1, pairs2, target, bin_mask))

                        # Skip if no valid bins in this batch
                        if len(batch_bin_indices) == 0:
                            continue

                        # Create tensor views for the bins in this batch (no copying!)
                        batched_hits = batch_tensor_hits[batch_bin_indices]  # [N, max_hit_input, 5]
                        batched_masks = batch_padding_hit[batch_bin_indices]  # [N, max_hit_input]
                        batched_particles = batch_tensor_particles[batch_bin_indices, :, 0:4]  # [N, max_hit_input, 4]

                        batch_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                        # Perform the hit embedding for all bins in one forward pass
                        encoded_space_points, attention_maps = model(
                            batched_hits, batched_masks
                        )  # encoded_space_points: [N, max_hit_input, dim_embedding]

                        # Compute reconstructed parameters if needed
                        if (
                            cfg.has_loss_component("MSE")
                            or cfg.has_loss_component("L1")
                            or cfg.has_loss_component("hit_BCE")
                        ):
                            ts_print(
                                "Computing reconstructed parameters for event ",
                                entry,
                                " bins ",
                                batch_bin_indices,
                            )
                            print("Currently not implemented.....")
                            reconstructed_params_batch = None
                            continue

                            if cfg.has_loss_component("MSE") or cfg.has_loss_component("L1"):
                                loss_type = "MSE" if cfg.has_loss_component("MSE") else "L1"
                                batch_reco_loss_dict = Losses.reconstruction_loss(
                                    reconstructed_params_batch,
                                    batched_particles,
                                    batched_masks,
                                    loss_type=loss_type,
                                )
                                # Normalize individual components and store in batch dict
                                batch_loss["reco_z0"] = batch_reco_loss_dict["z"] / (
                                    norm_factors["z0"] * norm_factors["z0"]
                                )
                                batch_loss["reco_eta"] = batch_reco_loss_dict["eta"] / (
                                    norm_factors["eta"] * norm_factors["eta"]
                                )
                                batch_loss["reco_phi"] = batch_reco_loss_dict["phi"]
                                batch_loss["reco_pt"] = batch_reco_loss_dict["pt"] / (
                                    norm_factors["pt"] * norm_factors["pt"]
                                )
                                # Average across components
                                batch_loss["reco"] = (
                                    batch_loss["reco_z0"]
                                    + batch_loss["reco_eta"]
                                    + batch_loss["reco_phi"]
                                    + batch_loss["reco_pt"]
                                ) / 4

                            # Note: L1 handled by Losses.reconstruction_loss via loss_type above

                            if cfg.has_loss_component("hit_BCE"):
                                batch_loss["hit_BCE"] = Losses.hit_classification_loss(
                                    reconstructed_params_batch[:, :, 4],
                                    batched_particles,
                                    batched_masks,
                                )

                        # Process each bin's results for pair-based losses
                        for idx_in_batch, bin_idx in enumerate(batch_bin_indices):
                            pairs1, pairs2, target, bin_mask = batch_pairs_list[idx_in_batch]

                            # Extract this bin's attention map and squeeze batch dim -> [seq_len, seq_len]
                            attention_map_bin = attention_maps[idx_in_batch].squeeze(0)

                            # Compute the attention loss
                            if cfg.has_loss_component("attention"):
                                if attention_map_bin is not None:
                                    batch_loss["attention"] += Losses.attention_loss(
                                        attention_map_bin, pairs1, pairs2, target
                                    )

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

                            num_valid_bins += 1

                        # Consolidate losses: sum into total, log components, and accumulate weighted grads
                        for key, value in batch_loss.items():
                            if key == "total":
                                continue
                            event_losses[key] += value.detach()
                            batch_loss["total"] += value
                            if status == "Training":
                                accumulated_loss = accumulated_loss + cfg.get_loss_weight(key) * value

                        event_losses["total"] += batch_loss["total"].detach()

                        # Apply gradients once per batch of bins (for training)
                        if status == "Training":
                            optimiser.zero_grad()
                            accumulated_loss.backward()
                            optimiser.step()
                            accumulated_loss = torch.tensor(0.0, device=cfg.device_acc, requires_grad=True)

                # Average the losses across valid bins
                if num_valid_bins > 0:
                    for key in event_losses.keys():
                        event_losses[key] = event_losses[key] / num_valid_bins

                if writer:
                    # Log gradients once per event for training
                    if status == "Training":
                        Utils.log_gradients(model, writer, epoch * nb_events + entry)

                    # write per-event losses
                    for key, value in event_losses.items():
                        writer.add_scalar(
                            f"loss_components/{key}/{status}",
                            value.item(),
                            epoch * nb_events + entry,
                        )

                    # write learning rate
                    if optimiser and scheduler:
                        writer.add_scalar(
                            "learning_rate/{}".format(status),
                            optimiser.param_groups[0]["lr"],
                            epoch * nb_events + entry,
                        )

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": (optimiser.state_dict() if optimiser else None),
                    "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
                    # Save model architecture parameters from config
                    "model_config": {
                        "nb_layers_t": cfg.nb_layers_t,
                        "dim_embedding": cfg.dim_embedding,
                        "nb_heads": cfg.nb_heads,
                        "dropout": cfg.dropout,
                        "num_frequencies": cfg.fourier_num_frequencies,
                    },
                },
                backup_path,
            )
            print(f"Saved backup checkpoint to {backup_path}")
        if optimiser and scheduler:
            scheduler.step()

    return model


def run_model(
    model: SeedTransformer,
    file_indices: list,
    dataset: DataLoader,
    cfg: config,
) -> tuple:
    """Run inference over files and return per-bin artifacts.

    Performs a forward pass (embedding + attention + dummy regression) and then executes
    a lightweight seed reconstruction on the CPU for every valid bin of each processed
    event. Outputs are kept as Python/NumPy containers to reduce peak memory usage.

    Returns (nested lists; indexing is [event_idx][bin_idx]):
        - seeds: Reconstruction seeds (list of seed tuples per bin). Each seed is a
          tuple (hit_indices: np.ndarray, avg_params: np.ndarray). Empty bins yield [].
        - reconstructed_parameters: per-bin parameters (np.ndarray[num_valid_hits, 5]) or None.
        - attention_maps: per-bin np.ndarray[num_valid_hits, num_valid_hits] used for seeding, or None.
    """

    # Initialize speed monitoring
    # Use high-resolution clock; sync once to avoid overlapping prior kernels
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        start_time = time.perf_counter()
    processing_times = []
    transformer_times = []
    regression_times = []
    seed_reconstruction_times = []

    # Loop over the number of epoch starting from start_epoch
    print("Starting the running of the transformer model for seed reconstruction")
    seeds = []  # refined seeds
    model_outputs = []
    reconstructed_parameters = []
    attention_maps = []

    event_counter = 0

    for file_idx in file_indices:
        batch_data = dataset.get_file(file_idx)

        tensor_hits = batch_data["tensor_hits"].to(cfg.device_acc)
        tensor_particles = batch_data["tensor_particles"].to(cfg.device_acc)
        tensor_ID = batch_data["tensor_ID"].to(cfg.device_acc)
        padding_mask_hit = batch_data["padding_mask_hit"].to(cfg.device_acc)
        all_pairs = batch_data["all_pairs"]

        # Process each event in the batch
        num_events_in_batch = tensor_hits.shape[0]
        for event_idx in range(num_events_in_batch):
            # for event_idx in range(min(num_events_in_batch, 20)):
            if cfg.timing_enabled:
                Utils.sync_device(cfg.device_acc)
                event_start_time = time.perf_counter()

            # Extract data for this specific event
            batch_tensor_hits = tensor_hits[event_idx]
            batch_padding_hit = padding_mask_hit[event_idx]

            event_seeds: List[Any] = []
            event_encoded_points: List[Optional[np.ndarray]] = []
            event_parameters: List[Optional[np.ndarray]] = []
            event_attention_maps: List[Optional[np.ndarray]] = []

            with torch.no_grad():
                # Timing: Transformer inference (encoding + attention)
                if cfg.timing_enabled:
                    Utils.sync_device(cfg.device_acc)
                    t0 = time.perf_counter()

                # Obtain encoded embeddings and attention weights from the model
                encoded_space_point, attention_weights = model(batch_tensor_hits, batch_padding_hit)

                if cfg.timing_enabled:
                    Utils.sync_device(cfg.device_acc)
                    transformer_duration = time.perf_counter() - t0

                # Timing: Parameter regression (+ optional pairwise scoring)
                if cfg.timing_enabled:
                    Utils.sync_device(cfg.device_acc)
                    r0 = time.perf_counter()

                ts_print("Reconstructed parameters computation not implemented....")
                # Initialize parameters with zeros having shape [bins, hits, 5]
                parameters = torch.zeros(
                    (batch_tensor_hits.shape[0], batch_tensor_hits.shape[1], 5),
                    device=cfg.device_acc,
                    dtype=batch_tensor_hits.dtype,
                )

                if cfg.timing_enabled:
                    Utils.sync_device(cfg.device_acc)
                    regression_duration = time.perf_counter() - r0

            # Timing: Seed reconstruction across all bins
            if cfg.timing_enabled:
                Utils.sync_device(cfg.device_acc)
                seed_reconstruction_start = time.perf_counter()

            for bin_idx in range(batch_tensor_hits.shape[0]):

                # Get valid hits in this bin
                bin_mask = ~batch_padding_hit[bin_idx].bool()
                if not torch.any(bin_mask):
                    # Maintain positional consistency with explicit None placeholders
                    event_encoded_points.append(None)
                    event_parameters.append(None)
                    event_attention_maps.append(None)
                    event_seeds.append([])
                    continue

                # Get encoded space points for valid hits
                valid_sp = encoded_space_point[bin_idx].cpu().detach()
                valid_parameters = parameters[bin_idx].cpu().detach()
                valid_attention_weights = attention_weights[bin_idx].squeeze(0).detach().cpu()
                bin_mask_cpu = bin_mask.detach().cpu()

                # Store the encoded space points
                event_encoded_points.append(valid_sp.numpy())
                event_parameters.append(valid_parameters.numpy())

                # Extract attention weights for this bin from single layer
                if valid_attention_weights is not None:

                    # Apply masking for valid hits only
                    neighbor_matrix_masked = valid_attention_weights[bin_mask_cpu, :][:, bin_mask_cpu]

                    # Choose clustering method based on loss configuration (mutually exclusive)
                    if cfg.has_loss_component("attention_next"):
                        # Use attention-next based reconstruction (row-normalized scores)
                        neighbor_matrix_masked.fill_diagonal_(float("-inf"))
                        neighbor_matrix_masked = torch.softmax(neighbor_matrix_masked, dim=-1)
                        bin_seeds = Reconstruction.chained_seed_reconstruction(
                            neighbor_matrix_masked,
                            valid_parameters,
                            score_threshold=0.2,
                            max_chain_length=5,
                        )
                    else:
                        # Fall back to attention-based reconstruction (thresholded k-NN)
                        neighbor_matrix_masked = torch.sigmoid(neighbor_matrix_masked)
                        bin_seeds = Reconstruction.topk_seed_reconstruction(
                            neighbor_matrix_masked,
                            valid_parameters,
                            threshold=0.8,
                            max_selection=5,
                        )
                    event_seeds.append(bin_seeds)

                    # Apply softmax row-wise to attention weights for monitoring
                    attention_softmax = torch.softmax(valid_attention_weights, dim=-1)
                    event_attention_maps.append(attention_softmax.cpu().detach().numpy())
                else:
                    # No attention weights or pairwise scores available - append placeholder to keep alignment
                    event_seeds.append([])
                    event_attention_maps.append(None)

            # End seed reconstruction timing
            if cfg.timing_enabled:
                Utils.sync_device(cfg.device_acc)
                seed_reconstruction_end = time.perf_counter()
                seed_reconstruction_duration = seed_reconstruction_end - seed_reconstruction_start

            seeds.append(event_seeds)
            model_outputs.append(event_encoded_points)
            reconstructed_parameters.append(event_parameters)
            attention_maps.append(event_attention_maps)

            # Record event processing times by component
            if cfg.timing_enabled:
                Utils.sync_device(cfg.device_acc)
                event_end_time = time.perf_counter()
                event_duration = event_end_time - event_start_time
                processing_times.append(event_duration)
                transformer_times.append(transformer_duration)
                regression_times.append(regression_duration)
                seed_reconstruction_times.append(seed_reconstruction_duration)

            # Explicitly delete large tensors after each event to free GPU memory
            del encoded_space_point, parameters
            if attention_weights is not None:
                del attention_weights
            del batch_tensor_hits, batch_padding_hit
            del (
                event_encoded_points,
                event_parameters,
                event_attention_maps,
                event_seeds,
            )

            event_counter += 1

            # More aggressive memory cleanup for test mode
            if event_counter % 1 == 0:  # Clean up every event
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Force garbage collection
                gc.collect()
                # Progress reporting
                if event_counter % 20 == 0:
                    print(f"Processed {event_counter} events...")

        # Clean up batch data after processing all events in this file
        del tensor_hits, tensor_particles, tensor_ID, padding_mask_hit, all_pairs

    # Final speed summary with component breakdown
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        total_time = time.perf_counter() - start_time
        avg_time_per_event = np.mean(processing_times)
        std_time_per_event = np.std(processing_times)
        min_time_per_event = np.min(processing_times)
        max_time_per_event = np.max(processing_times)

        # Component-wise statistics
        avg_transformer_time = np.mean(transformer_times)
        std_transformer_time = np.std(transformer_times)
        avg_regression_time = np.mean(regression_times)
        std_regression_time = np.std(regression_times)
        avg_seed_reconstruction_time = np.mean(seed_reconstruction_times)
        std_seed_reconstruction_time = np.std(seed_reconstruction_times)

        print("\n" + "=" * 80)
        print("PROCESSING SPEED SUMMARY")
        print("=" * 80)
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Number of events processed: {event_counter}")
        print(f"Average events per second: {event_counter / total_time:.2f}")
        print(f"Average events per minute: {event_counter / total_time * 60:.1f}")
        print()
        print("COMPONENT BREAKDOWN:")
        print("-" * 50)
        print(f"{'Component':<20} {'Avg Time':<12} {'Std':<8} {'% of Total':<12}")
        print("-" * 50)
        print(f"{'Total per event:':<20} {avg_time_per_event:.3f}s{'':<4} {std_time_per_event:.3f}s {'100.0%':<12}")
        print(
            f"{'Transformer:':<20} {avg_transformer_time:.3f}s{'':<4} {std_transformer_time:.3f}s "
            f"{avg_transformer_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print(
            f"{'Regression:':<20} {avg_regression_time:.3f}s{'':<4} {std_regression_time:.3f}s "
            f"{avg_regression_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print(
            f"{'Seed reconstruction:':<20} {avg_seed_reconstruction_time:.3f}s{'':<4}"
            f"{std_seed_reconstruction_time:.3f}s "
            f"{avg_seed_reconstruction_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print("-" * 50)
        print(f"Min/Max time per event: {min_time_per_event:.3f}s / {max_time_per_event:.3f}s")
    print("=" * 80)

    return seeds, model_outputs, reconstructed_parameters, attention_maps


def main():
    """
    Main function to run the training of the transformer model for seed reconstruction
    """
    # Parse the command line argument
    cfg = config()
    cfg.parse_args()

    # Print starting information
    ts_print("Starting the training of the transformer model for seed reconstruction")
    cfg.print_config()
    ts_print(f"Using device: {cfg.device_acc}")

    # TODO: make into an optional argument
    log_dir = "training_seeding"

    # Create the model using configuration parameters
    model = SeedTransformer(
        nb_layers_t=cfg.nb_layers_t,
        dim_embedding=cfg.dim_embedding,
        nb_heads=cfg.nb_heads,
        dropout=cfg.dropout,
        num_frequencies=cfg.fourier_num_frequencies,
        device_acc=cfg.device_acc,
    )
    model.to(cfg.device_acc)
    # Create optimizer right now we are using AdamW as it seem to perform well with transformer models
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Create learning rate scheduler with cosine annealing and minimum learning rate
    scheduler = Utils.create_cosine_schedule_with_min_lr(
        opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=100,
        min_lr_ratio=0.01,  # 1% of initial learning rate
    )

    start_epoch = 0

    # TODO: Create a unique suffix for the dataset based on configuration parameters
    tensor_file_suffix = "default"

    # TODO: Right  now we assume that the dataset already exists on disk
    # We will need to write a different file to create the dataset if it does not exist

    ts_print("Loading existing Seeding Dataset from disk")
    # Load the existing dataset (force_recreate=False will load existing files)
    tensor_list = {
        "tensor_hits",
        "tensor_particles",
        "tensor_ID",
        "padding_mask_hit",
        "all_pairs",
    }
    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=f"seeding_data_{tensor_file_suffix}",
        tensor_names=list(tensor_list),
        device=cfg.device_acc,
    )

    train_size = dataset.get_batch_size(0, len(dataset) - 1)

    # Resume training from existing model/checkpoint if specified
    if cfg.resume_training:
        ts_print(f"Resuming training from {cfg.model_path}...")
        # Check if a checkpoint file exists
        start_epoch = model.load(
            path=cfg.model_path,
            device=cfg.device_acc,
            optimizer=opt,
            scheduler=scheduler,
        )

        # Load previous tensorboard logs
        if os.path.exists(log_dir):
            # Calculate total events processed in previous epochs for correct purge_step
            purge_step = start_epoch * train_size
            writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)
        else:
            ts_print(f"Warning: TensorBoard log directory {log_dir} not found. Creating new logs.")
            writer = SummaryWriter(log_dir)
    else:
        # If not resuming, create new TensorBoard writer
        writer = SummaryWriter(log_dir)

    # Split the dataset at the file level to ensure proper train/test separation
    num_files = dataset.get_file_number()
    test_fraction = cfg.test_fraction
    train_files = math.ceil((1 - test_fraction) * num_files)

    ts_print(
        f"Dataset has {num_files} files, using {train_files} for training and {num_files - train_files} for testing"
    )

    # Create file-based train and test indices
    train_file_indices = list(range(train_files))
    test_file_indices = list(range(train_files, num_files))

    # Print model summary
    model.print_model_info()

    # Keep training file order deterministic (no shuffling)
    if not cfg.test_only:
        # Calculate the total number of training events across all training files

        ts_print(f"Training on {train_size} events across {len(train_file_indices)} files")
        # Very important: compile the model
        model = torch.compile(model)
        # Train the model
        ts_print("Starting training of the model")
        model = train_model(
            model,
            train_file_indices,
            dataset,
            train_size,
            cfg.batch_size,
            cfg,
            writer,
            opt,
            scheduler,
            start_epoch=start_epoch,
        )
        ts_print("Training completed")

        # Save the model with full state including architecture parameters
        model.save(
            epoch=start_epoch + cfg.epoch_nb - 1,
            path=cfg.model_path,
            optimizer=opt,
            scheduler=scheduler,
        )

        # Delete training variables to free some memory
        del model
        del opt
        del scheduler
        del train_file_indices
        torch.cuda.empty_cache()

    writer.close()
    # Load model configuration from checkpoint or use config defaults
    model_val = SeedTransformer()

    model_val.load(
        path=cfg.model_path,
        device=cfg.device_acc,
    )
    model_val.to(cfg.device_acc)
    model_val.eval()
    model_val = torch.compile(model_val)

    # Perform validation using test_file_indices
    print("Starting model evaluation with test dataset...")
    (
        seeds_test,
        _,
        reconstructed_parameters_test,
        attention_maps_test,
    ) = run_model(model_val, test_file_indices, dataset, cfg)

    # Collect all test data for monitoring from test_file_indices
    all_hits_test = []
    all_particles_test = []
    all_ID_test = []
    all_padding_mask_test = []
    all_pairs_test = []

    event_counter = 0

    # Iterate through test file indices to collect ground truth data
    for file_idx in test_file_indices:
        batch_data = dataset.get_file(file_idx)

        tensor_hits = batch_data["tensor_hits"]
        tensor_particles = batch_data["tensor_particles"]
        tensor_ID = batch_data["tensor_ID"]
        padding_mask_hit = batch_data["padding_mask_hit"]
        all_pairs = batch_data["all_pairs"]

        # Move tensors to device
        tensor_hits = tensor_hits.to(cfg.device_acc)
        tensor_particles = tensor_particles.to(cfg.device_acc)
        tensor_ID = tensor_ID.to(cfg.device_acc)
        padding_mask_hit = padding_mask_hit.to(cfg.device_acc)
        # all_pairs is already a dictionary, no need to extract from batch

        # Process each event in the batch
        num_events_in_batch = tensor_hits.shape[0]
        # for event_idx in range(min(num_events_in_batch, 20)):
        for event_idx in range(num_events_in_batch):
            # Extract event data
            event_hits = tensor_hits[event_idx]
            event_particles = tensor_particles[event_idx]
            event_ID = tensor_ID[event_idx]
            event_padding_mask = padding_mask_hit[event_idx]
            event_pairs = all_pairs[event_idx]

            # Store test data for monitoring
            all_hits_test.append(event_hits.cpu())
            all_particles_test.append(event_particles.cpu())
            all_ID_test.append(event_ID.cpu())
            all_padding_mask_test.append(event_padding_mask.cpu())
            all_pairs_test.append(event_pairs)

            event_counter += 1

    print(f"Collected data for {event_counter} test events")

    # Convert lists to tensors for monitoring
    hits_test = torch.stack(all_hits_test)
    particles_test = torch.stack(all_particles_test)
    ID_test = torch.stack(all_ID_test)
    padding_mask_hit_test = torch.stack(all_padding_mask_test)

    # Delete the lists immediately to free memory
    del all_hits_test, all_particles_test, all_ID_test, all_padding_mask_test
    gc.collect()

    # Convert tensors to NumPy for monitoring inputs
    hits_test_np = hits_test.cpu().numpy()
    particles_test_np = particles_test.cpu().numpy()
    ID_test_np = ID_test.cpu().numpy()
    padding_mask_hit_test_np = padding_mask_hit_test.cpu().numpy()

    # Create PerformanceMonitor with configuration-only args
    # Derive default monitoring indices if not provided (config may not define these)
    middle_bin = hits_test_np.shape[1] // 2
    event_idx_list = [0]
    bin_idx_list = [middle_bin]

    monitor = PerformanceMonitor(
        event_idx_list=event_idx_list,
        bin_idx_list=bin_idx_list,
        full_print=False,
        save_plots=True,
        min_common_hits=3,
        min_truth_hits=3,
        truth_r_tol=1e-3,
    )

    print("\n" + "=" * 80)
    print("STUDYING SEEDING PERFORMANCE")
    print("=" * 80)
    _monitor_start = time.perf_counter()
    seeding_results = monitor.seeding_performance(
        hits_test_np,
        particles_test_np,
        ID_test_np,
        padding_mask_hit_test_np,
        seeds_test,
        reconstructed_parameters_test,
        all_pairs_test,
        attention_maps_test,
    )
    _monitor_duration = time.perf_counter() - _monitor_start
    print(f"Monitoring finished in {_monitor_duration:.2f}s")

    # Cleanup large intermediates
    del hits_test, particles_test, ID_test, padding_mask_hit_test
    del hits_test_np, particles_test_np, ID_test_np, padding_mask_hit_test_np
    del seeds_test, reconstructed_parameters_test, all_pairs_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if seeding_results is not None:
        del seeding_results
    del monitor  # Delete the monitor object itself
    del attention_maps_test  # Delete the attention maps

    # Final garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Monitoring completed successfully with direct file index approach")


if __name__ == "__main__":
    sys.exit(main())
