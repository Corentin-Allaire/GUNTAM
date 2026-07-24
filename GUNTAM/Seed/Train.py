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
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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

    if optimiser and scheduler:
        scheduler.step()
        print(f"Initial learning rate: {scheduler.get_last_lr()}")

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
            hits_tensor = batch_data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
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
                        # Skip if no valid bins in this batch
                        if len(batch_bin_indices) == 0:
                            continue

                        batch_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)

                        # Perform the hit embedding for all bins in one forward pass
                        embedded_hits = model.embedding(batched_hits)
                        encoded_space_points, attention_maps = model.compute_adjacency(embedded_hits, batched_masks)

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
                    # if status == "Training":
                    #     Utils.log_gradients(model, writer, epoch * nb_events + entry)

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
            model.save(
                epoch=epoch,
                path=backup_path,
                optimizer=optimiser,
                scheduler=scheduler,
            )
            print(f"Saved backup checkpoint to {backup_path}")
        if optimiser and scheduler:
            scheduler.step()

    return model


def run_model(
    model: SeedTransformer,
    hits_tensor: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg: SeedConfig,
) -> tuple:
    """Run inference over files and return per-bin artifacts.

    Performs a forward pass (embedding + attention + dummy regression) and then executes
    a lightweight seed reconstruction on the CPU for every valid bin of each processed
    event. Outputs are kept as Python/NumPy containers to reduce peak memory usage.

    Returns (nested lists; indexing is [event_idx][bin_idx]):
        - seeds: Reconstruction seeds (list of seed tuples per bin). Each seed is a
          tuple (hit_indices: np.ndarray, avg_params: np.ndarray). Empty bins yield [].
        - attention_maps: per-bin np.ndarray[num_valid_hits, num_valid_hits] used for seeding, or None.
    """
    reco_width = 5  # max hits per seed (fixed for now, could be dynamic in the future)
    seed_score_threshold = 0.4  # minimum score for a seed to be considered valid (tunable)
    att_threshold = 0.2  # minimum attention score to consider an edge for seeding (tunable)

    event_duration = 0.0
    transformer_duration = 0.0
    seed_reconstruction_duration = 0.0

    # Initialize speed monitoring
    # Use high-resolution clock; sync once to avoid overlapping prior kernels
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)

    # for event_idx in range(min(num_events_in_batch, 20)):
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        event_start_time = time.perf_counter()

    event_seeds: List[Any] = []
    event_attention_maps: List[Optional[np.ndarray]] = []

    with torch.inference_mode():
        # Timing: Transformer inference (encoding + attention)
        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            t0 = time.perf_counter()

        hits_tensor = hits_tensor.to(cfg.device_acc, dtype=cfg.transformer_config.dtype)

        # Obtain encoded embeddings and attention weights from the model
        encoded_space_point, attention_edge = model(hits_tensor, padding_mask, reco_width)

        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            transformer_duration = time.perf_counter() - t0
            seed_reconstruction_start = time.perf_counter()

    # Determine reconstruction method once (same for every bin)
    reco_method = cfg.reconstruction_method
    if reco_method is None:
        if cfg.has_loss_component("attention_next"):
            reco_method = "beam_search"
        elif cfg.has_loss_component("attention_back"):
            reco_method = "beam_search_backward"
        else:
            raise ValueError(
                "No reconstruction method specified in config."
                "Please set cfg.reconstruction_method or include 'attention_next'/'attention_back' loss components."
            )

    beam_hits_t, beam_params_t, beam_scores_t = None, None, None

    raw_chain_len = cfg.raw_chain_length if cfg.radial_separation_constraint else 3

    if reco_method == "beam_search":

        valid_mask_gpu = (~padding_mask.bool()).squeeze(-1)
        beam_hits_t, beam_params_t, beam_scores_t = Reconstruction.batched_beam_search_seed_reconstruction(
            attention_edge.float(),
            valid_mask_gpu,
            att_threshold=att_threshold,
            max_chain_length=raw_chain_len,
            beam_width=5,
        )
    elif reco_method == "beam_search_backward":

        valid_mask_gpu = (~padding_mask.bool()).squeeze(-1)
        beam_hits_t, beam_params_t, beam_scores_t = Reconstruction.batched_beam_search_seed_reconstruction(
            attention_edge.float(),
            valid_mask_gpu,
            att_threshold=att_threshold,
            max_chain_length=raw_chain_len,
            beam_width=5,
            backward=True,
        )
    else:
        raise ValueError(f"Invalid reconstruction method: {reco_method}. Supported: 'beam_search', 'beam_search_backward'.")

    if cfg.radial_separation_constraint:
        # 3D spherical radius sqrt(x^2 + y^2 + z^2) per hit slot (intentionally includes z).
        r3d_bin_slot_space = torch.sqrt((hits_tensor[..., :3] ** 2).sum(dim=-1))
        beam_hits_t = Reconstruction.apply_radial_separation_filter(
            beam_hits_t,
            r3d_bin_slot_space,
            min_delta_rho_mm=cfg.min_delta_rho_mm,
            target_length=3,
        )

    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        seed_reconstruction_end = time.perf_counter()

    # Transfer the result to CPU and efficiency analysis. This is excluded from timing computation.
    hit_chains_all = beam_hits_t.cpu().numpy().astype(np.int64)  # [B, N, ML]
    scores_all = beam_scores_t.float().cpu().numpy()  # [B, N]
    params_all = beam_params_t.float().cpu().numpy()  # [B, N, F]
    # Scores are already post-softmax; scatter into sparse [B, N, N] for monitoring (zeros except top-k)
    _B, _N, _W, _ = attention_edge.shape
    attention_matrix = torch.zeros(_B, _N, _N, device=attention_edge.device, dtype=torch.float32)
    _b = torch.arange(_B, device=attention_edge.device)[:, None, None].expand(_B, _N, _W)
    attention_matrix[_b.reshape(-1), attention_edge[..., 0].long().reshape(-1), attention_edge[..., 1].long().reshape(-1)] = (
        attention_edge[..., 2].float().reshape(-1)
    )
    attention_softmax_cpu = attention_matrix.cpu().numpy()

    for bin_idx in range(hits_tensor.shape[0]):
        hit_chains_np = hit_chains_all[bin_idx]  # [N, ML]
        scores_np = scores_all[bin_idx]  # [N]
        params_np = params_all[bin_idx]  # [N, F]
        bin_seeds = []
        seen_bs: set = set()
        lengths = (hit_chains_np >= 0).sum(axis=1)  # [N] — vectorized length per chain
        # Only complete 3-hit chains are valid seeds: a finite score alone does not imply the radial
        # filter kept all three hits (it can drop hits or return an incomplete/-1-padded row).
        prefilter = (lengths == 3) & np.isfinite(scores_np) & (scores_np > seed_score_threshold)
        # Deduplicate keeping the highest-scoring instance of each hit-set, matching the exported
        # ONNX model's `_dedup_seeds` (which keeps the max score) so evaluation and inference agree.
        candidate_idx = np.where(prefilter)[0]
        candidate_idx = candidate_idx[np.argsort(-scores_np[candidate_idx], kind="stable")]
        for i in candidate_idx:
            chain_compact = hit_chains_np[i, : lengths[i]]
            key = tuple(sorted(chain_compact.tolist()))
            if key in seen_bs:
                continue
            seen_bs.add(key)
            bin_seeds.append((chain_compact, params_np[i], scores_np[i]))

        event_attention_maps.append(attention_softmax_cpu[bin_idx])
        event_seeds.append(bin_seeds)

    # End seed reconstruction timing
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        seed_reconstruction_duration = seed_reconstruction_end - seed_reconstruction_start
        event_duration = time.perf_counter() - event_start_time

    return (
        event_seeds,
        event_attention_maps,
        event_duration,
        transformer_duration,
        seed_reconstruction_duration,
    )


def main():
    """
    Main function to run the training of the transformer model for seed reconstruction
    """
    # Parse the command line argument
    cfg = SeedConfig()
    cfg.parse_args()

    # Print starting information
    ts_print("Starting the training of the transformer model for seed reconstruction")
    cfg.print_config()
    ts_print(f"Using device: {cfg.device_acc}")

    # If on CUDA, set matmul precision to high for potential speedup (requires PyTorch 2.0+ and compatible hardware)
    if cfg.device_acc.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # TODO: make into an optional argument
    log_dir = "training_seeding"

    # Create the model using configuration parameters
    model = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=cfg.device_acc,
        dtype=cfg.transformer_config.dtype,
    )
    model.to(cfg.device_acc)
    # Create optimizer right now we are using AdamW as it seem to perform well with transformer models
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Create learning rate scheduler with cosine annealing and minimum learning rate
    scheduler = Utils.create_cosine_schedule_with_min_lr(
        opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.num_training_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    start_epoch = 0

    # Check if the metadata file exist
    barcode = compute_barcode(cfg.preprocessing_config)
    metadata_path = f"{cfg.input_tensor_path}/metadata_{cfg.dataset_name}_{barcode}.pt"
    if not os.path.exists(metadata_path) or cfg.recompute_tensor:
        ts_print("Computing new tensor dataset from input files with barcode: ", barcode)
        # Set hit/particle features in preprocessing config
        cfg.preprocessing_config.hit_features = ["x", "y", "z", "r", "phi", "eta"]
        cfg.preprocessing_config.particle_features = ["d0", "z0", "phi", "eta", "pT", "q", "m"]
        # Pass preprocessing_config to prepare_tensor
        prepare_tensor(cfg.preprocessing_config)

    ts_print("Loading existing Seeding Dataset from disk")
    # Load the existing dataset (force_recreate=False will load existing files)
    tensor_list = {
        "hits_tensor",
        "particles_tensor",
        "hit_to_particle_tensor",
        "padding_mask",
        "good_pairs",
    }
    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=f"{cfg.dataset_name}_{barcode}",
        tensor_names=list(tensor_list),
        device=cfg.device_acc,
    )

    # Split the dataset at the file level to ensure proper train/test separation
    num_files = dataset.get_file_number()
    test_fraction = cfg.test_fraction
    train_files = math.ceil((1 - test_fraction) * num_files)

    ts_print(f"Dataset has {num_files} files, using {train_files} for training and {num_files - train_files} for testing")

    # Create file-based train and test indices
    train_file_indices = list(range(train_files))
    ts_print(f"Train file indices: {train_file_indices}")
    test_file_indices = list(range(train_files, num_files))
    ts_print(f"Test file indices: {test_file_indices}")

    train_size = dataset.get_batch_size(0, train_files - 1) if train_files > 0 else 0

    # Keep training file order deterministic (no shuffling)
    if cfg.epoch_nb > 0:

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
            # check if the model dtype matches the config, print a warning if not
            if model.dtype != cfg.transformer_config.dtype:
                ts_print(f"Warning: Loaded model dtype {model.dtype} does not match config dtype {cfg.transformer_config.dtype}.")

            # Load previous tensorboard logs
            if os.path.exists(log_dir):
                # Append to existing log dir so TensorBoard combines all runs
                writer = SummaryWriter(log_dir=log_dir)
            else:
                ts_print(f"Warning: TensorBoard log directory {log_dir} not found. Creating new logs.")
                writer = SummaryWriter(log_dir)
        else:
            # If not resuming, create new TensorBoard writer
            writer = SummaryWriter(log_dir)

        # Print model summary
        model.print_model_info()

        # Calculate the total number of training events across all training files
        ts_print(f"Training on {train_size} events across {len(train_file_indices)} files")
        # Very important: compile the model
        if cfg.device_acc.type != "mps":
            model = torch.compile(model)  # type: ignore[assignment]
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
    model_val = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=cfg.device_acc,
        dtype=cfg.transformer_config.dtype,
    )

    if cfg.no_test:
        ts_print("Skipping evaluation")
        return

    model_val.load(
        path=cfg.model_path,
        device=cfg.device_acc,
    )
    model_val.to(cfg.device_acc)
    model_val = model_val.to(cfg.transformer_config.dtype)
    model_val.eval()
    if cfg.device_acc.type != "mps":
        model_val = torch.compile(model_val)  # type: ignore[assignment]

    # Perform validation using test_file_indices
    print("Starting model evaluation with test dataset...")
    processing_times = []
    transformer_times = []
    seed_reconstruction_times = []
    total_time = 0.0
    start_event = 0
    event_counter = 0
    global_event_counter = 0  # monotonically increasing across all test files

    event_idx_list = [0]
    bin_idx_list = [55]

    min_common_hits = 3

    monitoring = PerformanceMonitor(
        full_print=False,
        save_plots=True,
        min_common_hits=min_common_hits,
        min_truth_hits=3,
        truth_r_tol=1e-3,
    )

    seed_features_dir = os.path.join(cfg.input_tensor_path, "seed_features")
    os.makedirs(seed_features_dir, exist_ok=True)
    feature_indices = cfg.transformer_config.high_level_features
    cosine_feature_indices = cfg.transformer_config.cosine_processing
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for file_idx in test_file_indices:
        batch_data = dataset.get_file(file_idx)
        end_event = start_event + dataset.get_batch_size(file_idx, file_idx)
        event_counter += end_event - start_event
        for event in range(start_event, end_event):
            event_idx = event - start_event
            start_time = time.perf_counter() if cfg.timing_enabled else 0.0
            (
                batch_seeds,
                batch_attention_maps,
                batch_processing_times,
                batch_transformer_times,
                batch_seed_reconstruction_times,
            ) = run_model(model_val, batch_data["hits_tensor"][event_idx], batch_data["padding_mask"][event_idx], cfg)
            total_time += time.perf_counter() - start_time if cfg.timing_enabled else 0.0

            # Accumulate timings for speed summary
            if cfg.timing_enabled:
                processing_times.append(batch_processing_times)
                transformer_times.append(batch_transformer_times)
                seed_reconstruction_times.append(batch_seed_reconstruction_times)
                del batch_processing_times
                del batch_transformer_times
                del batch_seed_reconstruction_times

            if event in event_idx_list:
                for bin_idx in bin_idx_list:
                    monitoring.analyse_bin_performance(
                        event_idx=event_idx,
                        bin_idx=bin_idx,
                        hits=batch_data["hits_tensor"][event_idx][bin_idx].cpu().float().numpy(),
                        particles=batch_data["particles_tensor"][event_idx].cpu().float().numpy(),
                        seeds=batch_seeds[bin_idx],
                        pairs=batch_data["good_pairs"][event_idx][bin_idx].cpu().float().numpy(),
                        attention_map=batch_attention_maps[bin_idx],
                        hit_to_particle=batch_data["hit_to_particle_tensor"][event_idx][bin_idx].cpu().float().numpy(),
                    )

            if cfg.write_seed_tensor:
                # Accumulate seed feature tensors across all bins
                hit_to_particle_event = batch_data["hit_to_particle_tensor"][event_idx]  # [num_bins, max_hits, 1]
                for bin_idx in range(batch_data["hits_tensor"][event_idx].shape[0]):
                    seeds = batch_seeds[bin_idx]
                    bin_hits_tensor = batch_data["hits_tensor"][event_idx][bin_idx]  # [N, num_features]
                    seed_tensor = torch.full((len(seeds), 3), -1, dtype=torch.long, device=bin_hits_tensor.device)
                    # Populate seed_tensor with hit indices for each seed, padding with -1 where necessary
                    for i, seed in enumerate(seeds):
                        idx = torch.tensor(seed[0], dtype=torch.long, device=bin_hits_tensor.device)
                        seed_tensor[i, : len(idx)] = idx
                    # Build feature tensor for this bin's seeds using the specified feature indices and cosine processing
                    features = Reconstruction.build_seed_features_tensor(
                        bin_hits_tensor,
                        seed_tensor,
                        feature_indices=feature_indices,
                        cosine_feature_indices=cosine_feature_indices,
                    )  # [num_seeds, 3, F]
                    # Flatten the last two dimensions to get a [num_seeds, 3*F] feature vector for each seed
                    all_features.append(features.float().cpu().flatten(start_dim=1))  # [num_seeds, 3*F]
                    htp = hit_to_particle_event[bin_idx].cpu().numpy().reshape(-1)  # [max_hits]
                    labels = torch.zeros(len(seeds), dtype=torch.long)
                    # Assign label 1 to seeds that have at least `min_common_hits` with any true particle, otherwise label 0
                    for i, seed in enumerate(seeds):
                        seed_htp = htp[list(seed[0])]
                        for pid in np.unique(seed_htp):
                            if pid >= 0 and int(np.sum(seed_htp == pid)) >= min_common_hits:
                                labels[i] = 1
                                break
                    all_labels.append(labels)

            monitoring.bin_seeding_performance(
                event_idx=global_event_counter,
                event_hits=batch_data["hits_tensor"][event_idx].cpu().float().numpy(),
                event_particles=batch_data["particles_tensor"][event_idx].cpu().float().numpy(),
                event_hit_to_particle=batch_data["hit_to_particle_tensor"][event_idx].cpu().float().numpy(),
                event_seeds=batch_seeds,
            )
            global_event_counter += 1

        start_event = end_event

    print("Model evaluation completed.")
    monitoring.performance_analysis()

    # Save aggregated seed feature tensors and labels to disk
    if cfg.write_seed_tensor and all_features:
        features_tensor = torch.cat(all_features, dim=0)  # [N_seed_tot, 3*F]
        labels_tensor = torch.cat(all_labels, dim=0)  # [N_seed_tot]
        out_path = os.path.join(seed_features_dir, "seed_features.pt")
        torch.save({"features": features_tensor, "labels": labels_tensor}, out_path)
        print(f"Saved {features_tensor.shape[0]} seed feature vectors to {out_path}")

    # Final speed summary with component breakdown
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        avg_time_per_event = np.mean(processing_times)
        std_time_per_event = np.std(processing_times)
        min_time_per_event = np.min(processing_times)
        max_time_per_event = np.max(processing_times)

        # Component-wise statistics
        avg_transformer_time = np.mean(transformer_times)
        std_transformer_time = np.std(transformer_times)
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
            f"{'Seed reconstruction:':<20} {avg_seed_reconstruction_time:.3f}s{'':<4}"
            f"{std_seed_reconstruction_time:.3f}s "
            f"{avg_seed_reconstruction_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print("-" * 50)
        print(f"Min/Max time per event: {min_time_per_event:.3f}s / {max_time_per_event:.3f}s")


if __name__ == "__main__":
    sys.exit(main())
