from typing import List, Tuple, Dict
import glob
import math
import multiprocessing
import os

import pandas as pd
import numpy as np
import torch

from GUNTAM.Transformer.BinTensor import global_bin, neighbor_bin, no_bin, margin_bin
from GUNTAM.IO.PreprocessingConfig import PreprocessingConfig

import h5py


def _particle_selection(
    data_batch: pd.DataFrame, particles_batch: pd.DataFrame, bins: pd.DataFrame, hit_to_particle: pd.Series, cfg
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Select particles the parameters in the config file

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have 'event_id' column.
        particles_batch: DataFrame containing the particle data for a batch of events, must have 'event_id' column.
        bins: DataFrame containing the bin indices for each hit, must have 'bin0', 'bin1', 'bin2' columns.
        hit_to_particle: Series containing the mapping from hits to particles, indexed the same as data_batch.
        cfg: Configuration object containing parameters for particle selection, specifically cfg.eta_range.

    Returns:
        Tuple of (data_batch, particles_batch, bins, hit_to_particle) where:
        - data_batch: Filtered DataFrame containing only hits associated with selected particles.
        - particles_batch: Filtered DataFrame containing only particles within the eta range.
        - bins: Filtered DataFrame containing bin indices for the remaining hits.
        - hit_to_particle: Filtered Series containing the mapping for the remaining hits.
    """

    # Select the particle in the eta range
    eta_range = cfg.eta_range
    mask = (
        (particles_batch["eta"] >= eta_range[0])
        & (particles_batch["eta"] <= eta_range[1])
        & (particles_batch["pT"] > 0)
        & (particles_batch["d0"] < cfg.vertex_cuts[0])
        & (particles_batch["z0"].abs() < cfg.vertex_cuts[1])
    )
    particles_batch = particles_batch[mask].reset_index(drop=True)

    # Using the hit_to_particle mapping, reclassify hits from removed particles as orphans
    valid_particle_ids = set(particles_batch["particle_id"].unique())

    # Reclassify hits from removed particles as orphans (particle_id = -1)
    invalid_particle_mask = ~hit_to_particle.isin(valid_particle_ids) & (hit_to_particle != -1)
    hit_to_particle.loc[invalid_particle_mask] = -1
    data_batch.loc[invalid_particle_mask, "particle_id"] = -1

    # Remap particle_id to sequential indices for each event
    for event_id in particles_batch["event_id"].unique():
        event_mask = particles_batch["event_id"] == event_id
        event_particles = particles_batch[event_mask]

        # Create mapping from old particle_id to new sequential index
        old_ids = event_particles["particle_id"].values
        particle_id_map = {old_id: new_idx for new_idx, old_id in enumerate(old_ids)}
        particle_id_map[-1] = -1  # Keep -1 for orphan/padding hits

        # Update particle_id in particles_batch to be sequential
        particles_batch.loc[event_mask, "particle_id"] = range(len(event_particles))

        # Update hit_to_particle mapping
        hit_event_mask = data_batch["event_id"] == event_id
        hit_to_particle.loc[hit_event_mask] = hit_to_particle.loc[hit_event_mask].map(lambda pid: particle_id_map.get(pid, -1))

    return data_batch, particles_batch, bins, hit_to_particle


def _hit_selection(data_batch: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Remove hits that fall outside the geometric acceptance defined by cfg.hit_range.

    Args:
        data_batch: DataFrame containing hit data.  Must have columns 'r' (transverse
            radius) and 'z'.
        cfg: Configuration object with a ``hit_range`` attribute
            ``[R_max, Z_max]``.

    Returns:
        Filtered DataFrame with the index reset.
    """
    r_max, z_max = cfg.hit_range
    geometric_mask = (data_batch["r"] <= r_max) & (data_batch["z"].abs() <= z_max)
    # Always keep empty hit (particle_id == -2) regardless of their geometric coordinates
    empty_hit_mask = data_batch["particle_id"] == -2

    return data_batch[geometric_mask | empty_hit_mask].reset_index(drop=True)


def _build_good_pairs_tensors(
    data_batch: pd.DataFrame,
    bins: pd.DataFrame,
    hit_to_particle: pd.Series,
    num_bins: int,
    pv_weight: int = 10,
    particles_batch: pd.DataFrame | None = None,
    z0_weight_bin: int = 0,
) -> torch.Tensor:
    """
    Build a tensor of all hit pairs and their labels (same particle or not) for a batch of events,
    organized by bins.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have 'event_id' column.
        bins: DataFrame containing the bin indices for each hit, must have 'bin0', 'bin1', 'bin2' columns.
        hit_to_particle: Series containing the mapping from hits to particles, indexed the same as data_batch.
        num_bins: Total number of bins used in the binning strategy.
        pv_weight: Weight applied to primary-vertex particle pairs.
        particles_batch: Optional DataFrame with columns 'event_id', 'particle_id', and 'z0' used to
            derive per-particle |z0| for bin-based weighting.
        z0_weight_bin: Bin size for z0-based weighting. The z0 contribution is 2**int(|z0| / z0_weight_bin) - 1,
            added directly to the pair label: final_label = pv_label + 2**int(|z0| / z0_weight_bin) - 1.
            Set to 0 to disable (default). With this formula a particle at weight bracket 0 contributes 0,
            so the minimum label for a non-PV pair is 1 (unchanged).
    Returns:
        A PyTorch tensor of shape [num_events, num_bins, num_pairs, 3] where each pair is represented as
        (hit_idx1, hit_idx2, label) and label is 1 if the hits belong to the same particle.
    """
    print("    Building good pairs tensor...")

    # Build (event_id, particle_id) -> z0 weight lookup: increment = int(|z0| / z0_weight_bin)
    z0_lookup: Dict[Tuple[int, int], int] = {}
    if z0_weight_bin != 0 and particles_batch is not None and "z0" in particles_batch.columns:
        for eid, pid, z0 in particles_batch[["event_id", "particle_id", "z0"]].to_numpy():
            abs_z0 = abs(float(z0))
            weight = int(abs_z0 / z0_weight_bin)
            z0_lookup[(int(eid), int(pid))] = 2**weight - 1

    unique_events = data_batch["event_id"].unique()
    unique_bins = bins["bin1"].unique()
    bin_only = bins[["bin0", "bin1", "bin2"]]
    global_bin_mask = []
    for bin_id in range(max(unique_bins) + 1):
        mask = bin_only.eq(bin_id).any(axis=1)
        global_bin_mask.append(mask)

    # Collect all pairs organized by event and bin
    all_pairs_by_event_bin: Dict[int, Dict[int, np.ndarray]] = {}
    max_pairs_per_bin = 0

    for event_id in unique_events:
        event_mask = data_batch["event_id"] == event_id
        all_pairs_by_event_bin[event_id] = {}

        for bin_id in range(max(unique_bins) + 1):
            # Get all hits in this bin for this event
            bin_mask = global_bin_mask[bin_id] & event_mask
            bin_hit_indices = data_batch[bin_mask].index
            bin_hit_indices_array = bin_hit_indices.to_numpy()
            bin_particle_ids = hit_to_particle.loc[bin_hit_indices_array].values
            n_hits = len(bin_hit_indices_array)
            if n_hits > 0:
                # Create all combinations using broadcasting
                i_indices = np.arange(n_hits)[:, None]  # Shape (n_hits, 1)
                j_indices = np.arange(n_hits)[None, :]  # Shape (1, n_hits)
                # Create masks for valid pairs
                not_self_mask = i_indices != j_indices
                same_particle_mask = bin_particle_ids[i_indices] == bin_particle_ids[j_indices]
                # Exclude orphan hits (particle_id == -1) and empty hits (particle_id == -2) from regular pairs
                not_orphan_mask = (
                    (bin_particle_ids[i_indices] != -1)
                    & (bin_particle_ids[j_indices] != -1)
                    & (bin_particle_ids[i_indices] != -2)
                    & (bin_particle_ids[j_indices] != -2)
                )
                valid_mask = not_self_mask & same_particle_mask & not_orphan_mask

                # Get valid pair indices
                i_valid, j_valid = np.where(valid_mask)

                # Create pairs array: (hit_idx1, hit_idx2, label) using bin-relative indices
                # Label = pv_label + z0_label
                if len(i_valid) > 0:
                    pv_col = "particle_id_pv"
                    if pv_col in data_batch.columns:
                        pv_flags = data_batch.loc[bin_hit_indices_array, pv_col].values == 1
                        is_pv_pair = pv_flags[i_valid]
                        labels = np.where(is_pv_pair, pv_weight, 1).astype(np.int64)
                    else:
                        labels = np.ones(len(i_valid), dtype=np.int64)

                    # Apply z0-based bracket weights additively
                    if z0_lookup:
                        pair_pids = bin_particle_ids[i_valid]
                        z0_labels = np.array(
                            [z0_lookup.get((int(event_id), int(pid)), 0) for pid in pair_pids],
                            dtype=np.int64,
                        )
                        labels = labels + z0_labels

                    pairs = np.stack([i_valid, j_valid, labels], axis=1)
                else:
                    pairs = np.empty((0, 3), dtype=np.int64)

                # Add pairs from non-padded orphan hits to the empty hit (particle_id == -2)
                empty_positions = np.where(bin_particle_ids == -2)[0]
                if len(empty_positions) > 0:
                    empty_pos = empty_positions[0]
                    bin_is_padding = data_batch.loc[bin_hit_indices_array, "is_padding"].values

                    orphan_positions = np.where((bin_particle_ids == -1) & (~bin_is_padding))[0]

                    if len(orphan_positions) > 0:
                        orphan_pairs = np.stack(
                            [
                                orphan_positions,
                                np.full(len(orphan_positions), empty_pos, dtype=np.int64),
                                np.ones(len(orphan_positions), dtype=np.int64),
                            ],
                            axis=1,
                        )
                        pairs = np.concatenate([pairs, orphan_pairs]) if len(pairs) > 0 else orphan_pairs
            else:
                pairs = np.empty((0, 3), dtype=np.int64)

            all_pairs_by_event_bin[event_id][bin_id] = pairs
            max_pairs_per_bin = max(max_pairs_per_bin, len(pairs))

    # Convert to tensor with shape [num_events, num_bins, max_pairs_per_bin, 3]
    num_events = len(unique_events)
    pairs_tensor = torch.zeros((num_events, num_bins, max_pairs_per_bin, 3), dtype=torch.long)

    for event_idx, event_id in enumerate(sorted(unique_events)):
        for bin_id in range(max(unique_bins) + 1):
            pairs = all_pairs_by_event_bin[event_id][bin_id]
            if len(pairs) > 0:
                pairs_tensor[event_idx, bin_id, : len(pairs), :] = torch.tensor(pairs, dtype=torch.long)

    return pairs_tensor


def _to_tensor(
    data_batch: pd.DataFrame,
    particles_batch: pd.DataFrame,
    bins: pd.DataFrame,
    hit_to_particle: pd.Series,
    hit_features: List[str],
    particle_features: List[str],
    num_bins: int,
    max_hits_per_bin: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep only feature of interest then convert the hits, particles, and hit_to_particle map into PyTorch tensors.
    Args:
    data_batch: DataFrame containing the hit data for a batch of events, must have 'event_id' column.
    particles_batch: DataFrame containing the particle data for a batch of events, must have 'event_id' column.
    bins: DataFrame containing the bin indices for each hit, must have 'bin0', 'bin1', 'bin2' columns.
    hit_to_particle: Series containing the mapping from hits to particles, indexed the same as data_batch.
    hit_features: List of column names in data_batch to use as hit features.
    particle_features: List of column names in particles_batch to use as particle features.
    num_bins: Total number of bins used in the binning strategy.
    cfg: Configuration object containing max_hit_input parameter.
    Returns:
    Tuple of (hits_tensor, particles_tensor, hit_to_particle_tensor) where:
    - hits_tensor: PyTorch tensor of shape [num_events, num_bins, cfg.max_hit_input, num_hit_features]
      containing the hit features organized by bins.
    - particles_tensor: PyTorch tensor of shape [num_events, num_particles, num_particle_features]
      containing the particle features.
    - hit_to_particle_tensor: PyTorch tensor of shape [num_events, num_bins, cfg.max_hit_input, 1]
      containing the mapping from hits to particles.
    """

    unique_events = data_batch["event_id"].unique()
    unique_bins = bins["bin1"].unique()
    num_events_batch = len(unique_events)
    bin_only = bins[["bin0", "bin1", "bin2"]]
    global_bin_mask = []
    for bin_id in range(max(unique_bins) + 1):
        mask = bin_only.eq(bin_id).any(axis=1)
        global_bin_mask.append(mask)

    # Determine max sizes for padding
    max_particles_per_event = particles_batch.groupby("event_id").size().max()

    # Initialize tensors with proper shapes [num_events, num_bins, cfg.max_hit_input, features]
    hits_tensor = torch.zeros((num_events_batch, num_bins, max_hits_per_bin, len(hit_features)), dtype=torch.float32)
    particles_tensor = torch.zeros((num_events_batch, max_particles_per_event, len(particle_features)), dtype=torch.float32)
    hit_to_particle_tensor = torch.full((num_events_batch, num_bins, max_hits_per_bin, 1), fill_value=-1, dtype=torch.int32)

    # Fill tensors event by event and bin by bin
    for event_idx, event_id in enumerate(sorted(unique_events)):
        event_mask = data_batch["event_id"] == event_id

        # Fill particles tensor
        event_particles = particles_batch[particles_batch["event_id"] == event_id]
        num_event_particles = len(event_particles)

        if num_event_particles > 0:
            particles_tensor[event_idx, :num_event_particles, :] = torch.tensor(
                event_particles[particle_features].values, dtype=torch.float32
            )

        # Fill bin-organized hit tensors
        for bin_id in range(max(unique_bins) + 1):
            # Get all hits in this bin for this event
            bin_mask = global_bin_mask[bin_id] & event_mask
            bin_hits = data_batch[bin_mask]
            num_bin_hits = len(bin_hits)

            if num_bin_hits > 0:
                # Fill hits tensor
                hits_tensor[event_idx, bin_id, :num_bin_hits, :] = torch.tensor(
                    bin_hits[hit_features].values, dtype=torch.float32
                )

                # Fill hit_to_particle tensor
                event_hit_to_particle = hit_to_particle.loc[bin_hits.index]
                hit_to_particle_tensor[event_idx, bin_id, :num_bin_hits, 0] = torch.tensor(
                    event_hit_to_particle.values, dtype=torch.int32
                )

    return hits_tensor, particles_tensor, hit_to_particle_tensor


def _add_padding(
    data_batch: pd.DataFrame,
    bins: pd.DataFrame,
    cfg: PreprocessingConfig,
    max_hits: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add padding hits to the data batch and corresponding bins to ensure each event has a consistent
    number of hits per bin.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events,
            must have 'event_id' columns.
        bins: DataFrame containing the bin indices for each hit, must have 'bin0', 'bin1', 'bin2' columns.
        cfg: Configuration object containing parameters for max_hit_input.
        max_hits: Target number of hits per bin. Defaults to cfg.max_hit_input when None.
    Returns:
        Tuple of (data_batch with padding hits added, bins with corresponding padding bins added)
    """
    target_hits = max_hits if max_hits is not None else cfg.max_hit_input

    # Prepare the padding for each event
    print("    Preparing padding for events...")
    data_batch["is_padding"] = False

    # Get unique bins and events
    unique_bins = bins["bin1"].unique()
    unique_events = data_batch["event_id"].unique()
    bin_only = bins[["bin0", "bin1", "bin2"]]
    global_bin_mask = []
    for bin_id in range(max(unique_bins) + 1):
        mask = bin_only.eq(bin_id).any(axis=1)
        global_bin_mask.append(mask)

    padding_data_rows = []
    padding_bin_rows = []

    for event_id in unique_events:
        event_mask = data_batch["event_id"] == event_id

        for bin_id in unique_bins:
            # Get all hits in this bin for this event (check all three bin columns)
            bin_mask = global_bin_mask[bin_id] & event_mask
            hits_in_bin_ = data_batch[bin_mask]
            num_hits = len(hits_in_bin_)
            # Remove excess hits if there are more than target_hits
            if num_hits > target_hits:
                # Too many hits - prioritize removing hits where bin1 != bin_id
                bins_in_bin = bins.loc[hits_in_bin_.index]
                primary_mask = bins_in_bin["bin1"] == bin_id
                duplicate_indices = hits_in_bin_.index[~primary_mask].tolist()

                hits_to_remove_count = num_hits - target_hits

                # Remove from duplicates first (from the start)
                if len(duplicate_indices) >= hits_to_remove_count:
                    # Enough duplicates to remove
                    indices_to_modify = duplicate_indices[:hits_to_remove_count]
                    bins_subset = bins.loc[indices_to_modify]

                    # Replace bin0 with bin1 where bin0 == bin_id
                    bins.loc[indices_to_modify, "bin0"] = np.where(
                        bins_subset["bin0"] == bin_id, bins_subset["bin1"], bins_subset["bin0"]
                    )

                    # Replace bin2 with bin1 where bin2 == bin_id
                    bins.loc[indices_to_modify, "bin2"] = np.where(
                        bins_subset["bin2"] == bin_id, bins_subset["bin1"], bins_subset["bin2"]
                    )
                else:
                    raise ValueError(
                        f"Not enough hits to remove for Event {event_id}, Bin {bin_id}. "
                        f"Consider adjusting max_hit_input or binning strategy."
                    )

            # Add padding if there are fewer than target_hits
            elif num_hits < target_hits and num_hits > 0:
                num_padding = target_hits - num_hits
                # Create padding entries with same structure as data
                for _ in range(num_padding):
                    # Create padding data row
                    padding_data_row = {}
                    padding_data_row["event_id"] = event_id
                    padding_data_row["particle_id"] = -1  # Padding has no particle
                    padding_data_row["is_padding"] = True  # Mark as padding

                    # Set other columns to default values (0 or NaN)
                    for col in data_batch.columns:
                        if col not in padding_data_row:
                            padding_data_row[col] = 0

                    # Create corresponding padding bin row
                    padding_bin_row = {"bin0": bin_id, "bin1": bin_id, "bin2": bin_id}

                    padding_data_rows.append(padding_data_row)
                    padding_bin_rows.append(padding_bin_row)

    # Add padding rows to both data_batch and bins
    if padding_data_rows:
        padding_data_df = pd.DataFrame(padding_data_rows)
        padding_bins_df = pd.DataFrame(padding_bin_rows)
        data_batch = pd.concat([data_batch, padding_data_df], ignore_index=True)
        bins = pd.concat([bins, padding_bins_df], ignore_index=True)
        print(f"    Added {len(padding_data_rows)} padding hits")

    return data_batch, bins


def _add_empty_hit(
    data_batch: pd.DataFrame,
    bins: pd.DataFrame,
    hit_to_particle: pd.Series,
    num_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepend one all-zeros empty hit (particle_id = -2) at position 0 of every bin for every event.

    The empty hit serves as an anchor: non-padded orphan hits will be paired with it during pair
    construction. It is distinct from padding (particle_id = -1) so it can be identified
    unambiguously later.

    Args:
        data_batch: DataFrame with hit data, must have 'event_id' and 'is_padding' columns.
        bins: DataFrame with 'bin0', 'bin1', 'bin2' columns, aligned with data_batch.
        hit_to_particle: Series mapping each hit (same index as data_batch) to its particle_id.
        num_bins: Total number of bins.

    Returns:
        Tuple of (data_batch, bins, hit_to_particle) with empty hits prepended.
    """
    print("    Adding empty hits at the start of each bin...")
    unique_events = sorted(data_batch["event_id"].unique())

    empty_data_rows = []
    empty_bin_rows = []

    for event_id in unique_events:
        for bin_id in range(num_bins):
            empty_row = {col: 0 for col in data_batch.columns}
            empty_row["event_id"] = event_id
            empty_row["particle_id"] = -2
            empty_row["is_padding"] = False
            empty_data_rows.append(empty_row)
            empty_bin_rows.append({"bin0": bin_id, "bin1": bin_id, "bin2": bin_id})

    num_empty = len(empty_data_rows)
    empty_data_df = pd.DataFrame(empty_data_rows)
    empty_bins_df = pd.DataFrame(empty_bin_rows)

    # Prepend empty hits so they occupy position 0 inside each bin slice
    data_batch = pd.concat([empty_data_df, data_batch], ignore_index=True)
    bins = pd.concat([empty_bins_df, bins], ignore_index=True)

    new_h2p_values = np.concatenate(
        [
            np.full(num_empty, -2, dtype=np.asarray(hit_to_particle.values).dtype),
            hit_to_particle.values,
        ]
    )
    hit_to_particle = pd.Series(new_h2p_values, index=range(len(new_h2p_values)))

    return data_batch, bins, hit_to_particle


def _create_padding_mask(
    data_batch: pd.DataFrame,
    bins: pd.DataFrame,
    num_bins: int,
    cfg: PreprocessingConfig,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Create a padding mask tensor indicating which positions in the hit input are padding.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events,
            must have 'event_id' and 'is_padding' columns.
        bins: DataFrame containing the bin indices for each hit,
            must have 'bin0', 'bin1', 'bin2' columns.
        num_bins: Total number of bins used in the binning strategy.
        cfg: Configuration object containing parameters for max_hit_input.
    Returns:
        data_batch : input DataFrame with 'is_padding' removed
        padding_mask: tensor of shape [num_events, num_bins, max_hit_input]
            where True indicates padding positions.
    """
    unique_bins = bins["bin1"].unique()
    unique_events = data_batch["event_id"].unique()
    bin_only = bins[["bin0", "bin1", "bin2"]]
    global_bin_mask = []
    for bin_id in range(max(unique_bins) + 1):
        mask = bin_only.eq(bin_id).any(axis=1)
        global_bin_mask.append(mask)

    # Create padding mask tensor [num_events, num_bins, max_hit_input]
    print("    Creating padding mask...")
    num_events_batch = len(unique_events)

    # Initialize padding mask as all False (PyTorch tensor)
    padding_mask = torch.zeros((num_events_batch, num_bins, cfg.max_hit_input), dtype=torch.bool)

    # Build a mapping from event_id to event index
    event_to_idx = {event_id: idx for idx, event_id in enumerate(sorted(unique_events))}

    # For each event and bin, count real hits and mask padding positions
    for event_id in unique_events:
        event_idx = event_to_idx[event_id]
        event_mask = data_batch["event_id"] == event_id

        for bin_id in unique_bins:
            # Count real hits (non-padding) in this bin for this event
            bin_mask = global_bin_mask[bin_id] & event_mask
            num_real_hits = (~data_batch[bin_mask]["is_padding"]).sum()

            if num_real_hits == 0:
                # Empty bin - mask everything
                padding_mask[event_idx, bin_id, :] = True
            elif num_real_hits < cfg.max_hit_input:
                # Create mask: True if position >= num_real_hits
                padding_mask[event_idx, bin_id, num_real_hits:] = True

    return data_batch, padding_mask


def _orphan_hit_removal(data_batch: pd.DataFrame, fraction_to_drop: float, random_state: int = 1993) -> pd.DataFrame:
    """
    Randomly drop a fraction of the hits with no associated particle (orphan hits) from the data batch.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have a 'particle_id' column.
        fraction_to_drop: Fraction of orphan hits to randomly drop (between 0 and 1).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with the specified fraction of orphan hits removed.
    """
    if fraction_to_drop <= 0.0:
        return data_batch

    orphan_hits = data_batch[data_batch["particle_id"] == -1]
    num_orphan_hits = len(orphan_hits)
    num_to_drop = int(num_orphan_hits * fraction_to_drop)

    if num_to_drop > 0:
        drop_indices = orphan_hits.sample(n=num_to_drop, random_state=random_state).index
        data_batch = data_batch.drop(index=drop_indices).reset_index(drop=True)

    return data_batch


def _bin_data(data_batch: pd.DataFrame, cfg: PreprocessingConfig) -> Tuple[pd.DataFrame, int]:
    """
    Bin the data according to the specified binning strategy in the config.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have a 'phi' column.
        cfg: Configuration object containing binning parameters.

    Returns:
        Tuple of (bins DataFrame with bin indices, number of bins)
    """
    phi_range = (-math.pi, math.pi)  # Assuming phi is in the range [-pi, pi]

    if cfg.binning_strategy == "no_bin" or cfg.bin_width > 2 * math.pi:
        bins, num_bins = no_bin(data_batch[["phi"]])
    elif cfg.binning_strategy == "global":
        bins, num_bins = global_bin(data_batch[["phi"]], cfg.bin_width, phi_range)
    elif cfg.binning_strategy == "neighbor":
        bins, num_bins = neighbor_bin(data_batch[["phi"]], cfg.bin_width, phi_range)
    elif cfg.binning_strategy == "margin":
        bins, num_bins = margin_bin(data_batch[["phi"]], cfg.bin_width, cfg.binning_margin, phi_range)
    else:
        raise ValueError(f"Unknown binning strategy: {cfg.binning_strategy}")

    # Reset indices to align bins with data_batch
    data_batch = data_batch.reset_index(drop=True)
    bins = bins.reset_index(drop=True)

    return bins, num_bins


def _save_tensor_data(
    file_data: Dict,
    file_path: str,
    tensor_format: str = "pt",
) -> None:
    """
    Save tensor data to file in the specified format.

    Args:
        file_data: Dictionary containing all tensor data to save
        file_path: Path to save the file (without extension)
        tensor_format: Format to save data in ('pt' for PyTorch or 'h5' for HDF5)
    """
    if tensor_format == "pt":
        # Save as PyTorch tensor file
        torch.save(file_data, f"{file_path}.pt")
    elif tensor_format == "h5":
        # Save as compressed HDF5 file
        with h5py.File(f"{file_path}.h5", "w") as f:
            # Save tensor data with gzip compression
            for key, value in file_data.items():
                if isinstance(value, torch.Tensor):
                    # Convert tensor to numpy array for HDF5 storage
                    numpy_value = value.numpy()
                    f.create_dataset(
                        key,
                        data=numpy_value,
                        dtype=numpy_value.dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                elif isinstance(value, list):
                    # Save lists as attributes or datasets depending on content
                    if key == "batch_events":
                        f.create_dataset(key, data=np.array(value), dtype=np.int64, compression="gzip")
                    else:
                        f.attrs[key] = value
                else:
                    # Save scalars as attributes
                    f.attrs[key] = value
    else:
        raise ValueError(f"Unsupported tensor format: {tensor_format}. Use 'pt' or 'h5'")


def _process_single_batch(args: Tuple) -> Tuple[str, Tuple[int, int], int, int]:
    """
    Process a single batch of events and save the resulting tensors to disk.

    Args:
        args: Tuple of (data_batch, particles_batch, cfg, hit_features, particle_features,
                        file_id, barcode, start_event, end_event)

    Returns:
        Tuple of (file_path, (start_event, end_event), num_events_processed, nb_bins_max)
    """
    data_batch, particles_batch, cfg, hit_features, particle_features, file_id, barcode, start_event, end_event = args

    print(f"  Processing events {start_event} to {end_event - 1}")

    # Optionally perform orphan hit removal (seed derived from file_id for independent removal per batch)
    data_batch = _orphan_hit_removal(data_batch, cfg.orphan_hit_fraction, random_state=1993 + file_id)

    # Apply geometric hit range cuts [R_max, Z_max]
    data_batch = _hit_selection(data_batch, cfg)

    # Perform binning and tensor preparation
    bins, nb_bins_max = _bin_data(data_batch, cfg)

    # Sort hits by r before padding so that padding (r=0) is not sorted to the front
    if "r" in data_batch.columns:
        data_batch = data_batch.sort_values("r", ascending=True)
        bins = bins.loc[data_batch.index].reset_index(drop=True)
        data_batch = data_batch.reset_index(drop=True)

    # Create the hit_to_particle mapping for this batch (after all reordering)
    hit_to_particle = data_batch["particle_id"].copy()

    # Apply specific particle selection as defined in the config
    data_batch, particles_batch, bins, hit_to_particle = _particle_selection(
        data_batch, particles_batch, bins, hit_to_particle, cfg
    )

    # Prepend one all-zeros empty hit (particle_id = -2) at position 0 of every bin
    if cfg.orphan_target:
        data_batch, bins, hit_to_particle = _add_empty_hit(data_batch, bins, hit_to_particle, nb_bins_max)

    # Create the padding mask tensor for this batch (after empty hits are inserted)
    data_batch, padding_mask = _create_padding_mask(data_batch, bins, nb_bins_max, cfg)

    # Create the good pairs tensor for this batch
    good_pairs = _build_good_pairs_tensors(
        data_batch,
        bins,
        hit_to_particle,
        nb_bins_max,
        pv_weight=cfg.pv_pair_weight,
        particles_batch=particles_batch,
        z0_weight_bin=cfg.z0_weight_bin,
    )

    data_batch = data_batch.drop(columns=["is_padding", "particle_id_pv"], errors="ignore")

    # Convert to tensors
    hits_tensor, particles_tensor, hit_to_particle_tensor = _to_tensor(
        data_batch,
        particles_batch,
        bins,
        hit_to_particle,
        hit_features,
        particle_features,
        nb_bins_max,
        cfg.max_hit_input,
    )

    full_path = f"{cfg.input_tensor_path}/{cfg.dataset_name}_{barcode}"
    path = f"/{cfg.dataset_name}_{barcode}"

    # Create the output directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    # Get the unique event IDs for this chunk
    batch_events = sorted(data_batch["event_id"].unique())

    # Create a single dictionary with all data
    file_data = {
        "hits_tensor": hits_tensor,
        "particles_tensor": particles_tensor,
        "good_pairs": good_pairs,
        "hit_to_particle_tensor": hit_to_particle_tensor,
        "padding_mask": padding_mask,
        "start_event": start_event,
        "end_event": end_event,
        "nb_bins": nb_bins_max,
        "batch_events": batch_events,
    }

    # Save tensor file in the specified format
    file_base = f"{full_path}/tensor_data_{file_id}_{barcode}"
    _save_tensor_data(file_data, file_base, cfg.tensor_format)

    # Determine file extension based on tensor format
    file_ext = ".pt" if cfg.tensor_format == "pt" else ".h5"

    print(f"  Saved tensor data for events {start_event} to {end_event - 1} ({cfg.tensor_format} format)")

    file_path = f"{path}/tensor_data_{file_id}_{barcode}{file_ext}"
    return file_path, (start_event, end_event), end_event - start_event, nb_bins_max


def compute_barcode(cfg: PreprocessingConfig) -> str:
    """
    Compute a barcode string based on the configuration parameters for easy identification of dataset variants.

    Args:
        cfg: Configuration object containing parameters that affect the dataset preparation.
    Returns:
        A string barcode that encodes key configuration parameters such as binning strategy,
        bin width, max hits, orphan hit fraction, and total event count.
    """
    barcode = f"BS{cfg.binning_strategy}_BW{cfg.bin_width}_MH{cfg.max_hit_input}_PW{int(cfg.pv_pair_weight)}"
    if cfg.orphan_hit_fraction > 0:
        barcode += f"_OF{int(cfg.orphan_hit_fraction * 100)}"
    if cfg.max_events > 0:
        barcode += f"_NE{cfg.max_events}"
    return barcode


def prepare_tensor(
    cfg: PreprocessingConfig,
) -> Dict:
    """
    Read CSV or HDF5 files produced by Read_ACTS_Csv.py or Read_ACTS_Root.py and prepare them for training
    by converting to PyTorch tensors.

    This function performs the complete data preprocessing pipeline including:
    - Loading hit and particle data from CSV or HDF5 files
    - Optionally removing a fraction of orphan hits (hits with no associated particle)
    - Binning hits according to the specified strategy
    - Adding padding to ensure consistent tensor sizes per bin
    - Creating padding masks for attention mechanisms
    - Filtering particles based on eta range and removing associated hits
    - Converting all data to PyTorch tensors

    For each batch of events, a single data file is saved to disk in either PyTorch (.pt) or
    compressed HDF5 (.h5) format containing:
    - `hits_tensor`: Hit data with shape [num_events, num_bins, num_hits, num_hit_features]
    - `particles_tensor`: Particle data with shape [num_events, num_particles, num_particle_features]
    - `good_pairs`: Good pairs tensor for training
    - `hit_to_particle_tensor`: Hit-to-particle mapping with shape [num_events, num_bins, num_hits, 1]
    - `padding_mask`: Padding mask with shape [num_events, num_bins, max_hit_input]
    - Metadata: start_event, end_event, nb_bins, batch_events

    A metadata file is also created containing information about the dataset structure and file paths.
    The output format (PyTorch or HDF5) is controlled by cfg.tensor_format.

    Args:
        cfg: Configuration object containing all parameters including:
            - input_path: Path to input data files
            - input_format: Format of input files ('csv' or 'h5')
            - input_tensor_path: Path where output tensors will be saved
            - events_per_file: Maximum number of events per output tensor file
            - orphan_hit_fraction: Fraction of orphan hits to remove
            - binning_strategy: Binning strategy to use ('global', 'neighbor', 'margin', 'no_bin')
            - bin_width: Width of bins for binning
            - max_hit_input: Maximum number of hits per bin
            - eta_range: Tuple specifying the eta range for particle selection
            - hit_features: List of column names to use as hit features
            - particle_features: List of column names to use as particle features
            - tensor_format: Output tensor file format ('pt' for PyTorch or 'h5' for compressed HDF5)

    Returns:
        Dictionary containing metadata about the processed dataset including:
        - total_events: Total number of events processed
        - events_per_file: Number of events per output file
        - nb_bins: Maximum number of bins used
        - orphan_hit_fraction: Fraction of orphan hits removed
        - eta_range: Eta range used for particle selection
        - file_paths: List of paths to generated tensor files
        - file_event_ranges: List of (start, end) event ranges for each file
    """

    # Use feature lists from config
    hit_features = cfg.hit_features
    particle_features = cfg.particle_features
    needed_columns = list(set(["event_id", "particle_id", "particle_id_pv", "phi"] + hit_features))
    # Metadata variable to be used to describe the dataset and keep track of file paths
    # and event ranges for each tensor file
    total_events = 0
    nb_bins_max = 0
    file_paths: List[str] = []
    file_event_ranges: List[Tuple[int, int]] = []
    orphan_hit_fraction = cfg.orphan_hit_fraction
    events_per_file = cfg.events_per_file
    barcode = compute_barcode(cfg)

    # Get file paths based on format
    input_path = cfg.input_path
    input_format = cfg.input_format
    data_files = []
    # Collect all data files and prepare for loop
    if input_format == "h5":
        # Get all HDF5 files
        data_files = sorted(glob.glob(f"{input_path}/processed_data*.h5"))  # type: ignore[arg-type]
        print(f"Found {len(data_files)} HDF5 file(s)")

    elif input_format == "csv":
        # Look for either space_points or hits CSV files
        space_points_files = sorted(glob.glob(f"{input_path}/space_points_small*.csv"))
        hits_files = sorted(glob.glob(f"{input_path}/hits_small*.csv"))
        particles_files = sorted(glob.glob(f"{input_path}/particles_small*.csv"))
        # Adapt the logic to handle both hits ans space points and pair with particles files
        if space_points_files:
            data_files = list(zip(space_points_files, particles_files))  # type: ignore[arg-type]
            data_type = "space_points"
            print(f"Found {len(space_points_files)} space_points CSV file(s)")
        elif hits_files:
            data_files = list(zip(hits_files, particles_files))  # type: ignore[arg-type]
            data_type = "hits"
            print(f"Found {len(hits_files)} hits CSV file(s)")
        else:
            raise FileNotFoundError(f"No space_points_small*.csv or hits_small*.csv files found in {input_path}")
    else:
        raise ValueError(f"Unsupported input format: {input_format}. Use 'h5' or 'csv'")

    file_id = 0
    tot_event = 0
    # Processing each file
    for file_idx, data_file in enumerate(data_files):
        if cfg.max_events > 0 and tot_event > cfg.max_events:
            print(f"Reached max_events limit of {cfg.max_events}. Stopping further processing.")
            break

        print(f"Processing file {file_idx + 1}/{len(data_files)}")

        if input_format == "h5":
            # Read data from HDF5 file
            # The file can contain either 'space_points' or 'hits' depending on how it was created
            with pd.HDFStore(data_file, mode="r") as store:
                # Check which key exists in the HDF5 file
                keys = [key.lstrip("/") for key in store.keys()]

                if "space_points" in keys:
                    data = store.select("space_points", columns=needed_columns)
                    print("  Loaded space_points data")
                elif "hits" in keys:
                    data = store.select("hits", columns=needed_columns)
                    print("  Loaded hits data")
                else:
                    raise KeyError(f"Neither 'space_points' nor 'hits' found in {data_file}. Available keys: {store.keys()}")

                # Load particles data
                particles = store.get("particles")

        elif input_format == "csv":
            # Read data from CSV files
            data_csv_file, particles_csv_file = data_file  # type: ignore[misc,str-unpack]
            data = pd.read_csv(data_csv_file)  # type: ignore[has-type]
            particles = pd.read_csv(particles_csv_file)  # type: ignore[has-type]
            print(f"  Loaded {data_type} data from {data_csv_file}")  # type: ignore[has-type]

        # Check if the number of events in data is divisible by the events_per_file
        num_events = len(data["event_id"].unique())
        tot_event += num_events
        # Apply max_events limit if specified
        if cfg.max_events > 0 and tot_event > cfg.max_events:
            print(f"  Limiting to {cfg.max_events} events (out of {num_events} available)")
            num_events = cfg.max_events + num_events - tot_event
            # Filter data to only include the first max_events events
            event_ids = sorted(data["event_id"].unique())[:num_events]
            data = data[data["event_id"].isin(event_ids)].reset_index(drop=True)
            particles = particles[particles["event_id"].isin(event_ids)].reset_index(drop=True)

        if num_events % events_per_file != 0:
            print(
                f"WARNING: Number of events in {data_file} ({num_events}) "
                f"is not divisible by events_per_file ({events_per_file})"
            )

        # Build arguments for each batch so they can be processed in parallel
        batch_args = []
        local_file_id = file_id
        for start_event in range(0, num_events, events_per_file):
            end_event = min(start_event + events_per_file, num_events)

            # Select data for this batch of events
            data_batch = data[(data["event_id"] >= start_event) & (data["event_id"] < end_event)].reset_index(drop=True)
            particles_batch = particles[(particles["event_id"] >= start_event) & (particles["event_id"] < end_event)].reset_index(
                drop=True
            )

            batch_args.append(
                (
                    data_batch,
                    particles_batch,
                    cfg,
                    hit_features,
                    particle_features,
                    local_file_id,
                    barcode,
                    start_event,
                    end_event,
                )
            )
            local_file_id += 1

        # Process batches – in parallel when num_workers > 1, sequentially otherwise
        num_workers = min(cfg.num_workers, len(batch_args))
        if num_workers > 1:
            print(f"  Spawning {num_workers} worker processes for {len(batch_args)} batch(es)")
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(_process_single_batch, batch_args)
        else:
            results = [_process_single_batch(args) for args in batch_args]

        # Collect results (results are already ordered by start_event)
        for file_path, event_range, n_events, nb_bins in results:
            file_paths.append(file_path)
            file_event_ranges.append(event_range)
            total_events += n_events
            nb_bins_max = max(nb_bins_max, nb_bins)

        file_id = local_file_id

    # Create the metadata file and save it to the current directory
    metadata = {
        "total_events": total_events,
        "events_per_file": events_per_file,
        "nb_bins": nb_bins_max,
        "orphan_hit_fraction": orphan_hit_fraction,
        "eta_range": cfg.eta_range,
        "tensor_format": cfg.tensor_format,
        "file_paths": file_paths,
        "file_event_ranges": file_event_ranges,
    }

    torch.save(metadata, f"{cfg.input_tensor_path}/metadata_{cfg.dataset_name}_{barcode}.pt")
    return metadata


if __name__ == "__main__":
    # Create config object and parse command line arguments
    cfg = PreprocessingConfig()
    cfg.parse_args()

    # Print the configuration
    cfg.print_config()
    # Prepare tensors using config settings
    prepare_tensor(cfg)
