import numpy as np
import torch
from typing import List, Tuple


def topk_seed_reconstruction(
    attention_map: torch.Tensor,
    reconstructed_parameters: torch.Tensor,
    threshold: float = 0.8,
    max_selection: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    K-nearest seeding with threshold: for each valid hit, create a seed consisting of the hit
    itself plus up to "max_selection" other hits with the highest attention values from that hit,
    keeping only those neighbors whose attention score is >= threshold.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        reconstructed_parameters: tensor [N, D] with per-hit parameters (includes score at index 4)
        threshold: Minimum attention score required to keep a neighbor (default: 0.8)
        max_selection: Maximum number of neighbors to select per hit (default: 5)

    Returns:
        clusters: List of (hit_indices, avg_parameters) tuples; one cluster per valid hit
        seeds: Empty list (kept for signature compatibility)
    """
    device = attention_map.device
    seeds: List[Tuple[np.ndarray, np.ndarray]] = []

    num_hits = attention_map.size(0)
    if num_hits == 0:
        return seeds

    # Use all hits (remove selection on scores)
    allowed_indices = torch.arange(num_hits, device=device)
    allowed_count = num_hits
    k = min(max_selection, max(0, allowed_count - 1))

    # Restrict attention matrix to allowed columns
    att_allowed = attention_map[:, allowed_indices].clone()

    # Forbid self-attention (set diagonal to -inf)
    if allowed_count == num_hits:
        att_allowed.fill_diagonal_(float("-inf"))
    else:
        arange = torch.arange(num_hits, device=device)
        common = torch.where((arange[:, None] == allowed_indices[None, :]))
        if common[0].numel() > 0:
            att_allowed[common] = float("-inf")

    if k > 0:
        # Get top-k attention scores and indices per row
        topk_vals, topk_idx = torch.topk(att_allowed, k, dim=1, largest=True, sorted=True)  # [N, k]
        topk_global = allowed_indices[topk_idx]  # [N, k]
    else:
        topk_vals = torch.empty((num_hits, 0), dtype=attention_map.dtype, device=device)
        topk_global = torch.empty((num_hits, 0), dtype=torch.long, device=device)

    # Build clusters per valid hit, applying attention threshold filter
    for i in range(num_hits):
        neighbor_scores = topk_vals[i]  # [k]
        neighbor_indices = topk_global[i]  # [k]

        # Keep only neighbors above threshold
        keep_mask = neighbor_scores >= threshold
        kept_neighbors = neighbor_indices[keep_mask]

        # Cluster = hit itself + kept neighbors (could be only the hit if none kept)
        cluster_idx = torch.cat([torch.tensor([i], device=device, dtype=torch.long), kept_neighbors], dim=0)

        # Compute average reconstructed parameters for this cluster
        seed_params = reconstructed_parameters[cluster_idx].mean(dim=0)

        # Append as numpy arrays
        seeds.append((cluster_idx.cpu().numpy(), seed_params.cpu().numpy()))

    return seeds


def chained_seed_reconstruction(
    attention_map: torch.Tensor,
    reconstructed_parameters: torch.Tensor,
    score_threshold: float = 0.001,
    max_chain_length: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chain-based seeding: starting from each hit, iteratively add the highest-attention
    neighbor with a greater index above a score threshold to form a chain of hits.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        reconstructed_parameters: tensor [N, D] with per-hit parameters (includes score at index 4)
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.5)
        max_chain_length: Maximum length of the chain (default: 5)
    Returns:
        clusters: List of (hit_indices, avg_parameters) tuples for initial per-hit chains
        seeds: Empty list (kept for signature compatibility)
    """
    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray]] = []

    if num_hits == 0:
        return seeds

    # Precompute things
    all_indices = torch.arange(num_hits, device=device)
    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    for start_idx in range(num_hits):
        if used_mask[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx

        for _ in range(max_chain_length - 1):
            # Get scores only for unused hits after current index
            att_scores = attention_map[current_idx]
            valid_mask = (all_indices > current_idx) & (~used_mask) & (att_scores >= score_threshold)
            if not torch.any(valid_mask):
                break

            # Get best next index directly
            next_idx = int(torch.argmax(att_scores * valid_mask.float()).item())
            if att_scores[next_idx] < score_threshold:
                break

            chain.append(next_idx)
            current_idx = next_idx

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = reconstructed_parameters[chain_indices].mean(dim=0)
            seeds.append((chain_indices.cpu().numpy(), chain_params.cpu().numpy()))

    return seeds
