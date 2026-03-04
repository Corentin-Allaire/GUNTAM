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
        max_selection: Maximum number of neighbors to select per hit (default: 4)

    Returns:
        List of (hit_indices, avg_parameters) tuples; one seed per valid hit
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
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chain-based seeding: starting from each hit, iteratively add the highest-attention
    neighbor with a greater index above a score threshold to form a chain of hits.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        reconstructed_parameters: tensor [N, D] with per-hit parameters (includes score at index 4)
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters) tuples for initial per-hit chains
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


def back_chained_seed_reconstruction(
    attention_map: torch.Tensor,
    reconstructed_parameters: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Backward chain-based seeding: starting from each hit, iteratively add the highest-attention
    neighbor with a smaller index above a score threshold to form a backward chain of hits.

    This is the backward counterpart to chained_seed_reconstruction, meant to be used with
    attention_backward_loss. It chains hits in reverse order (from later to earlier indices).

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        reconstructed_parameters: tensor [N, D] with per-hit parameters (includes score at index 4)
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters) tuples for initial per-hit backward chains
    """
    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray]] = []

    if num_hits == 0:
        return seeds

    # Precompute things
    all_indices = torch.arange(num_hits, device=device)
    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    # Process hits from last to first
    for start_idx in range(num_hits - 1, -1, -1):
        if used_mask[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx

        for _ in range(max_chain_length - 1):
            # Get scores only for unused hits before current index
            att_scores = attention_map[current_idx]
            valid_mask = (all_indices < current_idx) & (~used_mask) & (att_scores >= score_threshold)
            if not torch.any(valid_mask):
                break

            # Get best previous index directly
            prev_idx = int(torch.argmax(att_scores * valid_mask.float()).item())
            if att_scores[prev_idx] < score_threshold:
                break

            chain.append(prev_idx)
            current_idx = prev_idx

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = reconstructed_parameters[chain_indices].mean(dim=0)
            seeds.append((chain_indices.cpu().numpy(), chain_params.cpu().numpy()))

    return seeds


def weighted_chained_seed_reconstruction(
    attention_map: torch.Tensor,
    reconstructed_parameters: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
    pairs_per_hit: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Weighted chain seeding: build chains of hits by extending forward/backward
    along the highest-attention edges.

    For each hit, the `pairs_per_hit` highest-attention neighbors above `score_threshold`
    are selected. Only forward pairs (hit_i < hit_j) are retained, de-duplicated across
    hits, and processed in descending score order. Each pair seeds a chain that is
    iteratively extended: at every step the algorithm looks at the best outgoing edge
    from the current chain head (forward extension) and the best incoming edge to the
    current chain tail (backward extension), picks whichever has the higher score, and
    appends that hit. Once a hit has been added to a chain it is removed from the
    candidate pool so it cannot be reused. Chains with fewer than 3 hits are discarded.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        reconstructed_parameters: tensor [N, D] with per-hit parameters (includes score at index 4)
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)
        pairs_per_hit: Number of top-attention neighbors to consider per hit when
            building the initial pair list (default: 2).

    Returns:
        List of (hit_indices, avg_parameters) tuples for initial per-hit chains
    """

    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray]] = []

    # For each hit, keep the top pairs_per_hit neighbors above score_threshold
    k = min(pairs_per_hit, num_hits - 1)
    att = attention_map.clone()
    att.fill_diagonal_(float("-inf"))  # exclude self-pairs

    topk_vals, topk_idx = torch.topk(att, k, dim=1, largest=True, sorted=True)  # [N, k]

    # Flatten into (hit_i, hit_j, score) rows, filtering by threshold
    hit_i_list = torch.arange(num_hits, device=device).unsqueeze(1).expand_as(topk_idx).reshape(-1).long()
    hit_j_list = topk_idx.reshape(-1).long()
    scores_list = topk_vals.reshape(-1)

    above_thresh = scores_list >= score_threshold
    forward_pairs = hit_i_list < hit_j_list
    mask = above_thresh & forward_pairs
    hit_i_np = hit_i_list[mask].cpu().numpy()
    hit_j_np = hit_j_list[mask].cpu().numpy()
    scores_np = scores_list[mask].cpu().numpy()

    sort_order = np.argsort(-scores_np)
    hit_i_np = hit_i_np[sort_order]
    hit_j_np = hit_j_np[sort_order]
    scores_np = scores_np[sort_order]

    if len(hit_i_np) == 0:
        return seeds

    for i in range(len(hit_i_np)):
        hit_i = hit_i_np[i]
        hit_j = hit_j_np[i]
        score = scores_np[i]
        to_remove = []

        if score < 0:
            continue

        to_remove.append(hit_j)
        seed = [hit_i, hit_j]
        while len(seed) < max_chain_length:
            # Identify the highest forward and back attention scores for hit_i and hit_j
            backward_mask = hit_j_np == hit_i
            forward_mask = hit_i_np == hit_j
            max_forward_id = np.argmax(scores_np[forward_mask]) if forward_mask.any() else None
            max_backward_id = np.argmax(scores_np[backward_mask]) if backward_mask.any() else None
            max_forward_score = scores_np[forward_mask][max_forward_id] if forward_mask.any() else 0
            max_backward_score = scores_np[backward_mask][max_backward_id] if backward_mask.any() else 0
            if max_forward_score > max_backward_score:
                hit_j = hit_j_np[forward_mask][max_forward_id]
                seed.append(hit_j)
                to_remove.append(hit_j)
            elif max_backward_score > max_forward_score:
                to_remove.append(hit_i)
                hit_i = hit_i_np[backward_mask][max_backward_id]
                seed.append(hit_i)
            else:
                break

        if len(seed) >= 3:
            for hit in to_remove:
                scores_np[hit_j_np == hit] = 0
            seed_indices = torch.tensor(seed, device=device)
            seed_params = reconstructed_parameters[seed_indices].mean(dim=0)
            seeds.append((seed_indices.cpu().numpy(), seed_params.cpu().numpy()))

    return seeds
