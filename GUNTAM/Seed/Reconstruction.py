import numpy as np
import torch
from typing import List, Tuple


def topk_seed_reconstruction(
    attention_edge: torch.Tensor,
    max_selection: int = 4,
    att_threshold: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    K-nearest seeding: for each hit, create a seed consisting of the hit itself plus up to
    "max_selection" neighbors with the highest attention values.

    Args:
        attention_edge: 3D tensor [N, width, 3] of top-k edge triplets (source, target, score)
        max_selection: Maximum number of neighbors to select per hit (default: 4).
            Silently capped by width.

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples; one seed per hit.
    """
    device = attention_edge.device
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    num_hits = attention_edge.shape[0]
    if num_hits == 0:
        return seeds

    # Extract target indices and scores; mask self-edges
    topk_idx_all = attention_edge[:, :, 1].long()  # [N, width]
    topk_vals_all = attention_edge[:, :, 2].float()  # [N, width]
    self_mask = topk_idx_all == attention_edge[:, :, 0].long()
    topk_vals_all = topk_vals_all.masked_fill(self_mask, float("-inf"))

    # Select top-max_selection neighbors per hit
    k = min(max_selection, attention_edge.shape[1])
    if k > 0:
        _, top_pos = torch.topk(topk_vals_all, k, dim=1, largest=True, sorted=True)  # [N, k]
        topk_global = topk_idx_all.gather(1, top_pos)  # [N, k]
        topk_scores = topk_vals_all.gather(1, top_pos)  # [N, k]
    else:
        topk_global = torch.empty((num_hits, 0), dtype=torch.long, device=device)
        topk_scores = torch.empty((num_hits, 0), device=device)

    # Build clusters per hit
    for i in range(num_hits):
        # Exclude self-edges (score == -inf)
        valid_nbr = topk_scores[i] > att_threshold
        neighbor_indices = topk_global[i][valid_nbr]

        # Cluster = hit itself + all valid neighbors
        cluster_idx = torch.cat([torch.tensor([i], device=device, dtype=torch.long), neighbor_indices], dim=0)

        # Seed score: sum of attention to neighbors / cluster size
        att_sum = float(topk_scores[i][valid_nbr].sum().item())
        seed_score = att_sum / max(1, cluster_idx.numel())

        # No parameter reconstruction: seed params set to zero
        seed_params = np.zeros(5, dtype=np.float32)

        # Append as numpy arrays
        seeds.append((cluster_idx.cpu().numpy(), seed_params, seed_score))

    return seeds


def chained_seed_reconstruction(
    attention_edge: torch.Tensor,
    max_chain_length: int = 5,
    att_threshold: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Chain-based seeding: starting from each hit, iteratively add the highest-attention
    forward neighbor (greater index) to form a chain of hits.

    Args:
        attention_edge: 3D tensor [N, width, 3] of top-k edge triplets (source, target, score)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit chains.
    """
    device = attention_edge.device
    num_hits = attention_edge.shape[0]
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    if num_hits == 0:
        return seeds

    # Extract target indices and scores; mask self-edges
    topk_idx = attention_edge[:, :, 1].long()  # [N, width]
    topk_vals = attention_edge[:, :, 2].float()  # [N, width]
    self_mask = topk_idx == attention_edge[:, :, 0].long()
    topk_vals = topk_vals.masked_fill(self_mask, float("-inf"))

    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    for start_idx in range(num_hits):
        if used_mask[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx
        edge_score_sum = 0.0

        for _ in range(max_chain_length - 1):
            # Find best forward neighbor (j > current_idx) that is unused
            row_j = topk_idx[current_idx]  # [width]
            row_s = topk_vals[current_idx]  # [width]
            cand_mask = (row_j > current_idx) & ~used_mask[row_j]
            row_s = row_s.masked_fill(~cand_mask, float("-inf"))
            best_k = int(row_s.argmax().item())
            best_score = row_s[best_k].item()
            if best_score < att_threshold:
                break
            best_next = int(row_j[best_k].item())

            edge_score_sum += best_score
            chain.append(best_next)
            current_idx = best_next

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = np.zeros(5, dtype=np.float32)
            seed_score = edge_score_sum / len(chain)
            seeds.append((chain_indices.cpu().numpy(), chain_params, seed_score))

    return seeds


def back_chained_seed_reconstruction(
    attention_edge: torch.Tensor,
    max_chain_length: int = 5,
    att_threshold: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Backward chain-based seeding: starting from each hit, iteratively add the highest-attention
    backward neighbor (smaller index) to form a backward chain of hits.

    This is the backward counterpart to chained_seed_reconstruction, meant to be used with
    attention_backward_loss. It chains hits in reverse order (from later to earlier indices).

    Args:
        attention_edge: 3D tensor [N, width, 3] of top-k edge triplets (source, target, score)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit backward chains.
    """
    device = attention_edge.device
    num_hits = attention_edge.shape[0]
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    if num_hits == 0:
        return seeds

    # Extract target indices and scores; mask self-edges
    topk_idx = attention_edge[:, :, 1].long()  # [N, width]
    topk_vals = attention_edge[:, :, 2].float()  # [N, width]
    self_mask = topk_idx == attention_edge[:, :, 0].long()
    topk_vals = topk_vals.masked_fill(self_mask, float("-inf"))

    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    # Process hits from last to first
    for start_idx in range(num_hits - 1, -1, -1):
        if used_mask[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx
        edge_score_sum = 0.0

        for _ in range(max_chain_length - 1):
            # Find best backward neighbor (j < current_idx) that is unused
            row_j = topk_idx[current_idx]  # [width]
            row_s = topk_vals[current_idx]  # [width]
            cand_mask = (row_j < current_idx) & ~used_mask[row_j]
            row_s = row_s.masked_fill(~cand_mask, float("-inf"))
            best_k = int(row_s.argmax().item())
            best_score = row_s[best_k].item()
            if best_score < att_threshold:
                break
            best_prev = int(row_j[best_k].item())

            edge_score_sum += best_score
            chain.append(best_prev)
            current_idx = best_prev

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = np.zeros(1, dtype=np.float32)
            seed_score = edge_score_sum / len(chain)
            seeds.append((chain_indices.cpu().numpy(), chain_params, seed_score))

    return seeds


def weighted_chained_seed_reconstruction(
    attention_edge: torch.Tensor,
    max_chain_length: int = 5,
    pairs_per_hit: int = 2,
    att_threshold: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Weighted chain seeding: build chains of hits by extending forward/backward
    along the highest-attention edges.

    For each hit, the `pairs_per_hit` highest-attention neighbors are selected. Only
    forward pairs (hit_i < hit_j) are retained, de-duplicated across hits, and processed
    in descending score order. Each pair seeds a chain that is iteratively extended: at
    every step the algorithm looks at the best outgoing edge from the current chain head
    (forward extension) and the best incoming edge to the current chain tail (backward
    extension), picks whichever has the higher score, and appends that hit. Once a hit
    has been added to a chain it is removed from the candidate pool so it cannot be
    reused. Chains with fewer than 3 hits are discarded.

    Args:
        attention_edge: 3D tensor [N, width, 3] of top-k edge triplets (source, target, score)
        max_chain_length: Maximum length of the chain (default: 5)
        pairs_per_hit: Number of top-attention neighbors to consider per hit when
            building the initial pair list (default: 2). Silently capped by width.

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit chains.
    """

    device = attention_edge.device
    num_hits = attention_edge.shape[0]
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    # Extract target indices and scores; mask self-edges
    topk_idx_all = attention_edge[:, :, 1].long()  # [N, width]
    topk_vals_all = attention_edge[:, :, 2].float()  # [N, width]
    self_mask = topk_idx_all == attention_edge[:, :, 0].long()
    topk_vals_all = topk_vals_all.masked_fill(self_mask, float("-inf"))

    # Select top-pairs_per_hit neighbors per hit
    k = min(pairs_per_hit, attention_edge.shape[1])
    _, top_pos = torch.topk(topk_vals_all, k, dim=1, largest=True, sorted=True)  # [N, k]
    topk_idx = topk_idx_all.gather(1, top_pos)  # [N, k]
    topk_vals = topk_vals_all.gather(1, top_pos)  # [N, k]

    # Flatten into (hit_i, hit_j, score) rows, filtering by hit score
    hit_i_list = torch.arange(num_hits, device=device).unsqueeze(1).expand_as(topk_idx).reshape(-1).long()
    hit_j_list = topk_idx.reshape(-1).long()
    scores_list = topk_vals.reshape(-1)

    mask = hit_i_list < hit_j_list  # keep only forward pairs
    hit_i_np = hit_i_list[mask].numpy()
    hit_j_np = hit_j_list[mask].numpy()
    scores_np = scores_list[mask].float().numpy()

    sort_order = np.argsort(-scores_np)
    hit_i_np = hit_i_np[sort_order]
    hit_j_np = hit_j_np[sort_order]
    scores_np = scores_np[sort_order]

    if len(hit_i_np) == 0:
        return seeds

    forward: dict[int, list[int]] = {i: [] for i in range(num_hits)}
    backward: dict[int, list[int]] = {i: [] for i in range(num_hits)}
    for idx in range(len(hit_i_np)):
        forward[hit_i_np[idx]].append(idx)
        backward[hit_j_np[idx]].append(idx)
    forward_map = {i: np.array(v, dtype=np.intp) for i, v in forward.items()}
    backward_map = {i: np.array(v, dtype=np.intp) for i, v in backward.items()}

    for i in range(len(hit_i_np)):
        hit_i = hit_i_np[i]
        hit_j = hit_j_np[i]
        score = scores_np[i].item()
        to_remove = []

        if score < att_threshold:
            continue

        to_remove.append(hit_j)
        seed = [hit_i, hit_j]
        edge_score_sum = score
        while len(seed) < max_chain_length:
            # Identify the highest forward and back attention scores for hit_i and hit_j
            backward_mask = backward_map[hit_i]
            forward_mask = forward_map[hit_j]
            if len(forward_mask) > 0:
                max_forward_id = np.argmax(scores_np[forward_mask])
                max_forward_score = scores_np[forward_mask][max_forward_id].item()
            else:
                max_forward_id = None
                max_forward_score = 0.0

            if len(backward_mask) > 0:
                max_backward_id = np.argmax(scores_np[backward_mask])
                max_backward_score = scores_np[backward_mask][max_backward_id].item()
            else:
                max_backward_id = None
                max_backward_score = 0.0

            if max_forward_score > max_backward_score:
                hit_j = hit_j_np[forward_mask][max_forward_id]
                seed.append(hit_j)
                to_remove.append(hit_j)
                edge_score_sum += max_forward_score
            elif max_backward_score > max_forward_score:
                to_remove.append(hit_i)
                hit_i = hit_i_np[backward_mask][max_backward_id]
                seed.append(hit_i)
                edge_score_sum += max_backward_score
            else:
                break

        if len(seed) >= 3:
            if to_remove:
                scores_np[np.concatenate([backward_map[hit] for hit in to_remove])] = 0
            seed_arr = np.array(seed, dtype=np.int64)
            seed_params = np.zeros(5, dtype=np.float32)
            seed_score = edge_score_sum / len(seed)
            seeds.append((seed_arr, seed_params, seed_score))

    return seeds


def beam_search_seed_reconstruction(
    attention_edge: torch.Tensor,
    starting_mask: torch.Tensor,
    att_threshold: float = 0.2,
    max_chain_length: int = 5,
    beam_width: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Beam search seeding: for each hit allowed by `starting_mask`, maintain a beam of
    the top `beam_width` partial chains and iteratively extend them forward (to hits
    with a larger index) until no eligible neighbor remains or `max_chain_length` is
    reached.

    Scoring rule during search:
      - While a chain has fewer than 3 hits: cumulative sum of edge attention scores.
      - Once a chain has 3 or more hits: average edge attention score (cumulative sum
        divided by number of edges), which rewards compactness.

    The best-scoring chain from each beam is kept as a seed candidate.  Duplicate
    seeds (same set of hit indices) are discarded.  Only chains of length >= 3 are
    returned.

    Args:
        attention_edge: 3D tensor [N, width, 3] of top-k edge triplets (source, target, score)
        starting_mask: 1D boolean tensor [N] indicating which hits can be used as starting points
        max_chain_length: Maximum length of the chain (default: 5)
        beam_width: Number of top chains to keep at each step (default: 3). Silently capped by width.
    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples; one seed per unique best chain.
    """

    def _beam_score(cumulative: float, n_hits: int) -> float:
        """Return the score used to rank beam entries."""
        return cumulative / n_hits

    device = attention_edge.device
    num_hits = attention_edge.shape[0]
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    # Extract target indices and scores; mask self-edges
    topk_idx_all = attention_edge[:, :, 1].long()  # [N, width]
    topk_vals_all = attention_edge[:, :, 2].float()  # [N, width]
    self_mask = topk_idx_all == attention_edge[:, :, 0].long()
    topk_vals_all = topk_vals_all.masked_fill(self_mask, float("-inf"))

    # Select top-beam_width neighbors per hit
    k = min(beam_width, attention_edge.shape[1])
    _, top_pos = torch.topk(topk_vals_all, k, dim=1, largest=True, sorted=True)  # [N, k]
    topk_idx = topk_idx_all.gather(1, top_pos)  # [N, k]
    topk_vals = topk_vals_all.gather(1, top_pos)  # [N, k]

    # Build dict: hit_i -> [[hit_j, score], ...], keeping only forward pairs
    row_idx = torch.arange(num_hits, device=device).unsqueeze(1)  # [N, 1]
    valid_mask = topk_idx > row_idx  # [N, k]

    valid_mask_np = valid_mask.numpy()
    topk_idx_np = topk_idx.numpy()
    topk_vals_np = topk_vals.float().numpy()

    # Find all valid (row, col) entries across the [N, k] mask in one vectorized call,
    rows, cols_k = np.where(valid_mask_np)

    # Group neighbors by source hit: pairs_list[i] = [(neighbor_j, score), ...]
    pairs_list: list[list[tuple[int, float]]] = [[] for _ in range(num_hits)]
    for r, n, s in zip(rows.tolist(), topk_idx_np[rows, cols_k].tolist(), topk_vals_np[rows, cols_k].tolist()):
        pairs_list[r].append((n, s))

    starting_index = torch.nonzero(starting_mask, as_tuple=False).squeeze(1).tolist()

    for start in starting_index:
        beam = [([start], 0.0)]  # list of (chain, cumulative_score)
        best_chain = None
        best_score = att_threshold
        len_chain = 1
        while len_chain < max_chain_length and beam:
            new_beam = []
            for chain, beam_score in beam:
                last_hit = chain[-1]
                for neighbor, score in pairs_list[last_hit]:
                    new_score = beam_score + score
                    new_chain = chain + [neighbor]
                    new_beam.append((new_chain, new_score))

                    chain_score = _beam_score(new_score, len(new_chain))
                    if chain_score > best_score and len(new_chain) >= 3:
                        best_score = chain_score
                        best_chain = new_chain

            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            len_chain += 1

        if best_chain is not None:
            seed_params = np.zeros(5, dtype=np.float32)
            seeds.append((np.array(best_chain, dtype=np.int64), seed_params, best_score))

    return seeds


def batched_beam_search_seed_reconstruction(
    attention_edge: torch.Tensor,
    valid_mask: torch.Tensor,
    att_threshold: float = 0.0,
    max_chain_length: int = 5,
    beam_width: int = 3,
    backward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementation of the beam search seed reconstruction algorithm that runs in a fully vectorized manner on GPU or CPU.
    This implementation process an entire event at a time, performing the beam search for all hits in all bins in parallel.

    Args:
        attention_edge:   [B, N, width, 3] top-k edge triplets (source, target, score) from forward()
        valid_mask:       [B, N] True = valid (not padded) hit
        att_threshold:    minimum attention score to consider an edge (default: 0.0)
        max_chain_length: maximum number of hits per chain
        beam_width:       number of beams per starting hit
        backward:         if True, chains extend to hits with smaller indices (backward mode);
                          if False (default), chains extend to hits with larger indices (forward mode).

    Returns:
        Tuple of three GPU tensors:
          - compact_chains [B, N, max_chain_length]: compact hit indices per (bin, starting hit);
            -1 for unused slots.
          - params         [B, N, 5]:               seed parameters (all zeros).
          - best_scores    [B, N]:                  best average edge score (-inf if no valid chain).
    """
    device = attention_edge.device
    bin_nb = attention_edge.shape[0]  # B
    hit_nb = attention_edge.shape[1]  # N
    edge_width = attention_edge.shape[2]  # width

    if hit_nb < beam_width:
        raise ValueError(f"hit_nb ({hit_nb}) must be >= beam_width ({beam_width})")

    # Extract target indices and scores from triplets
    raw_idx = attention_edge[..., 1].long()  # [B, N, width]
    raw_vals = attention_edge[..., 2]  # [B, N, width]

    # Initialize a tensor with all the row indices
    row_idx = torch.arange(hit_nb, device=device)

    # valid_full: hit must be valid (not padded)
    valid_full = valid_mask  # [B, N]

    # Direction mask: kill edges that go the wrong way
    # Forward: keep j > i. Backward: keep j < i.
    if backward:
        dir_mask = raw_idx >= row_idx[None, :, None]  # kill j >= i
    else:
        dir_mask = raw_idx <= row_idx[None, :, None]  # kill j <= i

    # Remove edge from padded hits, hits below score threshold, and edges that go the wrong way
    b_exp = torch.arange(bin_nb, device=device)[:, None, None].expand_as(raw_idx)  # [B, N, width] for batch indexing
    tgt_valid = valid_full[b_exp, raw_idx]  # [B, N, width]
    raw_vals = raw_vals.masked_fill(dir_mask | ~tgt_valid | (raw_vals < att_threshold), float("-inf"))

    # Re-topk to enforce beam_width (clipped to available edge_width)
    beam_width = min(beam_width, edge_width)
    fwd_vals, top_pos = torch.topk(raw_vals, beam_width, dim=2, largest=True, sorted=True)
    fwd_idx = raw_idx.gather(2, top_pos)  # [B, N, beam_width]

    # Store all the hit chain in one matrix of shape [B, N, BW, CL], initialized to -1 (invalid hit index)
    chains = torch.full((bin_nb, hit_nb, beam_width, max_chain_length), -1, dtype=torch.long, device=device)
    chains[:, :, :, 0] = row_idx[None, :, None]  # [1, N, 1] broadcasts to [B, N, BW]

    # Initialise the current chain heads and cumulative scores
    heads = chains[:, :, :, 0].clone()  # [B, N, BW]
    chain_scores = torch.full((bin_nb, hit_nb, beam_width), float("-inf"), device=device, dtype=torch.float32)
    chain_scores = chain_scores.masked_fill(valid_full.unsqueeze(-1), 0.0)

    # Initialise the best chain tracking tensors
    best_chains = chains[:, :, 0, :].clone()  # [B, N, CL]
    best_scores = torch.full((bin_nb, hit_nb), float("-inf"), device=device, dtype=torch.float32)
    best_lens = torch.zeros(bin_nb, hit_nb, dtype=torch.long, device=device)

    # Iteratively extend the chains in the beam for max_chain_length steps
    for step in range(1, max_chain_length):
        # Gather neighbors for all current heads in one go: [B, N, BW] -> [B, N, BW, BW]
        b_idx = torch.arange(bin_nb, device=device)[:, None, None]
        cand_hits = fwd_idx[b_idx, heads]  # [B, N, BW, BW]
        cand_att = fwd_vals[b_idx, heads]  # [B, N, BW, BW]

        # Accumulate; -inf propagates through addition for dead beams
        cand_score = chain_scores.unsqueeze(3) + cand_att  # [B, N, BW, BW]

        # Combine all the candidates in one dimension
        flat_score = cand_score.reshape(bin_nb, hit_nb, beam_width * beam_width)  # [B, N, BW^2]
        flat_hits = cand_hits.reshape(bin_nb, hit_nb, beam_width * beam_width)  # [B, N, BW^2]

        # Select the top-k among all candidates for each (bin, starting hit)
        sel_scores, sel_pos = torch.topk(flat_score, beam_width, dim=2, largest=True, sorted=True)
        sel_parent = sel_pos // beam_width
        sel_hits = flat_hits.gather(2, sel_pos)  # [B, N, BW]

        # Inherit chain histories from selected parents, write new hit at position `step`
        sel_parent_exp = sel_parent.unsqueeze(3).expand(bin_nb, hit_nb, beam_width, max_chain_length)
        new_chains = chains.gather(2, sel_parent_exp)  # [B, N, BW, CL]
        new_chains[:, :, :, step] = sel_hits

        chains = new_chains
        heads = sel_hits
        chain_scores = sel_scores

        # Track best chain per (bin, starting hit): avg score, length >= 3 only
        chain_len = step + 1
        if chain_len >= 3:
            avg_scores = chain_scores / chain_len  # [B, N, BW]
            step_best, step_beam = avg_scores.max(dim=2)  # [B, N]

            improve = step_best > best_scores  # [B, N]
            if improve.any():
                best_scores = torch.where(improve, step_best, best_scores)
                best_lens = torch.where(improve, torch.full_like(best_lens, chain_len), best_lens)

                b_idx = torch.arange(bin_nb, device=device)[:, None]  # [B, 1]
                n_idx = torch.arange(hit_nb, device=device)[None, :]  # [1, N]
                winning_chains = chains[b_idx, n_idx, step_beam]  # [B, N, CL]
                best_chains = torch.where(improve.unsqueeze(-1), winning_chains, best_chains)

    params = torch.zeros(bin_nb, hit_nb, 5, device=device, dtype=torch.float32)

    return best_chains, params, best_scores


def build_seed_features_tensor(
    hits_tensor: torch.Tensor,
    seed_tensor: torch.Tensor,
    feature_indices: List[int] = [0, 1, 2, 3, 4, 5],
    cosine_feature_indices: List[int] = [4],
) -> torch.Tensor:
    """
    Build a feature tensor for each seed by gathering hit coordinates.
    This can then be passed to NN for parameter regression and good/fake classification.

    Args:
        hits_tensor: Float tensor of shape [N, num_features] containing the hit
            features for all hits in a single bin.
        seed_tensor: Long tensor of shape [num_seeds, max_seed_size] containing the
            per-seed hit indices.  A value of -1 indicates a padding slot.
        feature_indices: Ordered list of column indices from `hits_tensor` to
            include in the output.  Mirrors `cfg.embedding_feature` /
            `cfg.high_level_features`.  Default: [0, 1, 2, 3, 4, 5].
        cosine_feature_indices: Subset of `feature_indices` for which cos/sin
            decomposition is applied.  Mirrors `cfg.cosine_processing`.
            Default: [4] (phi).

    Returns:
        Float tensor of shape [num_seeds, max_seed_size, F] where F is
        len(feature_indices) + len(cosine_feature_indices) (each cosine-processed
        feature adds one extra column for sin).  Padding slots contain all zeros.
    """
    pad_mask = seed_tensor == -1  # [num_seeds, max_seed_size]
    ids = seed_tensor.clamp(min=0)  # replace -1 with 0 to avoid out-of-bounds indexing

    feats = hits_tensor[ids]  # [num_seeds, max_seed_size, num_features]

    cosine_set = set(cosine_feature_indices)
    parts: List[torch.Tensor] = []
    for idx in feature_indices:
        if idx in cosine_set:
            parts.append(torch.cos(feats[..., idx]))
            parts.append(torch.sin(feats[..., idx]))
        else:
            parts.append(feats[..., idx])

    result = torch.stack(parts, dim=-1)  # [num_seeds, max_seed_size, F]

    result[pad_mask] = 0.0

    return result
