import numpy as np
import torch
from typing import List, Tuple


def topk_seed_reconstruction(
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    threshold: float = 0.8,
    max_selection: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    K-nearest seeding with threshold: for each valid hit, create a seed consisting of the hit
    itself plus up to "max_selection" other hits with the highest attention values from that hit,
    keeping only those neighbors whose hit score is >= threshold.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        hit_score: tensor [N, 1] with per-hit scores
        threshold: Minimum hit score required to keep a neighbor (default: 0.8)
        max_selection: Maximum number of neighbors to select per hit (default: 4)

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples; one seed per valid hit.
    """
    device = attention_map.device
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

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
        _, topk_idx = torch.topk(att_allowed, k, dim=1, largest=True, sorted=True)  # [N, k]
        topk_global = allowed_indices[topk_idx]  # [N, k]
    else:
        topk_global = torch.empty((num_hits, 0), dtype=torch.long, device=device)

    # Build clusters per valid hit, applying attention threshold filter
    for i in range(num_hits):
        neighbor_indices = topk_global[i]  # [k]

        # Keep only neighbors whose hit score is above threshold
        keep_mask = hit_score[neighbor_indices].squeeze(-1) >= threshold
        kept_neighbors = neighbor_indices[keep_mask]

        # Cluster = hit itself + kept neighbors (could be only the hit if none kept)
        cluster_idx = torch.cat([torch.tensor([i], device=device, dtype=torch.long), kept_neighbors], dim=0)

        # Seed score: sum of attention to kept neighbors / cluster size
        if kept_neighbors.numel() > 0:
            att_sum = float(attention_map[i, kept_neighbors].sum().item())
        else:
            att_sum = 0.0
        seed_score = att_sum / max(1, cluster_idx.numel())

        # No parameter reconstruction: seed params set to zero
        seed_params = np.zeros(5, dtype=np.float32)

        # Append as numpy arrays
        seeds.append((cluster_idx.cpu().numpy(), seed_params, seed_score))

    return seeds


def chained_seed_reconstruction(
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Chain-based seeding: starting from each hit, iteratively add the highest-attention
    neighbor with a greater index above a score threshold to form a chain of hits.

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        hit_score: tensor [N, 1] with per-hit scores
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit chains.
    """
    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    if num_hits == 0:
        return seeds

    # Precompute things
    all_indices = torch.arange(num_hits, device=device)
    valid_hits = hit_score.squeeze(-1) >= score_threshold  # [N] filter by hit score
    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    for start_idx in range(num_hits):
        if used_mask[start_idx] or not valid_hits[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx
        edge_score_sum = 0.0

        for _ in range(max_chain_length - 1):
            # Get scores only for valid, unused hits after current index
            att_scores = attention_map[current_idx]
            valid_mask = (all_indices > current_idx) & (~used_mask) & valid_hits
            if not torch.any(valid_mask):
                break

            # Get best next index and associated attention_score directly
            next_idx = int(torch.argmax(att_scores * valid_mask.float()).item())
            if att_scores[next_idx].item() * valid_mask[next_idx].float() == 0:
                break

            edge_score_sum += att_scores[next_idx].item()
            chain.append(next_idx)
            current_idx = next_idx

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = np.zeros(5, dtype=np.float32)
            seed_score = edge_score_sum / len(chain)
            seeds.append((chain_indices.cpu().numpy(), chain_params, seed_score))

    return seeds


def back_chained_seed_reconstruction(
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Backward chain-based seeding: starting from each hit, iteratively add the highest-attention
    neighbor with a smaller index above a score threshold to form a backward chain of hits.

    This is the backward counterpart to chained_seed_reconstruction, meant to be used with
    attention_backward_loss. It chains hits in reverse order (from later to earlier indices).

    Args:
        attention_map: 2D tensor [N, N] with attention weights
        hit_score: tensor [N, 1] with per-hit scores
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit backward chains.
    """
    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    if num_hits == 0:
        return seeds

    # Precompute things
    all_indices = torch.arange(num_hits, device=device)
    valid_hits = hit_score.squeeze(-1) >= score_threshold  # [N] filter by hit score
    used_mask = torch.zeros(num_hits, dtype=torch.bool, device=device)

    # Process hits from last to first
    for start_idx in range(num_hits - 1, -1, -1):
        if used_mask[start_idx] or not valid_hits[start_idx]:
            continue

        chain = [start_idx]
        current_idx = start_idx
        edge_score_sum = 0.0

        for _ in range(max_chain_length - 1):
            # Get scores only for valid, unused hits before current index
            att_scores = attention_map[current_idx]
            valid_mask = (all_indices < current_idx) & (~used_mask) & valid_hits
            if not torch.any(valid_mask):
                break

            # Get best previous index directly
            prev_idx = int(torch.argmax(att_scores * valid_mask.float()).item())
            if att_scores[prev_idx].item() * valid_mask[prev_idx].float() == 0:
                break

            edge_score_sum += att_scores[prev_idx].item()
            chain.append(prev_idx)
            current_idx = prev_idx

        used_mask[chain] = True

        if len(chain) >= 3:
            chain_indices = torch.tensor(chain, device=device)
            chain_params = np.zeros(1, dtype=np.float32)
            seed_score = edge_score_sum / len(chain)
            seeds.append((chain_indices.cpu().numpy(), chain_params, seed_score))

    return seeds


def weighted_chained_seed_reconstruction(
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
    pairs_per_hit: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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
        hit_score: tensor [N, 1] with per-hit scores
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)
        pairs_per_hit: Number of top-attention neighbors to consider per hit when
            building the initial pair list (default: 2).

    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples for initial per-hit chains.
    """

    device = attention_map.device
    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    # Filter hits by score threshold
    valid_hits = hit_score.squeeze(-1) >= score_threshold  # [N]

    # For each hit, keep the top pairs_per_hit neighbors among valid hits
    k = min(pairs_per_hit, num_hits - 1)
    att = attention_map.clone()
    att.fill_diagonal_(float("-inf"))  # exclude self-pairs

    topk_vals, topk_idx = torch.topk(att, k, dim=1, largest=True, sorted=True)  # [N, k]
    # Flatten into (hit_i, hit_j, score) rows, filtering by hit score
    hit_i_list = torch.arange(num_hits, device=device).unsqueeze(1).expand_as(topk_idx).reshape(-1).long()
    hit_j_list = topk_idx.reshape(-1).long()
    scores_list = topk_vals.reshape(-1)

    both_valid = valid_hits[hit_i_list] & valid_hits[hit_j_list]
    forward_pairs = hit_i_list < hit_j_list
    mask = both_valid & forward_pairs
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

        if score <= 0.0:
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
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    starting_mask: torch.Tensor,
    score_threshold: float = 0.01,
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
        attention_map: 2D tensor [N, N] with attention weights
        hit_score: tensor [N, 1] with per-hit scores
        starting_mask: 1D boolean tensor [N] indicating which hits can be used as starting points
        score_threshold: Minimum attention score to add a hit to the chain (default: 0.01)
        max_chain_length: Maximum length of the chain (default: 5)
        beam_width: Number of top chains to keep at each step (default: 3)
    Returns:
        List of (hit_indices, avg_parameters, seed_score) tuples; one seed per unique best chain.
    """

    def _beam_score(cumulative: float, n_hits: int) -> float:
        """Return the score used to rank beam entries."""
        return cumulative / n_hits

    num_hits = attention_map.size(0)
    seeds: List[Tuple[np.ndarray, np.ndarray, float]] = []

    # Filter hits by score threshold
    valid_hits = hit_score.squeeze(-1) >= score_threshold  # [N]

    # For each hit, keep the top beam_width forward neighbors among valid hits
    k = min(beam_width, num_hits - 1)
    att = attention_map.clone()
    att.fill_diagonal_(float("-inf"))  # exclude self-pairs

    topk_vals, topk_idx = torch.topk(att, k, dim=1, largest=True, sorted=True)  # [N, k]
    # Build dict: hit_i -> [[hit_j, score], ...], keeping only forward pairs among valid hits
    row_idx = torch.arange(num_hits, device=att.device).unsqueeze(1)  # [N, 1]
    valid_mask = valid_hits[topk_idx] & (topk_idx > row_idx)  # [N, k] — filter by hit score

    valid_mask_np = valid_mask.numpy()
    topk_idx_np = topk_idx.numpy()
    topk_vals_np = topk_vals.float().numpy()

    # Find all valid (row, col) entries across the [N, k] mask in one vectorized call,
    rows, cols_k = np.where(valid_mask_np)

    # Group neighbors by source hit: pairs_list[i] = [(neighbor_j, score), ...]
    pairs_list: list[list[tuple[int, float]]] = [[] for _ in range(num_hits)]
    for r, n, s in zip(rows.tolist(), topk_idx_np[rows, cols_k].tolist(), topk_vals_np[rows, cols_k].tolist()):
        pairs_list[r].append((n, s))

    score_mask = starting_mask & valid_hits
    starting_index = torch.nonzero(score_mask, as_tuple=False).squeeze(1).tolist()

    for start in starting_index:
        beam = [([start], 0.0)]  # list of (chain, cumulative_score)
        best_chain = None
        best_score = float("-inf")
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


def multi_bin_batched_beam_search_seed_reconstruction(
    attention_map: torch.Tensor,
    hit_score: torch.Tensor,
    valid_mask: torch.Tensor,
    score_threshold: float = 0.01,
    max_chain_length: int = 5,
    beam_width: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-batched beam search seeding across all bins simultaneously. This algorithm rely on the pytorch topk
    and gather operations to efficiently maintain and extend the beams in parallel for all hits in all bins.

    Args:
        attention_map:    [B, N, N] raw attention logits on device (NOT yet softmaxed)
        hit_score:        [B, N, 1] per-hit scores
        valid_mask:       [B, N] True = valid (not padded) hit
        score_threshold:  minimum hit score to use a hit as chain source or destination
        max_chain_length: maximum number of hits per chain
        beam_width:       number of beams per starting hit

    Returns:
        Tuple of three GPU tensors:
          - compact_chains [B, N, max_chain_length]: compact hit indices per (bin, starting hit);
            -1 for unused slots.
          - params         [B, N, 5]:               seed parameters (all zeros).
          - best_scores    [B, N]:                  best average edge score (-inf if no valid chain).
    """
    device = attention_map.device
    bin_nb = attention_map.size(0)  # B
    hit_nb = attention_map.size(1)  # N

    if hit_nb < beam_width:
        raise ValueError("hit_nb must be greater than beam_width for valid seed reconstruction.")

    # Initialize a tensor with all the row indices
    row_idx = torch.arange(hit_nb, device=device)

    # valid_full: hit must be valid (not padded) AND score >= threshold
    valid_full = (hit_score.squeeze(-1) >= score_threshold) & valid_mask  # [B, N]

    att = attention_map.float().clone()  # [B, N, N]
    att = torch.softmax(att, dim=-1)

    # Mask j<=i and invalid hits: only forward neighbors among valid hits can be selected.
    combined_mask = (row_idx.unsqueeze(0) <= row_idx.unsqueeze(1)).unsqueeze(0) | ~valid_full.unsqueeze(1)
    att.masked_fill_(combined_mask, float("-inf"))

    # Collect for each hits the top-k neighbors and scores in one go
    fwd_vals, fwd_idx = torch.topk(att, beam_width, dim=2, largest=True, sorted=True)  # [B, N, CL]

    # Store all the hit chain in one matrix of shape [B, N, BW, CL], initialized to -1 (invalid hit index)
    chains = torch.full((bin_nb, hit_nb, beam_width, max_chain_length), -1, dtype=torch.long, device=device)
    chains[:, :, :, 0] = row_idx[None, :, None]  # [1, N, 1] broadcasts to [B, N, BW]

    # Initialise the current chain heads and cumulative scores
    heads = chains[:, :, :, 0].clone()  # [B, N, BW]
    chain_scores = torch.zeros(bin_nb, hit_nb, beam_width, device=device, dtype=torch.float32)
    chain_scores[~valid_full] = float("-inf")

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
