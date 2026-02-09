from typing import Dict
import torch
import torch.nn.functional as F


def attention_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Compute the attention loss using binary cross-entropy.

    Encourages high attention weights for positive pairs (same particle) and
    low attention weights for negative pairs (different particles).

    Args:
        attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
        pairs1: `[N_pairs]` first indices for evaluated pairs.
        pairs2: `[N_pairs]` second indices for evaluated pairs.
        target: `[N_pairs]` labels in `{+1 (same), -1 (different)}`.

    Note:
        `pairs1`/`pairs2` must be symmetric (include both `(i, j)` and `(j, i)`) and must not
        contain self-pairs `(i, i)`.

    Returns:
        Scalar attention loss tensor
    """

    # Guard: if no pairs, return zero loss
    if pairs1.numel() == 0 or pairs2.numel() == 0:
        # Preserve device from attention tensor
        return torch.tensor(0.0, device=attention_map_bin.device)

    # Direct indexing without masking to surface incorrectly built pairs
    logits = attention_map_bin[pairs1, pairs2]

    # Targets: map {-1, +1} -> {0, 1}
    targets = (target.float() + 1.0) * 0.5

    return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")


def full_attention_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Compute the attention loss using only the last transformer layer's attention.

    Encourages high attention weights for positive pairs (same particle) and
    low attention weights for negative pairs (different particles).
    Compared to attention_loss, this version doesn't use negative pairs, it instead looks
    at all possible pairs not in the positive set as negative. The value of the loss for positive
    and negative pairs is weighted by the number of pairs of each type.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for positive pairs.
            pairs2: `[N_pairs]` second indices for positive pairs.
            target: `[N_pairs]` labels where positives are `+1` (others ignored).

        Note:
            For consistency, positives in (`pairs1`,`pairs2`) should be symmetric (both `(i, j)` and `(j, i)`)
            and should exclude diagonal `(i, i)`.

    Returns:
        Scalar attention loss tensor
    """

    device = attention_map_bin.device
    pos_mask = target == 1
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Build a target matrix
    target = torch.zeros_like(attention_map_bin, device=device)
    target[pairs1[pos_mask], pairs2[pos_mask]] = 1.0

    full_mask = torch.ones_like(attention_map_bin, dtype=torch.bool)
    full_mask[num_valid_hits:, :] = False
    full_mask[:, num_valid_hits:] = False

    # Columns corresponding to hits NOT in any positive pair -> mask them out
    inactive_cols = torch.ones(attention_map_bin.shape[0], dtype=torch.bool, device=device)
    inactive_cols[pos_hits] = False
    full_mask[:, inactive_cols] = False

    logits = attention_map_bin[full_mask]
    targets = target[full_mask]

    pos_weight = 1 / max(pairs1[pos_mask].numel(), 1)
    neg_weight = 1 / max(full_mask.sum().item() - pairs1[pos_mask].numel(), 1)

    # Class weights
    weights = torch.where(targets > 0, pos_weight, neg_weight)

    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")


def top_attention_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Top-k attention loss using BCE-with-logits on masked entries, styled like full_attention_loss.

    - Positives: all provided positive pairs (same particle).
    - Negatives: top-N highest-scoring masked entries not in the positive set, with N = #positives.
    - Masking: restrict to a window up to max positive index and to columns involved in positives.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for positive pairs.
            pairs2: `[N_pairs]` second indices for positive pairs.
            target: `[N_pairs]` labels where positives are `+1` (others ignored).

        Note:
            Provide symmetric positive pairs `(i, j)` and `(j, i)` only; exclude `(i, i)`.

    Returns:
        Scalar attention loss tensor
    """
    device = attention_map_bin.device
    pos_mask = target == 1
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    # Hits involved in positives and valid window size
    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Build window + column mask focusing around positives
    full_mask = torch.ones_like(attention_map_bin, dtype=torch.bool)
    full_mask[num_valid_hits:, :] = False
    full_mask[:, num_valid_hits:] = False

    inactive_cols = torch.ones(attention_map_bin.shape[0], dtype=torch.bool, device=device)
    inactive_cols[pos_hits] = False
    full_mask[:, inactive_cols] = False

    # Positive logits
    pos_i = pairs1[pos_mask]
    pos_j = pairs2[pos_mask]
    pos_scores = attention_map_bin[pos_i, pos_j]
    num_pos = pos_scores.numel()

    # Negative candidates = masked entries excluding positives
    neg_mask = full_mask.clone()
    neg_mask[pos_i, pos_j] = False
    neg_scores = attention_map_bin[neg_mask]

    k = min(num_pos, neg_scores.numel())
    top_neg_scores, _ = torch.topk(neg_scores, k=k, largest=True, sorted=False)

    logits = torch.cat([pos_scores, top_neg_scores], dim=0)
    targets = torch.cat(
        [
            torch.ones(num_pos, device=device),
            torch.zeros(top_neg_scores.numel(), device=device),
        ],
        dim=0,
    )
    # Class-balanced weights
    pos_weight = 1.0 / max(num_pos, 1)
    neg_weight = 1.0 / max(top_neg_scores.numel(), 1)
    weights = torch.where(targets > 0, pos_weight, neg_weight)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")


def attention_next_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Attention loss using cross-entropy for sequential pairs.

    For each hit i in the sequence, if there exists a positive pair (i, i+1),
    we use the attention distribution from hit i as logits and apply cross-entropy
    loss with target = i+1. This encourages the model to attend to the next hit
    in the same particle track.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for pairs.
            pairs2: `[N_pairs]` second indices for pairs.
            target: `[N_pairs]` labels where positives are `+1` used to derive next targets.

        Note:
            Pairs should be symmetric across direction `(i, j)` and `(j, i)` and must not include `(i, i)`.

    """
    device = attention_map_bin.device
    pos_mask = target == 1
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Positive pairs within valid hit range
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]

    # Unique sources among valid hits
    unique_sources = torch.unique(sources)

    # For each source s: choose
    #  - next forward target: min t where t > s
    #  - else (no forward), last backward target: max t where t < s
    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)

    # Vectorized selection per source: prefer min forward target (> s), else max backward (< s)
    if unique_sources.numel() > 0 and sources.numel() > 0:
        # Build [S, M] match matrix (S=unique sources, M=pairs)
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)  # [S, M]

        # Pair-wise forward/backward masks (per pair, relative to its own source)
        forward_pairs_mask = targets > sources  # [M]
        backward_pairs_mask = targets <= sources  # [M]

        # Broadcast to [S, M]
        fwd_mask = source_eq & forward_pairs_mask.view(1, -1)
        back_mask = source_eq & backward_pairs_mask.view(1, -1)

        targets_row = targets.view(1, -1)

        # For forward: take min target; use sentinel = num_valid_hits when absent
        fwd_candidates = torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits))
        fwd_min, _ = torch.min(fwd_candidates, dim=1)  # [S]
        fwd_exists = fwd_mask.any(dim=1)  # [S]

        # For backward: take max target; use sentinel = -1 when absent
        back_candidates = torch.where(back_mask, targets_row, torch.full_like(targets_row, -1))
        back_max, _ = torch.max(back_candidates, dim=1)  # [S]

        # Prefer forward if exists, else backward
        selected_targets = torch.where(fwd_exists, fwd_min, back_max)
    # else: keep selected_targets as sentinel

    # Restrict logits to valid hits (slice the attention map, not the function)
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]
    loss = F.cross_entropy(attention_logits[unique_sources], selected_targets, reduction="sum")

    return loss


def attention_backward_loss(
    attention_map_bin: torch.Tensor,  # [seq_len, seq_len] attention map logits
    pairs1: torch.Tensor,  # [N_pairs] first hit indices of each pair
    pairs2: torch.Tensor,  # [N_pairs] second hit indices of each pair
    target: torch.Tensor,  # [N_pairs] target labels (-1 or 1)
) -> torch.Tensor:
    """
    Attention loss using cross-entropy for sequential pairs.

    For each hit i in the sequence, if there exists a positive pair (i, i-1),
    we use the attention distribution from hit i as logits and apply cross-entropy
    loss with target = i-1. This encourages the model to attend to the previous hit
    in the same particle track.

        Args:
            attention_map_bin: `[seq_len, seq_len]` attention logits matrix.
            pairs1: `[N_pairs]` first indices for pairs.
            pairs2: `[N_pairs]` second indices for pairs.
            target: `[N_pairs]` labels where positives are `+1` used to derive next targets.

        Note:
            Pairs should be symmetric across direction `(i, j)` and `(j, i)` and must not include `(i, i)`.

    """
    device = attention_map_bin.device
    pos_mask = target == 1
    if not torch.any(pos_mask):
        return torch.tensor(0.0, device=device)

    pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
    num_valid_hits = int(torch.max(pos_hits).item()) + 1

    # Positive pairs within valid hit range
    sources = pairs1[pos_mask]
    targets = pairs2[pos_mask]

    # Unique sources among valid hits
    unique_sources = torch.unique(sources)

    # For each source s: choose
    #  - last backward target: max t where t < s
    #  - else (no backward), next forward target: min t where t > s
    selected_targets = torch.full_like(unique_sources, fill_value=num_valid_hits)

    # Vectorized selection per source: prefer max backward target (< s), else min forward (> s)
    if unique_sources.numel() > 0 and sources.numel() > 0:
        # Build [S, M] match matrix (S=unique sources, M=pairs)
        source_eq = unique_sources.view(-1, 1) == sources.view(1, -1)  # [S, M]

        # Pair-wise forward/backward masks (per pair, relative to its own source)
        forward_pairs_mask = targets >= sources  # [M]
        backward_pairs_mask = targets < sources  # [M]

        # Broadcast to [S, M]
        fwd_mask = source_eq & forward_pairs_mask.view(1, -1)
        back_mask = source_eq & backward_pairs_mask.view(1, -1)

        targets_row = targets.view(1, -1)

        # For backward: take max target; use sentinel = -1 when absent
        back_candidates = torch.where(back_mask, targets_row, torch.full_like(targets_row, -1))
        back_max, _ = torch.max(back_candidates, dim=1)  # [S]
        back_exists = back_mask.any(dim=1)  # [S]

        # For forward: take min target; use sentinel = num_valid_hits when absent
        fwd_candidates = torch.where(fwd_mask, targets_row, torch.full_like(targets_row, num_valid_hits))
        fwd_min, _ = torch.min(fwd_candidates, dim=1)  # [S]

        # Prefer backward if exists, else forward
        selected_targets = torch.where(back_exists, back_max, fwd_min)
    # else: keep selected_targets as sentinel

    # Restrict logits to valid hits (slice the attention map, not the function)
    attention_logits = attention_map_bin[:num_valid_hits, :num_valid_hits]
    loss = F.cross_entropy(attention_logits[unique_sources], selected_targets, reduction="sum")

    return loss


def reconstruction_loss(
    reconstructed_particle: torch.Tensor,
    particles_data: torch.Tensor,
    padded_mask: torch.Tensor,
    loss_type: str = "MSE",
) -> Dict[str, torch.Tensor]:
    """
    Compute the reconstruction loss for the particles.

    Computes element-wise loss (MSE or L1) between reconstructed and original
    particle properties for each physical quantity. Excludes padded hits and
    hits without associated particles (pT == 0).

    Expected tensor layout:
    - `reconstructed_particle`: [batch, max_hits, 5] with components [z, eta, sin(phi), cos(phi), pT]
    - `particles_data`:        [batch, max_hits, 4] with components [z, eta, phi, pT]
    - `padded_mask`:           [batch, max_hits] boolean (True means padded)

    Args:
        reconstructed_particle: Reconstructed particle predictions.
        particles_data: Original particle data.
        padded_mask: Boolean mask for padded hits (True for padded).
        loss_type: "MSE" or "L1".

    Returns:
        Dictionary containing reconstruction losses for each particle property:
        {'z': loss_z, 'eta': loss_eta, 'phi': loss_phi, 'pt': loss_pt}
    """
    device = reconstructed_particle.device
    loss_function = None
    if loss_type == "MSE":
        loss_function = F.mse_loss
    elif loss_type == "L1":
        loss_function = F.l1_loss
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}' in reconstruction_loss")

    # Check if there are any valid (non-padded) hits
    # padded_mask is 1 for padded hits, so we need to invert it
    non_padded_mask = ~padded_mask.bool()

    # Filter out hits without associated particles (pT = 0.0 indicates orphan hits)
    # Only hits with true pT > 0 participate in any reconstruction component (z, eta, phi, pT)
    has_particle_mask = particles_data[:, :, 3] > 0.0  # True particle present
    valid_hits_mask = non_padded_mask & has_particle_mask

    if torch.sum(valid_hits_mask) == 0:
        # Return zero losses if no valid hits with particles
        return {
            "z": torch.tensor(0.0, device=device),
            "eta": torch.tensor(0.0, device=device),
            "phi": torch.tensor(0.0, device=device),
            "pt": torch.tensor(0.0, device=device),
        }
    # Safe clamp for pT inversion to avoid inf / NaN
    pred_pt = reconstructed_particle[:, :, 4][valid_hits_mask]
    pred_pt_clamped = torch.clamp(pred_pt, min=1e-6)

    # Compute component losses
    loss_z_part = loss_function(
        reconstructed_particle[:, :, 0][valid_hits_mask],
        particles_data[:, :, 0][valid_hits_mask],
        reduction="sum",
    )
    loss_eta_part = loss_function(
        reconstructed_particle[:, :, 1][valid_hits_mask],
        particles_data[:, :, 1][valid_hits_mask],
        reduction="sum",
    )
    loss_sin_phi_part = loss_function(
        reconstructed_particle[:, :, 2][valid_hits_mask],
        torch.sin(particles_data[:, :, 2][valid_hits_mask]),
        reduction="sum",
    )
    loss_cos_phi_part = loss_function(
        reconstructed_particle[:, :, 3][valid_hits_mask],
        torch.cos(particles_data[:, :, 2][valid_hits_mask]),
        reduction="sum",
    )
    loss_phi_part = loss_sin_phi_part + loss_cos_phi_part
    loss_pt_part = loss_function(
        (1.0 / pred_pt_clamped),
        (1.0 / particles_data[:, :, 3][valid_hits_mask]),
        reduction="mean",
    )

    # Build a dictionary to hold the losses
    rec_loss = {
        "z": loss_z_part,
        "eta": loss_eta_part,
        "phi": loss_phi_part,
        "pt": loss_pt_part,
    }

    return rec_loss


def hit_classification_loss(
    seed_hit_scores: torch.Tensor,
    particles_data: torch.Tensor,
    padded_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the binary cross-entropy loss for hit classification.

    This method classifies hits as seed hits (associated with a particle) or non-seed hits
    (not associated with any particle). It uses binary cross-entropy loss to train the model
    to distinguish between these two classes.

    Args:
        seed_hit_scores: Predicted seed hit scores (probabilities in [0,1]) [batch_size, max_hits]
            (typically taken from the 5th component of the regressed parameters).
        particles_data: Original particle data [batch_size, max_hits, 4]
        padded_mask: Boolean mask for padded hits (True for padded)

    Returns:
        Binary cross-entropy loss scalar tensor
    """
    device = seed_hit_scores.device
    # Check if there are any valid (non-padded) hits
    # padded_mask is 1 for padded hits, so we need to invert it
    non_padded_mask = ~padded_mask.bool()

    if torch.sum(non_padded_mask) == 0:
        # Return zero loss if no valid hits
        return torch.tensor(0.0, device=device)

    # Extract valid (non-padded) data
    valid_scores = seed_hit_scores[non_padded_mask]
    # Inputs are already sigmoid probabilities; clamp for numerical stability
    valid_probs = torch.clamp(valid_scores, min=1e-7, max=1 - 1e-7)
    valid_particles = particles_data[non_padded_mask]

    # Create target labels: 1.0 for hits with particles (seed hits), 0.0 for orphan hits
    # pT > 0 indicates valid particle association
    has_particle_mask = valid_particles[:, 3] > 0.0
    target_labels = has_particle_mask.float()

    # Class-balanced weights to mitigate imbalance between orphan (0) and seed (1) hits
    n_total = int(target_labels.numel())
    n_pos = int(torch.sum(target_labels).item())
    n_neg = n_total - n_pos

    if n_pos > 0 and n_neg > 0:
        # Inverse-frequency weights normalized so each class contributes ~ equally
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        sample_weights = torch.where(
            target_labels > 0.5,
            torch.full_like(target_labels, w_pos),
            torch.full_like(target_labels, w_neg),
        )
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, weight=sample_weights, reduction="sum")
    else:
        # Fallback to unweighted if a class is absent to avoid instability
        hit_bce_loss = F.binary_cross_entropy(valid_probs, target_labels, reduction="sum")

    return hit_bce_loss
