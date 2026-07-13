import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Transformer.BinTensor import global_bin_torch, neighbor_bin_torch, no_bin_torch, margin_bin_torch
import GUNTAM.Seed.Reconstruction as Reconstruction


class SeedReconstructionModel(nn.Module):
    """
    Full model for seed reconstruction from a list of hits, using a transformer architecture.
    The goal of this model is to be written to ONNX and run efficiently in C++ for inference.
    This is not meant for use in training, but rather as a standalone inference module.
    It implements:
        - Binning of input hits into a fixed-size sequence (with padding and masking).
        - Creation of adjacency matrix using a transformer encoder and a matching attention layer.
        - Reconstruction of seeds by selecting top-k connections for each hit based on attention scores.
        - To be added : Seed parameters regression and classification

    Attributes:
        - transformer (TransformerEncoder): Transformer encoder operating on embedded hits.
        - fourier_encoding (FourierPositionalEncoding): Fourier-based positional encoder for hit coordinates.
        - embedding_projection (nn.Linear): Linear layer projecting encoded features to `dim_embedding`.
        - matching_attention (MultiHeadAttention): Attention module producing matching scores and weights.
        - cfg (SeedConfig): Full architecture configuration.
        - device_acc (torch.device): Device on which the model's parameters are allocated.

    Args:
        - transformer_config (TransformerConfig): Architecture configuration object.
        - device_acc (torch.device, optional): Device to run the model on. Defaults to cpu.
    """

    def __init__(
        self,
        transformer_config: SeedConfig = SeedConfig(),
        transformer: SeedTransformer = SeedTransformer(),
        device_acc: torch.device = torch.device("cpu"),
        width: int = 5,
        max_seed_length: int = 3,
        radial_separation_constraint: bool = True,
        min_delta_rho_mm: float = 5.0,
        raw_chain_length: int = 5,
    ) -> None:
        super(SeedReconstructionModel, self).__init__()

        if raw_chain_length < 3:
            raise ValueError(
                f"raw_chain_length must be >= 3, got {raw_chain_length}. "
                "A shorter raw chain can never produce a valid 3-hit seed."
            )

        self.cfg = transformer_config
        self.device_acc = device_acc
        self.transformer = transformer
        self.width = width
        self.max_seed_length = max_seed_length
        self.radial_separation_constraint = radial_separation_constraint
        self.min_delta_rho_mm = min_delta_rho_mm
        self.raw_chain_length = raw_chain_length

    def bin_and_pad(self, hits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Bin the input hits into a fixed-size sequence and create a corresponding padding mask.
        Args:
            - hits (Tensor): Input hits tensor of shape [N, 3] with columns (x, y, z).
        Returns:
            - binned_hits (Tensor): Binned and padded hits of shape [num_bins, max_hit_input, 7]
              with columns (x, y, z, r, phi, eta, orig_idx).
            - padding_mask (Tensor): Padding mask of shape [num_bins, max_hit_input, 1];
              True where the slot is padding, False where a valid hit is present.
        """
        N = hits.shape[0]
        device = hits.device
        dtype = hits.dtype

        x, y, z = hits[:, 0], hits[:, 1], hits[:, 2]
        # Compute derived coordinates from (x, y, z)
        R = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        rho = torch.sqrt(x**2 + y**2 + z**2)
        cos_theta = z / rho
        eta = 0.5 * torch.log((1.0 + cos_theta) / (1.0 - cos_theta))

        orig_idx = torch.arange(N, device=device, dtype=dtype)
        # Build augmented hit matrix with columns (x, y, z, r, phi, eta, orig_idx)
        hits_matrix = torch.stack([x, y, z, R, phi, eta, orig_idx], dim=1)  # [N, 7]

        # Sort hits by R+rho ascending so that hits within each bin are radially ordered
        sort_order = torch.argsort(R + rho)
        hits_matrix = hits_matrix[sort_order]

        bin_width = self.cfg.preprocessing_config.bin_width
        max_hits = self.cfg.preprocessing_config.max_hit_input
        strategy = self.cfg.preprocessing_config.binning_strategy
        bin_margin = self.cfg.preprocessing_config.binning_margin
        phi_range = (-math.pi, math.pi)

        phi_val = hits_matrix[:, 4]  # [N] — phi in r-sorted order

        # Perform the binning based on the configured strategy:
        if strategy == "no_bin":
            bins_t, num_bins = no_bin_torch(phi_val)
        elif strategy == "global":
            bins_t, num_bins = global_bin_torch(phi_val, bin_width, phi_range)
        elif strategy == "neighbor":
            bins_t, num_bins = neighbor_bin_torch(phi_val, bin_width, phi_range)
        elif strategy == "margin":
            bins_t, num_bins = margin_bin_torch(phi_val, bin_width, bin_margin, phi_range)
        else:
            raise ValueError(f"Unknown binning_strategy: {strategy!r}")

        pos_idx = torch.arange(N, device=device, dtype=torch.long)
        b0 = bins_t[:, 0]
        b1 = bins_t[:, 1]
        b2 = bins_t[:, 2]

        if strategy in ("no_bin", "global"):
            bins_u = b1
            hit_pos_u = pos_idx
            is_secondary = torch.zeros(N, device=device, dtype=torch.long)
        elif strategy == "neighbor":
            bins_u = torch.cat([b0, b1, b2])
            hit_pos_u = pos_idx.repeat(3)
            is_secondary = (b1[hit_pos_u] != bins_u).long()  # 0 = primary, 1 = neighbor
        else:  # margin: hits near bin edges get an extra neighbor assignment → dedup needed
            pairs = torch.stack([torch.cat([b0, b1, b2]), pos_idx.repeat(3)], dim=1)  # [3N, 2]
            pairs = torch.unique(pairs, dim=0)  # sorts lexicographically, removing duplicates
            bins_u = pairs[:, 0]
            hit_pos_u = pairs[:, 1]
            is_secondary = (b1[hit_pos_u] != bins_u).long()  # 0 = primary, 1 = neighbor

        # Sort by (bin, is_secondary, hit_pos): primaries fill slots before neighbors on overflow;
        # bins_u is non-decreasing after this step.
        order = torch.argsort(bins_u * (2 * N) + is_secondary * N + hit_pos_u)
        bins_u = bins_u[order]
        hit_pos_u = hit_pos_u[order]

        # Compute the position of each hit within its assigned bin
        M = bins_u.shape[0]
        _is_new_u = torch.cat([bins_u.new_ones(1, dtype=torch.bool), bins_u[1:] != bins_u[:-1]])
        _bin_rank_u = torch.cumsum(_is_new_u.long(), dim=0) - 1  # 0-based bin index per element
        _first_starts_u = torch.arange(M, device=device, dtype=torch.long)[_is_new_u]  # start pos of each bin
        offset_in_bin = torch.arange(M, device=device, dtype=torch.long) - _first_starts_u[_bin_rank_u]
        valid = offset_in_bin < max_hits

        # Keep only the hits that fit within the max_hits limit per bin
        bins_v = bins_u[valid]
        hit_pos_v = hit_pos_u[valid]

        # Re-sort survivors by (bin, hit_pos) to restore the order within each bin
        reorder = torch.argsort(bins_v * N + hit_pos_v)
        bins_v = bins_v[reorder]
        hit_pos_v = hit_pos_v[reorder]

        # Compute the position of each hit within its assigned bin again after filtering and reordering
        M_v = bins_v.shape[0]
        _is_new_v = torch.cat([bins_v.new_ones(1, dtype=torch.bool), bins_v[1:] != bins_v[:-1]])
        _bin_rank_v = torch.cumsum(_is_new_v.long(), dim=0) - 1
        _first_starts_v = torch.arange(M_v, device=device, dtype=torch.long)[_is_new_v]
        offset_v = torch.arange(M_v, device=device, dtype=torch.long) - _first_starts_v[_bin_rank_v]

        binned = torch.zeros(num_bins, max_hits, 7, device=device, dtype=dtype)
        mask = torch.ones(num_bins, max_hits, 1, device=device, dtype=torch.bool)
        binned[bins_v, offset_v] = hits_matrix[hit_pos_v]
        mask[bins_v, offset_v, 0] = False

        return binned, mask

    def reconstruct_seed_triplets(
        self,
        triplets: Tensor,
        valid_mask: Tensor,
        att_threshold: float = 0.2,
        beam_width: int = 5,
        max_chain_length: int = 3,
        backward: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Reconstruct 3-hit seed triplets from the sparse edge tensor produced by the transformer,
        using the batched beam search algorithm from Reconstruction.py.

        Args:
            - triplets (Tensor): Shape [B, N, width, 3] — sparse edge tensor from forward(),
              columns (source_idx, target_idx, score).
            - valid_mask (Tensor): Shape [B, N] — True for valid (non-padding) hits.
            - att_threshold (float): Minimum attention score to consider an edge (default: 0.2).
            - score_threshold (float): Minimum per-hit score to use a hit as a chain source (default: 0.0).
            - beam_width (int): Number of beams per starting hit (default: 5).
            - max_chain_length (int): Maximum number of hits per seed chain (default: 3).
            - backward (bool): If True, extend chains to smaller indices (default: False).
        Returns:
            Tuple of:
              - chains     [B, N, max_chain_length]: hit indices per seed; -1 for invalid slots.
              - params     [B, N, 5]: seed parameters (all zeros).
              - scores     [B, N]:   best average edge score (-inf if no valid chain).
        """

        return Reconstruction.batched_beam_search_seed_reconstruction(
            triplets.float(),
            valid_mask,
            att_threshold=att_threshold,
            max_chain_length=max_chain_length,
            beam_width=beam_width,
            backward=backward,
        )

    def forward(
        self,
        hits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the full seed-reconstruction model.
        Args:
            - hits (Tensor): Raw flat hit tensor of shape [N, 3] with columns (x, y, z).
        Returns:
            - seed_triplets (Tensor): Shape [S, max_seed_length] — the S seeds that were
              successfully reconstructed, each row containing original hit IDs.
            - scores (Tensor): Shape [S] — score for each reconstructed seed.
        """
        binned_hits, padding_mask = self.bin_and_pad(hits)
        raw_len = self.raw_chain_length if self.radial_separation_constraint else self.max_seed_length
        # padding_mask is [B, N, 1]; the transformer expects a 2D key-padding mask [B, N].
        _, triplets = self.transformer(binned_hits[..., :6], padding_mask.squeeze(-1), self.width)

        valid_mask = (~padding_mask.bool()).squeeze(-1)  # [B, N_bin]
        chains, _, scores = self.reconstruct_seed_triplets(triplets, valid_mask, max_chain_length=raw_len)  # [B, N_bin, SL]

        if self.radial_separation_constraint:
            rho_bin_slot_space = torch.sqrt((binned_hits[..., :3] ** 2).sum(dim=-1))  # [B, N_bin]
            chains = Reconstruction.apply_radial_separation_filter(
                chains, rho_bin_slot_space, self.min_delta_rho_mm, self.max_seed_length
            )

        # Map bin-local indices → original hit IDs
        bin_nb, nb_max_hit = valid_mask.shape
        seed_nb = chains.shape[2]
        orig_idx_matrix = binned_hits[..., 6].long()  # [bin_nb, N_bin]

        # Create a mask for chain entry that need to be empty (no more hits) and clamp indices to valid range for indexing
        mask = chains >= 0  # [bin_nb, N_bin, seed_nb]
        slots_clamped = chains.clamp(0, nb_max_hit - 1)  # [bin_nb, N_bin, seed_nb]

        orig_expanded = orig_idx_matrix.unsqueeze(-1).expand(bin_nb, nb_max_hit, seed_nb)
        chains_orig = torch.gather(orig_expanded, 1, slots_clamped)  # [bin_nb, nb_max_hit, seed_nb]

        # Fill invalid hits slots with -1 (padding)
        chains_orig = chains_orig.masked_fill(~mask, -1)

        # Keep only valid seeds (beam search already excluded padding) then deduplicate
        return self._dedup_seeds(chains_orig.reshape(-1, seed_nb), scores.reshape(-1))

    def _dedup_seeds(self, chains_flat: Tensor, scores_flat: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Collapse duplicate seed rows (identical hit-ID tuples) to a single row each, keeping one
        associated score per unique seed. Split into its own method so the two tie-breaking
        strategies below can be unit-tested without needing a full transformer/binning pipeline.

        Args:
            - chains_flat (Tensor): [S, seed_nb] hit-ID rows (possibly containing duplicates and
              -1-padded invalid rows).
            - scores_flat (Tensor): [S] score per row, aligned with `chains_flat`.
        Returns:
            - unique_chains (Tensor): [U, seed_nb] deduplicated hit-ID rows.
            - unique_scores (Tensor): [U] score kept for each unique row.
        """
        has_seed = chains_flat[:, 0] >= 0
        unique_chains, inverse = torch.unique(chains_flat[has_seed], return_inverse=True, dim=0)
        scores_flat = scores_flat[has_seed]

        if self.radial_separation_constraint:
            # Duplicates can now arise from the same starting hit's raw chain filtering down to an
            # identical triple across two bin instances (margin/neighbor binning overlap) with
            # different raw scores per instance — keep the best one, not an arbitrary
            # first-occurrence pick.
            unique_scores = torch.full(
                (unique_chains.shape[0],), float("-inf"), device=scores_flat.device, dtype=scores_flat.dtype
            )
            unique_scores = unique_scores.scatter_reduce(0, inverse, scores_flat, reduce="amax", include_self=True)
        else:
            # Pre-existing behavior (first occurrence wins) — untouched when the flag is off.
            perm = torch.arange(inverse.shape[0], device=inverse.device)
            first = inverse.flip(0).new_empty(unique_chains.shape[0])
            first[inverse.flip(0)] = perm.flip(0)
            unique_scores = scores_flat[first]
        return unique_chains, unique_scores

    def export_onnx(
        self,
        path: str,
        example_hits: Tensor | None = None,
    ) -> None:
        """
        Export the model to an ONNX file.
        Args:
            - path (str): File path to save the ONNX model (.onnx).
            - example_hits (Tensor | None): Representative hits tensor [N, 3] with columns (x, y, z).
              If None, a small synthetic example is built from the config as fallback.
        Uses self.width and self.max_seed_length (set at construction time) as graph constants.
        """
        if example_hits is None:
            example_hits = torch.zeros(32, 3, dtype=torch.float32, device="cpu")

        example_hits = example_hits.float().cpu()

        was_training = self.training
        original_device = next(self.parameters()).device

        self.eval()
        self.to("cpu")

        try:
            torch.onnx.export(
                self,
                (example_hits,),
                path,
                input_names=["hits"],
                output_names=["seeds", "seed_scores"],
                dynamic_axes={
                    "hits": {0: "num_hits"},
                    "seeds": {0: "num_seeds"},
                    "seed_scores": {0: "num_seeds"},
                },
                opset_version=17,
            )
            print(f"Model exported to ONNX at {path}")
        finally:
            self.to(original_device)
            if was_training:
                self.train()
