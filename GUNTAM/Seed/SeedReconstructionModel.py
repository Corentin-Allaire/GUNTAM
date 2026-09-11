import math
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Transformer.BinTensor import global_bin_torch, neighbor_bin_torch, no_bin_torch, margin_bin_torch
from GUNTAM.Seed.SeedClassifier import SeedClassifier
from GUNTAM.Seed.Reconstruction import build_seed_features_tensor
import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.IO.OnnxRunTime_Interface import onnx_providers, run_onnx_iobinding


class SeedReconstructionModel(nn.Module):
    """
    Full model for seed reconstruction from a list of hits, using a transformer architecture.
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
        - transformer (SeedTransformer): Transformer instance.
        - classifier (SeedClassifier | None): Optional classifier instance.
        - device_acc (torch.device, optional): Device to run the model on. Defaults to cpu.
    """

    def __init__(
        self,
        transformer_config: SeedConfig = SeedConfig(),
        transformer: SeedTransformer = SeedTransformer(),
        classifier: SeedClassifier | None = None,
        device_acc: torch.device = torch.device("cpu"),
        width: int = 5,
        max_seed_length: int = 3,
        radial_separation_constraint: bool = True,
        min_delta_rho_mm: float = 5.0,
        raw_chain_length: int = 5,
    ) -> None:
        super(SeedReconstructionModel, self).__init__()

        if raw_chain_length < max_seed_length:
            raise ValueError(
                f"raw_chain_length ({raw_chain_length}) must be >= max_seed_length ({max_seed_length}). "
                "A raw chain shorter than the target seed can never produce a full-length seed."
            )
        if min_delta_rho_mm < 0:
            raise ValueError(f"min_delta_rho_mm must be >= 0, got {min_delta_rho_mm}.")

        self.cfg = transformer_config
        self.cfg.epoch_nb = 1
        self.cfg.transformer_config.embedding_mode = "MLP"
        self.device_acc = device_acc
        self.transformer = transformer
        self.width = width
        self.max_seed_length = max_seed_length
        self.radial_separation_constraint = radial_separation_constraint
        self.min_delta_rho_mm = min_delta_rho_mm
        self.raw_chain_length = raw_chain_length
        self.classifier = classifier

        self._transformer_onnx_session: ort.InferenceSession | None = None
        self._transformer_onnx_path: str | None = None
        self._classifier_onnx_session: ort.InferenceSession | None = None
        self._classifier_onnx_path: str | None = None

    def bin_and_pad(self, hits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        theta = torch.arctan2(R, z)
        eta = -torch.log(torch.tan(theta / 2))
        orig_idx = torch.arange(N, device=device, dtype=dtype)
        # Build augmented hit matrix with columns (x, y, z, r, phi, eta, orig_idx)
        hits_matrix = torch.stack([x, y, z, R, phi, eta, orig_idx], dim=1)  # [N, 7]
        flat_hits = hits_matrix.clone()

        # Sort hits by R ascending so that hits within each bin are radially ordered
        sort_order = torch.argsort(R)
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

        # Sort by (bin, is_secondary, hit_pos): primaries fill slots before neighbors on overflow.
        # Neighbor duplicates are ranked by descending hit_pos so that, on overflow, the
        # smallest-R duplicates are dropped first and the largest-R ones are kept
        secondary_rank = torch.where(is_secondary.bool(), N - 1 - hit_pos_u, hit_pos_u)
        order = torch.argsort(bins_u * (2 * N) + is_secondary * N + secondary_rank)
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

        return binned, mask, flat_hits

    def reconstruct_seed_triplets(
        self,
        binned_hits: Tensor,
        padding_mask: Tensor,
        triplets: Tensor,
        att_threshold: float = 0.2,
        beam_width: int = 5,
        max_chain_length: int = 3,
        backward: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Reconstruct 3-hit seed triplets from the sparse edge tensor produced by the transformer
        Then turn bin-local edge triplets into deduplicated seed chains expressed in original hit indices.
        Shared by `forward()` and `run_onnx_inference().
        Args:
            - binned_hits (Tensor): Binned and padded hits from `bin_and_pad`, shape [num_bins, max_hit_input, 7].
            - padding_mask (Tensor): Padding mask from `bin_and_pad`, shape [num_bins, max_hit_input, 1].
            - triplets (Tensor): Shape [B, N, width, 3] — sparse edge tensor, columns (source_idx, target_idx, score).
            - att_threshold (float): Minimum attention score to consider an edge (default: 0.2).
            - beam_width (int): Number of beams per starting hit (default: 5).
            - max_chain_length (int): Maximum number of hits per seed chain (default: 3).
            - backward (bool): If True, extend chains to smaller indices (default:
        Returns:
            - unique_chains (Tensor): Shape [S, seed_nb] — deduplicated seed chains in original hit indices.
            - best_scores (Tensor): Shape [S] — best average edge score for each unique chain.
        """
        valid_mask = (~padding_mask.bool()).squeeze(-1)  # [B, N_bin]
        chains, _, scores = Reconstruction.batched_beam_search_seed_reconstruction(
            triplets,
            valid_mask,
            att_threshold=att_threshold,
            max_chain_length=max_chain_length,
            beam_width=beam_width,
            backward=backward,
        )

        
        if self.radial_separation_constraint:
            # 3D spherical radius r3d = sqrt(x^2 + y^2 + z^2) per hit slot (intentionally includes z;
            # see apply_radial_separation_filter for why this is not the cylindrical detector rho).
            r3d_bin_slot_space = torch.sqrt((binned_hits[..., :3] ** 2).sum(dim=-1))  # [B, N_bin]
            chains = Reconstruction.apply_radial_separation_filter(
                chains, r3d_bin_slot_space, self.min_delta_rho_mm, self.max_seed_length
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
        chains_flat = chains_orig.reshape(-1, seed_nb)  # [bin_nb*N_bin, seed_nb]
        scores_flat = scores.reshape(-1)  # [bin_nb*N_bin]
        has_seed = chains_flat[:, 0] >= 0
        unique_chains, inverse = torch.unique(chains_flat[has_seed], return_inverse=True, dim=0)  # [S, seed_nb]
        scores_flat = scores_flat[has_seed]  # [num_valid_seeds]
        perm = torch.arange(inverse.shape[0], device=inverse.device)
        first = inverse.flip(0).new_empty(unique_chains.shape[0])
        first[inverse.flip(0)] = perm.flip(0)

        return unique_chains, scores_flat[first]
      
    def forward(self, hits: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the full seed-reconstruction model.
        Args:
            - hits (Tensor): Raw flat hit tensor of shape [N, 3] with columns (x, y, z).
        Returns:
            - signal: Long tensor of shape [number of classified true seeds, 3] containing the original
            input hit indices of all candidate seeds
            predicted as true seeds by the classifier. Each row corresponds to one reconstructed seed.
            - signal_scores: Float tensor of shape [number of classified true seeds] containing the classifier confidence score
            associated with each reconstructed seed. The i-th
            score corresponds to the i-th seed in `signal` and represents the predicted probability that this seed is a true seed.
        """
        binned_hits, padding_mask, flat_hits = self.bin_and_pad(hits)
        # padding_mask is [B, N, 1]; the transformer expects a 2D key-padding mask [B, N].
        _, triplets = self.transformer(binned_hits[..., :6], padding_mask.squeeze(-1), self.width)
        unique_chains, best_scores = self.reconstruct_seed_triplets(
            binned_hits, padding_mask, triplets, max_chain_length=self.max_seed_length
        )

        if self.classifier is None:
            signal = unique_chains
            signal_scores = best_scores

        else:
            seed_features = build_seed_features_tensor(
                hits_tensor=flat_hits, seed_tensor=unique_chains, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
            )
            # Flatten per-point features [N, 3, X] into per-seed features [N, 3*X]
            if seed_features.dim() == 3:
                seed_features = seed_features.reshape(seed_features.shape[0], -1)

            keep_mask, classifier_scores = self.classifier(seed_features, nb_hits_features=7)
            signal = unique_chains[keep_mask]
            signal_scores = classifier_scores[keep_mask]

        return signal, signal_scores

    def run_onnx_inference(
        self,
        hits: Tensor,
        transformer_path: str,
        classifier_path: str | None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Run the same pipeline as `forward()`, but executing the transformer (and, if a
        classifier ONNX model is provided, the classifier) through ONNX Runtime instead
        of the PyTorch modules, using `IOBinding` so tensors stay on-device.
        Binning, beam-search reconstruction and seed-feature building still
        run in PyTorch. Sessions are cached on `self` and only rebuilt when the requested
        path or device changes.
        Args:
            - hits (Tensor): Raw flat hit tensor of shape [N, 3] with columns (x, y, z).
            - transformer_path (str): Path to the exported transformer ONNX model.
            - classifier_path (str | None): Path to the exported classifier ONNX model.
                If None, every candidate seed is kept, mirroring `forward()` when
                `self.classifier` is None.
            - device (torch.device | None): Device to run ONNX Runtime on; selects the
                CUDA execution provider when available, otherwise falls back to CPU.
                Defaults to `self.device_acc`.
        Returns:
            Same as `forward()`: `(signal, signal_scores)`.
        """

        device = device if device is not None else self.device_acc
        providers = onnx_providers(device)
        ort_device_type = "cuda" if device.type == "cuda" else "cpu"
        ort_device_id = device.index if device.index is not None else 0

        if self._transformer_onnx_session is None or self._transformer_onnx_path != transformer_path:
            self._transformer_onnx_session = ort.InferenceSession(transformer_path, providers=providers)
            self._transformer_onnx_path = transformer_path

        if classifier_path is not None and (
            self._classifier_onnx_session is None or self._classifier_onnx_path != classifier_path
        ):
            self._classifier_onnx_session = ort.InferenceSession(classifier_path, providers=providers)
            self._classifier_onnx_path = classifier_path

        binned_hits, padding_mask, flat_hits = self.bin_and_pad(hits)
        # padding_mask is [B, N, 1]; the transformer expects a 2D key-padding mask [B, N].
        print(f"Shape of binned_hits: {binned_hits.shape}, padding_mask: {padding_mask.shape}")
        _, triplets = run_onnx_iobinding(
            self._transformer_onnx_session,
            [binned_hits[..., :6].float(), padding_mask.squeeze(-1)],
            ort_device_type,
            ort_device_id,
            {"width": torch.tensor(5, dtype=torch.int64)},
        )
        triplets = triplets.to(hits.device)

        unique_chains, best_scores = self.reconstruct_seed_triplets(
            binned_hits, padding_mask, triplets, max_chain_length=self.max_seed_length
        )

        if classifier_path is None:
            return unique_chains, best_scores

        seed_features = build_seed_features_tensor(
            hits_tensor=flat_hits, seed_tensor=unique_chains, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
        )
        if seed_features.dim() == 3:
            seed_features = seed_features.reshape(seed_features.shape[0], -1)

        keep_mask, classifier_scores = run_onnx_iobinding(
            self._classifier_onnx_session,
            [seed_features.float()],
            ort_device_type,
            ort_device_id,
            {"nb_hits_features": torch.tensor(7, dtype=torch.int64)},
        )
        keep_mask = keep_mask.to(hits.device).bool()
        classifier_scores = classifier_scores.to(hits.device)

        signal = unique_chains[keep_mask]
        signal_scores = classifier_scores[keep_mask]

        return signal, signal_scores
