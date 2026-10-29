import math
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config.SeedConfig import SeedConfig
from GUNTAM.Transformer.BinTensor import (
    global_bin_torch,
    neighbor_bin_torch,
    no_bin_torch,
    margin_bin_torch,
    neighbor_bin_Z_phi_torch,
)
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
        - Creation of an adjacency (edge) representation using a transformer encoder.
        - Reconstruction of seeds by running a beam search per starting hit over the edge scores.
        - Optional classification of candidate seeds via a ``SeedClassifier``.
        - To be added : Seed parameters regression.

    Attributes:
        - transformer (SeedTransformer): Transformer module producing the sparse edge (triplet) tensor.
        - classifier (SeedClassifier | None): Optional classifier instance used to filter candidate seeds.
        - cfg (SeedConfig): Full architecture configuration.
        - device_acc (torch.device): Device on which the model's parameters are allocated.

    Args:
        - transformer_config (SeedConfig): Architecture configuration object.
        - transformer (SeedTransformer): Transformer instance.
        - classifier (SeedClassifier | None): Optional classifier instance.
        - device_acc (torch.device, optional): Device to run the model on. Defaults to cpu.
        - width (int): Number of candidate edges kept per hit fed to the transformer (default: 5).
        - max_seed_length (int): Maximum number of hits per reconstructed seed (default: 3).
        - radial_separation_constraint (bool): Enable the minimum radial-separation filter (default: True).
        - min_delta_rho_mm (float): Minimum 3D radial separation between consecutive seed hits, in mm (default: 5.0).
        - raw_chain_length (int): Maximum chain length searched before trimming to ``max_seed_length`` (default: 5).
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
        elif strategy == "Z_Phi":
            z_val = hits_matrix[:, 2]  # [N] — z in r-sorted order
            r_val = hits_matrix[:, 3]  # [N] — r in r-sorted order
            bins_t, num_bins = neighbor_bin_Z_phi_torch(
                torch.stack([phi_val, z_val, r_val], dim=1),
                bin_width,
                phi_range,
                self.cfg.preprocessing_config.value_bins_eta,
                bin_margin,
            )

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

    def bin_and_pad_batch(self, events: List[Tensor]) -> Tuple[Tensor, Tensor, List[Tensor], int]:
        """
        Bin and pad several events independently, then stack their bins into a single
        batch dimension so the transformer and beam search can process all events in one call.
        Args:
            - events (List[Tensor]): One raw hit tensor of shape [N_e, 3] (columns x, y, z) per event.
        Returns:
            - binned_hits (Tensor): Shape [num_events * num_bins, max_hit_input, 7], events concatenated
              along dim 0 in the same order as `events`.
            - padding_mask (Tensor): Shape [num_events * num_bins, max_hit_input, 1], matching `binned_hits`.
            - flat_hits (List[Tensor]): Per-event flat hit tensor (as returned by `bin_and_pad`), needed to
              build seed features with event-local hit indices.
            - bins_per_event (int): Number of bins per event; every event yields the same count since it is
              only a function of the (fixed) binning configuration, not of the data.
        """
        if not events:
            raise ValueError("No events provided to bin_and_pad_batch().")

        binned_list: List[Tensor] = []
        mask_list: List[Tensor] = []
        flat_hits_list: List[Tensor] = []
        bins_per_event: int | None = None

        for event_hits in events:
            binned, mask, flat_hits = self.bin_and_pad(event_hits)
            if bins_per_event is None:
                bins_per_event = binned.shape[0]
            elif binned.shape[0] != bins_per_event:
                raise ValueError(
                    f"Inconsistent number of bins across events ({binned.shape[0]} != {bins_per_event}); "
                    "binning configuration must be identical for all events in a batch."
                )
            binned_list.append(binned)
            mask_list.append(mask)
            flat_hits_list.append(flat_hits)

        assert bins_per_event is not None  # events is non-empty, so the loop above ran at least once

        binned_hits = torch.cat(binned_list, dim=0)
        padding_mask = torch.cat(mask_list, dim=0)

        return binned_hits, padding_mask, flat_hits_list, bins_per_event

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
        Reconstruct raw seed chains from the sparse edge tensor produced by the transformer: run
        beam search per bin, then map bin-local edge triplets to original hit indices.
        Shared by `forward()` and `run_onnx_inference()`.
        Args:
            - binned_hits (Tensor): Binned and padded hits from `bin_and_pad`/`bin_and_pad_batch`,
              shape [num_bins, max_hit_input, 7] (or [num_events * num_bins, max_hit_input, 7] for a batch).
            - padding_mask (Tensor): Padding mask matching `binned_hits`, shape [num_bins, max_hit_input, 1].
            - triplets (Tensor): Shape [B, N, width, 3] — sparse edge tensor, columns (source_idx, target_idx, score).
            - att_threshold (float): Minimum attention score to consider an edge (default: 0.2).
            - beam_width (int): Number of beams per starting hit (default: 5).
            - max_chain_length (int): Maximum number of hits per seed chain (default: 3).
            - backward (bool): If True, extend chains to smaller indices (default: False).
        Returns:
            - chains_orig (Tensor): Shape [bin_nb, nb_max_hit, seed_nb], -1-padded, original hit
              indices (not yet deduplicated or split by event).
            - scores (Tensor): Shape [bin_nb, nb_max_hit], best average edge score per starting hit.
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
            chains, scores = Reconstruction.apply_radial_separation_filter(
                chains, scores, r3d_bin_slot_space, self.min_delta_rho_mm, self.max_seed_length
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

        return chains_orig, scores

    def split_by_event(self, chains_orig: Tensor, scores: Tensor, bins_per_event: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Split raw per-bin seed chains/scores into per-event chunks, undoing the concatenation
        performed by `bin_and_pad_batch()`.
        Args:
            - chains_orig (Tensor): Shape [bin_nb, nb_max_hit, seed_nb], as returned by `reconstruct_seed_triplets()`.
            - scores (Tensor): Shape [bin_nb, nb_max_hit], as returned by `reconstruct_seed_triplets()`.
            - bins_per_event (int): Number of bins per event; `bin_nb` must be a multiple of it.
        Returns:
            List of `(chains_orig_e, scores_e)` pairs, one per event, in concatenation order.
        """
        bin_nb = chains_orig.shape[0]
        if bin_nb % bins_per_event != 0:
            raise ValueError(f"bin_nb ({bin_nb}) is not a multiple of bins_per_event ({bins_per_event}).")
        num_events = bin_nb // bins_per_event

        chains_by_event = chains_orig.view(num_events, bins_per_event, *chains_orig.shape[1:])
        scores_by_event = scores.view(num_events, bins_per_event, *scores.shape[1:])

        return [(chains_by_event[e], scores_by_event[e]) for e in range(num_events)]

    def dedupe_chains(self, chains_orig: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Deduplicate bin-local seed chains already expressed in original hit indices, keeping the
        score of the first occurrence of each unique chain. Shared by single-event and per-event
        (batched) reconstruction paths.
        Args:
            - chains_orig (Tensor): Shape [bin_nb, nb_max_hit, seed_nb], -1-padded, original hit indices.
            - scores (Tensor): Shape [bin_nb, nb_max_hit], best average edge score per starting hit.
        Returns:
            - unique_chains (Tensor): Shape [S, seed_nb] — deduplicated seed chains.
            - best_scores (Tensor): Shape [S] — score of the first occurrence of each unique chain.
        """
        seed_nb = chains_orig.shape[-1]

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

    def classify_seeds(
        self,
        flat_hits_list: List[Tensor],
        per_event_chains_scores: List[Tuple[Tensor, Tensor]],
        classifier_fn,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Classify candidate seeds for several events in a single classifier call.
        Seed features must be built per event (seed indices are event-local, so each event
        needs its own `flat_hits`).
        Args:
            - flat_hits_list (List[Tensor]): Per-event flat hit tensor, as returned by `bin_and_pad_batch`.
            - per_event_chains_scores (List[Tuple[Tensor, Tensor]]): Per-event `(unique_chains, best_scores)`,
              as returned by `dedupe_chains()` after splitting with `split_by_event()`.
            - classifier_fn (Callable[[Tensor], Tuple[Tensor, Tensor]]): Takes the concatenated
              [S_total, input_shape] seed-feature tensor and returns `(keep_mask, classifier_scores)`.
        Returns:
            List of `(signal, signal_scores)` pairs, one per event, in the same order as `flat_hits_list`.
        """
        seed_features_list: List[Tensor] = []
        seed_counts: List[int] = []
        for flat_hits, (chains, _) in zip(flat_hits_list, per_event_chains_scores):
            seed_features = build_seed_features_tensor(
                hits_tensor=flat_hits, seed_tensor=chains, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
            )
            # Flatten per-point features [N, 3, X] into per-seed features [N, 3*X]
            if seed_features.dim() == 3:
                seed_features = seed_features.reshape(seed_features.shape[0], -1)
            seed_features_list.append(seed_features)
            seed_counts.append(seed_features.shape[0])

        keep_mask, classifier_scores = classifier_fn(torch.cat(seed_features_list, dim=0))

        results: List[Tuple[Tensor, Tensor]] = []
        start = 0
        for chains, _ in per_event_chains_scores:
            count = chains.shape[0]
            event_keep = keep_mask[start : start + count]
            results.append((chains[event_keep], classifier_scores[start : start + count][event_keep]))
            start += count

        return results

    def forward(self, hits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Reconstruct seeds for a single event. Thin wrapper around `forward_multi_events()`;
        see that method to batch several events into one transformer/beam-search call.
        Args:
            - hits (Tensor): Raw flat hit tensor of shape [N, 3] with columns (x, y, z).
        Returns:
            - signal: Long tensor of shape [number of classified true seeds, 3] containing the
              original input hit indices of all candidate seeds predicted as true seeds by the
              classifier. Each row corresponds to one reconstructed seed.
            - signal_scores: Float tensor of shape [number of classified true seeds] containing
              the classifier confidence score for each seed in `signal`.
        """
        return self.forward_multi_events([hits])[0]

    def forward_multi_events(self, events: List[Tensor]) -> List[Tuple[Tensor, Tensor]]:
        """
        Reconstruct seeds for several events in a single call, stacking their bins along the
        batch dimension so the transformer and beam search run once across all events.
        Args:
            - events (List[Tensor]): One raw flat hit tensor of shape [N_e, 3] (columns x, y, z)
              per event.
        Returns:
            List of `(signal, signal_scores)` pairs, one per event, in the same order as `events`;
            see `forward()` for the meaning of each pair.
        """
        binned_hits, padding_mask, flat_hits_list, bins_per_event = self.bin_and_pad_batch(events)
        # padding_mask is [B, N, 1]; the transformer expects a 2D key-padding mask [B, N].
        _, triplets = self.transformer(binned_hits[..., :6], padding_mask.squeeze(-1), self.width)
        chains_orig, scores = self.reconstruct_seed_triplets(
            binned_hits, padding_mask, triplets, max_chain_length=self.max_seed_length
        )
        # With a single event there is nothing to split: skip the (no-op) reshape.
        if len(events) == 1:
            per_event_raw = [(chains_orig, scores)]
        else:
            per_event_raw = self.split_by_event(chains_orig, scores, bins_per_event)
        per_event_chains_scores = [self.dedupe_chains(chains_e, scores_e) for chains_e, scores_e in per_event_raw]

        if self.classifier is None:
            return per_event_chains_scores

        return self.classify_seeds(
            flat_hits_list, per_event_chains_scores, lambda feats: self.classifier(feats, nb_hits_features=7)
        )

    def run_onnx_inference(
        self,
        hits: Tensor,
        transformer_path: str,
        classifier_path: str | None,
        device: torch.device | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Reconstruct seeds for a single event via ONNX Runtime. Thin wrapper around
        `run_onnx_inference_multi_events()`; see that method to batch several events into
        one ONNX call.
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
        return self.run_onnx_inference_multi_events([hits], transformer_path, classifier_path, device=device)[0]

    def run_onnx_inference_multi_events(
        self,
        events: List[Tensor],
        transformer_path: str,
        classifier_path: str | None,
        device: torch.device | None = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Run the same pipeline as `forward_multi_events()`, but executing the transformer (and,
        if a classifier ONNX model is provided, the classifier) through ONNX Runtime instead of
        the PyTorch modules, using `IOBinding` so tensors stay on-device. Binning, beam-search
        reconstruction and seed-feature building still run in PyTorch. Sessions are cached on
        `self` and only rebuilt when the requested path or device changes.
        Args:
            - events (List[Tensor]): One raw flat hit tensor of shape [N_e, 3] (columns x, y, z)
              per event.
            - transformer_path (str): Path to the exported transformer ONNX model.
            - classifier_path (str | None): Path to the exported classifier ONNX model.
                If None, every candidate seed is kept, mirroring `forward_multi_events()` when
                `self.classifier` is None.
            - device (torch.device | None): Device to run ONNX Runtime on; selects the
                CUDA execution provider when available, otherwise falls back to CPU.
                Defaults to `self.device_acc`.
        Returns:
            Same as `forward_multi_events()`: a list of `(signal, signal_scores)` pairs, one per event.
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

        binned_hits, padding_mask, flat_hits_list, bins_per_event = self.bin_and_pad_batch(events)
        # padding_mask is [B, N, 1]; the transformer expects a 2D key-padding mask [B, N].
        _, triplets = run_onnx_iobinding(
            self._transformer_onnx_session,
            [binned_hits[..., :6].float(), padding_mask.squeeze(-1)],
            ort_device_type,
            ort_device_id,
            {"width": torch.tensor(5, dtype=torch.int64)},
        )
        triplets = triplets.to(binned_hits.device)

        chains_orig, scores = self.reconstruct_seed_triplets(
            binned_hits, padding_mask, triplets, max_chain_length=self.max_seed_length
        )
        # With a single event there is nothing to split: skip the (no-op) reshape.
        if len(events) == 1:
            per_event_raw = [(chains_orig, scores)]
        else:
            per_event_raw = self.split_by_event(chains_orig, scores, bins_per_event)
        per_event_chains_scores = [self.dedupe_chains(chains_e, scores_e) for chains_e, scores_e in per_event_raw]

        if classifier_path is None:
            return per_event_chains_scores

        def _classifier_fn(seed_features: Tensor) -> Tuple[Tensor, Tensor]:
            keep_mask, classifier_scores = run_onnx_iobinding(
                self._classifier_onnx_session,
                [seed_features.float()],
                ort_device_type,
                ort_device_id,
                {"nb_hits_features": torch.tensor(7, dtype=torch.int64)},
            )
            return keep_mask.to(seed_features.device).bool(), classifier_scores.to(seed_features.device)

        return self.classify_seeds(flat_hits_list, per_event_chains_scores, _classifier_fn)
