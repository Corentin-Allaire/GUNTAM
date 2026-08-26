from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from GUNTAM.Seed.MonitoringPlot import (
    visualize_attention_map,
    create_seeding_performance_plots,
    create_particle_reconstruction_comparison_plots,
    create_efficiency_vs_truth_param_plots,
    create_fake_rate_vs_truth_param_plots,
    create_duplicate_rate_vs_truth_param_plots,
    create_2d_efficiency_heatmaps,
)


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Safely convert inputs to NumPy arrays.

    Handles CUDA/CPU torch.Tensors by detaching and moving to CPU.

    Args:
        x: Input (torch.Tensor or array-like)

    Returns:
        NumPy array
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def angular_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """Compute angular difference handling circular nature.

    Result is wrapped to [-π, π].

    Args:
        angle1: First angle(s) in radians
        angle2: Second angle(s) in radians

    Returns:
        Angular difference (angle1 - angle2) wrapped to [-π, π]
    """
    diff = angle1 - angle2
    # Wrap to [-π, π]
    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
    return diff


class PerformanceMonitor:

    def __init__(
        self,
        full_print: bool = False,
        save_plots: bool = True,
        min_common_hits: int = 3,
        min_truth_hits: int = 3,
        truth_r_tol: float = 1e-3,
    ):

        self.full_print = full_print
        self.save_plots = save_plots
        self.min_common_hits = min_common_hits
        self.min_truth_hits = min_truth_hits
        self.truth_r_tol = truth_r_tol

        self.seed_errors: dict[str, list[float]] = {
            "z": [],
            "eta": [],
            "phi": [],
            "pt": [],
        }

        self.seed_scores: dict[str, list[float]] = {"good": [], "fake": []}

        self.eligible_particles: list[dict[str, Any]] = []

        self.bin_summaries: list[dict[str, Any]] = []

        self.total_seeds = 0

    def _build_bin_particles(
        self,
        particles: np.ndarray,
        hit_to_particle: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """Build per-bin particle dictionary from valid hits and hit_to_particle mapping.

        hit_to_particle: shape [n_hits], each entry is the index of the associated particle in the event (or <0 for orphan)
        particles: shape [n_particles, ...], all particles in the event (not binned)

        Returns a dict mapping particle_id (index in event) -> {
            'hit_indices': list[int],
            'true_params': np.ndarray,  # parameters from particles[particle_id]
        }
        """
        hit_to_particle = np.asarray(hit_to_particle).reshape(-1)
        particles = np.asarray(particles)

        bin_particles = {}
        for hit_idx, particle_idx in enumerate(hit_to_particle):
            if particle_idx < 0:
                continue  # Orphan hit
            particle_idx = int(particle_idx)
            if particle_idx not in bin_particles:
                bin_particles[particle_idx] = {
                    "hit_indices": [],
                    "true_params": particles[particle_idx].copy(),
                }
            bin_particles[particle_idx]["hit_indices"].append(hit_idx)

        return bin_particles

    def _associate_seeds_to_particles(
        self,
        bin_seeds: Sequence[Tuple[Sequence[int], Sequence[float]]],
        hit_to_particle: np.ndarray,
        min_common_hits: int = 3,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Associate seeds to particles based on most hits in common.

        Args:
            bin_seeds: list of seed objects (each must have `.hits` and `.parameters`)
            bin_particles: dict mapping particle_id -> {
                "hit_indices": list[int],  # hit indices belonging to this particle
                "true_params": np.ndarray, # true particle parameters
            }
            hit_to_particle: array of particle IDs per hit for this bin (aligned with hits)
            min_common_hits: minimum number of common hits to form an association (default: 3)

        Returns:
            seeded_particle: dict mapping particle_id -> list of dicts:
                {
                    "seed_idx": int,
                    "n_common_hits": int,
                    "seed_params": np.ndarray,
                    "is_pure": bool,    # True if ALL hits of the seed belong to this particle
                }
        """
        seeded_particle: Dict[int, List[Dict[str, Any]]] = {}

        for seed_idx, seed in enumerate(bin_seeds):
            seed_hit_indices = set(seed[0])

            seed_ids = hit_to_particle[list(seed_hit_indices)]
            # Candidate particle IDs are those present among this seed's hits
            seed_unique_ids = set(np.unique(seed_ids))

            for particle_id in seed_unique_ids:
                if particle_id < 0:
                    continue

                n_hits_common = np.sum(seed_ids == particle_id)

                if n_hits_common >= min_common_hits:
                    # Seed is pure for this particle if ALL its hits belong to this particle
                    is_pure_for_particle = len(seed_unique_ids) == 1
                    seeded_particle.setdefault(particle_id, []).append(
                        {
                            "seed_idx": seed_idx,
                            "particle_id": particle_id,
                            "n_common_hits": n_hits_common,
                            "seed_params": np.array(seed[1], copy=True),
                            "is_pure": is_pure_for_particle,
                        }
                    )

        return seeded_particle

    def _select_best_seed_for_particles(
        self,
        seeded_particle: Dict[int, List[Dict[str, Any]]],
        bin_particles: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Select the best seed for each particle in a single bin.

        This function operates on per-bin data. For each particle present in the bin,
        it chooses the best associated seed (prioritizing pure seeds, then by number of common hits),
        and attaches this information to the bin's particle dictionary.
        The following keys are added to bin_particles[pid]:
            - 'best_seed_idx': index of the best seed in this bin
            - 'n_common_hits': number of hits in common with the best seed
            - 'param_distance': distance between seed and true parameters
            - 'seed_params': parameters of the best seed
            - 'is_pure': whether the best seed is pure

        Args:
                seeded_particle: dict mapping particle_id -> list of seed dicts (output of associate_seeds_to_particles)
                bin_particles: dict mapping particle_id -> particle info (with "true_params")

        Returns:
                bin_particles enriched with best seed association info per particle (for this bin only)
        """

        for particle_id, seed_list in seeded_particle.items():
            true_params = bin_particles[particle_id]["true_params"]

            # Selection priority: prefer pure seeds first, then by n_common_hits
            candidates = []
            for s in seed_list:
                seed_params = s["seed_params"]
                # Indices: 0-d0, 1-z0, 2-phi, 3-eta, 4-pT, 5-q, 6-m
                # For distance: use z0 (1), eta (3), phi (2)
                param_diff = np.array(
                    [
                        seed_params[1] - true_params[1],  # z0
                        seed_params[3] - true_params[3],  # eta
                        angular_difference(seed_params[2], true_params[2]),  # phi
                    ]
                )
                param_diff[0] /= 100.0  # scale z0
                param_distance = np.linalg.norm(param_diff)

                candidates.append(
                    {
                        "best_seed_idx": s["seed_idx"],
                        "n_common_hits": s["n_common_hits"],
                        "param_distance": param_distance,
                        "seed_params": seed_params,
                        "true_params": true_params,
                        "is_pure": s.get("is_pure", False),
                    }
                )

            # If any pure candidates exist, consider only them
            pure_candidates = [c for c in candidates if c.get("is_pure", False)]
            if pure_candidates:
                chosen = max(pure_candidates, key=lambda x: x["n_common_hits"])
            else:
                chosen = max(candidates, key=lambda x: x["n_common_hits"])

            # Enrich bin_particles with best candidate info (in-place)

            bin_particles[particle_id]["best_seed_idx"] = chosen["best_seed_idx"]
            bin_particles[particle_id]["nb_seed"] = len(seed_list)
            bin_particles[particle_id]["n_common_hits"] = chosen["n_common_hits"]
            bin_particles[particle_id]["param_distance"] = chosen["param_distance"]
            # store a copy to avoid external mutation
            bin_particles[particle_id]["seed_params"] = np.array(chosen["seed_params"], copy=True)
            bin_particles[particle_id]["is_pure"] = chosen.get("is_pure", False)

        return bin_particles

    def _update_event_particle_bins(
        self,
        event_particle_bins: Dict[int, Dict[str, Any]],
        bin_particles: Dict[int, Dict[str, Any]],
        hits: np.ndarray,
        bin_idx: int,
        truth_r_tol: float,
        n_fake_seeds_in_bin: float = 0.0,
    ) -> None:
        """Update per-event particle tracking with info from one bin.

        For each particle appearing in the current bin, this:
        - Initializes its entry in event_particle_bins if needed
        - Accumulates unique hit-radius keys (r quantized by truth_r_tol)
        - Records that the particle appeared in this bin
        - Records whether it has any associated seed in this bin
        - Records whether the selected association is pure in this bin
        - Records the number of associated seeds (and pure seeds) in this bin

        Args:
            event_particle_bins: dict mutated in place, keyed by particle_id
            bin_particles: dict produced by _build_bin_particles
            hits: numpy array of hits for this bin (unpadded)
            bin_idx: current bin index
            best_associations: mapping particle_id -> association dict for this bin
            truth_r_tol: radial tolerance used to deduplicate hits across bins
        """
        for particle_id, pdata in bin_particles.items():
            # Initialize per-particle structure if first time seen this event
            if particle_id not in event_particle_bins:
                event_particle_bins[particle_id] = {
                    "true_params": pdata["true_params"].copy(),
                    "bins": [],
                    "has_seed_in_bins": [],
                    "has_pure_seed_in_bins": [],
                    # Track unique hits across detector using r-quantization
                    "unique_r_keys": set(),
                    # Track seeds associated per bin
                    "n_seeds_in_bins": [],
                    # Track fake seeds (no particle match) per bin
                    "n_fake_seeds_in_bins": [],
                }

            # Accumulate unique r keys for this particle in this bin
            hit_indices = pdata.get("hit_indices", [])
            if len(hit_indices) > 0:
                tx_ty_tz = hits[hit_indices, :3]
                rs = np.sqrt(np.sum(np.square(tx_ty_tz), axis=1))
                r_keys = np.round(rs / truth_r_tol)
                for k in r_keys:
                    event_particle_bins[particle_id]["unique_r_keys"].add(int(k))

            # Record bin index and seed presence flags
            event_particle_bins[particle_id]["bins"].append(bin_idx)
            has_seed = pdata.get("best_seed_idx") is not None
            event_particle_bins[particle_id]["has_seed_in_bins"].append(has_seed)
            event_particle_bins[particle_id]["has_pure_seed_in_bins"].append(has_seed and pdata.get("is_pure", False))

            # Record number of seeds associated to this particle in this bin
            event_particle_bins[particle_id]["n_seeds_in_bins"].append(pdata.get("nb_seed", 0))
            # Record number of fake seeds in this bin (shared across all particles in the bin)
            event_particle_bins[particle_id]["n_fake_seeds_in_bins"].append(n_fake_seeds_in_bin)

    def _update_best_seed_selection(
        self,
        event_particle_best_seeds: Dict[int, Dict[str, Any]],
        particle_id: int,
        data: Dict[str, Any],
    ) -> None:
        """
        Update the best seed selection for each particle across all bins in an event.

        This function accumulates and compares the best seed info for each particle as bins are processed,
        ensuring that the event-level best seed is the purest and/or has the most common hits.
        If a new candidate is purer or has more common hits than the current event-level best, it replaces it.

        Selection rule:
        - Prefer pure seeds first
        - If same purity, prefer higher n_common_hits
        - If still tied, keep current selection (no distance tie-break)

        Args:
            event_particle_best_seeds: dict mapping particle_id -> best seed info across all bins
            particle_id: ID of the particle being updated
            data: best seed info from the current bin

        Returns:
            None (updates event_particle_best_seeds in place)
        """
        # Sanitize data by removing best seed identifier
        best_data = dict(data)
        best_data.pop("best_seed_idx", None)

        if particle_id not in event_particle_best_seeds:
            event_particle_best_seeds[particle_id] = best_data
            return

        cur = event_particle_best_seeds[particle_id]
        a_pure = bool(best_data.get("is_pure", False))
        b_pure = bool(cur.get("is_pure", False))
        if a_pure and not b_pure:
            event_particle_best_seeds[particle_id] = best_data
        elif a_pure == b_pure and best_data.get("n_common_hits", 0) > cur.get("n_common_hits", 0):
            event_particle_best_seeds[particle_id] = best_data

    def _process_bin_best_associations(self, best_associations: Dict[int, Dict[str, Any]]) -> None:
        """Record per-bin seed metrics for best associations.

        For each particle's best association in a bin, compute error components
        and append them into `seed_errors`, and store a metrics record into
        `all_seed_metrics` for downstream aggregation.
        """
        for particle_id, data in best_associations.items():
            if "best_seed_idx" not in data:
                continue
            seed_params = data["seed_params"]
            true_params = data["true_params"]

            # Errors for resolution metrics
            # Indices: 0-d0, 1-z0, 2-phi, 3-eta, 4-pT, 5-q, 6-m
            errors = np.zeros(4)
            errors[0] = seed_params[1] - true_params[1]  # z0
            errors[1] = seed_params[3] - true_params[3]  # eta
            errors[2] = angular_difference(seed_params[2], true_params[2])  # phi
            errors[3] = seed_params[4] - true_params[4]  # pT

            self.seed_errors["z"].append(errors[0])
            self.seed_errors["eta"].append(errors[1])
            self.seed_errors["phi"].append(errors[2])
            self.seed_errors["pt"].append(errors[3])

    def _finalize_event_particle_status(
        self,
        event_particle_bins: Dict[int, Dict[str, Any]],
        event_idx: int,
        min_truth_hits: int,
    ) -> None:
        """Finalize per-event particle status and append to a unified list.

        Applies the global eligibility rule (>= min_truth_hits unique r-deduped hits)
        and appends a per-particle info dict with seed flags to eligible_particles.
        """
        for particle_id, info in event_particle_bins.items():
            # Determine eligibility based on unique hit count across detector (dedup by r)
            n_truth_hits = int(len(info.get("unique_r_keys", set())))
            is_eligible = n_truth_hits >= min_truth_hits
            if not is_eligible:
                continue

            true_params = np.asarray(info["true_params"], dtype=float)

            # Particle has seed globally if it has seed in ANY bin
            has_seed_globally = any(bool(x) for x in info.get("has_seed_in_bins", []))
            has_pure_seed_globally = any(bool(x) for x in info.get("has_pure_seed_in_bins", []))

            # Aggregate seed counts across all bins for this particle
            n_seeds_total = int(np.sum(info.get("n_seeds_in_bins", [])))
            n_fake_seeds_total = float(np.sum(info.get("n_fake_seeds_in_bins", [])))

            info = {
                "particle_id": particle_id,
                "event_idx": event_idx,
                "bins_appeared": info.get("bins", []),
                "true_params": true_params.copy(),
                "n_hits": n_truth_hits,
                "had_seed": has_seed_globally,
                "had_pure_seed": has_pure_seed_globally,
                "n_seeds": n_seeds_total,
                "n_fake_seeds": n_fake_seeds_total,
            }

            self.eligible_particles.append(info)

    def _annotate_deltaR_min(self, eligible_particles: List[Dict[str, Any]]) -> None:
        """Compute and attach per-particle nearest-neighbor ΔR in (eta, phi).

        For each event, compute ΔR_i = min_{j != i} sqrt((Δη)^2 + (Δφ)^2), where Δφ is computed
        with angular wrapping. Mutates each particle dict in eligible_particles to add:
        - 'deltaR_min': float (np.inf if no other particle in the event)

        Notes:
        - Only particles that passed eligibility are considered for the neighborhood within each event.
        - Particles with no neighbor will have deltaR_min = np.inf (these are ignored by plotting).
        """
        if not eligible_particles:
            return

        # Group by event
        by_event: dict[int, list[dict]] = {}
        for p in eligible_particles:
            by_event.setdefault(p["event_idx"], []).append(p)

        # Compute per event
        for ev, plist in by_event.items():
            if len(plist) <= 1:
                # No neighbor available
                for p in plist:
                    p["deltaR_min"] = float("inf")
                continue

            etas = np.array([p["true_params"][3] for p in plist], dtype=float)  # index 3 = eta
            phis = np.array([p["true_params"][2] for p in plist], dtype=float)  # index 2 = phi

            # Build pairwise Δη, Δφ and ΔR
            dEta = etas[:, None] - etas[None, :]

            # Use angular_difference for φ wrapping
            def _ang_diff_matrix(a):
                # vectorized angular difference using existing helper
                # angular_difference expects two angles; we broadcast pairwise
                A = np.repeat(a[:, None], len(a), axis=1)
                B = np.repeat(a[None, :], len(a), axis=0)
                return angular_difference(A, B)

            dPhi = _ang_diff_matrix(phis)
            dR = np.sqrt(dEta**2 + dPhi**2)
            np.fill_diagonal(dR, np.inf)

            # Min ΔR per particle
            min_dR = np.min(dR, axis=1)
            for p, dr in zip(plist, min_dR):
                p["deltaR_min"] = float(dr)

    def bin_seeding_performance(
        self,
        event_idx: int,
        event_hits,
        event_particles,
        event_hit_to_particle,
        event_seeds,
    ):

        event_particle_best_seeds: Dict[int, Dict[str, Any]] = {}
        event_particle_bins: Dict[int, Dict[str, Any]] = {}

        for bin_idx in range(event_hits.shape[0]):
            hits = event_hits[bin_idx]
            particles = event_particles
            hit_to_particle = event_hit_to_particle[bin_idx]
            seeds = event_seeds[bin_idx]
            self.total_seeds += len(seeds)

            # --- Build bin_particles dict ---
            bin_particles = self._build_bin_particles(particles, hit_to_particle)

            # --- Step 1 & 2: associate and select best seeds ---
            seeded_particle = self._associate_seeds_to_particles(seeds, hit_to_particle, self.min_common_hits)
            bin_particles = self._select_best_seed_for_particles(seeded_particle, bin_particles)

            # Fake seeds: seeds that share <min_common_hits hits with any particle.
            seeded_seed_indices = {s["seed_idx"] for pl in seeded_particle.values() for s in pl}
            n_fake_seeds_in_bin = (len(seeds) - len(seeded_seed_indices)) / max(1, len(bin_particles))

            # --- Collect seed scores split by good/fake ---
            for seed_idx, seed in enumerate(seeds):
                if len(seed) >= 3:
                    category = "good" if seed_idx in seeded_seed_indices else "fake"
                    self.seed_scores[category].append(float(seed[2]))

            # --- Collect stats ---
            bin_total_particles = len(bin_particles)
            # Count particles with any associated seed in this bin
            bin_particles_with_seeds = sum(1 for pid, pdata in bin_particles.items() if "best_seed_idx" in pdata)
            # Count particles with a pure associated seed in this bin
            bin_particles_with_pure_seeds = sum(1 for pid, pdata in bin_particles.items() if pdata.get("is_pure", False))
            # Count particles with >=3 hits (excluding under-represented particles)
            bin_particles_min3 = {pid: pdata for pid, pdata in bin_particles.items() if len(pdata.get("hit_indices", [])) >= 3}
            bin_total_min3 = len(bin_particles_min3)
            bin_particles_with_seeds_min3 = sum(1 for pdata in bin_particles_min3.values() if "best_seed_idx" in pdata)
            bin_particles_with_pure_seeds_min3 = sum(1 for pdata in bin_particles_min3.values() if pdata.get("is_pure", False))

            self._update_event_particle_bins(
                event_particle_bins,
                bin_particles,
                hits,
                bin_idx,
                self.truth_r_tol,
                n_fake_seeds_in_bin,
            )

            # Update per-event best seed selection directly from seeding_performance
            for pid, pdata in bin_particles.items():
                self._update_best_seed_selection(event_particle_best_seeds, pid, pdata)

            # Store resolution data (per bin best seeds)
            self._process_bin_best_associations(
                bin_particles,
            )

            # Bin summary
            self.bin_summaries.append(
                {
                    "event_idx": event_idx,
                    "bin_idx": bin_idx,
                    "n_particles": bin_total_particles,
                    "n_seeds": len(seeds),
                    "particles_with_seeds": bin_particles_with_seeds,
                    "seeding_efficiency": (bin_particles_with_seeds / bin_total_particles if bin_total_particles > 0 else 0.0),
                    "particles_with_pure_seeds": bin_particles_with_pure_seeds,
                    "pure_seeding_efficiency": (
                        bin_particles_with_pure_seeds / bin_total_particles if bin_total_particles > 0 else 0.0
                    ),
                    "n_particles_min3hits": bin_total_min3,
                    "particles_with_seeds_min3hits": bin_particles_with_seeds_min3,
                    "seeding_efficiency_min3hits": (
                        bin_particles_with_seeds_min3 / bin_total_min3 if bin_total_min3 > 0 else 0.0
                    ),
                    "particles_with_pure_seeds_min3hits": bin_particles_with_pure_seeds_min3,
                    "pure_seeding_efficiency_min3hits": (
                        bin_particles_with_pure_seeds_min3 / bin_total_min3 if bin_total_min3 > 0 else 0.0
                    ),
                }
            )

        # After processing all bins in this event, determine global particle status
        self._finalize_event_particle_status(
            event_particle_bins,
            event_idx,
            self.min_truth_hits,
        )

    def performance_analysis(self):

        total_unique_particles = len(self.eligible_particles)
        particles_with_seeds_count = sum(1 for p in self.eligible_particles if p.get("had_seed", False))
        particles_with_pure_seeds_count = sum(1 for p in self.eligible_particles if p.get("had_pure_seed", False))
        seeding_efficiency = (particles_with_seeds_count / total_unique_particles) if total_unique_particles > 0 else 0.0
        pure_seeding_efficiency = (
            (particles_with_pure_seeds_count / total_unique_particles) if total_unique_particles > 0 else 0.0
        )

        resolution_metrics = {}
        for param, errors in self.seed_errors.items():
            if errors:
                arr = np.array(errors)
                resolution_metrics[param] = {
                    "mean_error": np.mean(arr),
                    "std_error": np.std(arr),
                    "rms_error": np.sqrt(np.mean(arr**2)),
                    "median_error": np.median(arr),
                    "n_seeds": len(arr),
                }
            else:
                resolution_metrics[param] = {k: 0.0 for k in ["mean_error", "std_error", "rms_error", "median_error"]}
                resolution_metrics[param]["n_seeds"] = 0

        total_fake_seeds = sum(p.get("n_fake_seeds", 0.0) for p in self.eligible_particles)
        total_duplicate_seeds = sum(p.get("n_seeds", 0) for p in self.eligible_particles)
        fake_rate = total_fake_seeds / total_unique_particles if total_unique_particles > 0 else 0.0
        duplicate_rate = total_duplicate_seeds / total_unique_particles if total_unique_particles > 0 else 0.0

        performance_results = {
            "efficiency_metrics": {
                "total_particles": total_unique_particles,
                "total_seeds": self.total_seeds,
                "particles_with_seeds": particles_with_seeds_count,
                "seeding_efficiency": seeding_efficiency,
                "particles_with_pure_seeds": particles_with_pure_seeds_count,
                "pure_seeding_efficiency": pure_seeding_efficiency,
                "total_bins_processed": len(self.bin_summaries),
                "fake_rate": fake_rate,
                "duplicate_rate": duplicate_rate,
            },
            "resolution_metrics": resolution_metrics,
            "bin_statistics": {
                "mean_efficiency": np.mean([b["seeding_efficiency"] for b in self.bin_summaries]),
                "std_efficiency": np.std([b["seeding_efficiency"] for b in self.bin_summaries]),
                "mean_pure_efficiency": np.mean([b["pure_seeding_efficiency"] for b in self.bin_summaries]),
                "std_pure_efficiency": np.std([b["pure_seeding_efficiency"] for b in self.bin_summaries]),
                "mean_efficiency_min3hits": np.mean([b["seeding_efficiency_min3hits"] for b in self.bin_summaries]),
                "std_efficiency_min3hits": np.std([b["seeding_efficiency_min3hits"] for b in self.bin_summaries]),
                "mean_pure_efficiency_min3hits": np.mean([b["pure_seeding_efficiency_min3hits"] for b in self.bin_summaries]),
                "std_pure_efficiency_min3hits": np.std([b["pure_seeding_efficiency_min3hits"] for b in self.bin_summaries]),
            },
            "bin_complexity_analysis": self._analyze_bin_complexity(self.bin_summaries),
            "seed_scores": self.seed_scores,
        }

        # Print
        self._print_seeding_performance_results(performance_results)

        # Plots
        if self.save_plots:
            create_seeding_performance_plots(performance_results, self.seed_errors, self.bin_summaries)
            if self.eligible_particles:
                create_particle_reconstruction_comparison_plots(self.eligible_particles)
            try:
                self._annotate_deltaR_min(self.eligible_particles)
                create_efficiency_vs_truth_param_plots(self.eligible_particles)
                create_fake_rate_vs_truth_param_plots(self.eligible_particles)
                create_duplicate_rate_vs_truth_param_plots(self.eligible_particles)
                create_2d_efficiency_heatmaps(self.eligible_particles)
            except Exception as e:
                print(f"Error creating efficiency-vs-parameter plots: {e}")

        return performance_results

    def analyse_bin_performance(
        self,
        event_idx: int,
        bin_idx: int,
        hits: np.ndarray,
        particles: np.ndarray,
        seeds: Sequence[Any],
        pairs: np.ndarray,
        attention_map: np.ndarray,
        hit_to_particle: Optional[np.ndarray] = None,
    ) -> None:
        """Detailed, optional analysis for a specific event/bin.

        Prints summary statistics, per-hit diagnostics (when `full_print=True`),
        and attention visualization/statistics.

        Args:
            event_idx: Event index (int)
            bin_idx: Bin index within event (int)
            hits: Array of valid hits for the bin, shape `[n_hits, 5]`
            particles: Array of truth parameters aligned to hits, shape `[n_hits, 4]`
            seeds: List of seeds `(hit_indices, parameters)` for the bin
            pairs: Tensor `[num_pairs, 3]` where columns are (pairs1, pairs2, targets)
            attention_map: Array `[n_hits, n_hits]` attention weights for the bin
            hit_to_particle: Optional array of particle IDs per hit (shape `[n_hits]`).
                When provided, fake seeds (< min_common_hits hits from any particle)
                are identified and up to 10 are printed with hit positions. Non-reconstructed
                particles (no associated seed) are also printed with their hits and the seed
                indices that contain those hits.
        """

        print(f"Number of valid hits in event {event_idx} bin {bin_idx}: {len(hits)}")
        print(f"Number of seeds reconstructed: {len(seeds)}")

        # --- Fake seed + non-reconstructed particle printout ---
        if hit_to_particle is not None:
            htp = np.asarray(hit_to_particle).reshape(-1)
            seeded_particle = self._associate_seeds_to_particles(seeds, htp, self.min_common_hits) if len(seeds) > 0 else {}
            seeded_seed_indices = {s["seed_idx"] for pl in seeded_particle.values() for s in pl}

            if len(seeds) > 0:
                fake_seeds = [(idx, s) for idx, s in enumerate(seeds) if idx not in seeded_seed_indices]
                print(f"\n  FAKE SEEDS ({len(fake_seeds)} / {len(seeds)} total):")
                for rank, (seed_idx, s) in enumerate(fake_seeds[:10]):
                    hit_indices = list(s[0])
                    print(f"   Fake seed {rank + 1} (seed #{seed_idx}): {len(hit_indices)} hits")
                    for i_h, h in enumerate(hit_indices):
                        if h < len(hits):
                            hx, hy, hz = hits[h, 0], hits[h, 1], hits[h, 2]
                            hr = hits[h, 3]
                            hphi = hits[h, 4]
                            heta = hits[h, 5]
                            pid = int(htp[h]) if h < len(htp) else -1
                            pid_str = f"particle {pid}" if pid >= 0 else "orphan"
                            next_h = hit_indices[i_h + 1] if i_h + 1 < len(hit_indices) else None
                            if next_h is not None and h < attention_map.shape[0] and next_h < attention_map.shape[1]:
                                attn_str = f"attn→next={attention_map[h, next_h]:.3f}"
                            else:
                                attn_str = "attn→next=  N/A"
                            print(
                                f"     hit {h:3d} | x={hx:8.2f} y={hy:8.2f} z={hz:8.2f}"
                                f" r={hr:7.2f} phi={hphi:6.3f} eta={heta:6.3f} | {attn_str} | {pid_str}"
                            )
                if len(fake_seeds) > 10:
                    print(f"   ... and {len(fake_seeds) - 10} more fake seeds (not shown)")

            # --- Non-reconstructed particle printout ---
            # Build a reverse map: seed_idx -> set of hit indices, for fast lookup
            seed_hit_sets = [set(s[0]) for s in seeds]

            unique_pids = sorted([int(p) for p in np.unique(htp) if p >= 0])
            non_reco_pids = [pid for pid in unique_pids if pid not in seeded_particle]
            non_reco_pids_min3 = [pid for pid in non_reco_pids if sum(1 for p in htp if int(p) == pid) >= 3]
            print(
                f"\n  NON-RECONSTRUCTED PARTICLES ({len(non_reco_pids_min3)} with >=3 hits"
                f" / {len(non_reco_pids)} total non-reco / {len(unique_pids)} total):"
            )
            for rank, pid in enumerate(non_reco_pids_min3[:10]):
                p_hit_indices = [i for i, p in enumerate(htp) if int(p) == pid]
                true_params_str = "N/A"
                if pid < len(particles):
                    p = particles[pid]
                    true_params_str = f"d0={p[0]:.2f} z0={p[1]:.2f} phi={p[2]:.3f} eta={p[3]:.3f} pT={p[4]:.2f}"
                print(f"   Particle {pid}: {len(p_hit_indices)} hits | {true_params_str}")
                for h in p_hit_indices:
                    if h >= len(hits):
                        continue
                    hx, hy, hz = hits[h, 0], hits[h, 1], hits[h, 2]
                    hr = hits[h, 3]
                    hphi = hits[h, 4]
                    heta = hits[h, 5]
                    seeds_with_hit = [
                        seed_idx for seed_idx, s_hits in enumerate(seed_hit_sets) if h in s_hits and len(seeds[seed_idx][0]) > 2
                    ]
                    seeds_str = f"in seeds {seeds_with_hit}" if seeds_with_hit else "in no seed"
                    print(
                        f"     hit {h:3d} | x={hx:8.2f} y={hy:8.2f} z={hz:8.2f}"
                        f" r={hr:7.2f} phi={hphi:6.3f} eta={heta:6.3f} | {seeds_str}"
                    )
            if len(non_reco_pids_min3) > 10:
                print(f"   ... and {len(non_reco_pids_min3) - 10} more non-reconstructed particles (not shown)")

        # If full_print is enabled, print hit-by-hit information for de purpose
        if self.full_print:
            print("\nHIT-BY-HIT ANALYSIS:")
            print("=" * 120)
            for i in range(len(hits)):
                hit = hits[i]
                particle = particles[i]
                is_orphan = particle[3] <= 0.0
                print(
                    f"  Coordinates (tx, ty, tz, phi, eta): "
                    f"[{hit[0]:8.4f}, {hit[1]:8.4f}, {hit[2]:8.4f}, {hit[3]:8.4f}, {hit[4]:8.4f}]"
                )

                if not is_orphan:
                    print(
                        f"  Particle (vz, eta, phi, pT): "
                        f"[{particle[0]:8.4f}, {particle[1]:6.3f}, {particle[2]:6.3f}, {particle[3]:8.4f}]"
                    )
                else:
                    print(f"  Particle: ORPHAN HIT (no associated particle, pT={particle[3]:.1f})")

            print(f"   Number of seeds reconstructed: {len(seeds)}")
            for seed_idx, s in enumerate(seeds):
                print(f"   Seed {seed_idx + 1}: {len(s[0])} hits {s[0]}")
                print(f"     Parameters: z={s[1][0]:.2f}, eta={s[1][1]:.3f}, phi={s[1][2]:.3f}, pT={s[1][3]:.2f}")

        pair_info = {}

        # Unbind pairs tensor to get individual columns
        if isinstance(pairs, torch.Tensor):
            pairs1, pairs2, targets = pairs.unbind(dim=1)
            pairs1 = _to_numpy(pairs1)
            pairs2 = _to_numpy(pairs2)
            targets = _to_numpy(targets)
        else:
            # Fallback for numpy arrays
            pairs_np = np.asarray(pairs)
            if pairs_np.ndim == 2 and pairs_np.shape[1] == 3:
                pairs1 = pairs_np[:, 0]
                pairs2 = pairs_np[:, 1]
                targets = pairs_np[:, 2]
            else:
                pairs1, pairs2, targets = pairs_np

        mask_valid = (pairs1 < len(hits)) & (pairs2 < len(hits))
        for p1, p2, t in zip(pairs1[mask_valid], pairs2[mask_valid], targets[mask_valid]):
            pair_info[(int(p1), int(p2))] = {
                "target": int(t),
                "is_good": int(t) == 1,
            }

        # Visualize attention map and print attention statistics
        print(f"\n{'=' * 60}")
        print("ATTENTION ANALYSIS")
        print(f"{'=' * 60}")
        visualize_attention_map(
            attention_map,
            pair_info,
            hits,
            event_idx,
            bin_idx,
            max_hits=600,
        )
        print("\nAttention statistics:")
        print(f"  Min attention: {np.min(attention_map):.4f}")
        print(f"  Max attention: {np.max(attention_map):.4f}")
        print(f"  Mean attention: {np.mean(attention_map):.4f}")
        print(f"  Std attention: {np.std(attention_map):.4f}")

        print("=" * 120)

    def _analyze_bin_complexity(self, bin_summaries) -> Dict[str, Any]:
        """Analyze seeding efficiency as a function of bin complexity (number of particles and seeds)."""
        if not bin_summaries:
            return {}

        # Extract bin characteristics
        n_particles = np.array([b["n_particles"] for b in bin_summaries])
        n_seeds = np.array([b["n_seeds"] for b in bin_summaries])
        efficiencies = np.array([b["seeding_efficiency"] for b in bin_summaries])

        # Create particle count bins
        particle_ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float("inf"))]
        particle_analysis = {}

        for min_p, max_p in particle_ranges:
            range_name = f"{min_p}-{max_p if max_p != float('inf') else '∞'}"
            mask = (n_particles >= min_p) & (n_particles < max_p)

            if np.any(mask):
                particle_analysis[range_name] = {
                    "n_bins": np.sum(mask),
                    "mean_particles": np.mean(n_particles[mask]),
                    "mean_seeds": np.mean(n_seeds[mask]),
                    "mean_efficiency": np.mean(efficiencies[mask]),
                    "std_efficiency": np.std(efficiencies[mask]),
                    "seeds_per_particle_ratio": np.mean(n_seeds[mask] / np.maximum(n_particles[mask], 1)),
                }

        # Create seed count bins
        seed_ranges = [(0, 5), (5, 15), (15, 30), (30, 60), (60, float("inf"))]
        seed_analysis = {}

        for min_s, max_s in seed_ranges:
            range_name = f"{min_s}-{max_s if max_s != float('inf') else '∞'}"
            mask = (n_seeds >= min_s) & (n_seeds < max_s)

            if np.any(mask):
                seed_analysis[range_name] = {
                    "n_bins": np.sum(mask),
                    "mean_particles": np.mean(n_particles[mask]),
                    "mean_seeds": np.mean(n_seeds[mask]),
                    "mean_efficiency": np.mean(efficiencies[mask]),
                    "std_efficiency": np.std(efficiencies[mask]),
                }

        # Analyze correlation between complexity and efficiency
        correlation_analysis = {
            "particles_vs_efficiency": (np.corrcoef(n_particles, efficiencies)[0, 1] if len(n_particles) > 1 else 0.0),
            "seeds_vs_efficiency": (np.corrcoef(n_seeds, efficiencies)[0, 1] if len(n_seeds) > 1 else 0.0),
            "particle_seed_ratio_vs_efficiency": (
                np.corrcoef(n_seeds / np.maximum(n_particles, 1), efficiencies)[0, 1] if len(n_particles) > 1 else 0.0
            ),
        }

        return {
            "by_particle_count": particle_analysis,
            "by_seed_count": seed_analysis,
            "correlations": correlation_analysis,
            "overall_stats": {
                "mean_particles_per_bin": np.mean(n_particles),
                "std_particles_per_bin": np.std(n_particles),
                "mean_seeds_per_bin": np.mean(n_seeds),
                "std_seeds_per_bin": np.std(n_seeds),
                "mean_seeds_per_particle": np.mean(n_seeds / np.maximum(n_particles, 1)),
            },
        }

    def _print_seeding_performance_results(self, results: Dict[str, Any]) -> None:
        """Print seeding performance results in a formatted way."""
        print("\n" + "=" * 80)
        print("SEEDING PERFORMANCE EVALUATION")
        print("=" * 80)

        eff_metrics = results["efficiency_metrics"]
        print("  EFFICIENCY OVERVIEW:")
        print(f"   Total particles analyzed: {eff_metrics['total_particles']}")
        print(f"   Total seeds reconstructed: {eff_metrics['total_seeds']}")
        print(f"   Total bins processed: {eff_metrics['total_bins_processed']}")
        print(f"   Average particles per bin: {eff_metrics['total_particles'] / eff_metrics['total_bins_processed']:.1f}")
        print(f"   Average seeds per bin: {eff_metrics['total_seeds'] / eff_metrics['total_bins_processed']:.1f}")
        print(f"   Seeds per particle ratio: {eff_metrics['total_seeds'] / eff_metrics['total_particles']:.2f}")

        print("\n  SEEDING EFFICIENCY:")
        print(
            "   Particles with associated seeds (≥3 hits): "
            f"{eff_metrics['particles_with_seeds']} ({eff_metrics['seeding_efficiency']:.1%})"
        )
        print(
            "   Particles with pure seeds (≥3 hits and all seed hits from same particle): "
            f"{eff_metrics['particles_with_pure_seeds']} ({eff_metrics['pure_seeding_efficiency']:.1%})"
        )
        print(
            "   Particles without seeds: "
            f"{eff_metrics['total_particles'] - eff_metrics['particles_with_seeds']} "
            f"({1 - eff_metrics['seeding_efficiency']:.1%})"
        )
        print(f"   Fake rate (fake seeds per particle):      {eff_metrics['fake_rate']:.4f}")
        print(f"   Duplicate rate (matched seeds per particle): {eff_metrics['duplicate_rate']:.4f}")
        print("     Detailed particle comparison plots: particle_reconstruction_comparison.png")

        bin_stats = results["bin_statistics"]
        print("\n BIN-WISE STATISTICS:")
        print("   Mean seeding efficiency per bin: " f"{bin_stats['mean_efficiency']:.1%} ± {bin_stats['std_efficiency']:.1%}")
        print(
            "   Mean pure seeding efficiency per bin: "
            f"{bin_stats['mean_pure_efficiency']:.1%} ± {bin_stats['std_pure_efficiency']:.1%}"
        )
        print(
            "   Mean seeding efficiency per bin (>=3 hits): "
            f"{bin_stats['mean_efficiency_min3hits']:.1%} ± {bin_stats['std_efficiency_min3hits']:.1%}"
        )
        print(
            "   Mean pure seeding efficiency per bin (>=3 hits): "
            f"{bin_stats['mean_pure_efficiency_min3hits']:.1%} ± {bin_stats['std_pure_efficiency_min3hits']:.1%}"
        )

        # Print bin complexity analysis
        if "bin_complexity_analysis" in results:
            self._print_bin_complexity_analysis(results["bin_complexity_analysis"])

        print("\n  SEED RESOLUTION (Best seed per particle):")
        resolution = results["resolution_metrics"]
        print(f"{'Parameter':<10} {'Mean Error':<12} {'Std Error':<12} {'RMS Error':<12} {'Median':<10} {'N Seeds':<10}")
        print("-" * 82)

        param_units = {"z": "[mm]", "eta": "", "phi": "[rad]", "pt": "[GeV]"}
        for param_name in ["z", "eta", "phi", "pt"]:
            if param_name in resolution:
                stats = resolution[param_name]
                unit = param_units.get(param_name, "")
                print(
                    f"{param_name + unit:<10} {stats['mean_error']:>11.4f} {stats['std_error']:>11.4f} "
                    f"{stats['rms_error']:>11.4f} {stats['median_error']:>9.4f} {stats['n_seeds']:>9d} "
                )

        print("=" * 82)
        print("Note: Resolution computed from best seed per particle (most hits in common, then closest parameters)")

    def _print_bin_complexity_analysis(self, complexity_analysis: Dict[str, Any]) -> None:
        """Print bin complexity analysis results."""
        print("\n  BIN COMPLEXITY ANALYSIS:")

        overall = complexity_analysis["overall_stats"]
        print("   Overall bin statistics:")
        print("     Mean particles per bin: " f"{overall['mean_particles_per_bin']:.1f} ± {overall['std_particles_per_bin']:.1f}")
        print(f"     Mean seeds per bin: {overall['mean_seeds_per_bin']:.1f} ± {overall['std_seeds_per_bin']:.1f}")
        print(f"     Mean seeds per particle: {overall['mean_seeds_per_particle']:.2f}")

        # Correlations
        corr = complexity_analysis["correlations"]
        print("\n     Correlations with seeding efficiency:")
        print(f"     Particles count: {corr['particles_vs_efficiency']:+.3f}")
        print(f"     Seeds count: {corr['seeds_vs_efficiency']:+.3f}")
        print(f"     Seeds/particles ratio: {corr['particle_seed_ratio_vs_efficiency']:+.3f}")

        print("\n     Detailed bin complexity analysis available in plots (bin_complexity_analysis.png)")
