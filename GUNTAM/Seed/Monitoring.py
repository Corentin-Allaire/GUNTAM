from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from GUNTAM.Seed.MonitoringPlot import (
    visualize_attention_map,
    plot_attention_score_distribution,
    create_seeding_performance_plots,
    create_particle_reconstruction_comparison_plots,
    create_efficiency_vs_truth_param_plots,
    create_seeds_per_particle_vs_truth_param_plots,
    create_2d_efficiency_heatmaps,
)

SeedErrors = Dict[str, List[float]]
SeedMetrics = List[Dict[str, Any]]
BinSummary = Dict[str, Any]
EventData = Tuple[
    np.ndarray,  # event_hits
    np.ndarray,  # event_particles
    np.ndarray,  # event_reconstructed
    Sequence[Any],  # event_pairs
    Sequence[Any],  # event_seeds
    np.ndarray,  # event_ID
    Optional[np.ndarray],  # event_map
    np.ndarray,  # event_mask
]


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
    """
    Monitor and visualize seeding and reconstruction performance across events/bins.
    - Computes:
        - Per-bin particle grouping and seed→particle associations (≥ `min_common_hits`).
        - Best seed selection per particle (prefer pure, then most common hits).
        - Global seeding and pure-seeding efficiencies and bin-wise statistics.
        - Seed-resolution metrics for `z`, `eta`, `phi`, and `pt`.
        - Bin complexity trends and optional attention-map diagnostics.
    - Plotting: if `save_plots=True`, summary and comparison figures are saved.

    Configuration:
    - `event_idx_list`, `bin_idx_list`: focus indices for detailed printing/visualization.
    - `full_print`: enable per-hit diagnostics in `analyze_event_bins`.
    - `min_common_hits`: hits threshold to accept a seed→particle association.
    - `min_truth_hits`: unique truth hits (across bins) required to count a particle.
    - `truth_r_tol`: radial tolerance used to deduplicate hits across bins.
    - `save_plots`: toggle creation of output plots.

    Input expectations for `seeding_performance(...)`:
    - All event-level arrays/lists share the same number of events.
    - Arrays (`hits_test`, `particles_test`, `ID_test`, `padding_mask_hit_test`)
        have identical bin dimensions per event.
    - `padding_mask_hit_test[event][bin]` marks padded hits; valid hits are
        those where the mask is False.
    """

    def __init__(
        self,
        event_idx_list: Optional[Sequence[int]] = None,
        bin_idx_list: Optional[Sequence[int]] = None,
        full_print: bool = False,
        save_plots: bool = True,
        min_common_hits: int = 3,
        min_truth_hits: int = 3,
        truth_r_tol: float = 1e-3,
    ) -> None:
        """
        Initialize the PerformanceMonitor with configuration only.

        Args:
            event_idx_list: Optional list of event indices to analyze in detail
            bin_idx_list: Optional list of bin indices to analyze in detail
            full_print: Enable detailed per-hit/seed diagnostics in analysis
            save_plots: Whether to save performance plots in `seeding_performance`
            min_common_hits: Minimum hits in common to accept seed→particle association
            min_truth_hits: Minimum unique truth hits per particle (global across bins)
            truth_r_tol: Radial tolerance to deduplicate hits across bins
        """
        self.event_idx_list = event_idx_list or []
        self.bin_idx_list = bin_idx_list or []
        self.full_print = full_print
        self.save_plots = save_plots
        self.min_common_hits = min_common_hits
        self.min_truth_hits = min_truth_hits
        self.truth_r_tol = truth_r_tol

    def _get_event(
        self,
        hits_test: np.ndarray,
        particles_test: np.ndarray,
        ID_test: np.ndarray,
        padding_mask_hit_test: np.ndarray,
        seeds_test: Sequence[Sequence[Any]],
        reconstructed_parameters: Sequence[Sequence[Any]],
        all_pairs_test: Sequence[Sequence[Any]],
        attention_maps: Optional[Sequence[Sequence[Any]]],
        event_idx: int,
    ) -> EventData:
        """
        Retrieve and package one event into an EventData tuple.

        Args:
            event_idx: Index of the event to retrieve.

        Returns:
            EventData: (
                event_hits, event_particles, event_reconstructed,
                event_pairs, event_seeds, event_ID, event_map, event_mask
            )
        """
        test_size = len(hits_test)
        if event_idx < 0 or event_idx >= test_size:
            raise IndexError(f"Event index {event_idx} is out of range (0 to {test_size - 1})")

        event_hits: np.ndarray = np.asarray(hits_test[event_idx])
        event_particles: np.ndarray = np.asarray(particles_test[event_idx])
        event_reconstructed: np.ndarray = np.asarray(reconstructed_parameters[event_idx])
        event_pairs: Sequence[Any] = all_pairs_test[event_idx]
        event_seeds: Sequence[Any] = seeds_test[event_idx]
        event_ID: np.ndarray = np.asarray(ID_test[event_idx])
        event_mask: np.ndarray = np.asarray(padding_mask_hit_test[event_idx])
        event_map: Optional[np.ndarray] = np.asarray(attention_maps[event_idx]) if attention_maps is not None else None

        return (
            event_hits,
            event_particles,
            event_reconstructed,
            event_pairs,
            event_seeds,
            event_ID,
            event_map,
            event_mask,
        )

    def _get_bin(
        self,
        event_data: EventData,
        bin_idx: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Sequence[Any],
        Optional[np.ndarray],
    ]:
        """
        Retrieve per-bin slices from an EventData tuple.

        Args:
            event_data: EventData produced by `_get_event`.
            bin_idx: Index of the bin to retrieve.

        Returns:
            Tuple of (bin_hits, bin_particles, bin_IDs, bin_reconstructed,
            bin_pairs, bin_seed, bin_map).
        """
        (
            event_hits,
            event_particles,
            event_reconstructed,
            event_pairs,
            event_seeds,
            event_ID,
            event_map,
            event_mask,
        ) = event_data

        if bin_idx < 0 or bin_idx >= event_hits.shape[0]:
            raise IndexError(f"Bin index {bin_idx} is out of range (0 to {event_hits.shape[0] - 1})")

        bin_mask = ~event_mask[bin_idx].astype(bool)

        bin_hits = event_hits[bin_idx][bin_mask]
        bin_particles = event_particles[bin_idx][bin_mask]
        bin_reconstructed = event_reconstructed[bin_idx][bin_mask]
        bin_pairs = event_pairs[bin_idx]
        bin_seed = event_seeds[bin_idx]
        bin_IDs = event_ID[bin_idx][bin_mask]
        bin_map = event_map[bin_idx] if event_map is not None else None

        return (
            bin_hits,
            bin_particles,
            bin_IDs,
            bin_reconstructed,
            bin_pairs,
            bin_seed,
            bin_map,
        )

    def _validate_inputs(
        self,
        hits_test: np.ndarray,
        particles_test: np.ndarray,
        ID_test: np.ndarray,
        padding_mask_hit_test: np.ndarray,
        seeds_test: Sequence[Sequence[Any]],
        reconstructed_parameters: Sequence[Sequence[Any]],
        all_pairs_test: Sequence[Sequence[Any]],
        attention_maps: Optional[Sequence[Sequence[Any]]],
    ) -> None:
        """Validate shapes and per-event bin counts for inputs.

        Ensures: same number of events across all inputs; same number of bins
        for tensor-like arrays; and consistent per-event bin lengths for
        nested-list inputs.
        """
        test_size = len(hits_test)
        if (
            len(particles_test) != test_size
            or len(ID_test) != test_size
            or len(padding_mask_hit_test) != test_size
            or len(seeds_test) != test_size
            or len(reconstructed_parameters) != test_size
            or len(all_pairs_test) != test_size
            or (attention_maps is not None and len(attention_maps) != test_size)
        ):
            raise ValueError(f"All inputs must have the same number of events ({test_size})")

        expected_bins = hits_test.shape[1] if test_size > 0 else 0
        if (
            expected_bins == 0
            or particles_test.shape[1] != expected_bins
            or ID_test.shape[1] != expected_bins
            or padding_mask_hit_test.shape[1] != expected_bins
        ):
            raise ValueError("hits/particles/ID/mask must have the same number of bins per event")

        for ev in range(test_size):
            if (
                len(seeds_test[ev]) != expected_bins
                or len(reconstructed_parameters[ev]) != expected_bins
                or len(all_pairs_test[ev]) != expected_bins
                or (attention_maps is not None and len(attention_maps[ev]) != expected_bins)
            ):
                raise ValueError(f"Event {ev}: list inputs must have {expected_bins} bins")

    def analyze_event_bins(
        self,
        event_idx: int,
        bin_idx: int,
        hits: np.ndarray,
        particles: np.ndarray,
        reco_params: np.ndarray,
        seeds: Sequence[Any],
        pairs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        attention_map: np.ndarray,
    ) -> None:
        """Detailed, optional analysis for a specific event/bin.

        Prints summary statistics, per-hit diagnostics (when `full_print=True`),
        and attention visualization/statistics.

        Args:
            event_idx: Event index (int)
            bin_idx: Bin index within event (int)
            hits: Array of valid hits for the bin, shape `[n_hits, 5]`
            particles: Array of truth parameters aligned to hits, shape `[n_hits, 4]`
            reco_params: Array of reconstructed parameters aligned to hits, shape `[n_hits, 4 or 5]`
            seeds: List of seeds `(hit_indices, parameters)` for the bin
            pairs: Tuple `(pairs1, pairs2, targets)` over valid hits
            attention_map: Array `[n_hits, n_hits]` attention weights for the bin
        """

        print(f"Number of valid hits in event {event_idx} bin {bin_idx}: {len(hits)}")

        # Count orphan hits (hits without particle associations)
        orphan_mask = particles[:, 3] <= 0.0  # pT <= 0 indicates orphan hit
        n_orphan_hits = np.sum(orphan_mask)
        n_particle_hits = len(hits) - n_orphan_hits

        print(f"  - Hits with particle associations: {n_particle_hits}")
        print(f"  - Orphan hits (detector noise): {n_orphan_hits}")

        # If full_print is enabled, print hit-by-hit information for de purpose
        if self.full_print:
            print("\nHIT-BY-HIT ANALYSIS:")
            print("=" * 120)
            for i in range(len(hits)):
                hit = hits[i]
                particle = particles[i]
                hit_params = reco_params[i]
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

                print(
                    f"  Reconstructed (vz, eta, phi, pT): "
                    f"[{hit_params[0]:8.4f}, {hit_params[1]:6.3f}, {hit_params[2]:6.3f}, {hit_params[3]:8.4f}]"
                )
                if len(hit_params) > 4:
                    print(f" Is_seed score {hit_params[4]}")
                # Calculate reconstruction errors
                error_vz = hit_params[0] - particle[0]
                error_eta = hit_params[1] - particle[1]
                error_phi = angular_difference(hit_params[2], particle[2])
                error_pt = hit_params[3] - particle[3]
                print(
                    f"  Reconstruction Errors (Δvz, Δeta, Δphi, ΔpT): "
                    f"[{error_vz:8.4f}, {error_eta:6.3f}, {error_phi:6.3f}, {error_pt:8.4f}]"
                )

            print(f"   Number of seeds reconstructed: {len(seeds)}")
            for seed_idx, s in enumerate(seeds):
                print(f"   Seed {seed_idx + 1}: {len(s[0])} hits {s[0]}")
                print(f"     Parameters: z={s[1][0]:.2f}, eta={s[1][1]:.3f}, phi={s[1][2]:.3f}, pT={s[1][3]:.2f}")

        pair_info = {}
        pairs1, pairs2, targets = pairs

        pairs1 = _to_numpy(pairs1)
        pairs2 = _to_numpy(pairs2)
        targets = _to_numpy(targets)

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
            max_hits=100,
        )
        print("\nAttention statistics:")
        print(f"  Min attention: {np.min(attention_map):.4f}")
        print(f"  Max attention: {np.max(attention_map):.4f}")
        print(f"  Mean attention: {np.mean(attention_map):.4f}")
        print(f"  Std attention: {np.std(attention_map):.4f}")

        # Plot attention score distribution for good vs bad pairs
        plot_attention_score_distribution(
            attention_map,
            pair_info,
            event_idx,
            bin_idx,
            save_path=f"attention_score_distribution_event{event_idx}_bin{bin_idx}.png",
        )

        print("=" * 120)

    def analyze_events(
        self,
        hits_test: np.ndarray,
        particles_test: np.ndarray,
        ID_test: np.ndarray,
        padding_mask_hit_test: np.ndarray,
        seeds_test: Sequence[Sequence[Any]],
        reconstructed_parameters: Sequence[Sequence[Any]],
        all_pairs_test: Sequence[Sequence[Any]],
        attention_maps: Sequence[Sequence[Any]],
    ) -> None:
        """
        Optional function letting us run the event analysis independently of the seeding performance function.
        In the futur this could be use to minimise the ammount of attention maps to be stored during inference.

        Args:
            hits_test, particles_test, ID_test, padding_mask_hit_test: arrays with per-event
            seeds_test, reconstructed_parameters, all_pairs_test, attention_maps: nested lists `[events][bins]`

        Returns:
            None
        """
        # Validate inputs and derive test sizes
        self._validate_inputs(
            hits_test,
            particles_test,
            ID_test,
            padding_mask_hit_test,
            seeds_test,
            reconstructed_parameters,
            all_pairs_test,
            attention_maps,
        )

        test_size = len(hits_test)
        print(f"\nAnalyzing {len(self.event_idx_list)} events for monitoring...")

        for event_idx in self.event_idx_list:
            if event_idx < 0 or event_idx >= test_size:
                raise IndexError(f"Event index {event_idx} is out of range (0 to {test_size - 1})")

            # Load event data via helper
            event = self._get_event(
                hits_test,
                particles_test,
                ID_test,
                padding_mask_hit_test,
                seeds_test,
                reconstructed_parameters,
                all_pairs_test,
                attention_maps,
                event_idx,
            )

            # === Loop over bins ===
            for bin_idx in range(event[0].shape[0]):
                hits, particles, IDs, reconstructed_params, pair, seeds, attn_map = self._get_bin(event, bin_idx)
                # If in the event/bin selection lists run the detailed analysis
                if attn_map is not None and event_idx in self.event_idx_list and bin_idx in self.bin_idx_list:
                    self.analyze_event_bins(
                        event_idx,
                        bin_idx,
                        hits,
                        particles,
                        reconstructed_params,
                        seeds,
                        pair,
                        attn_map,
                    )

    def seeding_performance(
        self,
        hits_test: np.ndarray,
        particles_test: np.ndarray,
        ID_test: np.ndarray,
        padding_mask_hit_test: np.ndarray,
        seeds_test: Sequence[Sequence[Any]],
        reconstructed_parameters: Sequence[Sequence[Any]],
        all_pairs_test: Sequence[Sequence[Any]],
        attention_maps: Optional[Sequence[Sequence[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate seeding performance by analyzing the relationship between reconstructed seeds and true particles.

        Steps:
            1. Build particle database per bin (true hits, params, reco params)
            2. Associate seeds to particles (≥ min_common_hits hits in common)
            3. Select best seed per particle (most hits, then closest params)
            4. Compute efficiencies and seed resolution
            5. Aggregate results and optionally create plots

        Args:
            hits_test, particles_test, ID_test, padding_mask_hit_test: arrays with per-event bins
            seeds_test, reconstructed_parameters, all_pairs_test, attention_maps: nested lists `[events][bins]`

        Returns:
            Dictionary containing seeding performance metrics
        """
        # Validate inputs and derive test sizes
        self._validate_inputs(
            hits_test,
            particles_test,
            ID_test,
            padding_mask_hit_test,
            seeds_test,
            reconstructed_parameters,
            all_pairs_test,
            attention_maps,
        )

        test_size = len(hits_test)
        print(f"\nEvaluating seeding performance across {test_size} events...")
        print(f"Using min_common_hits = {self.min_common_hits}")
        print(
            "Requiring at least "
            f"{self.min_truth_hits} unique truth hits per particle (global); "
            f"dedup by radius with tol={self.truth_r_tol}"
        )

        # Global accumulators
        total_seeds = 0
        eligible_particles: List[Dict[str, Any]] = []  # unified list with flags: had_seed, had_pure_seed

        seed_errors: SeedErrors = {"z": [], "eta": [], "phi": [], "pt": []}
        all_seed_metrics: SeedMetrics = []
        bin_summaries: List[BinSummary] = []

        # === Loop over events ===
        for event_idx in range(test_size):

            event_particle_best_seeds: Dict[int, Dict[str, Any]] = {}  # track per event best seed for each particle
            event_particle_bins: Dict[int, Dict[str, Any]] = (
                {}
            )  # track bins where each particle appears, seed status, and hit stats

            # Load event data via helper
            event = self._get_event(
                hits_test,
                particles_test,
                ID_test,
                padding_mask_hit_test,
                seeds_test,
                reconstructed_parameters,
                all_pairs_test,
                attention_maps,
                event_idx,
            )

            # === Loop over bins ===
            for bin_idx in range(event[0].shape[0]):

                hits, particles, IDs, reconstructed_params, pair, seeds, attn_map = self._get_bin(event, bin_idx)
                # If in the event/bin selection lists run the detailed analysis
                if attn_map is not None and event_idx in self.event_idx_list and bin_idx in self.bin_idx_list:
                    self.analyze_event_bins(
                        event_idx,
                        bin_idx,
                        hits,
                        particles,
                        reconstructed_params,
                        seeds,
                        pair,
                        attn_map,
                    )

                total_seeds += len(seeds)

                # --- Build bin_particles dict ---
                bin_particles = self._build_bin_particles(particles, IDs)

                # --- Step 1 & 2: associate and select best seeds ---
                seeded_particle = self._associate_seeds_to_particles(seeds, bin_particles, IDs, self.min_common_hits)
                bin_particles = self._select_best_seed_for_particles(seeded_particle, bin_particles)

                # --- Collect stats ---
                bin_total_particles = len(bin_particles)
                # Count particles with any associated seed in this bin
                bin_particles_with_seeds = sum(1 for pid, pdata in bin_particles.items() if "best_seed_idx" in pdata)
                # Count particles with a pure associated seed in this bin
                bin_particles_with_pure_seeds = sum(
                    1 for pid, pdata in bin_particles.items() if pdata.get("is_pure", False)
                )

                self._update_event_particle_bins(
                    event_particle_bins,
                    bin_particles,
                    hits,
                    bin_idx,
                    self.truth_r_tol,
                )

                # Update per-event best seed selection directly from seeding_performance
                for pid, pdata in bin_particles.items():
                    self._update_best_seed_selection(event_particle_best_seeds, pid, pdata)

                # Store resolution data (per bin best seeds)
                self._process_bin_best_associations(
                    bin_particles,
                    event_idx,
                    bin_idx,
                    seed_errors,
                    all_seed_metrics,
                )

                # Bin summary
                bin_summaries.append(
                    {
                        "event_idx": event_idx,
                        "bin_idx": bin_idx,
                        "n_particles": bin_total_particles,
                        "n_seeds": len(seeds),
                        "particles_with_seeds": bin_particles_with_seeds,
                        "seeding_efficiency": (
                            bin_particles_with_seeds / bin_total_particles if bin_total_particles > 0 else 0.0
                        ),
                        "particles_with_pure_seeds": bin_particles_with_pure_seeds,
                        "pure_seeding_efficiency": (
                            bin_particles_with_pure_seeds / bin_total_particles if bin_total_particles > 0 else 0.0
                        ),
                    }
                )

            # After processing all bins in this event, determine global particle status
            self._finalize_event_particle_status(
                event_particle_bins,
                event_idx,
                self.min_truth_hits,
                eligible_particles,
            )

        # === Aggregate results ===
        total_unique_particles = len(eligible_particles)
        particles_with_seeds_count = sum(1 for p in eligible_particles if p.get("had_seed", False))
        particles_with_pure_seeds_count = sum(1 for p in eligible_particles if p.get("had_pure_seed", False))
        seeding_efficiency = (
            (particles_with_seeds_count / total_unique_particles) if total_unique_particles > 0 else 0.0
        )
        pure_seeding_efficiency = (
            (particles_with_pure_seeds_count / total_unique_particles) if total_unique_particles > 0 else 0.0
        )

        resolution_metrics = {}
        for param, errors in seed_errors.items():
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

        performance_results = {
            "efficiency_metrics": {
                "total_particles": total_unique_particles,
                "total_seeds": total_seeds,
                "particles_with_seeds": particles_with_seeds_count,
                "seeding_efficiency": seeding_efficiency,
                "particles_with_pure_seeds": particles_with_pure_seeds_count,
                "pure_seeding_efficiency": pure_seeding_efficiency,
                "total_bins_processed": len(bin_summaries),
            },
            "resolution_metrics": resolution_metrics,
            "bin_statistics": {
                "mean_efficiency": np.mean([b["seeding_efficiency"] for b in bin_summaries]),
                "std_efficiency": np.std([b["seeding_efficiency"] for b in bin_summaries]),
                "mean_pure_efficiency": np.mean([b["pure_seeding_efficiency"] for b in bin_summaries]),
                "std_pure_efficiency": np.std([b["pure_seeding_efficiency"] for b in bin_summaries]),
            },
            "bin_complexity_analysis": self._analyze_bin_complexity(bin_summaries),
        }

        # Print
        self._print_seeding_performance_results(performance_results)

        # Plots
        if self.save_plots:
            create_seeding_performance_plots(performance_results, all_seed_metrics, seed_errors, bin_summaries)
            if eligible_particles:
                create_particle_reconstruction_comparison_plots(eligible_particles)
            try:
                self._annotate_deltaR_min(eligible_particles)
                create_efficiency_vs_truth_param_plots(eligible_particles)
                create_seeds_per_particle_vs_truth_param_plots(eligible_particles)
                create_2d_efficiency_heatmaps(eligible_particles)
            except Exception as e:
                print(f"Error creating efficiency-vs-parameter plots: {e}")

        # Build optional split lists for backward compatibility in return payload
        with_list = [p for p in eligible_particles if p.get("had_seed", False)]
        without_list = [p for p in eligible_particles if not p.get("had_seed", False)]

        return {
            "performance_results": performance_results,
            "seed_metrics": all_seed_metrics,
            "bin_summaries": bin_summaries,
            "eligible_particles": eligible_particles,
            "particles_with_seeds": with_list,
            "particles_without_seeds": without_list,
        }

    def _build_bin_particles(
        self,
        particles: np.ndarray,
        IDs: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """Build per-bin particle dictionary from valid hits and IDs.

        Returns a dict mapping particle_id -> {
            'hit_indices': list[int],
            'true_params': np.ndarray,  # parameters from first occurrence
        }

        Notes:
        - Skips Orphan hits (ID < 0 or pT <= 0)
        - Assumes reconstructed_params is aligned with hits
        """
        # Normalize shapes: expect flat per-hit IDs aligned with particles
        IDs = np.asarray(IDs).reshape(-1)
        particles = np.asarray(particles)

        # Apply mask to select non orphan hits
        pt = particles[:, 3]
        mask = (IDs > 0) & (pt > 0.0)

        # Indices of selected hits and corresponding IDs
        idx = np.nonzero(mask)[0]
        mask_ids = IDs[mask]

        # Group valid indices by particle ID using sorting + unique
        order = np.argsort(mask_ids, kind="stable")
        ids_sorted = mask_ids[order]
        idx_sorted = idx[order]

        unique_ids, _, counts = np.unique(ids_sorted, return_index=True, return_counts=True)

        bin_particles = {}
        # Split the sorted indices into per-particle groups
        split_points = np.cumsum(counts)[:-1]
        groups = np.split(idx_sorted, split_points)

        for pid, grp in zip(unique_ids.tolist(), groups):
            if grp.size == 0:
                continue
            first = int(grp[0])
            bin_particles[int(pid)] = {
                "hit_indices": grp.tolist(),
                "true_params": particles[first].copy(),
            }

        return bin_particles

    def _associate_seeds_to_particles(
        self,
        bin_seeds: Sequence[Tuple[Sequence[int], Sequence[float]]],
        bin_particles: Dict[int, Dict[str, Any]],
        IDs: np.ndarray,
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
            IDs: array of particle IDs per hit for this bin (aligned with hits)
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

        # Precompute available particle IDs in this bin
        bin_particle_ids = set(bin_particles.keys())

        for seed_idx, seed in enumerate(bin_seeds):
            seed_hit_indices = set(seed[0])

            # Candidate particle IDs are those present among this seed's hits
            seed_ids = set(np.unique(IDs[list(seed_hit_indices)]))
            candidate_particle_ids = {pid for pid in seed_ids if pid in bin_particle_ids}

            for particle_id in candidate_particle_ids:
                pdata = bin_particles[particle_id]
                particle_hit_indices = set(pdata["hit_indices"])
                n_hits_common = len(seed_hit_indices & particle_hit_indices)

                if n_hits_common >= min_common_hits:
                    # Seed is pure for this particle if ALL its hits belong to this particle
                    is_pure_for_particle = seed_hit_indices.issubset(particle_hit_indices)
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
        For each particle, select the best associated seed.

        Mutates bin_particles in place by attaching the best association info to
        each particle entry that has at least one candidate. The following keys
        are added to bin_particles[pid]:
          - 'best_seed_idx'
          - 'n_common_hits'
          - 'param_distance'
          - 'seed_params'
          - 'is_pure'

        Args:
            seeded_particle: dict mapping particle_id -> list of seed dicts
                            (output of associate_seeds_to_particles)
            bin_particles: dict mapping particle_id -> particle info (with "true_params")

        Returns:
            bin_particles enriched with best seed association info per particle
        """

        for particle_id, seed_list in seeded_particle.items():
            true_params = bin_particles[particle_id]["true_params"]

            # Selection priority: prefer pure seeds first, then by n_common_hits
            candidates = []
            for s in seed_list:
                seed_params = s["seed_params"]
                param_diff = np.array(seed_params[:3]) - np.array(true_params[:3])
                param_diff[0] /= 100.0  # scale z0
                param_diff[2] = angular_difference(seed_params[2], true_params[2])
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
                chosen = max(pure_candidates, key=lambda x: x["n_common_hits"])  # no distance tie-breaker
            else:
                chosen = max(candidates, key=lambda x: x["n_common_hits"])  # no distance tie-breaker

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
                }

            # Accumulate unique r keys for this particle in this bin
            hit_indices = pdata.get("hit_indices", [])
            if len(hit_indices) > 0:
                tx_ty = hits[hit_indices, :2]
                rs = np.sqrt(np.sum(np.square(tx_ty), axis=1))
                r_keys = np.round(rs / truth_r_tol)
                ur = event_particle_bins[particle_id]["unique_r_keys"]
                for k in r_keys:
                    ur.add(int(k))

            # Record bin index and seed presence flags
            event_particle_bins[particle_id]["bins"].append(bin_idx)
            has_seed = pdata.get("best_seed_idx") is not None
            event_particle_bins[particle_id]["has_seed_in_bins"].append(has_seed)
            event_particle_bins[particle_id]["has_pure_seed_in_bins"].append(has_seed and pdata.get("is_pure", False))

            # Record number of seeds associated to this particle in this bin
            event_particle_bins[particle_id]["n_seeds_in_bins"].append(pdata.get("nb_seed", 0))

    def _update_best_seed_selection(
        self,
        event_particle_best_seeds: Dict[int, Dict[str, Any]],
        particle_id: int,
        data: Dict[str, Any],
    ) -> None:
        """Update per-event best seed for a particle.

        Selection rule:
        - Prefer pure seeds first
        - If same purity, prefer higher n_common_hits
        - If still tied, keep current selection (no distance tie-break)
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

    def _process_bin_best_associations(
        self,
        best_associations: Dict[int, Dict[str, Any]],
        event_idx: int,
        bin_idx: int,
        seed_errors: SeedErrors,
        all_seed_metrics: SeedMetrics,
    ) -> None:
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
            errors = seed_params[0:4] - true_params
            errors[2] = angular_difference(seed_params[2], true_params[2])

            seed_errors["z"].append(errors[0])
            seed_errors["eta"].append(errors[1])
            seed_errors["phi"].append(errors[2])
            seed_errors["pt"].append(errors[3])

            all_seed_metrics.append(
                {
                    "event_idx": event_idx,
                    "bin_idx": bin_idx,
                    "particle_id": particle_id,
                    "seed_idx": data["best_seed_idx"],
                    "n_hits_common": data["n_common_hits"],
                    "is_pure": data.get("is_pure", False),
                    "param_distance": data["param_distance"],
                    "true_params": true_params,
                    "seed_params": seed_params.copy(),
                    "errors": errors.copy(),
                }
            )

    def _finalize_event_particle_status(
        self,
        event_particle_bins: Dict[int, Dict[str, Any]],
        event_idx: int,
        min_truth_hits: int,
        eligible_particles: List[Dict[str, Any]],
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

            info = {
                "particle_id": particle_id,
                "event_idx": event_idx,
                "bins_appeared": info.get("bins", []),
                "true_params": true_params.copy(),
                "n_hits": n_truth_hits,
                "had_seed": has_seed_globally,
                "had_pure_seed": has_pure_seed_globally,
                "n_seeds": n_seeds_total,
            }

            eligible_particles.append(info)

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

            etas = np.array([p["true_params"][1] for p in plist], dtype=float)
            phis = np.array([p["true_params"][2] for p in plist], dtype=float)

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

    def _analyze_bin_complexity(self, bin_summaries: Sequence[BinSummary]) -> Dict[str, Any]:
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
        print(
            f"   Average particles per bin: {eff_metrics['total_particles'] / eff_metrics['total_bins_processed']:.1f}"
        )
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
        print("     Detailed particle comparison plots: particle_reconstruction_comparison.png")

        bin_stats = results["bin_statistics"]
        print("\n BIN-WISE STATISTICS:")
        print(
            "   Mean seeding efficiency per bin: "
            f"{bin_stats['mean_efficiency']:.1%} ± {bin_stats['std_efficiency']:.1%}"
        )
        print(
            "   Mean pure seeding efficiency per bin: "
            f"{bin_stats['mean_pure_efficiency']:.1%} ± {bin_stats['std_pure_efficiency']:.1%}"
        )

        # Print bin complexity analysis
        if "bin_complexity_analysis" in results:
            self._print_bin_complexity_analysis(results["bin_complexity_analysis"])

        print("\n  SEED RESOLUTION (Best seed per particle):")
        resolution = results["resolution_metrics"]
        print(
            f"{'Parameter':<10} {'Mean Error':<12} {'Std Error':<12} {'RMS Error':<12} {'Median':<10} {'N Seeds':<10}"
        )
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
        print(
            "     Mean particles per bin: "
            f"{overall['mean_particles_per_bin']:.1f} ± {overall['std_particles_per_bin']:.1f}"
        )
        print(f"     Mean seeds per bin: {overall['mean_seeds_per_bin']:.1f} ± {overall['std_seeds_per_bin']:.1f}")
        print(f"     Mean seeds per particle: {overall['mean_seeds_per_particle']:.2f}")

        # Correlations
        corr = complexity_analysis["correlations"]
        print("\n     Correlations with seeding efficiency:")
        print(f"     Particles count: {corr['particles_vs_efficiency']:+.3f}")
        print(f"     Seeds count: {corr['seeds_vs_efficiency']:+.3f}")
        print(f"     Seeds/particles ratio: {corr['particle_seed_ratio_vs_efficiency']:+.3f}")

        print("\n     Detailed bin complexity analysis available in plots (bin_complexity_analysis.png)")
