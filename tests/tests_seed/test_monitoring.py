import numpy as np
import pytest

from GUNTAM.Seed.Monitoring import PerformanceMonitor


def make_synthetic_inputs(num_events=1, num_bins=1, hits_per_bin=6):
    """Generate pseudo inputs matching `seeding_performance` expectations.

    Shapes:
    - hits_test: [E, B, H, 5]
    - particles_test: [E, P, 4]
    - ID_test: [E, B, H]
    - seeds_test: nested list [E][B] of (hit_indices, params)
    - reconstructed_parameters: nested list [E][B] aligned to valid hits
    - all_pairs_test: nested list [E][B] of (pairs1, pairs2, targets)
    - attention_maps: nested list [E][B] of [H_valid, H_valid]
    """
    E, B, H = num_events, num_bins, hits_per_bin

    # Initialize arrays
    hits_test = np.zeros((E, B, H, 5), dtype=float)
    particles_test = np.zeros((E, 1, 7), dtype=float)  # 7 params: d0, z0, phi, eta, pT, q, m
    ID_test = np.full((E, B, H), -1, dtype=int)  # -1 for orphan by default

    # Construct one simple particle with 3 hits in bin 0 of event 0
    # True params: [d0, z0, phi, eta, pT, q, m]
    true_params = np.array([1.0, 10.0, 1.0, 0.5, 5.0, 1.0, 0.0], dtype=float)
    particle_id = 0
    particle_hit_indices = [0, 1, 2]

    e, b = 0, 0
    for hi in particle_hit_indices:
        # Hits coordinates: simple tx,ty forming distinct radii; tz, phi, eta placeholders
        hits_test[e, b, hi, :] = np.array([hi + 1.0, hi + 2.0, 0.0, true_params[2], true_params[3]], dtype=float)
        particles_test[e, particle_id, :] = true_params
        ID_test[e, b, hi] = particle_id

    # Remaining hits are orphans (-1 id, pt<=0)
    for hi in range(3, H):
        hits_test[e, b, hi, :] = np.array([0.1 * hi, 0.2 * hi, 0.0, 0.0, 0.0], dtype=float)
        # No particle entry for orphans
        ID_test[e, b, hi] = -1

    # Seeds: one pure seed matching the particle hits
    seed_params = true_params.copy()
    seeds_test = [[[(particle_hit_indices, seed_params)]]]  # nested [E][B]

    # Reconstructed parameters aligned to valid hits (copy true params + is_seed score)
    reco_bin = np.zeros((H, 7), dtype=float)  # 7 params + dummy score
    for hi in range(H):
        # Copy truth for simplicity; last column as a dummy score
        reco_bin[hi, :7] = true_params if hi in particle_hit_indices else 0.0
        # Optionally, add a dummy score as 8th column if needed elsewhere
    reconstructed_parameters = [[reco_bin]]

    # Pair info and attention maps (not used unless detailed analysis enabled)
    pairs1 = np.arange(H, dtype=int)
    pairs2 = np.arange(H, dtype=int)
    targets = np.zeros(H, dtype=int)
    all_pairs_test = [[(pairs1, pairs2, targets)]]
    attention_maps = [[np.zeros((H, H), dtype=float)]]

    return (
        hits_test,
        particles_test,
        ID_test,
        seeds_test,
        reconstructed_parameters,
        all_pairs_test,
        attention_maps,
    )

 
def make_multi_event_multi_bin_setup():
    """Create a setup with:
    - Event 1: 5 particles, 3 hits each, present in 4 bins
    - Event 2: 2 particles, 3 hits each, present in 3 bins (bin 3 empty)

    Uses a common bin count of 4 and H=32 hits per bin.
    Seeds are pure per particle in the bins where they appear.
    """
    E, B, H = 2, 4, 32

    hits_test = np.zeros((E, B, H, 5), dtype=float)
    max_particles = 5
    particles_test = np.zeros((E, max_particles, 7), dtype=float)  # 7 params: d0, z0, phi, eta, pT, q, m
    ID_test = np.full((E, B, H), -1, dtype=int)
    padding_mask_hit_test = np.zeros((E, B, H), dtype=bool)

    seeds_test = [[[] for _ in range(B)] for _ in range(E)]
    reconstructed_parameters = [[np.zeros((H, 7), dtype=float) for _ in range(B)] for _ in range(E)]
    all_pairs_test = [[(np.arange(H), np.arange(H), np.zeros(H, dtype=int)) for _ in range(B)] for _ in range(E)]
    attention_maps = [[np.zeros((H, H), dtype=float) for _ in range(B)] for _ in range(E)]

    # Helper to place a particle's 3 hits at distinct indices
    def place_particle(e, b, pid, base_idx, true_params):
        idxs = [base_idx, base_idx + 1, base_idx + 2]
        for k, hi in enumerate(idxs):
            hits_test[e, b, hi, :] = np.array(
                [
                    10 * pid + k + 1.0,
                    10 * pid + k + 2.0,
                    0.0,
                    true_params[2],  # phi
                    true_params[3],  # eta
                ]
            )
            particles_test[e, pid, :] = true_params
            ID_test[e, b, hi] = pid
            reconstructed_parameters[e][b][hi, :7] = true_params
            # Optionally, add a dummy score as 8th column if needed elsewhere
        # Pure seed over these hits
        seeds_test[e][b].append((idxs, true_params.copy()))

    # Event 1: 5 particles appear in all 4 bins
    for pid in range(5):
        # [d0, z0, phi, eta, pT, q, m]
        true = np.array([5.0 * pid, 0.1 * pid, 0.2 * pid, 2.0 + pid, 1.0 + pid, 1.0, 0.0], dtype=float)
        for b in range(4):
            base = pid * 3  # keep indices consistent across bins
            place_particle(0, b, pid, base, true)

    # Event 2: 2 particles appear in bins 0,1,2; bin 3 remains empty
    for pid in range(2):
        true = np.array([7.0 * pid, 0.2 * pid, 0.3 * pid, 3.0 + pid, 2.0 + pid, -1.0, 0.0], dtype=float)
        for b in range(3):
            base = pid * 3
            place_particle(1, b, pid, base, true)

    return (
        hits_test,
        particles_test,
        ID_test,
        padding_mask_hit_test,
        seeds_test,
        reconstructed_parameters,
        all_pairs_test,
        attention_maps,
    )


def test_seeding_performance_basic():
    """Basic sanity check: one pure seed for one eligible particle."""
    (
        hits_test,
        particles_test,
        ID_test,
        seeds_test,
        reconstructed_parameters,
        all_pairs_test,
        attention_maps,
    ) = make_synthetic_inputs(num_events=1, num_bins=1, hits_per_bin=6)
    
    monitor = PerformanceMonitor(
        save_plots=False,
        min_common_hits=3,
        min_truth_hits=3,
        truth_r_tol=1e-3,
    )
    # Process all events (usually one) using per-event inputs
    for ev in range(len(hits_test)):
        monitor.bin_seeding_performance(ev, hits_test[ev], particles_test[ev], ID_test[ev], seeds_test[ev])
    # Call detailed per-bin analysis for event 0 bin 0 (sanity: no exceptions)
    ev0, b0 = 0, 0
    # Build per-hit particle array expected by analyse_bin_performance: [vz, eta, phi, pT]
    H = hits_test.shape[2]
    particles_per_hit = np.zeros((H, 4), dtype=float)
    for hi in range(H):
        pid = ID_test[0, 0, hi]
        if pid >= 0:
            p = particles_test[0, pid]
            particles_per_hit[hi, 0] = p[1]  # vz / z0
            particles_per_hit[hi, 1] = p[3]  # eta
            particles_per_hit[hi, 2] = p[2]  # phi
            particles_per_hit[hi, 3] = p[4]  # pT

    pairs = all_pairs_test[0][0]
    if isinstance(pairs, tuple) and len(pairs) == 3:
        pairs_np = np.vstack(pairs).T
    else:
        pairs_np = np.asarray(pairs)

    monitor.analyse_bin_performance(
        ev0,
        b0,
        hits_test[0, 0],
        particles_per_hit,
        reconstructed_parameters[0][0],
        seeds_test[0][0],
        pairs_np,
        attention_maps[0][0],
    )
    monitor.performance_analysis()

    # Build performance and resolution dicts from monitor internal state
    total_particles = len(monitor.eligible_particles)
    particles_with_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_seed", False))
    particles_with_pure_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_pure_seed", False))
    perf = {
        "total_particles": total_particles,
        "total_seeds": monitor.total_seeds,
        "particles_with_seeds": particles_with_seeds,
        "seeding_efficiency": (particles_with_seeds / total_particles) if total_particles > 0 else 0.0,
        "particles_with_pure_seeds": particles_with_pure_seeds,
        "pure_seeding_efficiency": (particles_with_pure_seeds / total_particles) if total_particles > 0 else 0.0,
        "total_bins_processed": len(monitor.bin_summaries),
    }

    resolution = {}
    for param, errors in monitor.seed_errors.items():
        if errors:
            arr = np.array(errors)
            resolution[param] = {
                "mean_error": np.mean(arr),
                "std_error": np.std(arr),
                "rms_error": np.sqrt(np.mean(arr ** 2)),
                "median_error": np.median(arr),
                "n_seeds": len(arr),
            }
        else:
            resolution[param] = {k: 0.0 for k in ["mean_error", "std_error", "rms_error", "median_error"]}
            resolution[param]["n_seeds"] = 0

    # Eligible particles: 1 (3 unique hits), one pure associated seed
    assert perf["total_particles"] == 1
    assert perf["particles_with_seeds"] == 1
    assert pytest.approx(perf["seeding_efficiency"], rel=0, abs=1e-12) == 1.0
    assert perf["particles_with_pure_seeds"] == 1
    assert pytest.approx(perf["pure_seeding_efficiency"], rel=0, abs=1e-12) == 1.0

    # Resolution computed from best seed per particle; seed equals truth => zero errors
    for k in ("z", "eta", "phi", "pt"):
        stats = resolution[k]
        assert stats["n_seeds"] >= 1
        assert pytest.approx(stats["mean_error"], abs=1e-12) == 0.0
        assert pytest.approx(stats["rms_error"], abs=1e-12) == 0.0


def test_input_validation_mismatched_bins_raises():
    """Ensure _validate_inputs catches mismatched nested list lengths."""
    (
        hits_test,
        particles_test,
        ID_test,
        _,
        _,
        _,
        _,
    ) = make_synthetic_inputs(num_events=1, num_bins=1, hits_per_bin=4)

    monitor = PerformanceMonitor(save_plots=False)

    # Pass an event with mismatched per-event seeds (no bins) and expect an IndexError
    event_seeds_mismatched = []
    for ev in range(len(hits_test)):
        with pytest.raises(IndexError):
            monitor.bin_seeding_performance(ev, hits_test[ev], particles_test[ev], ID_test[ev], event_seeds_mismatched)


def test_min_common_hits_threshold_blocks_association():
    """Seed with insufficient common hits should not associate to a particle."""
    (
        hits_test,
        particles_test,
        ID_test,
        seeds_test,
        reconstructed_parameters,
        all_pairs_test,
        attention_maps,
    ) = make_synthetic_inputs(num_events=1, num_bins=1, hits_per_bin=5)

    # Replace the pure seed with only 2 common hits (threshold is 3)
    seeds_test = [[[([0, 1], np.array([10.0, 0.5, 1.0, 5.0]))]]]

    monitor = PerformanceMonitor(save_plots=False, min_common_hits=3, min_truth_hits=3)
    for ev in range(len(hits_test)):
        monitor.bin_seeding_performance(ev, hits_test[ev], particles_test[ev], ID_test[ev], seeds_test[ev])

    # Call detailed per-bin analysis for event 0 bin 0 (sanity: no exceptions)
    ev0, b0 = 0, 0
    H = hits_test.shape[2]
    particles_per_hit = np.zeros((H, 4), dtype=float)
    for hi in range(H):
        pid = ID_test[0, 0, hi]
        if pid >= 0:
            p = particles_test[0, pid]
            particles_per_hit[hi, 0] = p[1]
            particles_per_hit[hi, 1] = p[3]
            particles_per_hit[hi, 2] = p[2]
            particles_per_hit[hi, 3] = p[4]

    pairs = all_pairs_test[0][0]
    if isinstance(pairs, tuple) and len(pairs) == 3:
        pairs_np = np.vstack(pairs).T
    else:
        pairs_np = np.asarray(pairs)

    monitor.analyse_bin_performance(
        ev0,
        b0,
        hits_test[0, 0],
        particles_per_hit,
        reconstructed_parameters[0][0],
        seeds_test[0][0],
        pairs_np,
        attention_maps[0][0],
    )
    monitor.performance_analysis()

    # Build performance dict
    total_particles = len(monitor.eligible_particles)
    particles_with_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_seed", False))
    particles_with_pure_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_pure_seed", False))
    perf = {
        "total_particles": total_particles,
        "total_seeds": monitor.total_seeds,
        "particles_with_seeds": particles_with_seeds,
        "seeding_efficiency": (particles_with_seeds / total_particles) if total_particles > 0 else 0.0,
        "particles_with_pure_seeds": particles_with_pure_seeds,
        "pure_seeding_efficiency": (particles_with_pure_seeds / total_particles) if total_particles > 0 else 0.0,
        "total_bins_processed": len(monitor.bin_summaries),
    }
    # Eligible particle exists, but should have no seed association
    assert perf["total_particles"] == 1
    assert perf["particles_with_seeds"] == 0
    assert pytest.approx(perf["seeding_efficiency"], rel=0, abs=1e-12) == 0.0


def test_multi_event_multi_bin_distribution():
    (
        hits_test,
        particles_test,
        ID_test,
        _,
        seeds_test,
        _,
        _,
        _,
    ) = make_multi_event_multi_bin_setup()

    monitor = PerformanceMonitor(save_plots=False, min_common_hits=3, min_truth_hits=3)
    # Process each event separately using bin_seeding_performance
    for ev in range(len(hits_test)):
        monitor.bin_seeding_performance(ev, hits_test[ev], particles_test[ev], ID_test[ev], seeds_test[ev])
    monitor.performance_analysis()

    total_particles = len(monitor.eligible_particles)
    particles_with_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_seed", False))
    particles_with_pure_seeds = sum(1 for p in monitor.eligible_particles if p.get("had_pure_seed", False))
    perf = {
        "total_particles": total_particles,
        "total_seeds": monitor.total_seeds,
        "particles_with_seeds": particles_with_seeds,
        "seeding_efficiency": (particles_with_seeds / total_particles) if total_particles > 0 else 0.0,
        "particles_with_pure_seeds": particles_with_pure_seeds,
        "pure_seeding_efficiency": (particles_with_pure_seeds / total_particles) if total_particles > 0 else 0.0,
        "total_bins_processed": len(monitor.bin_summaries),
    }

    efficiencies = [b["seeding_efficiency"] for b in monitor.bin_summaries] if monitor.bin_summaries else []
    bin_stats = {"mean_efficiency": np.mean(efficiencies) if efficiencies else 0.0}

    # Particles: 5 in event 1 + 2 in event 2 = 7
    assert perf["total_particles"] == 7
    # Total seeds: event1 has 5 seeds per 4 bins = 20; event2 has 2 seeds per 3 bins = 6; total 26
    assert perf["total_seeds"] == 26
    # All eligible particles have seeds (and pure seeds)
    assert perf["particles_with_seeds"] == 7
    assert perf["particles_with_pure_seeds"] == 7
    assert pytest.approx(perf["seeding_efficiency"], rel=0, abs=1e-12) == 1.0
    assert pytest.approx(perf["pure_seeding_efficiency"], rel=0, abs=1e-12) == 1.0

    # Bin count and mean efficiency: 8 bins total, 7 bins with efficiency 1.0, 1 bin with 0 particles -> efficiency 0.0
    assert perf["total_bins_processed"] == 8
    assert pytest.approx(bin_stats["mean_efficiency"], abs=1e-12) == 7 / 8
