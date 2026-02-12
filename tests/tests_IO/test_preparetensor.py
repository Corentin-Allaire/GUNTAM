import pandas as pd
import torch

from GUNTAM.IO.PrepareTensor import (
    _particle_selection,
    _build_good_pairs_tensors,
    _to_tensor,
    _add_padding,
    _create_padding_mask,
    _orphan_hit_removal,
    _bin_data,
    compute_barcode,
)
from GUNTAM.Seed.Config import SeedConfig


class TestParticleSelection:
    def test_basic_selection(self):
        # Create test data
        cfg = SeedConfig()
        cfg.eta_range = [-2.0, 2.0]
        cfg.vertex_cuts = [1.0, 100.0]
        
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0, 0],
            "particle_id": [0, 0, 1, 2],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        
        particles_batch = pd.DataFrame({
            "event_id": [0, 0, 0],
            "particle_id": [0, 1, 2],
            "eta": [1.0, 2.5, -1.0],  # particle 1 outside eta range
            "pT": [10.0, 15.0, 20.0],
            "d0": [0.1, 0.2, 0.3],
            "z0": [5.0, 10.0, 15.0],
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 1, 2],
            "bin1": [0, 0, 1, 2],
            "bin2": [0, 0, 1, 2],
        })
        
        hit_to_particle = pd.Series([0, 0, 1, 2])
        
        data_result, particles_result, bins_result, hit_result = _particle_selection(
            data_batch, particles_batch, bins, hit_to_particle, cfg
        )
        
        # Particle 1 (eta=2.5) should be filtered out, leaving 2 particles
        assert len(particles_result) == 2
        
        # After filtering, particle IDs should be remapped to sequential indices [0, 1]
        assert set(particles_result["particle_id"].values) == {0, 1}
        
        # Original particles 0 and 2 should remain (1 was filtered)
        # Check by eta values: should have eta 1.0 and -1.0, not 2.5
        assert set(particles_result["eta"].values) == {1.0, -1.0}
        
        # Hits associated with original particle 1 should be removed
        assert len(data_result) == 3

    def test_vertex_cuts(self):
        cfg = SeedConfig()
        cfg.eta_range = [-3.0, 3.0]
        cfg.vertex_cuts = [0.5, 50.0]  # Strict d0 and z0 cuts
        
        particles_batch = pd.DataFrame({
            "event_id": [0, 0],
            "particle_id": [0, 1],
            "eta": [1.0, 1.5],
            "pT": [10.0, 15.0],
            "d0": [0.3, 0.8],  # particle 1 fails d0 cut
            "z0": [10.0, 20.0],
        })
        
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0],
            "particle_id": [0, 0, 1],
            "x": [1.0, 2.0, 3.0],
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 1],
            "bin1": [0, 0, 1],
            "bin2": [0, 0, 1],
        })
        
        hit_to_particle = pd.Series([0, 0, 1])
        
        data_result, particles_result, bins_result, hit_result = _particle_selection(
            data_batch, particles_batch, bins, hit_to_particle, cfg
        )
        
        # Only particle 0 should remain (particle 1 fails d0 cut)
        assert len(particles_result) == 1
        assert particles_result["particle_id"].iloc[0] == 0


class TestBuildGoodPairsTensors:
    def test_basic_pairs(self):
        # Create test data with 2 events
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0, 1, 1],
            "particle_id": [0, 0, 1, 0, -1],  # Last hit is orphan
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 1, 0, 1],
            "bin1": [0, 0, 1, 0, 1],
            "bin2": [0, 0, 1, 0, 1],
        })
        
        hit_to_particle = pd.Series([0, 0, 1, 0, -1])
        num_bins = 2
        
        result = _build_good_pairs_tensors(data_batch, bins, hit_to_particle, num_bins)
        
        # Check shape: [num_events, num_bins, max_pairs, 3]
        assert result.shape[0] == 2  # 2 events
        assert result.shape[1] == num_bins
        assert result.shape[3] == 3  # (hit_idx1, hit_idx2, label)
        
        # Check that pairs are built correctly
        # Event 0, bin 0: hits 0,1 both belong to particle 0 -> should have pair
        event0_bin0 = result[0, 0]
        # Find non-zero pairs
        non_zero_pairs = event0_bin0[event0_bin0[:, 2] == 1]
        assert len(non_zero_pairs) > 0  # Should have at least one positive pair


class TestToTensor:
    def test_basic_conversion(self):
        # Create test data
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0],
            "particle_id": [0, 0, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
            "r": [1.4, 2.8, 4.2],
            "phi": [0.0, 0.1, 0.2],
            "eta": [0.5, 1.0, 1.5],
            "varR": [0.01, 0.01, 0.01],
            "varZ": [0.01, 0.01, 0.01],
        })
        
        particles_batch = pd.DataFrame({
            "event_id": [0, 0],
            "particle_id": [0, 1],
            "d0": [0.1, 0.2],
            "z0": [5.0, 10.0],
            "phi": [0.0, 0.1],
            "eta": [0.5, 1.0],
            "pT": [10.0, 20.0],
            "q": [1, -1],
            "m": [0.105, 0.105],
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 1],
            "bin1": [0, 0, 1],
            "bin2": [0, 0, 1],
        })
        
        hit_to_particle = pd.Series([0, 0, 1])
        
        hit_features = ["x", "y", "z", "r", "phi", "eta", "varR", "varZ"]
        particle_features = ["d0", "z0", "phi", "eta", "pT", "q", "m"]
        num_bins = 2
        max_hits_per_bin = 5
        
        hits_tensor, particles_tensor, hit_to_particle_tensor = _to_tensor(
            data_batch, particles_batch, bins, hit_to_particle,
            hit_features, particle_features, num_bins, max_hits_per_bin
        )
        
        # Check shapes
        assert hits_tensor.shape == (1, num_bins, max_hits_per_bin, len(hit_features))
        assert particles_tensor.shape == (1, 2, len(particle_features))  # 2 particles
        assert hit_to_particle_tensor.shape == (1, num_bins, max_hits_per_bin, 1)
        
        # Check values for first bin
        assert torch.allclose(hits_tensor[0, 0, 0, 0], torch.tensor(1.0))  # x value
        
        # Check particle mapping
        assert hit_to_particle_tensor[0, 0, 0, 0] == 0  # First hit -> particle 0


class TestAddPadding:
    def test_adds_padding_correctly(self):
        cfg = SeedConfig()
        cfg.max_hit_input = 5
        
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0],
            "particle_id": [0, 0, 1],
            "x": [1.0, 2.0, 3.0],
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 1],
            "bin1": [0, 0, 1],
            "bin2": [0, 0, 1],
        })
        
        data_result, bins_result = _add_padding(data_batch, bins, cfg)
        
        # Should add padding to reach max_hit_input per bin
        # Bin 0 has 2 hits -> add 3 padding
        # Bin 1 has 1 hit -> add 4 padding
        assert len(data_result) > len(data_batch)
        assert "is_padding" in data_result.columns
        
        # Check padding hits have particle_id = -1
        padding_hits = data_result[data_result["is_padding"]]
        assert all(padding_hits["particle_id"] == -1)

    def test_removes_excess_hits(self):
        cfg = SeedConfig()
        cfg.max_hit_input = 2  # Small limit
        
        # Create more hits than max_hit_input in bin 0
        # Make some hits appear as duplicates (bin1 != bin0 for some)
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0, 0],
            "particle_id": [0, 0, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        
        # Set up bins so that some hits are duplicates in bin 0
        # Hit 0: primary in bin 0 (bin1=0)
        # Hit 1: primary in bin 0 (bin1=0)
        # Hit 2: duplicate in bin 0 (bin1=1, but also in bin0=0)
        # Hit 3: primary in bin 1
        bins = pd.DataFrame({
            "bin0": [0, 0, 0, 1],
            "bin1": [0, 0, 1, 1],  # Hit 2 is primary in bin 1, duplicate in bin 0
            "bin2": [0, 0, 0, 1],
        })
        
        data_result, bins_result = _add_padding(data_batch, bins, cfg)
        
        # Hit 2 should be removed from bin 0 by updating its bin0 value
        # After removal, bin 0 should have at most 2 hits as primary (bin1=0)
        event0_bin0_primary = bins_result[(bins_result["bin1"] == 0) & (data_result["event_id"] == 0)]
        assert len(event0_bin0_primary) <= cfg.max_hit_input


class TestCreatePaddingMask:
    def test_basic_mask(self):
        cfg = SeedConfig()
        cfg.max_hit_input = 5
        
        data_batch = pd.DataFrame({
            "event_id": [0, 0, 0, 0, 0],
            "is_padding": [False, False, True, True, True],
        })
        
        bins = pd.DataFrame({
            "bin0": [0, 0, 0, 0, 0],
            "bin1": [0, 0, 0, 0, 0],
            "bin2": [0, 0, 0, 0, 0],
        })
        
        num_bins = 1
        
        data_result, padding_mask = _create_padding_mask(data_batch, bins, num_bins, cfg)
        
        # Check shape: [num_events, num_bins, max_hit_input]
        assert padding_mask.shape == (1, num_bins, cfg.max_hit_input)
        
        # First 2 positions should be False (not padding), rest True
        assert not padding_mask[0, 0, 0]
        assert not padding_mask[0, 0, 1]
        assert padding_mask[0, 0, 2]
        assert padding_mask[0, 0, 3]
        assert padding_mask[0, 0, 4]


class TestOrphanHitRemoval:
    def test_removes_fraction(self):
        data_batch = pd.DataFrame({
            "particle_id": [0, 0, -1, -1, -1, -1, 1, 1],  # 4 orphan hits
        })
        
        # Remove 50% of orphan hits
        result = _orphan_hit_removal(data_batch, fraction_to_drop=0.5, random_state=42)
        
        # Should have removed ~2 orphan hits
        orphan_count_before = (data_batch["particle_id"] == -1).sum()
        orphan_count_after = (result["particle_id"] == -1).sum()
        
        assert orphan_count_after < orphan_count_before
        assert orphan_count_after == orphan_count_before - 2  # Exactly 2 removed (50% of 4)

    def test_zero_fraction_no_removal(self):
        data_batch = pd.DataFrame({
            "particle_id": [0, 0, -1, -1, 1, 1],
        })
        
        result = _orphan_hit_removal(data_batch, fraction_to_drop=0.0)
        
        # No orphan hits should be removed
        assert len(result) == len(data_batch)


class TestBinData:
    def test_no_bin_strategy(self):
        cfg = SeedConfig()
        cfg.binning_strategy = "no_bin"
        cfg.bin_width = 10.0  # Large value
        
        data_batch = pd.DataFrame({
            "phi": [0.0, 1.0, 2.0, 3.0],
        })
        
        bins, num_bins = _bin_data(data_batch, cfg)
        
        # Should have single bin
        assert num_bins == 1
        assert len(bins) == len(data_batch)

    def test_global_bin_strategy(self):
        cfg = SeedConfig()
        cfg.binning_strategy = "global"
        cfg.bin_width = 1.0  # 1 radian bins
        
        data_batch = pd.DataFrame({
            "phi": [0.0, 0.5, 1.5, 2.5, -1.0],
        })
        
        bins, num_bins = _bin_data(data_batch, cfg)
        
        # Should have multiple bins
        assert num_bins > 1
        assert len(bins) == len(data_batch)
        assert "bin1" in bins.columns


class TestComputeBarcode:
    def test_barcode_generation(self):
        cfg = SeedConfig()
        cfg.binning_strategy = "neighbor"
        cfg.bin_width = 0.05
        cfg.max_hit_input = 1200
        cfg.orphan_hit_fraction = 0.0
        
        barcode = compute_barcode(cfg)
        
        # Check format
        assert "BSneighbor" in barcode
        assert "BW0.05" in barcode
        assert "MH1200" in barcode
        assert "OF" not in barcode  # No orphan fraction since it's 0

    def test_barcode_with_orphan_fraction(self):
        cfg = SeedConfig()
        cfg.binning_strategy = "global"
        cfg.bin_width = 0.1
        cfg.max_hit_input = 800
        cfg.orphan_hit_fraction = 0.3
        
        barcode = compute_barcode(cfg)
        
        assert "BSglobal" in barcode
        assert "BW0.1" in barcode
        assert "MH800" in barcode
        assert "OF30" in barcode  # 30% orphan fraction
