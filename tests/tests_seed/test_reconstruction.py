import pytest
import torch
import numpy as np

from GUNTAM.Seed.Reconstruction import topk_seed_reconstruction, chained_seed_reconstruction, back_chained_seed_reconstruction


class TestTopKSeedReconstruction:
    def _make_inputs(self, n=6, d=6, device="cpu"):
        torch.manual_seed(0)
        attention = torch.rand(n, n, device=device)
        # strengthen diagonal slightly (will be set to -inf in logic)
        attention.fill_diagonal_(1.0)
        # reconstructed_parameters: arbitrary parameters; index 4 no longer used for validity
        params = torch.randn(n, d, device=device)
        # provide some values at column 4, but not used for filtering
        params[:, 4] = torch.tensor([0.1, 0.5, 0.9, -0.5, 0.2, 0.8], device=device)[:n]
        return attention, params

    def test_shapes_and_types(self):
        att, params = self._make_inputs(n=5, d=6)
        seeds = topk_seed_reconstruction(att, params, threshold=0.5, max_selection=3)
        assert isinstance(seeds, list)
        # Each cluster entry: (indices np.ndarray, avg_params np.ndarray)
        assert len(seeds) > 0
        idx, avg = seeds[0]
        assert isinstance(idx, np.ndarray)
        assert isinstance(avg, np.ndarray)
        assert avg.shape[-1] == params.size(1)

    def test_respects_threshold_and_max_selection(self):
        n = 6
        att = torch.zeros(n, n)
        # create clear high-attention neighbors for index 0
        att[0, 1] = 0.95
        att[0, 2] = 0.90
        att[0, 3] = 0.70  # below threshold
        att[0, 4] = 0.99
        att[0, 5] = 0.60
        # symmetric or unrelated values
        att[1, 0] = 0.2
        att.fill_diagonal_(1.0)

        params = torch.randn(n, 6)
        params[:, 4] = 0.2  # all allowed

        seeds = topk_seed_reconstruction(att, params, threshold=0.8, max_selection=3)

        # Find cluster for hit 0
        c0 = next((c for c in seeds if c[0][0] == 0), None)
        assert c0 is not None
        indices = c0[0]
        # Expected kept neighbors: 1,2,4 (>=0.8), limited by max_selection=3
        assert set(indices.tolist()) == {0, 1, 2, 4}

    def test_includes_all_hits(self):
        att, params = self._make_inputs(n=5, d=6)
        # Manipulate column 4 scores arbitrarily; should not affect inclusion
        params[1, 4] = -1.0
        params[3, 4] = -0.2
        seeds = topk_seed_reconstruction(att, params, threshold=0.5, max_selection=4)
        # One cluster per hit since score filtering is removed
        assert len(seeds) == params.size(0)
        all_indices = set(range(params.size(0)))
        for (idxs, _) in seeds:
            # All clusters must include their seed hit and may include neighbors by attention only
            assert set(idxs.tolist()).issubset(all_indices)

    def test_empty_input(self):
        att = torch.zeros(0, 0)
        params = torch.zeros(0, 5)
        seeds = topk_seed_reconstruction(att, params)
        assert seeds == []


class TestChainedSeedReconstruction:
    """Tests for the chained_seed_reconstruction function (forward chaining)."""

    def test_basic_chain_creation(self):
        """Test that basic forward chains are created correctly."""
        n = 6
        d = 6
        att = torch.zeros(n, n)
        # Build forward chains: 0 -> 1 -> 2 with sufficient scores, and 3 -> 4 -> 5
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.95
        att[4, 5] = 0.9
        params = torch.randn(n, d)
        params[:, 4] = 0.5

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        assert isinstance(seeds, list)
        # Should discover 2 chains
        assert len(seeds) == 2
        for idxs, avg in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray) and avg.shape[-1] == d
        # First chain should start from index 0 (processed first)
        assert seeds[0][0][0] == 0
        # Second chain should start from index 3
        assert seeds[1][0][0] == 3

    def test_forward_chain_direction(self):
        """Test that forward chains correctly follow increasing indices."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # Create a clear forward chain: 0 -> 1 -> 2 -> 3 -> 4
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        att[3, 4] = 0.80
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.75, max_chain_length=10)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should find a chain starting from index 0 going forward
        assert len(seeds) >= 1
        chain_indices, _ = seeds[0]
        # Chain should start with lowest index and go forward
        assert chain_indices[0] == 0
        # Indices should be increasing
        for i in range(len(chain_indices) - 1):
            assert chain_indices[i] < chain_indices[i + 1]

    def test_chain_stops_without_valid_next(self):
        """Test that forward chain stops when no valid next hit exists."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # 0 -> 1 valid, but 1 has no valid next above threshold
        att[0, 1] = 0.81
        att[1, 2] = 0.70  # below threshold
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.75, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        # No chain reaches length >=3 under this threshold
        assert seeds == []

    def test_forward_max_chain_length(self):
        """Test that forward chains respect max_chain_length."""
        n = 8
        d = 6
        att = torch.zeros(n, n)
        # Create a long forward chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
        for i in range(7):
            att[i, i + 1] = 0.9
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=4)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should have at least one chain
        assert len(seeds) >= 1
        # The first chain should have exactly max_chain_length hits
        chain_indices, _ = seeds[0]
        assert len(chain_indices) <= 4

    def test_forward_processes_from_first_to_last(self):
        """Test that forward reconstruction processes hits from first to last."""
        n = 6
        d = 6
        att = torch.zeros(n, n)
        # Create two forward chains: 0 -> 1 -> 2 and 3 -> 4 -> 5
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.9
        att[4, 5] = 0.85
        params = torch.randn(n, d)
        params[:, 4] = 0.5

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should find 2 chains
        assert len(seeds) == 2
        # First chain should start from index 0 (processed first)
        assert seeds[0][0][0] == 0
        # Second chain should start from index 3
        assert seeds[1][0][0] == 3

    def test_forward_prevents_reuse_of_hits(self):
        """Test that hits used in one chain cannot be reused in another."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # Create overlapping paths: 0 -> 1 -> 2 and 1 -> 2 -> 3
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should only create one chain (the first one found)
        # because hits 1 and 2 will be marked as used
        assert len(seeds) == 1
        # Collect all used indices
        all_indices = set()
        for idxs, _ in seeds:
            for idx in idxs:
                assert idx not in all_indices, "Hit reused in multiple chains"
                all_indices.add(idx)

    def test_empty_input(self):
        """Test forward reconstruction with empty input."""
        att = torch.zeros(0, 0)
        params = torch.zeros(0, 5)
        result = chained_seed_reconstruction(att, params)
        seeds = result[0] if isinstance(result, tuple) else result
        assert seeds == []

    def test_forward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # Create chains of different lengths
        att[0, 1] = 0.9  # Chain of length 2 (not enough)
        att[2, 3] = 0.9
        att[3, 4] = 0.85  # Chain of length 3 (valid)
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should only return the chain with length >= 3
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


class TestBackChainedSeedReconstruction:
    """Tests for the back_chained_seed_reconstruction function (backward chaining)."""

    def test_basic_backward_chain_creation(self):
        """Test that basic backward chains are created correctly."""
        n = 6
        d = 6
        att = torch.zeros(n, n)
        # Build backward chains: 2 <- 1 <- 0 with sufficient scores, and 5 <- 4 <- 3
        att[2, 1] = 0.9  # from 2, look back to 1
        att[1, 0] = 0.85  # from 1, look back to 0
        att[5, 4] = 0.95  # from 5, look back to 4
        att[4, 3] = 0.9  # from 4, look back to 3
        params = torch.randn(n, d)
        params[:, 4] = 0.5

        seeds = back_chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        assert isinstance(seeds, list)
        # Should discover at least 2 chains
        assert len(seeds) == 2
        for idxs, avg in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray) and avg.shape[-1] == d
        # First chain should start from index 5 (processed first as it's last)
        assert seeds[0][0][0] == 5
        # Second chain should start from index 2
        assert seeds[1][0][0] == 2

    def test_backward_chain_stops_without_valid_previous(self):
        """Test that backward chain stops when no valid previous hit exists."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # 4 <- 3 valid, but 3 has no valid previous above threshold
        att[4, 3] = 0.81
        att[3, 2] = 0.70  # below threshold
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        seeds = back_chained_seed_reconstruction(att, params, score_threshold=0.75, max_chain_length=5)
        # No chain reaches length >=3 under this threshold
        assert seeds == []

    def test_backward_max_chain_length(self):
        """Test that backward chains respect max_chain_length."""
        n = 8
        d = 6
        att = torch.zeros(n, n)
        # Create a long backward chain: 7 <- 6 <- 5 <- 4 <- 3 <- 2 <- 1 <- 0
        for i in range(7, 0, -1):
            att[i, i - 1] = 0.9
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        seeds = back_chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=4)

        # Should have at least one chain
        assert len(seeds) >= 1
        # The first chain should have exactly max_chain_length hits
        chain_indices, _ = seeds[0]
        assert len(chain_indices) <= 4

    def test_backward_prevents_reuse_of_hits(self):
        """Test that hits used in one chain cannot be reused in another."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # Create overlapping paths: 4 <- 3 <- 2 and 3 <- 2 <- 1
        att[4, 3] = 0.95
        att[3, 2] = 0.90
        att[2, 1] = 0.85
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        seeds = back_chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)

        # Should only create one chain (the first one found)
        # because hits 3 and 2 will be marked as used
        assert len(seeds) == 1
        # Collect all used indices
        all_indices = set()
        for idxs, _ in seeds:
            for idx in idxs:
                assert idx not in all_indices, "Hit reused in multiple chains"
                all_indices.add(idx)

    def test_backward_empty_input(self):
        """Test backward reconstruction with empty input."""
        att = torch.zeros(0, 0)
        params = torch.zeros(0, 5)
        seeds = back_chained_seed_reconstruction(att, params)
        assert seeds == []

    def test_backward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # Create chains of different lengths
        att[4, 3] = 0.9  # Chain of length 2 (not enough)
        att[2, 1] = 0.9
        att[1, 0] = 0.85  # Chain of length 3 (valid)
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        seeds = back_chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)

        # Should only return the chain with length >= 3
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
