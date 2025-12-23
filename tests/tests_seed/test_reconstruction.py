import pytest
import torch
import numpy as np

from GUNTAM.Seed.Reconstruction import topk_seed_reconstruction, chained_seed_reconstruction


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
    def test_basic_chain_creation(self):
        n = 6
        d = 6
        att = torch.zeros(n, n)
        # Build a chain 0 -> 1 -> 2 with sufficient scores, and 3 -> 4 -> 5
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.95
        att[4, 5] = 0.9
        params = torch.randn(n, d)
        params[:, 4] = 0.5

        result = chained_seed_reconstruction(att, params, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        assert isinstance(seeds, list)
        # Each discovered chain must have length >=3
        assert len(seeds) >= 2
        for idxs, avg in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray) and avg.shape[-1] == d

    def test_chain_stops_without_valid_next(self):
        n = 5
        d = 6
        att = torch.zeros(n, n)
        # 0 -> 1 valid, but 1 has no valid next above threshold
        att[0, 1] = 0.81
        params = torch.randn(n, d)
        params[:, 4] = 0.3

        result = chained_seed_reconstruction(att, params, score_threshold=0.82, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        # No chain reaches length >=3 under this threshold
        assert seeds == []

    def test_empty_input(self):
        att = torch.zeros(0, 0)
        params = torch.zeros(0, 5)
        result = chained_seed_reconstruction(att, params)
        seeds = result[0] if isinstance(result, tuple) else result
        assert seeds == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
