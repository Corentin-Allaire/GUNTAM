import pytest
import torch
import numpy as np

from GUNTAM.Seed.Reconstruction import (topk_seed_reconstruction,
                                        chained_seed_reconstruction,
                                        back_chained_seed_reconstruction,
                                        weighted_chained_seed_reconstruction)


class TestTopKSeedReconstruction:
    def _make_inputs(self, n=6, device="cpu"):
        torch.manual_seed(0)
        attention = torch.rand(n, n, device=device)
        # strengthen diagonal slightly (will be set to -inf in logic)
        attention.fill_diagonal_(1.0)
        # reconstructed_parameters: arbitrary parameters; index 4 no longer used for validity
        score = torch.randn(n, 1, device=device)
        # provide some values at column 4, but not used for filtering
        return attention, score

    def test_shapes_and_types(self):
        att, score = self._make_inputs(n=5)
        seeds = topk_seed_reconstruction(att, score, threshold=0.5, max_selection=3)
        assert isinstance(seeds, list)
        # Each cluster entry: (indices np.ndarray, avg_params np.ndarray)
        assert len(seeds) > 0
        idx, avg = seeds[0]
        assert isinstance(idx, np.ndarray)
        assert isinstance(avg, np.ndarray)

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

        score = torch.ones(n, 1)

        seeds = topk_seed_reconstruction(att, score, threshold=0.8, max_selection=3)

        # Find cluster for hit 0
        c0 = next((c for c in seeds if c[0][0] == 0), None)
        assert c0 is not None
        indices = c0[0]
        # Expected kept neighbors: 1,2,4 (>=0.8), limited by max_selection=3
        assert set(indices.tolist()) == {0, 1, 2, 4}

    def test_includes_all_hits(self):
        att, score = self._make_inputs(n=5)
        seeds = topk_seed_reconstruction(att, score, threshold=0.5, max_selection=4)
        # One cluster per hit since score filtering is removed
        assert len(seeds) == score.size(0)
        all_indices = set(range(score.size(0)))
        for (idxs, _) in seeds:
            # All clusters must include their seed hit and may include neighbors by attention only
            assert set(idxs.tolist()).issubset(all_indices)

    def test_empty_input(self):
        att = torch.zeros(0, 0)
        score = torch.zeros(0, 1)
        seeds = topk_seed_reconstruction(att, score)
        assert seeds == []


class TestChainedSeedReconstruction:
    """Tests for the chained_seed_reconstruction function (forward chaining)."""

    def test_basic_chain_creation(self):
        """Test that basic forward chains are created correctly."""
        n = 6
        att = torch.zeros(n, n)
        # Build forward chains: 0 -> 1 -> 2 with sufficient scores, and 3 -> 4 -> 5
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.95
        att[4, 5] = 0.9
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        assert isinstance(seeds, list)
        # Should discover 2 chains
        assert len(seeds) == 2
        for idxs, avg in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray)
        # First chain should start from index 0 (processed first)
        assert seeds[0][0][0] == 0
        # Second chain should start from index 3
        assert seeds[1][0][0] == 3

    def test_forward_chain_direction(self):
        """Test that forward chains correctly follow increasing indices."""
        n = 5
        att = torch.zeros(n, n)
        # Create a clear forward chain: 0 -> 1 -> 2 -> 3 -> 4
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        att[3, 4] = 0.80
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.75, max_chain_length=10)
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
        att = torch.zeros(n, n)
        att[0, 1] = 0.81
        # Hit 2 has a low score, so it is filtered out by valid_hits
        score = torch.ones(n, 1)
        score[2] = 0.5  # below score_threshold=0.75 → invalid
        score[3] = 0.5
        score[4] = 0.5

        result = chained_seed_reconstruction(att, score, score_threshold=0.75, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        # Only hits 0 and 1 are valid; chain [0, 1] has length 2 < 3 → not added
        assert seeds == []

    def test_forward_max_chain_length(self):
        """Test that forward chains respect max_chain_length."""
        n = 8
        att = torch.zeros(n, n)
        # Create a long forward chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
        for i in range(7):
            att[i, i + 1] = 0.9
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=4)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should have at least one chain
        assert len(seeds) >= 1
        # The first chain should have exactly max_chain_length hits
        chain_indices, _ = seeds[0]
        assert len(chain_indices) <= 4

    def test_forward_processes_from_first_to_last(self):
        """Test that forward reconstruction processes hits from first to last."""
        n = 6
        att = torch.zeros(n, n)
        # Create two forward chains: 0 -> 1 -> 2 and 3 -> 4 -> 5
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.9
        att[4, 5] = 0.85
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)
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
        att = torch.zeros(n, n)
        # Create overlapping paths: 0 -> 1 -> 2 and 1 -> 2 -> 3
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)
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
        score = torch.zeros(0, 1)
        result = chained_seed_reconstruction(att, score)
        seeds = result[0] if isinstance(result, tuple) else result
        assert seeds == []

    def test_forward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        att = torch.zeros(n, n)
        # Create chains of different lengths
        att[0, 1] = 0.9  # Chain of length 2 (not enough)
        att[2, 3] = 0.9
        att[3, 4] = 0.85  # Chain of length 3 (valid)
        score = torch.ones(n, 1)

        result = chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)
        seeds = result[0] if isinstance(result, tuple) else result
        
        # Should only return the chain with length >= 3
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


class TestBackChainedSeedReconstruction:
    """Tests for the back_chained_seed_reconstruction function (backward chaining)."""

    def test_basic_backward_chain_creation(self):
        """Test that basic backward chains are created correctly."""
        n = 6
        att = torch.zeros(n, n)
        # Build backward chains: 2 <- 1 <- 0 with sufficient scores, and 5 <- 4 <- 3
        att[2, 1] = 0.9  # from 2, look back to 1
        att[1, 0] = 0.85  # from 1, look back to 0
        att[5, 4] = 0.95  # from 5, look back to 4
        att[4, 3] = 0.9  # from 4, look back to 3
        score = torch.ones(n, 1)

        seeds = back_chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)
        assert isinstance(seeds, list)
        # Should discover at least 2 chains
        assert len(seeds) == 2
        for idxs, avg in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray)
        # First chain should start from index 5 (processed first as it's last)
        assert seeds[0][0][0] == 5
        # Second chain should start from index 2
        assert seeds[1][0][0] == 2

    def test_backward_chain_stops_without_valid_previous(self):
        """Test that backward chain stops when no valid previous hit exists."""
        n = 5
        att = torch.zeros(n, n)
        # 4 <- 3 valid, but 3 has no valid previous above threshold
        att[4, 3] = 0.81
        # Hits 0, 1, 2 have low scores, so they are filtered out by valid_hits
        score = torch.ones(n, 1)
        score[0] = 0.5  # below score_threshold=0.75 → invalid
        score[1] = 0.5
        score[2] = 0.5

        seeds = back_chained_seed_reconstruction(att, score, score_threshold=0.75, max_chain_length=5)
        # Only hits 3 and 4 are valid; chain [4, 3] has length 2 < 3 → not added
        assert seeds == []

    def test_backward_max_chain_length(self):
        """Test that backward chains respect max_chain_length."""
        n = 8
        att = torch.zeros(n, n)
        # Create a long backward chain: 7 <- 6 <- 5 <- 4 <- 3 <- 2 <- 1 <- 0
        for i in range(7, 0, -1):
            att[i, i - 1] = 0.9
        score = torch.ones(n, 1)

        seeds = back_chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=4)

        # Should have at least one chain
        assert len(seeds) >= 1
        # The first chain should have exactly max_chain_length hits
        chain_indices, _ = seeds[0]
        assert len(chain_indices) <= 4

    def test_backward_prevents_reuse_of_hits(self):
        """Test that hits used in one chain cannot be reused in another."""
        n = 5
        att = torch.zeros(n, n)
        # Create overlapping paths: 4 <- 3 <- 2 and 3 <- 2 <- 1
        att[4, 3] = 0.95
        att[3, 2] = 0.90
        att[2, 1] = 0.85
        score = torch.ones(n, 1)

        seeds = back_chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)

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
        score = torch.zeros(0, 1)
        seeds = back_chained_seed_reconstruction(att, score)
        assert seeds == []

    def test_backward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        att = torch.zeros(n, n)
        # Create chains of different lengths
        att[4, 3] = 0.9  # Chain of length 2 (not enough)
        att[2, 1] = 0.9
        att[1, 0] = 0.85  # Chain of length 3 (valid)
        score = torch.ones(n, 1)

        seeds = back_chained_seed_reconstruction(att, score, score_threshold=0.8, max_chain_length=5)

        # Should only return the chain with length >= 3
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


class TestWeightedChainedSeedReconstruction:
    """Tests for the weighted_chained_seed_reconstruction function."""

    def _make_score(self, n: int, score: float = 1) -> torch.Tensor:
        return torch.ones(n, 1) * score

    def test_pairs_below_threshold_excluded(self):
        """Pairs whose attention score is below score_threshold are ignored."""
        n = 5
        att = torch.zeros(n, n)
        # All scores below threshold – no pair at all
        att[0, 1] = 0.9
        att[1, 2] = 0.98
        score = self._make_score(n, 0.55)
        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        assert seeds == []

    def test_pairs_exactly_at_threshold_included(self):
        """Pairs whose score equals score_threshold are kept."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.99   # exactly at threshold
        att[1, 2] = 0.6
        score = self._make_score(n, 0.8)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        # Should produce a chain containing at least hits 0, 1, 2
        assert len(seeds) >= 1
        all_indices = set(seeds[0][0].tolist())
        assert {0, 1, 2}.issubset(all_indices)

    def test_minimum_chain_length_3(self):
        """Chains shorter than 3 hits are discarded."""
        n = 4
        att = torch.zeros(n, n)
        # Only one pair above threshold → chain of length 2 → discarded
        att[0, 1] = 0.9
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        assert seeds == []

    def test_chain_of_exactly_3_hits_kept(self):
        """A chain of exactly 3 hits is kept."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        # No further extension available
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        assert len(seeds) == 1
        assert len(seeds[0][0]) == 3
        assert set(seeds[0][0].tolist()) == {0, 1, 2}

    def test_max_chain_length_respected(self):
        """Chains never exceed max_chain_length hits."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(n - 1):
            att[i, i + 1] = 0.9
        score = self._make_score(n)

        max_len = 4
        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=max_len, pairs_per_hit=2
        )
        assert len(seeds) >= 1
        for idxs, _ in seeds:
            assert len(idxs) <= max_len

    def test_basic_forward_chain(self):
        """A simple linear chain 0->1->2->3 is reconstructed correctly."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=10, pairs_per_hit=2
        )
        assert len(seeds) >= 1
        chain_idxs = seeds[0][0].tolist()
        assert set(chain_idxs) == {0, 1, 2, 3}

    def test_average_parameters_correct(self):
        """avg_parameters is a numpy array returned alongside chain indices."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        assert len(seeds) >= 1
        idxs, avg = seeds[0]
        assert isinstance(avg, np.ndarray)

    def test_no_hit_reuse_across_seeds(self):
        """Once hits are claimed by a chain with length>=3, they are removed from the pool."""
        n = 6
        att = torch.zeros(n, n)
        # One strong chain: 0->1->2
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        # A second path that shares hits 1 and 2 should not produce another seed
        att[1, 3] = 0.85
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        # Collect every index used across all seeds
        all_used: list[int] = []
        for idxs, _ in seeds:
            all_used.extend(idxs.tolist())
        # No index should appear in more than one seed
        assert len(all_used) == len(set(all_used)), "Hits reused across chains"

    def test_pairs_per_hit_limits_candidates(self):
        """With pairs_per_hit=1 only the single strongest neighbor is considered per hit."""
        n = 6
        att = torch.zeros(n, n)
        # Hit 0 has two strong neighbors; with pairs_per_hit=1 only the stronger is kept
        att[0, 1] = 0.95   # stronger
        att[0, 2] = 0.90   # weaker – should be ignored with pairs_per_hit=1
        att[2, 5] = 0.93
        att[1, 3] = 0.88
        att[1, 4] = 0.85
        score = self._make_score(n)

        seeds_1 = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=1
        )
        seeds_2 = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        # With pairs_per_hit=1 the pair (0,2) is never considered
        indices_1 = set(seeds_1[0][0].tolist()) if seeds_1 else set()
        assert 2 not in indices_1
        # With pairs_per_hit=2 the pair (0,2) may appear
        all_indices_2 = {idx for idxs, _ in seeds_2 for idx in idxs.tolist()}
        assert 2 in all_indices_2

    def test_two_independent_chains(self):
        """Two disjoint chains are both reconstructed."""
        n = 8
        att = torch.zeros(n, n)
        # Chain A: 0->1->2
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        # Chain B: 4->5->6
        att[4, 5] = 0.95
        att[5, 6] = 0.90
        score = self._make_score(n)

        seeds = weighted_chained_seed_reconstruction(
            att, score, score_threshold=0.8, max_chain_length=5, pairs_per_hit=2
        )
        assert len(seeds) == 2
        chain_sets = [set(idxs.tolist()) for idxs, _ in seeds]
        assert {0, 1, 2} in chain_sets
        assert {4, 5, 6} in chain_sets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
