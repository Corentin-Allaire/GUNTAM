import pytest
import torch
import numpy as np

from GUNTAM.Seed.Reconstruction import (topk_seed_reconstruction,
                                        chained_seed_reconstruction,
                                        back_chained_seed_reconstruction,
                                        weighted_chained_seed_reconstruction,
                                        beam_search_seed_reconstruction,
                                        batched_beam_search_seed_reconstruction,
                                        build_seed_features_tensor)


def _to_edge(att: torch.Tensor, width: int = None) -> torch.Tensor:
    """Convert a dense [N, N] attention matrix to [N, width, 3] edge triplets."""
    N = att.shape[0]
    if N == 0:
        return torch.zeros(0, 0, 3)
    w = N if width is None else min(width, N)
    scores, targets = att.topk(w, dim=1, largest=True, sorted=True)  # [N, w]
    sources = torch.arange(N).unsqueeze(1).expand_as(targets)
    return torch.stack([sources.float(), targets.float(), scores], dim=2)


def _to_edge_batched(att: torch.Tensor, width: int = None) -> torch.Tensor:
    """Convert a dense [B, N, N] attention matrix to [B, N, width, 3] edge triplets."""
    B, N, _ = att.shape
    w = N if width is None else min(width, N)
    scores, targets = att.topk(w, dim=2, largest=True, sorted=True)  # [B, N, w]
    sources = torch.arange(N)[None, :, None].expand(B, N, w)
    return torch.stack([sources.float(), targets.float(), scores], dim=3)


class TestTopKSeedReconstruction:
    def _make_inputs(self, n=6, device="cpu"):
        torch.manual_seed(0)
        attention = torch.rand(n, n, device=device)
        attention.fill_diagonal_(1.0)
        return _to_edge(attention)

    def test_shapes_and_types(self):
        edge = self._make_inputs(n=5)
        seeds = topk_seed_reconstruction(edge, max_selection=3)
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        idx, avg, seed_score = seeds[0]
        assert isinstance(idx, np.ndarray)
        assert isinstance(avg, np.ndarray)

    def test_respects_max_selection(self):
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[0, 2] = 0.90
        att[0, 3] = 0.70
        att[0, 4] = 0.99
        att[0, 5] = 0.60
        att[1, 0] = 0.2
        att.fill_diagonal_(1.0)
        edge = _to_edge(att)
        seeds = topk_seed_reconstruction(edge, max_selection=3)
        c0 = next((c for c in seeds if c[0][0] == 0), None)
        assert c0 is not None
        indices = c0[0]
        # Top 3 non-self neighbors for hit 0: 4 (0.99), 1 (0.95), 2 (0.90)
        assert set(indices.tolist()) == {0, 4, 1, 2}

    def test_includes_all_hits(self):
        edge = self._make_inputs(n=5)
        seeds = topk_seed_reconstruction(edge, max_selection=4)
        assert len(seeds) == edge.shape[0]
        all_indices = set(range(edge.shape[0]))
        for (idxs, _, _seed_score) in seeds:
            assert set(idxs.tolist()).issubset(all_indices)

    def test_empty_input(self):
        edge = _to_edge(torch.zeros(0, 0))
        seeds = topk_seed_reconstruction(edge)
        assert seeds == []


class TestChainedSeedReconstruction:
    """Tests for the chained_seed_reconstruction function (forward chaining)."""

    def test_basic_chain_creation(self):
        """Test that basic forward chains are created correctly."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.95
        att[4, 5] = 0.9
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=5)
        assert isinstance(seeds, list)
        assert len(seeds) == 2
        for idxs, avg, seed_score in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray)
        assert seeds[0][0][0] == 0
        assert seeds[1][0][0] == 3

    def test_forward_chain_direction(self):
        """Test that forward chains correctly follow increasing indices."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        att[3, 4] = 0.80
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=10)
        assert len(seeds) >= 1
        chain_indices, _, _seed_score = seeds[0]
        assert chain_indices[0] == 0
        for i in range(len(chain_indices) - 1):
            assert chain_indices[i] < chain_indices[i + 1]

    def test_chain_stops_without_valid_next(self):
        """Test that chain stops when no forward neighbor has positive attention."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.81
        # No forward edges from hit 1 -> chain [0, 1] has length 2 < 3
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=5)
        assert seeds == []

    def test_forward_max_chain_length(self):
        """Test that forward chains respect max_chain_length."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(7):
            att[i, i + 1] = 0.9
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=4)
        assert len(seeds) >= 1
        chain_indices, _, _seed_score = seeds[0]
        assert len(chain_indices) <= 4

    def test_forward_processes_from_first_to_last(self):
        """Test that forward reconstruction processes hits from first to last."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.85
        att[3, 4] = 0.9
        att[4, 5] = 0.85
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=5)
        assert len(seeds) == 2
        assert seeds[0][0][0] == 0
        assert seeds[1][0][0] == 3

    def test_forward_prevents_reuse_of_hits(self):
        """Test that hits used in one chain cannot be reused in another."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=5)
        assert len(seeds) == 1
        all_indices = set()
        for idxs, _, _seed_score in seeds:
            for idx in idxs:
                assert idx not in all_indices, "Hit reused in multiple chains"
                all_indices.add(idx)

    def test_empty_input(self):
        """Test forward reconstruction with empty input."""
        edge = _to_edge(torch.zeros(0, 0))
        seeds = chained_seed_reconstruction(edge)
        assert seeds == []

    def test_forward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9  # Chain of length 2 (not enough)
        att[2, 3] = 0.9
        att[3, 4] = 0.85  # Chain of length 3 (valid)
        edge = _to_edge(att)
        seeds = chained_seed_reconstruction(edge, max_chain_length=5)
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


class TestBackChainedSeedReconstruction:
    """Tests for the back_chained_seed_reconstruction function (backward chaining)."""

    def test_basic_backward_chain_creation(self):
        """Test that basic backward chains are created correctly."""
        n = 6
        att = torch.zeros(n, n)
        att[2, 1] = 0.9
        att[1, 0] = 0.85
        att[5, 4] = 0.95
        att[4, 3] = 0.9
        edge = _to_edge(att)
        seeds = back_chained_seed_reconstruction(edge, max_chain_length=5)
        assert isinstance(seeds, list)
        assert len(seeds) == 2
        for idxs, avg, seed_score in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.size >= 3
            assert isinstance(avg, np.ndarray)
        assert seeds[0][0][0] == 5
        assert seeds[1][0][0] == 2

    def test_backward_chain_stops_without_valid_previous(self):
        """Test that backward chain stops when no backward neighbor has positive attention."""
        n = 5
        att = torch.zeros(n, n)
        att[4, 3] = 0.81
        # No backward edges from hit 3 -> chain [4, 3] has length 2 < 3
        edge = _to_edge(att)
        seeds = back_chained_seed_reconstruction(edge, max_chain_length=5)
        assert seeds == []

    def test_backward_max_chain_length(self):
        """Test that backward chains respect max_chain_length."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(7, 0, -1):
            att[i, i - 1] = 0.9
        edge = _to_edge(att)
        seeds = back_chained_seed_reconstruction(edge, max_chain_length=4)
        assert len(seeds) >= 1
        chain_indices, _, _seed_score = seeds[0]
        assert len(chain_indices) <= 4

    def test_backward_prevents_reuse_of_hits(self):
        """Test that hits used in one chain cannot be reused in another."""
        n = 5
        att = torch.zeros(n, n)
        att[4, 3] = 0.95
        att[3, 2] = 0.90
        att[2, 1] = 0.85
        edge = _to_edge(att)
        seeds = back_chained_seed_reconstruction(edge, max_chain_length=5)
        assert len(seeds) == 1
        all_indices = set()
        for idxs, _, _seed_score in seeds:
            for idx in idxs:
                assert idx not in all_indices, "Hit reused in multiple chains"
                all_indices.add(idx)

    def test_backward_empty_input(self):
        """Test backward reconstruction with empty input."""
        edge = _to_edge(torch.zeros(0, 0))
        seeds = back_chained_seed_reconstruction(edge)
        assert seeds == []

    def test_backward_minimum_chain_length(self):
        """Test that only chains with 3 or more hits are returned."""
        n = 5
        att = torch.zeros(n, n)
        att[4, 3] = 0.9  # Chain of length 2 (not enough)
        att[2, 1] = 0.9
        att[1, 0] = 0.85  # Chain of length 3 (valid)
        edge = _to_edge(att)
        seeds = back_chained_seed_reconstruction(edge, max_chain_length=5)
        assert len(seeds) == 1
        assert len(seeds[0][0]) >= 3


class TestWeightedChainedSeedReconstruction:
    """Tests for the weighted_chained_seed_reconstruction function."""

    def test_minimum_chain_length_3(self):
        """Chains shorter than 3 hits are discarded."""
        n = 4
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        assert seeds == []

    def test_chain_of_exactly_3_hits_kept(self):
        """A chain of exactly 3 hits is kept."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        assert len(seeds) == 1
        assert len(seeds[0][0]) == 3
        assert set(seeds[0][0].tolist()) == {0, 1, 2}

    def test_max_chain_length_respected(self):
        """Chains never exceed max_chain_length hits."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(n - 1):
            att[i, i + 1] = 0.9
        edge = _to_edge(att)
        max_len = 4
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=max_len, pairs_per_hit=2)
        assert len(seeds) >= 1
        for idxs, _, _seed_score in seeds:
            assert len(idxs) <= max_len

    def test_basic_forward_chain(self):
        """A simple linear chain 0->1->2->3 is reconstructed correctly."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=10, pairs_per_hit=2)
        assert len(seeds) >= 1
        chain_idxs = seeds[0][0].tolist()
        assert set(chain_idxs) == {0, 1, 2, 3}

    def test_average_parameters_correct(self):
        """avg_parameters is a numpy array returned alongside chain indices."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        assert len(seeds) >= 1
        idxs, avg, seed_score = seeds[0]
        assert isinstance(avg, np.ndarray)

    def test_no_hit_reuse_across_seeds(self):
        """Once hits are claimed by a chain with length>=3, they are removed from the pool."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[1, 3] = 0.85
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        all_used: list[int] = []
        for idxs, _, _seed_score in seeds:
            all_used.extend(idxs.tolist())
        assert len(all_used) == len(set(all_used)), "Hits reused across chains"

    def test_pairs_per_hit_limits_candidates(self):
        """With pairs_per_hit=1 only the single strongest neighbor is considered per hit."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95   # stronger
        att[0, 2] = 0.90   # weaker – should be ignored with pairs_per_hit=1
        att[2, 5] = 0.93
        att[1, 3] = 0.88
        att[1, 4] = 0.85
        edge = _to_edge(att)
        seeds_1 = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=1)
        seeds_2 = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        indices_1 = set(seeds_1[0][0].tolist()) if seeds_1 else set()
        assert 2 not in indices_1
        all_indices_2 = {idx for idxs, _, _s in seeds_2 for idx in idxs.tolist()}
        assert 2 in all_indices_2

    def test_two_independent_chains(self):
        """Two disjoint chains are both reconstructed."""
        n = 8
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[4, 5] = 0.95
        att[5, 6] = 0.90
        edge = _to_edge(att)
        seeds = weighted_chained_seed_reconstruction(edge, max_chain_length=5, pairs_per_hit=2)
        assert len(seeds) == 2
        chain_sets = [set(idxs.tolist()) for idxs, _, _s in seeds]
        assert {0, 1, 2} in chain_sets
        assert {4, 5, 6} in chain_sets


class TestBeamSearchSeedReconstruction:
    """Tests for beam_search_seed_reconstruction (CPU, single-bin)."""

    def _make_starting_mask(self, n: int) -> torch.Tensor:
        return torch.ones(n, dtype=torch.bool)

    def test_output_is_list_of_tuples(self):
        """Return type is a list of (np.ndarray[int64], np.ndarray[float32], float) tuples."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        assert isinstance(seeds, list)
        for idxs, params, seed_score in seeds:
            assert isinstance(idxs, np.ndarray)
            assert idxs.dtype == np.int64
            assert isinstance(params, np.ndarray)
            assert params.shape == (5,)
            assert isinstance(seed_score, float)

    def test_params_are_zeros(self):
        """avg_parameters is always a zero array of shape (5,) with float32 dtype."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        for _, params, _ in seeds:
            assert np.all(params == 0.0)
            assert params.dtype == np.float32

    def test_basic_forward_chain(self):
        """A clear linear chain 0->1->2->3 is recovered as the best seed."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.95
        att[1, 2] = 0.90
        att[2, 3] = 0.85
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        assert len(seeds) >= 1
        chain_idxs = seeds[0][0].tolist()
        assert set(chain_idxs).issuperset({0, 1, 2, 3})

    def test_chains_are_strictly_forward(self):
        """All returned chains contain strictly increasing hit indices (j > i)."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(n - 1):
            att[i, i + 1] = 0.9
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        for idxs, _, _ in seeds:
            for a, b in zip(idxs[:-1], idxs[1:]):
                assert b > a, f"Chain not strictly forward: {idxs}"

    def test_minimum_chain_length_is_three(self):
        """Only chains of length >= 3 are returned."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        att[2, 3] = 0.9
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        for idxs, _, _ in seeds:
            assert len(idxs) >= 3

    def test_max_chain_length_respected(self):
        """Chains never exceed max_chain_length hits."""
        n = 8
        att = torch.zeros(n, n)
        for i in range(n - 1):
            att[i, i + 1] = 0.9
        edge = _to_edge(att)
        max_len = 4
        seeds = beam_search_seed_reconstruction(
            edge, self._make_starting_mask(n), max_chain_length=max_len
        )
        for idxs, _, _ in seeds:
            assert len(idxs) <= max_len

    def test_starting_mask_limits_chain_starts(self):
        """Only hits allowed by starting_mask can start a chain."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        att[2, 3] = 0.9
        edge = _to_edge(att)
        starting_mask = self._make_starting_mask(n)
        starting_mask[0] = False
        seeds = beam_search_seed_reconstruction(edge, starting_mask)
        for idxs, _, _ in seeds:
            assert idxs[0] != 0

    def test_no_valid_chain_returns_empty_list(self):
        """When no chain of length >= 3 can be formed, return an empty list."""
        n = 2
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(edge, self._make_starting_mask(n))
        assert seeds == []

    def test_empty_starting_mask_returns_empty(self):
        """If no hit is in starting_mask, return an empty list."""
        n = 6
        att = torch.zeros(n, n)
        att[0, 1] = 0.9
        att[1, 2] = 0.9
        edge = _to_edge(att)
        starting_mask = torch.zeros(n, dtype=torch.bool)
        seeds = beam_search_seed_reconstruction(edge, starting_mask)
        assert seeds == []

    def test_seed_score_is_cumulative_over_chain_length(self):
        """Seed score equals sum of edge attention scores divided by chain length."""
        n = 5
        att = torch.zeros(n, n)
        att[0, 1] = 0.8
        att[1, 2] = 0.6
        edge = _to_edge(att)
        seeds = beam_search_seed_reconstruction(
            edge, self._make_starting_mask(n), max_chain_length=3
        )
        assert len(seeds) >= 1
        idxs, _, seed_score = seeds[0]
        assert idxs.tolist()[:3] == [0, 1, 2]
        expected = (0.8 + 0.6) / 3
        assert abs(seed_score - expected) < 1e-5


class TestBatchedBeamSearchSeedReconstruction:
    """Tests for batched_beam_search_seed_reconstruction (GPU-batched)."""

    def _make_inputs(self, B: int, N: int):
        att = torch.zeros(B, N, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        return _to_edge_batched(att), mask

    def test_output_shapes(self):
        """All three output tensors have the expected shapes."""
        B, N, CL = 2, 8, 5
        edge, mask = self._make_inputs(B, N)
        chains, params, best_scores = batched_beam_search_seed_reconstruction(
            edge, mask, max_chain_length=CL
        )
        assert chains.shape == (B, N, CL)
        assert params.shape == (B, N, 5)
        assert best_scores.shape == (B, N)

    def test_params_all_zeros(self):
        """The params output tensor is always entirely zero."""
        B, N = 1, 6
        edge, mask = self._make_inputs(B, N)
        _, params, _ = batched_beam_search_seed_reconstruction(edge, mask)
        assert params.eq(0.0).all()

    def test_raises_when_hit_nb_less_than_beam_width(self):
        """ValueError is raised when hit_nb < beam_width."""
        att = torch.zeros(1, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        edge = _to_edge_batched(att)
        with pytest.raises(ValueError):
            batched_beam_search_seed_reconstruction(edge, mask, beam_width=3)

    def test_all_hits_invalid_gives_neg_inf_scores(self):
        """When valid_mask is all False, best_scores stays -inf for every position."""
        B, N = 1, 6
        edge, mask = self._make_inputs(B, N)
        mask[:] = False
        _, _, best_scores = batched_beam_search_seed_reconstruction(edge, mask)
        assert (best_scores == float("-inf")).all()

    def test_unused_chain_slots_are_minus_one(self):
        """Positions beyond the chain's length in best_chains are filled with -1."""
        B, N = 1, 6
        att = torch.full((B, N, N), -10.0)
        att[0, 0, 1] = 10.0
        att[0, 1, 2] = 10.0
        mask = torch.ones(B, N, dtype=torch.bool)
        edge = _to_edge_batched(att)
        chains, _, _ = batched_beam_search_seed_reconstruction(
            edge, mask, max_chain_length=5, beam_width=2
        )
        chain = chains[0, 0, :]
        assert chain[3].item() == -1
        assert chain[4].item() == -1

    def test_forward_chains_have_increasing_indices(self):
        """In forward mode, all non-(-1) indices in best_chains are strictly increasing."""
        B, N = 1, 8
        att = torch.full((B, N, N), -10.0)
        for i in range(N - 1):
            att[0, i, i + 1] = 10.0
        mask = torch.ones(B, N, dtype=torch.bool)
        edge = _to_edge_batched(att)
        chains, _, best_scores = batched_beam_search_seed_reconstruction(
            edge, mask, max_chain_length=5, backward=False
        )
        assert best_scores[0, 0] > float("-inf")
        chain = chains[0, 0, :]
        valid = chain[chain >= 0].tolist()
        assert len(valid) >= 3
        for a, b in zip(valid[:-1], valid[1:]):
            assert b > a

    def test_backward_chains_have_decreasing_indices(self):
        """In backward mode, all non-(-1) indices in best_chains are strictly decreasing."""
        B, N = 1, 8
        att = torch.full((B, N, N), -10.0)
        for i in range(1, N):
            att[0, i, i - 1] = 10.0
        mask = torch.ones(B, N, dtype=torch.bool)
        edge = _to_edge_batched(att)
        chains, _, best_scores = batched_beam_search_seed_reconstruction(
            edge, mask, max_chain_length=5, backward=True
        )
        assert best_scores[0, N - 1] > float("-inf")
        chain = chains[0, N - 1, :]
        valid = chain[chain >= 0].tolist()
        assert len(valid) >= 3
        for a, b in zip(valid[:-1], valid[1:]):
            assert b < a

    def test_valid_mask_excludes_padded_hits(self):
        """Hits where valid_mask=False are excluded as destinations and have no valid chain."""
        B, N = 1, 7
        att = torch.full((B, N, N), -10.0)
        for i in range(N - 1):
            att[0, i, i + 1] = 10.0
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 5] = False
        edge = _to_edge_batched(att)
        chains, _, best_scores = batched_beam_search_seed_reconstruction(edge, mask)
        for n in range(N):
            if n != 5:
                assert 5 not in chains[0, n, :].tolist()
        assert best_scores[0, 5] == float("-inf")

    def test_att_threshold_masks_weak_edges(self):
        """Edges with attention below att_threshold are masked, preventing chains."""
        B, N = 1, 6
        edge, mask = self._make_inputs(B, N)
        # All attention scores are 0.0; threshold of 1.0 masks everything
        _, _, best_scores = batched_beam_search_seed_reconstruction(
            edge, mask, att_threshold=1.0
        )
        assert (best_scores == float("-inf")).all()

    def test_batch_bins_processed_independently(self):
        """Bins in the same batch are processed independently."""
        B, N = 2, 6
        att = torch.full((B, N, N), -10.0)
        mask = torch.ones(B, N, dtype=torch.bool)
        for i in range(N - 1):
            att[0, i, i + 1] = 10.0
        mask[1, :] = False
        edge = _to_edge_batched(att)
        _, _, best_scores = batched_beam_search_seed_reconstruction(edge, mask, backward=False)
        assert best_scores[0, 0] > float("-inf")
        assert (best_scores[1] == float("-inf")).all()


class TestBuildSeedFeaturesTensor:
    """Tests for build_seed_features_tensor."""

    def _make_hits(self, N: int = 5, num_features: int = 6) -> torch.Tensor:
        """Return a deterministic hits tensor [N, num_features]."""
        torch.manual_seed(42)
        return torch.rand(N, num_features)

    def test_output_shape_default(self):
        """Default indices: F = 6 features + 1 cosine expansion = 7 columns."""
        hits = self._make_hits(N=10, num_features=6)
        seeds = torch.tensor([[0, 1, 2], [3, 4, -1]])
        out = build_seed_features_tensor(hits, seeds)
        assert out.shape == (2, 3, 7)

    def test_output_shape_custom_indices(self):
        """Three direct features, no cosine expansion -> F = 3."""
        hits = self._make_hits(N=8, num_features=6)
        seeds = torch.tensor([[0, 1, 2, 3]])
        out = build_seed_features_tensor(hits, seeds, feature_indices=[0, 1, 2], cosine_feature_indices=[])
        assert out.shape == (1, 4, 3)

    def test_output_shape_multiple_cosine(self):
        """Two cosine-processed features -> each adds one extra column."""
        hits = self._make_hits(N=8, num_features=6)
        seeds = torch.tensor([[0, 1]])
        # feature_indices=[0,1,2], cosine on [0,1] -> columns: cos0,sin0,cos1,sin1,2 -> F=5
        out = build_seed_features_tensor(hits, seeds, feature_indices=[0, 1, 2], cosine_feature_indices=[0, 1])
        assert out.shape == (1, 2, 5)

    def test_padding_slots_are_zero(self):
        """Slots with hit index -1 must be all-zero regardless of the hit at index 0."""
        hits = torch.ones(5, 6) * 99.0  # extreme values to catch any leakage
        seeds = torch.tensor([[0, -1, 2]])
        out = build_seed_features_tensor(hits, seeds)
        assert out[0, 1, :].eq(0.0).all()

    def test_non_padding_slots_are_not_zero(self):
        """Non-padding slots for a non-zero hit must not be all zero."""
        hits = self._make_hits(N=5, num_features=6)
        seeds = torch.tensor([[0, 1]])
        out = build_seed_features_tensor(hits, seeds)
        assert not out[0, 0, :].eq(0.0).all()
        assert not out[0, 1, :].eq(0.0).all()

    def test_all_padding(self):
        """A fully-padded seed yields an all-zero feature row."""
        hits = torch.ones(3, 6) * 5.0
        seeds = torch.tensor([[-1, -1, -1]])
        out = build_seed_features_tensor(hits, seeds)
        assert out.eq(0.0).all()

    def test_direct_feature_values(self):
        """Direct (non-cosine) features are copied verbatim from hits_tensor."""
        hits = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        seeds = torch.tensor([[1, 3]])
        out = build_seed_features_tensor(hits, seeds, feature_indices=[0, 2], cosine_feature_indices=[])
        # Hit 1 -> features [5,6,7,8,9]; col 0 = 5, col 2 = 7
        assert out[0, 0, 0].item() == pytest.approx(hits[1, 0].item())
        assert out[0, 0, 1].item() == pytest.approx(hits[1, 2].item())
        # Hit 3 -> features [15,16,17,18,19]; col 0 = 15, col 2 = 17
        assert out[0, 1, 0].item() == pytest.approx(hits[3, 0].item())
        assert out[0, 1, 1].item() == pytest.approx(hits[3, 2].item())

    def test_cosine_feature_values(self):
        """Cosine-processed features produce cos then sin of the raw value."""
        phi_val = 1.2
        hits = torch.zeros(2, 6)
        hits[0, 4] = phi_val  # phi at column 4
        seeds = torch.tensor([[0]])
        out = build_seed_features_tensor(hits, seeds, feature_indices=[4], cosine_feature_indices=[4])
        assert out.shape == (1, 1, 2)
        assert out[0, 0, 0].item() == pytest.approx(torch.cos(torch.tensor(phi_val)).item(), rel=1e-5)
        assert out[0, 0, 1].item() == pytest.approx(torch.sin(torch.tensor(phi_val)).item(), rel=1e-5)

    def test_feature_ordering_preserved(self):
        """Features appear in the same order as feature_indices, with cos/sin in-place."""
        # feature_indices=[0, 4, 2], cosine on [4] -> output order: f0, cos(f4), sin(f4), f2
        hits = torch.zeros(1, 6)
        hits[0, 0] = 1.0
        hits[0, 2] = 3.0
        hits[0, 4] = 0.5
        seeds = torch.tensor([[0]])
        out = build_seed_features_tensor(hits, seeds, feature_indices=[0, 4, 2], cosine_feature_indices=[4])
        assert out.shape == (1, 1, 4)
        assert out[0, 0, 0].item() == pytest.approx(1.0)               # direct f0
        assert out[0, 0, 1].item() == pytest.approx(torch.cos(torch.tensor(0.5)).item(), rel=1e-5)  # cos(f4)
        assert out[0, 0, 2].item() == pytest.approx(torch.sin(torch.tensor(0.5)).item(), rel=1e-5)  # sin(f4)
        assert out[0, 0, 3].item() == pytest.approx(3.0)               # direct f2

    def test_multiple_seeds_independent(self):
        """Each seed row gathers its own hits independently."""
        hits = torch.eye(6)  # each hit is a one-hot vector
        seeds = torch.tensor([[0, 1], [4, 5]])
        out = build_seed_features_tensor(hits, seeds, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[])
        # Seed 0, slot 0 -> hit 0's row
        assert out[0, 0, :].tolist() == pytest.approx(hits[0, :].tolist())
        # Seed 1, slot 1 -> hit 5's row
        assert out[1, 1, :].tolist() == pytest.approx(hits[5, :].tolist())

    def test_output_dtype_float(self):
        """Output tensor has float dtype matching the hits tensor."""
        hits = self._make_hits().float()
        seeds = torch.tensor([[0, 1]])
        out = build_seed_features_tensor(hits, seeds)
        assert out.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
