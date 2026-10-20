import math

import numpy as np
import pytest
import pandas as pd
import torch
from GUNTAM.Transformer.BinTensor import (
    global_bin,
    global_bin_torch,
    margin_bin,
    margin_bin_torch,
    neighbor_bin,
    neighbor_bin_torch,
    neighbor_bin_Z_phi,
    neighbor_bin_Z_phi_torch,
    no_bin,
    no_bin_torch,
)


class TestNoBin:
    """Tests for the no_bin function."""

    def test_no_bin_returns_zeros(self):
        """Test that no_bin returns all zeros for bins."""
        values = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        bins_df, num_bins = no_bin(values)

        assert num_bins == 1
        assert bins_df.shape == (5, 3)
        assert (bins_df["bin0"] == 0).all()
        assert (bins_df["bin1"] == 0).all()
        assert (bins_df["bin2"] == 0).all()

    def test_no_bin_invalid_shape(self):
        """Test that no_bin raises error for invalid input shape."""
        values = pd.DataFrame({"value1": [1.0, 2.0], "value2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Input values must have shape \\[N, 1\\]"):
            no_bin(values)


class TestGlobalBin:
    """Tests for the global_bin function."""

    def test_global_bin_basic(self):
        """Test basic global binning functionality."""
        values = pd.DataFrame({"value": [0.0, 0.5, 1.0, 1.5, 2.0]})
        bins_df, num_bins = global_bin(values, bin_width=1.0, value_range=(0.0, 2.0))

        assert num_bins == 2
        assert bins_df.shape == (5, 3)
        # All three bin columns should have the same values
        assert (bins_df["bin0"] == bins_df["bin1"]).all()
        assert (bins_df["bin1"] == bins_df["bin2"]).all()

    def test_global_bin_proper_number_of_bins(self):
        """Test that global_bin creates the correct number of bins."""
        values = pd.DataFrame({"value": [0.0, 5.0, 10.0]})

        # Test case 1: Range 0-10 with bin_width 2 should give 5 bins
        _, num_bins = global_bin(values, bin_width=2.0, value_range=(0.0, 10.0))
        assert num_bins == 5

        # Test case 2: Range 0-10 with bin_width 3 should give 4 bins (ceil(10/3))
        _, num_bins = global_bin(values, bin_width=3.0, value_range=(0.0, 10.0))
        assert num_bins == 4

        # Test case 3: Range -5 to 5 with bin_width 2 should give 5 bins
        _, num_bins = global_bin(values, bin_width=2.0, value_range=(-5.0, 5.0))
        assert num_bins == 5

    def test_global_bin_boundary_values(self):
        """Test that boundary values are binned correctly."""
        values = pd.DataFrame({"value": [0.0, 5.0, 10.0]})
        bins_df, num_bins = global_bin(values, bin_width=5.0, value_range=(0.0, 10.0))

        assert num_bins == 2
        # 0.0 should be in bin 0, 5.0 at boundary should be in bin 1, 10.0 should be in bin 1 (clipped to max)
        assert bins_df.iloc[0]["bin1"] == 0
        assert bins_df.iloc[1]["bin1"] == 1  # 5.0 is at the boundary
        assert bins_df.iloc[2]["bin1"] == 1

    def test_global_bin_invalid_shape(self):
        """Test that global_bin raises error for invalid input shape."""
        values = pd.DataFrame({"value1": [1.0, 2.0], "value2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Input values must have shape \\[N, 1\\]"):
            global_bin(values, bin_width=1.0, value_range=(0.0, 5.0))

    def test_global_bin_invalid_range(self):
        """Test that global_bin raises error for invalid value range."""
        values = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="max_value must be greater than min_value"):
            global_bin(values, bin_width=1.0, value_range=(5.0, 5.0))

        with pytest.raises(ValueError, match="max_value must be greater than min_value"):
            global_bin(values, bin_width=1.0, value_range=(5.0, 2.0))

    def test_global_bin_invalid_bin_width(self):
        """Test that global_bin raises error for invalid bin width."""
        values = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="bin_width must be greater than 0"):
            global_bin(values, bin_width=0.0, value_range=(0.0, 5.0))

        with pytest.raises(ValueError, match="bin_width must be greater than 0"):
            global_bin(values, bin_width=-1.0, value_range=(0.0, 5.0))

    def test_global_bin_values_outside_range(self):
        """Test that values outside the range are clipped to valid bins."""
        values = pd.DataFrame({"value": [-5.0, 0.0, 5.0, 15.0]})
        bins_df, num_bins = global_bin(values, bin_width=2.0, value_range=(0.0, 10.0))

        # All bin indices should be within [0, num_bins-1]
        assert (bins_df["bin1"] >= 0).all()
        assert (bins_df["bin1"] < num_bins).all()


class TestNeighborBin:
    """Tests for the neighbor_bin function."""

    def test_neighbor_bin_basic(self):
        """Test basic neighbor binning functionality."""
        values = pd.DataFrame({"value": [1.0, 3.0, 5.0]})
        bins_df, num_bins = neighbor_bin(values, bin_width=2.0, value_range=(0.0, 6.0))

        assert num_bins == 3
        assert bins_df.shape == (3, 3)

        # Check that bin0 < bin1 < bin2 (modulo wrapping)
        for _, row in bins_df.iterrows():
            assert row["bin0"] == (row["bin1"] - 1) % num_bins
            assert row["bin2"] == (row["bin1"] + 1) % num_bins

    def test_neighbor_bin_edge_wrapping(self):
        """Test that neighbor bins wrap around at the edges of the value range."""
        values = pd.DataFrame({"value": [0.5, 9.5]})
        bins_df, num_bins = neighbor_bin(values, bin_width=2.0, value_range=(0.0, 10.0))

        assert num_bins == 5

        # For value 0.5 (bin 0), neighbors should be 4 (wraps) and 1
        first_bin = bins_df.iloc[0]
        assert first_bin["bin1"] == 0
        assert first_bin["bin0"] == 4  # Should wrap to 4
        assert first_bin["bin2"] == 1

        # For value 9.5 (bin 4), neighbors should be 3 and 0 (wraps)
        last_bin = bins_df.iloc[1]
        assert last_bin["bin1"] == 4
        assert last_bin["bin0"] == 3
        assert last_bin["bin2"] == 0  # Should wrap to 0

    def test_neighbor_bin_middle_bins(self):
        """Test that middle bins have correct neighbors without wrapping."""
        values = pd.DataFrame({"value": [2.5]})  # Should be in bin 1 (range 2-4)
        bins_df, num_bins = neighbor_bin(values, bin_width=2.0, value_range=(0.0, 10.0))

        assert num_bins == 5
        row = bins_df.iloc[0]
        assert row["bin1"] == 1
        assert row["bin0"] == 0
        assert row["bin2"] == 2

    def test_neighbor_bin_invalid_shape(self):
        """Test that neighbor_bin raises error for invalid input shape."""
        values = pd.DataFrame({"value1": [1.0, 2.0], "value2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Input values must have shape \\[N, 1\\]"):
            neighbor_bin(values, bin_width=1.0, value_range=(0.0, 5.0))


class TestMarginBin:
    """Tests for the margin_bin function."""

    def test_margin_bin_basic(self):
        """Test basic margin binning functionality."""
        values = pd.DataFrame({"value": [1.0, 3.0, 5.0]})
        bins_df, num_bins = margin_bin(values, bin_width=2.0, margin=0.5, value_range=(0.0, 6.0))

        assert num_bins == 3
        assert bins_df.shape == (3, 3)

    def test_margin_bin_start_margin(self):
        """Test that values in start margin get neighboring bin."""
        # Value 0.2 is in bin 0 [0-2], within start margin (< 0.5)
        values = pd.DataFrame({"value": [0.2]})
        bins_df, num_bins = margin_bin(values, bin_width=2.0, margin=0.5, value_range=(0.0, 10.0))

        row = bins_df.iloc[0]
        assert row["bin1"] == 0
        # bin0 should be set to previous bin (clipped to 0 since we're at the start)
        assert row["bin0"] == 4  # Can't go below 0

    def test_margin_bin_end_margin(self):
        """Test that values in end margin get neighboring bin."""
        # Value 1.8 is in bin 0 [0-2], within end margin (> 2-0.5 = 1.5)
        values = pd.DataFrame({"value": [1.8]})
        bins_df, num_bins = margin_bin(values, bin_width=2.0, margin=0.5, value_range=(0.0, 10.0))

        row = bins_df.iloc[0]
        assert row["bin0"] == 0
        assert row["bin1"] == 0
        # bin2 should be set to next bin
        assert row["bin2"] == 1

    def test_margin_bin_middle_of_bin(self):
        """Test that values in middle of bin have same bin for all three."""
        # Value 1.0 is exactly in middle of bin 0 [0-2], not in any margin
        values = pd.DataFrame({"value": [1.0]})
        bins_df, num_bins = margin_bin(values, bin_width=2.0, margin=0.3, value_range=(0.0, 10.0))

        row = bins_df.iloc[0]
        assert row["bin1"] == 0
        # Should have default values (same as bin1)
        assert row["bin0"] == 0
        assert row["bin2"] == 0

    def test_margin_bin_proper_number_of_bins(self):
        """Test that margin_bin creates the correct number of bins."""
        values = pd.DataFrame({"value": [0.0, 5.0, 10.0]})

        _, num_bins = margin_bin(values, bin_width=2.0, margin=0.5, value_range=(0.0, 10.0))
        assert num_bins == 5

    def test_margin_bin_invalid_margin(self):
        """Test that margin_bin raises error for invalid margin values."""
        values = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        # Margin cannot be negative
        with pytest.raises(ValueError, match="Margin must be non-negative"):
            margin_bin(values, bin_width=2.0, margin=-0.5, value_range=(0.0, 10.0))

        # Margin must be less than half of bin width
        with pytest.raises(ValueError, match="less than half of the bin width"):
            margin_bin(values, bin_width=2.0, margin=1.0, value_range=(0.0, 10.0))

        with pytest.raises(ValueError, match="less than half of the bin width"):
            margin_bin(values, bin_width=2.0, margin=1.5, value_range=(0.0, 10.0))

    def test_margin_bin_invalid_shape(self):
        """Test that margin_bin raises error for invalid input shape."""
        values = pd.DataFrame({"value1": [1.0, 2.0], "value2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Input values must have shape \\[N, 1\\]"):
            margin_bin(values, bin_width=2.0, margin=0.5, value_range=(0.0, 5.0))

    def test_margin_bin_at_boundaries(self):
        """Test margin binning behavior at value range boundaries."""
        # Test near the end of range
        values = pd.DataFrame({"value": [9.8]})  # Near end of [0, 10]
        bins_df, num_bins = margin_bin(values, bin_width=2.0, margin=0.3, value_range=(0.0, 10.0))

        row = bins_df.iloc[0]
        assert row["bin1"] == 4  # Last bin
        # Should be in end margin, so bin2 should be clipped to max
        assert row["bin2"] == 0


class TestNeighborBinZPhi:
    """Tests for the neighbor_bin_Z_phi function."""

    PHI_RANGE = (-math.pi, math.pi)
    BIN_WIDTH = math.pi / 2  # 4 phi bins
    ETA_EDGES = [1.0, 2.0]
    MARGIN = 0.1

    def _call(self, hits: pd.DataFrame):
        return neighbor_bin_Z_phi(hits, self.BIN_WIDTH, self.PHI_RANGE, self.ETA_EDGES, self.MARGIN)

    def test_output_shape_and_num_bins(self):
        """Test the number of output columns and the total number of bins."""
        hits = pd.DataFrame({"phi": [0.1, -2.0, 3.0], "z": [0.0, 300.0, -500.0], "r": [150.0, 100.0, 200.0]})
        bins_df, num_bins = self._call(hits)

        # 4 phi bins * 2 eta bins * 2 sides
        assert num_bins == 16
        assert bins_df.shape == (3, 6 * len(self.ETA_EDGES))

    def test_input_columns_unchanged(self):
        """Test that the temporary eta columns are removed from the input."""
        hits = pd.DataFrame({"phi": [0.1], "z": [0.0], "r": [150.0]})
        self._call(hits)
        assert list(hits.columns) == ["phi", "z", "r"]

    def test_known_assignment(self):
        """Test a hit with known eta values (eta_neg=0.881, eta_pos=-0.881)."""
        hits = pd.DataFrame({"phi": [0.1], "z": [0.0], "r": [150.0]})
        bins_df, _ = self._call(hits)

        # phi bin 2; assigned to the first positive and the first negative Z slot only
        expected = [1, 2, 3, -1, -1, -1, 9, 10, 11, -1, -1, -1]
        assert bins_df.iloc[0].tolist() == expected

    def test_hit_close_to_beam_assigned_to_all_slots(self):
        """Test that a hit at z=0 with a small r falls in every Z slot."""
        hits = pd.DataFrame({"phi": [0.1], "z": [0.0], "r": [10.0]})
        bins_df, _ = self._call(hits)
        assert (bins_df.iloc[0].values != -1).all()

    def test_unassigned_entries_are_triplets(self):
        """Test that each Z slot is either fully assigned or fully -1."""
        rng = np.random.default_rng(0)
        hits = pd.DataFrame(
            {
                "phi": rng.uniform(-math.pi, math.pi, 200),
                "z": rng.uniform(-1000, 1000, 200),
                "r": rng.uniform(30, 1000, 200),
            }
        )
        bins_df, num_bins = self._call(hits)
        slots = bins_df.values.reshape(200, -1, 3)
        assigned = (slots != -1).all(axis=2)
        unassigned = (slots == -1).all(axis=2)
        assert (assigned | unassigned).all()
        assert (slots[assigned] < num_bins).all()

    def test_eta_bins_are_sorted(self):
        """Test that the order of the eta edges does not matter."""
        hits = pd.DataFrame({"phi": [0.1, 1.0], "z": [0.0, 200.0], "r": [150.0, 50.0]})
        ref, _ = self._call(hits.copy())
        shuffled, _ = neighbor_bin_Z_phi(hits.copy(), self.BIN_WIDTH, self.PHI_RANGE, list(reversed(self.ETA_EDGES)), self.MARGIN)
        pd.testing.assert_frame_equal(ref, shuffled)


def _random_values(n: int = 500, seed: int = 0) -> np.ndarray:
    """Random 1D values, some outside of the [-4, 4] range to check the clipping."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-4.8, 4.8, n)


class TestTorchMatchesPandas:
    """The torch implementations must give the same bins as the pandas ones.

    Bin widths evenly divide the range, as the pandas global_bin splits the range in equal bins
    while the torch one uses bin_width directly.
    """

    VALUE_RANGE = (-4.0, 4.0)
    PHI_RANGE = (-math.pi, math.pi)

    def test_no_bin(self):
        values = _random_values()
        ref, n_ref = no_bin(pd.DataFrame({"value": values}))
        out, n_out = no_bin_torch(torch.tensor(values))

        assert n_ref == n_out
        assert np.array_equal(ref.values, out.numpy())

    @pytest.mark.parametrize("bin_width", [0.25, 0.5, 1.0, 2.0])
    def test_global_bin(self, bin_width):
        values = _random_values()
        ref, n_ref = global_bin(pd.DataFrame({"value": values}), bin_width, self.VALUE_RANGE)
        out, n_out = global_bin_torch(torch.tensor(values), bin_width, self.VALUE_RANGE)

        assert n_ref == n_out
        assert out.dtype == torch.long
        assert np.array_equal(ref.values, out.numpy())

    @pytest.mark.parametrize("bin_width", [0.25, 0.5, 1.0, 2.0])
    def test_neighbor_bin(self, bin_width):
        values = _random_values()
        ref, n_ref = neighbor_bin(pd.DataFrame({"value": values}), bin_width, self.VALUE_RANGE)
        out, n_out = neighbor_bin_torch(torch.tensor(values), bin_width, self.VALUE_RANGE)

        assert n_ref == n_out
        assert np.array_equal(ref.values, out.numpy())

    @pytest.mark.parametrize("bin_width, margin", [(0.5, 0.1), (1.0, 0.2), (2.0, 0.9)])
    def test_margin_bin(self, bin_width, margin):
        values = _random_values()
        ref, n_ref = margin_bin(pd.DataFrame({"value": values}), bin_width, margin, self.VALUE_RANGE)
        out, n_out = margin_bin_torch(torch.tensor(values), bin_width, margin, self.VALUE_RANGE)

        assert n_ref == n_out
        assert np.array_equal(ref.values, out.numpy())

    @pytest.mark.parametrize("bin_width", [math.pi / 4, math.pi / 2])
    @pytest.mark.parametrize("eta_edges", [[0.4, 0.9, 1.4, 2.5, 4.0], [1.0, 2.0]])
    def test_neighbor_bin_Z_phi(self, bin_width, eta_edges):
        rng = np.random.default_rng(1)
        n = 2000
        hits = pd.DataFrame(
            {
                "phi": rng.uniform(-math.pi, math.pi, n),
                "z": rng.uniform(-3000, 3000, n),
                "r": rng.uniform(30, 1000, n),
            }
        )
        ref, n_ref = neighbor_bin_Z_phi(hits.copy(), bin_width, self.PHI_RANGE, eta_edges, 0.05)
        out, n_out = neighbor_bin_Z_phi_torch(
            torch.tensor(hits[["phi", "z", "r"]].values), bin_width, self.PHI_RANGE, eta_edges, 0.05
        )

        assert n_ref == n_out
        assert out.shape == ref.shape
        assert np.array_equal(ref.values, out.numpy())

    def test_neighbor_bin_Z_phi_unsorted_edges_and_custom_threshold(self):
        rng = np.random.default_rng(2)
        n = 1000
        hits = pd.DataFrame(
            {
                "phi": rng.uniform(-math.pi, math.pi, n),
                "z": rng.uniform(-1000, 1000, n),
                "r": rng.uniform(30, 1000, n),
            }
        )
        edges = [2.5, 0.4, 1.4]
        ref, n_ref = neighbor_bin_Z_phi(hits.copy(), math.pi / 4, self.PHI_RANGE, edges, 0.1, Z_threashold=100)
        out, n_out = neighbor_bin_Z_phi_torch(
            torch.tensor(hits[["phi", "z", "r"]].values), math.pi / 4, self.PHI_RANGE, edges, 0.1, Z_threashold=100
        )

        assert n_ref == n_out
        assert np.array_equal(ref.values, out.numpy())

    def test_neighbor_bin_Z_phi_torch_invalid_shape(self):
        with pytest.raises(ValueError, match="Input values must have shape \\[N, 3\\]"):
            neighbor_bin_Z_phi_torch(torch.zeros(5, 2), 0.5, self.PHI_RANGE, [1.0], 0.1)
