import pytest
import pandas as pd
from GUNTAM.Transformer.BinTensor import global_bin, neighbor_bin, no_bin, margin_bin


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
