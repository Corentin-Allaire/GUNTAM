import pandas as pd
import numpy as np
from typing import Tuple


def no_bin(
    values: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    No binning, return 0 for all the bins.

    Args:
        values: DataFrame of shape [N] containing the values to bin.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """
    bins_df = pd.DataFrame(
        {
            "bin0": np.zeros(len(values), dtype=int),
            "bin1": np.zeros(len(values), dtype=int),
            "bin2": np.zeros(len(values), dtype=int),
        }
    )
    return bins_df, 1


def global_bin(
    values: pd.DataFrame,
    bin_width: float,
    value_range: Tuple[float, float],
) -> Tuple[pd.DataFrame, int]:
    """
    Global binning of the detector.

    Args:
        values: DataFrame of shape [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """
    min_value, max_value = value_range
    range_width = max_value - min_value

    if range_width <= 0:
        raise ValueError("max_value must be greater than min_value")
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0")

    num_bins = int(np.ceil(range_width / bin_width))
    bins = np.linspace(min_value, max_value, num_bins + 1)
    binned_values = np.digitize(values.values.flatten(), bins) - 1  # digitize returns 1-based indices
    binned_values = np.clip(binned_values, 0, num_bins - 1)  # Ensure values are within valid bin indices
    # Initialize all three columns with the same bin value
    bins_df = pd.DataFrame({"bin0": binned_values, "bin1": binned_values, "bin2": binned_values})
    return bins_df, num_bins


def neighbor_bin(
    values: pd.DataFrame,
    bin_width: float,
    value_range: Tuple[float, float],
) -> Tuple[pd.DataFrame, int]:
    """
    Binning of the detector with neighboring bins. For each value, we compute the bin it falls into (bin1)
    and also the neighboring bins (bin0 and bin2). This allows some redundancy and can help with edge cases
    where values are close to bin boundaries.

    Args:
        values: DataFrame of shape [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """
    bin, num_bins = global_bin(values, bin_width, value_range)
    # Set bin0 to bin1-1 and bin2 to bin1+1 looping when the value are negative or above the max value
    bin["bin0"] = (bin["bin1"] - 1) % num_bins
    bin["bin2"] = (bin["bin1"] + 1) % num_bins
    return bin, num_bins


def margin_bin(
    values: pd.DataFrame,
    bin_width: float,
    margin: float,
    value_range: Tuple[float, float],
) -> Tuple[pd.DataFrame, int]:
    """
    Binning of the detector with margin. For each value, we compute the bin it falls into (bin1).
    If the value falls within margin of the start/end of the bin, we also include the neighboring bin:
    bin0 if in the start margin and bin2 if in the end margin. This allows some redundancy and can help with edge cases.

    Args:
        values: DataFrame of shape [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        margin: Margin to consider for neighboring bins.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """

    if margin < 0 or margin >= bin_width / 2:
        raise ValueError("Margin must be non-negative and less than half of the bin width")

    min_value, _ = value_range
    bin, num_bins = global_bin(values, bin_width, value_range)

    # Calculate the position of each value within its bin
    values_array = values.values.flatten()
    bin_indices = bin["bin1"]

    # Calculate the start position of each bin
    bin_starts = min_value + bin_indices * bin_width

    # Calculate position within the bin (0 to bin_width)
    position_in_bin = values_array - bin_starts

    # Check if value is in start margin (set bin0 to previous bin)
    in_start_margin = position_in_bin < margin
    bin.loc[in_start_margin, "bin0"] = np.clip(bin_indices[in_start_margin] - 1, 0, bin_indices.max())

    # Check if value is in end margin (set bin2 to next bin)
    in_end_margin = position_in_bin > (bin_width - margin)
    bin.loc[in_end_margin, "bin2"] = np.clip(bin_indices[in_end_margin] + 1, 0, bin_indices.max())

    return bin, num_bins
