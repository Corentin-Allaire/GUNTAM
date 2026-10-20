import math

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple


def no_bin(
    values: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    No binning, return 0 for all the bins.

    Args:
        values: DataFrame of shape [N, 1] containing the values to bin.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """
    if values.shape[1] != 1:
        raise ValueError(f"Input values must have shape [N, 1], got shape {values.shape}")

    bins_df = pd.DataFrame(
        {
            "bin0": np.zeros(len(values), dtype=int),
            "bin1": np.zeros(len(values), dtype=int),
            "bin2": np.zeros(len(values), dtype=int),
        }
    )
    return bins_df, 1


def no_bin_torch(
    values: Tensor,
) -> Tuple[Tensor, int]:
    """
    No binning, return 0 for all the bins.

      Args:
          values: Tensor of shape [N] or [N, 1] containing the values to bin.

      Returns:
          Tuple of (bins tensor of shape [N, 3] with dtype=torch.long, num_bins=1)
    """

    values = values.flatten()
    N = values.shape[0]
    bins = torch.zeros(N, 3, dtype=torch.long, device=values.device)
    return bins, 1


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
    if values.shape[1] != 1:
        raise ValueError(f"Input values must have shape [N, 1], got shape {values.shape}")

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


def global_bin_torch(
    values: Tensor,
    bin_width: float,
    value_range: Tuple[float, float],
) -> Tuple[Tensor, int]:
    """
    Global binning of the detector.

    Args:
        values: Tensor of shape [N] or [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (bins tensor of shape [N, 3] with dtype=torch.long, num_bins)
    """
    min_value, max_value = value_range
    range_width = max_value - min_value

    if range_width <= 0:
        raise ValueError("max_value must be greater than min_value")
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0")

    values = values.flatten()
    num_bins = int(math.ceil(range_width / bin_width))
    primary = ((values - min_value) / bin_width).long().clamp(0, num_bins - 1)
    bins = torch.stack([primary, primary, primary], dim=1)  # [N, 3]
    return bins, num_bins


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
    bin_ref = bin["bin1"].to_numpy()
    # Set bin0 to bin1-1 and bin2 to bin1+1 looping when the value are negative or above the max value
    bin["bin0"] = (bin_ref - 1) % num_bins
    bin["bin2"] = (bin_ref + 1) % num_bins

    return bin, num_bins


def neighbor_bin_torch(
    values: Tensor,
    bin_width: float,
    value_range: Tuple[float, float],
) -> Tuple[Tensor, int]:
    """
    Binning with neighboring bins. Each hit is also assigned to the bin immediately
    before (bin0) and after (bin2) its primary bin, wrapping at the boundaries.

    Args:
        values: Tensor of shape [N] or [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (bins tensor of shape [N, 3] with dtype=torch.long, num_bins)
    """
    bins, num_bins = global_bin_torch(values, bin_width, value_range)
    primary = bins[:, 1]
    bins[:, 0] = (primary - 1) % num_bins
    bins[:, 2] = (primary + 1) % num_bins
    return bins, num_bins


def margin_bin(
    values: pd.DataFrame,
    bin_width: float,
    margin: float,
    value_range: Tuple[float, float],
) -> Tuple[pd.DataFrame, int]:
    """
    Binning of the detector with margin. For each value, we compute the bin it falls into (bin1).
    If the value falls within margin of the start/end of the bin, we also include the neighboring bin:
    bin0 if in the start margin and bin2 if in the end margin.

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
    bin.loc[in_start_margin, "bin0"] = (bin_indices[in_start_margin] - 1) % num_bins

    # Check if value is in end margin (set bin2 to next bin)
    in_end_margin = position_in_bin > (bin_width - margin)
    bin.loc[in_end_margin, "bin2"] = (bin_indices[in_end_margin] + 1) % num_bins

    return bin, num_bins


def margin_bin_torch(
    values: Tensor,
    bin_width: float,
    margin: float,
    value_range: Tuple[float, float],
) -> Tuple[Tensor, int]:
    """
    Binning with margin. A hit is additionally assigned to the previous bin (bin0) if
    it falls within `margin` of the bin's lower edge, or to the next bin (bin2) if it
    falls within `margin` of the upper edge.

    Args:
        values: Tensor of shape [N] or [N, 1] containing the values to bin.
        bin_width: Width of each bin.
        margin: Distance from a bin edge within which the neighboring bin is also included.
        value_range: Tuple (min_value, max_value) defining the range of values.

    Returns:
        Tuple of (bins tensor of shape [N, 3] with dtype=torch.long, num_bins)
    """
    if margin < 0 or margin >= bin_width / 2:
        raise ValueError("Margin must be non-negative and less than half of the bin width")

    min_value, _ = value_range
    bins, num_bins = global_bin_torch(values, bin_width, value_range)
    primary = bins[:, 1]

    values_flat = values.flatten()
    bin_starts = min_value + primary.float() * bin_width
    position_in_bin = values_flat - bin_starts

    bins[:, 0] = torch.where(position_in_bin < margin, (primary - 1) % num_bins, primary)
    bins[:, 2] = torch.where(position_in_bin > (bin_width - margin), (primary + 1) % num_bins, primary)
    return bins, num_bins


def neighbor_bin_Z_phi(
    values: pd.DataFrame,
    bin_width: float,
    value_range_phi: Tuple[float, float],
    value_bins_eta: List[float],
    margin_eta: float,
    Z_threashold: float = 150,
) -> Tuple[pd.DataFrame, int]:
    """
    Binning of the detector in Z and phi with neighboring bins. For each hit, we compute the bin it falls into (bin1)
    and also the neighboring bins (bin0 and bin2) in phi. The Z binning is determined by the eta value of the hit,
    which is computed from the r and z coordinates. The eta value is computed for two reference Z positions:
    -Z_threashold and Z_threashold. The hit is assigned to the Z bin corresponding
    to the eta value that falls within the defined eta bins, considering a margin for neighboring bins.

    Args:
        values: DataFrame of shape [N, 3] containing the values to bin. The columns should be ["r", "z", "phi"].
        bin_width: Width of each bin in phi.
        value_range_phi: Tuple (min_phi, max_phi) defining the range of phi values.
        value_bins_eta: List of eta bin edges for Z binning.
        margin_eta: Margin to consider for neighboring bins in eta.
        Z_threashold: Z value that will be used to compute the 2 extreme eta values for each hit. Default is 150.

    Returns:
        Tuple of (binned dataframe with shape (N, 3), num_bins)
    """
    nb_z_bin = len(value_bins_eta)
    # reorder the value_range_Z from the smallest to the largest
    value_bins_eta = sorted(value_bins_eta)
    bin, num_bins = global_bin(values[["phi"]], bin_width, value_range_phi)
    tot_num_bins = num_bins * nb_z_bin * 2
    bin_ref = bin["bin1"].to_numpy()

    # Compute the eta value for each hit for an origin at -1*Z_threashold and Z_threashold
    values["eta_neg"] = -1 * np.log(np.tan(np.arctan2(values["r"], values["z"] + Z_threashold) / 2))
    values["eta_pos"] = -1 * np.log(np.tan(np.arctan2(values["r"], values["z"] - Z_threashold) / 2))
    condition = values["z"] >= -1 * Z_threashold

    # Loop to create the positive bin
    for z_bin in range(nb_z_bin):
        condition &= values["eta_pos"] <= value_bins_eta[z_bin] + margin_eta
        b0 = "bin" + str(3 * z_bin)
        b1 = "bin" + str(3 * z_bin + 1)
        b2 = "bin" + str(3 * z_bin + 2)
        bin[b0] = -1
        bin[b1] = -1
        bin[b2] = -1
        bin_ref_cond = bin_ref[condition.to_numpy()]
        bin.loc[condition, b0] = ((num_bins * z_bin) + (bin_ref_cond - 1)) % tot_num_bins
        bin.loc[condition, b1] = (num_bins * z_bin) + bin_ref_cond
        bin.loc[condition, b2] = ((num_bins * z_bin) + (bin_ref_cond + 1)) % tot_num_bins
        condition = values["eta_neg"] > value_bins_eta[z_bin] - margin_eta

    condition = values["z"] <= Z_threashold
    # Loop to create the negative bin
    for z_bin in range(nb_z_bin):
        condition &= -1 * values["eta_neg"] <= value_bins_eta[z_bin] + margin_eta
        b0 = "bin" + str(3 * (nb_z_bin + z_bin))
        b1 = "bin" + str(3 * (nb_z_bin + z_bin) + 1)
        b2 = "bin" + str(3 * (nb_z_bin + z_bin) + 2)
        bin[b0] = -1
        bin[b1] = -1
        bin[b2] = -1
        offset = num_bins * (nb_z_bin + z_bin)
        bin_ref_cond = bin_ref[condition.to_numpy()]
        bin.loc[condition, b0] = (offset + (bin_ref_cond - 1)) % tot_num_bins
        bin.loc[condition, b1] = offset + bin_ref_cond
        bin.loc[condition, b2] = (offset + (bin_ref_cond + 1)) % tot_num_bins
        condition = -1 * values["eta_pos"] > value_bins_eta[z_bin] + margin_eta

    values.drop(columns=["eta_neg", "eta_pos"], inplace=True)
    return bin, tot_num_bins


def neighbor_bin_Z_phi_torch(
    values: Tensor,
    bin_width: float,
    value_range_phi: Tuple[float, float],
    value_bins_eta: List[float],
    margin_eta: float,
    Z_threashold: float = 150,
) -> Tuple[Tensor, int]:
    """
    Binning of the detector in Z and phi with neighboring bins. For each hit, we compute the bin it falls into (bin1)
    and also the neighboring bins (bin0 and bin2) in phi. The Z binning is determined by the eta value of the hit,
    which is computed from the r and z coordinates. The eta value is computed for two reference Z positions:
    -Z_threashold and Z_threashold. The hit is assigned to the Z bin corresponding to the eta value that falls
    within the defined eta bins, considering a margin for neighboring bins.

    Args:
        values: Tensor of shape [N, 3] containing the values to bin. The columns should be [phi, z, r].
        bin_width: Width of each bin in phi.
        value_range_phi: Tuple (min_phi, max_phi) defining the range of phi values.
        value_bins_eta: List of eta bin edges for Z binning.
        margin_eta: Margin to consider for neighboring bins in eta.
        Z_threashold: Z value that will be used to compute the 2 extreme eta values for each hit. Default is 150.

    Returns:
        Tuple of (bins tensor of shape [N, 6 * len(value_bins_eta)] with dtype=torch.long, num_bins).
        Columns are triplets (bin0, bin1, bin2) per Z bin, positive Z bins first then negative; -1 means unassigned.
    """
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Input values must have shape [N, 3], got shape {tuple(values.shape)}")

    phi, z, r = values[:, 0], values[:, 1], values[:, 2]
    value_bins_eta = sorted(value_bins_eta)
    nb_z_bin = len(value_bins_eta)

    phi_bins, num_bins = global_bin_torch(phi, bin_width, value_range_phi)
    tot_num_bins = num_bins * nb_z_bin * 2
    bin_ref = phi_bins[:, 1]

    # eta of each hit for an origin at -Z_threashold and +Z_threashold
    eta_neg = -torch.log(torch.tan(torch.atan2(r, z + Z_threashold) / 2))
    eta_pos = -torch.log(torch.tan(torch.atan2(r, z - Z_threashold) / 2))

    edges = torch.tensor(value_bins_eta, dtype=eta_neg.dtype, device=phi.device)  # [K]
    eta_neg = eta_neg[:, None]  # [N, 1]
    eta_pos = eta_pos[:, None]  # [N, 1]

    # Z-bin k requires an upper cut on its own edge and a lower cut on the previous edge (z-sign cut for k=0)
    upper_pos = eta_pos <= edges + margin_eta  # [N, K]
    lower_pos = torch.cat([(z >= -Z_threashold)[:, None], eta_neg > edges[:-1] - margin_eta], dim=1)
    upper_neg = -eta_neg <= edges + margin_eta
    lower_neg = torch.cat([(z <= Z_threashold)[:, None], -eta_pos > edges[:-1] + margin_eta], dim=1)
    condition = torch.cat([lower_pos & upper_pos, lower_neg & upper_neg], dim=1)  # [N, 2K]

    # Each of the 2K Z slots owns a block of num_bins phi bins
    offset = num_bins * torch.arange(2 * nb_z_bin, device=phi.device)  # [2K]
    bin1 = offset + bin_ref[:, None]  # [N, 2K]
    bins = torch.stack([(bin1 - 1) % tot_num_bins, bin1, (bin1 + 1) % tot_num_bins], dim=2)  # [N, 2K, 3]
    bins = bins.masked_fill(~condition[:, :, None], -1).reshape(phi.shape[0], 6 * nb_z_bin)

    return bins, tot_num_bins
