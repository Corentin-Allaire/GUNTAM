"""Shared color palette, rcParams, and the (bug-fixed) drawing routines for efficiency/profile panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import BoxStyle

RED = "#D1335B"
GREEN = "#03BD5B"
ORANGE = "#FF9947"
DARK = "#223A5A"
GRAY = "#929292"
LEG_BG = "#F4F8FB"
LEG_ED = "#D7E3EC"

PALETTE = (RED, GREEN, ORANGE, DARK)
MARKERS = ("o", "s", "^", "D", "v", "<")

FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ROW_LABEL = 19, 21, 25, 17, 28


def apply_style() -> None:
    """Apply the shared report-ready rcParams (ported from clean-file-poltting-example.py)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def assign_series_styles(n_series: int) -> list[tuple[str, str]]:
    """Return n_series (color, marker) pairs: fixed palette for n<=4, viridis sampling beyond that."""
    if n_series <= len(PALETTE):
        return [(PALETTE[i], MARKERS[i % len(MARKERS)]) for i in range(n_series)]

    cmap = plt.colormaps["viridis"]
    positions = np.linspace(0.0, 1.0, n_series)
    return [(cmap(pos), MARKERS[i % len(MARKERS)]) for i, pos in enumerate(positions)]


@dataclass(frozen=True)
class KeyStyle:
    label: str
    xlabel: str
    ylabel: str
    xlim: Optional[tuple[float, float]] = None
    fold_abs: bool = False


@dataclass(frozen=True)
class AxisStyle:
    xlabel: str
    xlim: Optional[tuple[float, float]] = None
    fold_abs: bool = False


# Physics-motivated x-axis decisions (fixed acceptance ranges, d0-folding rationale) live here,
# keyed by the ROOT key's "_vs_<axis>" suffix. One entry covers every metric plotted against
# that axis (trackeff_vs_eta, fakeRatio_vs_eta, nHoles_vs_eta, ... all share the "eta" entry).
AXIS_STYLES: dict[str, AxisStyle] = {
    "eta": AxisStyle(r"$\eta$", xlim=(-3.0, 3.0)),
    "phi": AxisStyle(r"$\phi$ [rad]", xlim=(-3.15, 3.15)),
    "d0": AxisStyle(r"$d_0$ [mm]", xlim=(0.0, 10.0), fold_abs=True),
    "pT": AxisStyle(r"$p_\mathrm{T}$ [GeV]"),
    "z0": AxisStyle(r"$z_0$ [mm]"),
}

# Pretty y-axis label per metric prefix, keyed by the ROOT key's "<metric>_vs_" prefix.
METRIC_LABELS: dict[str, str] = {
    "trackeff": "Track efficiency",
    "fakeRatio": "Fake ratio",
    "duplicationRatio": "Duplication ratio",
    "nDuplicated": "Mean duplicated tracks",
    "nHoles": "Mean holes",
    "nMeasurements": "Mean measurements",
    "nOutliers": "Mean outliers",
    "nSharedHits": "Mean shared hits",
    "nStates": "Mean states",
    "completeness": "Completeness",
    "purity": "Purity",
}


def get_key_style(key: str) -> KeyStyle:
    """Compose a KeyStyle from METRIC_LABELS (y-axis) and AXIS_STYLES (x-axis), split on '_vs_'.

    Falls back to the raw metric/axis substring for anything not in those tables, so unlisted
    quantities (e.g. a new ptRange split) still get a readable label instead of a code change.
    """
    base_key = key.split(";")[0]
    if "_vs_" not in base_key:
        return KeyStyle(label=base_key, xlabel=base_key, ylabel=base_key)

    metric, axis = base_key.split("_vs_", 1)
    ylabel = METRIC_LABELS.get(metric, metric)
    axis_style = AXIS_STYLES.get(axis)
    label = f"{ylabel} vs {axis}"

    if axis_style is None:
        return KeyStyle(label=label, xlabel=axis, ylabel=ylabel)

    return KeyStyle(
        label=label,
        xlabel=axis_style.xlabel,
        ylabel=ylabel,
        xlim=axis_style.xlim,
        fold_abs=axis_style.fold_abs,
    )


class EfficiencySeries(NamedTuple):
    label: str
    x: np.ndarray
    eff: np.ndarray
    err_lo: np.ndarray
    err_hi: np.ndarray
    color: object
    marker: str


class ProfileSeries(NamedTuple):
    label: str
    x: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    color: object
    marker: str


def _style_panel_axes(
    ax,
    key_style: KeyStyle,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    title: Optional[str],
) -> None:
    if key_style.xlim is not None:
        ax.set_xlim(*key_style.xlim)

    ax.axvline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.4, zorder=2)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.spines[["top", "right"]].set_visible(False)

    if show_xlabel:
        ax.set_xlabel(key_style.xlabel, fontsize=FS_LABEL, fontweight="bold", labelpad=4)
    if show_ylabel:
        ax.set_ylabel(key_style.ylabel, fontsize=FS_LABEL, fontweight="bold", labelpad=4)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=8)
    ax.tick_params(labelsize=FS_TICK)


def draw_efficiency_panel(
    ax,
    series: list[EfficiencySeries],
    key_style: KeyStyle,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw efficiency series with the correct y-limit: uses the full error-bar extent, not just point values."""
    lows = []
    highs = []
    for s in series:
        ax.errorbar(
            s.x,
            s.eff,
            yerr=[s.err_lo, s.err_hi],
            fmt=s.marker,
            color=s.color,
            markersize=5,
            linewidth=0,
            elinewidth=1.4,
            capsize=2.5,
            label=s.label,
            alpha=0.90,
            zorder=5,
        )
        lows.append(s.eff - s.err_lo)
        highs.append(s.eff + s.err_hi)

    all_lows = np.concatenate(lows) if lows else np.array([])
    all_highs = np.concatenate(highs) if highs else np.array([])
    all_lows = all_lows[~np.isnan(all_lows)]
    all_highs = all_highs[~np.isnan(all_highs)]

    y_lo = max(0.0, np.min(all_lows) - 0.025) if all_lows.size else 0.0
    y_hi = min(1.0, np.max(all_highs) + 0.015) if all_highs.size else 1.0
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    _style_panel_axes(ax, key_style, show_xlabel=show_xlabel, show_ylabel=show_ylabel, title=title)


def draw_profile_panel(
    ax,
    series: list[ProfileSeries],
    key_style: KeyStyle,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw TProfile series with the same class of y-limit fix: uses values +/- errors extent."""
    lows = []
    highs = []
    for s in series:
        mask = ~np.isnan(s.values)
        ax.errorbar(
            s.x[mask],
            s.values[mask],
            yerr=s.errors[mask],
            fmt=s.marker,
            color=s.color,
            markersize=5,
            linewidth=0,
            elinewidth=1.4,
            capsize=2.5,
            label=s.label,
            alpha=0.90,
            zorder=5,
        )
        lows.append(s.values[mask] - s.errors[mask])
        highs.append(s.values[mask] + s.errors[mask])

    all_lows = np.concatenate(lows) if lows else np.array([])
    all_highs = np.concatenate(highs) if highs else np.array([])
    all_lows = all_lows[~np.isnan(all_lows)]
    all_highs = all_highs[~np.isnan(all_highs)]

    if all_lows.size and all_highs.size:
        span = all_highs.max() - all_lows.min()
        pad = 0.05 * span if span > 0 else 0.05
        ax.set_ylim(all_lows.min() - pad, all_highs.max() + pad)

    _style_panel_axes(ax, key_style, show_xlabel=show_xlabel, show_ylabel=show_ylabel, title=title)


def add_legend(ax, loc: str = "upper right") -> None:
    leg = ax.legend(
        loc=loc,
        frameon=True,
        facecolor=LEG_BG,
        edgecolor=LEG_ED,
        framealpha=1.0,
        fancybox=True,
        fontsize=FS_LEGEND,
        borderpad=0.5,
        labelspacing=0.45,
        handlelength=1.4,
        handletextpad=0.5,
    )
    frame = leg.get_frame()
    frame.set_linewidth(1.2)
    frame.set_boxstyle(BoxStyle.Round(pad=0.1, rounding_size=0.25))
    leg.set_zorder(200)
