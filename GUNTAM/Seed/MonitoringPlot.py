from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


plt.switch_backend("Agg")


class PlotUtility:
    """Generic plotting utility class to reduce code duplication."""

    @staticmethod
    def create_histogram(
        ax: Axes,
        data: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        bins: int = 600,
        color: str = "skyblue",
        stats_text: Optional[str] = None,
        reference_lines: Optional[Sequence[Mapping[str, Any]]] = None,
        density: bool = False,
    ) -> None:
        if len(data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(title)
            return

        n_bins = min(bins, len(data) // 10) if len(data) > 10 else 10
        ax.hist(
            data,
            bins=n_bins,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            density=density,
        )

        x_min, x_max = np.percentile(data, [5, 95])
        ax.set_xlim(x_min, x_max)

        if stats_text:
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        if reference_lines:
            for line in reference_lines:
                ax.axvline(
                    line["value"],
                    color=line.get("color", "red"),
                    linestyle=line.get("style", "--"),
                    alpha=0.8,
                    label=line.get("label", ""),
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if reference_lines and any("label" in line for line in reference_lines):
            ax.legend()

    @staticmethod
    def create_scatter_2d(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        c: Optional[np.ndarray] = None,
        cmap: str = "viridis",
        s: float = 30,
        alpha: float = 0.7,
        colorbar_label: Optional[str] = None,
    ) -> None:
        if c is not None:
            scatter = ax.scatter(
                x,
                y,
                c=c,
                s=s,
                cmap=cmap,
                alpha=alpha,
                edgecolors="black",
                linewidth=0.5,
            )
            if colorbar_label:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(colorbar_label)
        else:
            ax.scatter(x, y, s=s, alpha=alpha, edgecolors="black", linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def create_2d_histogram(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        bins: Sequence[int] = (15, 20),
        cmap: str = "viridis",
        overlay_scatter: bool = True,
    ) -> Any:
        hist, xbins, ybins = np.histogram2d(x, y, bins=bins)

        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=(float(xbins[0]), float(xbins[-1]), float(ybins[0]), float(ybins[-1])),
            aspect="auto",
            cmap=cmap,
            alpha=0.8,
        )

        if overlay_scatter:
            ax.scatter(x, y, alpha=0.3, s=20, c="red", edgecolors="black", linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Bins")

        return im

    @staticmethod
    def create_bar_plot(
        ax: Axes,
        values: np.ndarray,
        counts: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str = "lightgreen",
    ) -> None:
        ax.bar(values, counts, alpha=0.7, color=color, edgecolor="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def add_stats_text(
        ax: Axes,
        data: np.ndarray,
        position: Tuple[float, float] = (0.95, 0.95),
        ha: str = "right",
        va: str = "top",
    ) -> None:
        if len(data) == 0:
            return

        stats_text = f"Mean: {np.mean(data):.2f}\n"
        stats_text += f"Std: {np.std(data):.2f}\n"
        if len(data) > 1:
            stats_text += f"Range: {np.min(data):.1f}-{np.max(data):.1f}\n"
        stats_text += f"Count: {len(data)}"

        ax.text(
            position[0],
            position[1],
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            horizontalalignment=ha,
            verticalalignment=va,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def visualize_attention_map(
    attention_weights: np.ndarray,
    pair_info: Mapping[Tuple[int, int], Mapping[str, Any]],
    valid_hits: np.ndarray,
    event_idx: int,
    bin_idx: int,
    max_hits: Optional[int] = None,
) -> None:
    """
    Visualize an attention-weight matrix for a bin, annotating positive pairs.

    Inputs
    - attention_weights: numpy.ndarray, shape (N, N), dtype float
        Square attention matrix for `N` hits in the bin. Values typically in [0, 1].
    - pair_info: Mapping[(int, int), Mapping[str, Any]]
        Metadata per hit index pair `(i, j)`; uses `"is_good"` to mark positives.
    - valid_hits: numpy.ndarray
        Zero-based hit indices aligned with the attention matrix (same order/length as its axes).
    - event_idx: int
        Event identifier for the figure title and output filename.
    - bin_idx: int
        Bin identifier for the figure title and output filename.
    - max_hits: Optional[int]
        If provided, crops the visualization to the first `max_hits` hits
        (top-left `max_hits x max_hits` sub-matrix).

    Plots
    - 2D image of the attention matrix with a colorbar.
    - Green check marks (✓) on diagonal cells and on pairs flagged as `is_good`.
    - Minor grid at cell boundaries; x-axis is query index, y-axis is key index.

    Output
    - Saves `attention_map_event_{event_idx}_bin_{bin_idx}.png`
        in the current working directory. Returns None.
    """

    print("\nATTENTION MAP VISUALIZATION:")

    if max_hits is not None and max_hits < attention_weights.shape[0]:
        attention_weights = attention_weights[:max_hits, :max_hits]
        valid_hits = valid_hits[:max_hits]
        print(f"Limiting visualization to first {max_hits} hits")

    try:
        print(f"Creating visualization for attention weights shape: {attention_weights.shape}")
        print(f"Attention weights dtype: {attention_weights.dtype}")
        print(f"Attention weights min/max: {np.min(attention_weights):.6f} / {np.max(attention_weights):.6f}")

        if np.any(np.isnan(attention_weights)):
            print("Warning: NaN values detected; replacing with small value")
            attention_weights = np.nan_to_num(attention_weights, nan=1e-2)
        if np.any(np.isinf(attention_weights)):
            print("Warning: Infinite values detected; clipping")
            attention_weights = np.clip(attention_weights, 1e-2, 1.0)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(
            attention_weights,
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=np.max(attention_weights) if np.max(attention_weights) > 0 else 1.0,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        positive_pairs = {k: v for k, v in pair_info.items() if v.get("is_good", False)}

        for i in range(min(attention_weights.shape)):
            ax.text(
                i,
                i,
                "✓",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="lime",
            )

        for (i, j), pair_data in positive_pairs.items():
            if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                color = "lime"
                symbol = "✓"
                ax.text(
                    j,
                    i,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )
                ax.text(
                    i,
                    j,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xlabel("Hit Index (Query)")
        ax.set_ylabel("Hit Index (Key)")
        title = f"Attention Map - Event {event_idx}, Bin {bin_idx}"
        title += "\nGreen ✓: Positive Pairs (Good Connections)"
        ax.set_title(title)

        ax.set_xticks(range(len(valid_hits)))
        ax.set_yticks(range(len(valid_hits)))
        ax.set_xticks(np.arange(-0.5, len(valid_hits), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(valid_hits), 1), minor=True)
        ax.grid(True, which="minor", alpha=0.3)

        plt.tight_layout()
        plot_filename = f"attention_map_event_{event_idx}_bin_{bin_idx}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print("Attention map saved as:", plot_filename)
        plt.close()
    except Exception as e:
        print(f"Error creating attention visualization: {e}")
        try:
            plt.close("all")
        except Exception:
            pass


def create_seeding_performance_plots(
    results: Mapping[str, Any],
    seed_metrics: Sequence[Mapping[str, Any]],
    error_arrays: Mapping[str, Sequence[float]],
    bin_summaries: Sequence[Mapping[str, Any]],
) -> None:
    """
    Visualize seed-level resolution, efficiency per bin, and hits-in-common stats.

    Inputs
    - results: Mapping[str, Any]
        Analysis results; if it contains `"bin_complexity_analysis"`, a
        complexity subplot collection is generated.
    - seed_metrics: Sequence[Mapping[str, Any]]
        Seed-particle association metrics; uses `"n_hits_common"` for distribution.
    - error_arrays: Mapping[str, Sequence[float]]
        Seed resolution errors keyed by parameter (`"z"`, `"eta"`, `"phi"`, `"pt"`).
        Units: `z` [mm], `eta` (unitless), `phi` [rad], `pt` [GeV]. Each array is shape (S,).
    - bin_summaries: Sequence[Mapping[str, Any]]
        Per-bin summaries including keys `"seeding_efficiency"`,
        optional `"pure_seeding_efficiency"`, `"n_particles"`, and `"n_seeds"`.

    Plots
    - 2x3 grid: four resolution histograms + efficiencies per bin + hits-in-common.
    - Efficiency panel compares regular vs pure efficiencies.

    Output
    - Saves `seeding_performance_analysis.png`. May also call `create_bin_complexity_plots`.
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    params = [
        {"key": "z", "label": "Z", "unit": "[mm]"},
        {"key": "eta", "label": "ETA", "unit": ""},
        {"key": "phi", "label": "PHI", "unit": "[rad]"},
        {"key": "pt", "label": "pT", "unit": "[GeV]"},
    ]

    error_map = {str(k).lower(): np.asarray(v) for k, v in error_arrays.items()}

    for i, meta in enumerate(params):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        errors_array = np.asarray(error_map.get(meta["key"], []))

        if errors_array.size > 0:
            stats_text = f"Mean: {np.mean(errors_array):.4f}\n"
            stats_text += f"Std: {np.std(errors_array):.4f}\n"
            stats_text += f"RMS: {np.sqrt(np.mean(errors_array**2)):.4f}\n"
            stats_text += f"Seeds: {len(errors_array)}"
            reference_lines = [
                {
                    "value": 0,
                    "color": "green",
                    "style": "-",
                    "label": "Perfect Reconstruction",
                },
                {
                    "value": np.mean(errors_array),
                    "color": "red",
                    "style": "--",
                    "label": "Mean Error",
                },
            ]
            PlotUtility.create_histogram(
                ax,
                errors_array,
                title=f"{meta['label']} Seed Resolution",
                xlabel=f"Seed Error: {meta['label']} {meta['unit']}",
                ylabel="Number of Seeds",
                color="lightcoral",
                stats_text=stats_text,
                reference_lines=reference_lines,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_title(f"{meta['label']} Seed Resolution")

    ax = axes[1, 2]
    efficiencies = [b["seeding_efficiency"] for b in bin_summaries]
    pure_efficiencies = [b.get("pure_seeding_efficiency", 0.0) for b in bin_summaries]
    all_vals = np.array(efficiencies + pure_efficiencies)
    bins = min(30, max(10, len(all_vals) // 10)) if len(all_vals) > 0 else 10
    ax.hist(
        efficiencies,
        bins=bins,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        linewidth=0.5,
        label=f"Regular (mean {np.mean(efficiencies):.1%} ± {np.std(efficiencies):.1%})",
    )
    ax.hist(
        pure_efficiencies,
        bins=bins,
        alpha=0.6,
        color="orange",
        edgecolor="black",
        linewidth=0.5,
        label=f"Pure (mean {np.mean(pure_efficiencies):.1%} ± {np.std(pure_efficiencies):.1%})",
    )
    ax.set_title("Bin-wise Seeding Efficiency (Regular vs Pure)")
    ax.set_xlabel("Seeding Efficiency")
    ax.set_ylabel("Number of Bins")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    hits_common = [s["n_hits_common"] for s in seed_metrics]
    if hits_common:
        hit_counts = np.bincount(hits_common)
        hit_values = np.arange(len(hit_counts))
        mask = hit_counts > 0
        PlotUtility.create_bar_plot(
            ax,
            hit_values[mask],
            hit_counts[mask],
            title="Distribution of Hits in Common",
            xlabel="Number of Hits in Common",
            ylabel="Number of Seed-Particle Associations",
            color="lightgreen",
        )
        stats_text = f"Mean: {np.mean(hits_common):.2f}\nMax: {np.max(hits_common)}\nAssociations: {len(hits_common)}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No seed-particle associations found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title("Distribution of Hits in Common")

    plt.tight_layout()
    plot_filename = "seeding_performance_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print("Seeding performance plots saved as:", plot_filename)
    plt.close()

    if "bin_complexity_analysis" in results:
        create_bin_complexity_plots(results["bin_complexity_analysis"], bin_summaries)


def create_bin_complexity_plots(
    complexity_analysis: Mapping[str, Any],
    bin_summaries: Sequence[Mapping[str, Any]],
) -> None:
    """
    Explore relationships between bin complexity and seeding efficiency.

    Inputs
    - complexity_analysis: Mapping[str, Any]
        Contains correlations, e.g.,
        `{"correlations": {"particle_seed_ratio_vs_efficiency": float}}`.
    - bin_summaries: Sequence[Mapping[str, Any]]
        Per-bin metrics with keys `"n_particles"`, `"n_seeds"`, `"seeding_efficiency"`.

    Plots
    - Heatmaps: efficiency vs particle count; efficiency vs seed count.
    - Scatter: seeds vs particles colored by efficiency.
    - Histograms: distributions of particles/bin and seeds/bin.
    - Efficiency vs seeds-per-particle ratio with correlation annotation.

    Output
    - Saves `bin_complexity_analysis.png`.
    """

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    n_particles = np.array([b["n_particles"] for b in bin_summaries])
    n_seeds = np.array([b["n_seeds"] for b in bin_summaries])
    efficiencies = np.array([b["seeding_efficiency"] for b in bin_summaries])

    ax = axes[0, 0]
    hist, xbins, ybins = np.histogram2d(n_particles, efficiencies, bins=[15, 20])
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
        aspect="auto",
        cmap="viridis",
        alpha=0.8,
    )
    ax.scatter(
        n_particles,
        efficiencies,
        alpha=0.3,
        s=20,
        c="red",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of Particles per Bin")
    ax.set_ylabel("Seeding Efficiency")
    ax.set_title("Seeding Efficiency vs Particle Count")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Bins")

    ax = axes[0, 1]
    hist, xbins, ybins = np.histogram2d(n_seeds, efficiencies, bins=[15, 20])
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
        aspect="auto",
        cmap="plasma",
        alpha=0.8,
    )
    ax.scatter(
        n_seeds,
        efficiencies,
        alpha=0.3,
        s=20,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of Seeds per Bin")
    ax.set_ylabel("Seeding Efficiency")
    ax.set_title("Seeding Efficiency vs Seed Count")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Bins")

    ax = axes[0, 2]
    scatter = ax.scatter(
        n_particles,
        n_seeds,
        c=efficiencies,
        s=30,
        cmap="RdYlBu_r",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of Particles per Bin")
    ax.set_ylabel("Number of Seeds per Bin")
    ax.set_title("Seeds vs Particles (colored by efficiency)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Seeding Efficiency")

    ax = axes[1, 0]
    ax.hist(n_particles, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
    ax.axvline(
        np.mean(n_particles),
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Mean: {np.mean(n_particles):.1f}",
    )
    ax.axvline(
        np.median(n_particles),
        color="darkred",
        linestyle="-",
        alpha=0.8,
        label=f"Median: {np.median(n_particles):.1f}",
    )
    ax.set_xlabel("Number of Particles per Bin")
    ax.set_ylabel("Number of Bins")
    ax.set_title("Distribution of Particles per Bin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(n_seeds, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
    ax.axvline(
        np.mean(n_seeds),
        color="green",
        linestyle="--",
        alpha=0.8,
        label=f"Mean: {np.mean(n_seeds):.1f}",
    )
    ax.axvline(
        np.median(n_seeds),
        color="darkgreen",
        linestyle="-",
        alpha=0.8,
        label=f"Median: {np.median(n_seeds):.1f}",
    )
    ax.set_xlabel("Number of Seeds per Bin")
    ax.set_ylabel("Number of Bins")
    ax.set_title("Distribution of Seeds per Bin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    seeds_per_particle = n_seeds / np.maximum(n_particles, 1)
    ratio_bins = np.linspace(0, np.percentile(seeds_per_particle, 95), 15)
    bin_indices = np.digitize(seeds_per_particle, ratio_bins)
    mean_effs = []
    std_effs = []
    bin_centers = []
    for i in range(1, len(ratio_bins)):
        mask = bin_indices == i
        if np.any(mask):
            mean_effs.append(np.mean(efficiencies[mask]))
            std_effs.append(np.std(efficiencies[mask]))
            bin_centers.append((ratio_bins[i - 1] + ratio_bins[i]) / 2)
    if len(mean_effs) > 0:
        ax.errorbar(
            bin_centers,
            mean_effs,
            yerr=std_effs,
            fmt="o-",
            capsize=5,
            capthick=2,
            alpha=0.8,
            color="purple",
        )
        ax.scatter(seeds_per_particle, efficiencies, alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Seeds per Particle Ratio")
    ax.set_ylabel("Seeding Efficiency")
    ax.set_title("Efficiency vs Seeds/Particles Ratio")
    ax.grid(True, alpha=0.3)
    corr = complexity_analysis["correlations"]["particle_seed_ratio_vs_efficiency"]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:+.3f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plot_filename = "bin_complexity_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print("Bin complexity plots saved as:", plot_filename)
    plt.close()


def create_particle_reconstruction_comparison_plots(eligible_particles: Sequence[Mapping[str, Any]]) -> None:
    """
    Compare distributions of truth parameters for particles with vs without seeds.

    Inputs
    - eligible_particles: Sequence[Mapping[str, Any]]
        Each mapping must include `"true_params"` (array-like length 4 `[z0, eta, phi, pT]`),
        `"n_hits"` (int), and `"had_seed"` (bool).

    Plots
    - Side-by-side histograms for `z0`, `eta`, `phi`, `pT`, and `n_hits` comparing
        particles that produced seeds vs those that did not, with basic stats.
    - Summary bar chart of counts and overall seeding efficiency.

    Output
    - Saves `particle_reconstruction_comparison.png`.
    """

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    param_names = ["z0", "eta", "phi", "pT", "n_hits"]
    param_units = ["[mm]", "", "[rad]", "[GeV]", ""]
    param_labels = ["Z0", "Eta", "Phi", "pT", "Number of Hits"]

    particles_with_seeds = [p for p in eligible_particles if p.get("had_seed", False)]
    particles_without_seeds = [p for p in eligible_particles if not p.get("had_seed", False)]

    with_seeds_data = {
        "z0": [p["true_params"][0] for p in particles_with_seeds],
        "eta": [p["true_params"][1] for p in particles_with_seeds],
        "phi": [p["true_params"][2] for p in particles_with_seeds],
        "pT": [p["true_params"][3] for p in particles_with_seeds],
        "n_hits": [p["n_hits"] for p in particles_with_seeds],
    }
    without_seeds_data = {
        "z0": [p["true_params"][0] for p in particles_without_seeds],
        "eta": [p["true_params"][1] for p in particles_without_seeds],
        "phi": [p["true_params"][2] for p in particles_without_seeds],
        "pT": [p["true_params"][3] for p in particles_without_seeds],
        "n_hits": [p["n_hits"] for p in particles_without_seeds],
    }

    for i, (param, unit, label) in enumerate(zip(param_names, param_units, param_labels)):
        if i < 5:
            row, col = i // 3, i % 3
            ax = axes[row, col]
            with_data = np.array(with_seeds_data[param])
            without_data = np.array(without_seeds_data[param])
            if len(with_data) > 0 and len(without_data) > 0:
                bins = (
                    min(50, max(len(with_data) // 10, len(without_data) // 10))
                    if max(len(with_data), len(without_data)) > 10
                    else 20
                )
                all_data = np.concatenate([with_data, without_data])
                data_range = [np.percentile(all_data, 5), np.percentile(all_data, 95)]
                ax.hist(
                    with_data,
                    bins=bins,
                    range=data_range,
                    alpha=0.7,
                    label=f"With seeds (n={len(with_data)})",
                    color="green",
                    density=False,
                )
                ax.hist(
                    without_data,
                    bins=bins,
                    range=data_range,
                    alpha=0.7,
                    label=f"Without seeds (n={len(without_data)})",
                    color="red",
                    density=False,
                )
                stats_text = f"With seeds:\n  Mean: {np.mean(with_data):.3f}\n  Std: {np.std(with_data):.3f}\n"
                stats_text += f"Without seeds:\n  Mean: {np.mean(without_data):.3f}\n  Std: {np.std(without_data):.3f}"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                ax.set_xlabel(f"{label} {unit}")
                ax.set_ylabel("Seeds")
                ax.set_title(f"{label} Distribution Comparison")
                ax.legend()
                ax.grid(True, alpha=0.3)
            elif len(with_data) > 0:
                ax.hist(
                    with_data,
                    bins=20,
                    alpha=0.7,
                    label=f"With seeds (n={len(with_data)})",
                    color="green",
                    density=False,
                )
                ax.text(
                    0.02,
                    0.98,
                    f"Only particles with seeds\nMean: {np.mean(with_data):.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                ax.set_xlabel(f"{label} {unit}")
                ax.set_ylabel("Seeds")
                ax.set_title(f"{label} Distribution (Seeds Only)")
                ax.legend()
                ax.grid(True, alpha=0.3)
            elif len(without_data) > 0:
                ax.hist(
                    without_data,
                    bins=20,
                    alpha=0.7,
                    label=f"Without seeds (n={len(without_data)})",
                    color="red",
                    density=False,
                )
                ax.text(
                    0.02,
                    0.98,
                    f"Only particles without seeds\nMean: {np.mean(without_data):.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                ax.set_xlabel(f"{label} {unit}")
                ax.set_ylabel("Seeds")
                ax.set_title(f"{label} Distribution (No Seeds)")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_title(f"{label} Distribution Comparison")

    ax = axes[1, 2]
    total_particles = len(particles_with_seeds) + len(particles_without_seeds)
    efficiency = len(particles_with_seeds) / total_particles if total_particles > 0 else 0.0
    categories = ["With Seeds", "Without Seeds"]
    counts = [len(particles_with_seeds), len(particles_without_seeds)]
    colors = ["green", "red"]
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{count}\n({count / total_particles:.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Number of Particles")
    ax.set_title(f"Particle Reconstruction Summary\nEfficiency: {efficiency:.1%}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_filename = "particle_reconstruction_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print("Particle reconstruction comparison plots saved as:", plot_filename)
    plt.close()


def create_efficiency_vs_truth_param_plots(eligible_particles: Sequence[Mapping[str, Any]]) -> None:
    """
    Plot seeding efficiency (regular and pure) versus truth parameters.

    Inputs
    - eligible_particles: Sequence[Mapping[str, Any]]
        Required keys per particle:
        - `"true_params"`: array-like length 4 `[z0, eta, phi, pT]`.
        - `"had_seed"`: bool flag for at least one seed.
        - `"had_pure_seed"`: bool flag for a pure seed.
        - Optional `"deltaR_min"`: float for ΔR to closest neighbor (used if present).

    Plots
    - Errorbar plots of efficiency vs each parameter: η, φ [rad], z0 [mm], pT [GeV],
        and ΔR when available. Both regular and pure efficiencies are shown.

    Output
    - Saves `seeding_efficiency_vs_truth_params.png`.
    """

    if not eligible_particles:
        print("No eligible particles for efficiency-vs-parameter plots")
        return

    truth_params = np.asarray([p.get("true_params") for p in eligible_particles], dtype=float)
    has_seed_flags = np.array([bool(p.get("had_seed", False)) for p in eligible_particles], dtype=bool)
    has_pure_seed_flags = np.array([bool(p.get("had_pure_seed", False)) for p in eligible_particles], dtype=bool)

    z0 = truth_params[:, 0]
    eta = truth_params[:, 1]
    phi = truth_params[:, 2]
    pt = truth_params[:, 3]
    phi = ((phi + np.pi) % (2 * np.pi)) - np.pi

    def compute_efficiency(
        xvals: np.ndarray,
        has_seed: np.ndarray,
        has_pure: np.ndarray,
        bins: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts_all, edges = np.histogram(xvals, bins=bins)
        counts_seed, _ = np.histogram(xvals[has_seed], bins=edges)
        counts_pure, _ = np.histogram(xvals[has_pure], bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(counts_all > 0, counts_seed / counts_all, 0.0)
            eff_pure = np.where(counts_all > 0, counts_pure / counts_all, 0.0)
            err_eff = np.where(counts_all > 0, np.sqrt(eff * (1.0 - eff) / counts_all), 0.0)
            err_eff_pure = np.where(counts_all > 0, np.sqrt(eff_pure * (1.0 - eff_pure) / counts_all), 0.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * (edges[1:] - edges[:-1])
        return centers, eff, err_eff, eff_pure, err_eff_pure, counts_all, half_widths

    def pct_bounds(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return (0.0, 1.0)
        arr_f = np.asarray(arr, dtype=float)
        lo = float(np.percentile(arr_f, 0.0))
        hi = float(np.percentile(arr_f, 100.0))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        return (lo, hi)

    bins_eta = np.linspace(*pct_bounds(eta), num=20)
    bins_phi = np.linspace(-np.pi, np.pi, num=21)
    bins_z0 = np.linspace(*pct_bounds(z0), num=20)
    bins_pt = np.linspace(*pct_bounds(pt), num=20)

    deltaR = np.array([p.get("deltaR_min", np.inf) for p in eligible_particles], dtype=float)
    finite_dr = np.isfinite(deltaR)
    if np.any(finite_dr):
        dr_hi = np.percentile(deltaR[finite_dr], 99)
        if not np.isfinite(dr_hi) or dr_hi <= 0:
            dr_hi = float(np.max(deltaR[finite_dr])) if np.any(finite_dr) else 1.0
        bins_dr = np.linspace(0.0, float(dr_hi), num=20)
        if np.unique(bins_dr).size < 2:
            bins_dr = np.linspace(0.0, float(dr_hi) + 1e-6, num=20)
    else:
        bins_dr = None

    c_eta, e_eta, e_eta_err, ep_eta, ep_eta_err, n_eta, w_eta = compute_efficiency(
        eta, has_seed_flags, has_pure_seed_flags, bins_eta
    )
    c_phi, e_phi, e_phi_err, ep_phi, ep_phi_err, n_phi, w_phi = compute_efficiency(
        phi, has_seed_flags, has_pure_seed_flags, bins_phi
    )
    c_z0, e_z0, e_z0_err, ep_z0, ep_z0_err, n_z0, w_z0 = compute_efficiency(
        z0, has_seed_flags, has_pure_seed_flags, bins_z0
    )
    c_pt, e_pt, e_pt_err, ep_pt, ep_pt_err, n_pt, w_pt = compute_efficiency(
        pt, has_seed_flags, has_pure_seed_flags, bins_pt
    )
    if bins_dr is not None:
        plot_dr = True
        c_dr, e_dr, e_dr_err, ep_dr, ep_dr_err, n_dr, w_dr = compute_efficiency(
            deltaR, has_seed_flags, has_pure_seed_flags, bins_dr
        )
    else:
        plot_dr = False

    try:
        ncols = 3
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
        axes = axes.flatten()

        def plot_ax(
            ax,
            centers,
            widths,
            counts_all,
            eff,
            eff_err,
            eff_pure,
            eff_pure_err,
            xlabel,
            title,
        ):
            ax.errorbar(
                centers,
                eff,
                xerr=widths,
                yerr=eff_err,
                fmt="o",
                linestyle="none",
                color="tab:blue",
                label="Regular",
                capsize=0,
                linewidth=1,
            )
            ax.errorbar(
                centers,
                eff_pure,
                xerr=widths,
                yerr=eff_pure_err,
                fmt="s",
                linestyle="none",
                color="tab:orange",
                label="Pure",
                capsize=0,
                linewidth=1,
            )
            valid = counts_all > 0
            if np.any(valid):
                min_eff = np.min(np.concatenate([eff[valid], eff_pure[valid]]))
            else:
                min_eff = 1.0
            if min_eff < 0.3:
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylim(0.3, 1.1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Efficiency")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plot_ax(
            axes[0],
            c_eta,
            w_eta,
            n_eta,
            e_eta,
            e_eta_err,
            ep_eta,
            ep_eta_err,
            "Truth η",
            "Efficiency vs η",
        )
        plot_ax(
            axes[1],
            c_phi,
            w_phi,
            n_phi,
            e_phi,
            e_phi_err,
            ep_phi,
            ep_phi_err,
            "Truth φ [rad]",
            "Efficiency vs φ",
        )
        plot_ax(
            axes[2],
            c_z0,
            w_z0,
            n_z0,
            e_z0,
            e_z0_err,
            ep_z0,
            ep_z0_err,
            "Truth z0 [mm]",
            "Efficiency vs z0",
        )
        plot_ax(
            axes[3],
            c_pt,
            w_pt,
            n_pt,
            e_pt,
            e_pt_err,
            ep_pt,
            ep_pt_err,
            "Truth pT [GeV]",
            "Efficiency vs pT",
        )
        if plot_dr:
            plot_ax(
                axes[4],
                c_dr,
                w_dr,
                n_dr,
                e_dr,
                e_dr_err,
                ep_dr,
                ep_dr_err,
                "ΔR to closest particle",
                "Efficiency vs ΔR",
            )

        plt.tight_layout()
        out_name = "seeding_efficiency_vs_truth_params.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print("Efficiency vs truth-parameter plots saved as:", out_name)
        plt.close()
    except Exception as e:
        print(f"Error while plotting efficiency vs parameters: {e}")


def create_seeds_per_particle_vs_truth_param_plots(
    eligible_particles: Sequence[Mapping[str, Any]],
) -> None:
    """
    Plot the mean number of seeds per particle versus truth parameters.

    Inputs
    - eligible_particles: Sequence[Mapping[str, Any]]
        Required keys per particle:
        - `"true_params"`: array-like length 4 `[z0, eta, phi, pT]`.
        - `"n_seeds"`: int number of seeds created for the particle.

    Plots
    - Errorbar plots of mean seeds-per-particle vs η, φ [rad], z0 [mm], pT [GeV].

    Output
    - Saves `seeds_per_particle_vs_truth_params.png`.
    """

    if not eligible_particles:
        print("No eligible particles for seeds-per-particle-vs-parameter plots")
        return

    truth_params = np.asarray([p.get("true_params") for p in eligible_particles], dtype=float)
    n_seeds = np.array([int(p.get("n_seeds", 0)) for p in eligible_particles])
    z0 = truth_params[:, 0]
    eta = truth_params[:, 1]
    phi = truth_params[:, 2]
    pt = truth_params[:, 3]
    phi = ((phi + np.pi) % (2 * np.pi)) - np.pi

    def pct_bounds(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return (0.0, 1.0)
        arr_f = np.asarray(arr, dtype=float)
        lo = float(np.percentile(arr_f, 1))
        hi = float(np.percentile(arr_f, 99))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        return (lo, hi)

    bins_eta = np.linspace(*pct_bounds(eta), num=20)
    bins_phi = np.linspace(-np.pi, np.pi, num=21)
    bins_z0 = np.linspace(*pct_bounds(z0), num=20)
    bins_pt = np.linspace(*pct_bounds(pt), num=20)

    def compute_mean_counts(xvals, counts, bins):
        sum_counts, edges = np.histogram(xvals, bins=bins, weights=counts)
        sum_squares, _ = np.histogram(xvals, bins=edges, weights=counts**2)
        denom, _ = np.histogram(xvals, bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_counts = np.where(denom > 0, sum_counts / denom, 0.0)
            var_counts = np.where(denom > 0, (sum_squares / denom) - mean_counts**2, 0.0)
            var_counts = np.maximum(var_counts, 0.0)
            sem_counts = np.where(denom > 0, np.sqrt(var_counts / denom), 0.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * (edges[1:] - edges[:-1])
        return centers, half_widths, mean_counts, sem_counts

    c_eta, w_eta, msp_eta, msp_eta_sem = compute_mean_counts(eta, n_seeds, bins_eta)
    c_phi, w_phi, msp_phi, msp_phi_sem = compute_mean_counts(phi, n_seeds, bins_phi)
    c_z0, w_z0, msp_z0, msp_z0_sem = compute_mean_counts(z0, n_seeds, bins_z0)
    c_pt, w_pt, msp_pt, msp_pt_sem = compute_mean_counts(pt, n_seeds, bins_pt)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        def plot_ax(ax, centers, widths, mean_counts, mean_sem, xlabel, title):
            ax.errorbar(
                centers,
                mean_counts,
                xerr=widths,
                yerr=mean_sem,
                fmt="o",
                linestyle="none",
                color="tab:blue",
                label="All seeds",
                capsize=0,
                linewidth=1,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean seeds per particle")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if mean_counts.size > 0:
                valid = np.isfinite(mean_counts)
                max_val = np.max(mean_counts[valid]) if np.any(valid) else 0.0
            else:
                max_val = 0.0
            y_max = max(1.0, float(np.ceil(max_val)))
            ax.set_ylim(0.0, y_max)
            ax.legend()

        plot_ax(
            axes[0],
            c_eta,
            w_eta,
            msp_eta,
            msp_eta_sem,
            "Truth η",
            "Seeds/particle vs η",
        )
        plot_ax(
            axes[1],
            c_phi,
            w_phi,
            msp_phi,
            msp_phi_sem,
            "Truth φ [rad]",
            "Seeds/particle vs φ",
        )
        plot_ax(
            axes[2],
            c_z0,
            w_z0,
            msp_z0,
            msp_z0_sem,
            "Truth z0 [mm]",
            "Seeds/particle vs z0",
        )
        plot_ax(
            axes[3],
            c_pt,
            w_pt,
            msp_pt,
            msp_pt_sem,
            "Truth pT [GeV]",
            "Seeds/particle vs pT",
        )

        plt.tight_layout()
        out_name = "seeds_per_particle_vs_truth_params.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        print("Seeds-per-particle vs truth-parameter plots saved as:", out_name)
        plt.close()
    except Exception as e:
        print(f"Error while plotting seeds-per-particle vs parameters: {e}")
