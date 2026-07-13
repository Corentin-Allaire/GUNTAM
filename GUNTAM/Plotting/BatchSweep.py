"""Entrance point: sweep an arbitrary set of quantities across an arbitrary set of ROOT files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from GUNTAM.Plotting import PlotStyle, RootIO, SummaryTables
from GUNTAM.Plotting.Config import PlottingConfig, parse_args


def resolve_labels(cfg: PlottingConfig) -> list[str]:
    if cfg.labels:
        return cfg.labels
    return [Path(f).stem for f in cfg.files]


def select_sweep_quantities(cfg: PlottingConfig, sample_file: str) -> dict[str, str]:
    """Return {key: classname} to sweep: full introspected set for 'all', else a validated subset."""
    available = RootIO.list_plottable_keys(sample_file)
    if cfg.quantities == ["all"]:
        return available

    missing = [q for q in cfg.quantities if q not in available]
    if missing:
        raise ValueError(f"Quantities not found in {sample_file}: {missing}. Available: {sorted(available)}")
    return {q: available[q] for q in cfg.quantities}


def sweep_quantity(files: list[str], labels: list[str], key: str, classname: str, key_style: PlotStyle.KeyStyle):
    """Build one single-panel figure overlaying `key` across all `files`."""
    styles = PlotStyle.assign_series_styles(len(files))
    fig, ax = plt.subplots(figsize=(12, 6))

    if classname == "TEfficiency":
        eff_series = []
        for path, label, (color, marker) in zip(files, labels, styles):
            eff_data = RootIO.read_efficiency(path, key, fold_abs=key_style.fold_abs)
            eff_series.append(
                PlotStyle.EfficiencySeries(label, eff_data.x, eff_data.eff, eff_data.err_lo, eff_data.err_hi, color, marker)
            )
        PlotStyle.draw_efficiency_panel(ax, eff_series, key_style, title=key_style.label)
    else:
        profile_series = []
        for path, label, (color, marker) in zip(files, labels, styles):
            profile_data = RootIO.read_profile(path, key)
            profile_series.append(
                PlotStyle.ProfileSeries(label, profile_data.x, profile_data.values, profile_data.errors, color, marker)
            )
        PlotStyle.draw_profile_panel(ax, profile_series, key_style, title=key_style.label)

    PlotStyle.add_legend(ax)
    return fig


def run_sweep(cfg: PlottingConfig) -> dict[str, list[Path]]:
    PlotStyle.apply_style()

    labels = resolve_labels(cfg)
    quantities = select_sweep_quantities(cfg, cfg.files[0])

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, list[Path]] = {}
    for key, classname in quantities.items():
        key_style = PlotStyle.get_key_style(key)
        fig = sweep_quantity(cfg.files, labels, key, classname, key_style)

        base_name = key.split(";")[0]
        paths = []
        for ext in cfg.output_formats:
            out_path = output_dir / f"{base_name}.{ext}"
            fig.savefig(out_path, bbox_inches="tight", dpi=cfg.dpi)
            paths.append(out_path)
            print(f"Saved -> {out_path}")
        saved[key] = paths
        plt.close(fig)

    SummaryTables.write_summary(cfg.files, labels, list(quantities.keys()), cfg.output_dir)
    return saved


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_args(argv, include_compare=False)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
