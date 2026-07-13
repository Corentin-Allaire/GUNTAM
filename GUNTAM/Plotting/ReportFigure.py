"""Entrance point: report-ready 2x2 seeding-vs-finding efficiency comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from GUNTAM.Plotting import PlotStyle, RootIO
from GUNTAM.Plotting.Config import PlottingConfig, parse_args, stage_filenames_for_compare

STAGE_DISPLAY_NAME = {
    "ckf": "CKF",
    "ambi": "Ambiguity resolution",
}


def resolve_dataset_paths(directories: list[str], filename: str) -> list[Path]:
    """Return one Path per dataset directory, pointing at `filename` inside it."""
    return [Path(directory) / filename for directory in directories]


def build_report_figure(cfg: PlottingConfig):
    """Build the 2x2 report figure: rows = stage (compare-stage, Seeding), columns = cfg.quantities."""
    if len(cfg.quantities) != 2:
        raise ValueError(f"ReportFigure requires exactly 2 --quantities, got {len(cfg.quantities)}: {cfg.quantities}")

    PlotStyle.apply_style()

    seeding_filename, compare_filename = stage_filenames_for_compare(cfg.compare)
    compare_stage = cfg.compare.split("-vs-")[1]
    compare_title = STAGE_DISPLAY_NAME.get(compare_stage, compare_stage)

    seeding_paths = resolve_dataset_paths(cfg.files, seeding_filename)
    compare_paths = resolve_dataset_paths(cfg.files, compare_filename)
    labels = cfg.labels if cfg.labels else [Path(d).name for d in cfg.files]
    styles = PlotStyle.assign_series_styles(len(cfg.files))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 11),
        facecolor="white",
        gridspec_kw={"hspace": 0.20, "wspace": 0.20},
    )

    stage_rows = [(compare_paths, compare_title), (seeding_paths, "Seeding")]
    last_row = len(stage_rows) - 1
    last_col = len(cfg.quantities) - 1

    for row, (stage_paths, stage_title) in enumerate(stage_rows):
        for col, quantity in enumerate(cfg.quantities):
            key_style = PlotStyle.get_key_style(quantity)
            ax = axes[row][col]
            series = []
            for path, label, (color, marker) in zip(stage_paths, labels, styles):
                data = RootIO.read_efficiency(str(path), quantity, fold_abs=key_style.fold_abs)
                series.append(PlotStyle.EfficiencySeries(label, data.x, data.eff, data.err_lo, data.err_hi, color, marker))

            title = key_style.label if row == 0 else None
            PlotStyle.draw_efficiency_panel(
                ax,
                series,
                key_style,
                show_xlabel=(row == last_row),
                show_ylabel=(col == 0),
                title=title,
            )
            if row == 0 and col == last_col:
                PlotStyle.add_legend(ax, loc="lower center")

    # One rotated stage label per row (dataset legend colors/markers are shared figure-wide,
    # so only the stage identity needs calling out per row, mirroring the reference figure's
    # "Phase 1"/"Phase 2" side labels).
    for row, (_, stage_title) in enumerate(stage_rows):
        ax_row = axes[row][0]
        bb = ax_row.get_position()
        y_c = bb.y0 + bb.height / 2
        fig.text(
            0.005,
            y_c,
            stage_title,
            va="center",
            ha="left",
            fontsize=PlotStyle.FS_ROW_LABEL,
            fontweight="bold",
            rotation=90,
            color=PlotStyle.DARK,
        )

    return fig


def save_figure(fig, cfg: PlottingConfig, quantities: list[str]) -> list[Path]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q1, q2 = (q.split(";")[0] for q in quantities)
    saved_paths = []
    for ext in cfg.output_formats:
        out_path = output_dir / f"report_{q1}_{q2}_{cfg.compare}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=cfg.dpi)
        saved_paths.append(out_path)
        print(f"Saved -> {out_path}")
    return saved_paths


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_args(argv, include_compare=True)
    fig = build_report_figure(cfg)
    save_figure(fig, cfg, cfg.quantities)
    plt.close(fig)


if __name__ == "__main__":
    main()
