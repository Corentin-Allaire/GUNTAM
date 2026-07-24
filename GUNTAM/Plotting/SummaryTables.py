"""CSV/Markdown summary tables: one row per quantity, one column per dataset."""

from __future__ import annotations

import csv
from pathlib import Path

from GUNTAM.Plotting import PlotStyle, RootIO


def format_efficiency(eff: float, err_lo: float, err_hi: float) -> str:
    return f"{eff * 100:.2f}% (+{err_hi * 100:.2f}/-{err_lo * 100:.2f})"


def format_profile_combined(mean: float, err: float) -> str:
    return f"{mean:.3f} ± {err:.3f}"


def compute_summary_table(files: list[str], labels: list[str], quantities: dict[str, str]) -> dict[str, dict[str, str]]:
    """Return {quantity_label: {dataset_label: formatted_value}}."""
    table: dict[str, dict[str, str]] = {}
    for key, classname in quantities.items():
        key_style = PlotStyle.get_key_style(key)
        row: dict[str, str] = {}
        for path, label in zip(files, labels):
            if classname == "TEfficiency":
                pooled = RootIO.pooled_efficiency(path, key)
                row[label] = format_efficiency(pooled.eff, pooled.err_lo, pooled.err_hi)
            else:
                combined = RootIO.combine_profile_inverse_variance(path, key)
                row[label] = format_profile_combined(combined.mean, combined.err)
        table[key_style.label] = row
    return table


def _markdown_table_lines(table: dict[str, dict[str, str]], labels: list[str], row_header: str) -> list[str]:
    lines = [
        f"| {row_header} | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for row_label, row in table.items():
        lines.append(f"| {row_label} | " + " | ".join(row[label] for label in labels) + " |")
    return lines


def write_csv(table: dict[str, dict[str, str]], labels: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Quantity", *labels])
        for quantity_label, row in table.items():
            writer.writerow([quantity_label, *(row[label] for label in labels)])


def write_markdown(table: dict[str, dict[str, str]], labels: list[str], path: Path) -> None:
    # No pooled "Overall" row is emitted: the per-axis re-binnings (trackeff_vs_eta, _vs_pT, ...)
    # are the SAME tracks re-histogrammed, so inverse-variance-combining them would treat one
    # measurement as many independent ones and understate the uncertainty. Each quantity is
    # reported on its own row instead.
    lines = _markdown_table_lines(table, labels, "Quantity")
    path.write_text("\n".join(lines) + "\n")


def write_summary(files: list[str], labels: list[str], quantities: list[str], output_dir: str) -> tuple[Path, Path]:
    sample_available = RootIO.list_plottable_keys(files[0])
    quantities_with_class = {q: sample_available[q] for q in quantities if q in sample_available}

    table = compute_summary_table(files, labels, quantities_with_class)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    write_csv(table, labels, csv_path)
    write_markdown(table, labels, md_path)
    print(f"Saved -> {csv_path}")
    print(f"Saved -> {md_path}")
    return csv_path, md_path
