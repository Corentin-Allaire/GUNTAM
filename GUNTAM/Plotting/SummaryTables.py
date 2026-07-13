"""CSV/Markdown summary tables: one row per quantity, one column per dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from GUNTAM.Plotting import PlotStyle, RootIO

# A TEfficiency group member must retain at least this fraction of the group's largest pooled
# sample size to count as a full-population re-binning (vs. a conditional slice, e.g. a pT-range
# cut) when building the "Overall" aggregate row.
FULL_POPULATION_FRACTION = 0.9


def format_efficiency(eff: float, err_lo: float, err_hi: float) -> str:
    return f"{eff * 100:.2f}% (+{err_hi * 100:.2f}/-{err_lo * 100:.2f})"


def format_profile_combined(mean: float, err: float) -> str:
    return f"{mean:.3f} ± {err:.3f}"


def format_aggregated_efficiency(mean: float, err: float) -> str:
    return f"{mean * 100:.2f}% ± {err * 100:.2f}%"


def metric_group(key: str) -> str:
    """The '<metric>' part of a '<metric>_vs_<axis>' ROOT key, e.g. 'trackeff' for 'trackeff_vs_eta'."""
    return key.split("_vs_", 1)[0] if "_vs_" in key else key


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


def _full_population_keys(sample_file: str, teff_keys: list[str]) -> set[str]:
    """Keep only TEfficiency members whose pooled sample size is close to the group's largest,
    using `sample_file` as the reference so the same set of quantities is aggregated for every
    dataset column.
    """
    if not teff_keys:
        return set()
    totals = {key: RootIO.pooled_efficiency(sample_file, key).n_tot_total for key in teff_keys}
    max_total = max(totals.values())
    return {key for key, total in totals.items() if total >= FULL_POPULATION_FRACTION * max_total}


def compute_group_aggregates(files: list[str], labels: list[str], quantities: dict[str, str]) -> dict[str, dict[str, str]]:
    """One row per metric group (e.g. 'trackeff'): an inverse-variance-weighted combination across
    every full-population quantity sharing that metric (trackeff_vs_eta, trackeff_vs_pT, ...).

    Conditional slices (e.g. a pT-range cut) are excluded via `_full_population_keys` since they
    measure a genuinely different sub-population, not another estimate of the same overall number.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for key, classname in quantities.items():
        groups.setdefault(metric_group(key), []).append((key, classname))

    aggregate_table: dict[str, dict[str, str]] = {}
    for metric, members in groups.items():
        classname = members[0][1]
        teff_keys = [key for key, cls in members if cls == "TEfficiency"]
        full_teff_keys = _full_population_keys(files[0], teff_keys)
        included = [(key, cls) for key, cls in members if cls != "TEfficiency" or key in full_teff_keys]

        metric_label = PlotStyle.METRIC_LABELS.get(metric, metric)
        row: dict[str, str] = {}
        for path, label in zip(files, labels):
            values = []
            errors = []
            for key, cls in included:
                if cls == "TEfficiency":
                    pooled = RootIO.pooled_efficiency(path, key)
                    values.append(pooled.eff)
                    errors.append((pooled.err_lo + pooled.err_hi) / 2)
                else:
                    combined = RootIO.combine_profile_inverse_variance(path, key)
                    values.append(combined.mean)
                    errors.append(combined.err)
            agg = RootIO.inverse_variance_combine(np.array(values), np.array(errors))
            if classname == "TEfficiency":
                row[label] = format_aggregated_efficiency(agg.mean, agg.err)
            else:
                row[label] = format_profile_combined(agg.mean, agg.err)
        aggregate_table[metric_label] = row
    return aggregate_table


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


def write_markdown(
    table: dict[str, dict[str, str]],
    labels: list[str],
    path: Path,
    *,
    aggregate_table: Optional[dict[str, dict[str, str]]] = None,
) -> None:
    lines: list[str] = []
    if aggregate_table:
        lines += [
            "## Overall",
            "",
            "Inverse-variance-weighted combination across every full-population quantity sharing "
            "each metric (e.g. all `trackeff_vs_*` re-binnings); conditional slices such as a "
            "pT-range cut are excluded here but appear in the full breakdown below.",
            "",
        ]
        lines += _markdown_table_lines(aggregate_table, labels, "Metric")
        lines += ["", "## Full breakdown", ""]
    lines += _markdown_table_lines(table, labels, "Quantity")
    path.write_text("\n".join(lines) + "\n")


def write_summary(files: list[str], labels: list[str], quantities: list[str], output_dir: str) -> tuple[Path, Path]:
    sample_available = RootIO.list_plottable_keys(files[0])
    quantities_with_class = {q: sample_available[q] for q in quantities if q in sample_available}

    table = compute_summary_table(files, labels, quantities_with_class)
    aggregate_table = compute_group_aggregates(files, labels, quantities_with_class)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    write_csv(table, labels, csv_path)
    write_markdown(table, labels, md_path, aggregate_table=aggregate_table)
    print(f"Saved -> {csv_path}")
    print(f"Saved -> {md_path}")
    return csv_path, md_path
