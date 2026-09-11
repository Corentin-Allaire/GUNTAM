"""Shared configuration (dataclass + JSON + CLI) for the Plotting entrance scripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

COMPARE_CHOICES = ("seeding-vs-ckf", "seeding-vs-ambi")

STAGE_FILENAMES = {
    "seeding": "performance_seeding.root",
    "ckf": "performance_finding_ckf.root",
    "ambi": "performance_finding_ambi.root",
}


def stage_filenames_for_compare(compare: str) -> tuple[str, str]:
    """Return (seeding_filename, compare_stage_filename) for a `--compare` choice."""
    if compare not in COMPARE_CHOICES:
        raise ValueError(f"Unknown compare option '{compare}', expected one of {COMPARE_CHOICES}")
    _, stage = compare.split("-vs-")
    return STAGE_FILENAMES["seeding"], STAGE_FILENAMES[stage]


@dataclass
class PlottingConfig:
    """Configuration shared by the ReportFigure and BatchSweep entrance scripts."""

    files: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    compare: str = "seeding-vs-ckf"
    quantities: list[str] = field(default_factory=lambda: ["all"])
    output_dir: str = "plots"
    output_formats: list[str] = field(default_factory=lambda: ["png"])
    dpi: int = 150

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.labels and len(self.labels) != len(self.files):
            raise ValueError(f"Number of labels ({len(self.labels)}) must match number of files ({len(self.files)})")
        if self.compare not in COMPARE_CHOICES:
            raise ValueError(f"Unknown compare option '{self.compare}', expected one of {COMPARE_CHOICES}")
        if not self.quantities:
            raise ValueError("quantities must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PlottingConfig":
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PlottingConfig":
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        print(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)

    def print_config(self) -> None:
        print("=" * 60)
        print("Plotting Configuration")
        print("=" * 60)
        print("Files: ", self.files)
        print("Labels: ", self.labels)
        print("Compare: ", self.compare)
        print("Quantities: ", self.quantities)
        print("Output dir: ", self.output_dir)
        print("Output formats: ", self.output_formats)
        print("DPI: ", self.dpi)
        print("=" * 60)


def add_args(parser: argparse.ArgumentParser, *, include_compare: bool = False) -> None:
    """Register Plotting CLI flags on `parser`. Non-passed flags default to None (sentinel)."""
    parser.add_argument("--files", nargs="+", default=None, help="ROOT files or dataset directories to plot")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels, one per --files entry")
    if include_compare:
        parser.add_argument(
            "--compare",
            type=str,
            default=None,
            choices=COMPARE_CHOICES,
            help="Which finding stage to compare against seeding",
        )
    parser.add_argument("--quantities", nargs="+", default=None, help="ROOT key names to plot, or 'all'")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write plots/tables to")
    parser.add_argument("--output_formats", nargs="+", default=None, help="Output image formats (e.g. png pdf)")
    parser.add_argument("--dpi", type=int, default=None, help="Output image resolution")
    parser.add_argument("--config", type=str, default=None, help="Load configuration from a JSON file")
    parser.add_argument("--save_config", type=str, default=None, help="Save the resolved configuration to a JSON file")


def from_args(args: argparse.Namespace, *, include_compare: bool = False) -> PlottingConfig:
    """Build a PlottingConfig from parsed args: JSON (if --config given), then CLI overrides."""
    cfg = PlottingConfig.load(args.config) if args.config else PlottingConfig()

    if args.files is not None:
        cfg.files = args.files
    if args.labels is not None:
        cfg.labels = args.labels
    if include_compare and getattr(args, "compare", None) is not None:
        cfg.compare = args.compare
    if args.quantities is not None:
        cfg.quantities = args.quantities
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.output_formats is not None:
        cfg.output_formats = args.output_formats
    if args.dpi is not None:
        cfg.dpi = args.dpi

    cfg._validate()

    if args.save_config:
        cfg.save(args.save_config)

    return cfg


def parse_args(argv: Optional[list[str]] = None, *, include_compare: bool = False) -> PlottingConfig:
    parser = argparse.ArgumentParser(description="Configure ROOT comparison plotting")
    add_args(parser, include_compare=include_compare)
    args = parser.parse_args(argv)
    return from_args(args, include_compare=include_compare)
