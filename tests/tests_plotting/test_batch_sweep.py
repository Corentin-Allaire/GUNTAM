"""Tests for GUNTAM.Plotting.BatchSweep."""

import sys
from pathlib import Path

import matplotlib
import pytest

from GUNTAM.Plotting import BatchSweep, PlotStyle
from GUNTAM.Plotting.Config import PlottingConfig

matplotlib.use("Agg")

DATA_DIR = Path(__file__).parent.parent / "data" / "geant4_pu200_ttbar_odd_5000evt"
SEEDING_FILES = [
    str(DATA_DIR / "guntam" / "performance_seeding.root"),
    str(DATA_DIR / "triplet_grid" / "performance_seeding.root"),
]


class TestResolveLabels:
    def test_uses_configured_labels(self):
        cfg = PlottingConfig(files=SEEDING_FILES, labels=["A", "B"])
        assert BatchSweep.resolve_labels(cfg) == ["A", "B"]

    def test_falls_back_to_file_stems(self):
        cfg = PlottingConfig(files=SEEDING_FILES)
        assert BatchSweep.resolve_labels(cfg) == ["performance_seeding", "performance_seeding"]


class TestSelectSweepQuantities:
    def test_all_returns_full_introspected_set(self):
        cfg = PlottingConfig(files=SEEDING_FILES, quantities=["all"])
        result = BatchSweep.select_sweep_quantities(cfg, SEEDING_FILES[0])
        assert "trackeff_vs_eta" in result
        assert "nDuplicated_vs_pT" in result

    def test_explicit_missing_key_raises(self):
        cfg = PlottingConfig(files=SEEDING_FILES, quantities=["trackeff_vs_eta", "not_a_real_key"])
        with pytest.raises(ValueError, match="not_a_real_key"):
            BatchSweep.select_sweep_quantities(cfg, SEEDING_FILES[0])

    def test_explicit_valid_subset(self):
        cfg = PlottingConfig(files=SEEDING_FILES, quantities=["trackeff_vs_eta"])
        result = BatchSweep.select_sweep_quantities(cfg, SEEDING_FILES[0])
        assert result == {"trackeff_vs_eta": "TEfficiency"}


class TestSweepQuantity:
    def test_teff_dispatch(self):
        key_style = PlotStyle.get_key_style("trackeff_vs_eta")
        fig = BatchSweep.sweep_quantity(SEEDING_FILES, ["A", "B"], "trackeff_vs_eta", "TEfficiency", key_style)
        assert len(fig.axes[0].containers) == 2

    def test_tprofile_dispatch(self):
        key_style = PlotStyle.get_key_style("nDuplicated_vs_pT")
        fig = BatchSweep.sweep_quantity(SEEDING_FILES, ["A", "B"], "nDuplicated_vs_pT", "TProfile", key_style)
        assert len(fig.axes[0].containers) == 2


class TestRunSweepEndToEnd:
    def test_produces_plots_and_summary(self, tmp_path):
        cfg = PlottingConfig(
            files=SEEDING_FILES,
            labels=["GUNTAM", "Triplet grid"],
            quantities=["trackeff_vs_eta", "nDuplicated_vs_pT"],
            output_dir=str(tmp_path),
            output_formats=["png"],
        )
        saved = BatchSweep.run_sweep(cfg)
        assert len(saved) == 2
        assert (tmp_path / "trackeff_vs_eta.png").exists()
        assert (tmp_path / "nDuplicated_vs_pT.png").exists()
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "summary.md").exists()


class TestMainCLISmoke:
    def test_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--files",
                *SEEDING_FILES,
                "--labels",
                "GUNTAM",
                "Triplet grid",
                "--quantities",
                "trackeff_vs_eta",
                "--output_dir",
                str(tmp_path),
            ],
        )
        BatchSweep.main()
        assert (tmp_path / "trackeff_vs_eta.png").exists()
        assert (tmp_path / "summary.csv").exists()
