"""Tests for GUNTAM.Plotting.ReportFigure."""

import sys
from pathlib import Path

import matplotlib
import pytest

from GUNTAM.Plotting import ReportFigure
from GUNTAM.Plotting.Config import PlottingConfig

matplotlib.use("Agg")

DATA_DIR = Path(__file__).parent.parent / "data" / "geant4_pu200_ttbar_odd_5000evt"
GUNTAM_DIR = str(DATA_DIR / "guntam")
TRIPLET_DIR = str(DATA_DIR / "triplet_grid")


class TestResolveDatasetPaths:
    def test_resolves_filenames_under_each_directory(self):
        paths = ReportFigure.resolve_dataset_paths([GUNTAM_DIR, TRIPLET_DIR], "performance_seeding.root")
        assert paths == [Path(GUNTAM_DIR) / "performance_seeding.root", Path(TRIPLET_DIR) / "performance_seeding.root"]


class TestBuildReportFigureQuantitiesValidation:
    def test_requires_exactly_two_quantities(self):
        cfg = PlottingConfig(files=[GUNTAM_DIR], quantities=["trackeff_vs_eta"])
        with pytest.raises(ValueError, match="exactly 2"):
            ReportFigure.build_report_figure(cfg)

    def test_all_is_never_valid(self):
        cfg = PlottingConfig(files=[GUNTAM_DIR], quantities=["all"])
        with pytest.raises(ValueError, match="exactly 2"):
            ReportFigure.build_report_figure(cfg)

    def test_three_quantities_rejected(self):
        cfg = PlottingConfig(files=[GUNTAM_DIR], quantities=["trackeff_vs_eta", "trackeff_vs_phi", "trackeff_vs_pT"])
        with pytest.raises(ValueError, match="exactly 2"):
            ReportFigure.build_report_figure(cfg)


class TestBuildReportFigureStructural:
    def _build(self, compare="seeding-vs-ckf"):
        cfg = PlottingConfig(
            files=[GUNTAM_DIR, TRIPLET_DIR],
            labels=["GUNTAM", "Triplet grid"],
            compare=compare,
            quantities=["trackeff_vs_eta", "trackeff_vs_pT"],
        )
        return ReportFigure.build_report_figure(cfg), cfg

    def test_four_axes(self):
        fig, _ = self._build()
        assert len(fig.axes) == 4

    def test_one_container_per_dataset_per_panel(self):
        fig, cfg = self._build()
        for ax in fig.axes:
            assert len(ax.containers) == len(cfg.files)

    def test_column_titles_show_quantity_labels(self):
        fig, _ = self._build()
        titles = [ax.get_title() for ax in fig.axes]
        assert any("eta" in t for t in titles)
        assert any("pT" in t for t in titles)

    def test_row_labels_show_stage_names_for_seeding_vs_ckf(self):
        fig, _ = self._build(compare="seeding-vs-ckf")
        row_labels = [t.get_text() for t in fig.texts]
        assert "Seeding" in row_labels
        assert "CKF" in row_labels

    def test_row_labels_show_stage_names_for_seeding_vs_ambi(self):
        fig, _ = self._build(compare="seeding-vs-ambi")
        row_labels = [t.get_text() for t in fig.texts]
        assert "Seeding" in row_labels
        assert "Ambiguity resolution" in row_labels

    def test_only_one_legend_for_whole_figure(self):
        fig, _ = self._build()
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
        assert len(legends) == 1

    def test_only_top_row_has_titles(self):
        fig, _ = self._build()
        top_titles = [axes.get_title() for axes in fig.axes[:2]]
        bottom_titles = [axes.get_title() for axes in fig.axes[2:]]
        assert all(t for t in top_titles)
        assert all(t == "" for t in bottom_titles)


class TestSaveFigure:
    def test_files_are_written(self, tmp_path):
        cfg = PlottingConfig(
            files=[GUNTAM_DIR, TRIPLET_DIR],
            labels=["GUNTAM", "Triplet grid"],
            quantities=["trackeff_vs_eta", "trackeff_vs_pT"],
            output_dir=str(tmp_path),
            output_formats=["png"],
        )
        fig = ReportFigure.build_report_figure(cfg)
        paths = ReportFigure.save_figure(fig, cfg, cfg.quantities)
        assert len(paths) == 1
        assert paths[0].exists()


class TestMainCLISmoke:
    def test_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--files",
                GUNTAM_DIR,
                TRIPLET_DIR,
                "--labels",
                "GUNTAM",
                "Triplet grid",
                "--quantities",
                "trackeff_vs_eta",
                "trackeff_vs_pT",
                "--compare",
                "seeding-vs-ckf",
                "--output_dir",
                str(tmp_path),
            ],
        )
        ReportFigure.main()
        outputs = list(tmp_path.glob("report_*.png"))
        assert len(outputs) == 1
