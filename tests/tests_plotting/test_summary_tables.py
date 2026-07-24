"""Tests for GUNTAM.Plotting.SummaryTables."""

from pathlib import Path

from GUNTAM.Plotting import RootIO, SummaryTables

DATA_DIR = Path(__file__).parent.parent / "data" / "geant4_pu200_ttbar_odd_5000evt"
SEEDING_FILES = [
    str(DATA_DIR / "guntam" / "performance_seeding.root"),
    str(DATA_DIR / "triplet_grid" / "performance_seeding.root"),
]


class TestFormatting:
    def test_format_efficiency(self):
        assert SummaryTables.format_efficiency(0.9642, 0.0034, 0.0031) == "96.42% (+0.31/-0.34)"

    def test_format_profile_combined(self):
        assert SummaryTables.format_profile_combined(1.842, 0.021) == "1.842 ± 0.021"


class TestComputeSummaryTableMonkeypatched:
    def test_shape_with_monkeypatched_root_io(self, monkeypatch):
        monkeypatch.setattr(
            RootIO,
            "pooled_efficiency",
            lambda path, key: RootIO.PooledEfficiency(0.9, 0.01, 0.01, 900.0, 1000.0),
        )
        monkeypatch.setattr(
            RootIO,
            "combine_profile_inverse_variance",
            lambda path, key: RootIO.ProfileCombined(2.0, 0.1),
        )

        quantities = {"trackeff_vs_eta": "TEfficiency", "nDuplicated_vs_pT": "TProfile"}
        table = SummaryTables.compute_summary_table(["a.root", "b.root"], ["A", "B"], quantities)

        assert set(table.keys()) == {"Track efficiency vs eta", "Mean duplicated tracks vs pT"}
        assert table["Track efficiency vs eta"]["A"] == "90.00% (+1.00/-1.00)"
        assert table["Mean duplicated tracks vs pT"]["B"] == "2.000 ± 0.100"


class TestPivotedLayout:
    def test_csv_layout(self, tmp_path):
        table = {"Quantity One": {"A": "1", "B": "2"}, "Quantity Two": {"A": "3", "B": "4"}}
        path = tmp_path / "summary.csv"
        SummaryTables.write_csv(table, ["A", "B"], path)

        lines = path.read_text().splitlines()
        assert lines[0] == "Quantity,A,B"
        assert lines[1] == "Quantity One,1,2"
        assert lines[2] == "Quantity Two,3,4"

    def test_markdown_layout_has_no_overall_section(self, tmp_path):
        table = {"Quantity One": {"A": "1", "B": "2"}}
        path = tmp_path / "summary.md"
        SummaryTables.write_markdown(table, ["A", "B"], path)

        text = path.read_text()
        assert "| Quantity | A | B |" in text
        assert "| Quantity One | 1 | 2 |" in text
        # The pooled "Overall" aggregate was removed as statistically overconfident.
        assert "## Overall" not in text
        assert "## Full breakdown" not in text


class TestWriteSummaryEndToEnd:
    def test_mixed_efficiency_and_profile(self, tmp_path):
        csv_path, md_path = SummaryTables.write_summary(
            SEEDING_FILES,
            ["GUNTAM", "Triplet grid"],
            ["trackeff_vs_eta", "nDuplicated_vs_pT"],
            str(tmp_path),
        )
        assert csv_path.exists()
        assert md_path.exists()

        csv_text = csv_path.read_text()
        assert "GUNTAM" in csv_text
        assert "Triplet grid" in csv_text
        assert "%" in csv_text  # efficiency row
        assert "±" in csv_path.read_text()  # profile row (± survives to file, UTF-8)

        # Both CSV and markdown are a single flat table with no pooled "Overall" section.
        assert "## Overall" not in csv_text
        md_text = md_path.read_text()
        assert "## Overall" not in md_text
        assert "Track efficiency vs eta" in md_text
        assert "Mean duplicated tracks vs pT" in md_text
