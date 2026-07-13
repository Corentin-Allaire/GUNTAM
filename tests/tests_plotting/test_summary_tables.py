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

    def test_format_aggregated_efficiency(self):
        assert SummaryTables.format_aggregated_efficiency(0.9611, 0.0001) == "96.11% ± 0.01%"


class TestMetricGroup:
    def test_splits_on_vs(self):
        assert SummaryTables.metric_group("trackeff_vs_eta") == "trackeff"
        assert SummaryTables.metric_group("duplicationRatio_vs_phi") == "duplicationRatio"

    def test_no_vs_returns_whole_key(self):
        assert SummaryTables.metric_group("eff_tracks") == "eff_tracks"


class TestFullPopulationKeys:
    def test_excludes_small_slice_keeps_full_rebinnings(self, monkeypatch):
        totals = {"trackeff_vs_eta": 4_140_000.0, "trackeff_vs_pT": 4_140_000.0, "trackeff_vs_eta_ptRange_0": 693_000.0}
        monkeypatch.setattr(
            RootIO,
            "pooled_efficiency",
            lambda path, key: RootIO.PooledEfficiency(0.96, 0.001, 0.001, 0.0, totals[key]),
        )
        full_keys = SummaryTables._full_population_keys("dummy.root", list(totals))
        assert full_keys == {"trackeff_vs_eta", "trackeff_vs_pT"}

    def test_empty_input_returns_empty_set(self):
        assert SummaryTables._full_population_keys("dummy.root", []) == set()


class TestComputeGroupAggregates:
    def test_slice_excluded_from_aggregate(self, monkeypatch):
        totals = {"trackeff_vs_eta": 4_140_000.0, "trackeff_vs_eta_ptRange_0": 693_000.0}
        effs = {"trackeff_vs_eta": 0.961, "trackeff_vs_eta_ptRange_0": 0.935}

        def fake_pooled(path, key):
            return RootIO.PooledEfficiency(effs[key], 0.0001, 0.0001, 0.0, totals[key])

        monkeypatch.setattr(RootIO, "pooled_efficiency", fake_pooled)

        quantities = {"trackeff_vs_eta": "TEfficiency", "trackeff_vs_eta_ptRange_0": "TEfficiency"}
        aggregate = SummaryTables.compute_group_aggregates(["a.root"], ["A"], quantities)

        # If the slice leaked into the aggregate, the combined value would be pulled well below 96.1%.
        assert aggregate["Track efficiency"]["A"].startswith("96.1")

    def test_single_member_group_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            RootIO,
            "combine_profile_inverse_variance",
            lambda path, key: RootIO.ProfileCombined(2.0, 0.1),
        )
        quantities = {"nDuplicated_vs_pT": "TProfile"}
        aggregate = SummaryTables.compute_group_aggregates(["a.root"], ["A"], quantities)
        assert aggregate["Mean duplicated tracks"]["A"] == "2.000 ± 0.100"


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

    def test_markdown_layout_without_aggregate(self, tmp_path):
        table = {"Quantity One": {"A": "1", "B": "2"}}
        path = tmp_path / "summary.md"
        SummaryTables.write_markdown(table, ["A", "B"], path)

        text = path.read_text()
        assert "| Quantity | A | B |" in text
        assert "| Quantity One | 1 | 2 |" in text
        assert "## Overall" not in text

    def test_markdown_layout_with_aggregate_puts_overall_first(self, tmp_path):
        table = {"Quantity One": {"A": "1", "B": "2"}}
        aggregate_table = {"Metric One": {"A": "1.5", "B": "2.5"}}
        path = tmp_path / "summary.md"
        SummaryTables.write_markdown(table, ["A", "B"], path, aggregate_table=aggregate_table)

        text = path.read_text()
        assert "| Metric | A | B |" in text
        assert "| Metric One | 1.5 | 2.5 |" in text
        assert "| Quantity One | 1 | 2 |" in text
        assert text.index("## Overall") < text.index("## Full breakdown")
        assert text.index("Metric One") < text.index("Quantity One")


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

        # CSV stays a single flat table (no "Overall" section); markdown gets both sections.
        assert "## Overall" not in csv_text

        md_text = md_path.read_text()
        assert "## Overall" in md_text
        assert "## Full breakdown" in md_text
        assert md_text.index("## Overall") < md_text.index("## Full breakdown")

    def test_slice_and_full_rebinning_together_excludes_slice_from_overall(self, tmp_path):
        _, md_path = SummaryTables.write_summary(
            SEEDING_FILES,
            ["GUNTAM", "Triplet grid"],
            ["trackeff_vs_eta", "trackeff_vs_eta_ptRange_0"],
            str(tmp_path),
        )
        md_text = md_path.read_text()
        overall_section = md_text.split("## Overall", 1)[1].split("## Full breakdown", 1)[0]

        # exactly one combined "Track efficiency" row in the aggregate section...
        assert overall_section.count("| Track efficiency |") == 1
        # ...and its GUNTAM value tracks the full-population trackeff_vs_eta (~96.1%), not the
        # pT-range slice (~93.5%), confirming the slice was excluded from the combination.
        overall_row = next(line for line in overall_section.splitlines() if line.startswith("| Track efficiency |"))
        guntam_value = overall_row.split("|")[2].strip()
        assert guntam_value.startswith("96.")

        # both quantities still show up individually in the detailed breakdown below.
        full_breakdown = md_text.split("## Full breakdown", 1)[1]
        assert "Track efficiency vs eta_ptRange_0" in full_breakdown
        assert "Track efficiency vs eta" in full_breakdown
