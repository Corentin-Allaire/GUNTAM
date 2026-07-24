"""Tests for GUNTAM.Plotting.PlotStyle."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from GUNTAM.Plotting import PlotStyle

matplotlib.use("Agg")


class TestApplyStyle:
    def test_sets_expected_rcparams(self):
        PlotStyle.apply_style()
        assert plt.rcParams["font.family"] == ["DejaVu Sans"]
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["ps.fonttype"] == 42
        assert plt.rcParams["axes.linewidth"] == 0.9


class TestAssignSeriesStyles:
    def test_fixed_pairing_for_n_leq_4(self):
        styles = PlotStyle.assign_series_styles(3)
        assert styles == [
            (PlotStyle.RED, "o"),
            (PlotStyle.GREEN, "s"),
            (PlotStyle.ORANGE, "^"),
        ]

    def test_exactly_four(self):
        styles = PlotStyle.assign_series_styles(4)
        assert [c for c, _ in styles] == list(PlotStyle.PALETTE)

    def test_colormap_overflow_and_marker_cycling_for_n_geq_5(self):
        n = 8
        styles = PlotStyle.assign_series_styles(n)
        assert len(styles) == n
        markers = [m for _, m in styles]
        expected_markers = [PlotStyle.MARKERS[i % len(PlotStyle.MARKERS)] for i in range(n)]
        assert markers == expected_markers
        # Colors should not be the fixed 4-color palette; should be distinct across series
        colors = [c for c, _ in styles]
        assert len(set(colors)) == n


class TestGetKeyStyle:
    def test_known_key_d0_fold_and_xlim(self):
        style = PlotStyle.get_key_style("trackeff_vs_d0;1")
        assert style.fold_abs is True
        assert style.xlim == (0.0, 10.0)

    def test_known_key_z0_no_fold_no_xlim(self):
        style = PlotStyle.get_key_style("trackeff_vs_z0;1")
        assert style.fold_abs is False
        assert style.xlim is None

    def test_unknown_key_fallback(self):
        style = PlotStyle.get_key_style("some_unrecognized_key;1")
        assert style.label == "some_unrecognized_key"
        assert style.xlim is None
        assert style.fold_abs is False

    def test_metric_not_in_table_still_composes_with_known_axis(self):
        """A metric with no METRIC_LABELS entry should still pick up the axis's xlim/xlabel."""
        style = PlotStyle.get_key_style("nHoles_vs_eta;1")
        assert style.ylabel == "Mean holes"
        assert style.xlabel == r"$\eta$"
        assert style.xlim == (-3.0, 3.0)

    def test_unknown_axis_with_known_metric_falls_back_to_raw_axis(self):
        """An axis with no AXIS_STYLES entry (e.g. a ptRange split) shouldn't crash or lose the metric label."""
        style = PlotStyle.get_key_style("trackeff_vs_eta_ptRange_0;1")
        assert style.ylabel == "Track efficiency"
        assert style.xlabel == "eta_ptRange_0"
        assert style.xlim is None

    def test_composed_label_matches_metric_and_axis(self):
        style = PlotStyle.get_key_style("duplicationRatio_vs_phi;1")
        assert style.label == "Duplication ratio vs phi"
        assert style.xlim == (-3.15, 3.15)


class TestDrawEfficiencyPanel:
    def test_ylim_uses_full_error_bar_extent(self):
        """Regression test for the y-axis clipping bug: ylim must reflect eff +/- err, not just eff."""
        fig, ax = plt.subplots()
        key_style = PlotStyle.KeyStyle(label="test", xlabel="x", ylabel="y")

        # A single point with a large asymmetric error bar near the top of the axis.
        x = np.array([0.0])
        eff = np.array([0.95])
        err_lo = np.array([0.02])
        err_hi = np.array([0.20])  # pushes the true extent to 1.15, clipped to 1.0 by the eff-only bug

        series = [PlotStyle.EfficiencySeries("s1", x, eff, err_lo, err_hi, PlotStyle.RED, "o")]
        PlotStyle.draw_efficiency_panel(ax, series, key_style)

        y_lo, y_hi = ax.get_ylim()
        # The old (buggy) computation would derive y_hi from eff alone (~0.95+0.015=0.965).
        # The fix must push y_hi up toward the full error extent (eff + err_hi), clipped at 1.0.
        assert y_hi > 0.98
        plt.close(fig)

    def test_series_count_via_containers(self):
        fig, ax = plt.subplots()
        key_style = PlotStyle.KeyStyle(label="test", xlabel="x", ylabel="y")
        x = np.array([0.0, 1.0])
        eff = np.array([0.9, 0.8])
        err = np.array([0.05, 0.05])

        series = [
            PlotStyle.EfficiencySeries("s1", x, eff, err, err, PlotStyle.RED, "o"),
            PlotStyle.EfficiencySeries("s2", x, eff, err, err, PlotStyle.GREEN, "s"),
        ]
        PlotStyle.draw_efficiency_panel(ax, series, key_style)
        assert len(ax.containers) == 2
        plt.close(fig)

    def test_nan_bins_excluded_from_ylim(self):
        fig, ax = plt.subplots()
        key_style = PlotStyle.KeyStyle(label="test", xlabel="x", ylabel="y")
        x = np.array([0.0, 1.0])
        eff = np.array([0.9, np.nan])
        err = np.array([0.05, np.nan])

        series = [PlotStyle.EfficiencySeries("s1", x, eff, err, err, PlotStyle.RED, "o")]
        # Should not raise despite NaN bin.
        PlotStyle.draw_efficiency_panel(ax, series, key_style)
        plt.close(fig)


class TestDrawProfilePanel:
    def test_ylim_uses_full_error_extent(self):
        fig, ax = plt.subplots()
        key_style = PlotStyle.KeyStyle(label="test", xlabel="x", ylabel="y")
        x = np.array([0.0])
        values = np.array([2.0])
        errors = np.array([1.0])

        series = [PlotStyle.ProfileSeries("s1", x, values, errors, PlotStyle.RED, "o")]
        PlotStyle.draw_profile_panel(ax, series, key_style)

        y_lo, y_hi = ax.get_ylim()
        assert y_hi > 3.0
        assert y_lo < 1.0
        plt.close(fig)


class TestAddLegend:
    def test_legend_present(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="test series")
        PlotStyle.add_legend(ax)
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)
