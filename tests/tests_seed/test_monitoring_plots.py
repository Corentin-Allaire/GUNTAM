import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from GUNTAM.Seed.MonitoringPlot import PlotUtility


class TestPlotUtility:
    """Tests for the PlotUtility class."""
    matplotlib.use("Agg")

    def test_create_histogram_basic(self):
        """Test basic histogram creation."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Test Histogram",
            xlabel="Value",
            ylabel="Frequency"
        )

        # Check that plot was created
        assert ax.get_title() == "Test Histogram"
        assert ax.get_xlabel() == "Value"
        assert ax.get_ylabel() == "Frequency"
        assert len(ax.patches) > 0  # Histogram bars exist
        plt.close(fig)

    def test_create_histogram_empty_data(self):
        """Test histogram with empty data shows message."""
        fig, ax = plt.subplots()
        data = np.array([])

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Empty Histogram",
            xlabel="Value",
            ylabel="Frequency"
        )

        # Check that "No data available" message is displayed
        assert ax.get_title() == "Empty Histogram"
        texts = [t.get_text() for t in ax.texts]
        assert "No data available" in texts
        plt.close(fig)

    def test_create_histogram_with_stats_text(self):
        """Test histogram with statistics text."""
        fig, ax = plt.subplots()
        data = np.random.randn(100)
        stats_text = "Mean: 0.5\nStd: 1.0"

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Histogram with Stats",
            xlabel="Value",
            ylabel="Frequency",
            stats_text=stats_text
        )

        # Check that stats text is present
        texts = [t.get_text() for t in ax.texts]
        assert any(stats_text in t for t in texts)
        plt.close(fig)

    def test_create_histogram_with_reference_lines(self):
        """Test histogram with reference lines."""
        fig, ax = plt.subplots()
        data = np.random.randn(100)
        reference_lines = [
            {"value": 0.0, "color": "red", "style": "--", "label": "Mean"},
            {"value": 1.0, "color": "green", "style": "-.", "label": "Threshold"}
        ]

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Histogram with References",
            xlabel="Value",
            ylabel="Frequency",
            reference_lines=reference_lines
        )

        # Check that reference lines were added
        assert len(ax.lines) >= 2  # At least the reference lines
        # Check legend was created
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_create_histogram_with_custom_bins(self):
        """Test histogram with custom number of bins."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Custom Bins",
            xlabel="Value",
            ylabel="Frequency",
            bins=30,
            color="red"
        )

        # Check that histogram exists
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_histogram_density_mode(self):
        """Test histogram with density=True."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Density Histogram",
            xlabel="Value",
            ylabel="Density",
            density=True
        )

        # Check that plot was created
        assert ax.get_ylabel() == "Density"
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_histogram_small_dataset(self):
        """Test histogram with very small dataset."""
        fig, ax = plt.subplots()
        data = np.array([1.0, 2.0, 3.0])

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Small Dataset",
            xlabel="Value",
            ylabel="Frequency"
        )

        # Should still create a plot
        assert ax.get_title() == "Small Dataset"
        plt.close(fig)

    def test_create_scatter_2d_basic(self):
        """Test basic 2D scatter plot creation."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        PlotUtility.create_scatter_2d(
            ax=ax,
            x=x,
            y=y,
            title="Scatter Plot",
            xlabel="X",
            ylabel="Y"
        )

        # Check that plot was created
        assert ax.get_title() == "Scatter Plot"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        # Check that scatter points exist
        collections = ax.collections
        assert len(collections) > 0
        plt.close(fig)

    def test_create_scatter_2d_with_color(self):
        """Test scatter plot with color mapping."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        c = np.random.rand(100)

        PlotUtility.create_scatter_2d(
            ax=ax,
            x=x,
            y=y,
            c=c,
            title="Colored Scatter",
            xlabel="X",
            ylabel="Y",
            colorbar_label="Color Value"
        )

        # Check that scatter and colorbar exist
        assert len(ax.collections) > 0
        # Colorbar should be present
        assert len(fig.axes) == 2  # Main axis + colorbar axis
        plt.close(fig)

    def test_create_scatter_2d_custom_params(self):
        """Test scatter plot with custom parameters."""
        fig, ax = plt.subplots()
        x = np.random.randn(50)
        y = np.random.randn(50)

        PlotUtility.create_scatter_2d(
            ax=ax,
            x=x,
            y=y,
            title="Custom Scatter",
            xlabel="X",
            ylabel="Y",
            s=50,
            alpha=0.5,
            cmap="plasma"
        )

        assert ax.get_title() == "Custom Scatter"
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_create_2d_histogram_basic(self):
        """Test basic 2D histogram creation."""
        fig, ax = plt.subplots()
        x = np.random.randn(200)
        y = np.random.randn(200)

        PlotUtility.create_2d_histogram(
            ax=ax,
            x=x,
            y=y,
            title="2D Histogram",
            xlabel="X",
            ylabel="Y"
        )

        # Check that plot was created
        assert ax.get_title() == "2D Histogram"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        # Check that image exists
        assert len(ax.images) > 0
        plt.close(fig)

    def test_create_2d_histogram_with_scatter_overlay(self):
        """Test 2D histogram with scatter overlay."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        PlotUtility.create_2d_histogram(
            ax=ax,
            x=x,
            y=y,
            title="2D Hist + Scatter",
            xlabel="X",
            ylabel="Y",
            overlay_scatter=True
        )

        # Check that both image and scatter exist
        assert len(ax.images) > 0  # Histogram image
        assert len(ax.collections) > 0  # Scatter points
        plt.close(fig)

    def test_create_2d_histogram_without_scatter(self):
        """Test 2D histogram without scatter overlay."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        PlotUtility.create_2d_histogram(
            ax=ax,
            x=x,
            y=y,
            title="2D Hist Only",
            xlabel="X",
            ylabel="Y",
            overlay_scatter=False
        )

        # Check that only image exists, no scatter
        assert len(ax.images) > 0
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_create_2d_histogram_custom_bins(self):
        """Test 2D histogram with custom bin counts."""
        fig, ax = plt.subplots()
        x = np.random.randn(200)
        y = np.random.randn(200)

        PlotUtility.create_2d_histogram(
            ax=ax,
            x=x,
            y=y,
            title="Custom Bins 2D",
            xlabel="X",
            ylabel="Y",
            bins=(20, 25)
        )

        assert len(ax.images) > 0
        plt.close(fig)

    def test_create_2d_histogram_custom_cmap(self):
        """Test 2D histogram with custom colormap."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        PlotUtility.create_2d_histogram(
            ax=ax,
            x=x,
            y=y,
            title="Custom Colormap",
            xlabel="X",
            ylabel="Y",
            cmap="coolwarm"
        )

        assert len(ax.images) > 0
        # Check colorbar exists
        assert len(fig.axes) == 2  # Main axis + colorbar
        plt.close(fig)

    def test_histogram_xlim_percentiles(self):
        """Test that histogram uses 5th and 95th percentiles for xlim."""
        fig, ax = plt.subplots()
        # Create data with outliers
        data = np.concatenate([
            np.random.randn(90),  # Main distribution
            np.array([-50, 50])   # Outliers
        ])

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="With Outliers",
            xlabel="Value",
            ylabel="Frequency"
        )

        # Check that xlim excludes extreme outliers
        xlim = ax.get_xlim()
        p5, p95 = np.percentile(data, [5, 95])
        assert xlim[0] == pytest.approx(p5, abs=0.1)
        assert xlim[1] == pytest.approx(p95, abs=0.1)
        plt.close(fig)

    def test_plots_have_grid(self):
        """Test that all plots have grid enabled."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        data = np.random.randn(100)
        x = np.random.randn(100)
        y = np.random.randn(100)

        PlotUtility.create_histogram(axes[0], data, "Hist", "X", "Y")
        PlotUtility.create_scatter_2d(axes[1], x, y, "Scatter", "X", "Y")
        PlotUtility.create_2d_histogram(axes[2], x, y, "2D Hist", "X", "Y")

        # Check all have grid
        for ax in axes:
            assert ax.xaxis.get_gridlines()[0].get_visible()

        plt.close(fig)

    def test_multiple_reference_lines(self):
        """Test histogram with multiple reference lines of different styles."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)

        reference_lines = [
            {"value": -1.0, "color": "blue", "style": "-", "label": "Lower"},
            {"value": 0.0, "color": "red", "style": "--", "label": "Middle"},
            {"value": 1.0, "color": "green", "style": ":", "label": "Upper"}
        ]

        PlotUtility.create_histogram(
            ax=ax,
            data=data,
            title="Multiple References",
            xlabel="Value",
            ylabel="Frequency",
            reference_lines=reference_lines
        )

        # Check that all reference lines are present
        assert len(ax.lines) >= 3
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
