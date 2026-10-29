import pathlib
import pytest
from unittest.mock import Mock, patch

from GUNTAM.Seed.Config.SeedConfig import SeedConfig
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Monitoring.PermutationImportances import (
    perm_imp_featuress,
    feature_channel_indices,
    plot_freq_loss,
    plot_all_loss,
    plot_freq_eff,
    plot_all_eff,
    config_model_dataset,
)


class _FakeFourierEncoding:
    """Minimal stand-in for FourierPositionalEncoding (only the attributes used by the analysis)."""

    def __init__(self, num_frequencies, output_dim):
        self.num_frequencies = list(num_frequencies)
        self.output_dim = output_dim


class _FakeCfg:
    """Minimal stand-in for TransformerConfig."""

    def __init__(self, embedding_feature, high_level_features, cosine_processing):
        self.embedding_feature = list(embedding_feature)
        self.high_level_features = list(high_level_features)
        self.cosine_processing = list(cosine_processing)


class _FakeModel:
    """Minimal stand-in for SeedTransformer exposing cfg and fourier_encoding."""

    def __init__(
        self, num_frequencies, output_dim=None, embedding_feature=(0, 1, 2, 3), high_level_features=(4, 5), cosine_processing=(4,)
    ):
        if output_dim is None:
            # [cos(phi), sin(phi), plain...] tail width
            high_level_tail = len(high_level_features) + len(set(high_level_features) & set(cosine_processing))
            output_dim = sum(num_frequencies) * 2 + high_level_tail
        self.cfg = _FakeCfg(embedding_feature, high_level_features, cosine_processing)
        self.fourier_encoding = _FakeFourierEncoding(num_frequencies, output_dim)


@pytest.fixture
def default_model():
    """Model matching the default config: 4 coord dims x 15 freqs, high level (phi, eta) with cos/sin(phi)."""
    return _FakeModel(num_frequencies=[15, 15, 15, 15], output_dim=123)


data_path = pathlib.Path(__file__).parents[1] / "data" / "odd_output_new_5"
model_path = pathlib.Path(__file__).parents[1] / "data" / "transformer.pt"


@pytest.fixture
def mock_dataset():
    """A mock dataset with file_paths."""
    dataset = Mock(spec=DataLoader)
    dataset.file_paths = ["file_a", "file_b"]
    return dataset


@pytest.fixture
def mock_model():
    """A mock model."""
    return Mock(spec=SeedTransformer)


@pytest.fixture
def mock_cfg():
    """A mock config."""
    return Mock(spec=SeedConfig)


class TestPermImpFeatures:
    """Unit tests for perm_imp_featuress."""

    @patch("GUNTAM.Monitoring.PermutationImportances.validate_function")
    def test_method_loss_calls_validate_function(self, mock_validate, mock_dataset, mock_model, mock_cfg):
        """method='loss' should call validate_function with the right args."""
        mock_validate.return_value = 0.5
        result = perm_imp_featuress(features=[1, 2], method="loss", dataset=mock_dataset, model=mock_model, cfg=mock_cfg)

        mock_validate.assert_called_once_with(
            model=mock_model,
            file_indices=[0, 1],
            dataset=mock_dataset,
            cfg=mock_cfg,
            features=[1, 2],
        )
        assert result == 0.5

    @patch("GUNTAM.Monitoring.PermutationImportances.efficiency_reconstructed_seeds")
    def test_method_seed_eff_calls_efficiency(self, mock_efficiency, mock_dataset, mock_model, mock_cfg):
        """method='seed_eff' should call efficiency_reconstructed_seeds with the right args."""
        mock_efficiency.return_value = 0.85
        result = perm_imp_featuress(features=[0], method="seed_eff", dataset=mock_dataset, model=mock_model, cfg=mock_cfg)

        mock_efficiency.assert_called_once_with(
            model=mock_model,
            file_indices=[0, 1],
            dataset=mock_dataset,
            cfg=mock_cfg,
            features=[0],
        )
        assert result == 0.85

    def test_invalid_method_raises_value_error(self, mock_dataset, mock_model, mock_cfg):
        """An unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            perm_imp_featuress(features=[0], method="invalid", dataset=mock_dataset, model=mock_model, cfg=mock_cfg)

    def test_empty_features_passes_empty_list(self, mock_dataset, mock_model, mock_cfg):
        """Empty features list should be passed through."""
        with patch("GUNTAM.Monitoring.PermutationImportances.validate_function") as mock_validate:
            mock_validate.return_value = 0.3
            perm_imp_featuress(features=[], method="loss", dataset=mock_dataset, model=mock_model, cfg=mock_cfg)
            mock_validate.assert_called_once_with(
                model=mock_model,
                file_indices=[0, 1],
                dataset=mock_dataset,
                cfg=mock_cfg,
                features=[],
            )


class TestPlotFunctions:
    """Unit tests for the plot functions."""

    @pytest.fixture(autouse=True)
    def _set_agg_backend(self):
        """Use Agg backend to avoid GUI windows."""
        import matplotlib

        matplotlib.use("Agg")
        yield

    def test_plot_all_loss_creates_file(self, tmp_path, monkeypatch):
        """plot_all_loss should save a PNG file."""
        monkeypatch.chdir(tmp_path)
        plot_all_loss(avg_loss_ref=0.5, avg_loss_total=[0.6, 0.7, 0.8])
        assert (tmp_path / "perm_imp_all_features_loss.png").exists()

    def test_plot_all_eff_creates_file(self, tmp_path, monkeypatch):
        """plot_all_eff should save a PNG file."""
        monkeypatch.chdir(tmp_path)
        plot_all_eff(seed_eff_total=[0.8, 0.7, 0.9])
        assert (tmp_path / "perm_imp_all_features_eff.png").exists()

    @pytest.mark.parametrize("feature", ["x", "y", "z", "r", "cos_phi", "sin_phi", "eta"])
    def test_plot_freq_loss_creates_file(self, feature, tmp_path, monkeypatch, default_model):
        """plot_freq_loss should save a PNG for each feature under the frequency-blocked layout."""
        output_dim = default_model.fourier_encoding.output_dim
        avg_loss_total = list(range(output_dim))
        monkeypatch.chdir(tmp_path)

        plot_freq_loss(avg_loss_ref=0.5, avg_loss_total=avg_loss_total, model=default_model, feature=feature)

        assert (tmp_path / f"perm_imp_freq_{feature}_loss.png").exists()

    @pytest.mark.parametrize("feature", ["x", "y", "z", "r", "cos_phi", "sin_phi", "eta"])
    def test_plot_freq_eff_creates_file(self, feature, tmp_path, monkeypatch, default_model):
        """plot_freq_eff should save a PNG for each feature under the frequency-blocked layout."""
        output_dim = default_model.fourier_encoding.output_dim
        seed_eff_total = [0.8] * output_dim
        monkeypatch.chdir(tmp_path)

        plot_freq_eff(seed_eff_total=seed_eff_total, model=default_model, feature=feature)

        assert (tmp_path / f"perm_imp_freq_{feature}_eff.png").exists()

    def test_plot_freq_loss_invalid_feature_raises(self, tmp_path, monkeypatch, default_model):
        """plot_freq_loss with an unknown feature should raise ValueError."""
        monkeypatch.chdir(tmp_path)
        output_dim = default_model.fourier_encoding.output_dim
        with pytest.raises(ValueError, match="Unknown feature"):
            plot_freq_loss(avg_loss_ref=0.5, avg_loss_total=[0.6] * output_dim, model=default_model, feature="invalid")

    def test_plot_freq_eff_invalid_feature_raises(self, tmp_path, monkeypatch, default_model):
        """plot_freq_eff with an unknown feature should raise ValueError."""
        monkeypatch.chdir(tmp_path)
        output_dim = default_model.fourier_encoding.output_dim
        with pytest.raises(ValueError, match="Unknown feature"):
            plot_freq_eff(seed_eff_total=[0.8] * output_dim, model=default_model, feature="invalid")


class TestFeatureChannelIndices:
    """Unit tests for feature_channel_indices (frequency-blocked layout)."""

    def test_uniform_coordinate_feature(self, default_model):
        """With uniform frequencies, a coordinate feature gets one (sin, cos) pair per level.

        For 4 dims with 15 freqs: level k block starts at k * 8, sin first
        (feature at position d -> sin at block_start + d, cos at block_start + 4 + d).
        """
        expected = [(k * 8 + 0, k * 8 + 4) for k in range(15)]  # x is dim 0
        assert feature_channel_indices(default_model, "x") == expected
        expected_z = [(k * 8 + 2, k * 8 + 6) for k in range(15)]  # z is dim 2
        assert feature_channel_indices(default_model, "z") == expected_z
        expected_r = [(k * 8 + 3, k * 8 + 7) for k in range(15)]  # r is dim 3
        assert feature_channel_indices(default_model, "r") == expected_r

    def test_ragged_frequencies(self):
        """Dimensions with fewer frequencies drop out of higher levels, in dimension order."""
        # dims: x n=4, y n=2, z n=6 -> active dims per level:
        # k=0,1: [x,y,z]; k=2..5: [x,z]
        model = _FakeModel(num_frequencies=[4, 2, 6], output_dim=4 * 2 + 2 * 2 + 6 * 2 + 3)
        # Active dims per level: k=0,1 -> [x,y,z] (block width 6); k=2..5 -> [x,z] (block width 4)
        # Block bases: k=0 -> 0, k=1 -> 6; k=2..5 -> 12 + (k-2)*4
        # Returns (sin, cos) pairs per frequency level, at the dim's active position:
        # x (pos 0; pos 0 after y drops): cos at base + n_active
        expected_x = [(0, 3), (6, 9), (12, 14), (16, 18)]
        assert feature_channel_indices(model, "x") == expected_x
        # y (active only at levels 0 and 1, position 1)
        expected_y = [(1, 4), (7, 10)]
        assert feature_channel_indices(model, "y") == expected_y
        # z (position 2 at levels 0,1; position 1 at levels 2..5)
        expected_z = [(2, 5), (8, 11), (13, 15), (17, 19), (20, 21), (22, 23)]
        assert feature_channel_indices(model, "z") == expected_z

    def test_high_level_features(self, default_model):
        """High-level features live in the tail: [cos(phi), sin(phi), eta]."""
        output_dim = default_model.fourier_encoding.output_dim  # 123
        assert feature_channel_indices(default_model, "cos_phi") == [output_dim - 3]
        assert feature_channel_indices(default_model, "sin_phi") == [output_dim - 2]
        assert feature_channel_indices(default_model, "eta") == [output_dim - 1]

    def test_all_feature_channels_are_disjoint(self, default_model):
        """Every embedding channel belongs to exactly one feature."""
        all_channels: list = []
        for feature in ["x", "y", "z", "r", "cos_phi", "sin_phi", "eta"]:
            for entry in feature_channel_indices(default_model, feature):
                if isinstance(entry, tuple):
                    all_channels.extend(entry)  # (sin, cos) pair
                else:
                    all_channels.append(entry)  # single high-level channel
        assert len(all_channels) == len(set(all_channels))
        assert all(0 <= c < default_model.fourier_encoding.output_dim for c in all_channels)

    def test_unknown_feature_raises(self, default_model):
        """An unknown feature name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown feature"):
            feature_channel_indices(default_model, "invalid")


class TestConfigModelDataset:
    """Integration test for config_model_dataset (minimal — uses real data)."""

    def test_config_model_dataset_returns_dataset_and_model(self):
        """Should return a DataLoader and a SeedTransformer."""
        cfg, dataset, model = config_model_dataset(
            path=str(data_path),
            dataset_name="odd_output_new_5",
            model_name=str(model_path),
        )

        assert isinstance(cfg, SeedConfig)
        assert isinstance(dataset, DataLoader)
        assert isinstance(model, SeedTransformer)
        assert len(dataset.file_paths) > 0
