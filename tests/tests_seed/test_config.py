import sys
import pytest

from GUNTAM.Seed.Config import SeedConfig


def test_defaults_initialization():
    cfg = SeedConfig()
    
    # Preprocessing config exists and has correct defaults
    assert cfg.preprocessing_config is not None
    assert cfg.preprocessing_config.max_hit_input == 1200
    assert cfg.preprocessing_config.vertex_cuts == [10, 200]
    assert cfg.preprocessing_config.bin_width == 0.05
    assert cfg.preprocessing_config.eta_range == [-3.0, 3.0]
    assert cfg.preprocessing_config.binning_strategy == "neighbor"
    assert cfg.preprocessing_config.binning_margin == 0.01
    assert cfg.preprocessing_config.max_events == -1
    assert cfg.preprocessing_config.orphan_hit_fraction == 0.0
    assert cfg.preprocessing_config.input_path == "odd_output"
    assert cfg.preprocessing_config.input_format == "csv"
    assert cfg.preprocessing_config.hit_features == ["x", "y", "z"]
    assert cfg.preprocessing_config.particle_features == ["eta", "phi", "pT"]

    # Training defaults
    assert cfg.epoch_nb == 10
    assert cfg.num_warmup_steps == 5
    assert cfg.val_fraction == 0.2
    assert cfg.test_fraction == 0.02
    assert cfg.learning_rate == pytest.approx(5e-5)
    assert cfg.weight_decay == pytest.approx(0.01)
    assert cfg.batch_size == 1

    # Paths (overlapping with preprocessing_config)
    assert cfg.input_tensor_path == "odd_output"
    assert cfg.dataset_name == "seeding_data"
    assert cfg.model_path == "transformer.pt"
    assert cfg.recompute_tensor is False

    # Model architecture
    assert cfg.nb_layers_t == 6
    assert cfg.dim_embedding == 128
    assert cfg.nb_heads == 4
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.fourier_num_frequencies is None

    # Loss configuration
    assert cfg.loss_components == ["attention_next"]
    assert cfg.loss_weights == []


def test_to_from_dict_roundtrip():
    cfg = SeedConfig()
    # customize a few fields and provide consistent loss/weights
    cfg.epoch_nb = 3
    cfg.preprocessing_config.input_path = "data/in"
    cfg.preprocessing_config.max_hit_input = 1500
    cfg.loss_components = ["cosine", "L1", "attention"]
    cfg.loss_weights = [0.5, 2.0, 1.5]
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))

    payload = cfg.to_dict()
    # device should be serialized as string
    assert isinstance(payload["device_acc"], str)
    # preprocessing_config should be serialized as dict
    assert isinstance(payload["preprocessing_config"], dict)

    new_cfg = SeedConfig()
    new_cfg.from_dict(payload)

    # spot check important values and that loss_config is restored
    assert new_cfg.epoch_nb == 3
    assert new_cfg.preprocessing_config.input_path == "data/in"
    assert new_cfg.preprocessing_config.max_hit_input == 1500
    assert new_cfg.loss_components == ["cosine", "L1", "attention"]
    assert new_cfg.loss_weights == [0.5, 2.0, 1.5]
    assert new_cfg.get_loss_weight("L1") == 2.0
    assert new_cfg.get_loss_weight("missing") == 0.0


def test_save_and_load_config(tmp_path):
    cfg = SeedConfig()
    cfg.epoch_nb = 42
    cfg.loss_components = ["cosine"]
    cfg.loss_weights = [3.0]
    cfg.loss_config = {"cosine": 3.0}

    file_path = tmp_path / "conf.json"
    cfg.save_config(str(file_path))
    assert file_path.exists()

    loaded = SeedConfig()
    loaded.load_config(str(file_path))

    assert loaded.epoch_nb == 42
    assert loaded.loss_components == ["cosine"]
    assert loaded.get_loss_weight("cosine") == 3.0


def test_parse_args_conflicting_losses_raises(monkeypatch):
    # Provide both MSE and L1 -> should raise
    argv = [
        "prog",
        "--loss_components",
        "MSE",
        "L1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        SeedConfig().parse_args()


def test_parse_args_infers_default_weights(monkeypatch):
    # Provide components but no weights -> defaults to 1.0 each
    argv = [
        "prog",
        "--loss_components",
        "cosine",
        "attention",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.loss_components == ["cosine", "attention"]
    assert cfg.loss_weights == [1.0, 1.0]
    assert cfg.get_loss_weight("cosine") == 1.0
    assert cfg.get_loss_weight("attention") == 1.0


def test_parse_args_mismatched_lengths_raises(monkeypatch):
    argv = [
        "prog",
        "--loss_components",
        "cosine",
        "MSE",
        "--loss_weights",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        SeedConfig().parse_args()


def test_orphan_hit_fraction_validation(monkeypatch):
    # Valid value should work
    argv = ["prog", "--orphan_hit_fraction", "0.5"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.preprocessing_config.orphan_hit_fraction == 0.5


def test_has_loss_component_and_get_weight_helpers():
    cfg = SeedConfig()
    cfg.loss_components = ["cosine", "attention"]
    cfg.loss_weights = [0.7, 1.3]
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))
    assert cfg.has_loss_component("cosine")
    assert not cfg.has_loss_component("MSE")
    assert cfg.get_loss_weight("attention") == 1.3
    assert cfg.get_loss_weight("MSE") == 0.0
