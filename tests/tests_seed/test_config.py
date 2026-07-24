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
    assert cfg.num_training_steps == 100
    assert cfg.min_lr_ratio == pytest.approx(0.01)
    assert cfg.val_fraction == 0.1
    assert cfg.test_fraction == 0.1
    assert cfg.learning_rate == pytest.approx(5e-5)
    assert cfg.weight_decay == pytest.approx(0.01)
    assert cfg.batch_size == 5

    # Paths (overlapping with preprocessing_config)
    assert cfg.input_tensor_path == "odd_output"
    assert cfg.dataset_name == "seeding_data"
    assert cfg.model_path == "transformer.pt"
    assert cfg.recompute_tensor is False
    assert cfg.no_test is False
    assert cfg.resume_training is False

    # Model architecture (delegated to transformer_config)
    assert cfg.transformer_config.nb_layers_t == 4
    assert cfg.transformer_config.dim_embedding == 128
    assert cfg.transformer_config.nb_heads == 2
    assert cfg.transformer_config.dropout == pytest.approx(0.1)
    assert cfg.transformer_config.fourier_num_frequencies == [15, 15, 15, 15]

    # Loss configuration
    assert cfg.loss_components == ["attention_next"]
    assert cfg.loss_weights == []

    # Radial separation filter defaults (on by default: flat efficiency improvement, no extra cost)
    assert cfg.radial_separation_constraint is True
    assert cfg.min_delta_rho_mm == pytest.approx(5.0)
    assert cfg.raw_chain_length == 5


def test_to_from_dict_roundtrip():
    cfg = SeedConfig()
    # customize a few fields and provide consistent loss/weights
    cfg.epoch_nb = 3
    cfg.preprocessing_config.input_path = "data/in"
    cfg.preprocessing_config.max_hit_input = 1500
    cfg.loss_components = ["hit_BCE", "attention_next"]
    cfg.loss_weights = [0.5, 2.0]
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))
    cfg.radial_separation_constraint = False
    cfg.min_delta_rho_mm = 7.5
    cfg.raw_chain_length = 6

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
    assert new_cfg.loss_components == ["hit_BCE", "attention_next"]
    assert new_cfg.loss_weights == [0.5, 2.0]
    assert new_cfg.get_loss_weight("missing") == 0.0
    assert new_cfg.radial_separation_constraint is False
    assert new_cfg.min_delta_rho_mm == pytest.approx(7.5)
    assert new_cfg.raw_chain_length == 6


def test_save_and_load_config(tmp_path):
    cfg = SeedConfig()
    cfg.epoch_nb = 42
    cfg.loss_components = ["hit_BCE"]
    cfg.loss_weights = [3.0]
    cfg.loss_config = {"hit_BCE": 3.0}
    cfg.radial_separation_constraint = False
    cfg.min_delta_rho_mm = 8.0
    cfg.raw_chain_length = 7

    file_path = tmp_path / "conf.json"
    cfg.save_config(str(file_path))
    assert file_path.exists()

    loaded = SeedConfig()
    loaded.load_config(str(file_path))

    assert loaded.epoch_nb == 42
    assert loaded.loss_components == ["hit_BCE"]
    assert loaded.get_loss_weight("hit_BCE") == 3.0
    assert loaded.radial_separation_constraint is False
    assert loaded.min_delta_rho_mm == pytest.approx(8.0)
    assert loaded.raw_chain_length == 7


def test_parse_args_infers_default_weights(monkeypatch):
    # Provide components but no weights -> defaults to 1.0 each
    argv = ["prog", "--loss_components", "hit_BCE", "attention_next", "--regression"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.loss_components == ["hit_BCE", "attention_next"]
    assert cfg.loss_weights == [1.0, 1.0]
    assert cfg.get_loss_weight("hit_BCE") == 1.0
    assert cfg.get_loss_weight("attention_next") == 1.0


def test_parse_args_mismatched_lengths_raises(monkeypatch):
    argv = [
        "prog",
        "--loss_components",
        "hit_BCE",
        "attention_next",
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
    cfg.loss_components = ["hit_BCE", "attention_next"]
    cfg.loss_weights = [0.7, 1.3]
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))
    assert cfg.has_loss_component("hit_BCE")
    assert not cfg.has_loss_component("attention_back")
    assert cfg.get_loss_weight("attention_next") == 1.3
    assert cfg.get_loss_weight("attention_back") == 0.0


def test_reconstruction_method_valid_choices(monkeypatch):
    """beam_search and beam_search_backward are the only accepted reconstruction methods."""
    for method in ("beam_search", "beam_search_backward"):
        argv = ["prog", "--reconstruction_method", method]
        monkeypatch.setattr(sys, "argv", argv)
        cfg = SeedConfig()
        cfg.parse_args()
        assert cfg.reconstruction_method == method


def test_reconstruction_method_invalid_choice_rejected(monkeypatch):
    """Old method names must be rejected by argparse."""
    for bad in ("chained", "back_chained", "weighted_chained", "topk"):
        argv = ["prog", "--reconstruction_method", bad]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            SeedConfig().parse_args()


def test_reconstruction_method_default_is_none():
    """Default reconstruction_method must be None (auto-select from loss config)."""
    cfg = SeedConfig()
    assert cfg.reconstruction_method is None


def test_radial_separation_constraint_default_true_without_flag(monkeypatch):
    """radial_separation_constraint is on by default even with no CLI args passed."""
    argv = ["prog"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.radial_separation_constraint is True


def test_no_radial_separation_constraint_cli_flag_disables(monkeypatch):
    """--no-radial_separation_constraint is the way to turn off the on-by-default filter
    (argparse.BooleanOptionalAction inserts the 'no-' prefix after the leading '--', not
    after the option name's own underscores)."""
    argv = ["prog", "--no-radial_separation_constraint"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.radial_separation_constraint is False


def test_min_delta_rho_mm_cli_flag(monkeypatch):
    argv = ["prog", "--min_delta_rho_mm", "7.5"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.min_delta_rho_mm == pytest.approx(7.5)


def test_raw_chain_length_cli_flag(monkeypatch):
    argv = ["prog", "--raw_chain_length", "7"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.raw_chain_length == 7


def test_raw_chain_length_below_minimum_raises(monkeypatch):
    """A raw_chain_length < 3 can never produce a valid 3-hit seed and must be rejected eagerly."""
    argv = ["prog", "--raw_chain_length", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        SeedConfig().parse_args()


def test_raw_chain_length_boundary_three_accepted(monkeypatch):
    argv = ["prog", "--raw_chain_length", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg = SeedConfig()
    cfg.parse_args()
    assert cfg.raw_chain_length == 3


def test_print_config(capsys):
    """Test that print_config outputs all configuration sections."""
    cfg = SeedConfig()
    cfg.epoch_nb = 15
    cfg.loss_components = ["hit_BCE", "attention_next"]
    cfg.loss_weights = [0.5, 1.0]
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))
    cfg.model_path = "custom_model.pt"

    cfg.print_config()

    captured = capsys.readouterr()
    output = captured.out

    # Check main sections are present
    assert "Seed Training Configuration" in output
    assert "Preprocessing Settings" in output
    assert "Training Settings:" in output
    assert "File Settings:" in output
    assert "Model Architecture:" in output
    assert "Loss Configuration:" in output
    assert "Radial separation constraint" in output
    assert "Min delta rho" in output
    assert "Raw chain length" in output
