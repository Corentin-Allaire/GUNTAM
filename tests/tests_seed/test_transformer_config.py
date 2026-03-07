import sys
import pytest

from GUNTAM.Seed.TransformerConfig import TransformerConfig


def test_defaults_initialization():
    cfg = TransformerConfig()
    assert cfg.nb_layers_t == 4
    assert cfg.nb_heads == 2
    assert cfg.dim_embedding == 128
    assert cfg.feed_forward_ratio == 2
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.embedding_feature == [0, 1, 2, 3]
    assert cfg.high_level_features == [4, 5]
    assert cfg.cosine_processing == [4]
    assert cfg.fourier_num_frequencies == [15, 15, 15, 15]
    assert cfg.dim_max == [400.0, 400.0, 2000.0, 500]
    assert cfg.shift == [200, 200, 1000.0, 0.0]


def test_to_from_dict_roundtrip():
    cfg = TransformerConfig()
    cfg.nb_layers_t = 6
    cfg.nb_heads = 4
    cfg.dim_embedding = 64
    cfg.embedding_feature = [0, 1, 2]
    cfg.dim_max = [400.0, 400.0, 2000.0]
    cfg.shift = [200.0, 200.0, 1000.0]
    cfg.cosine_processing = []

    d = cfg.to_dict()
    assert d["nb_layers_t"] == 6
    assert d["dim_embedding"] == 64

    new_cfg = TransformerConfig()
    new_cfg.from_dict(d)
    assert new_cfg.nb_layers_t == 6
    assert new_cfg.nb_heads == 4
    assert new_cfg.dim_embedding == 64
    assert new_cfg.embedding_feature == [0, 1, 2]
    assert new_cfg.cosine_processing == []


def test_save_and_load_config(tmp_path):
    cfg = TransformerConfig()
    cfg.nb_layers_t = 3
    cfg.dropout = 0.2

    path = str(tmp_path / "tc.json")
    cfg.save_config(path)

    loaded = TransformerConfig()
    loaded.load_config(path)
    assert loaded.nb_layers_t == 3
    assert loaded.dropout == pytest.approx(0.2)


def test_load_config_missing_file():
    cfg = TransformerConfig()
    with pytest.raises(FileNotFoundError):
        cfg.load_config("/nonexistent/path/tc.json")


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    cfg = TransformerConfig()
    cfg.parse_args()
    assert cfg.nb_layers_t == 4
    assert cfg.nb_heads == 2
    assert cfg.dim_embedding == 128
    # fourier_num_frequencies is derived automatically when None
    assert cfg.fourier_num_frequencies is not None
    assert len(cfg.fourier_num_frequencies) == len(cfg.embedding_feature)


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--nb_layers_t", "2",
        "--nb_heads", "4",
        "--dim_embedding", "64",
        "--dropout", "0.05",
        "--embedding_feature", "0", "1", "2",
        "--high_level_features", "3",
        "--cosine_processing", "3",
        "--dim_max", "400.0", "400.0", "2000.0",
        "--shift", "200.0", "200.0", "1000.0",
    ])
    cfg = TransformerConfig()
    cfg.parse_args()
    assert cfg.nb_layers_t == 2
    assert cfg.nb_heads == 4
    assert cfg.dim_embedding == 64
    assert cfg.dropout == pytest.approx(0.05)
    assert cfg.embedding_feature == [0, 1, 2]
    assert cfg.high_level_features == [3]
    assert cfg.cosine_processing == [3]


def test_parse_args_fourier_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1", "2", "3",
        "--high_level_features", "4",
        "--cosine_processing",
        "--dim_max", "400.0", "400.0", "2000.0", "500.0",
        "--shift", "200.0", "200.0", "1000.0", "0.0",
        "--fourier_num_frequencies", "8", "8", "16", "4",
    ])
    cfg = TransformerConfig()
    cfg.parse_args()
    assert cfg.fourier_num_frequencies == [8, 8, 16, 4]


def test_validation_dropout_out_of_range(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--dropout", "1.0"])
    with pytest.raises(ValueError, match="dropout"):
        TransformerConfig().parse_args()


def test_validation_cosine_not_in_features(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1",
        "--high_level_features", "2",
        "--cosine_processing", "9",   # 9 not in {0,1,2}
        "--dim_max", "400.0", "400.0",
        "--shift", "0.0", "0.0",
    ])
    with pytest.raises(ValueError, match="cosine_processing"):
        TransformerConfig().parse_args()


def test_validation_fourier_length_mismatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1", "2", "3",
        "--fourier_num_frequencies", "8", "8",   # length 2 != 4
        "--dim_max", "400.0", "400.0", "2000.0", "500.0",
        "--shift", "200.0", "200.0", "1000.0", "0.0",
    ])
    with pytest.raises(ValueError, match="fourier_num_frequencies"):
        TransformerConfig().parse_args()


def test_validation_dim_max_length_mismatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1",
        "--dim_max", "400.0",          # length 1 != 2
        "--shift", "0.0", "0.0",
    ])
    with pytest.raises(ValueError, match="dim_max"):
        TransformerConfig().parse_args()


def test_validation_shift_length_mismatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1",
        "--dim_max", "400.0", "400.0",
        "--shift", "0.0",              # length 1 != 2
    ])
    with pytest.raises(ValueError, match="shift"):
        TransformerConfig().parse_args()


def test_validation_nb_layers_less_than_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--nb_layers_t", "0"])
    with pytest.raises(ValueError, match="nb_layers_t"):
        TransformerConfig().parse_args()


def test_fourier_auto_derivation(monkeypatch):
    """fourier_num_frequencies should be auto-derived when not supplied."""
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--dim_embedding", "64",
        "--embedding_feature", "0", "1", "2", "3",
        "--high_level_features", "4",
        "--cosine_processing",
        "--dim_max", "400.0", "400.0", "2000.0", "500.0",
        "--shift", "200.0", "200.0", "1000.0", "0.0",
    ])
    cfg = TransformerConfig()
    cfg.parse_args()
    expected = max(1, (64 - 1) // (2 * 4))
    assert cfg.fourier_num_frequencies == [expected] * 4


def test_high_level_features_can_be_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1",
        "--high_level_features",        # empty
        "--cosine_processing",          # empty
        "--dim_max", "400.0", "400.0",
        "--shift", "0.0", "0.0",
    ])
    cfg = TransformerConfig()
    cfg.parse_args()
    assert cfg.high_level_features == []
    assert cfg.cosine_processing == []


def test_print_config(capsys):
    cfg = TransformerConfig()
    cfg.print_config()
    out = capsys.readouterr().out
    assert "Transformer Architecture Configuration" in out
    assert "Number of layers" in out
    assert "Embedding features" in out


def test_dim_max_shift_match_coord_dim_with_cosine_in_embedding(monkeypatch):
    """When cosine_processing overlaps embedding_feature, coord_dim > n_embed.
    dim_max and shift must have coord_dim length, not n_embed length."""
    # embedding_feature = [0, 1, 2], cosine_processing = [0]
    # intersection = {0} → coord_dim = 3 + 1 = 4, n_embed = 3
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1", "2",
        "--high_level_features",
        "--cosine_processing", "0",
        "--dim_max", "400.0", "400.0", "2000.0", "400.0",  # length 4 == coord_dim
        "--shift", "200.0", "200.0", "1000.0", "200.0",    # length 4 == coord_dim
        "--fourier_num_frequencies", "4", "4", "4", "4",
    ])
    cfg = TransformerConfig()
    cfg.parse_args()
    assert len(cfg.dim_max) == 4
    assert len(cfg.shift) == 4


def test_validation_dim_max_must_match_coord_dim_not_n_embed(monkeypatch):
    """Providing dim_max with n_embed length instead of coord_dim length should fail
    when cosine_processing overlaps embedding_feature."""
    # embedding_feature = [0, 1, 2], cosine_processing = [0]
    # coord_dim = 4, n_embed = 3 → dim_max length 3 should raise
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1", "2",
        "--high_level_features",
        "--cosine_processing", "0",
        "--dim_max", "400.0", "400.0", "2000.0",   # length 3 == n_embed, but coord_dim == 4
        "--shift", "200.0", "200.0", "1000.0", "200.0",
    ])
    with pytest.raises(ValueError, match="dim_max"):
        TransformerConfig().parse_args()


def test_validation_shift_must_match_coord_dim_not_n_embed(monkeypatch):
    """Providing shift with n_embed length instead of coord_dim length should fail
    when cosine_processing overlaps embedding_feature."""
    # embedding_feature = [0, 1, 2], cosine_processing = [0]
    # coord_dim = 4, n_embed = 3 → shift length 3 should raise
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--embedding_feature", "0", "1", "2",
        "--high_level_features",
        "--cosine_processing", "0",
        "--dim_max", "400.0", "400.0", "2000.0", "400.0",
        "--shift", "200.0", "200.0", "1000.0",             # length 3 == n_embed, but coord_dim == 4
    ])
    with pytest.raises(ValueError, match="shift"):
        TransformerConfig().parse_args()
