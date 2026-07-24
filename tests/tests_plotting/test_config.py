"""Tests for GUNTAM.Plotting.Config."""

import sys

import pytest

from GUNTAM.Plotting.Config import PlottingConfig, parse_args


class TestPlottingConfigDefaults:
    def test_defaults(self):
        cfg = PlottingConfig()
        assert cfg.files == []
        assert cfg.labels == []
        assert cfg.compare == "seeding-vs-ckf"
        assert cfg.quantities == ["all"]
        assert cfg.output_dir == "plots"
        assert cfg.output_formats == ["png"]
        assert cfg.dpi == 150

    def test_labels_files_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels"):
            PlottingConfig(files=["a.root", "b.root"], labels=["only_one"])

    def test_bad_compare_raises(self):
        with pytest.raises(ValueError, match="compare"):
            PlottingConfig(compare="not-a-real-option")

    def test_empty_quantities_raises(self):
        with pytest.raises(ValueError, match="quantities"):
            PlottingConfig(quantities=[])


class TestPlottingConfigSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        cfg = PlottingConfig(files=["a.root"], labels=["A"], compare="seeding-vs-ambi", quantities=["trackeff_vs_eta"])
        restored = PlottingConfig.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_save_load_json_roundtrip(self, tmp_path):
        cfg = PlottingConfig(files=["a.root", "b.root"], labels=["A", "B"], quantities=["trackeff_vs_pT"])
        path = tmp_path / "config.json"
        cfg.save(str(path))
        loaded = PlottingConfig.load(str(path))
        assert loaded == cfg

    def test_print_config(self, capsys):
        cfg = PlottingConfig(files=["a.root"])
        cfg.print_config()
        out = capsys.readouterr().out
        assert "Plotting Configuration" in out
        assert "a.root" in out


class TestParseArgsCLIOnly:
    def test_cli_only(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--files", "a.root", "b.root", "--labels", "A", "B", "--quantities", "trackeff_vs_eta"],
        )
        cfg = parse_args(include_compare=False)
        assert cfg.files == ["a.root", "b.root"]
        assert cfg.labels == ["A", "B"]
        assert cfg.quantities == ["trackeff_vs_eta"]

    def test_compare_gated_when_not_included(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["prog", "--compare", "seeding-vs-ambi"])
        with pytest.raises(SystemExit):
            parse_args(include_compare=False)

    def test_compare_available_when_included(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--files", "a.root", "--compare", "seeding-vs-ambi"])
        cfg = parse_args(include_compare=True)
        assert cfg.compare == "seeding-vs-ambi"


class TestConfigOverrideSemantics:
    def test_json_then_cli_files_replace_wholesale(self, tmp_path, monkeypatch):
        base = PlottingConfig(files=["a.root", "b.root", "c.root"])
        config_path = tmp_path / "config.json"
        base.save(str(config_path))

        monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--files", "x.root", "y.root"])
        cfg = parse_args(include_compare=False)

        # files replaced wholesale, not merged/appended with the JSON's files
        assert cfg.files == ["x.root", "y.root"]

    def test_final_files_length_validated_against_final_labels(self, tmp_path, monkeypatch):
        base = PlottingConfig(files=["a.root", "b.root"], labels=["A", "B"])
        config_path = tmp_path / "config.json"
        base.save(str(config_path))

        # Overriding files to 3 entries while JSON labels has only 2 must fail validation.
        monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--files", "x.root", "y.root", "z.root"])
        with pytest.raises(ValueError, match="labels"):
            parse_args(include_compare=False)

    def test_cli_overrides_json_when_both_given(self, tmp_path, monkeypatch):
        base = PlottingConfig(output_dir="from_json", dpi=100)
        config_path = tmp_path / "config.json"
        base.save(str(config_path))

        monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--files", "a.root", "--output_dir", "from_cli"])
        cfg = parse_args(include_compare=False)
        assert cfg.output_dir == "from_cli"
        assert cfg.dpi == 100  # untouched by CLI, kept from JSON

    def test_save_config_writes_file(self, tmp_path, monkeypatch):
        out_path = tmp_path / "saved.json"
        monkeypatch.setattr(sys, "argv", ["prog", "--files", "a.root", "--save_config", str(out_path)])
        parse_args(include_compare=False)
        assert out_path.exists()
