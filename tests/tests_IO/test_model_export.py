"""
Tests for GUNTAM/IO/Export_full_model.py

Tests cover:
1. Argument parsing (parse_args)
2. Exporting the transformer to ONNX using checkpoints from tests/data/
3. Exporting the classifier to ONNX
5. Configuration file handling
"""

import os
import sys
from pathlib import Path

import onnx
import pytest

from GUNTAM.IO.Export_full_model import parse_args

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent.parent / "data"
TRANSFORMER_CHECKPOINT = TEST_DATA_DIR / "transformer.pt"
CLASSIFIER_CHECKPOINT = TEST_DATA_DIR / "classifier.pt"

# ---------------------------------------------------------------------------
# Test argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for the parse_args function."""

    def test_required_args_only(self, monkeypatch, tmp_path):
        """Test that only --checkpoint is required."""
        monkeypatch.setattr(sys, "argv", ["prog", "--checkpoint", str(TRANSFORMER_CHECKPOINT)])
        args = parse_args()

        assert args.checkpoint == str(TRANSFORMER_CHECKPOINT)
        # Defaults
        assert args.config is None
        assert args.classifier_checkpoint is None
        assert args.classifier_config is None
        assert args.classifier_threshold is None
        assert args.transformer_output == "transformer.onnx"
        assert args.classifier_output == "classifier.onnx"
        assert args.width == 5
        assert args.max_seed_length == 3
        assert args.num_example_hits == 50000
        assert args.device == "cpu"

    def test_missing_checkpoint_raises_system_exit(self, monkeypatch):
        """--checkpoint is required; absence should raise SystemExit."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_all_arguments_with_classifier(self, monkeypatch, tmp_path):
        """Test that all arguments are parsed correctly."""
        dummy_config = tmp_path / "seed_config.json"
        dummy_classifier_config = tmp_path / "classifier_config.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--config",
                str(dummy_config),
                "--classifier_checkpoint",
                str(CLASSIFIER_CHECKPOINT),
                "--classifier_config",
                str(dummy_classifier_config),
                "--classifier_threshold",
                "0.7",
                "--transformer_output",
                str(tmp_path / "transformer_test.onnx"),
                "--classifier_output",
                str(tmp_path / "classifier_test.onnx"),
                "--width",
                "7",
                "--max_seed_length",
                "4",
                "--num_example_hits",
                "1000",
                "--device",
                "cpu",
            ],
        )
        args = parse_args()

        assert args.checkpoint == str(TRANSFORMER_CHECKPOINT)
        assert args.config == str(dummy_config)
        assert args.classifier_checkpoint == str(CLASSIFIER_CHECKPOINT)
        assert args.classifier_config == str(dummy_classifier_config)
        assert args.classifier_threshold == 0.7
        assert args.transformer_output == str(tmp_path / "transformer_test.onnx")
        assert args.classifier_output == str(tmp_path / "classifier_test.onnx")
        assert args.width == 7
        assert args.max_seed_length == 4
        assert args.num_example_hits == 1000
        assert args.device == "cpu"

    def test_transformer_only_arguments(self, monkeypatch, tmp_path):
        """Test parsing with only transformer options, no classifier."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--transformer_output",
                str(tmp_path / "t.onnx"),
                "--width",
                "3",
                "--max_seed_length",
                "2",
                "--num_example_hits",
                "100",
            ],
        )
        args = parse_args()

        assert args.checkpoint == str(TRANSFORMER_CHECKPOINT)
        assert args.transformer_output == str(tmp_path / "t.onnx")
        assert args.width == 3
        assert args.max_seed_length == 2
        assert args.num_example_hits == 100
        assert args.classifier_checkpoint is None

    def test_custom_device(self, monkeypatch):
        """Test that a custom device string is parsed correctly."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--checkpoint", str(TRANSFORMER_CHECKPOINT), "--device", "cuda:0"],
        )
        args = parse_args()
        assert args.device == "cuda:0"

    def test_num_example_hits_custom_value(self, monkeypatch):
        """Test that num_example_hits is parsed as an int."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--checkpoint", str(TRANSFORMER_CHECKPOINT), "--num_example_hits", "12345"],
        )
        args = parse_args()
        assert args.num_example_hits == 12345


# ---------------------------------------------------------------------------
# Test exporting the transformer
# ---------------------------------------------------------------------------


class TestExportTransformerOnly:
    """Tests for exporting only the transformer to ONNX."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, tmp_path):
        """Clean up any side-effect files (like training_config.sh)."""
        yield
        # Remove any training_config.sh that might be created in the cwd
        cwd_config = Path.cwd() / "training_config.sh"
        if cwd_config.exists():
            cwd_config.unlink()

    def test_export_transformer(self, tmp_path):
        """Test exporting the transformer checkpoint to ONNX."""
        from GUNTAM.IO.Export_full_model import main

        transformer_output = str(tmp_path / "transformer_test.onnx")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--transformer_output",
                transformer_output,
                "--num_example_hits",
                "200",
                "--width",
                "5",
                "--max_seed_length",
                "3",
                "--device",
                "cpu",
            ],
        )

        try:
            main()
        finally:
            monkeypatch.undo()

        # Verify the ONNX file was created and is non-empty
        assert os.path.exists(transformer_output)
        assert os.path.getsize(transformer_output) > 0

        # Validate the ONNX model structure
        model = onnx.load(transformer_output)
        onnx.checker.check_model(model)

        # Verify input/output names
        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]

        assert "hits" in input_names
        assert "padding_mask" in input_names
        assert "output" in output_names
        assert "attention_weights" in output_names

    def test_export_transformer_with_classifier(self, tmp_path):
        """Test exporting both transformer and classifier to ONNX."""
        from GUNTAM.IO.Export_full_model import main

        transformer_output = str(tmp_path / "transformer_full.onnx")
        classifier_output = str(tmp_path / "classifier_full.onnx")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--classifier_checkpoint",
                str(CLASSIFIER_CHECKPOINT),
                "--transformer_output",
                transformer_output,
                "--classifier_output",
                classifier_output,
                "--num_example_hits",
                "200",
                "--width",
                "5",
                "--max_seed_length",
                "3",
                "--device",
                "cpu",
            ],
        )

        try:
            main()
        finally:
            monkeypatch.undo()

        # Verify both ONNX files were created
        assert os.path.exists(transformer_output)
        assert os.path.exists(classifier_output)
        assert os.path.getsize(transformer_output) > 0
        assert os.path.getsize(classifier_output) > 0

        # Validate the transformer ONNX model
        transformer_model = onnx.load(transformer_output)
        onnx.checker.check_model(transformer_model)

        # Validate the classifier ONNX model
        classifier_model = onnx.load(classifier_output)
        onnx.checker.check_model(classifier_model)

        # Verify classifier input/output names
        cls_input_names = [inp.name for inp in classifier_model.graph.input]
        cls_output_names = [out.name for out in classifier_model.graph.output]
        assert "seed_features" in cls_input_names
        assert "keep_mask" in cls_output_names
        assert "scores" in cls_output_names


class TestExportWithConfig:
    """Tests for exporting with configuration files."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, tmp_path):
        """Clean up any side-effect files."""
        yield
        cwd_config = Path.cwd() / "training_config.sh"
        if cwd_config.exists():
            cwd_config.unlink()

    def test_export_with_seed_config(self, tmp_path):
        """Test exporting with a custom SeedConfig JSON file."""
        from GUNTAM.IO.Export_full_model import main
        from GUNTAM.Seed.Config.SeedConfig import SeedConfig

        # Create a SeedConfig and save it as JSON
        cfg = SeedConfig()
        config_path = tmp_path / "seed_config.json"
        cfg.save_config(str(config_path))

        transformer_output = str(tmp_path / "transformer_cfg.onnx")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--config",
                str(config_path),
                "--transformer_output",
                transformer_output,
                "--num_example_hits",
                "100",
                "--device",
                "cpu",
            ],
        )

        try:
            main()
        finally:
            monkeypatch.undo()

        assert os.path.exists(transformer_output)

    def test_export_with_classifier_config(self, tmp_path):
        """Test exporting with a custom ClassifierConfig JSON file."""
        from GUNTAM.IO.Export_full_model import main
        from GUNTAM.Seed.Config.ClassifierConfig import ClassifierConfig

        # Create a classifier config with a custom threshold
        classifier_cfg = ClassifierConfig()
        classifier_cfg.threshold = 0.7
        config_path = tmp_path / "classifier_config.json"
        classifier_cfg.save_config(str(config_path))

        transformer_output = str(tmp_path / "t_cfg.onnx")
        classifier_output = str(tmp_path / "c_cfg.onnx")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--classifier_checkpoint",
                str(CLASSIFIER_CHECKPOINT),
                "--classifier_config",
                str(config_path),
                "--transformer_output",
                transformer_output,
                "--classifier_output",
                classifier_output,
                "--num_example_hits",
                "100",
                "--device",
                "cpu",
            ],
        )

        try:
            main()
        finally:
            monkeypatch.undo()

        assert os.path.exists(transformer_output)
        assert os.path.exists(classifier_output)

    def test_export_with_threshold_override(self, tmp_path):
        """Test that --classifier_threshold overrides the config/checkpoint value."""
        from GUNTAM.IO.Export_full_model import main

        transformer_output = str(tmp_path / "t_thresh.onnx")
        classifier_output = str(tmp_path / "c_thresh.onnx")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--checkpoint",
                str(TRANSFORMER_CHECKPOINT),
                "--classifier_checkpoint",
                str(CLASSIFIER_CHECKPOINT),
                "--classifier_threshold",
                "0.8",
                "--transformer_output",
                transformer_output,
                "--classifier_output",
                classifier_output,
                "--num_example_hits",
                "100",
                "--device",
                "cpu",
            ],
        )

        try:
            main()
        finally:
            monkeypatch.undo()

        assert os.path.exists(transformer_output)
        assert os.path.exists(classifier_output)
