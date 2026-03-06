import pytest
import os
import sys
import tempfile
from GUNTAM.IO.PreprocessingConfig import PreprocessingConfig


class TestPreprocessingConfig:
    """Tests for the PreprocessingConfig class."""

    def test_default_initialization(self):
        """Test that default values are initialized correctly."""
        config = PreprocessingConfig()

        # Input/Output paths
        assert config.input_path == "odd_output"
        assert config.input_format == "csv"
        assert config.tensor_format == "pt"
        assert config.input_tensor_path == "odd_output"
        assert config.dataset_name == "seeding_data"

        # Processing parameters
        assert config.events_per_file == 100
        assert config.max_events == -1

        # Orphan hit removal
        assert config.orphan_hit_fraction == 0.0

        # Binning parameters
        assert config.binning_strategy == "neighbor"
        assert config.bin_width == 0.05
        assert config.binning_margin == 0.01
        assert config.max_hit_input == 1200

        # Selection parameters
        assert config.eta_range == [-3.0, 3.0]
        assert config.vertex_cuts == [10, 200]

        # Feature lists
        assert config.hit_features == ["x", "y", "z"]
        assert config.particle_features == ["eta", "phi", "pT"]

        # Event weights
        assert config.pv_pair_weight == 10

    def test_eta_range_list_structure(self):
        """Test that eta_range is a list with two elements."""
        config = PreprocessingConfig()

        assert isinstance(config.eta_range, list)
        assert len(config.eta_range) == 2
        assert config.eta_range[0] < config.eta_range[1]  # min < max

    def test_vertex_cuts_list_structure(self):
        """Test that vertex_cuts is a list with two elements."""
        config = PreprocessingConfig()

        assert isinstance(config.vertex_cuts, list)
        assert len(config.vertex_cuts) == 2
        # Both should be positive
        assert all(v > 0 for v in config.vertex_cuts)

    def test_feature_lists_are_lists(self):
        """Test that feature lists are properly initialized as lists."""
        config = PreprocessingConfig()

        assert isinstance(config.hit_features, list)
        assert isinstance(config.particle_features, list)
        assert len(config.hit_features) > 0
        assert len(config.particle_features) > 0

    def test_positive_processing_parameters(self):
        """Test that processing parameters have sensible values."""
        config = PreprocessingConfig()

        assert config.events_per_file > 0
        assert config.max_hit_input > 0
        assert config.bin_width > 0
        assert config.binning_margin >= 0

    def test_to_dict(self):
        """Test conversion of config to dictionary."""
        config = PreprocessingConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["input_path"] == "odd_output"
        assert config_dict["input_format"] == "csv"
        assert config_dict["tensor_format"] == "pt"
        assert config_dict["binning_strategy"] == "neighbor"
        assert config_dict["eta_range"] == [-3.0, 3.0]
        assert config_dict["pv_pair_weight"] == 10

    def test_from_dict(self):
        """Test loading config from dictionary."""
        config = PreprocessingConfig()

        test_dict = {
            "input_path": "test_path",
            "input_format": "h5",
            "tensor_format": "h5",
            "events_per_file": 50,
            "max_events": 1000,
            "orphan_hit_fraction": 0.5,
            "binning_strategy": "global",
            "bin_width": 0.1,
            "eta_range": [-2.0, 2.0],
        }

        config.from_dict(test_dict)

        assert config.input_path == "test_path"
        assert config.input_format == "h5"
        assert config.tensor_format == "h5"
        assert config.events_per_file == 50
        assert config.max_events == 1000
        assert config.orphan_hit_fraction == 0.5
        assert config.binning_strategy == "global"
        assert config.bin_width == 0.1
        assert config.eta_range == [-2.0, 2.0]

    def test_config_format_choices(self):
        """Test that valid input and output format choices are available."""
        config = PreprocessingConfig()

        # Valid input formats
        valid_input_formats = ["csv", "h5"]
        for fmt in valid_input_formats:
            config.input_format = fmt
            assert config.input_format == fmt
        
        # Valid tensor formats
        valid_tensor_formats = ["pt", "h5"]
        for fmt in valid_tensor_formats:
            config.tensor_format = fmt
            assert config.tensor_format == fmt

    def test_binning_strategy_choices(self):
        """Test that valid binning strategy choices are available."""
        config = PreprocessingConfig()

        # Valid strategies
        valid_strategies = ["global", "neighbor", "margin", "no_bin"]
        for strategy in valid_strategies:
            config.binning_strategy = strategy
            assert config.binning_strategy == strategy

    def test_save_config_creates_directory(self):
        """Test that save_config creates directories if they don't exist."""
        config = PreprocessingConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "subdir", "test_config.json")

            # Save should create the subdir
            config.save_config(config_file)
            assert os.path.exists(config_file)

    def test_load_config_file_not_found(self):
        """Test that loading a non-existent config file raises FileNotFoundError."""
        config = PreprocessingConfig()

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            config.load_config("nonexistent_config.json")

    def test_config_persistence_all_fields(self):
        """Test that all configuration fields are saved and loaded correctly."""
        config = PreprocessingConfig()

        # Modify all fields
        config.input_path = "modified_path"
        config.input_format = "h5"
        config.tensor_format = "h5"
        config.input_tensor_path = "tensor_path"
        config.dataset_name = "modified_dataset"
        config.events_per_file = 150
        config.max_events = 5000
        config.orphan_hit_fraction = 0.3
        config.binning_strategy = "global"
        config.bin_width = 0.08
        config.binning_margin = 0.02
        config.max_hit_input = 1500
        config.eta_range = [-2.0, 2.0]
        config.vertex_cuts = [15, 250]
        config.hit_features = ["x", "y", "z", "r"]
        config.particle_features = ["eta", "phi", "pT", "d0"]
        config.pv_pair_weight = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "full_test.json")

            # Save and load
            config.save_config(config_file)
            loaded_config = PreprocessingConfig()
            loaded_config.load_config(config_file)

            # Verify all fields
            assert loaded_config.input_path == "modified_path"
            assert loaded_config.input_format == "h5"
            assert loaded_config.tensor_format == "h5"
            assert loaded_config.input_tensor_path == "tensor_path"
            assert loaded_config.dataset_name == "modified_dataset"
            assert loaded_config.events_per_file == 150
            assert loaded_config.max_events == 5000
            assert loaded_config.orphan_hit_fraction == 0.3
            assert loaded_config.binning_strategy == "global"
            assert loaded_config.bin_width == 0.08
            assert loaded_config.binning_margin == 0.02
            assert loaded_config.max_hit_input == 1500
            assert loaded_config.eta_range == [-2.0, 2.0]
            assert loaded_config.vertex_cuts == [15, 250]
            assert loaded_config.hit_features == ["x", "y", "z", "r"]
            assert loaded_config.particle_features == ["eta", "phi", "pT", "d0"]
            assert loaded_config.pv_pair_weight == 5

    def test_parse_args_basic(self, monkeypatch):
        """Test basic argument parsing with various parameters."""
        argv = [
            "prog",
            "--input_path", "test_data",
            "--input_format", "h5",
            "--tensor_format", "h5",
            "--max_events", "500",
            "--binning_strategy", "global",
            "--bin_width", "0.08",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        
        config = PreprocessingConfig()
        config.parse_args()
        
        assert config.input_path == "test_data"
        assert config.input_format == "h5"
        assert config.tensor_format == "h5"
        assert config.max_events == 500
        assert config.binning_strategy == "global"
        assert config.bin_width == 0.08

    def test_print_config(self, capsys):
        """Test that print_config outputs all configuration sections."""
        config = PreprocessingConfig()
        config.input_path = "custom_path"
        config.input_format = "h5"
        config.binning_strategy = "global"
        config.max_events = 2000
        config.orphan_hit_fraction = 0.25
        
        config.print_config()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check main sections are present
        assert "Preprocessing Configuration:" in output
        assert "Input/Output:" in output
        assert "Processing:" in output
        assert "Orphan Hit Removal:" in output
        assert "Binning:" in output
        assert "Selection:" in output
        assert "Features:" in output
        assert "Event Weights:" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
