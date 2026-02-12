import argparse
import json
import os


class PreprocessingConfig:
    """
    Class to store the configuration variables for preprocessing and tensor preparation
    """

    def __init__(self):
        """
        Initialise the preprocessing configuration
        Members:
            - input_path: str: Path to the input data files
            - input_format: str: Format of input files ('csv' or 'h5')
            - input_tensor_path: str: Path where output tensors will be saved
            - dataset_name: str: Base name for dataset files (will be combined with barcode)

            - events_per_file: int: Maximum number of events per output tensor file
            - max_events: int: Maximum number of events to process, -1 for all events

            - orphan_hit_fraction: float: Fraction of orphan hits to remove (0.0 to 1.0)

            - binning_strategy: str: Binning strategy ('no_bin', 'global', 'neighbor', or 'margin')
            - bin_width: float: Width of bins for binning in phi
            - binning_margin: float: Margin for margin binning strategy (fraction of bin_width)
            - max_hit_input: int: Maximum number of hits per bin

            - eta_range: list[float, float]: Eta range for particle selection [min, max]
            - vertex_cuts: list[float, float]: Cuts on d0 and z0 for primary vertex selection

            - hit_features: list[str]: List of hit features to extract from data
            - particle_features: list[str]: List of particle features to extract from data
        """

        # Input/Output paths
        self.input_path = "odd_output"  # Path to input data files
        self.input_format = "csv"  # Format of input files: 'csv' or 'h5'
        self.input_tensor_path = "odd_output"  # Path where output tensors will be saved
        self.dataset_name = "seeding_data"  # Base name for dataset files

        # Processing parameters
        self.events_per_file = 100  # Maximum number of events per output tensor file
        self.max_events = -1  # Maximum number of events to process (-1 for all events)

        # Orphan hit removal
        self.orphan_hit_fraction = 0.0  # Fraction of orphan hits to remove (0.0 to 1.0)

        # Binning parameters
        self.binning_strategy = "neighbor"  # Binning strategy: 'no_bin', 'global', 'neighbor', or 'margin'
        self.bin_width = 0.05  # Width of bins for binning in phi
        self.binning_margin = 0.01  # Margin for margin binning strategy (fraction of bin_width)
        self.max_hit_input = 1200  # Maximum number of hits per bin

        # Selection parameters
        self.eta_range = [-3.0, 3.0]  # Eta range for particle selection [min, max]
        self.vertex_cuts = [10, 200]  # Cuts on d0 and z0 for primary vertex selection

        # Feature lists
        self.hit_features = ["x", "y", "z"]  # List of hit features to extract
        self.particle_features = ["eta", "phi", "pT"]  # List of particle features to extract

    def parse_args(self):
        """
        Parse the command line arguments to fill the configuration
        """
        parser = argparse.ArgumentParser(
            description="Configure preprocessing and tensor preparation from the command line",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Input/Output paths
        parser.add_argument(
            "--input_path",
            type=str,
            default=self.input_path,
            help="Path to input data files",
        )
        parser.add_argument(
            "--input_format",
            type=str,
            default=self.input_format,
            choices=["csv", "h5"],
            help="Format of input files ('csv' or 'h5')",
        )
        parser.add_argument(
            "--input_tensor_path",
            type=str,
            default=self.input_tensor_path,
            help="Path where output tensors will be saved (if None, uses input_path)",
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default=self.dataset_name,
            help="Base name for dataset files (will be combined with barcode)",
        )

        # Processing parameters
        parser.add_argument(
            "--events_per_file",
            type=int,
            default=self.events_per_file,
            help="Maximum number of events per output tensor file",
        )
        parser.add_argument(
            "--max_events",
            type=int,
            default=self.max_events,
            help="Maximum number of events to process (-1 for all events)",
        )

        # Orphan hit removal
        parser.add_argument(
            "--orphan_hit_fraction",
            type=float,
            default=self.orphan_hit_fraction,
            help="Fraction of orphan hits to remove (0.0 to 1.0)",
        )

        # Binning parameters
        parser.add_argument(
            "--binning_strategy",
            type=str,
            default=self.binning_strategy,
            choices=["global", "neighbor", "margin", "no_bin"],
            help="Binning strategy to use",
        )
        parser.add_argument(
            "--bin_width",
            type=float,
            default=self.bin_width,
            help="Width of bins for binning in phi",
        )
        parser.add_argument(
            "--binning_margin",
            type=float,
            default=self.binning_margin,
            help="Margin for margin binning strategy (fraction of bin_width)",
        )
        parser.add_argument(
            "--max_hit_input",
            type=int,
            default=self.max_hit_input,
            help="Maximum number of hits per bin",
        )

        # Selection parameters
        parser.add_argument(
            "--eta_range",
            nargs=2,
            type=float,
            default=self.eta_range,
            help="Eta range for particle selection [min, max]",
        )
        parser.add_argument(
            "--vertex_cuts",
            nargs=2,
            type=float,
            default=self.vertex_cuts,
            help="Cuts on d0 and z0 for primary vertex selection [d0_max, z0_max]",
        )

        # Feature lists
        parser.add_argument(
            "--hit_features",
            nargs="+",
            default=self.hit_features,
            help="List of hit features to extract from data",
        )
        parser.add_argument(
            "--particle_features",
            nargs="+",
            default=self.particle_features,
            help="List of particle features to extract from data",
        )

        # Configuration file arguments
        parser.add_argument(
            "--save_config",
            type=str,
            help="Save current configuration to a JSON file",
        )
        parser.add_argument(
            "--load_config",
            type=str,
            help="Load configuration from a JSON file",
        )

        args = parser.parse_args()

        # Handle config file loading first (before setting other args)
        if args.load_config:
            self.load_config(args.load_config)
            print(f"Configuration loaded from {args.load_config}")
            print("All the other arguments will be overridden by the loaded configuration.")
            return

        # Update configuration with parsed arguments
        self.input_path = args.input_path
        self.input_format = args.input_format
        self.input_tensor_path = args.input_tensor_path if args.input_tensor_path else args.input_path
        self.dataset_name = args.dataset_name

        self.events_per_file = args.events_per_file
        self.max_events = args.max_events

        self.orphan_hit_fraction = args.orphan_hit_fraction

        self.binning_strategy = args.binning_strategy
        self.bin_width = args.bin_width
        self.binning_margin = args.binning_margin
        self.max_hit_input = args.max_hit_input

        self.eta_range = args.eta_range
        self.vertex_cuts = args.vertex_cuts

        self.hit_features = args.hit_features
        self.particle_features = args.particle_features

        # Validate orphan_hit_fraction range
        if self.orphan_hit_fraction < 0.0 or self.orphan_hit_fraction > 1.0:
            raise ValueError(f"orphan_hit_fraction must be between 0.0 and 1.0, got {self.orphan_hit_fraction}")

        # Handle config file saving (after all configuration is set)
        if args.save_config:
            self.save_config(args.save_config)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            config_dict[key] = value
        return config_dict

    def from_dict(self, config_dict: dict):
        """Load configuration from dictionary"""
        for key, value in config_dict.items():
            setattr(self, key, value)

    def save_config(self, filepath: str):
        """Save configuration to a JSON file"""
        config_dict = self.to_dict()

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from a JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        self.from_dict(config_dict)
        print(f"Configuration loaded from {filepath}")

    def print_config(self):
        """
        Print the configuration
        """
        print("Preprocessing Configuration:")
        print("\nInput/Output:")
        print("  Input path: ", self.input_path)
        print("  Input format: ", self.input_format)
        print("  Input tensor path: ", self.input_tensor_path)
        print("  Dataset name: ", self.dataset_name)

        print("\nProcessing:")
        print("  Events per file: ", self.events_per_file)
        print("  Max events: ", self.max_events)

        print("\nOrphan Hit Removal:")
        print("  Orphan hit fraction: ", self.orphan_hit_fraction)

        print("\nBinning:")
        print("  Binning strategy: ", self.binning_strategy)
        print("  Bin width (phi): ", self.bin_width)
        print("  Binning margin: ", self.binning_margin)
        print("  Max hit input: ", self.max_hit_input)

        print("\nSelection:")
        print("  Eta range: ", self.eta_range)
        print("  Vertex cuts (d0, z0): ", self.vertex_cuts)

        print("\nFeatures:")
        print("  Hit features: ", self.hit_features)
        print("  Particle features: ", self.particle_features)

        print("\nConfiguration file operations available:")
        print("  --save_config <filename>     : Save current config to JSON file")
        print("  --load_config <filename>     : Load config from JSON file")
