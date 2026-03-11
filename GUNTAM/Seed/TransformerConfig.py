import argparse
import json
import os


class TransformerConfig:
    """
    Class to store the configuration variables for the transformer architecture.
    """

    def __init__(self):
        """
        Initialise the transformer architecture configuration.
        Members:
            - nb_layers_t: int: Number of Transformer encoder layers
            - nb_heads: int: Number of attention heads in the Transformer encoder
            - dim_embedding: int: Dimension of the internal embedding space
            - feed_forward_ratio: int: Ratio of the feed-forward layer dimension to dim_embedding
            - dropout: float: Dropout rate used in Transformer and attention layers
            - fourier_num_frequencies: list[int] | None: Number of Fourier frequency bands per
                spatial dimension [x, y, z, r]. If None, derived automatically from dim_embedding
                so that the encoded dimension is close to dim_embedding.
            - embedding_feature: list[int]: Indices of the hit features to be embedded via Fourier
                encoding (e.g. [0,1,2,3] for x, y, z, r).
            - high_level_features: list[int]: Indices of the high-level features to be concatenated
                directly (e.g. [4,5] for phi, eta).
            - cosine_processing: list[int]: Indices of features to which cosine/sine decomposition
                is applied (e.g. [4] for phi).
            - dim_max: list[float]: Maximum expected value per encoded coordinate dimension
                (after cosine expansion, length == coord_dim), used to normalise inputs before
                Fourier encoding [x_max, y_max, z_max, r_max].
            - shift: list[float]: Shift applied to each encoded coordinate dimension before
                normalisation (length == coord_dim) [x_shift, y_shift, z_shift, r_shift].
        """

        # Transformer architecture
        self.nb_layers_t = 4  # Number of Transformer encoder layers
        self.nb_heads = 2  # Number of attention heads
        self.dim_embedding = 128  # Dimension of the internal embedding space
        self.feed_forward_ratio = 2  # Ratio of feed-forward dimension to embedding dimension
        self.dropout = 0.1  # Dropout rate
        self.embedding_feature = [0, 1, 2, 3]  # list of indices of the features to be embedded (0-3 for x,y,z,r)
        self.high_level_features = [4, 5]  # list of indices of the high-level features to be concatenated (4-5 phi, eta)
        self.cosine_processing = [4]  # list of indices of the features to apply cosine processing to (e.g., phi)
        self.fourier_num_frequencies = None  # list[int] of length 4, one per dimension [x, y, z, r]
        self.dim_max = [400.0, 400.0, 2000.0, 500]
        self.shift = [200, 200, 1000.0, 0.0]

        # Regression head
        self.regression = False  # Whether to enable the regression MLP head
        self.num_regression_parameters = 5  # Number of output parameters for the regression head

        coord_dim = len(self.embedding_feature) + len(set(self.embedding_feature) & set(self.cosine_processing))
        if self.fourier_num_frequencies is None:
            high_dim = len(self.high_level_features) + len(set(self.high_level_features) & set(self.cosine_processing))
            self.fourier_num_frequencies = [max(1, (self.dim_embedding - high_dim) // (coord_dim * 2))] * coord_dim

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Add transformer architecture arguments to an existing ArgumentParser.
        Allows sharing argument definitions with a parent parser (e.g. SeedConfig).
        """
        # Transformer architecture
        parser.add_argument(
            "--nb_layers_t",
            type=int,
            default=self.nb_layers_t,
            help="Number of Transformer encoder layers",
        )
        parser.add_argument(
            "--nb_heads",
            type=int,
            default=self.nb_heads,
            help="Number of attention heads in the Transformer encoder",
        )
        parser.add_argument(
            "--dim_embedding",
            type=int,
            default=self.dim_embedding,
            help="Dimension of the internal embedding space",
        )
        parser.add_argument(
            "--feed_forward_ratio",
            type=int,
            default=self.feed_forward_ratio,
            help="Ratio of the feed-forward layer dimension to dim_embedding",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=self.dropout,
            help="Dropout rate used in Transformer and attention layers",
        )

        # Fourier positional encoding
        parser.add_argument(
            "--fourier_num_frequencies",
            nargs="+",
            type=int,
            default=None,
            help=(
                "Number of Fourier frequency bands per embedded feature dimension. "
                "Length must match --embedding_feature. "
                "If not provided, derived automatically from dim_embedding."
            ),
        )
        parser.add_argument(
            "--dim_max",
            nargs="+",
            type=float,
            default=self.dim_max,
            help="Maximum expected value per encoded coordinate dimension for Fourier normalisation. "
            "Length must match coord_dim (embedding_feature + cosine-expanded dims).",
        )
        parser.add_argument(
            "--shift",
            nargs="+",
            type=float,
            default=self.shift,
            help="Shift applied to each encoded coordinate dimension before Fourier normalisation. "
            "Length must match coord_dim (embedding_feature + cosine-expanded dims).",
        )

        # Feature selection
        parser.add_argument(
            "--embedding_feature",
            nargs="+",
            type=int,
            default=self.embedding_feature,
            help="Indices of hit features to embed via Fourier encoding (e.g. 0 1 2 3 for x y z r)",
        )
        parser.add_argument(
            "--high_level_features",
            nargs="*",
            type=int,
            default=self.high_level_features,
            help="Indices of high-level features to concatenate directly (e.g. 4 5 for phi eta)",
        )
        parser.add_argument(
            "--cosine_processing",
            nargs="*",
            type=int,
            default=self.cosine_processing,
            help="Indices of features to apply cosine/sine decomposition to (e.g. 4 for phi)",
        )
        parser.add_argument(
            "--regression",
            action="store_true",
            default=self.regression,
            help="Enable the regression MLP head on top of the transformer encoder",
        )
        parser.add_argument(
            "--num_regression_parameters",
            type=int,
            default=self.num_regression_parameters,
            help="Number of output parameters for the regression MLP head",
        )

    def apply_args(self, args: argparse.Namespace) -> None:
        """
        Apply the values from a parsed Namespace to the configuration.
        Can be called after a shared parent parser has been parsed.
        """
        self.nb_layers_t = args.nb_layers_t
        self.nb_heads = args.nb_heads
        self.dim_embedding = args.dim_embedding
        self.feed_forward_ratio = args.feed_forward_ratio
        self.embedding_feature = args.embedding_feature
        self.high_level_features = args.high_level_features
        self.cosine_processing = args.cosine_processing
        self.dropout = args.dropout
        self.fourier_num_frequencies = args.fourier_num_frequencies
        coord_dim = len(self.embedding_feature) + len(set(self.embedding_feature) & set(self.cosine_processing))
        if self.fourier_num_frequencies is None:
            high_dim = len(self.high_level_features) + len(set(self.high_level_features) & set(self.cosine_processing))
            self.fourier_num_frequencies = [max(1, (self.dim_embedding - high_dim) // (coord_dim * 2))] * coord_dim
        self.dim_max = args.dim_max
        self.shift = args.shift
        self.regression = args.regression
        self.num_regression_parameters = args.num_regression_parameters

        # Validation
        if self.nb_layers_t < 1:
            raise ValueError(f"nb_layers_t must be >= 1, got {self.nb_layers_t}")
        if self.nb_heads < 1:
            raise ValueError(f"nb_heads must be >= 1, got {self.nb_heads}")
        if self.dim_embedding < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {self.dim_embedding}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if not self.embedding_feature:
            raise ValueError("embedding_feature must not be empty")
        if self.fourier_num_frequencies is not None:
            if len(self.fourier_num_frequencies) != coord_dim:
                raise ValueError(
                    f"fourier_num_frequencies length ({len(self.fourier_num_frequencies)}) "
                    f"must match coord_dim ({coord_dim}) "
                )
            if any(f < 1 for f in self.fourier_num_frequencies):
                raise ValueError(f"All fourier_num_frequencies values must be >= 1, got {self.fourier_num_frequencies}")
        if len(self.dim_max) != coord_dim:
            raise ValueError(f"dim_max length ({len(self.dim_max)}) must match coord_dim ({coord_dim})")
        if len(self.shift) != coord_dim:
            raise ValueError(f"shift length ({len(self.shift)}) must match coord_dim ({coord_dim})")
        if self.num_regression_parameters < 1:
            raise ValueError(f"num_regression_parameters must be >= 1, got {self.num_regression_parameters}")
        valid_indices = set(self.embedding_feature) | set(self.high_level_features)
        invalid = [i for i in self.cosine_processing if i not in valid_indices]
        if invalid:
            raise ValueError(
                f"cosine_processing contains indices {invalid} that are not in "
                f"embedding_feature {self.embedding_feature} or high_level_features {self.high_level_features}"
            )

    def parse_args(self):
        """
        Parse the command line arguments to fill the configuration.
        """
        parser = argparse.ArgumentParser(
            description="Configure the transformer architecture from the command line",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.add_args(parser)

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

        # Handle config file loading first (overrides all other args)
        if args.load_config:
            self.load_config(args.load_config)
            print(f"Configuration loaded from {args.load_config}")
            print("All the other arguments will be overridden by the loaded configuration.")
            return

        self.apply_args(args)

        # Handle config file saving (after all configuration is set)
        if args.save_config:
            self.save_config(args.save_config)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization."""
        return {key: value for key, value in self.__dict__.items()}

    def from_dict(self, config_dict: dict):
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)

    def save_config(self, filepath: str):
        """Save configuration to a JSON file."""
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)
        print(f"Configuration loaded from {filepath}")

    def print_config(self):
        """Print the transformer architecture configuration."""
        print("Transformer Architecture Configuration:")
        print("\nEncoder:")
        print("  Number of layers:     ", self.nb_layers_t)
        print("  Number of heads:      ", self.nb_heads)
        print("  Embedding dimension:  ", self.dim_embedding)
        print("  Feed-forward ratio:   ", self.feed_forward_ratio)
        print("  Dropout rate:         ", self.dropout)
        print(f"\nFourier Positional Encoding: {self.fourier_num_frequencies}")
        print(f"  dim_max [x,y,z,r]:     {self.dim_max}")
        print(f"  shift   [x,y,z,r]:     {self.shift}")
        print("\nFeature Selection:")
        print(f"  Embedding features:    {self.embedding_feature}")
        print(f"  High-level features:   {self.high_level_features}")
        print(f"  Cosine processing:     {self.cosine_processing}")
        print("\nRegression Head:")
        print(f"  Enabled:               {self.regression}")
        print(f"  Output parameters:     {self.num_regression_parameters}")
        print("\nConfiguration file operations available:")
        print("  --save_config <filename>     : Save current config to JSON file")
        print("  --load_config <filename>     : Load config from JSON file")
