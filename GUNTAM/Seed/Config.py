import torch
import argparse
import json
import os


class config:
    """
    Class to store the configuration variables for training and preprocessing
    """

    def __init__(self):
        """
        Initialise the configuration
        Members:
            - epoch_nb: int: Number of epochs
            - max_hit_input: int: Maximum number of hits in the encoder input
            - vertex_cuts: list[int, int, int]: Cuts on the vertex position to keep the primary vertex
            - bin_width: float: Width of the bin in phi
            - eta_bin_width: float: Width of the bin in eta
            - eta_range: list[float, float]: Fixed eta range [min, max] for binning
            - num_warmup_steps: int: Number of warmup steps for the scheduler
            - val_fraction: float: Fraction of the data to use for validation
            - test_fraction: float: Fraction of the data to use for testing
            - input_path: str: Path to the input data
            - input_tensor_path: str: Path to read/write preprocessed tensor .pt files
            - model_path: str: Path to save/load the model
            - test_only: bool: Only perform testing (requires a saved model)
            - resume_training: bool: Resume training from an existing model checkpoint
            - device_acc: torch.device: The device to use (cpu/gpu)
            - max_events: int: Maximum number of events to process, -1 for all events

            - nb_layers_t: int: Number of transformer layers
            - dim_embedding: int: Embedding dimension
            - nb_heads: int: Number of attention heads
            - dropout: float: Dropout rate
            - fourier_num_frequencies: int | list[int] | None: Number of Fourier frequency bands for embeddings.
                Can be int (same for all 3 dimensions) or list of 3 ints (one per dimension x,y,z).

            - loss_components: list[str]: Active loss components (e.g., 'cosine', 'MSE', 'attention')
            - loss_weights: list[float]: Corresponding weights for each loss component

            - orphan_hit_fraction: float: Fraction of orphan hits to keep per bin (0.0 to 1.0)
        """

        # Preprocessing / Binning variables
        self.max_hit_input = -1
        self.vertex_cuts = [10, 10, 200]
        self.bin_width = 0.05
        self.eta_bin_width = 0.5  # Width of the bin in eta
        self.eta_range = [
            -3.0,
            3.0,
        ]
        self.max_events = -1
        self.orphan_hit_fraction = 0.0  # Fraction of orphan hits to keep per bin (0.0 to 1.0)

        # Training loop variables
        self.epoch_nb = 10
        self.num_warmup_steps = 5
        self.val_fraction = 0.2
        self.test_fraction = 0.02
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.batch_size = 1
        self.device_acc = torch.device("cpu")

        # File paths
        self.input_path = "odd_output"  # Read/write path for input data
        self.input_tensor_path = "."  # Read/write path for input tensor .pt files
        self.model_path = "transformer.pt"

        # Model architecture parameters
        self.nb_layers_t = 6
        self.dim_embedding = 128
        self.nb_heads = 4
        self.dropout = 0.1
        # Fourier embedding specific (optional): number of frequency bands; if None, derived from dim_embedding
        self.fourier_num_frequencies = None

        # Loss configuration using lists
        self.loss_components = ["cosine", "MSE", "attention"]
        self.loss_weights = []  # Default weights matching loss_components

        # Boolean configurations
        self.test_only = False
        self.resume_training = False
        self.timing_enabled = False  # Timing measurements during training/testing

    def parse_args(self):
        """
        Parse the command line argument to fill the configuration
        """
        parser = argparse.ArgumentParser(description="Configure training and preprocessing from the command line")
        parser.add_argument("--epoch_nb", type=int, default=self.epoch_nb, help="Number of epoch")
        parser.add_argument(
            "--max_hit_input",
            type=int,
            default=self.max_hit_input,
            help="Maximum number of hit input",
        )
        parser.add_argument("--vertex_cuts", type=list, default=self.vertex_cuts, help="Vertex cuts")
        parser.add_argument(
            "--bin_width",
            type=float,
            default=self.bin_width,
            help="Width of the bin in phi",
        )
        parser.add_argument(
            "--eta_bin_width",
            type=float,
            default=self.eta_bin_width,
            help="Width of the bin in eta",
        )
        parser.add_argument(
            "--eta_range",
            nargs=2,
            type=float,
            default=self.eta_range,
            help="Fixed eta range [min, max] for binning (hits/particles outside are filtered)",
        )

        parser.add_argument(
            "--val_fraction",
            type=float,
            default=self.val_fraction,
            help="Fraction of the data to use for validation",
        )
        parser.add_argument(
            "--test_fraction",
            type=float,
            default=self.test_fraction,
            help="Fraction of the data to use for testing",
        )
        parser.add_argument(
            "--input_path",
            type=str,
            default=self.input_path,
            help="Path to the input data",
        )
        parser.add_argument(
            "--input_tensor_path",
            type=str,
            default=self.input_tensor_path,
            help="Directory to read/write preprocessed tensor .pt files",
        )
        parser.add_argument(
            "--test_only",
            action="store_true",
            help="Only perform testing on the model (requires a saved model)",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=self.model_path,
            help="Path to save/load the model",
        )
        parser.add_argument(
            "--resume_training",
            action="store_true",
            help="Resume training from an existing model checkpoint",
        )
        parser.add_argument(
            "--num_warmup_steps",
            type=int,
            default=self.num_warmup_steps,
            help="Number of warmup steps for the learning rate scheduler",
        )
        parser.add_argument(
            "--max_events",
            type=int,
            default=self.max_events,
            help="Maximum number of events to process (-1 for all events)",
        )
        # Model architecture arguments
        parser.add_argument(
            "--nb_layers_t",
            type=int,
            default=self.nb_layers_t,
            help="Number of transformer layers",
        )
        parser.add_argument(
            "--dim_embedding",
            type=int,
            default=self.dim_embedding,
            help="Embedding dimension",
        )
        parser.add_argument(
            "--nb_heads",
            type=int,
            default=self.nb_heads,
            help="Number of attention heads",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=self.dropout,
            help="Dropout rate",
        )
        parser.add_argument(
            "--fourier_num_frequencies",
            nargs="*",
            type=int,
            default=self.fourier_num_frequencies,
            help=(
                "Number of Fourier frequency bands (int or list of 3 ints for x,y,z); "
                "if omitted, it's derived from dim_embedding"
            ),
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=self.learning_rate,
            help="Learning rate for the optimizer",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=self.weight_decay,
            help="Weight decay (L2 regularization) for the optimizer",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=self.batch_size,
            help="Number of bins to accumulate gradients before updating (gradient accumulation)",
        )
        parser.add_argument(
            "--timing_enabled",
            action="store_true",
            help="Enable detailed timing measurements during training/testing",
        )

        # Loss configuration arguments using lists
        parser.add_argument(
            "--loss_components",
            nargs="+",
            default=self.loss_components,
            choices=[
                "cosine",
                "MSE",
                "L1",
                "attention",
                "full_attention",
                "topk_attention",
                "attention_next",
                "hit_BCE",
            ],
            help="List of loss components to use",
        )
        parser.add_argument(
            "--loss_weights",
            nargs="+",
            type=float,
            default=self.loss_weights,
            help="List of weights for each loss component (must match the number of loss_components)",
        )

        # Simple case configuration arguments
        parser.add_argument(
            "--orphan_hit_fraction",
            type=float,
            default=self.orphan_hit_fraction,
            help="Fraction of orphan hits (hits without particles) to keep per bin (0.0 to 1.0)",
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

        self.epoch_nb = args.epoch_nb
        self.max_hit_input = args.max_hit_input
        self.vertex_cuts = args.vertex_cuts
        self.bin_width = args.bin_width
        self.eta_bin_width = args.eta_bin_width
        self.eta_range = args.eta_range
        self.num_warmup_steps = args.num_warmup_steps
        self.max_events = args.max_events
        self.val_fraction = args.val_fraction
        self.test_fraction = args.test_fraction
        self.input_path = args.input_path
        self.input_tensor_path = args.input_tensor_path
        self.test_only = args.test_only
        self.resume_training = args.resume_training
        self.model_path = args.model_path

        # Parse model architecture parameters
        self.nb_layers_t = args.nb_layers_t
        self.dim_embedding = args.dim_embedding
        self.nb_heads = args.nb_heads
        self.dropout = args.dropout
        # Handle fourier_num_frequencies as int or list
        if args.fourier_num_frequencies:
            if len(args.fourier_num_frequencies) == 1:
                self.fourier_num_frequencies = args.fourier_num_frequencies[0]
            elif len(args.fourier_num_frequencies) == 3:
                self.fourier_num_frequencies = args.fourier_num_frequencies
            else:
                raise ValueError(
                    f"fourier_num_frequencies must be either 1 value (for all dims) or 3 values (x,y,z), "
                    f"got {len(args.fourier_num_frequencies)} values"
                )
        else:
            self.fourier_num_frequencies = None
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.timing_enabled = args.timing_enabled

        # Parse loss configuration lists
        self.loss_components = args.loss_components
        self.loss_weights = args.loss_weights

        # Check that the loss_components does not contain but MSE and L1
        if "MSE" in self.loss_components and "L1" in self.loss_components:
            raise ValueError("Cannot use both MSE and L1 loss components at the same time. Please choose one.")

        # Set default weights to 1.0 if no weights provided
        if not self.loss_weights or len(self.loss_weights) == 0:
            self.loss_weights = [1.0] * len(self.loss_components)
            print(
                "No loss weights specified, using default weight 1.0 for all "
                f"{len(self.loss_components)} loss components"
            )

        self.orphan_hit_fraction = args.orphan_hit_fraction

        # Validate orphan_hit_fraction range
        if self.orphan_hit_fraction < 0.0 or self.orphan_hit_fraction > 1.0:
            raise ValueError(f"orphan_hit_fraction must be between 0.0 and 1.0, got {self.orphan_hit_fraction}")

        # Validate that loss_components and loss_weights have the same length
        if len(self.loss_components) != len(self.loss_weights):
            raise ValueError(
                (
                    f"Number of loss components ({len(self.loss_components)}) must match number of weights "
                    f"({len(self.loss_weights)})"
                )
            )

        # Create a dictionary mapping loss components to weights for easy lookup
        self.loss_config = dict(zip(self.loss_components, self.loss_weights))

        self.device_acc = torch.device("cuda:0")

        # Handle config file saving (after all configuration is set)
        if args.save_config:
            self.save_config(args.save_config)

    def has_loss_component(self, component_name: str) -> bool:
        """Check if a loss component is active"""
        return component_name in self.loss_config

    def get_loss_weight(self, component_name: str) -> float:
        """Get the weight for a specific loss component, returns 0.0 if not active"""
        return self.loss_config.get(component_name, 0.0)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            # Convert torch.device to string for JSON serialization
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def from_dict(self, config_dict: dict):
        """Load configuration from dictionary"""
        for key, value in config_dict.items():
            if key == "device_acc":
                # Convert string back to torch.device
                self.device_acc = torch.device(value)
            else:
                setattr(self, key, value)

        # Recreate loss_config dictionary if components and weights are present
        if hasattr(self, "loss_components") and hasattr(self, "loss_weights"):
            self.loss_config = dict(zip(self.loss_components, self.loss_weights))

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
        print("Configuration:")
        print("Epoch number: ", self.epoch_nb)
        print("Max hit input: ", self.max_hit_input)
        print("Vertex cuts: ", self.vertex_cuts)
        print("Bin width (phi): ", self.bin_width)
        print("Bin width (eta): ", self.eta_bin_width)
        print("Eta range: ", self.eta_range)
        print("Validation fraction: ", self.val_fraction)
        print("Test fraction: ", self.test_fraction)
        print("Input path: ", self.input_path)
        print("Input tensor path: ", self.input_tensor_path)
        print("Test only mode: ", self.test_only)
        print("Model path: ", self.model_path)
        print("Resume training: ", self.resume_training)
        print("Device: ", self.device_acc)
        print("Cuda available: ", torch.cuda.is_available())
        print("Max events: ", self.max_events)
        print("Timing enabled: ", self.timing_enabled)

        # Print model architecture
        print("\nModel Architecture:")
        print("Transformer layers: ", self.nb_layers_t)
        print("Embedding dimension: ", self.dim_embedding)
        print("Attention heads: ", self.nb_heads)
        print("Dropout rate: ", self.dropout)
        print("Learning rate: ", self.learning_rate)
        print("Weight decay: ", self.weight_decay)
        print("Fourier num_frequencies: ", self.fourier_num_frequencies)

        # Print loss configuration
        print("\nLoss Configuration:")
        print("Active loss components: ", self.loss_components)
        print("Loss weights: ", self.loss_weights)
        for component, weight in self.loss_config.items():
            print(f"  {component}: {weight}")

        # Print simple case configuration
        print("\nSimple Case Configuration:")
        print("Orphan hit fraction: ", self.orphan_hit_fraction)

        print("\nConfiguration file operations available:")
        print("  --save_config <filename>     : Save current config to JSON file")
        print("  --load_config <filename>     : Load config from JSON file")
