import torch
import argparse
import json
import os

from GUNTAM.IO.PreprocessingConfig import PreprocessingConfig
from typing import Any

from GUNTAM.Seed.TransformerConfig import TransformerConfig


class SeedConfig:
    """
    Class to store the configuration variables for training and preprocessing
    """

    def __init__(self):
        """
        Initialise the configuration
        Members:
            - preprocessing_config: PreprocessingConfig: Preprocessing configuration (contains binning, I/O, selection params)
            - transformer_config: TransformerConfig: Transformer architecture configuration
            - epoch_nb: int: Number of epochs
            - num_warmup_steps: int: Number of warmup steps for the scheduler
            - val_fraction: float: Fraction of the data to use for validation
            - test_fraction: float: Fraction of the data to use for testing
            - input_tensor_path: str: Path to read/write preprocessed tensor .pt files (also in preprocessing_config)
            - dataset_name: str: Base name for dataset files (also in preprocessing_config)
            - recompute_tensor: bool: Whether to recompute tensors even if they already exist
            - model_path: str: Path to save/load the model
            - no_test: bool: Skip testing/evaluation phase after training
            - resume_training: bool: Resume training from an existing model checkpoint
            - device_acc: torch.device: The device to use (cpu/gpu)

            - loss_components: list[str]: Active loss components (e.g., 'cosine', 'MSE', 'attention')
            - loss_weights: list[float]: Corresponding weights for each loss component
        """

        # Preprocessing configuration (handles binning, I/O, selection parameters)
        self.preprocessing_config = PreprocessingConfig()
        # Model architecture configuration (transformer layers, embedding dimension, attention heads, dropout, Fourier encoding)
        self.transformer_config = TransformerConfig()

        # Training loop variables
        self.epoch_nb = 10
        self.num_warmup_steps = 5
        self.num_training_steps = 100
        self.min_lr_ratio = 0.01
        self.val_fraction = 0.1
        self.test_fraction = 0.1
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.batch_size = 5
        self.device_acc = torch.device("cpu")

        # File paths - synced with preprocessing_config in parse_args
        self.input_tensor_path = "odd_output"  # Path to read/write preprocessed tensor .pt files
        self.dataset_name = "seeding_data"  # Base name for dataset files
        self.recompute_tensor = False  # Whether to recompute tensors even if they already exist
        self.model_path = "transformer.pt"

        # Loss configuration using lists
        self.loss_components = ["attention_next"]  # List of active loss components
        self.loss_weights = []  # Default weights matching loss_components

        # Reconstruction method for inference (None = auto-select from loss config)
        # Choices: None (auto), "beam_search", "beam_search_backward"
        self.reconstruction_method = None

        # Post-hoc radial-separation greedy filter on raw beam-search chains (on by default:
        # flat efficiency improvement, no extra inference cost)
        self.radial_separation_constraint = True
        self.min_delta_rho_mm = 5.0
        self.raw_chain_length = 5

        # Boolean configurations
        self.no_test = False
        self.resume_training = False
        self.timing_enabled = False  # Timing measurements during training/testing
        self.write_seed_tensor = False  # Write seed feature tensors to disk after evaluation

    def parse_args(self):
        """
        Parse the command line argument to fill the configuration
        """
        parser = argparse.ArgumentParser(description="Configure training and preprocessing from the command line")

        # Training-specific arguments
        parser.add_argument("--epoch_nb", type=int, default=self.epoch_nb, help="Number of epoch")
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
            "--num_warmup_steps",
            type=int,
            default=self.num_warmup_steps,
            help="Number of warmup steps for the learning rate scheduler",
        )
        parser.add_argument(
            "--num_training_steps",
            type=int,
            default=self.num_training_steps,
            help="Total number of training steps for the cosine LR scheduler",
        )
        parser.add_argument(
            "--min_lr_ratio",
            type=float,
            default=self.min_lr_ratio,
            help="Minimum learning rate as a fraction of the initial learning rate (cosine scheduler)",
        )

        # Training control flags and paths
        parser.add_argument(
            "--recompute_tensor",
            action="store_true",
            help="Recompute tensors even if preprocessed files already exist",
        )
        parser.add_argument(
            "--no_test",
            action="store_true",
            help="Skip testing/evaluation phase after training",
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

        # Preprocessing arguments (delegated to preprocessing_config)
        self.preprocessing_config.add_args(parser)

        # Model architecture arguments (delegated to transformer_config)
        self.transformer_config.add_args(parser)
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
        parser.add_argument(
            "--write_seed_tensor",
            action="store_true",
            help="Write seed feature tensors and labels to disk after evaluation",
        )
        parser.add_argument(
            "--reconstruction_method",
            type=str,
            default=self.reconstruction_method,
            choices=["beam_search", "beam_search_backward"],
            help=(
                "Seed reconstruction method to use during inference. "
                "If omitted, the method is auto-selected from the active loss components. "
                "Choices: 'beam_search' (forward) or 'beam_search_backward' (backward)."
            ),
        )
        parser.add_argument(
            "--radial_separation_constraint",
            action=argparse.BooleanOptionalAction,
            default=self.radial_separation_constraint,
            help=(
                "Post-hoc 3D radial-separation greedy filter on raw beam-search chains. "
                "On by default; use --no-radial_separation_constraint to disable."
            ),
        )
        parser.add_argument(
            "--min_delta_rho_mm",
            type=float,
            default=self.min_delta_rho_mm,
            help="Minimum strict 3D radial separation (mm) required between consecutive kept hits.",
        )
        parser.add_argument(
            "--raw_chain_length",
            type=int,
            default=self.raw_chain_length,
            help="max_chain_length used for the raw beam search when radial_separation_constraint is enabled (must be >= 3).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0" if torch.cuda.is_available() else "cpu",
            help="Device to use for training (e.g., 'cpu', 'cuda:0', 'cuda:1')",
        )

        # Loss configuration arguments using lists
        parser.add_argument(
            "--loss_components",
            nargs="+",
            default=self.loss_components,
            choices=[
                "attention_next",
                "attention_back",
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

        # Parse training-specific parameters
        self.epoch_nb = args.epoch_nb
        self.val_fraction = args.val_fraction
        self.test_fraction = args.test_fraction
        self.num_warmup_steps = args.num_warmup_steps
        self.num_training_steps = args.num_training_steps
        self.min_lr_ratio = args.min_lr_ratio
        self.recompute_tensor = args.recompute_tensor
        self.no_test = args.no_test
        self.resume_training = args.resume_training
        self.model_path = args.model_path

        # Apply preprocessing parameters (also sets input_tensor_path, dataset_name)
        self.preprocessing_config.apply_args(args)
        # Sync overlapping path fields to SeedConfig level
        self.input_tensor_path = self.preprocessing_config.input_tensor_path
        self.dataset_name = self.preprocessing_config.dataset_name

        # Apply model architecture parameters
        self.transformer_config.apply_args(args)
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.timing_enabled = args.timing_enabled
        self.write_seed_tensor = args.write_seed_tensor
        self.reconstruction_method = args.reconstruction_method
        self.radial_separation_constraint = args.radial_separation_constraint
        self.min_delta_rho_mm = args.min_delta_rho_mm
        self.raw_chain_length = args.raw_chain_length
        if self.raw_chain_length < 3:
            raise ValueError(
                f"--raw_chain_length must be >= 3, got {self.raw_chain_length}. "
                "A shorter raw chain can never produce a valid 3-hit seed."
            )
        self.device_acc = torch.device(args.device)

        # Parse loss configuration lists
        self.loss_components = args.loss_components
        self.loss_weights = args.loss_weights

        # Set default weights to 1.0 if no weights provided
        if not self.loss_weights or len(self.loss_weights) == 0:
            self.loss_weights = [1.0] * len(self.loss_components)
            print("No loss weights specified, using default weight 1.0 for all " f"{len(self.loss_components)} loss components")

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

        # hit_BCE relies on the regression head output; raise early rather than silently train on zeros
        if "hit_BCE" in self.loss_config and not self.transformer_config.regression:
            raise ValueError(
                "Loss component 'hit_BCE' requires --regression to be enabled. "
                "Please add --regression to your command-line arguments."
            )

        # Handle config file saving (after all configuration is set)
        if args.save_config:
            self.save_config(args.save_config)

    def has_loss_component(self, component_name: str) -> bool:
        """Check if a loss component is active"""
        return component_name in self.loss_config

    def get_loss_weight(self, component_name: str) -> float:
        """Get the weight for a specific loss component, returns 0.0 if not active"""
        return self.loss_config.get(component_name, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization"""
        config_dict: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            # Convert torch.device to string for JSON serialization
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            # Convert PreprocessingConfig to dict
            elif isinstance(value, PreprocessingConfig):
                config_dict[key] = value.to_dict()
            # Convert TransformerConfig to dict
            elif isinstance(value, TransformerConfig):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def from_dict(self, config_dict: dict[str, Any]):
        """Load configuration from dictionary"""
        for key, value in config_dict.items():
            if key == "device_acc":
                # Convert string back to torch.device
                self.device_acc = torch.device(value)
            elif key == "preprocessing_config":
                # Handle nested preprocessing config
                if isinstance(value, dict):
                    self.preprocessing_config = PreprocessingConfig()
                    self.preprocessing_config.from_dict(value)
                else:
                    self.preprocessing_config = value
            elif key == "transformer_config":
                # Handle nested transformer config
                if isinstance(value, dict):
                    self.transformer_config = TransformerConfig()
                    self.transformer_config.from_dict(value)
                else:
                    self.transformer_config = value
            else:
                setattr(self, key, value)

        # Recreate loss_config dictionary if components and weights are present
        if hasattr(self, "loss_components") and hasattr(self, "loss_weights"):
            self.loss_config = dict(zip(self.loss_components, self.loss_weights))

    def save_sh(self, filepath: str):
        """Save configuration to a bash shell script with COMMON_ARGS format."""

        def _fmt(value) -> str:
            if isinstance(value, list):
                return " ".join(str(v) for v in value)
            return str(value)

        def _line(flag: str, value) -> str:
            if isinstance(value, bool):
                return f"    --{flag}" if value else f"    # --{flag}"
            return f"    --{flag} {_fmt(value)}"

        d = self.to_dict()

        # Flatten nested sub-configs so every key is at the same level
        flat: dict[str, Any] = {}
        for key, value in d.items():
            if key in ("preprocessing_config", "transformer_config"):
                flat.update(value)
            else:
                flat[key] = value

        # Keys that are not CLI arguments (derived or top-level duplicates of sub-config fields)
        skip = {"loss_config"}

        # CLI flag name differs from the dict key
        rename = {"device_acc": "device", "reconstruction_method": "reconstruction_method"}

        # Compute fourier_num_frequencies fallback when None
        if flat.get("fourier_num_frequencies") is None:
            flat["fourier_num_frequencies"] = [
                max(1, (flat["dim_embedding"] - len(flat["high_level_features"])) // (2 * len(flat["embedding_feature"])))
            ] * len(flat["embedding_feature"])

        # Use default loss weights when none were specified
        if not flat.get("loss_weights"):
            flat["loss_weights"] = [1.0] * len(flat["loss_components"])

        # Section header is injected when the first key of that group is encountered
        section_start = {
            next(iter(self.preprocessing_config.to_dict())): "# --- Preprocessing ---",
            next(iter(self.transformer_config.to_dict())): "# --- Transformer architecture ---",
            "epoch_nb": "# --- Training ---",
        }

        lines: list[str] = ["#!/bin/bash", "", "COMMON_ARGS=("]
        for key, value in flat.items():
            if key in skip:
                continue
            if key in section_start:
                lines.append(f"    {section_start[key]}")
            if value is None:
                lines.append(f"    # --{rename.get(key, key)}")
            else:
                lines.append(_line(rename.get(key, key), value))
        lines += [")", "", '# Usage: python -m GUNTAM.Seed.Train "${COMMON_ARGS[@]}"', ""]

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Shell configuration saved to {filepath}")

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

        self.save_sh("training_config.sh")

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
        print("=" * 60)
        print("Seed Training Configuration")
        print("=" * 60)

        # Print preprocessing configuration
        print("\n" + "=" * 60)
        print("Preprocessing Settings (from preprocessing_config):")
        print("=" * 60)
        self.preprocessing_config.print_config()

        print("\n" + "=" * 60)
        print("Training Settings:")
        print("=" * 60)
        print("Epoch number: ", self.epoch_nb)
        print("Validation fraction: ", self.val_fraction)
        print("Test fraction: ", self.test_fraction)
        print("Learning rate: ", self.learning_rate)
        print("Weight decay: ", self.weight_decay)
        print("Batch size: ", self.batch_size)
        print("Warmup steps: ", self.num_warmup_steps)
        print("Num training steps: ", self.num_training_steps)
        print("Min LR ratio: ", self.min_lr_ratio)
        print("Device: ", self.device_acc)
        print("Cuda available: ", torch.cuda.is_available())
        print("Timing enabled: ", self.timing_enabled)
        print("Reconstruction method: ", self.reconstruction_method if self.reconstruction_method else "auto (from loss config)")
        print("Radial separation constraint: ", self.radial_separation_constraint)
        print("Min delta rho (mm): ", self.min_delta_rho_mm)
        print("Raw chain length: ", self.raw_chain_length)

        print("\nFile Settings:")
        print("Model path: ", self.model_path)
        print("Input tensor path: ", self.input_tensor_path)
        print("Dataset name: ", self.dataset_name)
        print("Recompute tensor: ", self.recompute_tensor)
        print("Skip testing: ", self.no_test)
        print("Resume training: ", self.resume_training)

        # Print model architecture
        print("\n" + "=" * 60)
        print("Model Architecture:")
        print("=" * 60)
        self.transformer_config.print_config()

        # Print loss configuration
        print("\n" + "=" * 60)
        print("Loss Configuration:")
        print("=" * 60)
        print("Active loss components: ", self.loss_components)
        print("Loss weights: ", self.loss_weights)
        for component, weight in self.loss_config.items():
            print(f"  {component}: {weight}")

        print("\n" + "=" * 60)
        print("Configuration file operations available:")
        print("  --save_config <filename>     : Save current config to JSON file")
        print("  --load_config <filename>     : Load config from JSON file")
        print("=" * 60)


if __name__ == "__main__":
    config = SeedConfig()
    config.parse_args()
    config.print_config()
    config.save_sh("training_config.sh")
