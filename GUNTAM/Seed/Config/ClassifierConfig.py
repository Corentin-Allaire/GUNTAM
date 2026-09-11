import argparse
import json
import os

import torch


class ClassifierConfig:
    """
    Class to store the configuration variables for the seed classifier (OutputClassifier) architecture and training.
    """

    def __init__(self):
        """
        Initialise the classifier configuration.
        Members:
            - input_shape: int: Number of input features (per seed)
            - hidden_layers: list[int]: Number of neurons in each hidden layer, in order
            - p: float: Dropout coefficient
            - n_epochs: int: Number of training epochs
            - icing_epochs: int: Number of epochs for the output-layer-only fine-tuning phase
            - lr: float: Learning rate for the optimizer
            - icing_lr: float: Learning rate for the output-layer-only fine-tuning phase
            - batch_size: int: Batch size used by the training dataloader
            - model_path: str: Path to save/load the model checkpoint
            - resume_training: bool: Resume training from an existing model checkpoint
            - test_fraction: float: Fraction of the data to hold out for final testing (ROC/AUC)
            - val_fraction: float: Fraction of the remaining (non-test) data held out for validation,
              monitored during training
            - device_acc: torch.device: The device to use (cpu/gpu)
        """

        self.input_shape = 28
        self.hidden_layers = [256, 128, 64]
        self.p = 0.0
        self.threshold = 0.5
        self.activation = "ReLU"
        self.n_epochs = 10
        self.output_shape = 2
        self.icing_epochs = 5
        self.lr = 1e-3
        self.icing_lr = 1e-3
        self.batch_size = 100
        self.model_path = "classifier.pt"
        self.input_tensor_path = "seed_features.pt"
        self.resume_training = False
        self.test_fraction = 0.1
        self.val_fraction = 0.1
        self.device_acc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Add classifier architecture/training arguments to an existing ArgumentParser.
        """
        parser.add_argument("--input_shape", type=int, default=self.input_shape, help="Number of input features per seed")
        parser.add_argument(
            "--hidden_layers",
            nargs="+",
            type=int,
            default=self.hidden_layers,
            help="Number of neurons in each hidden layer, in order (e.g. 512 256 128 64)",
        )
        parser.add_argument("--p", type=float, default=self.p, help="Dropout coefficient")
        parser.add_argument(
            "--activation",
            type=str,
            default=self.activation,
            help="activation function for the hidden layers (e.g., ReLU, Sigmoid, Tanh)",
        )
        parser.add_argument("--n_epochs", type=int, default=self.n_epochs, help="Number of training epochs")
        parser.add_argument(
            "--icing_epochs",
            type=int,
            default=self.icing_epochs,
            help="Number of epochs for the output-layer-only fine-tuning phase",
        )
        parser.add_argument("--lr", type=float, default=self.lr, help="Learning rate for the optimizer")
        parser.add_argument(
            "--icing_lr",
            type=float,
            default=self.icing_lr,
            help="Learning rate for the output-layer-only fine-tuning phase",
        )
        parser.add_argument("--batch_size", type=int, default=self.batch_size, help="Batch size for the training dataloader")
        parser.add_argument(
            "--classifier_model_path",
            type=str,
            default=self.model_path,
            help="Path to save/load the classifier model checkpoint",
        )
        parser.add_argument(
            "--input_tensor_path",
            type=str,
            default=self.input_tensor_path,
            help="Path to the input tensor file (seed features)",
        )
        parser.add_argument(
            "--classifier_resume_training",
            action="store_true",
            help="Resume classifier training from an existing model checkpoint",
        )
        parser.add_argument(
            "--test_fraction",
            type=float,
            default=self.test_fraction,
            help="Fraction of the data to hold out for final testing (ROC/AUC)",
        )
        parser.add_argument(
            "--val_fraction",
            type=float,
            default=self.val_fraction,
            help="Fraction of the remaining (non-test) data held out for validation, monitored during training",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=str(self.device_acc),
            help="Device to use for training (e.g., 'cpu', 'cuda:0', 'cuda:1')",
        )

    def apply_args(self, args: argparse.Namespace) -> None:
        """
        Apply the values from a parsed Namespace to the configuration.
        """
        self.input_shape = args.input_shape
        self.hidden_layers = args.hidden_layers
        self.p = args.p
        self.activation = "ReLU"
        self.n_epochs = args.n_epochs
        self.icing_epochs = args.icing_epochs
        self.lr = args.lr
        self.icing_lr = args.icing_lr
        self.batch_size = args.batch_size
        self.model_path = args.classifier_model_path
        self.input_tensor_path = args.input_tensor_path
        self.resume_training = args.classifier_resume_training
        self.test_fraction = args.test_fraction
        self.val_fraction = args.val_fraction
        self.device_acc = torch.device(args.device)

        if self.input_shape < 1:
            raise ValueError(f"input_shape must be >= 1, got {self.input_shape}")
        if not self.hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        if not (0.0 <= self.p < 1.0):
            raise ValueError(f"p must be in [0.0, 1.0), got {self.p}")
        if not (0.0 <= self.test_fraction < 1.0):
            raise ValueError(f"test_fraction must be in [0.0, 1.0), got {self.test_fraction}")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0.0, 1.0), got {self.val_fraction}")

    def parse_args(self):
        """
        Parse the command line arguments to fill the configuration.
        """
        parser = argparse.ArgumentParser(
            description="Configure the seed classifier from the command line",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.add_args(parser)

        parser.add_argument("--save_config", type=str, help="Save current configuration to a JSON file")
        parser.add_argument("--load_config", type=str, help="Load configuration from a JSON file")

        args = parser.parse_args()

        if args.load_config:
            self.load_config(args.load_config)
            print(f"Configuration loaded from {args.load_config}")
            print("All the other arguments will be overridden by the loaded configuration.")
            return

        self.apply_args(args)

        if args.save_config:
            self.save_config(args.save_config)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for JSON serialization."""
        result = dict(self.__dict__)
        result["device_acc"] = str(self.device_acc)
        return result

    def from_dict(self, config_dict: dict):
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            if key == "device_acc":
                value = torch.device(value)
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
        """Print the classifier configuration."""
        print("Classifier Configuration:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")
