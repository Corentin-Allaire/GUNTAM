from numpy.char import center
import torch
import torch.nn as nn
from torch import Tensor, normal

from GUNTAM.Seed.ClassifierConfig import ClassifierConfig
from GUNTAM.Transformer.OutputClassifier import OutputClassifier
from GUNTAM.Transformer.Transformer import load_state_dict_flex


class SeedClassifier(nn.Module):
    """
    MLP-based classifier used to accept/reject candidate seeds.

    This module wraps an `OutputClassifier` MLP, keeping track of the
    configuration used to build it so that checkpoints can be saved,
    loaded, and exported to ONNX.

    Attributes:
        - classifier (OutputClassifier): MLP classifying candidate seeds as true/false.
        - cfg (ClassifierConfig): Full architecture/training configuration.
        - device_acc (torch.device): Device on which the model's parameters are allocated.

    Args:
        - classifier_config (ClassifierConfig): Architecture configuration object.
        - device_acc (torch.device, optional): Device to run the model on. Defaults to cpu.
        - dtype (torch.dtype, optional): Floating-point precision for the model's weights.
    """

    def __init__(
        self,
        classifier_config: ClassifierConfig = ClassifierConfig(),
        device_acc: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(SeedClassifier, self).__init__()

        self.cfg = classifier_config
        self.device_acc = device_acc
        self.dtype = dtype
        self.threshold: float = classifier_config.threshold
        self._setup_modules()
        self.to(dtype)

    def _setup_modules(
        self,
    ) -> None:
        """
        Initialize or rebuild all submodules with the provided hyperparameters.
        """

        self.classifier = OutputClassifier(
            input_shape=self.cfg.input_shape,
            hidden_layers=self.cfg.hidden_layers,
            output_shape=self.cfg.output_shape,
            p=self.cfg.p,
            activation=self.cfg.activation,
            threshold=self.cfg.threshold,
            device=self.device_acc,
        )

    def forward(self, x: Tensor, nb_hits_features: int = 7) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the classifier network.
        Args:
            - x (Tensor): Input features for a batch of seeds.
        Returns:
            - out (Tensor): Class logits (use `seed_classification` to get probabilities/keep mask).
        """
        seed = self.prepare_seed_features(x, nb_hits_features)
        proba = self.classifier(seed)
        scores = proba[:, 1]
        keep_mask = scores >= self.threshold
        return keep_mask, scores

    def prepare_seed_features(self, data: Tensor, nb_hits_features: int = 7) -> Tensor:
        """
        We add nb_hits_features new features for each seed. These features are the parameters of the circle made of the three points in each seed
        Args:
            data: Tensor of shape [N, 21] containing the features of the N seeds.
            Each seed is represented by 3 points, each point having nb_hits_features features,
            the first 3 need to be the coordinates of the point (x, y, z).

        Returns:
            Tensor of shape [N, 10] containing the new features for each seed.
            The new features are the parameters of the circle made of the three points in each seed:
            - center_x, center_y, center_z: coordinates of the center of the circle
            - radius: radius of the circle
            - normal_x, normal_y, normal_z: components of the normal vector to the plane of the circle
        """

        P1 = data[:, 0:3]  # positional features of the first point of the seed : x1, y1, z1
        P2 = data[:, (nb_hits_features) : (nb_hits_features + 3)]  # x2, y2, z2
        P3 = data[:, (2 * nb_hits_features) : (2 * nb_hits_features + 3)]  # x3, y3, z3
        # Computing the normal vector to the plane formed by the 3 points : P1, P2, P3
        a = P2 - P1
        b = P3 - P1
        normal = torch.linalg.cross(a, b, dim=1)

        # Handling degenerate cases:
        norm_val = torch.linalg.norm(normal, dim=1, keepdim=True)
        degenerate = norm_val[:, 0] < 1e-10
        norm_val = torch.where(norm_val < 1e-10, torch.ones_like(norm_val), norm_val)
        normal = normal / norm_val

        # Building a linear system A @ center = b_vec to find the center:
        row1 = 2 * (P2 - P1)
        row2 = 2 * (P3 - P1)
        row3 = normal

        b_vec = torch.stack(
            [
                (P2**2).sum(dim=1) - (P1**2).sum(dim=1),
                (P3**2).sum(dim=1) - (P1**2).sum(dim=1),
                (normal * P1).sum(dim=1),
            ],
            dim=1,
        )

        R0, R1, R2 = row1, row2, row3

        cross_R1_R2 = torch.linalg.cross(R1, R2, dim=1)  # (N, 3)
        cross_R2_R0 = torch.linalg.cross(R2, R0, dim=1)  # (N, 3)
        cross_R0_R1 = torch.linalg.cross(R0, R1, dim=1)  # (N, 3)

        det = (R0 * cross_R1_R2).sum(dim=1, keepdim=True)  # (N, 1)
        det_safe = torch.where(det.abs() < 1e-12, torch.ones_like(det), det)

        b0 = b_vec[:, 0:1]
        b1 = b_vec[:, 1:2]
        b2 = b_vec[:, 2:3]

        center = (b0 * cross_R1_R2 + b1 * cross_R2_R0 + b2 * cross_R0_R1) / det_safe  # (N, 3)

        # Computing the radius:
        radius = torch.linalg.norm(center - P1, dim=1, keepdim=True)

        # Cleaning up degenerate cases:
        degenerate_col = degenerate.unsqueeze(-1)
        center = torch.where(degenerate_col, torch.zeros_like(center), center)
        radius = torch.where(degenerate_col, torch.zeros_like(radius), radius)
        normal = torch.where(degenerate_col, torch.zeros_like(normal), normal)

        seeds = torch.cat([data, center, radius, normal], dim=1)

        return seeds

    def print_model_info(self) -> None:
        """
        Print model information including number of layers and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("SeedClassifier Model Info:")
        print(f"  - Hidden layers: {self.cfg.hidden_layers}")
        print(f"  - Total parameters: {total_params}")
        print(f"  - Trainable parameters: {trainable_params}")

    def save(
        self,
        epoch: int,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """
        Save the model state to a file.
        Args:
            - epoch (int): Last completed epoch, used to resume training.
            - path (str): File path to save the checkpoint.
            - optimizer (torch.optim.Optimizer | None): Optimizer to save the state of, if any.
            - scheduler (torch.optim.lr_scheduler._LRScheduler | None): Scheduler to save the state of, if any.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                # Save full classifier config so the architecture can be rebuilt on load
                "classifier_config": self.cfg.to_dict(),
                "dtype": str(self.dtype).replace("torch.", ""),
            },
            path,
        )

    def load(
        self,
        path: str,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> int:
        """
        Load the model state from a file.
        Args:
            - path (str): File path to load the checkpoint from.
            - device (torch.device): Device to load the model onto.
            - optimizer (torch.optim.Optimizer | None): Optimizer to restore the state of, if any.
            - scheduler (torch.optim.lr_scheduler._LRScheduler | None): Scheduler to restore the state of, if any.
        Returns:
            - start_epoch (int): Epoch to resume training from.
        """
        start_epoch = 0
        try:
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                print("Checkpoint missing 'model_state_dict'; starting from scratch.")
            else:
                # Rebuild architecture to match the checkpoint if hidden layers/activation/etc differ
                self._rebuild_from_checkpoint_config(checkpoint.get("classifier_config"), device)
                load_state_dict_flex(self, state_dict, desc="classifier resume")
                self.to(device)
                if "dtype" in checkpoint:
                    saved_dtype = getattr(torch, checkpoint["dtype"], None)
                    if saved_dtype is not None:
                        self.dtype = saved_dtype
                        self.to(saved_dtype)
                if (
                    "optimizer_state_dict" in checkpoint
                    and checkpoint["optimizer_state_dict"] is not None
                    and optimizer is not None
                ):
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if (
                    "scheduler_state_dict" in checkpoint
                    and checkpoint["scheduler_state_dict"] is not None
                    and scheduler is not None
                ):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                    print(f"Resumed classifier training from epoch {start_epoch}")
        except FileNotFoundError:
            print(f"Error: No checkpoint found at {path}. Starting training from scratch.")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}); starting from scratch.")
        return start_epoch

    def _rebuild_from_checkpoint_config(self, model_cfg: dict | None, device: torch.device) -> None:
        """
        Recreate architecture modules to match a checkpoint config.
        Allows loading checkpoints with different architecture parameters.
        When a field differs between the CLI config and the checkpoint config, the CLI
        value takes precedence if it was explicitly changed from the default; otherwise
        the checkpoint value is used.
        Args:
            - model_cfg (dict | None): Model configuration from checkpoint.
            - device (torch.device): Device to allocate rebuilt modules on.
        Returns:
            - None
        """

        # Config fields that define the network architecture; used to decide whether a checkpoint requires a rebuild
        _ARCHITECTURE_FIELDS = ("input_shape", "hidden_layers", "output_shape", "p", "activation", "threshold")

        if not model_cfg:
            return

        default_cfg = ClassifierConfig().to_dict()
        cli_cfg = self.cfg.to_dict()

        # Start from checkpoint config, then let non-default CLI values win (architecture fields only)
        new_cfg = ClassifierConfig()
        new_cfg.from_dict(cli_cfg)  # start from current (CLI) config
        new_cfg.from_dict({k: v for k, v in model_cfg.items() if k in _ARCHITECTURE_FIELDS})  # overlay with checkpoint

        for key in _ARCHITECTURE_FIELDS:
            if key not in model_cfg:
                continue
            ckpt_val = model_cfg[key]
            cli_val = cli_cfg.get(key)
            default_val = default_cfg.get(key)
            if cli_val != ckpt_val and cli_val != default_val:
                print(
                    f"Warning: '{key}' mismatch — checkpoint={ckpt_val}, CLI={cli_val} (not default={default_val}). "
                    f"Using CLI value."
                )
                setattr(new_cfg, key, cli_val)

        if all(getattr(new_cfg, key) == getattr(self.cfg, key) for key in _ARCHITECTURE_FIELDS):
            return

        print("Rebuilding SeedClassifier modules to match checkpoint configuration...")
        self.cfg = new_cfg
        self.device_acc = device
        self._setup_modules()

    def export_onnx(
        self,
        path: str,
        example_input: Tensor | None = None,
    ) -> None:
        """
        Export the model to an ONNX file.
        Args:
            - path (str): File path to save the ONNX model (.onnx).
            - example_input (Tensor | None): Representative seed features tensor [batch, input_shape].
              If None, a zero tensor is built from the config as fallback.
        """
        if example_input is None:
            example_input = torch.zeros(1, self.cfg.input_shape, dtype=torch.float32, device="cpu")

        example_input = example_input.float().cpu()

        was_training = self.training
        original_device = next(self.parameters()).device
        original_dtype = self.dtype

        self.eval()
        self.to(torch.float32)
        self.to("cpu")

        try:
            torch.onnx.export(
                self,
                (example_input,),
                path,
                input_names=["seed_features"],
                output_names=["keep_mask", "scores"],
                dynamic_axes={
                    "seed_features": {0: "batch_size"},
                    "keep_mask": {0: "batch_size"},
                    "scores": {0: "batch_size"},
                },
                opset_version=18,
            )
            print(f"Model exported to ONNX at {path}")
        finally:
            self.to(original_device)
            self.to(original_dtype)
            if was_training:
                self.train()
