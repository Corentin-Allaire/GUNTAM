import math
from datetime import datetime
import torch
from torch.optim.lr_scheduler import LRScheduler


def ts_print(*args, **kwargs) -> None:
    """Timestamped print for logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ", *args, **kwargs)


def log_gradients(model, writer=None, step=None):
    """
    Log gradient norms and means to TensorBoard writer if provided.
    Args:
        model: The model containing parameters with gradients.
        writer: TensorBoard SummaryWriter instance (optional).
        step: Current training step for logging (optional).
    Returns:
        A dictionary mapping parameter names to their gradient norms.
    """
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.abs().mean().item()
            grad_stats[name] = grad_norm

            if writer and step is not None:
                writer.add_scalar(f"GradNorm/{name}", grad_norm, step)
                writer.add_scalar(f"GradMean/{name}", grad_mean, step)

    return grad_stats


def sync_device(dev: torch.device):
    """
    Synchronize CUDA if cfg.device_acc is CUDA; CPU is a no-op.

    Args:
        dev: The device to synchronize.
    """
    if isinstance(dev, torch.device):
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=dev)


class CosineScheduleWithMinLR(LRScheduler):
    """
    Custom scheduler with warmup, cosine annealing, and minimum learning rate

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (e.g., 0.01 for 1%).
    """

    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.last_epoch = 0

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr_ratio(self, current_step):
        """
        Calculate the learning rate ratio for the current step

        Args:
            current_step: The current training step.
        Returns:
            The learning rate ratio to apply.
        """
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
        if current_step >= self.num_training_steps:
            # After training steps, return minimum learning rate ratio
            return self.min_lr_ratio

        # Cosine annealing with minimum
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Ensure learning rate doesn't go below min_lr_ratio
        return max(self.min_lr_ratio, cosine_factor)

    def get_last_lr(self):
        """Get the last learning rates"""
        return self.optimizer.param_groups[0]["lr"]

    def get_lr(self):
        """Get the current learning rates"""
        ratio = self.get_lr_ratio(self.last_epoch)
        return [base_lr * ratio for base_lr in self.base_lrs]

    def step(self):
        """Step the scheduler"""
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def is_in_warmup(self):
        """Check if the scheduler is currently in the warmup phase"""
        return self.last_epoch < self.num_warmup_steps

    def get_ratio(self):
        """Get the current learning rate ratio"""
        return self.get_lr_ratio(self.last_epoch)

    def state_dict(self):
        """Return the state of the scheduler"""
        return {
            "last_epoch": self.last_epoch,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        """Load the state of the scheduler"""
        self.last_epoch = state_dict["last_epoch"]
        self.num_warmup_steps = state_dict["num_warmup_steps"]
        self.num_training_steps = state_dict["num_training_steps"]
        self.min_lr_ratio = state_dict["min_lr_ratio"]
        self.base_lrs = state_dict["base_lrs"]


def create_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """
    Create a cosine schedule with warmup and minimum learning rate.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (e.g., 0.01 for 1%)
    """
    return CosineScheduleWithMinLR(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio)
