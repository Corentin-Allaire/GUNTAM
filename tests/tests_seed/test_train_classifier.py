import torch

from GUNTAM.Seed.ClassifierConfig import ClassifierConfig
from GUNTAM.Seed.Train_classifier import (
    SeedDataset,
    balance_dataset,
)
from torch.utils.data import DataLoader as TorchDataLoader


def make_config(model_path: str) -> ClassifierConfig:
    """Build a minimal ClassifierConfig for fast tests."""
    cfg = ClassifierConfig()
    cfg.input_shape = 8
    cfg.hidden_layers = [16, 8]
    cfg.n_epochs = 1
    cfg.icing_epochs = 1
    cfg.lr = 1e-3
    cfg.icing_lr = 1e-3
    cfg.batch_size = 4
    cfg.model_path = model_path
    cfg.device_acc = torch.device("cpu")
    return cfg


def make_dataloader(cfg: ClassifierConfig, n_samples: int = 16) -> TorchDataLoader:
    X = torch.randn(n_samples, cfg.input_shape)
    y = torch.randint(0, 2, (n_samples,))
    dataset = SeedDataset(X, y)
    return TorchDataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)


class TestBalanceDataset:
    def test_balances_class_counts(self):
        features = torch.randn(10, 4)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        balanced_features, balanced_labels = balance_dataset(features, labels)

        values, counts = torch.unique(balanced_labels, return_counts=True)
        assert set(values.tolist()) == {0, 1}
        assert counts[0].item() == counts[1].item() == 3
        assert balanced_features.shape[0] == balanced_labels.shape[0]
