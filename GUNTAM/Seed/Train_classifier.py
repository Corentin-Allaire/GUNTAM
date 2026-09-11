import os

import torch
from torch.utils.tensorboard import SummaryWriter
from GUNTAM.Seed.Config.ClassifierConfig import ClassifierConfig
from GUNTAM.Seed.SeedClassifier import SeedClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader


class SeedDataset(Dataset):
    """
    Transforms our data into a proper dataset.

    Args:
        Seeds
        Features

    Returns:
        A dataset containing the seeds and features.

    """

    def __init__(self, X, y):
        # The init class is used to define your dataset attributes.
        # X contains our seeds, we convert them to float PyTorch tensor
        self.X = X
        # The labels are integeres (0 or 1), we convert them to long (integer) PyTorch tensors.
        self.y = y

    def __len__(self):
        length = len(self.y)
        return length

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def balance_dataset(features, labels):
    """
    We balance the dataset in order to have the same number of fake and true seeds in the dataset
    Args:
    features: features of each seed
    labels: labels of each seeds

    Returns:
    A balanced dataset

    """

    # Separating the true and fake seeds (avoid torch.unique: it may omit a class entirely)
    idx_fake = (labels == 0).nonzero(as_tuple=True)[0]
    idx_true = (labels == 1).nonzero(as_tuple=True)[0]

    print("Number of fake seeds (pre-balancing):", len(idx_fake))
    print("Number of true seeds (pre-balancing):", len(idx_true))

    n_min = min(len(idx_true), len(idx_fake))

    idx_balanced = torch.cat([idx_true[:n_min], idx_fake[:n_min]])

    features = features[idx_balanced]
    labels = labels[idx_balanced]

    # Shuffle
    perm = torch.randperm(len(labels))
    features = features[perm]
    labels = labels[perm]

    return features, labels


def split_train_val(features, labels, val_fraction: float = 0.1):
    """
    Randomly splits features/labels into training and validation subsets.

    Args:
        features: features of each seed
        labels: labels of each seed
        val_fraction: fraction of the data kept for validation

    Returns:
        (train_features, train_labels, val_features, val_labels)

    """
    n = len(labels)
    perm = torch.randperm(n)
    n_val = int(val_fraction * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


def train_loop_Classifier(
    trainloader,
    model: SeedClassifier,
    n_epochs: int,
    optimizer,
    criterion,
    device,
    writer: SummaryWriter | None = None,
    start_epoch: int = 0,
    val_dataloader=None,
):
    """
    Returns the trained model.
    """
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):

        print("epoch:", epoch)
        epoch_losses = []

        for i, (X, y) in enumerate(trainloader):
            inputs = X.to(device)
            labels = y.to(device)

            # Gradients to zero
            optimizer.zero_grad()
            # Compute prediction and loss
            pred = model.classifier(inputs)
            loss = criterion(pred, labels)

            # Update gradients and Update model's weights:
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if writer and epoch_losses:
            writer.add_scalar("loss_epoch/Training", sum(epoch_losses) / len(epoch_losses), start_epoch + epoch)

        if val_dataloader is not None:
            val_loss = evaluate_loss_Classifier(val_dataloader, model, criterion, device)
            print(f"  validation loss: {val_loss:.4f}")
            if writer:
                writer.add_scalar("loss_epoch/Validation", val_loss, start_epoch + epoch)
            model.train()

    return model


def extract_features_Classifier(dataloader, model: SeedClassifier, device):
    """
    extract features per batch
    """
    model.eval()

    for X, y in dataloader:
        X = X.to(device)
        out = X
        for layer in model.classifier.layers:
            out = layer(out)
            out = model.classifier.activation(out)
        yield out, y


def evaluate_loss_Classifier(dataloader, model: SeedClassifier, criterion, device) -> float:
    """Computes the average loss of the model over a dataloader, without updating weights."""
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            inputs = X.to(device)
            labels = y.to(device)
            pred = model.classifier(inputs)
            loss = criterion(pred, labels)
            losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("nan")


def icing_on_the_cake_Classifier(
    trainloader,
    model: SeedClassifier,
    n_epochs: int,
    lr: float,
    device,
    writer: SummaryWriter | None = None,
    start_epoch: int = 0,
    val_dataloader=None,
):
    """
    Re-training the output_layer only
    """
    optimizer = torch.optim.Adam(model.classifier.output_layer.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.train()
    for epoch in range(n_epochs):
        print(f"ICK epoch: {epoch}")
        epoch_losses = []
        for features, labels in extract_features_Classifier(trainloader, model, device):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model.classifier.output_layer(features)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if writer and epoch_losses:
            writer.add_scalar("loss_epoch/Icing", sum(epoch_losses) / len(epoch_losses), epoch)

        if val_dataloader is not None:
            val_losses = []
            with torch.no_grad():
                for features, labels in extract_features_Classifier(val_dataloader, model, device):
                    features = features.to(device)
                    labels = labels.to(device)
                    pred = model.classifier.output_layer(features)
                    val_losses.append(criterion(pred, labels).item())
            if val_losses:
                val_loss = sum(val_losses) / len(val_losses)
                print(f"  ICK validation loss: {val_loss:.4f}")
                if writer:
                    writer.add_scalar("loss_epoch/Validation_Icing", val_loss, start_epoch + epoch)
            model.train()

    return model


def train_classifier(
    train_dataloader,
    model: SeedClassifier,
    optimizer,
    cfg: ClassifierConfig,
    start_epoch: int = 0,
    criterion=None,
    writer: SummaryWriter | None = None,
    val_dataloader=None,
) -> SeedClassifier:

    device = cfg.device_acc
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    model = train_loop_Classifier(
        train_dataloader,
        model,
        cfg.n_epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        writer=writer,
        start_epoch=start_epoch,
        val_dataloader=val_dataloader,
    )
    model = icing_on_the_cake_Classifier(
        train_dataloader,
        model,
        n_epochs=cfg.icing_epochs,
        lr=cfg.icing_lr,
        device=device,
        writer=writer,
        start_epoch=start_epoch + cfg.n_epochs,
        val_dataloader=val_dataloader,
    )
    return model


def evaluate_classifier(model: SeedClassifier, dataloader, device):
    """
    Runs the model over a dataloader and collects the true-seed discriminant score (softmax
    probability of label 1) together with the ground-truth labels (0: fake seed, 1: true seed).
    """
    model = model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            probs = torch.softmax(model.classifier(X), dim=1)
            all_scores.append(probs[:, 1].cpu())
            all_labels.append(y.cpu())

    return torch.cat(all_scores), torch.cat(all_labels)


def compute_roc_curve(labels, scores):
    """
    Computes the ROC curve for the true-seed discriminant.

    Args:
        labels: ground-truth labels (0: fake seed, 1: true seed)
        scores: discriminant, i.e. the probability of being a true seed

    Returns:
        (fpr, tpr, thresholds), all sorted by decreasing threshold.
    """
    order = torch.argsort(scores, descending=True)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    is_signal = labels_sorted == 1
    is_background = labels_sorted == 0

    n_signal = is_signal.sum()
    n_background = is_background.sum()
    if n_signal == 0 or n_background == 0:
        raise ValueError("Both true and fake seeds are required to compute a ROC curve.")

    tpr = torch.cumsum(is_signal, dim=0) / n_signal
    fpr = torch.cumsum(is_background, dim=0) / n_background

    # Prepend the (0, 0) point corresponding to an infinite threshold.
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])
    thresholds = torch.cat([torch.tensor([float("inf")]), scores_sorted])

    return fpr, tpr, thresholds


def compute_auc(fpr, tpr) -> float:
    """Area under the ROC curve, via the trapezoidal rule."""
    return torch.trapz(tpr, fpr).item()


def fpr_at_tpr(fpr, tpr, target_tpr: float = 0.99) -> float:
    """False positive rate at the first operating point reaching target_tpr."""
    idx = min(torch.searchsorted(tpr, torch.tensor(target_tpr)).item(), len(tpr) - 1)
    return fpr[idx].item()


def threshold_at_tpr(tpr, thresholds, target_tpr: float = 0.99) -> float:
    """Discriminant threshold at the first operating point reaching target_tpr."""
    idx = min(torch.searchsorted(tpr, torch.tensor(target_tpr)).item(), len(tpr) - 1)
    return thresholds[idx].item()


def plot_roc_curve(fpr, tpr, auc: float):
    """Builds a matplotlib figure of the ROC curve, annotated with the AUC."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(fpr.numpy(), tpr.numpy(), label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate (fake seed accepted)")
    ax.set_ylabel("True Positive Rate (true seed accepted)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    return fig


def plot_discriminant(scores, labels):
    """Builds a matplotlib figure of the discriminant distribution, split by true/fake seed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    bins = 50
    ax.hist(scores[labels == 1].numpy(), bins=bins, range=(0, 1), alpha=0.5, label="True seeds", density=True)
    ax.hist(scores[labels == 0].numpy(), bins=bins, range=(0, 1), alpha=0.5, label="Fake seeds", density=True)
    ax.set_xlabel("Discriminant (probability of true seed)")
    ax.set_ylabel("Density")
    ax.set_title("Discriminant distribution")
    ax.legend(loc="upper center")

    return fig


def test_classifier(model: SeedClassifier, dataloader, device, writer: SummaryWriter | None = None, global_step: int = 0):
    """
    Evaluates the classifier: prints the discriminant, the ROC AUC, and the false positive
    rate at a 99% true positive rate.
    """
    scores, labels = evaluate_classifier(model, dataloader, device)
    print("Discriminant (probability of true seed):", scores)

    fpr, tpr, thresholds = compute_roc_curve(labels, scores)
    auc = compute_auc(fpr, tpr)
    fpr_99 = fpr_at_tpr(fpr, tpr, target_tpr=0.99)
    threshold_99 = threshold_at_tpr(tpr, thresholds, target_tpr=0.99)

    print(f"ROC AUC: {auc:.4f}")
    print(f"False positive rate at 99% TPR: {fpr_99:.4f}")
    print(f"Threshold at 99% TPR: {threshold_99:.4f}")

    if writer:
        writer.add_scalar("test/AUC", auc, global_step)
        writer.add_scalar("test/FPR_at_99_TPR", fpr_99, global_step)
        writer.add_scalar("test/Threshold_at_99_TPR", threshold_99, global_step)
        writer.add_figure("test/ROC_curve", plot_roc_curve(fpr, tpr, auc), global_step)
        writer.add_figure("test/discriminant", plot_discriminant(scores, labels), global_step)

    return {
        "scores": scores,
        "labels": labels,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "fpr_at_99_tpr": fpr_99,
        "threshold_at_99_tpr": threshold_99,
    }


if __name__ == "__main__":

    cfg = ClassifierConfig()
    cfg.parse_args()

    model_Classifier = SeedClassifier(
        classifier_config=cfg,
        device_acc=cfg.device_acc,
    )
    optimizer_Classifier = torch.optim.Adam(model_Classifier.parameters(), cfg.lr)

    start_epoch = 0
    if cfg.resume_training:
        print(f"Resuming classifier training from {cfg.model_path}...")
        start_epoch = model_Classifier.load(path=cfg.model_path, device=cfg.device_acc, optimizer=optimizer_Classifier)

    seed_features = torch.load(cfg.input_tensor_path, weights_only=True)

    X = seed_features["features"]
    y = seed_features["labels"].squeeze()

    X, y = balance_dataset(X, y)

    if X.dim() == 1:
        X = X.reshape(X.shape[0], -1)

    seeds = model_Classifier.prepare_seed_features(X)

    X_trainval, y_trainval, X_test, y_test = split_train_val(seeds, y, val_fraction=cfg.test_fraction)
    X_train, y_train, X_val, y_val = split_train_val(X_trainval, y_trainval, val_fraction=cfg.val_fraction)

    Seed_Dataset = SeedDataset(X_train, y_train)
    Val_Dataset = SeedDataset(X_val, y_val)
    Test_Dataset = SeedDataset(X_test, y_test)

    Seed_dataloader = TorchDataLoader(dataset=Seed_Dataset, batch_size=cfg.batch_size, shuffle=False)
    Val_dataloader = TorchDataLoader(dataset=Val_Dataset, batch_size=cfg.batch_size, shuffle=False)
    Test_dataloader = TorchDataLoader(dataset=Test_Dataset, batch_size=cfg.batch_size, shuffle=False)

    log_dir = "training_classifier"
    writer = SummaryWriter(log_dir=log_dir) if os.path.exists(log_dir) else SummaryWriter(log_dir)
    print(
        f"A total of {len(Seed_Dataset)} seeds are used for training, {len(Val_Dataset)} for validation "
        f"and {len(Test_Dataset)} for testing."
    )
    model = train_classifier(
        train_dataloader=Seed_dataloader,
        model=model_Classifier,
        optimizer=optimizer_Classifier,
        cfg=cfg,
        start_epoch=start_epoch,
        criterion=torch.nn.CrossEntropyLoss().to(cfg.device_acc),
        writer=writer,
        val_dataloader=Val_dataloader,
    )

    res = test_classifier(
        model=model_Classifier,
        dataloader=Test_dataloader,
        device=cfg.device_acc,
        writer=writer,
        global_step=start_epoch + cfg.n_epochs + cfg.icing_epochs,
    )
    model.threshold = res["threshold_at_99_tpr"]
    model.save(epoch=start_epoch + cfg.n_epochs - 1, path=cfg.model_path, optimizer=optimizer_Classifier)

    writer.close()
