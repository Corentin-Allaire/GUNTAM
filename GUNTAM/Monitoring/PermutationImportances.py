# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""" PERMUTATION IMPORTANCES """""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from typing import List, Sequence, Tuple

import torch
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config.SeedConfig import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Monitoring.PermutationMetricEvaluator import validate_function
from GUNTAM.Monitoring.PermutationMetricEvaluator import efficiency_reconstructed_seeds
from GUNTAM.Monitoring.PermutationMetricEvaluator import PermutationMetricEvaluator


def parse_args():
    """
    Command-line argument parser for the permutation importance analysis.

    This function defines the CLI arguments used to select the dataset path,
    the dataset name, and the transformer model checkpoint to analyze.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following fields:
            - path (str): Path to the directory containing the dataset.
            - dataset_name (str): Name of the dataset to load.
            - model_name (str): Name of the trained transformer checkpoint file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/justine/Documents/GUNTAM/odd_output_new_5")
    parser.add_argument("--dataset_name", type=str, default="odd_output_new_5")
    parser.add_argument("--model_name", type=str, default="transformer_98_seed_eff.pt")
    return parser.parse_args()


def config_model_dataset(path: str, dataset_name: str, model_name: str):
    """
    Load the dataset and the pre-trained transformer model.

    This function initialises the seed configuration, loads the specified
    dataset via the DataLoader, and instantiates the SeedTransformer with
    the loaded weights from the given model checkpoint.

    Args:
        path (str): Path to the directory containing the dataset.
        dataset_name (str): Name of the dataset to load.
        model_name (str): Name of the transformer checkpoint file (.pt) to load.

    Returns:
        tuple: A tuple containing:
            - cfg (SeedConfig): Configuration used to load the dataset and model.
            - dataset (DataLoader): Loaded dataset object with the required tensors.
            - model (SeedTransformer): Initialised transformer model with loaded weights.
    """

    cfg = SeedConfig()
    # cfg.parse_args()
    # Use default weights (1.0) for each loss component if none are specified
    if not cfg.loss_weights:
        cfg.loss_weights = [1.0] * len(cfg.loss_components)
    cfg.loss_config = dict(zip(cfg.loss_components, cfg.loss_weights))
    cfg.epoch_nb = 1
    cfg.transformer_config.embedding_mode = "MLP"

    cfg.input_tensor_path = path

    tensor_list = {
        "hits_tensor",
        "particles_tensor",
        "hit_to_particle_tensor",
        "padding_mask",
        "good_pairs",
    }

    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=dataset_name,
        tensor_names=list(tensor_list),
        device=cfg.device_acc,
    )

    model = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=cfg.device_acc,
        dtype=torch.float32,
    )
    model.to(cfg.device_acc)
    model.load(path=model_name, device=cfg.device_acc)

    return cfg, dataset, model


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""" SHUFFLE PER FEATURES """"""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# multiprocessing :


def perm_imp_per_features_loss(enc_hits_shape: int, dataset, model, cfg):
    """
    Compute permutation importance per feature using the loss metric.

    This function uses multiprocessing to compute the average loss for
    each feature by shuffling one feature at a time across the dataset.
    Each feature index is processed in a separate subprocess to speed
    up the computation.

    Args:
        enc_hits_shape (int): Number of features in the dataset after Fourier encoding.
        dataset: The dataset object containing the events to evaluate.
        model: The trained transformer model to evaluate.
        cfg: Configuration object with training and evaluation parameters.

    Returns:
        list: List of average losses, one per shuffled feature index.
    """

    validation = PermutationMetricEvaluator(metric_fn=validate_function, model=model, dataset=dataset, cfg=cfg)

    args = list(range(enc_hits_shape))

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=1) as pool:
        results = pool.map(validation, args)

    avg_loss_total = results

    return avg_loss_total


def perm_imp_per_features_eff(enc_hits_shape: int, dataset, model, cfg):
    """
    Compute permutation importance per feature using the seeding efficiency metric.

    This function uses multiprocessing to compute the seeding efficiency
    for each feature by shuffling one feature at a time across the dataset.
    Each feature index is processed in a separate subprocess to speed
    up the computation.

    Args:
        enc_hits_shape (int): Number of features in the dataset after Fourier encoding.
        dataset: The dataset object containing the events to evaluate.
        model: The trained transformer model to evaluate.
        cfg: Configuration object with training and evaluation parameters.

    Returns:
        list: List of seeding efficiencies, one per shuffled feature index.
    """

    efficiency = PermutationMetricEvaluator(metric_fn=efficiency_reconstructed_seeds, model=model, dataset=dataset, cfg=cfg)

    args = list(range(enc_hits_shape))

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=1) as pool:
        results = pool.map(efficiency, args)

    seed_eff_total = results

    return seed_eff_total


# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""" SHUFFLE FEATURES """"""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def perm_imp_featuress(features: List[int], method: str, dataset, model, cfg):
    """
    Compute permutation importance when shuffling multiple features simultaneously.

    This function computes the average validation loss or the seeding
    efficiency of the model when the specified set of features is shuffled
    together across the dataset.

    Args:
        features (List[int]): Indices of the features to shuffle simultaneously.
        method (str): Metric to compute. Must be either "loss" or "seed_eff".
        dataset: The dataset object containing the events to evaluate.
        model: The trained transformer model to evaluate.
        cfg: Configuration object with training and evaluation parameters.

    Returns:
        float: The average loss (if method="loss") or the seeding efficiency
               (if method="seed_eff") after shuffling the given features.
    """

    if method == "loss":
        return validate_function(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, features=features
        )
    if method == "seed_eff":
        return efficiency_reconstructed_seeds(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, features=features
        )

    raise ValueError(f"Unknown method: {method}. Must be 'loss' or 'seed_eff'.")


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""" PLOTS """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def feature_channel_indices(model, feature: str, hit_feature_names: List[str] | None = None) -> Sequence[int | Tuple[int, int]]:
    """
    Return the Fourier-embedding channel indices belonging to a physical feature,
    following the frequency-blocked layout of FourierPositionalEncoding.

    For a coordinate feature (in ``embedding_feature``), returns a list of
    ``(sin_channel, cos_channel)`` tuples, one per frequency level at which the
    feature is active, in level order (level 0 first). For a high-level feature
    (in ``high_level_features``), returns a list with its single channel in the
    tail of the embedding (cos/sin-expanded features get one channel each).

    Args:
        model: The SeedTransformer (provides ``fourier_encoding`` and the config).
        feature: Feature name (e.g. "x", "y", "z", "r", "phi", "eta", or
            "cos_phi"/"sin_phi" for cosine-expanded features).
        hit_feature_names: Names of the raw hit columns, indexed by the
            ``embedding_feature`` / ``high_level_features`` / ``cosine_processing``
            indices in the config.

    Returns:
        List[Tuple[int, int]]: One (sin, cos) channel pair per frequency level
            (coordinate features), or List[int] with the single channel index
            (high-level features).
    """
    if hit_feature_names is None:
        hit_feature_names = ["x", "y", "z", "r", "phi", "eta"]

    cfg = model.cfg
    fpe = model.fourier_encoding
    num_freq = list(fpe.num_frequencies)

    emb_cos = [i for i in cfg.embedding_feature if i in cfg.cosine_processing]
    emb_plain = [i for i in cfg.embedding_feature if i not in cfg.cosine_processing]
    high_cos = [i for i in cfg.high_level_features if i in cfg.cosine_processing]
    high_plain = [i for i in cfg.high_level_features if i not in cfg.cosine_processing]

    # Coordinate dimensions in embedding order: plain, then cos(φ), sin(φ)
    coord_names = (
        [hit_feature_names[i] for i in emb_plain]
        + [f"cos_{hit_feature_names[i]}" for i in emb_cos]
        + [f"sin_{hit_feature_names[i]}" for i in emb_cos]
    )

    if feature in coord_names:
        d = coord_names.index(feature)
        indices: List[Tuple[int, int]] = []
        base = 0
        for k in range(max(num_freq)):
            active = [m for m, n in enumerate(num_freq) if n > k]
            if d in active:
                pos = active.index(d)
                # (sin, cos) channel pair for this feature at level k
                indices.append((base + pos, base + len(active) + pos))
            base += 2 * len(active)
        return indices

    # High-level features live in the tail: [cos(φ), sin(φ), ..., plain]
    high_names = (
        [f"cos_{hit_feature_names[i]}" for i in high_cos]
        + [f"sin_{hit_feature_names[i]}" for i in high_cos]
        + [hit_feature_names[i] for i in high_plain]
    )
    if feature in high_names:
        return [fpe.output_dim - len(high_names) + high_names.index(feature)]

    raise ValueError(f"Unknown feature: {feature} (known: {coord_names + high_names})")


def plot_freq_loss(avg_loss_ref: float, avg_loss_total: list, model, feature: str, hit_feature_names: List[str] | None = None):
    """
    Plot the variation in average loss for each Fourier frequency of a given feature.

    Under the frequency-blocked embedding layout, the channels of a feature are
    spread across the embedding (one sin/cos pair per frequency level), so this
    function collects the feature's channels via feature_channel_indices and
    plots the absolute deviation from the reference loss per frequency level.
    For high-level (non-Fourier) features, one point per channel is plotted.

    Args:
        avg_loss_ref (float): Reference average loss (computed without any shuffling).
        avg_loss_total (list): List of average losses for each shuffled feature.
        model: The SeedTransformer used to derive the embedding channel layout.
        feature (str): The feature to analyse (e.g. "x", "y", "z", "r", "phi", "eta").
        hit_feature_names (List[str], optional): Names of the raw hit columns.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """
    if hit_feature_names is None:
        hit_feature_names = ["x", "y", "z", "r", "phi", "eta"]

    indices = feature_channel_indices(model, feature, hit_feature_names)

    plt.figure()

    for j, channel_indices in enumerate(indices):
        if isinstance(channel_indices, tuple):
            sin_idx, cos_idx = channel_indices
            plt.plot(abs(avg_loss_total[sin_idx] - avg_loss_ref), j + 1, "o", color="red")
            plt.plot(abs(avg_loss_total[cos_idx] - avg_loss_ref), j + 1, "o", color="blue")
        else:
            idx = channel_indices
            plt.plot(abs(avg_loss_total[idx] - avg_loss_ref), j + 1, "o")

    plt.xlabel("delta_loss")
    plt.ylabel("k")
    plt.title(f"permutation importances for frequencies of {feature}")
    plt.savefig(f"perm_imp_freq_{feature}_loss.png")

    plt.close()


def plot_all_loss(avg_loss_ref: float, avg_loss_total: list):
    """
    Plot the permutation importance (loss deviation) for every feature.

    This function produces a scatter plot showing the absolute difference
    between the reference loss and the loss obtained when each individual
    feature is shuffled.

    Args:
        avg_loss_ref (float): Reference average loss (computed without any shuffling).
        avg_loss_total (list): List of average losses for each shuffled feature.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """

    start = 0
    finish = len(avg_loss_total)

    plt.figure()

    for j in range(start, finish):

        plt.plot(abs(avg_loss_total[j] - avg_loss_ref), j + 1, "o")

    plt.xlabel("delta_loss")
    plt.ylabel("features")
    plt.title("permutation importances for each features")
    plt.savefig("perm_imp_all_features_loss.png")

    plt.close()


def plot_freq_eff(seed_eff_total: list, model, feature: str, hit_feature_names: List[str] | None = None):
    """
    Plot the seeding efficiency for each Fourier frequency of a given feature.

    Under the frequency-blocked embedding layout, the channels of a feature are
    spread across the embedding (one sin/cos pair per frequency level), so this
    function collects the feature's channels via feature_channel_indices and
    plots the efficiency values per frequency level.
    For high-level (non-Fourier) features, one point per channel is plotted.

    Args:
        seed_eff_total (list): List of seeding efficiencies for each shuffled feature.
        model: The SeedTransformer used to derive the embedding channel layout.
        feature (str): The feature to analyse (e.g. "x", "y", "z", "r", "phi", "eta").
        hit_feature_names (List[str], optional): Names of the raw hit columns.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """
    if hit_feature_names is None:
        hit_feature_names = ["x", "y", "z", "r", "phi", "eta"]

    indices = feature_channel_indices(model, feature, hit_feature_names)

    plt.figure()

    for j, channel_indices in enumerate(indices):
        if isinstance(channel_indices, tuple):
            sin_idx, cos_idx = channel_indices
            plt.plot(abs(seed_eff_total[sin_idx]), j + 1, "o", color="red")
            plt.plot(abs(seed_eff_total[cos_idx]), j + 1, "o", color="blue")
        else:
            idx = channel_indices
            plt.plot(abs(seed_eff_total[idx]), j + 1, "o")

    plt.xlabel("seeding efficiency")
    plt.ylabel("k")
    plt.xlim(0.70, 0.99)
    plt.title(f"permutation importances for frequencies of {feature}")
    plt.savefig(f"perm_imp_freq_{feature}_eff.png")

    plt.close()


def plot_all_eff(seed_eff_total: list):
    """
    Plot the seeding efficiency for every feature.

    This function produces a scatter plot showing the seeding efficiency
    obtained when each individual feature is shuffled.

    Args:
        seed_eff_total (list): List of seeding efficiencies for each shuffled feature.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """

    start = 0
    finish = len(seed_eff_total)

    plt.figure()

    for j in range(start, finish):

        plt.plot(abs(seed_eff_total[j]), j + 1, "o")

    plt.xlabel("seeding efficiency")
    plt.ylabel("features")
    plt.title("permutation importances for all features")
    plt.savefig("perm_imp_all_features_eff.png")

    plt.close()


# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""" MAIN """""""""""""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

if __name__ == "__main__":

    args = parse_args()

    cfg, dataset, model = config_model_dataset(
        path=args.path,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )

    # Number of channels in the Fourier embedding (derived from the model, not hardcoded)
    enc_hits_shape = model.fourier_encoding.output_dim

    # Computing the average loss and the seeding efficiency for every feature we shuffle:
    avg_loss_total = perm_imp_per_features_loss(enc_hits_shape=enc_hits_shape, dataset=dataset, model=model, cfg=cfg)
    seed_eff_total = perm_imp_per_features_eff(enc_hits_shape=enc_hits_shape, dataset=dataset, model=model, cfg=cfg)

    # Computing the average loss without shuffling any feature (average loss of reference):
    avg_loss_ref = validate_function(model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg)
    print("loss=", avg_loss_ref)

    # Plotting every average loss for each feature we shuffle:
    plot_all_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total)
    # Plotting the average loss for each feature of Emb(x) we shuffle:
    plot_freq_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total, model=model, feature="x")

    # Plotting every seeding efficiency for each feature we shuffle:
    plot_all_eff(seed_eff_total=seed_eff_total)
    # Plotting the seeding efficiency for each feature of Emb(x) we shuffle:
    plot_freq_eff(seed_eff_total=seed_eff_total, model=model, feature="x")
