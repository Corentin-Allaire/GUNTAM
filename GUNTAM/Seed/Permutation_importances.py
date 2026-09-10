# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""" PERMUTATION IMPORTANCES """""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from typing import List

import torch
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.PermutationMetricEvaluator import validate_function
from GUNTAM.Seed.PermutationMetricEvaluator import efficiency_reconstructed_seeds
from GUNTAM.Seed.PermutationMetricEvaluator import PermutationMetricEvaluator


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
            - dataset (DataLoader): Loaded dataset object with the required tensors.
            - model (SeedTransformer): Initialised transformer model with loaded weights.
    """

    cfg = SeedConfig()
    # cfg.parse_args()
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

    return dataset, model


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
        result = validate_function(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, features=features
        )
    if method == "seed_eff":
        result = efficiency_reconstructed_seeds(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, features=features
        )

    return result


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""""" PLOTS """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def plot_freq_loss(avg_loss_ref: float, avg_loss_total: list, feature: str, fourier_freq: int = 20):
    """
    Plot the variation in average loss for each Fourier frequency of a given feature.

    This function selects the slice of the total loss vector corresponding to the
    Fourier frequencies of the specified feature and plots the absolute deviation
    from the reference loss for each frequency index.

    Args:
        avg_loss_ref (float): Reference average loss (computed without any shuffling).
        avg_loss_total (list): List of average losses for each shuffled feature.
        feature (str): The feature to analyse (one of "phi", "x", "y", "z", "r", "eta").
        fourier_freq (int, optional): Number of Fourier frequencies used in the encoding. Defaults to 20.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """

    # Frequencies of the dataset (unique for each dataset)

    if feature == "z":
        avg_loss_total = avg_loss_total[(fourier_freq * 6) : (fourier_freq * 8)]
    elif feature == "r":
        avg_loss_total = avg_loss_total[(fourier_freq * 8) : (fourier_freq * 10)]
    elif feature == "x":
        avg_loss_total = avg_loss_total[(fourier_freq * 2) : (fourier_freq * 4)]
    elif feature == "y":
        avg_loss_total = avg_loss_total[(fourier_freq * 4) : (fourier_freq * 6)]
    elif feature == "eta":
        avg_loss_total = avg_loss_total[(fourier_freq * 10) : (fourier_freq * 12)]
    elif feature == "phi":
        avg_loss_total = avg_loss_total[0 : (fourier_freq * 2)]
    else:
        raise ValueError(f"shuffled feature: {feature}")

    plt.figure()

    for j in range(0, int(len(avg_loss_total) / 2)):

        plt.plot(abs(avg_loss_total[2 * j] - avg_loss_ref), j + 1, "o", color="red")
        plt.plot(abs(avg_loss_total[2 * j + 1] - avg_loss_ref), j + 1, "o", color="blue")

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


def plot_freq_eff(seed_eff_total: list, feature: str, fourier_freq: int = 20):
    """
    Plot the seeding efficiency for each Fourier frequency of a given feature.

    This function selects the slice of the total efficiency vector corresponding
    to the Fourier frequencies of the specified feature and plots the efficiency
    values for each frequency index.

    Args:
        seed_eff_total (list): List of seeding efficiencies for each shuffled feature.
        feature (str): The feature to analyse (one of "phi", "x", "y", "z", "r", "eta").
        fourier_freq (int, optional): Number of Fourier frequencies used in the encoding. Defaults to 20.

    Returns:
        None: This function generates and saves a matplotlib plot.
    """

    # Frequencies of the dataset (unique for each dataset)

    if feature == "z":
        seed_eff_total = seed_eff_total[(fourier_freq * 6) : (fourier_freq * 8)]
    elif feature == "r":
        seed_eff_total = seed_eff_total[(fourier_freq * 8) : (fourier_freq * 10)]
    elif feature == "x":
        seed_eff_total = seed_eff_total[(fourier_freq * 2) : (fourier_freq * 4)]
    elif feature == "y":
        seed_eff_total = seed_eff_total[(fourier_freq * 4) : (fourier_freq * 6)]
    elif feature == "eta":
        seed_eff_total = seed_eff_total[(fourier_freq * 10) : (fourier_freq * 12)]
    elif feature == "phi":
        seed_eff_total = seed_eff_total[0 : (fourier_freq * 2)]
    else:
        raise ValueError(f"shuffled feature: {feature}")

    plt.figure()

    for j in range(0, int(len(seed_eff_total) / 2)):

        plt.plot(abs(seed_eff_total[2 * j]), j + 1, "o", color="red")
        plt.plot(abs(seed_eff_total[2 * j + 1]), j + 1, "o", color="blue")

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

    # Computing the average loss and the seeding efficiency for every feature we shuffle:
    avg_loss_total = perm_imp_per_features_loss(enc_hits_shape=45, dataset=dataset, model=model, cfg=cfg)
    seed_eff_total = perm_imp_per_features_eff(enc_hits_shape=45, dataset=dataset, model=model, cfg=cfg)

    # Computing the average loss without shuffling any feature (average loss of reference):
    avg_loss_ref = validate_function(model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg)
    print("loss=", avg_loss_ref)

    # Plotting every average loss for each feature we shuffle:
    plot_all_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total)
    # Plotting the average loss for each feature of Emb(x) we shuffle:
    plot_freq_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total, feature="x")

    # Plotting every seeding efficiency for each feature we shuffle:
    plot_all_eff(seed_eff_total=seed_eff_total)
    # Plotting the seeding efficiency for each feature of Emb(x) we shuffle:
    plot_freq_eff(seed_eff_total=seed_eff_total, feature="x")
