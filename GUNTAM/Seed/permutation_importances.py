# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""" PERMUTATION IMPORTANCES """"""""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from Validation_loss_function import validate_function
from Validation_loss_class import Validation_class
from Efficiency_seed_function import efficiency_reconstructed_seeds
from Efficiency_seed_class import Efficiency_class


def config_model_dataset(path: str, dataset_name: str, model_name: str):
    """
    We load the dataset and the transformer.
    Args:
    path: dataset path
    dataset_name: name of the dataset
    model_name: name of the tranformer.pt we want to load

    Returns:
    cfg, dataset and model
    """

    cfg = SeedConfig()
    cfg.parse_args()
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
# """""""""""""""""""""""""""""""""" SHUFFLE PER FEATURES """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# multiprocessing :


def perm_imp_per_features_loss(enc_hits_shape: int, dataset, model, cfg):
    """
    We use multiprocessing to speed up the process of computing each average loss for each feature we shuffle.
    Args:
    enc_hits_shape: number of features in the dataset after the Fourier encoding

    Returns:
    if method=="loss", list of each average loss for each feature we shuffle.
    if method=="seed_eff", list of each efficiency for each feature we shuffle.
    """
    avg_loss_total = []

    validation = Validation_class(model=model, dataset=dataset, cfg=cfg)

    args = []

    for i in range(enc_hits_shape):
        args.append(i)

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=1) as pool:
        results = pool.map(validation.validate_class, args)

    avg_loss_total = results

    return avg_loss_total


def perm_imp_per_features_eff(enc_hits_shape: int, dataset, model, cfg):
    """
    We use multiprocessing to speed up the process of computing each average loss for each feature we shuffle.
    Args:
    enc_hits_shape: number of features in the dataset after the Fourier encoding

    Returns:
    if method=="loss", list of each average loss for each feature we shuffle.
    if method=="seed_eff", list of each efficiency for each feature we shuffle.
    """
    seed_eff_total = []

    efficiency = Efficiency_class(model=model, dataset=dataset, cfg=cfg)

    args = []

    for i in range(enc_hits_shape):
        args.append(i)

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=1) as pool:
        results = pool.map(efficiency.seed_eff_class, args)

    seed_eff_total = results

    return seed_eff_total


# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""" SHUFFLE SITUATIONS """""""""""""""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def perm_imp_situations(situation: str, method: str, dataset, model, cfg):
    """
    We compute the average loss of the model for different situations where we shuffle multiple features.
    Args:
    situation: name of the situation where we shuffle multiple features

    Returns:
    if method=="loss", average loss of the situation.
    if method=="seed_eff", efficiency of the situation.
    """

    if method == "loss":
        result = validate_function(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, situation="xyz"
        )
    if method == "seed_eff":
        result = efficiency_reconstructed_seeds(
            model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg, situation="xyz"
        )

    return result


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""" PLOTS """""""""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def plot_freq_loss(avg_loss_ref: int, avg_loss_total: list, feature: str):
    """
    We plot the average loss of each model in which we shuflle the frequencies of each embedded feature.
    Args:
    avg_loss_ref: average loss of reference (without any shuffling)
    avg_loss_total: average loss of the situation

    Returns:
    Plots
    """

    # Frequencies of the dataset (unique for each dataset)

    if feature == "z":
        avg_loss_total = avg_loss_total[120:160]
    elif feature == "r":
        avg_loss_total = avg_loss_total[160:200]
    elif feature == "x":
        avg_loss_total = avg_loss_total[40:80]
    elif feature == "y":
        avg_loss_total = avg_loss_total[80:120]
    elif feature == "eta":
        avg_loss_total = avg_loss_total[200:240]
    elif feature == "phi":
        avg_loss_total = avg_loss_total[0:40]
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


def plot_all_loss(avg_loss_ref: int, avg_loss_total: list):
    """
    We plot the average loss of each model in which we shuffle one feature at a time.
    Args:
    avg_loss_ref: average loss of reference (without any shuffling)
    avg_loss_total: average loss of the situation

    Returns:
    Plots
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


def plot_freq_eff(seed_eff_total: list, feature: str):
    """
    We plot the seeding efficiency of each model in which we shuflle the frequencies of each embedded feature.
    Args:
    avg_loss_ref: average loss of reference (without any shuffling)
    avg_loss_total: average loss of the situation

    Returns:
    Plots
    """

    # Frequencies of the dataset (unique for each dataset)

    if feature == "z":
        seed_eff_total = seed_eff_total[120:160]
    elif feature == "r":
        seed_eff_total = seed_eff_total[160:200]
    elif feature == "x":
        seed_eff_total = seed_eff_total[40:80]
    elif feature == "y":
        seed_eff_total = seed_eff_total[80:120]
    elif feature == "phi":
        seed_eff_total = seed_eff_total[0:40]
    elif feature == "eta":
        seed_eff_total = seed_eff_total[200:240]
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
    We plot the average loss of each model in which we shuffle one feature at a time.
    Args:
    avg_loss_ref: average loss of reference (without any shuffling)
    avg_loss_total: average loss of the situation

    Returns:
    Plots
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

    cfg, dataset, model = config_model_dataset(
        path="/home/justine/Documents/GUNTAM/odd_output_new_5",
        dataset_name="odd_output_new_5",
        model_name="transformer_98_seed_eff.pt",
    )

    # avg_loss_total = perm_imp_per_features_loss(enc_hits_shape=45, dataset=dataset, model=model, cfg=cfg)
    # seed_eff_total = perm_imp_per_features_eff(enc_hits_shape, dataset=dataset, model=model, cfg=cfg)

    avg_loss_ref = validate_function(
        model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg
    )  # shuffle_v=None and situation=None
    print("loss=", avg_loss_ref)
    # avg_loss_ref = 75.48

    # plot_all_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total)
    # plot_freq_loss(avg_loss_ref=avg_loss_ref, avg_loss_total=avg_loss_total, feature="x")

    # plot_all_eff(seed_eff_total=seed_eff_total)
    # plot_freq_eff(seed_eff_total=seed_eff_total, feature="x")
