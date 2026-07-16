import torch
import numpy as np
from typing import List
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.Reconstruction import batched_beam_search_seed_reconstruction
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# """"""""""""""" TRANSFORMER LOADING """"""""""""""""


def transformer_loading(transformer_name):
    """
    Args:
    transformer_name: name of the trained transformer

    Returns:
    trained transformer loaded

    """

    cfg = SeedConfig()
    cfg.parse_args()
    cfg.epoch_nb = 1

    transformer = SeedTransformer(transformer_config=cfg.transformer_config, device_acc=cfg.device_acc, dtype=torch.float32)
    transformer.to(cfg.device_acc)
    transformer.load(path=transformer_name, device=cfg.device_acc)

    return transformer


# """" EXTRACTING SEEDS/HITS FROM THE ATTENTION MAPS OF THE TRANSFORMER """"


def transformer_seed_reconstruction(
    model: SeedTransformer,
    file_indices: list,
    dataset,
    cfg: SeedConfig,
):
    """
    We reconstruct the seeds once they've been through the transformer

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.

    Returns:
        hits_tensor: list of the coordinates of all the hits per bin
        seed_tensor: list of the coordinates of the hits per seed

    """

    model.eval()
    model_dtype = model.dtype

    with torch.no_grad():

        # We work on each file at a time:

        for file_idx in file_indices:
            print(file_idx)
            data = dataset.get_file(file_idx)

            # We define all the different information that are in data:

            hits_tensor = data["hits_tensor"]
            hits_tensor = hits_tensor.to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = data["particles_tensor"]
            particles_tensor = particles_tensor.to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = data["hit_to_particle_tensor"]
            hit_to_particle_tensor = hit_to_particle_tensor.to(cfg.device_acc)
            padding_mask = data["padding_mask"].to(cfg.device_acc)

            num_events = hits_tensor.shape[0]  # = 5 for odd_output_new_5

            # We work on each event at a time:

            for event_idx in range(num_events):

                # We define all the information above for one event:

                event_hits_tensor = hits_tensor[event_idx]
                event_padding_mask = padding_mask[event_idx]
                encoded_space_points, attention_maps = model(event_hits_tensor, event_padding_mask)

                if cfg.transformer_config.regression:  # sert à rien
                    hits_score = encoded_space_points

                else:

                    hits_score = attention_maps.squeeze(1)
                    hits_score = hits_score.max(dim=-1)
                    hits_score = hits_score.values.unsqueeze(-1)

                # We reconstruct the hits:

                chains, params, scores = batched_beam_search_seed_reconstruction(
                    attention_edge=attention_maps,
                    valid_mask=~event_padding_mask.bool(),
                    att_threshold=0.0,
                    max_chain_length=5,
                    beam_width=3,
                    backward=False,
                )

                # Extraire le seed_tensor en filtrant les seeds invalides
                chains = chains.squeeze(0)  # [N, max_chain_length]
                scores = scores.squeeze(0)  # [N]

                valid_seeds = scores > float("-inf")  # seeds avec au moins 3 hits
                seed_tensor = chains[valid_seeds]  # [num_valid_seeds, max_chain_length]
                # les -1 sont déjà là pour le padding, c'est le bon format pour build_seed_features_tensor

    return hits_tensor, seed_tensor


# """""""""""""""""""""""""""""""""""" PUTTING THESE SEEDS IN A FILE """"""""""""""""""""""""""""""""""""


def build_seed_features_tensor(
    hits_tensor: torch.Tensor,
    seed_tensor: torch.Tensor,
    feature_indices: List[int] = [0, 1, 2, 3, 4, 5],
    cosine_feature_indices: List[int] = [4],
) -> torch.Tensor:
    """
    Build a feature tensor for each seed by gathering hit coordinates.
    This can then be passed to NN for parameter regression and good/fake classification.

    Args:
        hits_tensor: Float tensor of shape [N, num_features] containing the hit
            features for all hits in a single bin.
        seed_tensor: Long tensor of shape [num_seeds, max_seed_size] containing the
            per-seed hit indices.  A value of -1 indicates a padding slot.
        feature_indices: Ordered list of column indices from `hits_tensor` to
            include in the output.  Mirrors `cfg.embedding_feature` /
            `cfg.high_level_features`.  Default: [0, 1, 2, 3, 4, 5].
        cosine_feature_indices: Subset of `feature_indices` for which cos/sin
            decomposition is applied.  Mirrors `cfg.cosine_processing`.
            Default: [4] (phi).

    Returns:
        Float tensor of shape [num_seeds, max_seed_size, F] where F is
        len(feature_indices) + len(cosine_feature_indices) (each cosine-processed
        feature adds one extra column for sin).  Padding slots contain all zeros.
    """
    pad_mask = seed_tensor == -1  # [num_seeds, max_seed_size]
    ids = seed_tensor.clamp(min=0)  # replace -1 with 0 to avoid out-of-bounds indexing

    feats = hits_tensor[ids]  # [num_seeds, max_seed_size, num_features]

    cosine_set = set(cosine_feature_indices)
    parts: List[torch.Tensor] = []
    for idx in feature_indices:
        if idx in cosine_set:
            parts.append(torch.cos(feats[..., idx]))
            parts.append(torch.sin(feats[..., idx]))
        else:
            parts.append(feats[..., idx])

    result = torch.stack(parts, dim=-1)  # [num_seeds, max_seed_size, F]

    result[pad_mask] = 0.0

    return result


def seed_features_file(input_tensor_path, dataset_name, transformer_name):
    """
    Args:
    input_tensor_path: where is the dataset used to train the transformer
    dataset_name: name of the dataset used to train the transformer
    transformer_name: name of the trained transformer

    Returns:
    seed_features: tensor of seeds and their labels reconstructed with the attention maps of the transformer

    """

    cfg = SeedConfig()

    cfg.input_tensor_path = input_tensor_path

    tensor_list = {
        "hits_tensor",
        "particles_tensor",
        "hit_to_particle_tensor",
        "padding_mask",
        "good_pairs",
    }

    # données pour le modèle :
    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=dataset_name,
        tensor_names=list(tensor_list),
        device=cfg.device_acc,
    )

    transformer = transformer_loading(transformer_name=transformer_name)

    hits_tensor, seed_tensor = transformer_seed_reconstruction(
        model=transformer, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg
    )

    seed_features = build_seed_features_tensor(
        hits_tensor=hits_tensor, seed_tensor=seed_tensor, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
    )

    torch.save(seed_features, "seed_features.pt")

    return seed_features


# """""""""""""""""""""""""""""""""""""""""""""""""""" PREPARING THIS FILE TO BE USED """"""""""""""""""""""""""""""""""""


def balance_dataset(X, y):
    """
    Args:
    X: features of each seed
    y: labels of each seeds

    Returns:
    A balanced dataset

    """

    values, counts = torch.unique(y, return_counts=True)  # proportion of true and fake seeds

    # Separating the true and fake seeds
    idx_true = (y == 0).nonzero(as_tuple=True)[0]
    idx_fake = (y == 1).nonzero(as_tuple=True)[0]

    n_min = min(len(idx_true), len(idx_fake))

    idx_balanced = torch.cat([idx_true[:n_min], idx_fake[:n_min]])

    X = X[idx_balanced]
    y = y[idx_balanced]

    # Shuffle
    perm = torch.randperm(len(y))
    X = X[perm]
    y = y[perm]

    return X, y


def circle_3_points_batch(X_np):
    """
    Args:
    X_np: X as a numpy object

    Returns:
    - coordonitates of the center of the circle
    - radius of the circle
    - coordinates of the normal of the circle

    """
    X_np = X_np.astype(np.float64)

    P1 = X_np[:, 0:3]
    P2 = X_np[:, 7:10]
    P3 = X_np[:, 14:17]

    print("P1 shape:", P1.shape)  # should be (N, 3)
    print("P1 dtype:", P1.dtype)  # should be float64

    a = P2 - P1
    b = P3 - P1
    normal = np.cross(a, b)

    print("normal shape:", normal.shape)  # should be (N, 3)

    norm_val = np.linalg.norm(normal, axis=1, keepdims=True)
    degenerate = norm_val[:, 0] < 1e-10
    norm_val = np.where(norm_val < 1e-10, 1.0, norm_val)
    normal /= norm_val

    row1 = 2 * (P2 - P1)
    row2 = 2 * (P3 - P1)
    row3 = normal

    print("row1 shape:", row1.shape)  # should be (N, 3)
    print("row2 shape:", row2.shape)
    print("row3 shape:", row3.shape)

    A = np.stack([row1, row2, row3], axis=1)
    b_vec = np.stack(
        [
            np.sum(P2**2, axis=1) - np.sum(P1**2, axis=1),
            np.sum(P3**2, axis=1) - np.sum(P1**2, axis=1),
            np.sum(normal * P1, axis=1),
        ],
        axis=1,
    )

    print("A shape:", A.shape)  # should be (N, 3, 3)
    print("b_vec shape:", b_vec.shape)  # should be (N, 3)

    A_inv = np.linalg.pinv(A)  # (N, 3, 3)
    center = np.einsum("nij,nj->ni", A_inv, b_vec)  # (N, 3)
    radius = np.linalg.norm(center - P1, axis=1, keepdims=True)

    center[degenerate] = 0.0
    radius[degenerate] = 0.0
    normal[degenerate] = 0.0

    return np.concatenate([center, radius, normal], axis=1)


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
        self.X = X.float()
        # The labels are integeres (0 or 1), we convert them to long (integer) PyTorch tensors.
        self.y = y.long()

    def __len__(self):
        length = len(self.y)
        return length

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def seed_features_file_adjustment(data, batch_size=1000):
    """

    Adjusting the datas of seed features to give it to the Classifier

    Args:
    data: seed_features created by seed_features_file

    Returns:
    Adjusted seed_features

    """

    X = data["features"]
    y = data["labels"].squeeze().long()

    X, y = balance_dataset(X, y)

    X_np = X.numpy().astype(np.float64)

    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 21)
        print("Shape après reshape :", X_np.shape)  # should be (N, 21)

    extra = circle_3_points_batch(X_np)

    extra_tensor = torch.tensor(extra, dtype=torch.float32)
    X = torch.cat([X, extra_tensor], dim=1)  # (11884838, 28)

    Seed_Dataset = SeedDataset(X, y)

    Seed_dataloader = DataLoader(dataset=Seed_Dataset, batch_size=batch_size)

    return Seed_dataloader
