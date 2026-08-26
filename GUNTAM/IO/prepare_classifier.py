import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.Reconstruction import batched_beam_search_seed_reconstruction
from GUNTAM.IO.DataLoader import DataLoader as GUNTAMDataLoader
from GUNTAM.Seed.Reconstruction import build_seed_features_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# """"""""""""""" TRANSFORMER LOADING """"""""""""""""


def transformer_loading(transformer_name):
    """
    This function is able to load the trained transformer saved as "transformer_name.pt"
    Args:
    transformer_name: name of the trained transformer

    Returns:
    trained transformer loaded

    """

    cfg = SeedConfig()
    # cfg.parse_args()
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
    We reconstruct the seeds once they've been through the transformer thanks to batched_beam_search_seed_reconstruction

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data (DataLoader).
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


def create_seed_features_file(input_tensor_path, dataset_name, transformer_name):
    """
    This function creates a file made of the reconstructed seeds, their features and labels
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
    dataset = GUNTAMDataLoader(
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

    return seed_features


# """""""""""""""""""""""""""""""""""""""""""""""""""" PREPARING THIS FILE TO BE USED """"""""""""""""""""""""""""""""""""


def balance_dataset(features, labels):
    """
    We balance the dataset in order to have the same number of fake and true seeds in the dataset
    Args:
    features: features of each seed
    labels: labels of each seeds

    Returns:
    A balanced dataset

    """

    values, counts = torch.unique(labels, return_counts=True)  # proportion of true and fake seeds

    # Separating the true and fake seeds
    idx_true = (labels == 0).nonzero(as_tuple=True)[0]
    idx_fake = (labels == 1).nonzero(as_tuple=True)[0]

    n_min = min(len(idx_true), len(idx_fake))

    idx_balanced = torch.cat([idx_true[:n_min], idx_fake[:n_min]])

    features = features[idx_balanced]
    labels = labels[idx_balanced]

    # Shuffle
    perm = torch.randperm(len(labels))
    features = features[perm]
    labels = labels[perm]

    return features, labels


def circle_3_points_batch(features: torch.Tensor) -> torch.Tensor:
    """
    We add 7 new features for each seed. These features are the parameters of the circle made of the three points in each seed
    Args:
    features: features tensor as a PyTorch tensor, shape [N, >=17]

    Returns:
    Float64 tensor of shape [N, 7]: concatenation of
    - coordinates of the center of the circle (3)
    - radius of the circle (1)
    - coordinates of the normal of the circle (3)

    """
    features = features.double()

    P1 = features[:, 0:3]  # positional features of the first point of the seed : x1, y1, z1
    P2 = features[:, 7:10]  # x2, y2, z2
    P3 = features[:, 14:17]  # x3, y3, z3

    # print("P1 shape:", P1.shape)  # should be (N, 3)
    # print("P1 dtype:", P1.dtype)  # should be float64

    # Computing the normal vector to the plane formed by the 3 points : P1, P2, P3
    a = P2 - P1
    b = P3 - P1
    normal = torch.linalg.cross(a, b, dim=1)

    # print("normal shape:", normal.shape)  # should be (N, 3)

    # Handling degenerate cases:
    norm_val = torch.linalg.norm(normal, dim=1, keepdim=True)
    degenerate = norm_val[:, 0] < 1e-10
    norm_val = torch.where(norm_val < 1e-10, torch.ones_like(norm_val), norm_val)
    normal = normal / norm_val

    # Building a linear system A @ center = b_vec to find the center:
    row1 = 2 * (P2 - P1)
    row2 = 2 * (P3 - P1)
    row3 = normal

    # print("row1 shape:", row1.shape)  # should be (N, 3)
    # print("row2 shape:", row2.shape)
    # print("row3 shape:", row3.shape)

    b_vec = torch.stack(
        [
            (P2**2).sum(dim=1) - (P1**2).sum(dim=1),
            (P3**2).sum(dim=1) - (P1**2).sum(dim=1),
            (normal * P1).sum(dim=1),
        ],
        dim=1,
    )

    # print("b_vec shape:", b_vec.shape)  # should be (N, 3)

    # Solving the 3x3 system in closed form (Cramer's rule via cross products) instead of
    # torch.linalg.pinv/torch.linalg.solve: those LAPACK-backed ops have no registered ONNX
    # export function (aten.linalg_pinv / aten.linalg_solve are not convertible), whereas this
    # closed-form solution only uses elementary tensor ops (cross product, sums, division),
    # which are exportable.
    #
    # For a 3x3 system with rows R0, R1, R2 (here row1, row2, row3) and right-hand side b_vec,
    # the solution is:
    #   det   = R0 . (R1 x R2)
    #   center = (b0 * (R1 x R2) + b1 * (R2 x R0) + b2 * (R0 x R1)) / det
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

    return torch.cat([center, radius, normal], dim=1)


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


class InferenceSeedDataset(Dataset):
    """
    Same as SeedDataset but WITHOUT labels, for use at inference time when no ground truth is
    available (e.g. inside SeedReconstructionModel.forward()).

    Args:
        X: Float tensor of seed features, shape [num_seeds, num_features]

    Returns:
        A dataset yielding only the feature vector for each seed (no label).
    """

    def __init__(self, X):
        self.X = X.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def seed_features_file_adjustment(data, batch_size=1000):
    """
    Used for TRAINING the Classifier
    preparing the dataset of reconstructed seeds (features + labels) we give to the Classifier by :
        - balancing the dataset
        - adding new features
        - turning it into a Dataloader

    Args:
    data: seed_features created by create_seed_features_file (must be a dict)

    Returns:
    Adjusted seed_features

    """

    X = data["features"]
    y = data["labels"].squeeze().long()

    X, y = balance_dataset(X, y)

    if X.dim() == 1:
        X = X.reshape(-1, 21)
        print("Shape after reshape :", X.shape)  # should be (N, 21)

    extra_tensor = circle_3_points_batch(X).float()
    X = torch.cat([X, extra_tensor], dim=1)  # (N, 28)

    Seed_Dataset = SeedDataset(X, y)

    Seed_dataloader = TorchDataLoader(dataset=Seed_Dataset, batch_size=batch_size, shuffle=False)

    return Seed_dataloader


def build_inference_seed_features(data):
    """
    Preparing the reconstructed seed features for the Classifier at INFERENCE time,
    when there is no ground-truth label available (e.g. inside SeedReconstructionModel.forward()).

    Unlike seed_features_file_adjustment / seed_features_inference_adjustment, this function does
    NOT use torch.utils.data.Dataset/DataLoader. The number of seeds N is a data-dependent/dynamic
    size at export time (it depends on how many valid seeds the beam search found at runtime), and
    Dataset/DataLoader rely on Python-level operations (len(), per-element indexing, batching loops)
    that require N to be a concrete, known integer. This breaks torch.export/ONNX export
    (GuardOnDataDependentSymNode / "Could not extract specialized integer from ... u_"). This
    function stays purely tensor-based so it can be traced.

    Args:
    data: dict with a single key "features", a Float tensor of shape
        [num_seeds, max_seed_size * num_base_features], e.g. built in forward() via
        build_seed_features_tensor(...).flatten(start_dim=1)

    Returns:
    X: Float tensor [num_seeds, 28], ready to be fed directly to the classifier MLP (see
    Classifier_architecture.run_classifier_tensor), in the same order as the input seeds.
    """

    X = data["features"]

    if X.dim() == 1:
        X = X.reshape(-1, 21)
        print("Shape after reshape :", X.shape)  # should be (N, 21)

    extra_tensor = circle_3_points_batch(X).float()
    X = torch.cat([X, extra_tensor], dim=1)  # [num_seeds, 28]

    return X
