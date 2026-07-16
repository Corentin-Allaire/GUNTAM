import torch
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.IO.prepare_classifier import (
    build_seed_features_tensor,
    balance_dataset,
    transformer_seed_reconstruction,
    transformer_loading,
)


def test_transformer_seed_reconstruction(
    path="/homeijclab/lesecq/stageM1/GUNTAM/tests/data/", dataset_name="event000000001-hits.csv", model_name="transformer.pt"
):
    """
    Testing if the transformer is built with the transformer class
    Testing if the reconstructed hits and seeds are tensors
    Testing if the file containing all of the reconstructed seeds is a tensor
    Testing if the seed features are well balanced between true and fake
    """
    cfg = SeedConfig()

    transformer = transformer_loading(transformer_name=model_name)

    tensor_list = {
        "hits_tensor",
        "particles_tensor",
        "hit_to_particle_tensor",
        "padding_mask",
        "good_pairs",
    }

    dataset = DataLoader(
        dataset_dir=path,
        dataset_name=dataset_name,
        tensor_names=list(tensor_list),
        device="cuda:0",
    )

    hits_tensor, seed_tensor = transformer_seed_reconstruction(
        model=transformer, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg
    )

    seed_features = build_seed_features_tensor(
        hits_tensor=hits_tensor, seed_tensor=seed_tensor, feature_indices=[0, 1, 2, 3, 4, 5], cosine_feature_indices=[4]
    )

    X = seed_features["features"]
    y = seed_features["labels"].squeeze().long()

    X, y = balance_dataset(X, y)

    assert isinstance(transformer, SeedTransformer)

    assert isinstance(hits_tensor, torch.Tensor)
    assert isinstance(hits_tensor, torch.Tensor)

    assert isinstance(seed_features, torch.Tensor)

    assert len(X) == len(y)
