import torch
import pytest
from GUNTAM.Seed.SeedTransformer_JUSTINE import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.Validation_loss_function import validate_function


def test_validation_function(path="/homeijclab/lesecq/stageM1/GUNTAM/tests/data/", dataset_name="event000000001-hits.csv", model_name="transformer.pt"):

    cfg = SeedConfig()
    cfg.parse_args(argv=[])
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

    file_indices=list(range(len(dataset.file_paths)))

    with pytest.raises(ValueError):
        validate_function(model, file_indices, dataset, cfg, shuffle_v=4, situation="xyz")

    assert round(validate_function(model=model, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg)) == 50

