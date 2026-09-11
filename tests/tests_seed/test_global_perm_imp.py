from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.PermutationMetricEvaluator import validate_function
from GUNTAM.Seed.Permutation_importances import config_model_dataset


import pathlib

data_path = pathlib.Path(__file__).parents[1] / "data" / "odd_output_new_5"
model_path = pathlib.Path(__file__).parents[1] / "data" / "transformer.pt"


def test_validation_function(data_path=data_path, dataset_name="odd_output_new_5", model_path=model_path):

    cfg = SeedConfig()

    dataset, transformer = config_model_dataset(
        path=data_path,
        dataset_name=dataset_name,
        model_name=model_path,
    )

    result = validate_function(model=transformer, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg)

    assert round(result) == 878
