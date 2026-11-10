from GUNTAM.Monitoring.PermutationMetricEvaluator import validate_function
from GUNTAM.Monitoring.PermutationImportances import config_model_dataset


import pathlib

data_path = pathlib.Path(__file__).parents[1] / "data" / "odd_output_new_5"
model_path = pathlib.Path(__file__).parents[1] / "data" / "transformer.pt"


def test_validation_function(data_path=data_path, dataset_name="odd_output_new_5", model_path=model_path):

    cfg, dataset, transformer = config_model_dataset(
        path=data_path,
        dataset_name=dataset_name,
        model_name=model_path,
    )

    result = validate_function(model=transformer, file_indices=list(range(len(dataset.file_paths))), dataset=dataset, cfg=cfg)

<<<<<<< HEAD
    assert round(result) == 1005
=======
    assert round(result) == 44
>>>>>>> b4e1cc5 (fix issue with attention not return logit in training)
