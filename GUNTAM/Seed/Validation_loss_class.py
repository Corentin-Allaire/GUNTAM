# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""" VALIDATION CLASS """""""""""""""""""""""""""""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch.nn as nn
from GUNTAM.Seed.Config import SeedConfig
from Validation_loss_function import validate_function
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.IO.DataLoader import DataLoader


class Validation_class(nn.Module):
    """
    built a class to call the validation function in order to speed up the process in the main part

    Args:
        model: The transformer model to be validated. 
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.

    Returns:
        average loss for the dataset with the feature i that has been shuffle.

    """

    def __init__(self, model: SeedTransformer, dataset: DataLoader, cfg=SeedConfig):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

    def validate_class(self, i: int):
        return validate_function(
            model=self.model,
            dataset=self.dataset,
            file_indices=list(range(len(self.dataset.file_paths))),
            cfg=self.cfg,
            shuffle_v=i

        )
