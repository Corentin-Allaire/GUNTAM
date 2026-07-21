# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """""""""""""""""""""""""""""""""""""" EFFICIENCY CLASS """"""""""""""""""""""""""""""""""""""""""""""""""""""""
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch.nn as nn
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Transformer.Utils import ts_print
from GUNTAM.Seed.Efficiency_seed_function import efficiency_reconstructed_seeds
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.IO.DataLoader import DataLoader


class Efficiency_class(nn.Module):
    """
    built a class to call the efficiency_reconstructed_seeds function to speed up the process in the main part

    Args:
        model: The transformer model to be validated.
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.

    Returns:
        Seeding efficiency for the dataset with the feature i that has been shuffle.

    """

    def __init__(self, model: SeedTransformer, dataset: DataLoader, cfg=SeedConfig):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

    def seed_eff_class(self, i: int):
        ts_print("i :", i)
        return efficiency_reconstructed_seeds(
            model=self.model,
            dataset=self.dataset,
            file_indices=list(range(len(self.dataset.file_paths))),
            shuffle_v=i,
            cfg=self.cfg,
        )
