# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""""" VALIDATION FUNCTION """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import torch
from typing import Optional
import torch.nn as nn
from GUNTAM.Seed.SeedTransformer import SeedTransformer
import GUNTAM.Seed.SeedLoss as Losses
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader


def initialize_loss_dictionary(active_components: list, device: torch.device):
    """
    Initialize a loss dictionary with zero values for active loss components.

    Args:
        active_components: List of active loss component names.
        device: Torch device for tensor initialization.

    Returns:
        Initialized loss dictionary with zero values.
    """

    # Helper to add a key lazily
    def add_loss_key(key: str):
        if key not in event_loss:
            event_loss[key] = torch.tensor(0.0, device=device)

    # Initialize per-event losses dynamically based on active loss components
    event_loss = {"total": torch.tensor(0.0, device=device)}

    # Attention variants
    if "attention" in active_components:
        add_loss_key("attention")
    if "topk_attention" in active_components:
        add_loss_key("topk_attention")
    if "full_attention" in active_components:
        add_loss_key("full_attention")
    if "attention_next" in active_components:
        add_loss_key("attention_next")
    if "attention_back" in active_components:
        add_loss_key("attention_back")

    # Classification losses
    if "hit_BCE" in active_components:
        add_loss_key("hit_BCE")

    return event_loss


def validate_function(
    model: SeedTransformer,
    file_indices: list,
    dataset: DataLoader,
    cfg: SeedConfig,
    *,
    shuffle_v: Optional[int] = None,
    situation: Optional[str] = None,
) -> float:
    """
    Validation on all datas (no batch, no bin).

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.
        shuffle_v: Indice of the feature we shuffle.
        situation: name of the situation corresponding to which features we want to shuffle together.

    Returns:
        Average loss.

    """

    if situation is not None and shuffle_v is not None:
        raise ValueError("`situation` or `shuffle_v` are not well defined")

    torch.set_num_threads(1)

    model.eval()
    model_dtype = model.dtype

    with torch.no_grad():
        liste_all_files: list = []

        # We work on each file at a time:

        for file_idx in file_indices:
            data = dataset.get_file(file_idx)

            # We define all the different information that are in the dataset:

            hits_tensor = data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = data["particles_tensor"].to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = data["hit_to_particle_tensor"].to(cfg.device_acc)
            padding_mask = data["padding_mask"].to(cfg.device_acc)
            good_pairs = data["good_pairs"].to(cfg.device_acc)

            num_events = hits_tensor.shape[0]

            nb_total_events = 0
            liste_event_file = []

            # We work on each event at a time:

            for event_idx in range(num_events):

                # We define all the information above for one event:

                event_hits_tensor = hits_tensor[event_idx]
                event_good_pairs = good_pairs[event_idx]
                event_padding_mask = padding_mask[event_idx]
                event_hit_to_particle_tensor = hit_to_particle_tensor[event_idx]
                event_particle_tensor = particles_tensor[event_idx]
                valid_bins = torch.where(event_good_pairs.sum(dim=(1, 2)) > 0)[0].tolist()

                event_hits = event_hits_tensor[valid_bins]
                event_masks = event_padding_mask[valid_bins]
                event_hit_to_particle_indices = event_hit_to_particle_tensor[valid_bins].squeeze(-1)
                event_particles = event_particle_tensor[event_hit_to_particle_indices]
                event_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)
                if situation is not None and shuffle_v is None:
                    encoded_hits = model.embedding(hits=event_hits, situation=situation)
                    transformer_output, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_masks)
                if situation is None and shuffle_v is not None:
                    encoded_hits = model.embedding(hits=event_hits, shuffle_v=shuffle_v)
                    transformer_output, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_masks)
                if situation is None and shuffle_v is None:
                    encoded_hits = model.embedding(hits=event_hits)
                    transformer_output, attention_maps = model.compute_adjacency(
                        encoded_hits=encoded_hits, mask=event_masks
                    )  # average loss of reference

                # We define which loss type we want to use:

                if cfg.transformer_config.regression and cfg.has_loss_component("hit_BCE"):
                    hits_score = transformer_output
                    event_loss["hit_BCE"] = Losses.hit_classification_loss(
                        hits_score,
                        event_particles,
                        event_masks,
                    )

                for idx_valid_bins, truc in enumerate(valid_bins):
                    pairs1, pairs2, target = event_good_pairs[idx_valid_bins].unbind(dim=1)

                    if target.sum() == 0:
                        continue

                    attention_map_bin = attention_maps[idx_valid_bins].squeeze(0)

                if cfg.has_loss_component("attention_next"):
                    event_loss["attention_next"] += Losses.attention_next_loss(attention_map_bin, pairs1, pairs2, target)

                liste_event_file.append(event_loss["attention_next"].item())  # list of losses for one file

                nb_total_events += 1

            liste_all_files.append(liste_event_file)  # list of list : list of losses for all the files

        liste_all_files_flatten = [x for sous_liste in liste_all_files for x in sous_liste]

    avg_loss = sum(liste_all_files_flatten) / len(liste_all_files_flatten)  # average loss for all the files

    return avg_loss


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
            shuffle_v=i,
        )
