# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""""" EVENT ENCODING """"""""""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import random
import torch
import numpy as np
from typing import List
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.Reconstruction import batched_beam_search_seed_reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor
import GUNTAM.Seed.SeedLoss as Losses


class PermutationMetricEvaluator:
    """
    Wraps a metric function (validate_function or efficiency_reconstructed_seeds) so it can be called
    per shuffled feature index via multiprocessing.Pool.map.
    """

    def __init__(self, metric_fn, model, dataset, cfg):
        self.metric_fn = metric_fn
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

    def __call__(self, i: int):
        return self.metric_fn(
            model=self.model,
            dataset=self.dataset,
            file_indices=list(range(len(self.dataset.file_paths))),
            cfg=self.cfg,
            features=[i],
        )


def prepare_event_encoding(
    model: SeedTransformer,
    event_hits: torch.Tensor,
    event_mask: torch.Tensor,
    features: List[int],
):
    """
    Shared preprocessing used by both the validation-loss and seeding-efficiency permutation-importance
    functions: embed the hits, optionally shuffle chosen features across hits, project them, then run
    the model's attention/adjacency computation.

    Args:
        model: The transformer model.
        event_hits: Hits tensor for a single event.
        event_mask: Padding mask for the same event.
        features: Indices of the Fourier-embedding features to shuffle (independently) across hits.

    Returns:
        Tuple of (encoded_hits, transformer_output, attention_maps).
    """

    encoded_hits = model.fourier_embedding(event_hits)
    for feature in features:
        idx = list(range(encoded_hits.size(1)))
        random.shuffle(idx)
        tensor_idx = torch.tensor(idx)
        encoded_hits[:, :, feature] = encoded_hits[:, tensor_idx, feature]
    encoded_hits = model.feature_projection(encoded_hits)

    transformer_output, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_mask)

    return encoded_hits, transformer_output, attention_maps


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""""" SEED RECONSTRUCTION """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def efficiency_reconstructed_seeds(
    model: SeedTransformer,
    file_indices: list,
    dataset: DataLoader,
    cfg: SeedConfig,
    features: List[int] = [],
) -> float:
    """

    We reconstruct the seeds once they've been through the transformer (we have the attention matrices)

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.
        features: List of indices of the features (last dimension) to shuffle.
        Each feature is shuffled with its own independent permutation.

    Returns:
        Seeding efficiency.

    """

    model.eval()
    model_dtype = model.dtype

    monitoring = PerformanceMonitor(
        full_print=False,
        save_plots=True,
        min_common_hits=3,
        min_truth_hits=3,
        truth_r_tol=1e-3,
    )

    with torch.no_grad():

        # We work on each file at a time:

        for file_idx in file_indices:
            data = dataset.get_file(file_idx)

            # We define all the different information that are in data:

            hits_tensor = data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = data["particles_tensor"].to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = data["hit_to_particle_tensor"].to(cfg.device_acc)
            padding_mask = data["padding_mask"].to(cfg.device_acc)

            num_events = hits_tensor.shape[0]  # = 5 for odd_output_new_5

            # We work on each event at a time:

            for event_idx in range(num_events):

                # We define all the information above for one event:
                event_hits_tensor = hits_tensor[event_idx]
                event_padding_mask = padding_mask[event_idx]
                event_hit_to_particle_tensor = hit_to_particle_tensor[event_idx]
                event_particle_tensor = particles_tensor[event_idx]

                # We shuffle the features in the Fourier embedding for the current event:
                encoded_hits, _, attention_maps = prepare_event_encoding(
                    model=model, event_hits=event_hits_tensor, event_mask=event_padding_mask, features=features
                )
                _ = model.triplet_extraction(attention_maps, width=5)

                if cfg.transformer_config.regression:
                    hits_score = encoded_hits

                else:
                    # Compute hit score as the row-wise max of the attention weights [bins, hits, 1]
                    hits_score = attention_maps.squeeze(1).max(dim=-1).values.unsqueeze(-1)

                # We reconstruct the hits:

                chains, params, scores = batched_beam_search_seed_reconstruction(
                    attention_edge=attention_maps,
                    valid_mask=~event_padding_mask.bool(),
                    att_threshold=0.0,
                    max_chain_length=5,
                    beam_width=5,
                    backward=False,
                )

                event_seeds = []
                event_hit_scores = []
                event_attention_maps = []

                # Transfer the result to CPU and efficiency analysis. This is excluded from timing computation.
                hit_chains_all = chains.cpu().numpy().astype(np.int64)  # [B, N, ML]
                scores_all = scores.cpu().numpy()  # [B, N]
                params_all = params.cpu().numpy()  # [B, N, F]
                attention_softmax = torch.softmax(attention_maps.squeeze(1), dim=-1)  # [B, N, N]
                attention_softmax_cpu = attention_softmax.cpu().numpy()
                hit_score_all = hits_score.squeeze(-1).cpu().detach().float().numpy()  # [B, N]

                for bin_idx in range(event_hits_tensor.shape[0]):
                    hit_chains_np = hit_chains_all[bin_idx]  # [N, ML]
                    scores_np = scores_all[bin_idx]  # [N]
                    params_np = params_all[bin_idx]  # [N, F]
                    bin_seeds = []
                    seen_bs: set = set()
                    lengths = (hit_chains_np >= 0).sum(axis=1)  # [N] — vectorized length per chain
                    prefilter = np.isfinite(scores_np) & (scores_np > 0.3)
                    for i in np.where(prefilter)[0]:
                        chain_compact = hit_chains_np[i, : lengths[i]]
                        key = tuple(sorted(chain_compact.tolist()))
                        if key in seen_bs:
                            continue
                        seen_bs.add(key)
                        bin_seeds.append((chain_compact, params_np[i], scores_np[i]))

                    event_attention_maps.append(attention_softmax_cpu[bin_idx])
                    event_seeds.append(bin_seeds)
                    event_hit_scores.append(hit_score_all[bin_idx])

                monitoring.bin_seeding_performance(
                    event_idx=event_idx,
                    event_hits=event_hits_tensor.cpu().float().numpy(),
                    event_particles=event_particle_tensor.cpu().float().numpy(),
                    event_hit_to_particle=event_hit_to_particle_tensor.cpu().float().numpy(),
                    event_seeds=event_seeds,
                )

    performance_results = monitoring.performance_analysis()

    efficiency = performance_results["efficiency_metrics"]["seeding_efficiency"]

    return efficiency


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""""" VALIDATION FUNCTION """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


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
    features: List[int] = [],
) -> float:
    """
    Validation on all datas (no batch, no bin).

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data.
        cfg: Full architecture configuration.
        features: name of the features corresponding to which features we want to shuffle together.

    Returns:
        Average loss.

    """
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

                # We shuffle the features in the Fourier embedding for the current event:
                encoded_hits, transformer_output, attention_maps = prepare_event_encoding(
                    model=model, event_hits=event_hits, event_mask=event_masks, features=features
                )

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
