# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# """"""""""""""""""""""""""""""""""""""""""" SEED RECONSTRUCTION """"""""""""""""""""""""""""""""""""""""""""""""
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from GUNTAM.Transformer.Utils import ts_print
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.Reconstruction import batched_beam_search_seed_reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor


def efficiency_reconstructed_seeds(
    model: SeedTransformer,
    file_indices: list,
    dataset: DataLoader,
    cfg: SeedConfig,
    *,
    shuffle_v: Optional[int] = None,
    situation: Optional[str] = None,
) -> float:
    """

    We reconstruct the seeds once they've been through the transformer (we have the attention matrices)

    Args:
        model: The transformer model to be validated.
        file_indices: List of indices indexing the files we use.
        dataset: The dataset object containing trained data.
        shuffle_v: Indice of the feature we shuffle. The signification test is done on this feature.
        cfg: Full architecture configuration.

    Returns:
        Seeding efficiency.

    """

    if situation is not None and shuffle_v is not None:
        raise ValueError("`situation` or `shuffle_v` are not well defined")

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

                if situation is not None and shuffle_v is None:
                    encoded_hits = model.embedding(hits=event_hits_tensor, situation=situation)
                    _, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_padding_mask)
                    _, triplets = model(hits=event_hits_tensor, mask=event_padding_mask, width=5, situation=situation)
                if situation is None and shuffle_v is not None:
                    encoded_hits = model.embedding(hits=event_hits_tensor, shuffle_v=shuffle_v)
                    _, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_padding_mask)
                    _, triplets = model(hits=event_hits_tensor, mask=event_padding_mask, width=5, shuffle_v=shuffle_v)
                if situation is None and shuffle_v is None:
                    encoded_hits = model.embedding(hits=event_hits_tensor)
                    _, attention_maps = model.compute_adjacency(encoded_hits=encoded_hits, mask=event_padding_mask)
                    _, triplets = model(
                        hits=event_hits_tensor, mask=event_padding_mask, width=5
                    )  # seeding efficiency of reference

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
                    beam_width=3,
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
