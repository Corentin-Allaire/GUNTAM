#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#"""""""""""""""""""""""""""" Validation """"""""""""""""""""""""""""""""""""""""""""""""
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import random
import sys
import os
import math
import torch
import time
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from GUNTAM.Seed.SeedTransformer import SeedTransformer
import GUNTAM.Seed.SeedLoss as Losses
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Transformer.Utils import ts_print
import GUNTAM.Transformer.Utils as Utils
import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor
from GUNTAM.IO.PrepareTensor import compute_barcode, prepare_tensor


def initialize_loss_dictionary(active_components: list, device: torch.device) -> Dict[str, torch.Tensor]:
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


def validate_rapide_model(
    model: SeedTransformer,
    file_indices: list,
    nb_events,
    dataset,
    cfg: SeedConfig,
):
    """
    Validation uniquement sur toutes les données.
    Pas d'entraînement, pas de backward, pas de split train/val.
    """

    ts_print("Starting validation on all available data")

    model.eval()
    model_dtype = model.dtype

    global_losses = []
    

    ts_print("start no_grad")

    with torch.no_grad():
        ts_print("start file")
        liste_all_files = []
        nb_file = 0
        for file_idx in file_indices:

            ts_print("start get")
            data = dataset.get_file(file_idx)

            hits_tensor = data["hits_tensor"].to(cfg.device_acc, dtype=model_dtype)
            particles_tensor = data["particles_tensor"].to(cfg.device_acc, dtype=model_dtype)
            hit_to_particle_tensor = data["hit_to_particle_tensor"].to(cfg.device_acc)
            padding_mask = data["padding_mask"].to(cfg.device_acc)
            good_pairs = data["good_pairs"].to(cfg.device_acc)

            num_events = hits_tensor.shape[0] # = 5
            ts_print("finish get")

            nb_total_events = 0
            liste_event_file =[]

            ts_print("start for") # ici que c'est long !!!!!
            for event_idx in range(num_events): #pour chaque event
                
                event_hits_tensor = hits_tensor[event_idx]
                event_good_pairs = good_pairs[event_idx]
                event_padding_mask = padding_mask[event_idx]
                event_hit_to_particle_tensor = hit_to_particle_tensor[event_idx]
                event_particle_tensor = particles_tensor[event_idx]

                valid_bins = torch.where(event_good_pairs.sum(dim=(1, 2)) > 0)[0].tolist()
                #indices des bins qui contiennent au moins une paire valide (on trie les bins non utiles)
                
                #il faut que j'enlève batch_size dans range et ligne 123 et que je modifie lignes 130 à 134
                #l'algo est très très lent, on essaye le rendre + rapide
                #Pour aller + vite, on peut directement donner toutes les données au transformer (on use plus de batch)
                #en utilisant des batchs, on donne au modèle un bin à la fois
                #en enlevant la notion de batch, on donne au modèle un évenements à la fois
                #la notion de bins ne sert plus à rien ici aussi

                
                event_hits = event_hits_tensor[valid_bins] #bon hits
                # event_hits_tensor = tous les hits
                event_masks = event_padding_mask[valid_bins]
                event_pairs = event_good_pairs[valid_bins]
                event_hit_to_particle_indices = event_hit_to_particle_tensor[valid_bins].squeeze(-1)
                event_particles = event_particle_tensor[event_hit_to_particle_indices]

                event_loss = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)
                encoded_space_points, attention_maps = model(event_hits, event_masks)

                #ts_print("start if") PAS LUIIIIIIII
                if cfg.transformer_config.regression and cfg.has_loss_component("hit_BCE"):
                    hits_score = encoded_space_points
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
                    event_loss["attention_next"] += Losses.attention_next_loss(
                        attention_map_bin, pairs1, pairs2, target
                    )

                liste_event_file.append(event_loss["attention_next"].item())
                print(liste_event_file)
                print(f"[Validation] Event {nb_total_events} - loss = {event_loss["attention_next"].item():.6f}")
                nb_total_events += 1

            liste_all_files.append(liste_event_file) #liste de liste
            print(liste_all_files)
            # for key, value in event_loss.items():
            #     if key == "total":
            #         continue
            #     event_loss["total"] += value
        liste_all_files_applatie = [x for sous_liste in liste_all_files for x in sous_liste]
        print(liste_all_files_applatie)
        #global_losses.append(event_loss["total"].item())
        ts_print("finish for")
        ts_print("finish file")
    ts_print("finish no_grad")

    
    avg_loss = sum(liste_all_files_applatie) / len(liste_all_files_applatie) 

    ts_print(f"Validation finished on {nb_file} files")
    ts_print(f"Average validation loss on all data: {avg_loss:.6f}")

    return avg_loss, liste_all_files_applatie


