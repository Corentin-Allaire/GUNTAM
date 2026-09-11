"""Inference macro for profiling SeedReconstructionModel with py-spy.

This script mirrors the space-point preprocessing path used by Read_ACTS_Csv,
builds one hit tensor per event (shape [N, 3] with columns x, y, z), and runs
SeedReconstructionModel inference on each event.

Example
-------
py-spy top -- python -m GUNTAM.Seed.InferReconstructionModel \
  --input-path tests/data \
  --checkpoint tests/data/transformer.pt \
  --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch

from GUNTAM.IO.Read_ACTS_Csv import (
    _create_particle_id_column,
    _process_hits_data,
    _process_particles_data,
    _process_space_points_data,
)
from GUNTAM.Seed.ClassifierConfig import ClassifierConfig
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedClassifier import SeedClassifier
from GUNTAM.Seed.SeedReconstructionModel import SeedReconstructionModel
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.IO.OnnxRunTime_Interface import onnx_providers


def _load_spacepoint_event_tensors(
    input_path: str,
    min_hits_per_particle: int,
    hit_range: tuple[float, float],
    max_events: int | None,
) -> list[torch.Tensor]:
    """
    Load per-event ACTS CSV files and build one raw xyz space-point tensor per event.

    Mirrors the space-point preprocessing path used by Read_ACTS_Csv: hits,
    particles, space points and the measurement-simhit map are read from disk,
    particles with too few hits are dropped, and hits are filtered to a
    cylindrical region before space points are reconstructed.

    Args:
        input_path: Directory containing `event*-hits.csv`,
            `event*-particles_selected.csv`, `event*-spacepoint.csv` and
            `event*-measurement-simhit-map.csv` files.
        min_hits_per_particle: Minimum number of hits a particle must have to
            be kept; particles below this threshold are treated as noise.
        hit_range: Tuple `(R_max, Z_max)` giving the cylindrical radius and
            absolute z bounds used to select hits during preprocessing.
        max_events: Optional cap on the number of events to load. If None,
            all complete event file sets found in `input_path` are used.

    Returns:
        List of float32 tensors, one per event, each of shape [N, 3] with
        columns x, y, z for that event's reconstructed space points.
    """
    hit_files = sorted(glob.glob(os.path.join(input_path, "event*-hits.csv")))
    particle_files = sorted(glob.glob(os.path.join(input_path, "event*-particles_selected.csv")))
    spacepoint_files = sorted(glob.glob(os.path.join(input_path, "event*-spacepoint.csv")))
    measurement_map_files = sorted(glob.glob(os.path.join(input_path, "event*-measurement-simhit-map.csv")))

    n_events = min(len(hit_files), len(particle_files), len(spacepoint_files), len(measurement_map_files))
    if n_events == 0:
        raise RuntimeError(f"No complete event file sets found in: {input_path}")

    if max_events is not None:
        n_events = min(n_events, max_events)

    event_tensors: list[torch.Tensor] = []

    for event_idx in range(n_events):
        hits = pd.read_csv(hit_files[event_idx], dtype={"geometry_id": np.int64})
        particles = pd.read_csv(particle_files[event_idx], dtype={"particle_id": np.int64})
        space_points = pd.read_csv(spacepoint_files[event_idx])
        measurement_map = pd.read_csv(measurement_map_files[event_idx])

        _create_particle_id_column(hits, particles)

        particle_counts = hits["particle_id"].value_counts()
        valid_particle_ids = particle_counts[particle_counts >= min_hits_per_particle].index

        hits = _process_hits_data(hits, R_max=hit_range[0], Z_max=hit_range[1])
        particles = _process_particles_data(particles, valid_particle_ids)

        hits["event_id"] = event_idx
        particles["event_id"] = event_idx

        kept_particle_ids = set(particles["particle_id"].values)
        mask_missing_particle = hits["particle_id"].isna() | ~hits["particle_id"].isin(kept_particle_ids)
        hits.loc[mask_missing_particle, "particle_id"] = -1

        processed_space_points = _process_space_points_data(space_points, measurement_map, hits)
        xyz_np = processed_space_points[["x", "y", "z"]].dropna().to_numpy(dtype=np.float32)

        event_tensor = torch.from_numpy(xyz_np)
        event_tensors.append(event_tensor)

    return event_tensors


def _build_reconstruction_model(
    checkpoint: str,
    config_path: str | None,
    device: torch.device,
    width: int,
    max_seed_length: int,
    classifier_checkpoint: str | None = None,
    classifier_config_path: str | None = None,
) -> SeedReconstructionModel:
    """
    Load a trained SeedTransformer checkpoint wrapped in a SeedReconstructionModel.

    Args:
        checkpoint: Path to the trained SeedTransformer checkpoint file (.pt).
        config_path: Optional path to a SeedConfig JSON file. If None, the
            default SeedConfig is used before being overwritten by the
            checkpoint's own transformer config.
        device: Device the transformer and reconstruction model are moved to.
        width: Top-k width used by SeedReconstructionModel during inference.
        max_seed_length: Maximum number of hits per reconstructed seed.
        classifier_checkpoint: Optional path to a SeedClassifier checkpoint (.pt).
            If provided, a SeedClassifier is loaded and attached to the model.
        classifier_config_path: Optional path to a ClassifierConfig JSON file,
            used to build the classifier before loading its checkpoint.

    Returns:
        A SeedReconstructionModel in eval mode, moved to `device`, wrapping the
        loaded SeedTransformer and, if requested, the loaded SeedClassifier.
    """
    cfg = SeedConfig()
    if config_path is not None:
        cfg.load_config(config_path)

    transformer = SeedTransformer(transformer_config=cfg.transformer_config, device_acc=device)
    transformer.load(checkpoint, device=device)
    transformer.eval()

    cfg.transformer_config = transformer.cfg

    classifier = None
    if classifier_checkpoint is not None:
        classifier_cfg = ClassifierConfig()
        if classifier_config_path is not None:
            classifier_cfg.load_config(classifier_config_path)
        classifier = SeedClassifier(classifier_config=classifier_cfg, device_acc=device)
        classifier.load(classifier_checkpoint, device=device)
        classifier.eval()

    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=transformer,
        classifier=classifier,
        device_acc=device,
        width=width,
        max_seed_length=max_seed_length,
    )
    model.to(device)
    model.eval()
    return model


def _build_onnx_reconstruction_model(
    config_path: str | None,
    device: torch.device,
    width: int,
    max_seed_length: int,
) -> SeedReconstructionModel:
    """
    Build a SeedReconstructionModel used only to drive `run_onnx_inference()`.

    No checkpoint is loaded here: binning and beam-search reconstruction (the
    only PyTorch-side logic `run_onnx_inference` relies on) only depend on
    `cfg.preprocessing_config`, `width` and `max_seed_length`, not on the
    transformer/classifier weights.

    Args:
        config_path: Optional path to a SeedConfig JSON file. If None, the
            default SeedConfig is used.
        device: Device to run ONNX Runtime and the PyTorch-side steps on.
        width: Top-k width used by SeedReconstructionModel during inference.
        max_seed_length: Maximum number of hits per reconstructed seed.

    Returns:
        A SeedReconstructionModel in eval mode, moved to `device`.
    """
    cfg = SeedConfig()
    if config_path is not None:
        cfg.load_config(config_path)

    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=SeedTransformer(transformer_config=cfg.transformer_config, device_acc=device),
        classifier=None,
        device_acc=device,
        width=width,
        max_seed_length=max_seed_length,
    )
    model.to(device)
    model.eval()
    return model


def _benchmark_pytorch(
    model: SeedReconstructionModel,
    events: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    """
    Run warmup + timed PyTorch inference over all events and return summary stats.

    Args:
        model: SeedReconstructionModel to benchmark, already moved to `device`.
        events: List of per-event hit tensors (shape [N, 3]) to run inference on.
        device: Device inference is run on; used to trigger the appropriate
            synchronization primitive before timing each run.
        warmup: Number of untimed passes over all events run before timing,
            used to stabilize JIT/kernel caches before measurement.
        repeat: Number of timed passes over all events used to compute the
            mean runtime.

    Returns:
        Dict with keys:
            events: Number of events processed per timed run.
            seeds: Number of seeds reconstructed in the last timed run.
            mean_time: Mean wall-clock time (seconds) per timed run over all events.
            event_rate: Mean event throughput (events/second).
    """
    with torch.no_grad():
        for _ in range(warmup):
            for event in events:
                model(event.to(device))

    timed_runs: list[float] = []
    total_events = len(events)
    total_seeds = 0

    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            run_seed_count = 0
            for event in events:
                seeds, _ = model(event.to(device))
                run_seed_count += int(seeds.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            timed_runs.append(elapsed)
            total_seeds = run_seed_count

    mean_time = sum(timed_runs) / len(timed_runs)
    event_rate = total_events / mean_time if mean_time > 0 else 0.0

    return {
        "events": total_events,
        "seeds": total_seeds,
        "mean_time": mean_time,
        "event_rate": event_rate,
    }


def _benchmark_parallel_pytorch(
    model: SeedReconstructionModel,
    events: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    nb_workers: int = 4,
) -> dict[str, Any]:
    """
    Run warmup, then `nb_workers` concurrent workers over all events and measure wall-clock throughput.

    Each worker independently runs inference over the full `events` list on a
    shared thread pool, so this measures aggregate throughput under
    concurrent load rather than single-stream latency.

    Args:
        model: SeedReconstructionModel to benchmark, already moved to `device`.
        events: List of per-event hit tensors (shape [N, 3]) to run inference on.
        device: Device inference is run on; used to trigger the appropriate
            synchronization primitive before timing.
        warmup: Number of untimed passes over all events run before timing,
            used to stabilize JIT/kernel caches before measurement.
        nb_workers: Number of concurrent worker threads, each processing the
            full `events` list once.

    Returns:
        Dict with keys:
            events: Total number of events processed across all workers.
            seeds: Total number of seeds reconstructed across all workers.
            mean_time: Wall-clock time (seconds) for all workers to complete.
            event_rate: Aggregate event throughput (events/second).
    """
    with torch.no_grad():
        for _ in range(warmup):
            for event in events:
                model(event.to(device))

    def _run_worker() -> int:
        seed_count = 0
        with torch.no_grad():
            for event in events:
                seeds, _ = model(event.to(device))
                seed_count += int(seeds.shape[0])
        return seed_count

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        futures = [executor.submit(_run_worker) for _ in range(nb_workers)]
        seed_counts = [future.result() for future in futures]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    total_events = len(events) * nb_workers
    total_seeds = sum(seed_counts)
    event_rate = total_events / elapsed if elapsed > 0 else 0.0

    return {
        "events": total_events,
        "seeds": total_seeds,
        "mean_time": elapsed,
        "event_rate": event_rate,
    }


def _benchmark_onnx(
    model: SeedReconstructionModel,
    events: list[torch.Tensor],
    transformer_path: str,
    classifier_path: str | None,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    """
    Run warmup + timed ONNX Runtime inference over all events, via
    `SeedReconstructionModel.run_onnx_inference`, and return summary stats.

    Args:
        model: SeedReconstructionModel used to drive `run_onnx_inference`
            (binning and beam-search reconstruction run in PyTorch on `device`).
        events: List of per-event hit tensors (shape [N, 3]) to run inference on.
        transformer_path: Path to the exported transformer ONNX model.
        classifier_path: Path to the exported classifier ONNX model, or None to
            skip classification and keep every reconstructed seed.
        device: Device inference is run on; used to trigger the appropriate
            synchronization primitive before timing each run.
        warmup: Number of untimed passes over all events run before timing,
            used to stabilize the runtime before measurement.
        repeat: Number of timed passes over all events used to compute the
            mean runtime.

    Returns:
        Dict with keys:
            events: Number of events processed per timed run.
            seeds: Number of seeds reconstructed in the last timed run.
            mean_time: Mean wall-clock time (seconds) per timed run over all events.
            event_rate: Mean event throughput (events/second).
    """
    with torch.no_grad():
        for _ in range(warmup):
            for event in events:
                model.run_onnx_inference(event.to(device), transformer_path, classifier_path, device=device)

    timed_runs: list[float] = []
    total_events = len(events)
    total_seeds = 0

    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            run_seed_count = 0
            for event in events:
                print(f"Running ONNX inference for event with {event.shape[0]} hits...")
                seeds, _ = model.run_onnx_inference(event.to(device), transformer_path, classifier_path, device=device)
                run_seed_count += int(seeds.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            timed_runs.append(elapsed)
            total_seeds = run_seed_count

    mean_time = sum(timed_runs) / len(timed_runs)
    event_rate = total_events / mean_time if mean_time > 0 else 0.0

    return {
        "events": total_events,
        "seeds": total_seeds,
        "mean_time": mean_time,
        "event_rate": event_rate,
    }


def main() -> None:

    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser(description="Run SeedReconstructionModel inference event-by-event for profiling.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="tests/data",
        help="Directory containing event*-hits/particles/spacepoint/measurement CSV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tests/data/transformer.pt",
        help="Path to the trained SeedTransformer checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional SeedConfig JSON file. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a SeedClassifier checkpoint (.pt). If provided, the classifier is "
            "attached to the model and used to filter reconstructed seeds."
        ),
    )
    parser.add_argument(
        "--classifier-config",
        type=str,
        default=None,
        help="Optional ClassifierConfig JSON file used to build the classifier before loading its checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="Top-k width used by SeedReconstructionModel.",
    )
    parser.add_argument(
        "--max-seed-length",
        type=int,
        default=3,
        help="Maximum seed length used by reconstruction.",
    )
    parser.add_argument(
        "--min-hits-per-particle",
        type=int,
        default=9,
        help="Minimum number of hits required to keep a particle (same as Read_ACTS_Csv default).",
    )
    parser.add_argument(
        "--hit-range",
        nargs=2,
        type=float,
        default=[500.0, 1000.0],
        metavar=("R_MAX", "Z_MAX"),
        help="Spatial hit selection limits [R_max, Z_max] used during preprocessing.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to process.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup inference passes over all events.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timed inference passes over all events.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["pytorch", "parallel_pytorch", "onnx", "both"],
        help="Inference backend to benchmark: pytorch, onnx, or both.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the reconstruction model (PyTorch 2.x).",
    )
    parser.add_argument(
        "--transformer-onnx",
        type=str,
        default="Work/transformer.onnx",
        help="Path to the exported transformer ONNX model, used for ONNX backend benchmarking.",
    )
    parser.add_argument(
        "--classifier-onnx",
        type=str,
        default=None,
        help=(
            "Optional path to the exported classifier ONNX model. If omitted, every "
            "reconstructed seed is kept (no classification step) for the ONNX backend."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    hit_range = (args.hit_range[0], args.hit_range[1])

    print("Loading and preprocessing space points...")
    event_tensors = _load_spacepoint_event_tensors(
        input_path=args.input_path,
        min_hits_per_particle=args.min_hits_per_particle,
        hit_range=hit_range,
        max_events=args.max_events,
    )

    non_empty_events = [t for t in event_tensors if t.shape[0] > 0]
    if not non_empty_events:
        raise RuntimeError("All selected events are empty after preprocessing.")

    print(f"Prepared {len(non_empty_events)} events.")
    print(f"Average space points per event: {sum(t.shape[0] for t in non_empty_events) / len(non_empty_events):.1f}")

    results: dict[str, dict[str, Any]] = {}

    if args.backend in ("pytorch", "both"):
        model = _build_reconstruction_model(
            checkpoint=args.checkpoint,
            config_path=args.config,
            device=device,
            width=args.width,
            max_seed_length=args.max_seed_length,
            classifier_checkpoint=args.classifier_checkpoint,
            classifier_config_path=args.classifier_config,
        )
        model.eval()
        if args.compile:
            print("Compiling PyTorch model with torch.compile...")
            model = torch.compile(model)  # type: ignore[assignment]

        results["pytorch"] = _benchmark_pytorch(
            model=model,
            events=non_empty_events,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    if args.backend in ("onnx", "both"):
        if not os.path.exists(args.transformer_onnx):
            raise FileNotFoundError(
                f"Transformer ONNX model not found: {args.transformer_onnx}. Provide --transformer-onnx or export first."
            )
        if args.classifier_onnx is not None and not os.path.exists(args.classifier_onnx):
            raise FileNotFoundError(f"Classifier ONNX model not found: {args.classifier_onnx}.")

        onnx_model = _build_onnx_reconstruction_model(
            config_path=args.config,
            device=device,
            width=args.width,
            max_seed_length=args.max_seed_length,
        )
        providers = onnx_providers(device)
        print(f"ONNX Runtime providers: {providers}")

        results["onnx"] = _benchmark_onnx(
            model=onnx_model,
            events=non_empty_events,
            transformer_path=args.transformer_onnx,
            classifier_path=args.classifier_onnx,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    if args.backend == "parallel_pytorch":
        model = _build_reconstruction_model(
            checkpoint=args.checkpoint,
            config_path=args.config,
            device=device,
            width=args.width,
            max_seed_length=args.max_seed_length,
            classifier_checkpoint=args.classifier_checkpoint,
            classifier_config_path=args.classifier_config,
        )
        model.eval()
        if args.compile:
            print("Compiling PyTorch model with torch.compile...")
            model = torch.compile(model)  # type: ignore[assignment]
        results["parallel_pytorch"] = _benchmark_parallel_pytorch(
            model=model,
            events=non_empty_events,
            device=device,
            warmup=args.warmup,
            nb_workers=args.repeat,
        )

    print("Inference completed.")
    for backend_name, stats in results.items():
        print(f"[{backend_name}] Events per timed run: {stats['events']}")
        print(f"[{backend_name}] Seeds reconstructed (last run): {stats['seeds']}")
        print(f"[{backend_name}] Mean runtime per run: {stats['mean_time']:.6f} s")
        print(f"[{backend_name}] Event throughput: {stats['event_rate']:.2f} events/s")

    if "pytorch" in results:
        print(f"[pytorch] Compiled: {args.compile}")


if __name__ == "__main__":
    main()
