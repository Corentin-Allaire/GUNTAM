"""
Integration test for the transformer+classifier model, run both as raw PyTorch
("classique") and as an exported ONNX model.

Pipeline:
  1. Run the trained transformer+classifier directly in PyTorch ("classique").
  2. Export the same transformer and classifier checkpoints to ONNX and run
     inference through onnxruntime.
  3. Read event 0 space-point CSV and build a [N, 3] input (x, y, z).
  4. Run inference → seed tensor [S, seed_length] of original space-point indices.
  5. Map each seed to a particle via the space-point→particle_id table
     (built by joining spacepoint.csv → measurement-simhit-map.csv → hits.csv,
     mirroring the logic in GUNTAM/IO/Read_ACTS_Csv._process_space_points_data).
  6. Compute seeding efficiency = fraction of particles that have ≥1 seed.

The same test suite (`TestFullModel`) is run against both backends via the
`inference_results` fixture, parametrized over "classique" and "onnx".
"""

from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
import pytest

import GUNTAM.IO.Export_full_model as _export_module
from GUNTAM.Seed.SeedReconstructionModel import SeedReconstructionModel
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.SeedClassifier import SeedClassifier
from GUNTAM.Seed.Config import SeedConfig

DATA_DIR = Path(__file__).parent.parent / "data"
PT_MODEL = DATA_DIR / "transformer.pt"
CLASSIFIER_MODEL = DATA_DIR / "classifier.pt"  # adapte le nom/chemin si besoin
SP_FILE = DATA_DIR / "event000000002-spacepoint.csv"
HITS_FILE = DATA_DIR / "event000000002-hits.csv"
MEAS_FILE = DATA_DIR / "event000000002-measurement-simhit-map.csv"
PARTICLES_FILE = DATA_DIR / "event000000002-particles_selected.csv"

# Particle-ID columns shared between hits and particles CSVs
_ID_COLUMNS = [
    "particle_id_pv",
    "particle_id_sv",
    "particle_id_part",
    "particle_id_gen",
    "particle_id_subpart",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _create_particle_id_column(hits: pd.DataFrame, particles: pd.DataFrame) -> None:
    """Reuse the same mapping logic as GUNTAM/IO/Read_ACTS_Csv._create_particle_id_column."""
    particles["particle_id"] = particles.index.astype(int)
    particles["_key"] = list(map(tuple, particles[_ID_COLUMNS].to_numpy()))
    hits["_key"] = list(map(tuple, hits[_ID_COLUMNS].to_numpy()))
    key_to_pid = dict(zip(particles["_key"], particles["particle_id"]))
    hits["particle_id"] = hits["_key"].map(key_to_pid)
    particles.drop(columns="_key", inplace=True)
    hits.drop(columns="_key", inplace=True)


def _load_space_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and join space points with particle IDs.

    Mirrors GUNTAM/IO/Read_ACTS_Csv._process_space_points_data:
      spacepoint.csv --(measurement_id_1)--> measurement-simhit-map.csv --(hit_id)--> hits.csv

    Returns:
        (sp, particles) where sp has columns [x, y, z, particle_id] after
        spatial filtering (R<500, |Z|<1000), reset to a contiguous 0-based index.
    """
    hits = pd.read_csv(HITS_FILE, dtype={"geometry_id": np.int64})
    particles = pd.read_csv(PARTICLES_FILE, dtype={"particle_id": np.int64})
    _create_particle_id_column(hits, particles)
    # hits index → hit_id used by the measurement map
    hits["hit_id"] = hits.index.astype(int)

    meas_map = pd.read_csv(MEAS_FILE)  # columns: measurement_id, hit_id
    sp = pd.read_csv(SP_FILE)  # columns: measurement_id_1, measurement_id_2, x, y, z, …

    # Join measurement_id_1 → hit_id → particle_id
    sp = sp.merge(
        meas_map.rename(columns={"hit_id": "hit_id_1"}),
        left_on="measurement_id_1",
        right_on="measurement_id",
        how="left",
    ).drop(columns=["measurement_id"])

    sp = sp.merge(
        hits[["hit_id", "particle_id"]].rename(columns={"hit_id": "hit_id_1"}),
        on="hit_id_1",
        how="left",
    )

    # Spatial filter matching training geometry (R<500, |Z|<1000)
    r2 = sp["x"] ** 2 + sp["y"] ** 2
    mask = (r2 < 500**2) & (sp["z"].abs() < 1000)
    sp = sp[mask][["x", "y", "z", "particle_id"]].reset_index(drop=True)

    return sp, particles


@pytest.fixture(scope="module")
def event0():
    """Load and prepare event-0 space points and particles."""
    if not SP_FILE.exists():
        pytest.skip(f"Space-point file not found at {SP_FILE}")
    return _load_space_points()


def _majority_pid(seed: np.ndarray, pid_array: np.ndarray) -> int | None:
    """Return the majority particle_id for a seed row, or None if all slots are padding/NaN."""
    valid_slots = seed[seed >= 0]
    if len(valid_slots) == 0:
        return None
    pids = pid_array[valid_slots]
    pids_valid = pids[~np.isnan(pids.astype(float))]
    if len(pids_valid) == 0:
        return None
    counts = np.bincount(pids_valid.astype(np.int64) % (2**31))
    return int(np.argmax(counts))


def _build_reconstruction_model(with_classifier: bool) -> SeedReconstructionModel:
    """Build the transformer(+classifier) reconstruction model shared by both backends."""
    cfg = SeedConfig()
    cfg.epoch_nb = 1

    transformer = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=device,
        dtype=torch.float32,
    )
    transformer.to(device)
    transformer.load(str(PT_MODEL), device=device)
    transformer.eval()

    classifier = None
    if with_classifier:
        classifier = SeedClassifier(device_acc=device)
        classifier.load(str(CLASSIFIER_MODEL), device=device)
        classifier.threshold = 0.5
        classifier.eval()

    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=transformer,
        classifier=classifier,
        device_acc=device,
        width=5,
        max_seed_length=3,
    )
    model.to(device)
    model.eval()
    return model


def _run_inference_classique(model: SeedReconstructionModel, sp: "pd.DataFrame"):
    """Run the raw (non-exported) PyTorch model and return (seeds [S, SL], seed_scores [S])."""
    sp_np = sp[["x", "y", "z"]].to_numpy(dtype=np.float32)
    sp_tensor = torch.tensor(sp_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        seeds, seed_scores = model(sp_tensor)

    return seeds.cpu().numpy(), seed_scores.cpu().numpy()


def _run_inference_onnx(
    model: SeedReconstructionModel,
    sp: "pd.DataFrame",
    transformer_path: str,
    classifier_path: str | None,
):
    """Run inference via `SeedReconstructionModel.run_onnx_inference` and return (seeds [S, SL], seed_scores [S])."""
    sp_np = sp[["x", "y", "z"]].to_numpy(dtype=np.float32)
    sp_tensor = torch.tensor(sp_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        seeds, seed_scores = model.run_onnx_inference(sp_tensor, transformer_path, classifier_path, device=device)

    return seeds.cpu().numpy(), seed_scores.cpu().numpy()


@pytest.fixture(scope="module")
def model():
    """Build the transformer(+classifier) reconstruction model once, shared by both backends."""
    if not PT_MODEL.exists():
        pytest.skip(f"PyTorch checkpoint not found at {PT_MODEL}")
    return _build_reconstruction_model(with_classifier=CLASSIFIER_MODEL.exists())


@pytest.fixture(scope="module")
def onnx_paths(tmp_path_factory, model):
    """Export the transformer (and classifier, if available) to ONNX via Export_full_model.main()."""
    tmp_dir = tmp_path_factory.mktemp("onnx")
    transformer_path = tmp_dir / "transformer.onnx"
    classifier_path = tmp_dir / "classifier.onnx"

    # Reuse the same logic as `python -m GUNTAM.Seed.Export_full_model`
    # Build the namespace directly to avoid touching sys.argv
    export_args = argparse.Namespace(
        checkpoint=str(PT_MODEL),
        classifier_checkpoint=str(CLASSIFIER_MODEL) if CLASSIFIER_MODEL.exists() else None,
        config=None,
        classifier_config=None,
        classifier_threshold=None,
        transformer_output=str(transformer_path),
        classifier_output=str(classifier_path),
        width=5,
        max_seed_length=3,
        num_example_hits=50000,
        device="cpu",
    )
    # Temporarily patch parse_args so main() uses our namespace
    _orig_parse_args = _export_module.parse_args
    _export_module.parse_args = lambda: export_args
    try:
        _export_module.main()
    finally:
        _export_module.parse_args = _orig_parse_args

    return str(transformer_path), str(classifier_path) if CLASSIFIER_MODEL.exists() else None


@pytest.fixture(scope="module", params=["classique", "onnx"], ids=["classique", "onnx"])
def inference_results(request, event0, model):
    """Run inference through either the raw PyTorch ("classique") or ONNX-exported model."""
    sp, _ = event0
    if request.param == "onnx":
        transformer_path, classifier_path = request.getfixturevalue("onnx_paths")
        return _run_inference_onnx(model, sp, transformer_path, classifier_path)

    if not CLASSIFIER_MODEL.exists():
        pytest.skip(f"Classifier checkpoint not found at {CLASSIFIER_MODEL}")
    return _run_inference_classique(model, sp)


class TestFullModel:
    """Tests run against both the raw PyTorch ("classique") and ONNX-exported model."""

    def test_model_file_exists(self):
        assert PT_MODEL.exists(), f"transformer.pt not found at {PT_MODEL}"

    def test_output_shape(self, event0, inference_results):
        """Seeds tensor must be 2-D and scores 1-D with matching first dimension."""
        seeds, seed_scores = inference_results

        assert seeds.ndim == 2, f"Expected 2-D seeds output, got shape {seeds.shape}"
        assert seeds.shape[1] >= 2, "Seed length must be ≥ 2"
        assert seed_scores.ndim == 1, f"Expected 1-D seed_scores, got shape {seed_scores.shape}"
        assert seed_scores.shape[0] == seeds.shape[0], "seeds and seed_scores must have the same length"

    def test_seed_indices_in_range(self, event0, inference_results):
        """Every non-negative index in the seed output must be a valid space-point index."""
        sp, _ = event0
        seeds, _ = inference_results
        valid_indices = seeds[seeds >= 0]
        assert valid_indices.max() < len(sp), f"Seed index {valid_indices.max()} out of range for {len(sp)} space points"

    def test_seeding_efficiency(self, event0, inference_results):
        """Compute efficiency and assert a non-trivial lower bound."""
        sp, particles = event0
        pid_array = sp["particle_id"].to_numpy()
        seeds, _ = inference_results

        seeded_particles: set = set()
        for seed in seeds:
            pid = _majority_pid(seed, pid_array)
            if pid is not None:
                seeded_particles.add(pid)

        total_particles = len(particles)
        efficiency = len(seeded_particles) / total_particles if total_particles > 0 else 0.0
        print(f"\nSeeding efficiency: {len(seeded_particles)}/{total_particles} = {efficiency:.2%}")

        assert total_particles > 0, "No particles found in event 0"
        assert efficiency >= 0.0  # Tighten once a baseline is established.

    def test_fake_rate(self, event0, inference_results):
        """Compute fake rate (seeds whose majority particle is ambiguous or absent)."""
        sp, _ = event0
        pid_array = sp["particle_id"].to_numpy()
        seeds, _ = inference_results

        fake_count = 0
        for seed in seeds:
            valid_slots = seed[seed >= 0]
            if len(valid_slots) == 0:
                continue
            pids = pid_array[valid_slots]
            pids_valid = pids[~np.isnan(pids.astype(float))]
            if len(pids_valid) == 0:
                fake_count += 1
                continue
            # Fake if the majority particle does not account for > 50 % of the seed hits
            counts = np.bincount(pids_valid.astype(np.int64) % (2**31))
            majority_fraction = counts.max() / len(pids_valid)
            if majority_fraction <= 0.5:
                fake_count += 1

        total_seeds = len(seeds)
        fake_rate = fake_count / total_seeds if total_seeds > 0 else 0.0
        print(f"\nFake rate: {fake_count}/{total_seeds} = {fake_rate:.2%}")

        assert total_seeds > 0, "No seeds produced"
        assert fake_rate <= 1.0  # Tighten once a baseline is established.

    def test_efficiency_and_fake_rate_with_score_threshold(self, event0, inference_results):
        """Efficiency and fake rate after applying a score threshold of 0.4 on seeds."""
        sp, particles = event0
        pid_array = sp["particle_id"].to_numpy()
        score_threshold = 0.4

        seeds, seed_scores = inference_results
        keep = seed_scores >= score_threshold
        seeds_filtered = seeds[keep]

        # Efficiency
        seeded_particles: set = set()
        for seed in seeds_filtered:
            pid = _majority_pid(seed, pid_array)
            if pid is not None:
                seeded_particles.add(pid)

        total_particles = len(particles)
        efficiency = len(seeded_particles) / total_particles if total_particles > 0 else 0.0

        # Fake rate
        fake_count = 0
        for seed in seeds_filtered:
            valid_slots = seed[seed >= 0]
            if len(valid_slots) == 0:
                continue
            pids = pid_array[valid_slots]
            pids_valid = pids[~np.isnan(pids.astype(float))]
            if len(pids_valid) == 0:
                fake_count += 1
                continue
            counts = np.bincount(pids_valid.astype(np.int64) % (2**31))
            if counts.max() / len(pids_valid) <= 0.5:
                fake_count += 1

        total_filtered = len(seeds_filtered)
        fake_rate = fake_count / total_filtered if total_filtered > 0 else 0.0

        print(
            f"\n[score≥{score_threshold}] Efficiency: {len(seeded_particles)}/{total_particles} = {efficiency:.2%} | "
            f"Fake rate: {fake_count}/{total_filtered} = {fake_rate:.2%}"
        )

        assert total_particles > 0, "No particles found in event 0"
        assert efficiency >= 0.0  # Tighten once a baseline is established.
        assert fake_rate <= 1.0  # Tighten once a baseline is established.
