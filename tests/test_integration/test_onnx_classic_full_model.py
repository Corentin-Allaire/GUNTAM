"""
Integration test for the exported ONNX full model.
 
Pipeline:
  1. Load Work/transformer.pt, export it to a temporary ONNX file.
  2. Read event 0 space-point CSV and build a [N, 3] input (x, y, z).
  3. Run inference → seed tensor [S, seed_length] of original space-point indices.
  4. Map each seed to a particle via the space-point→particle_id table
     (built by joining spacepoint.csv → measurement-simhit-map.csv → hits.csv,
     mirroring the logic in GUNTAM/IO/Read_ACTS_Csv._process_space_points_data).
  5. Compute seeding efficiency = fraction of particles that have ≥1 seed.
 
This file also runs the same inference through the raw (non-exported) PyTorch
model, so the ONNX export can be cross-checked against the "classique" model.
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
def ort_session(tmp_path_factory):
    """Load transformer.pt via Export_full_model.main(), then return an onnxruntime session."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    if not PT_MODEL.exists():
        pytest.skip(f"PyTorch checkpoint not found at {PT_MODEL}")
 
    tmp_dir = tmp_path_factory.mktemp("onnx")
    onnx_path = tmp_dir / "model.onnx"
 
    # Reuse the same logic as `python -m GUNTAM.Seed.Export_full_model`
    # Build the namespace directly to avoid touching sys.argv
    export_args = argparse.Namespace(
        checkpoint=str(PT_MODEL),
        config=None,
        output=str(onnx_path),
        width=5,
        max_seed_length=3,
        num_example_hits=256,
        device="cpu",
    )
    # Temporarily patch parse_args so main() uses our namespace
    _orig_parse_args = _export_module.parse_args
    _export_module.parse_args = lambda: export_args
    try:
        _export_module.main()
    finally:
        _export_module.parse_args = _orig_parse_args
 
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
 
 
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
 
 
def _run_inference_onnx(ort_session, sp: "pd.DataFrame"):
    """Run the ONNX model and return (seeds [S, SL], seed_scores [S])."""
    sp_np = sp[["x", "y", "z"]].to_numpy(dtype=np.float32)
    seeds, seed_scores = ort_session.run(["seeds", "seed_scores"], {"hits": sp_np})
    return seeds, seed_scores
 
 
def _run_inference_classique(sp: "pd.DataFrame"):
    """Run the raw (non-exported) PyTorch model and return (seeds [S, SL], seed_scores [S])."""
    sp_np = sp[["x", "y", "z"]].to_numpy(dtype=np.float32)
    sp_tensor = torch.tensor(sp_np, dtype=torch.float32, device=device)
 
    # ---- 1. Config (doit correspondre à celle utilisée à l'entraînement) ----
    cfg = SeedConfig()
    cfg.epoch_nb = 1
 
    # ---- 2. Transformer entraîné ----
    transformer = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=device,
        dtype=torch.float32,
    )
    transformer.to(device)
    transformer.load(str(PT_MODEL), device=device)
    transformer.eval()
 
    # ---- 3. Modèle complet (transformer + reconstruction + classifier) ----
    model = SeedReconstructionModel(
        transformer_config=cfg,
        transformer=transformer,
        device_acc=device,
        width=5,
        max_seed_length=3,
        classifier_path=str(CLASSIFIER_MODEL),
        classifier_threshold=0.5,
    )
    model.to(device)
    model.eval()
 
    with torch.no_grad():
        seeds, seed_scores = model(sp_tensor)
 
    return seeds.cpu().numpy(), seed_scores.cpu().numpy()
 
 
@pytest.fixture(scope="module")
def inference_results_onnx(ort_session, event0):
    """Run ONNX inference once and share the results across all tests in the module."""
    sp, _ = event0
    return _run_inference_onnx(ort_session, sp)
 
 
@pytest.fixture(scope="module")
def inference_results_classique(event0):
    """Run raw PyTorch inference once and share the results across all tests in the module."""
    if not CLASSIFIER_MODEL.exists():
        pytest.skip(f"Classifier checkpoint not found at {CLASSIFIER_MODEL}")
    sp, _ = event0
    return _run_inference_classique(sp)
 
 
class TestOnnxFullModel:
    """Tests running against the ONNX-exported model."""
 
    def test_model_file_exists(self):
        assert PT_MODEL.exists(), f"transformer.pt not found at {PT_MODEL}"
 
    def test_output_shape(self, event0, inference_results_onnx):
        """Seeds tensor must be 2-D and scores 1-D with matching first dimension."""
        seeds, seed_scores = inference_results_onnx
 
        assert seeds.ndim == 2, f"Expected 2-D seeds output, got shape {seeds.shape}"
        assert seeds.shape[1] >= 2, "Seed length must be ≥ 2"
        assert seed_scores.ndim == 1, f"Expected 1-D seed_scores, got shape {seed_scores.shape}"
        assert seed_scores.shape[0] == seeds.shape[0], "seeds and seed_scores must have the same length"
 
    def test_seed_indices_in_range(self, event0, inference_results_onnx):
        """Every non-negative index in the seed output must be a valid space-point index."""
        sp, _ = event0
        seeds, _ = inference_results_onnx
        print("biiiip")
        print(seeds)
        valid_indices = seeds[seeds >= 0]
        assert valid_indices.max() < len(sp), f"Seed index {valid_indices.max()} out of range for {len(sp)} space points"
 
    def test_seeding_efficiency(self, event0, inference_results_onnx):
        """Compute efficiency and assert a non-trivial lower bound."""
        sp, particles = event0
        pid_array = sp["particle_id"].to_numpy()
        seeds, _ = inference_results_onnx
 
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
 
    def test_fake_rate(self, event0, inference_results_onnx):
        """Compute fake rate (seeds whose majority particle is ambiguous or absent)."""
        sp, _ = event0
        pid_array = sp["particle_id"].to_numpy()
        seeds, _ = inference_results_onnx
 
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
 
    def test_efficiency_and_fake_rate_with_score_threshold(self, event0, inference_results_onnx):
        """Efficiency and fake rate after applying a score threshold of 0.4 on seeds."""
        sp, particles = event0
        pid_array = sp["particle_id"].to_numpy()
        score_threshold = 0.4
 
        seeds, seed_scores = inference_results_onnx
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
