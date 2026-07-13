import pytest
import torch

from GUNTAM.IO.PreprocessingConfig import PreprocessingConfig
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedReconstructionModel import SeedReconstructionModel
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.TransformerConfig import TransformerConfig
import GUNTAM.Seed.Reconstruction as Reconstruction


def _make_seed_cfg() -> SeedConfig:
    """A small, fast, fully-deterministic-shape config: single bin (no_bin strategy) so
    per-bin hit-slot order is simply hits sorted by R+rho."""
    seed_cfg = SeedConfig()
    seed_cfg.preprocessing_config = PreprocessingConfig()
    seed_cfg.preprocessing_config.binning_strategy = "no_bin"
    seed_cfg.preprocessing_config.max_hit_input = 32

    trans_cfg = TransformerConfig()
    trans_cfg.nb_layers_t = 1
    trans_cfg.nb_heads = 1
    trans_cfg.dim_embedding = 16
    trans_cfg.feed_forward_ratio = 1
    trans_cfg.dropout = 0.0
    seed_cfg.transformer_config = trans_cfg
    return seed_cfg


def _make_model(**kwargs) -> SeedReconstructionModel:
    seed_cfg = _make_seed_cfg()
    torch.manual_seed(0)
    transformer = SeedTransformer(seed_cfg.transformer_config)
    transformer.eval()
    return SeedReconstructionModel(
        transformer_config=seed_cfg,
        transformer=transformer,
        width=5,
        max_seed_length=3,
        **kwargs,
    )


def _make_hits(n: int = 10) -> torch.Tensor:
    """Hits along the x-axis so rho == x; some clustered (Δρ<5) and some spread out (Δρ>5)."""
    xs = torch.tensor([0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 40.0, 41.0, 60.0])[:n]
    hits = torch.zeros(n, 3)
    hits[:, 0] = xs
    return hits


class TestSeedReconstructionModelRadialSeparationConstraint:
    def test_raw_chain_length_below_minimum_raises_at_construction(self):
        with pytest.raises(ValueError):
            _make_model(raw_chain_length=2)

    def test_raw_chain_length_boundary_three_accepted(self):
        model = _make_model(raw_chain_length=3)
        assert model.raw_chain_length == 3

    def test_flag_off_never_invokes_filter(self, monkeypatch):
        """The filter must never be called when radial_separation_constraint is False."""

        def _raise(*args, **kwargs):
            raise AssertionError("apply_radial_separation_filter must not be called when the flag is off")

        monkeypatch.setattr(Reconstruction, "apply_radial_separation_filter", _raise)

        model = _make_model(radial_separation_constraint=False)
        hits = _make_hits()
        with torch.inference_mode():
            seeds, scores = model(hits)
        assert seeds.shape[1] == 3
        assert scores.shape[0] == seeds.shape[0]

    @staticmethod
    def _assert_respects_min_delta_rho(seeds: torch.Tensor, hits: torch.Tensor, min_delta_rho_mm: float):
        rho = torch.sqrt((hits**2).sum(dim=-1))
        for row in seeds:
            valid = row[row >= 0]
            for a, b in zip(valid[:-1].tolist(), valid[1:].tolist()):
                delta = abs(rho[b].item() - rho[a].item())
                assert delta > min_delta_rho_mm - 1e-6

    def test_flag_on_respects_min_delta_rho(self):
        """radial_separation_constraint defaults to True; verify the wired-in filter (not the
        standalone function) actually enforces min_delta_rho_mm on the real forward() path."""
        min_delta = 5.0
        model = _make_model(radial_separation_constraint=True, raw_chain_length=5, min_delta_rho_mm=min_delta)
        hits = _make_hits()
        with torch.inference_mode():
            seeds, scores = model(hits)
        assert seeds.shape[1] == 3
        self._assert_respects_min_delta_rho(seeds, hits, min_delta)

    def test_returned_seeds_are_always_complete(self):
        """Regression guard for the incomplete-seed leak: every emitted row must be a full 3-hit
        seed. Before the fix, an incomplete chain such as [hit, -1, -1] passed the first-slot-only
        validity gate and was returned as a bogus 3-hit seed."""
        for flag in (True, False):
            model = _make_model(radial_separation_constraint=flag, raw_chain_length=5)
            hits = _make_hits()
            with torch.inference_mode():
                seeds, scores = model(hits)
            assert (seeds >= 0).all(), f"incomplete seed leaked with radial_separation_constraint={flag}: {seeds}"
            assert scores.shape[0] == seeds.shape[0]

    def test_onnx_export_with_flag_on_does_not_raise(self, tmp_path):
        """Authoritative proof that the filter (and the flag-gated scatter_reduce dedup fix)
        introduce no data-dependent Python control flow: a real torch.onnx.export attempt with
        radial_separation_constraint=True (the default) must succeed and the resulting graph must
        run in onnxruntime. Any data-dependent branching would raise GuardOnDataDependentSymNode
        here. Uses a tiny synthetic checkpoint, not the full trained model, so this stays fast
        enough for routine CI."""
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

        model = _make_model(radial_separation_constraint=True, raw_chain_length=5, min_delta_rho_mm=5.0)
        hits = _make_hits()

        out_path = str(tmp_path / "model.onnx")
        model.export_onnx(out_path, example_hits=hits)

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        seeds, scores = sess.run(None, {"hits": hits.numpy()})
        assert seeds.shape[1] == 3
        assert scores.shape[0] == seeds.shape[0]


class TestDedupSeeds:
    """Direct unit tests of SeedReconstructionModel._dedup_seeds, extracted from forward() so the
    two tie-breaking strategies (best-score-wins vs first-occurrence) can be tested deterministically
    without needing a full transformer/binning pipeline."""

    def test_flag_on_keeps_best_score_not_first_occurrence(self):
        model = _make_model(radial_separation_constraint=True)
        chains_flat = torch.tensor([[1, 2, 3], [1, 2, 3], [5, 6, 7]], dtype=torch.long)
        scores_flat = torch.tensor([0.5, 0.9, 0.3])
        unique_chains, unique_scores = model._dedup_seeds(chains_flat, scores_flat)

        rows = unique_chains.tolist()
        idx_123 = rows.index([1, 2, 3])
        idx_567 = rows.index([5, 6, 7])
        assert unique_chains.shape[0] == 2
        assert unique_scores[idx_123].item() == pytest.approx(0.9)
        assert unique_scores[idx_567].item() == pytest.approx(0.3)

    def test_flag_off_keeps_first_occurrence_unchanged(self):
        model = _make_model(radial_separation_constraint=False)
        chains_flat = torch.tensor([[1, 2, 3], [1, 2, 3], [5, 6, 7]], dtype=torch.long)
        scores_flat = torch.tensor([0.5, 0.9, 0.3])
        unique_chains, unique_scores = model._dedup_seeds(chains_flat, scores_flat)

        rows = unique_chains.tolist()
        idx_123 = rows.index([1, 2, 3])
        assert unique_scores[idx_123].item() == pytest.approx(0.5)  # first occurrence, not max

    def test_invalid_rows_excluded(self):
        model = _make_model(radial_separation_constraint=False)
        chains_flat = torch.tensor([[-1, -1, -1], [1, 2, 3]], dtype=torch.long)
        scores_flat = torch.tensor([0.9, 0.1])
        unique_chains, unique_scores = model._dedup_seeds(chains_flat, scores_flat)
        assert unique_chains.shape[0] == 1
        assert unique_chains.tolist() == [[1, 2, 3]]
        assert unique_scores.tolist() == pytest.approx([0.1])

    @pytest.mark.parametrize("flag", [True, False])
    def test_incomplete_rows_excluded(self, flag):
        """A partially-filled row (valid first slot, -1 tail) is not a 3-hit seed and must be
        dropped. Gating on the first slot alone would emit [2,-1,-1] as a reconstructed seed."""
        model = _make_model(radial_separation_constraint=flag)
        chains_flat = torch.tensor([[2, -1, -1], [4, 5, -1], [1, 2, 3]], dtype=torch.long)
        scores_flat = torch.tensor([0.9, 0.8, 0.1])
        unique_chains, unique_scores = model._dedup_seeds(chains_flat, scores_flat)
        assert unique_chains.tolist() == [[1, 2, 3]]
        assert unique_scores.tolist() == pytest.approx([0.1])
