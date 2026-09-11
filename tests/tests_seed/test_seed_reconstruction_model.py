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
