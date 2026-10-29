import random
import pytest
import torch
from unittest.mock import Mock, patch

from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config.TransformerConfig import TransformerConfig
from GUNTAM.Monitoring.PermutationMetricEvaluator import (
    prepare_event_encoding,
    PermutationMetricEvaluator,
)


def make_config(
    nb_layers_t: int = 2,
    nb_heads: int = 2,
    dim_embedding: int = 64,
    dropout: float = 0.0,
    fourier_num_frequencies: list | None = None,
) -> TransformerConfig:
    """Build a minimal TransformerConfig for tests.

    Mirrors the auto-derivation logic from TransformerConfig.apply_args so that
    fourier_num_frequencies is always a valid list (never None) when the model is
    constructed.
    """
    cfg = TransformerConfig()
    cfg.nb_layers_t = nb_layers_t
    cfg.nb_heads = nb_heads
    cfg.dim_embedding = dim_embedding
    cfg.dropout = dropout
    if fourier_num_frequencies is not None:
        cfg.fourier_num_frequencies = fourier_num_frequencies
    else:
        # Replicate apply_args auto-derivation:
        # cosine-processed embedding features expand to cos+sin (+1 per such feature)
        embed_cosine_n = len(set(cfg.embedding_feature) & set(cfg.cosine_processing))
        coord_dim = len(cfg.embedding_feature) + embed_cosine_n
        n_high = len(cfg.high_level_features) + len(set(cfg.high_level_features) & set(cfg.cosine_processing))
        cfg.fourier_num_frequencies = [max(1, (dim_embedding - n_high) // (2 * coord_dim))] * coord_dim
    return cfg


@pytest.fixture
def model():
    """A small SeedTransformer for synthetic tests."""
    torch.manual_seed(42)
    cfg = make_config(nb_layers_t=1, nb_heads=1, dim_embedding=32, dropout=0.0)
    return SeedTransformer(transformer_config=cfg)


@pytest.fixture
def event_hits():
    """Random hits tensor [B=2, N=8, F=6]."""
    torch.manual_seed(0)
    return torch.rand(2, 8, 6)


@pytest.fixture
def event_mask():
    """Padding mask [B=2, N=8] — all valid (False = valid)."""
    return torch.zeros(2, 8, dtype=torch.bool)


class TestPrepareEventEncoding:
    """Unit tests for prepare_event_encoding."""

    def test_returns_three_outputs(self, model, event_hits, event_mask):
        encoded_hits, transformer_output, attention_maps = prepare_event_encoding(
            model=model, event_hits=event_hits, event_mask=event_mask, features=[]
        )
        assert encoded_hits is not None
        assert transformer_output is not None
        assert attention_maps is not None

    def test_no_features_no_shuffle(self, model, event_hits, event_mask):
        """With features=[], encoded_hits should equal the unshuffled embedding."""
        expected = model.embedding(event_hits)  # fourier_embedding + feature_projection
        encoded_hits, _, _ = prepare_event_encoding(model=model, event_hits=event_hits, event_mask=event_mask, features=[])
        assert torch.allclose(encoded_hits, expected)

    def test_shuffle_single_feature_only_affects_that_feature(self, model, event_hits, event_mask):
        """Shuffling feature k should only change column k of the fourier-encoded hits."""
        feature_idx = 2

        # Compute the fourier-encoded hits directly (before feature_projection)
        fourier_encoded = model.fourier_embedding(event_hits)

        # Manually apply the same shuffle to the fourier-encoded hits
        random.seed(42)
        idx = list(range(fourier_encoded.size(1)))
        random.shuffle(idx)
        tensor_idx = torch.tensor(idx)
        expected_shuffled = fourier_encoded.clone()
        expected_shuffled[:, :, feature_idx] = fourier_encoded[:, tensor_idx, feature_idx]

        # Apply feature_projection to get the expected output
        expected_output = model.feature_projection(expected_shuffled)

        # Now call prepare_event_encoding with the same seed
        random.seed(42)
        shuffled, _, _ = prepare_event_encoding(model=model, event_hits=event_hits, event_mask=event_mask, features=[feature_idx])

        assert torch.allclose(shuffled, expected_output)

    def test_shuffle_multiple_features_independently(self, model, event_hits, event_mask):
        """Shuffling features [0, 2] should change both columns 0 and 2 of the fourier-encoded hits."""
        fourier_encoded = model.fourier_embedding(event_hits)

        # Manually apply independent shuffles to features 0 and 2 sequentially
        # (matching the loop in prepare_event_encoding)
        random.seed(42)
        idx0 = list(range(fourier_encoded.size(1)))
        random.shuffle(idx0)
        tensor_idx0 = torch.tensor(idx0)

        idx2 = list(range(fourier_encoded.size(1)))
        random.shuffle(idx2)
        tensor_idx2 = torch.tensor(idx2)

        expected_shuffled = fourier_encoded.clone()
        expected_shuffled[:, :, 0] = fourier_encoded[:, tensor_idx0, 0]
        expected_shuffled[:, :, 2] = fourier_encoded[:, tensor_idx2, 2]

        expected_output = model.feature_projection(expected_shuffled)

        # Now call prepare_event_encoding with the same seed
        random.seed(42)
        shuffled, _, _ = prepare_event_encoding(model=model, event_hits=event_hits, event_mask=event_mask, features=[0, 2])

        assert torch.allclose(shuffled, expected_output)

    def test_shuffle_is_permutation_not_replacement(self, model, event_hits, event_mask):
        """The multiset of values in a shuffled fourier-encoded column must be preserved."""
        feature_idx = 1

        # Compute the fourier-encoded hits directly
        fourier_encoded = model.fourier_embedding(event_hits)

        # Manually apply the shuffle to the fourier-encoded hits
        random.seed(42)
        idx = list(range(fourier_encoded.size(1)))
        random.shuffle(idx)
        tensor_idx = torch.tensor(idx)
        expected_shuffled = fourier_encoded.clone()
        expected_shuffled[:, :, feature_idx] = fourier_encoded[:, tensor_idx, feature_idx]

        # The multiset of values in the shuffled column must be preserved
        baseline_sorted, _ = torch.sort(fourier_encoded[:, :, feature_idx].flatten())
        shuffled_sorted, _ = torch.sort(expected_shuffled[:, :, feature_idx].flatten())
        assert torch.allclose(baseline_sorted, shuffled_sorted)

        # And the shuffled column should differ from the original
        assert not torch.allclose(expected_shuffled[:, :, feature_idx], fourier_encoded[:, :, feature_idx])

    def test_shuffle_uses_independent_permutations(self, model, event_hits, event_mask):
        """Shuffling two features should use different permutations (with high probability)."""
        torch.manual_seed(123)
        # Use a larger N to make accidental identical permutations very unlikely
        hits = torch.rand(1, 20, 6)
        mask = torch.zeros(1, 20, dtype=torch.bool)

        # Record the permutations applied by patching random.shuffle
        permutations = []

        original_shuffle = random.shuffle

        def recording_shuffle(lst):
            original_shuffle(lst)
            permutations.append(list(lst))

        with patch("GUNTAM.Monitoring.PermutationMetricEvaluator.random.shuffle", side_effect=recording_shuffle):
            prepare_event_encoding(model=model, event_hits=hits, event_mask=mask, features=[0, 1])

        # Two shuffles should have been performed (one per feature)
        assert len(permutations) == 2
        # The two permutations should be different (with high probability for N=20)
        assert permutations[0] != permutations[1]

    def test_attention_maps_shape(self, model, event_hits, event_mask):
        """attention_maps should have shape [B, N, N] (single head, squeezed)."""
        _, _, attention_maps = prepare_event_encoding(model=model, event_hits=event_hits, event_mask=event_mask, features=[])
        B, N, _ = event_hits.shape
        assert attention_maps.shape == (B, N, N)


class TestPermutationMetricEvaluator:
    """Unit tests for PermutationMetricEvaluator."""

    def test_call_invokes_metric_fn_with_correct_args(self):
        """The evaluator should call metric_fn with the right arguments."""
        metric_fn = Mock(return_value=0.42)
        model = Mock()
        dataset = Mock()
        dataset.file_paths = ["file_a", "file_b", "file_c"]
        cfg = Mock()

        evaluator = PermutationMetricEvaluator(metric_fn=metric_fn, model=model, dataset=dataset, cfg=cfg)
        result = evaluator(3)

        metric_fn.assert_called_once_with(
            model=model,
            dataset=dataset,
            file_indices=[0, 1, 2],
            cfg=cfg,
            features=[3],
        )
        assert result == 0.42

    def test_call_returns_metric_result(self):
        """The evaluator should pass through the metric function's return value."""

        def fake_metric_fn(**kwargs):
            return 0.99

        model = Mock()
        dataset = Mock()
        dataset.file_paths = ["file_a"]
        cfg = Mock()

        evaluator = PermutationMetricEvaluator(metric_fn=fake_metric_fn, model=model, dataset=dataset, cfg=cfg)
        assert evaluator(0) == 0.99

    def test_call_with_different_feature_indices(self):
        """Each call should pass the correct feature index."""
        metric_fn = Mock(return_value=0.0)
        model = Mock()
        dataset = Mock()
        dataset.file_paths = ["file_a", "file_b"]
        cfg = Mock()

        evaluator = PermutationMetricEvaluator(metric_fn=metric_fn, model=model, dataset=dataset, cfg=cfg)
        evaluator(0)
        evaluator(5)

        assert metric_fn.call_count == 2
        assert metric_fn.call_args_list[0].kwargs["features"] == [0]
        assert metric_fn.call_args_list[1].kwargs["features"] == [5]
