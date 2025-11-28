import math
import pytest
import torch

from GUNTAM.Transformer.Utils import (
	ts_print,
	log_gradients,
	sync_device,
	load_state_dict_flex,
	_normalize_state_dict_keys,
	CosineScheduleWithMinLR,
	create_cosine_schedule_with_min_lr,
)


class DummyWriter:
	"""Minimal TensorBoard-like writer for tests."""

	def __init__(self):
		self.logged = []

	def add_scalar(self, tag, value, step):
		self.logged.append((tag, value, step))


class TestTsPrint:
	"""Basic test for timestamped print utility."""

	def test_ts_print_runs(self, capsys):
		ts_print("hello", 123)
		captured = capsys.readouterr()
		assert "hello" in captured.out
		assert "123" in captured.out
		assert "[" in captured.out and "]" in captured.out


class TestLogGradients:
	"""Tests for gradient logging utility."""

	def test_logs_norms_and_means_when_grads_present(self):
		model = torch.nn.Sequential(
			torch.nn.Linear(8, 4),
			torch.nn.ReLU(),
			torch.nn.Linear(4, 2),
		)
		x = torch.randn(3, 8)
		y = model(x).sum()
		y.backward()

		writer = DummyWriter()
		stats = log_gradients(model, writer=writer, step=10)

		# Should have entries for parameters that received gradients
		assert isinstance(stats, dict)
		assert len(stats) > 0
		for name, param in model.named_parameters():
			if param.grad is not None:
				assert name in stats
				assert stats[name] >= 0

		# Writer should have logged GradNorm and GradMean entries
		tags = [t for t, _, _ in writer.logged]
		assert any(tag.startswith("GradNorm/") for tag in tags)
		assert any(tag.startswith("GradMean/") for tag in tags)

	def test_ignores_params_without_grad(self):
		# A model where we detach outputs to avoid grads
		lin = torch.nn.Linear(4, 4)
		with torch.no_grad():
			_ = lin(torch.randn(2, 4))

		stats = log_gradients(lin, writer=None, step=None)
		# No grads collected
		assert stats == {}


class TestSyncDevice:
	"""Tests for device synchronization helper."""

	def test_cpu_noop(self):
		dev = torch.device("cpu")
		# Should not raise
		sync_device(dev)

	@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
	def test_cuda_sync(self):
		dev = torch.device("cuda")
		# Should not raise; synchronize current device
		sync_device(dev)


class TestNormalizeStateDictKeys:
	"""Tests for normalization of state_dict keys."""

	def test_strips_compile_and_dp_prefixes(self):
		sd = {
			"_orig_mod.module.layer.weight": torch.randn(3, 3),
			"_orig_mod.module.layer.bias": torch.randn(3),
			"module.other": torch.tensor(1.0),
			"regular": torch.tensor(2.0),
		}
		norm = _normalize_state_dict_keys(sd)
		assert "layer.weight" in norm
		assert "layer.bias" in norm
		assert "other" in norm
		assert "regular" in norm
		# Ensure original prefixed keys removed
		assert not any(k.startswith("_orig_mod.") or k.startswith("module.") for k in norm)


class TestLoadStateDictFlex:
	"""Tests for flexible state_dict loading utility."""

	class Tiny(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.layer = torch.nn.Linear(4, 4)

	def test_strict_success_after_normalization(self, capsys):
		m = self.Tiny()
		# Create a state_dict with prefixes that match the internal parameter names
		raw = {}
		for name, param in m.named_parameters():
			raw[f"_orig_mod.module.{name}"] = param.detach().clone()

		load_state_dict_flex(m, raw, desc="tiny")
		out = capsys.readouterr().out
		assert "Loaded tiny state_dict (strict)" in out

	def test_non_strict_when_missing_key(self, capsys):
		m = self.Tiny()
		# Provide only weight, miss bias
		raw = {"_orig_mod.module.layer.weight": torch.randn_like(m.layer.weight)}

		load_state_dict_flex(m, raw, desc="tiny")
		out = capsys.readouterr().out
		assert "Warning: strict load failed" in out
		assert "Non-strict load results" in out


class TestCosineScheduleWithMinLR:
	"""Tests for the custom cosine scheduler with warmup and min LR."""

	def test_warmup_then_cosine_then_min(self):
		model = torch.nn.Linear(4, 4)
		opt = torch.optim.SGD(model.parameters(), lr=0.1)

		sched = CosineScheduleWithMinLR(opt, num_warmup_steps=5, num_training_steps=20, min_lr_ratio=0.05)

		# Step through a few epochs and check ratios
		ratios = []
		for _ in range(25):
			ratios.append(sched.get_ratio())
			sched.step()

		# Warmup should be increasing 0 -> 1 over 5 steps
		assert pytest.approx(ratios[0], rel=0, abs=1e-6) == 0.0
		assert ratios[1] > ratios[0]
		assert pytest.approx(ratios[5], rel=0, abs=1e-6) == 1.0

		# After training steps, ratio should be min_lr_ratio
		assert pytest.approx(ratios[21], rel=0, abs=1e-6) == 0.05
		assert pytest.approx(ratios[-1], rel=0, abs=1e-6) == 0.05

		# Optimizer lrs should be updated accordingly
		lrs = [group["lr"] for group in opt.param_groups]
		for lr in lrs:
			assert lr >= 0.1 * 0.05

	def test_get_lr_matches_ratio_times_base(self):
		model = torch.nn.Linear(2, 2)
		opt = torch.optim.Adam(model.parameters(), lr=0.01)
		sched = CosineScheduleWithMinLR(opt, num_warmup_steps=2, num_training_steps=10, min_lr_ratio=0.1)

		# At start (last_epoch=0), ratio=0
		lrs0 = sched.get_lr()
		assert all(pytest.approx(l, rel=0, abs=1e-9) == 0.0 for l in lrs0)

		sched.step()  # last_epoch=1
		ratio1 = sched.get_ratio()
		lrs1 = sched.get_lr()
		for base, lr in zip(sched.base_lrs, lrs1):
			assert pytest.approx(lr, rel=0, abs=1e-9) == base * ratio1

	def test_state_dict_roundtrip(self):
		model = torch.nn.Linear(3, 3)
		opt = torch.optim.SGD(model.parameters(), lr=0.2)
		sched = create_cosine_schedule_with_min_lr(opt, 3, 12, min_lr_ratio=0.2)

		# Advance some steps
		for _ in range(5):
			sched.step()

		sd = sched.state_dict()
		# Create a fresh scheduler and load state
		sched2 = CosineScheduleWithMinLR(opt, 1, 2, min_lr_ratio=0.5)
		sched2.load_state_dict(sd)

		assert sched2.last_epoch == sched.last_epoch
		assert sched2.num_warmup_steps == sched.num_warmup_steps
		assert sched2.num_training_steps == sched.num_training_steps
		assert pytest.approx(sched2.min_lr_ratio, rel=0, abs=1e-12) == sched.min_lr_ratio
		assert sched2.base_lrs == sched.base_lrs


if __name__ == "__main__":
	pytest.main([__file__, "-v"])

