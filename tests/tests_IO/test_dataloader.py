import os
from typing import List
from pathlib import Path

import pytest
import torch

from GUNTAM.IO.DataLoader import DataLoader


class TestDataLoader:
    @staticmethod
    def _create_synthetic_dataset(tmp_path, dataset_name: str = "my_data") -> str:
        """Create a tiny synthetic dataset on disk compatible with DataLoader.

        Layout:
        - 3 files with events counts [2, 2, 1]
        - Tensors: "x" -> (n, 3) float32, "y" -> (n,) long
        - start/end events are cumulative
        - Metadata saved to f"{dataset_name}_metadata.pt"

        Returns:
            dataset_dir path as str
        """
        dataset_dir = str(tmp_path)

        per_file_events: List[int] = [2, 2, 1]
        file_paths: List[str] = []
        file_event_ranges: List[tuple[int, int]] = []

        start = 0
        for i, n in enumerate(per_file_events):
            end = start + n
            fname = f"{dataset_name}_{i}.pt"
            path = os.path.join(dataset_dir, fname)
            # Store only filename for metadata (DataLoader will reconstruct full path)
            file_paths.append(fname)

            # Create tensors with [E, B, ...] shape where B=3
            x = torch.arange(start * 3, end * 3, dtype=torch.float32).reshape(n, 3)  # [E, B]
            y = (
                torch.arange(start, end, dtype=torch.long)
                .unsqueeze(1)
                .expand(-1, 3)
            )  # [E, B]

            torch.save(
                {
                    "start_event": start,
                    "end_event": end,
                    "x": x,
                    "y": y,
                },
                path,
            )
            file_event_ranges.append((start, end))
            start = end

        metadata = {
            "total_events": file_event_ranges[-1][1],
            "nb_bins": 3,
            # DataLoader expects just filenames, not full paths
            "file_paths": file_paths,
            "file_event_ranges": file_event_ranges,
        }
        torch.save(metadata, os.path.join(dataset_dir, f"metadata_{dataset_name}.pt"))
        return dataset_dir

    @pytest.fixture()
    def synthetic_dataset(self, tmp_path):
        dataset_name = "my_data"
        dataset_dir = self._create_synthetic_dataset(tmp_path, dataset_name)
        try:
            yield dataset_dir, dataset_name
        finally:
            # Explicitly remove generated .pt files after tests
            for p in Path(dataset_dir).glob("*.pt"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    def test_get_file_returns_requested_tensors_and_events(self, synthetic_dataset):
        dataset_dir, dataset_name = synthetic_dataset
        dl = DataLoader(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            tensor_names=["x", "y"],
            device=torch.device("cpu"),
        )

        out = dl.get_file(1)  # second file: events [2, 4)

        assert "x" in out and isinstance(out["x"], torch.Tensor)
        assert "y" in out and isinstance(out["y"], torch.Tensor)
        # Expect [E, B, ...] with B=3 from metadata
        assert out["x"].shape == (2, 3)
        assert out["y"].shape == (2, 3)
        assert out["start_event"] == 2
        assert out["end_event"] == 4

    def test_get_file_raises_for_missing_tensor(self, synthetic_dataset):
        dataset_dir, dataset_name = synthetic_dataset
        dl = DataLoader(dataset_dir=dataset_dir, dataset_name=dataset_name, tensor_names=["nonexistent"])

        with pytest.raises(KeyError):
            _ = dl.get_file(0)  # noqa: F841

    def test_get_file_size(self, synthetic_dataset):
        dataset_dir, dataset_name = synthetic_dataset
        dl = DataLoader(dataset_dir=dataset_dir, dataset_name=dataset_name)

        # Third file has 1 event according to the fixture
        assert dl.get_file_size(2) == 1

    def test_get_batch_files_concatenates_and_sets_events(self, synthetic_dataset):
        dataset_dir, dataset_name = synthetic_dataset
        dl = DataLoader(dataset_dir=dataset_dir, dataset_name=dataset_name, tensor_names=["x", "y"])

        # First two files -> total 4 events
        batch = dl.get_batch_files((0, 1))

        assert "x" in batch and isinstance(batch["x"], torch.Tensor)
        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        # Expect [E, B, ...] with B=3 from metadata
        assert batch["x"].shape == (4, 3)
        assert batch["y"].shape == (4, 3)
        assert batch["start_event"] == 0
        assert batch["end_event"] == 4

    def test_get_batch_files_invalid_range_raises(self, synthetic_dataset):
        dataset_dir, dataset_name = synthetic_dataset
        dl = DataLoader(dataset_dir=dataset_dir, dataset_name=dataset_name, tensor_names=["x"])

        with pytest.raises(IndexError):
            _ = dl.get_batch_files((1, 0))  # end < start

        with pytest.raises(IndexError):
            _ = dl.get_batch_files((-1, 1))  # negative start

        with pytest.raises(IndexError):
            _ = dl.get_batch_files((0, 10))  # end out of bounds

        with pytest.raises(IndexError):
            _ = dl.get_batch_files((0,))  # invalid tuple size
