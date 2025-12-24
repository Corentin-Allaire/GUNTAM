import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from GUNTAM.Transformer.Utils import ts_print


class DataLoader:
    """
    DataLoader class for on-demand loading of dataset files during the training.

    """

    def __init__(
        self,
        dataset_dir: str = ".",
        dataset_name: str = "seeding_data",
        tensor_names: List[str] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.tensor_names = tensor_names if tensor_names is not None else []

        # File-based storage attributes
        self.file_paths: List[str] = []
        # Each element is a (start_event, end_event) tuple (end exclusive)
        self.file_event_ranges: List[Tuple[int, int]] = []  # (start_event, end_event)
        self.total_events = 0
        self.nb_bins = 0

        # Load dataset files
        os.makedirs(self.dataset_dir, exist_ok=True)
        metadata_path = os.path.join(self.dataset_dir, f"{self.dataset_name}_metadata.pt")
        ts_print(f"Metadata: {metadata_path}")
        self._load_metadata()

    def _load_file(self, file_idx: int) -> Dict[str, Any]:
        """Load a dataset file.

        Args:
            file_idx: Index of file to load

        Returns:
            Dictionary containing batch data
        """
        """Load a dataset file directly without caching."""
        if file_idx >= len(self.file_paths):
            raise IndexError(f"File index {file_idx} out of range")

        # Load file directly to GPU
        file_data = torch.load(
            self.file_paths[file_idx],
            map_location=self.device,
            weights_only=False,
        )
        return file_data

    def _load_metadata(self) -> None:
        """Load dataset metadata from file."""
        import os

        metadata_path = os.path.join(self.dataset_dir, f"{self.dataset_name}_metadata.pt")
        metadata = torch.load(metadata_path, map_location=self.device, weights_only=False)

        self.total_events = metadata["total_events"]
        self.nb_bins = metadata["nb_bins"]
        self.file_event_ranges = metadata["file_event_ranges"]

        # Reconstruct file paths using current dataset_dir and filenames from metadata
        # This allows the dataset to be moved to a different directory
        saved_file_paths = metadata["file_paths"]
        self.file_paths = []
        for saved_path in saved_file_paths:
            # Extract just the filename from the saved path
            filename = os.path.basename(saved_path)
            # Reconstruct the full path using current dataset_dir
            new_path = os.path.join(self.dataset_dir, filename)
            self.file_paths.append(new_path)

        # Verify files exist
        missing_files = [path for path in self.file_paths if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing dataset files: {missing_files}")

        ts_print(f"Loaded dataset metadata: {len(self.file_paths)} files, {self.total_events} total events")

    def __len__(self) -> int:
        """Return number of files in dataset"""
        return len(self.file_paths)

    def get_file(self, file_idx: int) -> Dict[str, Any]:
        """
        Get a selection of tensors/fields from a dataset file.

        Only accesses keys explicitly requested via `tensor_names`.

        Args:
            file_idx: Index of the file to read.

        Returns:
            Dict[str, Any]: Mapping from requested names to data.

        Raises:
            IndexError: If file_idx is out of range.
            KeyError: If a requested name cannot be resolved in the file.
        """
        if file_idx >= len(self.file_paths):
            raise IndexError(f"File index {file_idx} out of range (0-{len(self.file_paths) - 1})")

        file_data = self._load_file(file_idx)

        result: Dict[str, Any] = {}
        for name in self.tensor_names:
            if name not in file_data:
                available = ", ".join(sorted(file_data.keys()))
                raise KeyError(f"Requested tensor '{name}' not found in file. Available keys: {available}")
            value = file_data[name]
            if isinstance(value, torch.Tensor):
                if value.ndim < 2 or value.shape[1] != self.nb_bins:
                    raise ValueError(
                        f"Tensor '{name}' must have shape [E, B,...] with B={self.nb_bins} as dim 1; "
                        f"got {tuple(value.shape)}"
                    )
            result[name] = value

        result["start_event"] = file_data["start_event"]
        result["end_event"] = file_data["end_event"]
        result["file_idx"] = file_idx
        return result

    def get_file_number(self) -> int:
        """Get number of files in dataset.

        Returns:
            Number of files
        """
        return len(self.file_paths)

    def get_file_size(self, file_idx: int) -> int:
        """Get number of events in specified file.

        Args:
            file_idx: Index of file

        Returns:
            Number of events in file
        """
        if file_idx >= len(self.file_event_ranges):
            raise IndexError(f"File index {file_idx} out of range")

        start_event, end_event = self.file_event_ranges[file_idx]
        return end_event - start_event

    def get_batch_files(self, batch_file_range: Tuple[int, int]) -> Dict[str, Any]:
        """
        Get the data from a consecutive range of dataset files as a single batch.

        Args:
            batch_file_range: Tuple (start_idx, end_idx) inclusive file indices to read.

        Returns:
            Dict[str, Any]: Same structure as `get_file`, where each requested
            tensor is the concatenation across the specified files. The
            `start_event` corresponds to the first file in the range and the
            `end_event` to the last file in the range.

        Raises:
            IndexError: If the provided range is invalid or out of bounds.
            KeyError: If a requested tensor name is missing in any file.
        """
        if not isinstance(batch_file_range, tuple) or len(batch_file_range) != 2:
            raise IndexError("batch_file_range must be a tuple of (start_idx, end_idx)")

        start_idx, end_idx = batch_file_range
        if start_idx < 0 or end_idx < start_idx or end_idx >= len(self.file_paths):
            raise IndexError(f"Invalid file range ({start_idx}, {end_idx}) for 0-{len(self.file_paths) - 1}")

        per_name_values: Dict[str, List[Any]] = {name: [] for name in self.tensor_names}
        first_start_event: Optional[int] = None
        last_end_event: Optional[int] = None

        for file_idx in range(start_idx, end_idx + 1):
            file_data = self.get_file(file_idx)

            # Track global start/end event across the batch
            if first_start_event is None:
                first_start_event = int(file_data["start_event"])  # type: ignore[arg-type]
            last_end_event = int(file_data["end_event"])  # type: ignore[arg-type]

            # Collect values per requested tensor name
            for name in self.tensor_names:
                per_name_values[name].append(file_data[name])

        # Concatenate tensors when possible; otherwise keep as list
        result: Dict[str, Any] = {}
        for name, items in per_name_values.items():
            if len(items) == 0:
                continue
            if not isinstance(items[0], torch.Tensor):
                raise TypeError(f"Expected tensor for key '{name}', got {type(items[0])}")

            result[name] = torch.cat(items, dim=0)

        # Set batch start/end events
        result["start_event"] = first_start_event if first_start_event is not None else 0
        result["end_event"] = last_end_event if last_end_event is not None else 0

        return result

    def get_batch_size(self, batch_file_start: int, batch_file_end: int) -> int:
        """
        Get total number of events across a range of dataset files.

        Args:
            batch_file_range: Tuple (start_idx, end_idx) inclusive file indices.

        Returns:
            Total number of events across specified files.
        """
        if batch_file_start < 0 or batch_file_end < batch_file_start or batch_file_end >= len(self.file_paths):
            raise IndexError(
                f"Invalid file range ({batch_file_start}, {batch_file_end}) for 0-{len(self.file_paths) - 1}"
            )

        total_size = 0
        for file_idx in range(batch_file_start, batch_file_end + 1):
            total_size += self.get_file_size(file_idx)

        return total_size
