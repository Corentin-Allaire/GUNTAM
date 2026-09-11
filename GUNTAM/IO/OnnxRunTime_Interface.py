import torch
from torch import Tensor
import onnxruntime as ort
import numpy as np

# Maps torch dtypes to the numpy dtype ONNX Runtime's IOBinding expects for `element_type`.
_TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


def onnx_providers(device: torch.device) -> list[str]:
    """Pick ONNX Runtime execution providers matching the given torch device, falling back to CPU."""
    if device.type == "cuda":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def run_onnx_iobinding(
    session: ort.InferenceSession,
    inputs: list[Tensor],
    device_type: str,
    device_id: int,
    extra_inputs: dict[str, Tensor] | None = None,
) -> list[Tensor]:
    """
    Run an ONNX Runtime session via `IOBinding`, binding directly to the input tensors'
    device memory and returning outputs as torch tensors on that same device, without
    any host (numpy) round-trip.
    Args:
        - session (ort.InferenceSession): Session to run.
        - inputs (list[Tensor]): Input tensors, in the same order as `session.get_inputs()`.
        - device_type (str): ONNX Runtime device type ("cuda" or "cpu") to bind inputs/outputs to.
        - device_id (int): Device index (e.g. GPU index) to bind inputs/outputs to.
        - extra_inputs: Optional additional ONNX inputs, keyed by their ONNX input name. For example:
            {"width": torch.tensor(1200, dtype=torch.int64)}
    Returns:
        List of output tensors, in the same order as `session.get_outputs()`.
    """
    extra_inputs = extra_inputs or {}
    io_binding = session.io_binding()
    # Keep contiguous copies alive until run_with_iobinding() executes.
    bound_inputs = [tensor.contiguous() for tensor in inputs]
    extra_inputs = {name: tensor for name, tensor in (extra_inputs or {}).items()}

    ort_inputs = session.get_inputs()

    for i, ort_input in enumerate(ort_inputs):
        if i < len(bound_inputs):
            tensor = bound_inputs[i]
        elif ort_input.name in extra_inputs:
            tensor = extra_inputs[ort_input.name]
        else:
            raise ValueError(f"Missing ONNX input: {ort_input.name!r}. " f"Expected inputs: {[x.name for x in ort_inputs]}")

        io_binding.bind_input(
            name=ort_input.name,
            device_type=device_type,
            device_id=device_id,
            element_type=_TORCH_TO_NUMPY_DTYPE[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    for ort_output in session.get_outputs():
        io_binding.bind_output(
            ort_output.name,
            device_type=device_type,
            device_id=device_id,
        )

    session.run_with_iobinding(io_binding)

    return [torch.from_dlpack(ortvalue) for ortvalue in io_binding.get_outputs()]
