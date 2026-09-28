"""Hardware acceleration / CPU tuning helpers.

IOPaint is very often used on machines without a discrete NVIDIA GPU (for
example an AMD Ryzen APU with a Radeon iGPU). This module contains the two
things that matter on such machines:

* :func:`configure_cpu_threads` - make PyTorch/OpenCV/ONNXRuntime actually use
  all the physical cores, without oversubscribing them (which is slower).
* DirectML helpers - on Windows, any DirectX 12 GPU (AMD/Intel iGPU included)
  can run ONNX models through ``onnxruntime-directml`` and torch models through
  ``torch-directml``.
"""

import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import List, Optional

from loguru import logger

_CPU_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@lru_cache(maxsize=1)
def detect_physical_cores() -> int:
    """Best effort physical (non-SMT) core count."""
    logical = os.cpu_count() or 1

    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            ids = set()
            core_id = pkg_id = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    elif line.startswith("physical id"):
                        pkg_id = line.split(":")[1].strip()
                    elif not line.strip() and core_id is not None:
                        ids.add((pkg_id, core_id))
                        core_id = pkg_id = None
            if ids:
                return len(ids)
        elif system == "Windows":
            out = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            cores = [int(x) for x in out.split() if x.isdigit()]
            if cores:
                return sum(cores)
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if out.isdigit():
                return int(out)
    except Exception:
        pass

    # Assume SMT/Hyper-Threading when the logical count is even and >= 4.
    return max(1, logical // 2) if logical >= 4 and logical % 2 == 0 else logical


def configure_cpu_threads(num_threads: Optional[int] = None) -> int:
    """Set thread related env vars. Call this *before* importing torch.

    Returns the number of threads that will be used.
    """
    threads = int(num_threads) if num_threads else detect_physical_cores()
    threads = max(1, threads)

    for var in _CPU_ENV_VARS:
        os.environ.setdefault(var, str(threads))
    # Intel/AMD OpenMP: spinning threads burn power and hurt laptops/APUs.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    return threads


def apply_torch_cpu_tuning(num_threads: Optional[int] = None) -> int:
    """Apply the thread settings to an already imported torch/cv2."""
    threads = configure_cpu_threads(num_threads)

    try:
        import torch

        torch.set_num_threads(threads)
        try:
            # inter-op parallelism does not help for a single inference request
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # already initialised, ignore
            pass
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception as e:  # pragma: no cover - torch is always present in prod
        logger.debug(f"torch cpu tuning skipped: {e}")

    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass

    logger.info(f"CPU threads: {threads} (logical cores: {os.cpu_count()})")
    return threads


@lru_cache(maxsize=1)
def is_onnx_directml_available() -> bool:
    try:
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_torch_directml_available() -> bool:
    try:
        import torch_directml  # type: ignore

        return bool(torch_directml.is_available())
    except Exception:
        return False


def _device_str(device) -> str:
    """Accept both a plain string and a Device enum member."""
    value = getattr(device, "value", device)
    return str(value)


def onnx_providers(device: str) -> List[str]:
    """ONNXRuntime execution providers for a given IOPaint device string."""
    device = _device_str(device)
    providers: List[str] = []
    if device == "directml" and is_onnx_directml_available():
        providers.append("DmlExecutionProvider")
    if device == "cuda":
        try:
            import onnxruntime

            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.append("CUDAExecutionProvider")
        except Exception:
            pass
    providers.append("CPUExecutionProvider")
    return providers


def torch_device(device: str):
    """Map an IOPaint device string to a real torch device object."""
    import torch

    device = _device_str(device)
    if device == "directml":
        if is_torch_directml_available():
            import torch_directml  # type: ignore

            return torch_directml.device()
        logger.warning(
            "torch-directml is not installed, falling back to CPU for torch models. "
            "Install it with: pip install torch-directml"
        )
        return torch.device("cpu")
    return torch.device(device)


def describe_acceleration() -> str:
    parts = [f"physical cores: {detect_physical_cores()}"]
    if platform.system() == "Windows":
        parts.append(f"onnx-directml: {is_onnx_directml_available()}")
        parts.append(f"torch-directml: {is_torch_directml_available()}")
    return ", ".join(parts)
