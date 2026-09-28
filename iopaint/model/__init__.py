"""Model registry.

Erase models (LaMa, MAT, MI-GAN, ...) only need ``torch``; the diffusion models
additionally need ``diffusers``/``transformers``/``peft``, which are several GB
of dependencies. To make a light-weight "erase + background removal only"
install possible, the diffusion models are imported lazily: importing this
package never imports ``diffusers``.

Install the diffusion extras with::

    pip install iopaint[diffusion]
"""

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Dict, Iterator, Tuple

from iopaint.const import (
    ANYTEXT_NAME,
    INSTRUCT_PIX2PIX_NAME,
    KANDINSKY22_NAME,
    POWERPAINT_NAME,
)

from .fcf import FcF
from .lama import AnimeLaMa, LaMa
from .ldm import LDM
from .manga import Manga
from .mat import MAT
from .mi_gan import MIGAN
from .opencv2 import OpenCV2
from .zits import ZITS

# Models that only require torch. Always available.
erase_models: Dict[str, Any] = {
    LaMa.name: LaMa,
    AnimeLaMa.name: AnimeLaMa,
    LDM.name: LDM,
    ZITS.name: ZITS,
    MAT.name: MAT,
    FcF.name: FcF,
    OpenCV2.name: OpenCV2,
    Manga.name: Manga,
    MIGAN.name: MIGAN,
}

# name -> (module, class). Imported on first use.
_LAZY_DIFFUSION_MODELS: Dict[str, Tuple[str, str]] = {
    "runwayml/stable-diffusion-inpainting": (".sd", "SD15"),
    "Sanster/anything-4.0-inpainting": (".sd", "Anything4"),
    "Sanster/Realistic_Vision_V1.4-inpainting": (".sd", "RealisticVision14"),
    "stabilityai/stable-diffusion-2-inpainting": (".sd", "SD2"),
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": (".sdxl", "SDXL"),
    "Fantasy-Studio/Paint-by-Example": (".paint_by_example", "PaintByExample"),
    INSTRUCT_PIX2PIX_NAME: (".instruct_pix2pix", "InstructPix2Pix"),
    KANDINSKY22_NAME: (".kandinsky", "Kandinsky22"),
    POWERPAINT_NAME: (".power_paint.power_paint", "PowerPaint"),
    ANYTEXT_NAME: (".anytext.anytext_model", "AnyText"),
}

# Other lazily exported symbols (`from iopaint.model import ControlNet`)
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "ControlNet": (".controlnet", "ControlNet"),
    "SD": (".sd", "SD"),
    "SD15": (".sd", "SD15"),
    "SD2": (".sd", "SD2"),
    "Anything4": (".sd", "Anything4"),
    "RealisticVision14": (".sd", "RealisticVision14"),
    "SDXL": (".sdxl", "SDXL"),
    "PaintByExample": (".paint_by_example", "PaintByExample"),
    "InstructPix2Pix": (".instruct_pix2pix", "InstructPix2Pix"),
    "Kandinsky22": (".kandinsky", "Kandinsky22"),
    "PowerPaint": (".power_paint.power_paint", "PowerPaint"),
    "AnyText": (".anytext.anytext_model", "AnyText"),
}

_DIFFUSION_INSTALL_HINT = (
    "This model needs the diffusion dependencies, which are not installed. "
    "Install them with:  pip install 'iopaint[diffusion]'  "
    "(or: pip install diffusers transformers peft accelerate)"
)


def _lazy_import(module: str, attr: str):
    try:
        mod = import_module(module, __name__)
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(f"{_DIFFUSION_INSTALL_HINT}\nOriginal error: {e}") from e
    return getattr(mod, attr)


class _ModelRegistry(Mapping):
    """Dict-like registry that imports diffusion models on demand."""

    def __getitem__(self, key: str):
        if key in erase_models:
            return erase_models[key]
        if key in _LAZY_DIFFUSION_MODELS:
            return _lazy_import(*_LAZY_DIFFUSION_MODELS[key])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from erase_models
        yield from _LAZY_DIFFUSION_MODELS

    def __len__(self) -> int:
        return len(erase_models) + len(_LAZY_DIFFUSION_MODELS)

    def __contains__(self, key) -> bool:
        return key in erase_models or key in _LAZY_DIFFUSION_MODELS


models = _ModelRegistry()


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        return _lazy_import(*_LAZY_ATTRS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_diffusion_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False
