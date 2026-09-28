"""Filename and metadata handling for saved results.

Goals:

* never silently overwrite an existing file;
* the extension must match the actual image data (a transparent cutout is a
  PNG, even if the source was a .jpg);
* the name says what produced the file (model / plugin), not only that it is
  a new version.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from iopaint.helper import get_image_ext

# Formats that can store an alpha channel
ALPHA_CAPABLE = {"png", "webp", "tiff"}

SAVE_POLICIES = ("unique", "timestamp", "overwrite")

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING = re.compile(r"[ .]+$")


def sanitize_stem(name: str, fallback: str = "image") -> str:
    """Make a filename component safe on Windows/macOS/Linux."""
    stem = Path(str(name)).name
    stem = Path(stem).stem or fallback
    stem = _UNSAFE.sub("_", stem)
    stem = _TRAILING.sub("", stem).strip()
    # reserved device names on Windows
    if stem.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = f"_{stem}"
    return stem[:120] or fallback


def sanitize_tool(tool: Optional[str]) -> str:
    """Normalise a model/plugin name so it can go into a filename."""
    if not tool:
        return ""
    tool = str(tool).strip().lower()
    tool = tool.replace("/", "-").replace(" ", "-")
    tool = re.sub(r"[^a-z0-9._-]+", "-", tool)
    tool = re.sub(r"-{2,}", "-", tool).strip("-.")
    return tool[:40]


def image_ext_from_bytes(image_bytes: bytes, has_alpha: bool = False) -> str:
    """Extension that actually matches the bytes (png wins if alpha is used)."""
    ext = get_image_ext(image_bytes)
    if ext == "jpeg":
        ext = "jpg"
    if has_alpha and ext not in ALPHA_CAPABLE:
        return "png"
    return ext


def build_filename(
    original_name: str,
    tool: Optional[str] = None,
    ext: str = "png",
    version: Optional[int] = None,
) -> str:
    """``photo.jpg`` + tool ``lama`` + version 2 -> ``photo_lama_v2.png``"""
    stem = sanitize_stem(original_name)
    tool = sanitize_tool(tool)

    parts = [stem]
    if tool:
        parts.append(tool)
    if version and version > 1:
        parts.append(f"v{version}")
    return f"{'_'.join(parts)}.{ext.lstrip('.')}"


def resolve_save_path(
    output_dir: Path, filename: str, policy: str = "unique"
) -> Tuple[Path, bool]:
    """Return the final path and whether an existing file is being replaced."""
    if policy not in SAVE_POLICIES:
        policy = "unique"

    output_dir = Path(output_dir)
    name = Path(filename).name
    stem, ext = Path(name).stem, Path(name).suffix
    path = output_dir / name

    if not path.exists():
        return path, False

    if policy == "overwrite":
        return path, True

    if policy == "timestamp":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return output_dir / f"{stem}_{stamp}{ext}", False

    # "unique": photo.png -> photo_2.png, photo_3.png ...
    counter = 2
    while True:
        candidate = output_dir / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate, False
        counter += 1
        if counter > 9999:  # pragma: no cover - pathological case
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return output_dir / f"{stem}_{stamp}{ext}", False


def build_infos(
    base_infos: Optional[Dict] = None,
    tool: Optional[str] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict:
    """Provenance metadata embedded into the saved image."""
    from iopaint import __version__ as iopaint_version

    infos = dict(base_infos or {})
    software = f"IOPaint {version or iopaint_version}"
    if tool:
        software += f" ({tool})"
    infos["Software"] = software
    infos["iopaint-version"] = version or iopaint_version
    if tool:
        infos["iopaint-tool"] = str(tool)
    if model:
        infos["iopaint-model"] = str(model)
    infos["iopaint-created"] = datetime.now().isoformat(timespec="seconds")
    return infos
