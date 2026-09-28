"""Where IOPaint writes things on disk, how much it takes, and how to clean it.

Everything IOPaint downloads or caches is supposed to live under a single
"data directory" (``--model-dir``), so that nothing is scattered over the
system drive. This module resolves those locations, measures them and can
delete them again.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Logical name -> human description. Used by the API, the UI and the CLI.
STORAGE_TARGETS = {
    "erase_models": "Erase models (LaMa, MAT, MI-GAN, ...)",
    "plugin_models": "Plugin models (RemoveBG, AnimeSeg, SAM, upscalers)",
    "huggingface": "HuggingFace cache (diffusion models, RMBG)",
    "torch_hub": "Torch hub cache",
    "thumbnails": "File manager thumbnails",
    "outputs": "Saved result images",
}


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def dir_size(path: Optional[Path]) -> int:
    if path is None:
        return 0
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path, onerror=lambda e: None):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def setup_data_dirs(data_dir: Path) -> Path:
    """Point every cache used by IOPaint's dependencies inside ``data_dir``.

    Must run before huggingface_hub / torch.hub are imported.
    """
    data_dir = Path(data_dir).expanduser().absolute()
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = str(data_dir)
    os.environ["U2NET_HOME"] = str(data_dir / "rembg")
    os.environ["TORCH_HOME"] = str(data_dir / "torch")
    os.environ["HF_HOME"] = str(data_dir / "huggingface")
    # HF_HUB_CACHE is what actually holds the weights
    os.environ["HF_HUB_CACHE"] = str(data_dir / "huggingface" / "hub")
    return data_dir


def storage_paths(
    data_dir: Path, output_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """Resolve all interesting locations with their size on disk."""
    data_dir = Path(data_dir).expanduser().absolute()

    locations = {
        "erase_models": data_dir / "torch" / "hub" / "checkpoints",
        "plugin_models": data_dir / "rembg",
        "huggingface": data_dir / "huggingface",
        "torch_hub": data_dir / "torch",
        "thumbnails": data_dir / "thumbnails",
        "outputs": Path(output_dir).expanduser().absolute() if output_dir else None,
    }

    result = {}
    for key, path in locations.items():
        result[key] = {
            "name": key,
            "description": STORAGE_TARGETS[key],
            "path": str(path) if path else "",
            "exists": bool(path and path.exists()),
            "size": dir_size(path),
            "size_human": human_size(dir_size(path)),
            # deleting the outputs means deleting the user's own results
            "safe_to_delete": key != "outputs",
        }

    total = sum(
        v["size"] for k, v in result.items() if k not in ("torch_hub", "outputs")
    )
    result["_total"] = {
        "name": "_total",
        "description": "Total cache used by IOPaint",
        "path": str(data_dir),
        "exists": data_dir.exists(),
        "size": total,
        "size_human": human_size(total),
        "safe_to_delete": False,
    }
    return result


def cleanup(
    data_dir: Path,
    targets: List[str],
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict:
    """Delete the selected caches. Returns what was (or would be) removed."""
    paths = storage_paths(data_dir, output_dir)
    removed, freed, errors = [], 0, []

    for target in targets:
        if target not in STORAGE_TARGETS:
            errors.append(f"unknown target: {target}")
            continue
        info = paths[target]
        path = Path(info["path"]) if info["path"] else None
        if not path or not path.exists():
            continue

        freed += info["size"]
        removed.append({"target": target, "path": str(path), "size": info["size"]})
        if dry_run:
            continue
        try:
            if target == "outputs":
                # keep the folder the user configured, drop its content
                for child in path.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                shutil.rmtree(path)
            logger.info(f"Removed {path} ({human_size(info['size'])})")
        except OSError as e:
            errors.append(f"{path}: {e}")
            freed -= info["size"]

    return {
        "removed": removed,
        "freed": freed,
        "freed_human": human_size(freed),
        "errors": errors,
        "dry_run": dry_run,
    }
