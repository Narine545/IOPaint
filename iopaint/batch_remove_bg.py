"""Batch background removal from the command line.

Example::

    iopaint remove-bg --image ./photos --output ./cutouts \
        --model isnet-general-use --alpha-matting quality
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from iopaint.batch_processing import glob_images
from iopaint.schema import AlphaMattingMethod, RunPluginRequest


def batch_remove_bg(
    image: Path,
    output: Path,
    model: str = "isnet-general-use",
    device: str = "cpu",
    alpha_matting: str = AlphaMattingMethod.fast,
    fg_threshold: int = 240,
    bg_threshold: int = 15,
    erode_size: int = 8,
    matting_max_size: int = 1280,
    remove_color_spill: bool = True,
    save_mask: bool = False,
    anime: bool = False,
):
    if image.is_dir() and output.is_file():
        logger.error(
            "invalid --output: when image is a directory, output should be a directory"
        )
        exit(-1)
    output.mkdir(parents=True, exist_ok=True)

    image_paths = glob_images(image)
    if not image_paths:
        logger.error("invalid --image: no png/jpg/jpeg found")
        exit(-1)

    if anime:
        from iopaint.plugins.anime_seg import AnimeSeg

        plugin = AnimeSeg(
            alpha_matting=alpha_matting,
            fg_threshold=fg_threshold,
            bg_threshold=bg_threshold,
            erode_size=erode_size,
            matting_max_size=matting_max_size,
            remove_color_spill=remove_color_spill,
        )
    else:
        from iopaint.plugins.remove_bg import RemoveBG

        plugin = RemoveBG(
            model,
            device,
            alpha_matting=alpha_matting,
            fg_threshold=fg_threshold,
            bg_threshold=bg_threshold,
            erode_size=erode_size,
            matting_max_size=matting_max_size,
            remove_color_spill=remove_color_spill,
        )

    req = RunPluginRequest(name=plugin.name, image="")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Remove background", total=len(image_paths))
        for stem, path in image_paths.items():
            bgr = cv2.imdecode(
                np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is None:
                logger.warning(f"Skip unreadable image: {path}")
                progress.update(task, advance=1)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if save_mask:
                result = plugin.gen_mask(rgb, req)
                out_path = output / f"{stem}_mask.png"
                cv2.imencode(".png", result)[1].tofile(str(out_path))
            else:
                rgba = plugin.gen_image(rgb, req)
                out_path = output / f"{stem}.png"
                bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                cv2.imencode(".png", bgra)[1].tofile(str(out_path))

            progress.update(task, advance=1)

    logger.info(f"Done. Results saved to {output}")
