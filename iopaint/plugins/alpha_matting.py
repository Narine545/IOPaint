"""Alpha matting / cutout post-processing utilities.

This module is intentionally free of any ``torch`` dependency so that it can be
unit tested (and reused) without loading the heavy deep learning stack.

The typical pipeline is:

1. a segmentation model produces a coarse alpha (``0..255`` uint8 mask)
2. :func:`refine_alpha` turns that coarse alpha into a soft, edge accurate alpha
   (hair, fur, anime hair strands, semi transparent areas ...)
3. :func:`cutout` composites the RGBA result and removes the colour spill of the
   old background (very visible when the original background is white).

Two refinement methods are available:

``fast``
    Edge aware refinement based on a guided filter. Runs in a few hundred
    milliseconds even for 4K images and uses almost no extra memory. This is the
    recommended default on CPU-only machines.

``quality``
    Closed form matting (``pymatting``). Much more accurate for hair/fur, but it
    solves a large sparse linear system. To keep RAM usage sane on 16 GB
    machines the solve is done on a downscaled copy of the image
    (``max_size``) and the result is then upsampled and re-sharpened with the
    guided filter at full resolution.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "guided_filter",
    "generate_trimap",
    "post_process_mask",
    "refine_alpha",
    "estimate_background_color",
    "decontaminate",
    "cutout",
    "is_pymatting_available",
]

# Default parameters, tuned for portrait/anime cutouts on a light background.
DEFAULT_FOREGROUND_THRESHOLD = 240
DEFAULT_BACKGROUND_THRESHOLD = 15
DEFAULT_ERODE_SIZE = 8
DEFAULT_MAX_SIZE = 1280


def is_pymatting_available() -> bool:
    try:
        import pymatting  # noqa: F401

        return True
    except Exception:
        return False


def _as_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-4
) -> np.ndarray:
    """Edge preserving filter (He et al. 2010), grayscale guide.

    ``guide`` and ``src`` are float32 arrays in ``[0, 1]``.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    radius = max(1, int(radius))
    ksize = (radius * 2 + 1, radius * 2 + 1)

    def box(x):
        return cv2.boxFilter(x, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_g = box(guide)
    mean_s = box(src)
    corr_gg = box(guide * guide)
    corr_gs = box(guide * src)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = box(a)
    mean_b = box(b)
    return mean_a * guide + mean_b


def post_process_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove speckles / fill pinholes in a binary-ish mask (uint8)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return np.clip(out.astype(np.int32) * 2 - 255, 0, 255).astype(np.uint8)


def generate_trimap(
    alpha: np.ndarray,
    foreground_threshold: int = DEFAULT_FOREGROUND_THRESHOLD,
    background_threshold: int = DEFAULT_BACKGROUND_THRESHOLD,
    erode_size: int = DEFAULT_ERODE_SIZE,
) -> np.ndarray:
    """Build a trimap (0 = background, 128 = unknown, 255 = foreground)."""
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]

    is_fg = (alpha > foreground_threshold).astype(np.uint8)
    is_bg = (alpha < background_threshold).astype(np.uint8)

    if erode_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size * 2 + 1, erode_size * 2 + 1)
        )
        is_fg = cv2.erode(is_fg, kernel, iterations=1)
        is_bg = cv2.erode(is_bg, kernel, iterations=1)

    trimap = np.full(alpha.shape[:2], 128, dtype=np.uint8)
    trimap[is_fg > 0] = 255
    trimap[is_bg > 0] = 0
    return trimap


def _resize_to_max(img: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_size <= 0 or longest <= max_size:
        return img, 1.0
    scale = max_size / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def _closed_form_alpha(
    rgb: np.ndarray, trimap: np.ndarray, max_size: int
) -> Optional[np.ndarray]:
    """Closed form matting through pymatting. Returns float32 alpha in [0, 1]."""
    try:
        from pymatting import estimate_alpha_cf
    except Exception:
        return None

    small_rgb, scale = _resize_to_max(rgb, max_size)
    if scale != 1.0:
        small_trimap = cv2.resize(
            trimap, (small_rgb.shape[1], small_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        small_trimap = trimap

    # pymatting is compiled (numba) for float64 inputs
    image_f64 = _as_float(small_rgb).astype(np.float64)
    trimap_f64 = _as_float(small_trimap).astype(np.float64)
    if trimap_f64.ndim == 3:
        trimap_f64 = trimap_f64[:, :, 0]

    alpha = estimate_alpha_cf(image_f64, trimap_f64).astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)

    if scale != 1.0:
        alpha = cv2.resize(
            alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return alpha


def refine_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    method: str = "fast",
    foreground_threshold: int = DEFAULT_FOREGROUND_THRESHOLD,
    background_threshold: int = DEFAULT_BACKGROUND_THRESHOLD,
    erode_size: int = DEFAULT_ERODE_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> np.ndarray:
    """Refine a coarse alpha mask. Returns uint8 alpha with the shape of ``rgb``.

    ``rgb``  : HxWx3 uint8 RGB image
    ``alpha``: HxW (or HxWx1) uint8 coarse mask coming from a segmentation model
    """
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(
            alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    method = (method or "none").lower()
    if method in ("none", "off", "disable", "disabled"):
        return alpha

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    guide = _as_float(gray)
    coarse = _as_float(alpha)

    # Radius follows the image size so that behaviour is resolution independent.
    radius = max(2, int(round(min(rgb.shape[:2]) / 250)))

    refined = None
    if method == "quality":
        trimap = generate_trimap(
            alpha, foreground_threshold, background_threshold, erode_size
        )
        refined = _closed_form_alpha(rgb, trimap, max_size)
        if refined is not None:
            # Re-sharpen the upsampled alpha using the full resolution guide.
            refined = guided_filter(guide, refined, radius=radius, eps=1e-5)
            known_fg = trimap == 255
            known_bg = trimap == 0
            refined[known_fg] = 1.0
            refined[known_bg] = 0.0

    if refined is None:
        # fast (guided filter) path, also the fallback when pymatting is missing
        refined = guided_filter(guide, coarse, radius=radius, eps=1e-4)
        lo = background_threshold / 255.0
        hi = max(foreground_threshold / 255.0, lo + 1e-3)
        refined = (refined - lo) / (hi - lo)

    refined = np.clip(refined, 0.0, 1.0)
    return (refined * 255.0 + 0.5).astype(np.uint8)


def estimate_background_color(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Median colour of the pixels that are confidently background."""
    bg_pixels = rgb[alpha < 16]
    if bg_pixels.size < 30:
        # fall back to the image border
        border = np.concatenate(
            [
                rgb[0, :, :].reshape(-1, 3),
                rgb[-1, :, :].reshape(-1, 3),
                rgb[:, 0, :].reshape(-1, 3),
                rgb[:, -1, :].reshape(-1, 3),
            ]
        )
        bg_pixels = border
    return np.median(bg_pixels.reshape(-1, 3), axis=0).astype(np.float32)


def decontaminate(
    rgb: np.ndarray, alpha: np.ndarray, bg_color: Optional[np.ndarray] = None
) -> np.ndarray:
    """Remove the old background colour bleeding into semi transparent pixels.

    Solving ``I = a * F + (1 - a) * B`` for ``F`` removes the bright halo you get
    when cutting a subject out of a white background.
    """
    if bg_color is None:
        bg_color = estimate_background_color(rgb, alpha)

    a = _as_float(alpha)[:, :, None]
    img = _as_float(rgb)
    bg = (bg_color.reshape(1, 1, 3) / 255.0).astype(np.float32)

    # Only touch the transition band, keep opaque pixels untouched.
    safe_a = np.maximum(a, 1e-3)
    fg = (img - (1.0 - a) * bg) / safe_a
    fg = np.clip(fg, 0.0, 1.0)

    blend = np.clip((a - 0.02) / 0.96, 0.0, 1.0)
    out = fg * (1.0 - blend) + img * blend
    # fully opaque pixels must stay exactly as they were
    out = np.where(a >= 0.995, img, out)
    return (out * 255.0 + 0.5).astype(np.uint8)


def cutout(
    rgb: np.ndarray, alpha: np.ndarray, remove_color_spill: bool = True
) -> np.ndarray:
    """Compose an RGBA cutout from an RGB image and an uint8 alpha mask."""
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    rgb_out = decontaminate(rgb, alpha) if remove_color_spill else rgb
    return np.dstack([rgb_out, alpha]).astype(np.uint8)
