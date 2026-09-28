"""Tests for the torch-free alpha matting helpers.

These tests intentionally only need numpy/opencv (+ optionally pymatting), so
they run fast and without downloading any model.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import the module directly from its file: it is deliberately torch-free, so
# the tests can run without the (heavy) deep learning stack installed.
_spec = importlib.util.spec_from_file_location(
    "iopaint_alpha_matting",
    Path(__file__).resolve().parents[1] / "plugins" / "alpha_matting.py",
)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)

cutout = am.cutout
decontaminate = am.decontaminate
estimate_background_color = am.estimate_background_color
generate_trimap = am.generate_trimap
guided_filter = am.guided_filter
is_pymatting_available = am.is_pymatting_available
post_process_mask = am.post_process_mask
refine_alpha = am.refine_alpha


def _synthetic_scene(size=128, radius=40, bg=(255, 255, 255), fg=(200, 40, 40)):
    """A red disc on a (by default white) background + its ground truth alpha."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :] = bg
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), radius, 255, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)

    a = alpha.astype(np.float32)[:, :, None] / 255.0
    rgb = (a * np.array(fg, np.float32) + (1 - a) * np.array(bg, np.float32)).astype(
        np.uint8
    )
    return rgb, alpha


def test_guided_filter_preserves_range():
    rng = np.random.default_rng(0)
    guide = rng.random((64, 64)).astype(np.float32)
    src = rng.random((64, 64)).astype(np.float32)
    out = guided_filter(guide, src, radius=4)
    assert out.shape == src.shape
    assert out.dtype == np.float32
    assert -0.2 <= out.min() and out.max() <= 1.2


def test_generate_trimap_values():
    _, alpha = _synthetic_scene()
    trimap = generate_trimap(alpha, 240, 15, 6)
    assert set(np.unique(trimap)).issubset({0, 128, 255})
    # all three regions must exist for a soft edged object
    assert (trimap == 255).any() and (trimap == 0).any() and (trimap == 128).any()


def test_post_process_mask_removes_speckles():
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), 20, 255, -1)
    mask[2, 2] = 255  # single pixel noise
    out = post_process_mask(mask)
    assert out[2, 2] == 0
    assert out[32, 32] == 255


@pytest.mark.parametrize("method", ["none", "fast", "quality"])
def test_refine_alpha_shapes_and_accuracy(method):
    rgb, alpha = _synthetic_scene()
    # simulate a coarse/binary model output
    coarse = (alpha > 127).astype(np.uint8) * 255

    refined = refine_alpha(rgb, coarse, method=method, max_size=128)
    assert refined.shape == rgb.shape[:2]
    assert refined.dtype == np.uint8

    # interior stays opaque, far background stays transparent
    assert refined[64, 64] > 240
    assert refined[2, 2] < 15

    if method != "none":
        # refinement must not be worse than the coarse mask
        err_coarse = np.abs(coarse.astype(int) - alpha.astype(int)).mean()
        err_refined = np.abs(refined.astype(int) - alpha.astype(int)).mean()
        assert err_refined <= err_coarse + 1.0


def test_refine_alpha_resizes_mismatched_mask():
    rgb, alpha = _synthetic_scene(size=128)
    small = cv2.resize(alpha, (64, 64))
    refined = refine_alpha(rgb, small, method="fast")
    assert refined.shape == (128, 128)


def test_estimate_background_color_white():
    rgb, alpha = _synthetic_scene(bg=(255, 255, 255))
    bg = estimate_background_color(rgb, alpha)
    assert np.allclose(bg, [255, 255, 255], atol=2)


def test_decontaminate_removes_white_halo():
    rgb, alpha = _synthetic_scene(bg=(255, 255, 255), fg=(200, 40, 40))
    out = decontaminate(rgb, alpha)

    band = (alpha > 60) & (alpha < 200)
    assert band.any()
    # in the transition band the colour must move towards the foreground colour
    before = rgb[band].astype(np.float32).mean(axis=0)
    after = out[band].astype(np.float32).mean(axis=0)
    assert after.mean() < before.mean()
    assert np.abs(after - np.array([200, 40, 40])).mean() < np.abs(
        before - np.array([200, 40, 40])
    ).mean()

    # opaque pixels must be untouched
    opaque = alpha >= 254
    assert np.array_equal(out[opaque], rgb[opaque])


def test_cutout_returns_rgba():
    rgb, alpha = _synthetic_scene()
    rgba = cutout(rgb, alpha)
    assert rgba.shape == (128, 128, 4)
    assert rgba.dtype == np.uint8
    assert np.array_equal(rgba[:, :, 3], alpha)


def test_quality_falls_back_without_pymatting(monkeypatch):
    rgb, alpha = _synthetic_scene()

    monkeypatch.setattr(am, "_closed_form_alpha", lambda *a, **k: None)
    out = am.refine_alpha(rgb, alpha, method="quality")
    assert out.shape == rgb.shape[:2]


@pytest.mark.skipif(not is_pymatting_available(), reason="pymatting not installed")
def test_quality_downscales_big_images():
    rgb, alpha = _synthetic_scene(size=512, radius=160)
    coarse = (alpha > 127).astype(np.uint8) * 255
    refined = refine_alpha(rgb, coarse, method="quality", max_size=128)
    assert refined.shape == (512, 512)
