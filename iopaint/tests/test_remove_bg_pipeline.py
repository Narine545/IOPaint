"""End-to-end test of the RemoveBG/AnimeSeg post-processing pipeline.

No model is downloaded: the segmentation step is replaced by a fake predictor,
so only IOPaint's own code (alpha matting, colour spill removal, request
overrides) is exercised.
"""

import cv2
import numpy as np
import pytest

from iopaint.plugins.anime_seg import AnimeSeg
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import AlphaMattingMethod, RunPluginRequest


def _scene(size=96, radius=30):
    """Red disc on a white background (the classic 'white studio' case)."""
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), radius, 255, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    rgb = (a * np.float32([200, 40, 40]) + (1 - a) * np.float32([255, 255, 255])).astype(
        np.uint8
    )
    return rgb, alpha


def _fake_remove_bg(alpha, **kwargs):
    """A RemoveBG instance whose model always returns ``alpha``."""
    plugin = RemoveBG.__new__(RemoveBG)
    plugin.model_name = "fake"
    plugin.device = "cpu"
    plugin.alpha_matting = kwargs.get("alpha_matting", AlphaMattingMethod.none.value)
    plugin.fg_threshold = kwargs.get("fg_threshold", 240)
    plugin.bg_threshold = kwargs.get("bg_threshold", 15)
    plugin.erode_size = kwargs.get("erode_size", 8)
    plugin.matting_max_size = kwargs.get("matting_max_size", 1280)
    plugin.remove_color_spill = kwargs.get("remove_color_spill", True)
    plugin.predict = lambda bgr: alpha
    return plugin


def _req(**kwargs):
    return RunPluginRequest(name="RemoveBG", image="", **kwargs)


def test_gen_image_returns_rgba_cutout():
    rgb, alpha = _scene()
    plugin = _fake_remove_bg((alpha > 127).astype(np.uint8) * 255)

    out = plugin.gen_image(rgb, _req())
    assert out.shape == (96, 96, 4)
    assert out.dtype == np.uint8
    assert out[48, 48, 3] == 255  # object is opaque
    assert out[1, 1, 3] == 0  # corner is transparent


def test_alpha_matting_softens_edges():
    rgb, alpha = _scene()
    hard = (alpha > 127).astype(np.uint8) * 255
    plugin = _fake_remove_bg(hard)

    hard_out = plugin.gen_mask(rgb, _req())
    soft_out = plugin.gen_mask(rgb, _req(alpha_matting=AlphaMattingMethod.fast))

    def n_semi(mask):
        return int(((mask > 20) & (mask < 235)).sum())

    assert n_semi(soft_out) > n_semi(hard_out)


def test_request_overrides_server_defaults():
    rgb, alpha = _scene()
    plugin = _fake_remove_bg(
        (alpha > 127).astype(np.uint8) * 255,
        alpha_matting=AlphaMattingMethod.fast.value,
    )

    params = plugin._matting_params(_req())
    assert params["method"] == "fast"

    params = plugin._matting_params(
        _req(alpha_matting=AlphaMattingMethod.none, fg_threshold=200, erode_size=3)
    )
    assert params["method"] == "none"
    assert params["foreground_threshold"] == 200
    assert params["erode_size"] == 3


def test_color_spill_removal_kills_the_white_halo():
    rgb, alpha = _scene()
    plugin = _fake_remove_bg(alpha)

    with_spill = plugin.gen_image(rgb, _req(remove_color_spill=False))
    without_spill = plugin.gen_image(rgb, _req(remove_color_spill=True))

    band = (alpha > 60) & (alpha < 200)
    # edge pixels must be less "white" once the old background is removed
    assert without_spill[:, :, :3][band].mean() < with_spill[:, :, :3][band].mean()
    # alpha channel is untouched by the decontamination
    assert np.array_equal(with_spill[:, :, 3], without_spill[:, :, 3])


def test_mask_is_resized_to_image():
    rgb, alpha = _scene(size=96)
    small = cv2.resize(alpha, (48, 48))
    plugin = _fake_remove_bg(small)
    assert plugin.gen_mask(rgb, _req()).shape == (96, 96)


@pytest.mark.parametrize("method", ["none", "fast", "quality"])
def test_anime_seg_pipeline(method):
    rgb, alpha = _scene()
    plugin = AnimeSeg.__new__(AnimeSeg)
    plugin.alpha_matting = method
    plugin.fg_threshold = 240
    plugin.bg_threshold = 15
    plugin.erode_size = 8
    plugin.matting_max_size = 1280
    plugin.remove_color_spill = True
    plugin.forward = lambda img: (alpha > 127).astype(np.uint8) * 255

    out = plugin.gen_image(rgb, RunPluginRequest(name="AnimeSeg", image=""))
    assert out.shape == (96, 96, 4)
    assert out[48, 48, 3] == 255
