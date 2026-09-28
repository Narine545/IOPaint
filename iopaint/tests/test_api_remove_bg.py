"""API level test for the background removal endpoints.

Uses the `cv2` erase model (no weights to download) and a fake RemoveBG plugin,
so it runs offline and in a few seconds.
"""

import base64
import io

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image

from iopaint.api import Api
from iopaint.schema import AlphaMattingMethod, ApiConfig, Device, RunPluginRequest


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _scene(size=64, radius=20):
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), radius, 255, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    rgb = (a * np.float32([200, 40, 40]) + (1 - a) * np.float32([255, 255, 255])).astype(
        np.uint8
    )
    return rgb, alpha


class FakeRemoveBG:
    name = "RemoveBG"
    support_gen_image = True
    support_gen_mask = True

    def __init__(self, alpha):
        from iopaint.plugins.remove_bg import RemoveBG

        self._impl = RemoveBG.__new__(RemoveBG)
        self._impl.model_name = "fake"
        self._impl.device = "cpu"
        self._impl.alpha_matting = AlphaMattingMethod.fast.value
        self._impl.fg_threshold = 240
        self._impl.bg_threshold = 15
        self._impl.erode_size = 8
        self._impl.matting_max_size = 1280
        self._impl.remove_color_spill = True
        self._impl.predict = lambda bgr: alpha

    def gen_image(self, rgb, req):
        return self._impl.gen_image(rgb, req)

    def gen_mask(self, rgb, req):
        return self._impl.gen_mask(rgb, req)


@pytest.fixture(scope="module")
def api():
    config = ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="cv2",
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=True,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=None,
        quality=95,
        enable_interactive_seg=False,
        interactive_seg_model="sam2_1_tiny",
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model="briaai/RMBG-1.4",
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model="realesr-general-x4v3",
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
    )
    return Api(FastAPI(), config)


def test_server_config_exposes_alpha_matting(api):
    cfg = api.api_server_config()
    assert cfg.removeBGAlphaMatting == AlphaMattingMethod.none
    assert cfg.alphaMattingMethods == ["none", "fast", "quality"]


def test_run_plugin_gen_image_returns_transparent_png(api):
    rgb, alpha = _scene()
    api.plugins["RemoveBG"] = FakeRemoveBG((alpha > 127).astype(np.uint8) * 255)

    resp = api.api_run_plugin_gen_image(
        RunPluginRequest(
            name="RemoveBG",
            image=_png_b64(rgb),
            alpha_matting=AlphaMattingMethod.fast,
        )
    )
    assert resp.media_type == "image/png"

    out = np.array(Image.open(io.BytesIO(resp.body)))
    assert out.shape[2] == 4
    assert out[32, 32, 3] == 255
    assert out[1, 1, 3] == 0


def test_run_plugin_gen_mask(api):
    rgb, alpha = _scene()
    api.plugins["RemoveBG"] = FakeRemoveBG((alpha > 127).astype(np.uint8) * 255)

    resp = api.api_run_plugin_gen_mask(
        RunPluginRequest(name="RemoveBG", image=_png_b64(rgb))
    )
    out = np.array(Image.open(io.BytesIO(resp.body)))
    assert out.ndim in (2, 3)
