"""Saving must never lie about the format and never overwrite silently."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.api import Api
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    RealESRGANModel,
    RemoveBGModel,
)


def make_client(output_dir: Path, save_policy: str = "unique"):
    config = ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="lama",
        no_half=False,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=False,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=output_dir,
        model_dir=output_dir.parent / "data",
        save_policy=save_policy,
        quality=95,
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.sam2_1_tiny,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model=RemoveBGModel.briaai_rmbg_1_4,
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
    )
    app = FastAPI()
    with patch("iopaint.api.ModelManager"):
        Api(app, config)
    return TestClient(app)


def transparent_png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 4), np.uint8), "RGBA").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        output_dir.mkdir()
        yield output_dir


def test_cutout_uploaded_as_jpg_is_stored_as_png(workspace):
    client = make_client(workspace)
    res = client.post(
        "/api/v1/save_image",
        files={"file": ("photo.jpg", transparent_png(), "image/jpeg")},
        data={"tool": "RemoveBG-isnet-general-use", "version": "2"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "photo_removebg-isnet-general-use_v2.png"
    assert Path(body["path"]).exists()
    assert body["replaced"] is False


def test_same_name_twice_does_not_overwrite(workspace):
    client = make_client(workspace)
    names = []
    for _ in range(3):
        res = client.post(
            "/api/v1/save_image",
            files={"file": ("photo.jpg", transparent_png(), "image/jpeg")},
            data={"tool": "lama", "version": "1"},
        )
        names.append(res.json()["filename"])
    assert len(set(names)) == 3
    assert len(list(workspace.iterdir())) == 3


def test_overwrite_policy_is_reported(workspace):
    client = make_client(workspace, save_policy="overwrite")
    first = client.post(
        "/api/v1/save_image",
        files={"file": ("photo.png", transparent_png(), "image/png")},
        data={"tool": "lama"},
    ).json()
    second = client.post(
        "/api/v1/save_image",
        files={"file": ("photo.png", transparent_png(), "image/png")},
        data={"tool": "lama"},
    ).json()
    assert first["path"] == second["path"]
    assert second["replaced"] is True


def test_non_image_is_rejected(workspace):
    client = make_client(workspace)
    res = client.post(
        "/api/v1/save_image",
        files={"file": ("evil.png", b"not an image", "image/png")},
    )
    assert res.status_code == 400


def test_storage_endpoints(workspace):
    client = make_client(workspace)
    client.post(
        "/api/v1/save_image",
        files={"file": ("photo.png", transparent_png(), "image/png")},
        data={"tool": "lama"},
    )

    info = client.get("/api/v1/storage").json()
    assert info["output_dir"] == str(workspace)
    assert {item["name"] for item in info["items"]} >= {"erase_models", "outputs"}

    dry = client.post(
        "/api/v1/storage/cleanup", json={"targets": ["outputs"], "dry_run": True}
    ).json()
    assert dry["dry_run"] is True
    assert list(workspace.iterdir()), "dry run must not delete anything"

    client.post("/api/v1/storage/cleanup", json={"targets": ["outputs"]})
    assert list(workspace.iterdir()) == []
