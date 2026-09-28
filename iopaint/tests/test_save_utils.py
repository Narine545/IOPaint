import importlib.util
from pathlib import Path

import pytest

CURRENT_DIR = Path(__file__).parent.absolute().resolve()
ROOT = CURRENT_DIR.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


save_utils = _load("save_utils")
storage = _load("storage")

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
    b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 16


class TestFileNames:
    def test_tool_and_version_in_name(self):
        assert (
            save_utils.build_filename("photo.jpg", "lama", "png", 2)
            == "photo_lama_v2.png"
        )

    def test_no_version_for_first_result(self):
        assert save_utils.build_filename("photo.jpg", "lama", "png", 1) == (
            "photo_lama.png"
        )
        assert save_utils.build_filename("photo.jpg", "lama", "png", None) == (
            "photo_lama.png"
        )

    def test_plugin_name_is_slugified(self):
        name = save_utils.build_filename(
            "My Photo.JPG", "RemoveBG-isnet-general-use", "png", 3
        )
        assert name == "My Photo_removebg-isnet-general-use_v3.png"

    def test_path_traversal_is_stripped(self):
        name = save_utils.build_filename("../../etc/passwd", "lama", "png", 1)
        assert "/" not in name and ".." not in name

    def test_extension_follows_real_bytes_not_original_name(self):
        # the bug the user hit: a transparent cutout saved as .jpg
        assert save_utils.image_ext_from_bytes(PNG_BYTES, has_alpha=True) == "png"
        assert save_utils.image_ext_from_bytes(JPEG_BYTES, has_alpha=True) == "png"
        assert save_utils.image_ext_from_bytes(JPEG_BYTES, has_alpha=False) == "jpg"


class TestOverwriteProtection:
    def test_unique_never_overwrites(self, tmp_path):
        (tmp_path / "a.png").write_bytes(PNG_BYTES)
        path, replaced = save_utils.resolve_save_path(tmp_path, "a.png", "unique")
        assert path.name == "a_2.png"
        assert replaced is False

        path.write_bytes(PNG_BYTES)
        path2, _ = save_utils.resolve_save_path(tmp_path, "a.png", "unique")
        assert path2.name == "a_3.png"

    def test_timestamp_policy(self, tmp_path):
        (tmp_path / "a.png").write_bytes(PNG_BYTES)
        path, replaced = save_utils.resolve_save_path(tmp_path, "a.png", "timestamp")
        assert path.name != "a.png" and path.name.endswith(".png")
        assert replaced is False

    def test_overwrite_is_explicit_and_reported(self, tmp_path):
        (tmp_path / "a.png").write_bytes(PNG_BYTES)
        path, replaced = save_utils.resolve_save_path(tmp_path, "a.png", "overwrite")
        assert path.name == "a.png"
        assert replaced is True

    def test_free_name_is_used_as_is(self, tmp_path):
        path, replaced = save_utils.resolve_save_path(tmp_path, "a.png", "unique")
        assert path.name == "a.png"
        assert replaced is False


class TestInfos:
    def test_provenance_fields(self):
        infos = save_utils.build_infos(
            {"parameters": "x"}, tool="RemoveBG-isnet", model="isnet-general-use"
        )
        assert infos["parameters"] == "x"
        assert infos["iopaint-tool"] == "RemoveBG-isnet"
        assert infos["iopaint-model"] == "isnet-general-use"
        assert "iopaint-version" in infos
        assert infos["Software"].startswith("IOPaint ")


class TestStorage:
    def test_paths_are_inside_data_dir(self, tmp_path):
        paths = storage.storage_paths(tmp_path, tmp_path / "out")
        assert "_total" in paths
        for name, item in paths.items():
            if name in ("_total", "outputs"):
                continue
            assert str(tmp_path) in item["path"]

    def test_cleanup_removes_only_requested_target(self, tmp_path):
        models = tmp_path / "torch" / "hub" / "checkpoints"
        models.mkdir(parents=True)
        (models / "big.pt").write_bytes(b"0" * 2048)
        hf = tmp_path / "huggingface"
        hf.mkdir()
        (hf / "x.bin").write_bytes(b"0" * 1024)

        result = storage.cleanup(tmp_path, ["erase_models"])
        assert not models.exists()
        assert hf.exists()
        assert result["freed"] >= 2048

    def test_dry_run_deletes_nothing(self, tmp_path):
        models = tmp_path / "torch" / "hub" / "checkpoints"
        models.mkdir(parents=True)
        (models / "big.pt").write_bytes(b"0" * 2048)

        result = storage.cleanup(tmp_path, ["erase_models"], dry_run=True)
        assert models.exists()
        assert result["freed"] >= 2048
        assert result["dry_run"] is True

    def test_unknown_target_is_reported(self, tmp_path):
        result = storage.cleanup(tmp_path, ["not-a-target"])
        assert result["errors"]

    def test_outputs_require_explicit_dir(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "r.png").write_bytes(PNG_BYTES)
        storage.cleanup(tmp_path, ["outputs"], output_dir=out)
        # the folder itself stays, the files are gone
        assert out.exists()
        assert list(out.iterdir()) == []

    def test_human_size(self):
        assert storage.human_size(0) == "0 B"
        assert storage.human_size(1536).startswith("1.5")
        assert storage.human_size(3 * 1024**3).endswith("GB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
