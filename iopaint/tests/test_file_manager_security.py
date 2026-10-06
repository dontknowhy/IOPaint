"""路径遍历 / 未配置目录 / Content-Type 一致性（SEC-1、SEC-2、CONS-3）。
"""
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.file_manager import FileManager


def _save_png(directory, name, size=(32, 32), color=(10, 20, 30)):
    directory.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    path = directory / name
    img.save(path, format="PNG")
    return path


def _build(tmp_path, mask_dir=None, output_dir=None):
    input_dir = tmp_path / "input"
    output_dir = output_dir if output_dir is not None else (tmp_path / "output")
    app = FastAPI()
    FileManager(app=app, input_dir=input_dir, mask_dir=mask_dir, output_dir=output_dir)
    return app, input_dir, output_dir


class TestPathTraversal:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        # 输入目录里放一张真图，宿主目录里放一个“机密”文件
        self.app, self.input_dir, self.output_dir = _build(tmp_path)
        _save_png(self.input_dir, "ok.png")
        self.secret = tmp_path / "secret.txt"
        self.secret.write_text("TOP-SECRET")
        self.client = TestClient(self.app)

    def _get(self, filename, tab="input"):
        return self.client.get(
            "/api/v1/media_file", params={"tab": tab, "filename": filename}
        )

    def test_dotdot_traversal_rejected(self):
        resp = self._get("../secret.txt")
        assert resp.status_code == 422
        assert "TOP-SECRET" not in resp.text

    def test_deep_traversal_rejected(self):
        resp = self._get("../../etc/passwd")
        assert resp.status_code == 422
        assert "root:" not in resp.text

    def test_absolute_path_rejected(self):
        resp = self._get("/etc/passwd")
        assert resp.status_code == 422
        assert "root:" not in resp.text

    def test_embedded_traversal_rejected(self):
        resp = self._get("sub/../../secret.txt")
        assert resp.status_code == 422
        assert "TOP-SECRET" not in resp.text

    def test_valid_filename_still_works(self):
        resp = self._get("ok.png")
        assert resp.status_code == 200
        assert resp.content.startswith(b"\x89PNG")

    def test_missing_file_is_422(self):
        resp = self._get("nope.png")
        assert resp.status_code == 422

    def test_thumbnail_traversal_rejected(self):
        resp = self.client.get(
            "/api/v1/media_thumbnail_file",
            params={"tab": "input", "filename": "../secret.txt", "width": 16, "height": 16},
        )
        assert resp.status_code == 422
        assert "TOP-SECRET" not in resp.text

    def test_thumbnail_valid(self):
        resp = self.client.get(
            "/api/v1/media_thumbnail_file",
            params={"tab": "input", "filename": "ok.png", "width": 16, "height": 16},
        )
        assert resp.status_code == 200
        # D-6: 缩略图 Content-Type 必须跟字节一致（源是 PNG 就不能回 image/jpeg）
        assert resp.headers["content-type"].startswith("image/png")


class TestUnconfiguredDirs:
    def test_mask_dir_not_configured_is_422_not_500(self, tmp_path):
        app, input_dir, output_dir = _build(tmp_path, mask_dir=None)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/media_file", params={"tab": "mask", "filename": "a.png"}
        )
        assert resp.status_code == 422
        assert "not configured" in resp.text

    def test_mask_dir_listing_returns_empty(self, tmp_path):
        app, input_dir, output_dir = _build(tmp_path, mask_dir=None)
        client = TestClient(app)
        resp = client.get("/api/v1/medias", params={"tab": "mask"})
        assert resp.status_code == 200
        assert resp.json() == []


class TestMediaType:
    def test_jpg_served_as_image_jpeg(self, tmp_path):
        app, input_dir, output_dir = _build(tmp_path)
        input_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), color=(1, 2, 3))
        img.save(input_dir / "a.jpg", format="JPEG")
        client = TestClient(app)
        resp = client.get(
            "/api/v1/media_file", params={"tab": "input", "filename": "a.jpg"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_png_served_as_image_png(self, tmp_path):
        app, input_dir, output_dir = _build(tmp_path)
        _save_png(input_dir, "a.png")
        client = TestClient(app)
        resp = client.get(
            "/api/v1/media_file", params={"tab": "input", "filename": "a.png"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_sniff_ignores_wrong_extension(self, tmp_path):
        """文件名扩展名与真实格式不符时，以字节为准（实测复现的错配根源）。"""
        app, input_dir, output_dir = _build(tmp_path)
        input_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), color=(1, 2, 3))
        # 真实是 PNG，却叫 .jpg
        img.save(input_dir / "lie.jpg", format="PNG")
        client = TestClient(app)
        resp = client.get(
            "/api/v1/media_file", params={"tab": "input", "filename": "lie.jpg"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"


class TestOutputDirMissing:
    def test_manager_requires_output_dir(self, tmp_path):
        """FileManager 构造期就需要 output_dir（thumbnail_directory）。

        这里记录当前行为：cli 在 --input 为目录时强制要求 --output-dir，
        所以 output_dir=None 不会走到构造函数；_safe_join 的 None 分支
        实际由 mask_dir=None 触发（见 TestUnconfiguredDirs）。
        """
        input_dir = tmp_path / "input"
        app = FastAPI()
        with pytest.raises(TypeError):
            FileManager(
                app=app, input_dir=input_dir, mask_dir=None, output_dir=None
            )
