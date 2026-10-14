"""helper.py 缺口补测：缓存目录创建、本地路径短路、md5 校验三分支、
handle_error 两分支、load_jit_model / load_model 的加载/损坏/下载三分支、
numpy_to_bytes 兜底格式、load_img / decode_base64_to_image 的 exif 容错。

全部离线：假权重用真 torch.jit / torch.save 生成，下载函数一律打桩。
"""
import base64
import hashlib
import io
import os

import cv2
import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from iopaint import helper
from iopaint.exceptions import ModelLoadError
from iopaint.helper import (
    decode_base64_to_image,
    download_model,
    get_cache_path_by_url,
    handle_error,
    load_img,
    load_jit_model,
    load_model,
    md5sum,
    numpy_to_bytes,
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _png_bytes(size=(8, 6), color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class _JitToy(torch.nn.Module):
    def forward(self, x):
        return x * 2


def _save_jit(path):
    torch.jit.script(_JitToy()).save(str(path))


class TestGetCachePathByUrl:
    def test_creates_checkpoints_dir_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(helper, "get_dir", lambda: str(tmp_path / "hub"))

        path = get_cache_path_by_url("https://example.com/weights/model.pt")

        assert (tmp_path / "hub" / "checkpoints").is_dir()
        assert path == str(tmp_path / "hub" / "checkpoints" / "model.pt")


class TestDownloadModel:
    def test_local_path_short_circuits_download(self, tmp_path, monkeypatch):
        local = tmp_path / "model.pt"
        local.write_bytes(b"local weights")

        def boom(*args, **kwargs):
            raise AssertionError("本地已存在的路径不该触发下载")

        monkeypatch.setattr(helper, "download_url_to_file", boom)

        assert download_model(str(local), model_md5="unused") == str(local)

    def test_md5_match_logs_success(self, tmp_path, monkeypatch):
        content = b"real weights"
        cached = tmp_path / "cached.pt"
        monkeypatch.setattr(helper, "get_cache_path_by_url", lambda url: str(cached))

        def fake_download(url, dst, hash_prefix, progress=True):
            with open(dst, "wb") as fw:
                fw.write(content)

        monkeypatch.setattr(helper, "download_url_to_file", fake_download)

        messages = []
        sink = logger.add(messages.append, level="INFO")
        try:
            result = download_model(
                "https://example.com/model.pt", model_md5=_md5(content)
            )
        finally:
            logger.remove(sink)

        assert result == str(cached)
        assert "Download model success" in "".join(str(m) for m in messages)

    def test_md5_mismatch_remove_failure_still_raises(self, tmp_path, monkeypatch):
        cached = tmp_path / "cached.pt"
        monkeypatch.setattr(helper, "get_cache_path_by_url", lambda url: str(cached))

        def fake_download(url, dst, hash_prefix, progress=True):
            with open(dst, "wb") as fw:
                fw.write(b"corrupted")

        monkeypatch.setattr(helper, "download_url_to_file", fake_download)

        def boom(path):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(os, "remove", boom)

        messages = []
        sink = logger.add(messages.append, level="ERROR")
        with pytest.raises(ModelLoadError):
            try:
                download_model("https://example.com/model.pt", model_md5="0" * 32)
            finally:
                logger.remove(sink)

        text = "".join(str(m) for m in messages)
        assert "please delete" in text
        assert cached.exists()  # 删不掉 → 文件保留，等用户手动删


class TestHandleError:
    def test_md5_match_reports_load_failure(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"weights")

        messages = []
        sink = logger.add(messages.append, level="ERROR")
        with pytest.raises(ModelLoadError):
            try:
                handle_error(str(p), _md5(b"weights"), RuntimeError("cuda error"))
            finally:
                logger.remove(sink)

        text = "".join(str(m) for m in messages)
        assert "Failed to load model" in text
        assert "submit an issue" in text
        assert p.exists()  # md5 一致 → 不是坏文件，不删

    def test_md5_mismatch_remove_failure_logs_delete_hint(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pt"
        p.write_bytes(b"weights")

        def boom(path):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(os, "remove", boom)

        messages = []
        sink = logger.add(messages.append, level="ERROR")
        with pytest.raises(ModelLoadError):
            try:
                handle_error(str(p), "0" * 32, RuntimeError("boom"))
            finally:
                logger.remove(sink)

        text = "".join(str(m) for m in messages)
        assert f"please delete {p}" in text
        assert p.exists()


class TestLoadJitModel:
    def test_loads_local_jit_model(self, tmp_path):
        p = tmp_path / "model.pt"
        _save_jit(p)

        model = load_jit_model(str(p), torch.device("cpu"), md5sum(str(p)))

        assert model(torch.ones(1)).item() == 2.0
        assert model.training is False

    def test_corrupted_file_raises_and_deletes(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"not a jit model")

        with pytest.raises(ModelLoadError):
            load_jit_model(str(p), torch.device("cpu"), "0" * 32)
        assert not p.exists()  # md5 不匹配 → 坏文件被删

    def test_downloads_when_url_not_local(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pt"
        _save_jit(p)
        monkeypatch.setattr(
            helper, "download_model", lambda url, md5=None: str(p)
        )

        model = load_jit_model(
            "https://example.com/model.pt", torch.device("cpu"), "0" * 32
        )

        assert model(torch.ones(1)).item() == 2.0


class TestLoadModel:
    def test_loads_local_state_dict(self, tmp_path):
        src = torch.nn.Linear(4, 2)
        p = tmp_path / "model.pt"
        torch.save(src.state_dict(), str(p))

        model = torch.nn.Linear(4, 2)
        loaded = load_model(model, str(p), torch.device("cpu"), md5sum(str(p)))

        assert loaded is model
        assert torch.equal(loaded.weight, src.weight)
        assert model.training is False

    def test_corrupted_file_raises_and_deletes(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"not a state dict")

        with pytest.raises(ModelLoadError):
            load_model(
                torch.nn.Linear(4, 2), str(p), torch.device("cpu"), "0" * 32
            )
        assert not p.exists()

    def test_downloads_when_url_not_local(self, tmp_path, monkeypatch):
        src = torch.nn.Linear(4, 2)
        p = tmp_path / "model.pt"
        torch.save(src.state_dict(), str(p))
        monkeypatch.setattr(
            helper, "download_model", lambda url, md5=None: str(p)
        )

        loaded = load_model(
            torch.nn.Linear(4, 2),
            "https://example.com/model.pt",
            torch.device("cpu"),
            "0" * 32,
        )

        assert torch.equal(loaded.weight, src.weight)


class TestNumpyToBytes:
    def test_unknown_format_uses_no_params(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        data = numpy_to_bytes(img, "bmp")

        assert len(data) > 0
        back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert back.shape == (8, 8, 3)


class TestExifTolerance:
    def test_load_img_swallows_exif_transpose_error(self, monkeypatch):
        def boom(image, **kwargs):
            raise RuntimeError("bad exif")

        monkeypatch.setattr(helper.ImageOps, "exif_transpose", boom)

        np_img, alpha = load_img(_png_bytes())

        assert np_img.shape == (6, 8, 3)
        assert alpha is None

    def test_decode_base64_swallows_exif_transpose_error(self, monkeypatch):
        def boom(image, **kwargs):
            raise RuntimeError("bad exif")

        monkeypatch.setattr(helper.ImageOps, "exif_transpose", boom)

        encoding = base64.b64encode(_png_bytes()).decode()
        np_img, alpha, infos, ext = decode_base64_to_image(encoding)

        assert np_img.shape == (6, 8, 3)
        assert ext == "png"
