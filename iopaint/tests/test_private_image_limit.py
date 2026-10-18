"""按来源（内网/外网）区分图片像素上限。

外网来源沿用 1 亿像素上限（>上限告警、>2×上限抛错，与 Pillow 原生语义一致），
内网来源（回环 / 私有网段 / 链路本地）不限制。判定依据 request.client.host，
直连部署下即真实来源 IP。这里用 TestClient 的 client= 注入来源，并把
api.EXTERNAL_MAX_IMAGE_PIXELS 调小到 4096，避免真的构造上亿像素的图。
"""
import base64
import io
import warnings
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from iopaint import api as api_mod
from iopaint.api import Api, _max_pixels_for_request
from iopaint.helper import (
    EXTERNAL_MAX_IMAGE_PIXELS,
    check_image_pixels,
    decode_base64_to_image,
    is_private_host,
)
from iopaint.schema import ApiConfig, InpaintRequest

PRIVATE_CLIENT = ("192.168.1.50", 12345)
PUBLIC_CLIENT = ("203.0.113.9", 12345)


def _b64_png(size: tuple[int, int], mode: str = "RGB") -> str:
    img = Image.new(mode, size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_mask(size: tuple[int, int]) -> str:
    pixels = np.zeros((size[1], size[0]), dtype="uint8")
    pixels[1:-1, 1:-1] = 255
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _minimal_config(tmp_path: Path) -> ApiConfig:
    return ApiConfig.model_validate(
        {
            "host": "127.0.0.1",
            "port": 8080,
            "inbrowser": False,
            "model": "cv2",
            "no_half": True,
            "low_mem": False,
            "cpu_offload": False,
            "disable_nsfw_checker": True,
            "local_files_only": True,
            "cpu_textencoder": False,
            "device": "cpu",
            "input": None,
            "mask_dir": None,
            "output_dir": tmp_path,
            "quality": 95,
            "empty_cache_after_inpaint": False,
            "enable_interactive_seg": False,
            "interactive_seg_model": "vit_b",
            "interactive_seg_device": "cpu",
            "enable_remove_bg": False,
            "remove_bg_device": "cpu",
            "remove_bg_model": "u2net",
            "enable_anime_seg": False,
            "enable_realesrgan": False,
            "realesrgan_device": "cpu",
            "realesrgan_model": "realesr-general-x4v3",
            "enable_gfpgan": False,
            "gfpgan_device": "cpu",
            "enable_restoreformer": False,
            "restoreformer_device": "cpu",
        }
    )


class _FakeRequest:
    """只带 client.host 的 Request 替身，供 _max_pixels_for_request 单测。"""

    class _Client:
        def __init__(self, host: str) -> None:
            self.host = host

    def __init__(self, host: str) -> None:
        self.client = self._Client(host)


def _fake_request(host: str) -> Request:
    return cast(Request, _FakeRequest(host))


class TestIsPrivateHost:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.1.2.3",
            "::1",
            "10.0.0.1",
            "192.168.31.7",
            "172.16.0.9",
            "172.31.255.254",
            "169.254.1.1",
            "fe80::1",
            "fc00::1",
            "::ffff:192.168.1.5",
            "localhost",
        ],
    )
    def test_private_sources(self, host: str) -> None:
        assert is_private_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "172.32.0.1",  # 紧邻 172.16/12 之外
            "11.0.0.1",
            "203.0.113.9",
            "2001:4860:4860::8888",
            "::ffff:8.8.8.8",
            "testclient",  # TestClient 默认的非 IP 来源
            "",
            None,
        ],
    )
    def test_public_or_unknown_sources(self, host: str) -> None:
        assert is_private_host(host) is False


class TestCheckImagePixels:
    def test_none_limit_skips_check(self) -> None:
        # 内网：即便像素数巨大也不告警、不抛错
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_image_pixels((100_000, 100_000), None)
        assert not record

    def test_at_limit_is_ok(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_image_pixels((100, 100), 10_000)
        assert not record

    def test_over_limit_warns_only(self) -> None:
        # 外网：超过上限但未到 2× 上限只告警，仍可继续处理（Pillow 原生语义）
        with pytest.warns(Image.DecompressionBombWarning):
            check_image_pixels((101, 100), 10_000)

    def test_over_double_limit_raises(self) -> None:
        with pytest.raises(Image.DecompressionBombError):
            check_image_pixels((201, 100), 10_000)


class TestMaxPixelsForRequest:
    def test_private_request_is_unlimited(self) -> None:
        assert _max_pixels_for_request(_fake_request("192.168.1.5")) is None

    def test_public_request_uses_external_limit(self) -> None:
        assert (
            _max_pixels_for_request(_fake_request("8.8.8.8"))
            == EXTERNAL_MAX_IMAGE_PIXELS
        )

    def test_missing_request_is_external(self) -> None:
        assert _max_pixels_for_request(None) == EXTERNAL_MAX_IMAGE_PIXELS


class TestDecodeMaxPixels:
    def test_decode_private_allows_over_limit(self) -> None:
        # 100x100=10000 像素，limit=4096 时本应抛错；unlimited(None) 应正常解码
        _, _, _, ext = decode_base64_to_image(_b64_png((100, 100)), max_pixels=None)
        assert ext == "png"

    def test_decode_external_rejects_double_limit(self) -> None:
        with pytest.raises(Image.DecompressionBombError):
            decode_base64_to_image(_b64_png((100, 100)), max_pixels=4096)


class TestApiSourceDependentLimit:
    """端到端：同一张超限图，内网来源放行、外网来源按原上限拦截。"""

    CAP = 4096

    def _client(
        self, tmp_path: Path, client: tuple[str, int]
    ) -> TestClient:
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        return TestClient(app, client=client, raise_server_exceptions=False)

    def _post_inpaint(
        self, client: TestClient, size: tuple[int, int]
    ) -> httpx.Response:
        req = InpaintRequest.model_validate(
            {"image": _b64_png(size), "mask": _b64_mask(size), "hd_strategy": "Original"}
        )
        return client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )

    def test_private_client_bypasses_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_mod, "EXTERNAL_MAX_IMAGE_PIXELS", self.CAP)
        client = self._client(tmp_path, PRIVATE_CLIENT)
        # 10000 像素 > 2×CAP，外网会直接报错；内网应正常出图
        resp = self._post_inpaint(client, (100, 100))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_public_client_over_double_limit_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_mod, "EXTERNAL_MAX_IMAGE_PIXELS", self.CAP)
        client = self._client(tmp_path, PUBLIC_CLIENT)
        resp = self._post_inpaint(client, (100, 100))
        assert resp.status_code == 500
        assert resp.json()["error"] == "DecompressionBombError"

    def test_public_client_within_double_limit_still_processes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 6400 像素 > CAP 但 < 2×CAP：与 Pillow 原生一样只告警，不拦截
        monkeypatch.setattr(api_mod, "EXTERNAL_MAX_IMAGE_PIXELS", self.CAP)
        client = self._client(tmp_path, PUBLIC_CLIENT)
        resp = self._post_inpaint(client, (80, 80))
        assert resp.status_code == 200
