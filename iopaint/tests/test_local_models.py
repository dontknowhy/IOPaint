"""Tests for locally available erase models (lama, fcf, cv2).

These models are already downloaded and don't need GPU.
Run: ``pytest iopaint/tests/test_local_models.py -v``
"""
import base64
import io
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api
from iopaint.download import scan_models
from iopaint.helper import load_img
from iopaint.schema import ApiConfig, Device, HDStrategy, InpaintRequest
from iopaint.tests.utils import form_fields, multipart

# 依赖本地已下载的 lama/fcf 权重，没有就 assert 失败
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Discover which local models are actually available
# ---------------------------------------------------------------------------

_available = scan_models()
_LOCAL_MODELS = [m.name for m in _available if m.model_type.value == "inpaint"]
assert len(_LOCAL_MODELS) > 0, "No local erase models found – run the download first"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png(size: tuple = (128, 128)) -> bytes:
    img = Image.new("RGB", size, color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_mask(size: tuple = (128, 128)) -> bytes:
    mask = np.zeros(size, dtype=np.uint8)
    mask[32:96, 32:96] = 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return buf.getvalue()


def _b64_mask(size: tuple = (128, 128)) -> str:
    """/adjust_mask 仍是 JSON + base64 的端点。"""
    return base64.b64encode(_png_mask(size)).decode()


def _make_config(tmp_path, model_name, **overrides):
    cfg = ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model=model_name,
        no_half=True,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=True,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=tmp_path,
        quality=100,
        empty_cache_after_inpaint=False,
        enable_interactive_seg=False,
        interactive_seg_model="vit_b",
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model="u2net",
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model="realesr-general-x4v3",
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Test each local model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", _LOCAL_MODELS)
class TestLocalModelInpaint:
    """Inpaint with each locally available erase model via the HTTP API."""

    def test_inpaint_original(self, model_name, tmp_path):
        cfg = _make_config(tmp_path, model_name)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        t0 = time.time()
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(form_fields(req), image=_png(), mask=_png_mask()),
        )
        elapsed = (time.time() - t0) * 1000
        assert resp.status_code == 200, f"{model_name} failed: {resp.text}"
        assert resp.headers["content-type"].startswith("image/")
        # Response should be valid image bytes
        img_bytes = resp.content
        assert len(img_bytes) > 0
        # Decode to verify it's a valid image
        np_img, _ = load_img(img_bytes)
        assert np_img.shape[2] == 3
        print(f"  {model_name} ORIGINAL: {elapsed:.0f}ms, output {np_img.shape}")

    def test_inpaint_resize(self, model_name, tmp_path):
        cfg = _make_config(tmp_path, model_name)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(hd_strategy=HDStrategy.RESIZE, hd_strategy_resize_limit=256)
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(form_fields(req), image=_png(), mask=_png_mask()),
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_inpaint_crop(self, model_name, tmp_path):
        cfg = _make_config(tmp_path, model_name)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=16,
            hd_strategy_crop_trigger_size=64,
        )
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(form_fields(req), image=_png(), mask=_png_mask()),
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_adjust_mask_expand(self, model_name, tmp_path):
        cfg = _make_config(tmp_path, model_name)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import AdjustMaskRequest
        req = AdjustMaskRequest(mask=_b64_mask(), operate="expand", kernel_size=5)
        resp = client.post(
            "/api/v1/adjust_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_model_info(self, model_name, tmp_path):
        """GET /api/v1/model should report the correct model."""
        cfg = _make_config(tmp_path, model_name)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        resp = client.get("/api/v1/model")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == model_name
        assert body["model_type"] == "inpaint"


# ---------------------------------------------------------------------------
# Cross-model switching
# ---------------------------------------------------------------------------

class TestModelSwitching:
    def test_switch_model(self, tmp_path):
        if len(_LOCAL_MODELS) < 2:
            pytest.skip("Need at least 2 local models to test switching")
        m1, m2 = _LOCAL_MODELS[0], _LOCAL_MODELS[1]

        cfg = _make_config(tmp_path, m1)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        # Start with m1
        resp = client.get("/api/v1/model")
        assert resp.json()["name"] == m1

        # Switch to m2
        from iopaint.schema import SwitchModelRequest
        resp = client.post(
            "/api/v1/model",
            content=SwitchModelRequest(name=m2).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == m2

        # Inpaint with m2
        req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(form_fields(req), image=_png(), mask=_png_mask()),
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_rgba_image(self, tmp_path):
        """Inpaint with an RGBA image (has alpha channel)."""
        cfg = _make_config(tmp_path, "cv2")
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        # Create RGBA image
        img = Image.new("RGBA", (128, 128), color=(200, 100, 50, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_rgba = buf.getvalue()

        req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(form_fields(req), image=png_rgba, mask=_png_mask()),
        )
        assert resp.status_code == 200

    def test_small_image(self, tmp_path):
        """Inpaint with a very small image."""
        cfg = _make_config(tmp_path, "cv2")
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        img = Image.new("RGB", (16, 16), color=(100, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_small = buf.getvalue()

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        mask_buf = io.BytesIO()
        Image.fromarray(mask).save(mask_buf, format="PNG")
        png_mask_small = mask_buf.getvalue()

        req = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(
                form_fields(req), image=png_small, mask=png_mask_small
            ),
        )
        assert resp.status_code == 200

    def test_large_image_resize_strategy(self, tmp_path):
        """Larger image with RESIZE strategy."""
        cfg = _make_config(tmp_path, "cv2")
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        img = Image.new("RGB", (600, 400), color=(100, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_large = buf.getvalue()

        mask = np.zeros((400, 600), dtype=np.uint8)
        mask[100:300, 150:450] = 255
        mask_buf = io.BytesIO()
        Image.fromarray(mask).save(mask_buf, format="PNG")
        png_mask_large = mask_buf.getvalue()

        req = InpaintRequest(
            hd_strategy=HDStrategy.RESIZE, hd_strategy_resize_limit=512
        )
        resp = client.post(
            "/api/v1/inpaint",
            **multipart(
                form_fields(req), image=png_large, mask=png_mask_large
            ),
        )
        assert resp.status_code == 200
        np_img, _ = load_img(resp.content)
        # Output should preserve original dimensions
        assert np_img.shape[:2] == (400, 600)
