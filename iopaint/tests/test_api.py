"""Tests for iopaint.api – HTTP endpoints using FastAPI TestClient."""
import base64
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api, api_middleware, diffuser_callback
from iopaint.schema import ApiConfig, Device, InpaintRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_image(mode="RGB", size=(64, 64)):
    img = Image.new(mode, size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_mask(size=(64, 64)):
    mask = Image.new("L", size, color=0)
    # Draw a white square in the center
    pixels = np.array(mask)
    pixels[16:48, 16:48] = 255
    mask = Image.fromarray(pixels)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _minimal_config(tmp_path):
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="cv2",
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


# ---------------------------------------------------------------------------
# api_middleware
# ---------------------------------------------------------------------------

class TestApiMiddleware:
    def test_cors_headers(self):
        app = FastAPI()
        api_middleware(app)

        @app.get("/test")
        def test_route():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_exception_handling(self):
        app = FastAPI()
        api_middleware(app)

        @app.get("/error")
        def error_route():
            raise ValueError("test error")

        client = TestClient(app)
        resp = client.get("/error")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "ValueError"


# ---------------------------------------------------------------------------
# diffuser_callback
# ---------------------------------------------------------------------------

class TestDiffuserCallback:
    def test_callback_returns_dict(self):
        result = diffuser_callback(None, step=1, timestep=500)
        assert isinstance(result, dict)

    def test_callback_with_kwargs(self):
        result = diffuser_callback(None, step=1, timestep=500, callback_kwargs={"key": "val"})
        assert isinstance(result, dict)

    def test_callback_none_kwargs(self):
        result = diffuser_callback(None, step=1, timestep=500, callback_kwargs=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Api.api_server_config
# ---------------------------------------------------------------------------

class TestApiServerConfig:
    def test_server_config(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/server-config")
        assert resp.status_code == 200
        body = resp.json()
        assert "plugins" in body
        assert "modelInfos" in body
        assert "samplers" in body
        assert isinstance(body["samplers"], list)

    def test_samplers(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/samplers")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) > 0


# ---------------------------------------------------------------------------
# Api.api_current_model
# ---------------------------------------------------------------------------

class TestApiCurrentModel:
    def test_get_model(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/model")
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body
        assert "model_type" in body


# ---------------------------------------------------------------------------
# Api.api_inpaint
# ---------------------------------------------------------------------------

class TestApiInpaint:
    def test_inpaint_with_cv2(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(
            image=_b64_image(),
            mask=_b64_mask(),
            hd_strategy="Original",
            cv2_radius=3,
        )
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_inpaint_size_mismatch(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        # 64x64 image but 32x32 mask
        mask_32 = _b64_mask(size=(32, 32))
        req = InpaintRequest(
            image=_b64_image(),
            mask=mask_32,
            hd_strategy="Original",
        )
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_inpaint_empty_cache(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        cfg.empty_cache_after_inpaint = True
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(
            image=_b64_image(),
            mask=_b64_mask(),
            hd_strategy="Original",
        )
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Api.api_adjust_mask
# ---------------------------------------------------------------------------

class TestApiAdjustMask:
    def test_adjust_mask_expand(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import AdjustMaskRequest
        req = AdjustMaskRequest(mask=_b64_mask(), operate="expand", kernel_size=5)
        resp = client.post(
            "/api/v1/adjust_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_adjust_mask_shrink(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import AdjustMaskRequest
        req = AdjustMaskRequest(mask=_b64_mask(), operate="shrink", kernel_size=5)
        resp = client.post(
            "/api/v1/adjust_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_adjust_mask_reverse(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import AdjustMaskRequest
        req = AdjustMaskRequest(mask=_b64_mask(), operate="reverse")
        resp = client.post(
            "/api/v1/adjust_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Api.api_input_image
# ---------------------------------------------------------------------------

class TestApiInputImage:
    def test_no_input_returns_204(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        cfg.input = None
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/inputimage")
        assert resp.status_code == 204

    def test_valid_input_file(self, tmp_path):
        # Create a test image
        img_path = tmp_path / "test_input.png"
        img = Image.new("RGB", (32, 32), color=(100, 200, 50))
        img.save(str(img_path))

        cfg = _minimal_config(tmp_path)
        cfg.input = img_path
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/inputimage")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Api.api_save_image
# ---------------------------------------------------------------------------

class TestApiSaveImage:
    def test_save_image(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        # Create a test PNG file to upload
        img = Image.new("RGB", (16, 16), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/v1/save_image",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 200
        assert (tmp_path / "test.png").exists()


# ---------------------------------------------------------------------------
# Api.api_run_plugin_gen_mask / gen_image (plugin not found)
# ---------------------------------------------------------------------------

class TestPluginNotFound:
    def test_gen_mask_plugin_not_found(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import RunPluginRequest
        req = RunPluginRequest(name="nonexistent", image=_b64_image())
        resp = client.post(
            "/api/v1/run_plugin_gen_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_gen_image_plugin_not_found(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)
        client = TestClient(app)

        from iopaint.schema import RunPluginRequest
        req = RunPluginRequest(name="nonexistent", image=_b64_image())
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
