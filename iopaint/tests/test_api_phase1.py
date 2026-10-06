"""api 层 Phase 1 行为：CORS、请求体上限、image/mask 缺省、事件循环不被阻塞。
"""
import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api, api_middleware
from iopaint.schema import ApiConfig, Device, InpaintRequest


def _b64_image(mode="RGB", size=(64, 64)):
    img = Image.new(mode, size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_mask(size=(64, 64)):
    mask = Image.new("L", size, color=0)
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
        quality=95,
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


class TestCorsHardening:
    def test_no_credentials_header_with_wildcard_origin(self):
        app = FastAPI()
        api_middleware(app)

        @app.get("/t")
        def t():
            return {"ok": 1}

        client = TestClient(app)
        resp = client.get(
            "/t", headers={"Origin": "http://evil.example"}
        )
        assert resp.status_code == 200
        # SEC-3: "origins=* + credentials=true" 是规范禁止的组合，必须不再出现
        assert "access-control-allow-credentials" not in {
            k.lower() for k in resp.headers.keys()
        }

    def test_cors_still_works_for_dev_origin(self):
        """保持 allow_origins=* —— dev 的 5173 跨源依赖它。"""
        app = FastAPI()
        api_middleware(app)

        @app.get("/t")
        def t():
            return {"ok": 1}

        client = TestClient(app)
        resp = client.get(
            "/t",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "*"


class TestBodyLimit:
    def _app(self):
        app = FastAPI()
        api_middleware(app)

        @app.post("/echo")
        def echo():
            return {"ok": 1}

        return app

    def test_oversize_body_413(self):
        client = TestClient(self._app())
        resp = client.post(
            "/echo",
            content=b"x" * 1024,
            headers={"Content-Type": "application/json"},
        )
        # 直接设一个超限的 content-length 头（服务器按头判断）
        assert resp.status_code == 200  # 1KB 正常

        resp = client.post(
            "/echo",
            content=b"x",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(512 * 1024 * 1024),
            },
        )
        assert resp.status_code == 413
        assert resp.json()["error"] == "RequestEntityTooLarge"


class TestInpaintMissingImage:
    def test_missing_image_is_400_not_500(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(image=None, mask=_b64_mask())
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_missing_mask_is_400(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        req = InpaintRequest(image=_b64_image(), mask=None)
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_oversize_image_field_422(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/inpaint",
            content='{"image": "%s"}' % ("x" * 64_000_001),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestCropperRectValidation:
    def _post(self, tmp_path, **kwargs):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)
        req = InpaintRequest(
            image=_b64_image(), mask=_b64_mask(), hd_strategy="Original", **kwargs
        )
        return client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )

    def test_cropper_far_outside_image_is_400(self, tmp_path):
        """cropper 框完全在 64x64 图之外 → 400，而不是 forward 时的空切片错误。"""
        resp = self._post(
            tmp_path, use_croper=True, croper_x=10_000, croper_y=10_000
        )
        assert resp.status_code == 400
        assert "cropper" in resp.json()["detail"]

    def test_cropper_partially_outside_is_ok(self, tmp_path):
        """部分超出是正常的（_apply_cropper 会 clamp），不能误伤。"""
        resp = self._post(
            tmp_path, use_croper=True, croper_x=32, croper_y=32, croper_width=1000
        )
        assert resp.status_code == 200

    def test_cropper_disabled_ignores_coords(self, tmp_path):
        resp = self._post(tmp_path, use_croper=False, croper_x=9_999)
        assert resp.status_code == 200

    def test_extender_far_outside_is_400(self, tmp_path):
        resp = self._post(
            tmp_path, use_extender=True, extender_x=-15_000, extender_y=-15_000
        )
        assert resp.status_code == 400
        assert "extender" in resp.json()["detail"]

    def test_extender_negative_origin_is_ok(self, tmp_path):
        """extender 坐标可为负（部分扩展到画布左上），相交即合法。"""
        resp = self._post(
            tmp_path, use_extender=True, extender_x=-16, extender_y=-16
        )
        assert resp.status_code == 200


class TestModelLoadErrorMiddleware:
    def test_model_load_error_is_500_with_chinese_detail(self):
        """REL-1：ModelLoadError 必须被异常中间件接住并给出可读提示，
        而不像 SystemExit 那样绕过处理直接杀进程。"""
        from iopaint.api import api_middleware
        from iopaint.exceptions import ModelLoadError

        app = FastAPI()
        api_middleware(app)

        @app.get("/boom")
        def boom():
            raise ModelLoadError("模型文件 md5 不匹配")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "ModelLoadError"
        assert "模型加载失败" in body["detail"]


class TestEventLoopNotBlocked:
    """PERF-1：inpaint 期间事件循环必须还能服务其它请求。

    用一个“慢编码”模拟 12MP PNG 编码（实测 7s），断言期间轻请求能立刻返回。
    """

    SLOW_ENCODE_SECONDS = 1.5

    def test_samplers_responsive_during_slow_inpaint(self, tmp_path):
        """慢编码（12MP PNG 实测 ~7s）发生在事件循环上时，samplers 会一起卡住。

        这里用 Event 确保 inpaint **确实已经进入编码阶段**，再给 samplers 计时：
        若编码跑在事件循环上，samplers 至少要等 SLOW_ENCODE_SECONDS 才返回。
        """
        from iopaint import api as api_mod

        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        api = Api(app, cfg)

        real_pil_to_bytes = api_mod.pil_to_bytes
        encoding_started = threading.Event()

        def slow_pil_to_bytes(*args, **kwargs):
            encoding_started.set()
            time.sleep(self.SLOW_ENCODE_SECONDS)
            return real_pil_to_bytes(*args, **kwargs)

        api_mod.pil_to_bytes = slow_pil_to_bytes
        try:
            # 必须进 with 块：starlette 的 TestClient 在 with 之外会为**每个请求**
            # 单独开一个 portal（独立事件循环，testclient.py:_portal_factory），
            # 那样阻塞就感知不到，测试对修复前后都会“通过”。
            with TestClient(app) as client:
                req = InpaintRequest(
                    image=_b64_image(), mask=_b64_mask(), hd_strategy="Original"
                )
                result = {}

                def do_inpaint():
                    result["resp"] = client.post(
                        "/api/v1/inpaint",
                        content=req.model_dump_json(),
                        headers={"Content-Type": "application/json"},
                    )

                t = threading.Thread(target=do_inpaint, daemon=True)
                t.start()

                # 1) 确认 inpaint 已经进入慢编码
                assert encoding_started.wait(timeout=10), "inpaint 没有走到编码阶段"

                # 2) 此刻发轻请求，必须很快返回
                t0 = time.time()
                resp = client.get("/api/v1/samplers")
                elapsed = time.time() - t0
                assert resp.status_code == 200, f"samplers 失败: {resp.status_code}"

                t.join(timeout=30)
                assert result["resp"].status_code == 200
        finally:
            api_mod.pil_to_bytes = real_pil_to_bytes

        assert elapsed < self.SLOW_ENCODE_SECONDS / 2, (
            f"samplers 耗时 {elapsed:.2f}s，说明编码阻塞了事件循环"
        )

    def test_inpaint_still_returns_image(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        app = FastAPI()
        Api(app, cfg)
        client = TestClient(app)
        req = InpaintRequest(
            image=_b64_image(), mask=_b64_mask(), hd_strategy="Original"
        )
        resp = client.post(
            "/api/v1/inpaint",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")
