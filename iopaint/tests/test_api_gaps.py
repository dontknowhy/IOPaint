"""api.py 的剩余缺口：静态目录 503、socketio 线程发射、插件路由与 geninfo/save_image 边界。

全部走 in-process TestClient（不绑端口）；插件用假插件注入 api.plugins，
不加载任何权重。与 test_api.py/test_api_phase1.py 的分工：那里覆盖主干
（inpaint/adjust_mask/CORS/请求体上限），这里补 4xx/5xx 边界与未触达分支。
"""
import asyncio
import io
import sys
import threading

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import iopaint.api as api_mod
from iopaint.api import Api, WebAppStaticFiles, api_middleware, emit_from_thread
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import (
    AdjustMaskRequest,
    InpaintRequest,
    RunPluginRequest,
    SwitchPluginModelRequest,
)
from iopaint.tests.test_api_phase1 import _b64_image, _b64_mask, _minimal_config


class FakePlugin(BasePlugin):
    """记录调用的假插件；support_* 用例参数控制，不加载权重。"""

    def __init__(self, name="FakePlugin", gen_image=None, gen_mask=None,
                 support_gen_image=True, support_gen_mask=True):
        # 不调 super().__init__（它会跑 check_dep）；直接建锁
        self.lock = threading.Lock()
        self.name = name
        self.support_gen_image = support_gen_image
        self.support_gen_mask = support_gen_mask
        self._gen_image = gen_image
        self._gen_mask = gen_mask
        self.switched = []
        self.lock_held_during_switch = False

    def gen_image(self, rgb_np_img, req):
        return self._gen_image

    def gen_mask(self, rgb_np_img, req):
        return self._gen_mask

    def switch_model(self, new_model_name):
        self.lock_held_during_switch = self.lock.locked()
        self.switched.append(new_model_name)


def _client(tmp_path, **cfg_overrides):
    cfg = _minimal_config(tmp_path)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    app = FastAPI()
    api = Api(app, cfg)
    return api, TestClient(app)


# ---------------------------------------------------------------------------
# WebAppStaticFiles：目录缺失必须是可读的 503，而不是 RuntimeError→500
# ---------------------------------------------------------------------------


class TestWebAppStaticFiles:
    def test_missing_dir_returns_503_with_hint(self, tmp_path):
        app = FastAPI()
        app.mount(
            "/",
            WebAppStaticFiles(directory=tmp_path / "not-built", html=True),
            name="assets",
        )
        resp = TestClient(app).get("/")
        assert resp.status_code == 503
        # 提示要可操作：告诉用户去构建前端
        assert "npm run build" in resp.text

    def test_existing_dir_serves_files(self, tmp_path):
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "app.js").write_text("console.log(1)")
        app = FastAPI()
        app.mount("/", WebAppStaticFiles(directory=tmp_path / "assets"), name="assets")
        resp = TestClient(app).get("/app.js")
        assert resp.status_code == 200
        assert "console.log" in resp.text


# ---------------------------------------------------------------------------
# 异常中间件：WEBUI_RICH_EXCEPTIONS 打开时走 rich 打印分支，响应不变
# ---------------------------------------------------------------------------


class TestRichExceptionPath:
    def test_rich_path_still_returns_500_json(self, monkeypatch):
        monkeypatch.setenv("WEBUI_RICH_EXCEPTIONS", "1")
        app = FastAPI()
        api_middleware(app)

        @app.get("/boom")
        def boom():
            raise ValueError("rich path error")

        resp = TestClient(app, raise_server_exceptions=False).get("/boom")
        assert resp.status_code == 500
        assert resp.json()["error"] == "ValueError"


# ---------------------------------------------------------------------------
# emit_from_thread / _log_emit_error：socketio 进度事件的线程侧安全网
# ---------------------------------------------------------------------------


class TestEmitFromThread:
    def test_noop_without_main_loop(self):
        api_mod.main_loop = None
        # 没有捕获到主循环时静默返回，不抛
        assert emit_from_thread(object(), "evt", {"a": 1}) is None

    def test_schedules_on_main_loop(self):
        """必须在主循环上调度，且事件/数据原样送达。"""

        class RecordingSio:
            def __init__(self):
                self.events = []

            async def emit(self, event, data=None):
                self.events.append((event, data))

        async def scenario():
            api_mod.main_loop = asyncio.get_running_loop()
            sio = RecordingSio()
            emit_from_thread(sio, "diffusion_progress", {"step": 3})
            # 让调度的协程跑完
            await asyncio.sleep(0)
            return sio.events

        try:
            events = asyncio.run(scenario())
        finally:
            api_mod.main_loop = None
        assert events == [("diffusion_progress", {"step": 3})]

    def test_emit_failure_is_logged_not_raised(self):
        """sio.emit 抛异常时只记 warning，不能把 worker 线程打挂。"""

        class BoomSio:
            async def emit(self, event, data=None):
                raise RuntimeError("socketio down")

        async def scenario():
            api_mod.main_loop = asyncio.get_running_loop()
            emit_from_thread(BoomSio(), "diffusion_finish")
            await asyncio.sleep(0)  # 触发 done callback

        try:
            asyncio.run(scenario())  # 不应抛出
        finally:
            api_mod.main_loop = None

    def test_log_emit_error_reads_future_exception(self):
        async def scenario():
            fut = asyncio.get_running_loop().create_future()
            fut.set_exception(RuntimeError("emit failed"))
            api_mod._log_emit_error(fut)  # 记 warning，不抛
            # 无异常的 future 也不该抛
            ok = asyncio.get_running_loop().create_future()
            ok.set_result(None)
            api_mod._log_emit_error(ok)

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 插件路由：422 分支 + 成功路径（假插件，无权重）
# ---------------------------------------------------------------------------


class TestPluginGenImage:
    def test_plugin_not_found_is_422(self, tmp_path):
        _, client = _client(tmp_path)
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=RunPluginRequest(name="nope", image=_b64_image()).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_gen_image_not_supported_is_422(self, tmp_path):
        api, client = _client(tmp_path)
        api.plugins["FakePlugin"] = FakePlugin(support_gen_image=False)
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert "does not support" in resp.json()["detail"]

    def test_gen_image_success_bgr_result(self, tmp_path):
        """插件返回 BGR 3 通道 → 转 RGB + 拼回 alpha → image/png。"""
        api, client = _client(tmp_path)
        bgr = np.full((64, 64, 3), (30, 60, 90), np.uint8)
        api.plugins["FakePlugin"] = FakePlugin(gen_image=bgr)
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        out = np.array(Image.open(io.BytesIO(resp.content)))
        # 输入 PNG 无 alpha → alpha_channel=None → concat 原样返回 RGB
        assert out.shape == (64, 64, 3)
        # BGR→RGB 转换生效：BGR(30,60,90) 应变成 RGB(90,60,30)
        assert tuple(out[32, 32]) == (90, 60, 30)

    def test_gen_image_success_rgba_result_passthrough(self, tmp_path):
        """插件返回 RGBA 4 通道（如 RemoveBG）→ 不做色彩转换直接编码。"""
        api, client = _client(tmp_path)
        rgba = np.zeros((64, 64, 4), np.uint8)
        rgba[..., 3] = 200
        api.plugins["FakePlugin"] = FakePlugin(gen_image=rgba)
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        out = np.array(Image.open(io.BytesIO(resp.content)))
        assert out.shape == (64, 64, 4)
        assert (out[..., 3] == 200).all()

    def test_empty_cache_after_inpaint_triggers_gc(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(api_mod, "torch_gc", lambda: calls.append(1))
        api, client = _client(tmp_path, empty_cache_after_inpaint=True)
        api.plugins["FakePlugin"] = FakePlugin(
            gen_image=np.zeros((64, 64, 3), np.uint8)
        )
        resp = client.post(
            "/api/v1/run_plugin_gen_image",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert calls == [1]


class TestPluginGenMask:
    def test_plugin_not_found_is_422(self, tmp_path):
        _, client = _client(tmp_path)
        resp = client.post(
            "/api/v1/run_plugin_gen_mask",
            content=RunPluginRequest(name="nope", image=_b64_image()).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_gen_mask_not_supported_is_422(self, tmp_path):
        api, client = _client(tmp_path)
        api.plugins["FakePlugin"] = FakePlugin(support_gen_mask=False)
        resp = client.post(
            "/api/v1/run_plugin_gen_mask",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_gen_mask_success_returns_brush_color(self, tmp_path):
        """灰色 mask → gen_frontend_mask 膨胀并染成前端笔刷色 ffcc00bb(RGBA)。"""
        api, client = _client(tmp_path)
        gray = np.zeros((64, 64), np.uint8)
        gray[16:48, 16:48] = 255
        api.plugins["FakePlugin"] = FakePlugin(gen_mask=gray)
        resp = client.post(
            "/api/v1/run_plugin_gen_mask",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        out = np.array(Image.open(io.BytesIO(resp.content)))
        assert out.shape == (64, 64, 4)
        # 中心区域应是笔刷色 (255,203,0,alpha)
        center = out[32, 32]
        assert tuple(center[:3]) == (255, 203, 0)

    def test_empty_cache_after_inpaint_triggers_gc(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(api_mod, "torch_gc", lambda: calls.append(1))
        api, client = _client(tmp_path, empty_cache_after_inpaint=True)
        api.plugins["FakePlugin"] = FakePlugin(
            gen_mask=np.full((64, 64), 255, np.uint8)
        )
        resp = client.post(
            "/api/v1/run_plugin_gen_mask",
            content=RunPluginRequest(
                name="FakePlugin", image=_b64_image()
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert calls == [1]


# ---------------------------------------------------------------------------
# switch_plugin_model：切换要持插件锁（CON-1），并回写 config
# ---------------------------------------------------------------------------


class TestSwitchPluginModel:
    def test_switch_holds_plugin_lock_and_records(self, tmp_path):
        api, client = _client(tmp_path)
        plugin = FakePlugin()
        api.plugins["FakePlugin"] = plugin

        resp = client.post(
            "/api/v1/switch_plugin_model",
            content=SwitchPluginModelRequest(
                plugin_name="FakePlugin", model_name="new-model"
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert plugin.switched == ["new-model"]
        # CON-1: switch_model 必须在插件锁内执行（防与推理互踩）
        assert plugin.lock_held_during_switch is True

    def test_switch_removebg_updates_config(self, tmp_path):
        api, client = _client(tmp_path, remove_bg_model="u2net")
        api.plugins["RemoveBG"] = FakePlugin(name="RemoveBG")
        client.post(
            "/api/v1/switch_plugin_model",
            content=SwitchPluginModelRequest(
                plugin_name="RemoveBG", model_name="silueta"
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert api.config.remove_bg_model == "silueta"

    def test_switch_realesrgan_updates_config(self, tmp_path):
        api, client = _client(tmp_path)
        api.plugins["RealESRGAN"] = FakePlugin(name="RealESRGAN")
        client.post(
            "/api/v1/switch_plugin_model",
            content=SwitchPluginModelRequest(
                plugin_name="RealESRGAN", model_name="RealESRGAN_x4plus"
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert api.config.realesrgan_model == "RealESRGAN_x4plus"

    def test_switch_interactive_seg_updates_config(self, tmp_path):
        api, client = _client(tmp_path)
        api.plugins["InteractiveSeg"] = FakePlugin(name="InteractiveSeg")
        client.post(
            "/api/v1/switch_plugin_model",
            content=SwitchPluginModelRequest(
                plugin_name="InteractiveSeg", model_name="vit_h"
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert api.config.interactive_seg_model == "vit_h"

    def test_switch_unknown_plugin_is_noop(self, tmp_path):
        api, client = _client(tmp_path, remove_bg_model="u2net")
        resp = client.post(
            "/api/v1/switch_plugin_model",
            content=SwitchPluginModelRequest(
                plugin_name="nope", model_name="x"
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert api.config.remove_bg_model == "u2net"  # 没被改


# ---------------------------------------------------------------------------
# geninfo / save_image / server-config / switch-model 边界
# ---------------------------------------------------------------------------


class TestGenInfo:
    def test_parses_prompt_and_negative(self, tmp_path):
        """Automatic1111 风格 PNG parameters 块：首段是 prompt，Negative prompt: 后是负向。"""
        _, client = _client(tmp_path)
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=None)
        # 手工塞 parameters 文本块
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        info.add_text(
            "parameters",
            "a cat sitting\nNegative prompt: blurry, low quality\nSteps: 20, Sampler: Euler",
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=info)
        buf.seek(0)

        resp = client.post(
            "/api/v1/gen-info", files={"file": ("t.png", buf, "image/png")}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt"] == "a cat sitting"
        assert body["negative_prompt"] == "blurry, low quality"

    def test_no_negative_prompt_section(self, tmp_path):
        _, client = _client(tmp_path)
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        info.add_text("parameters", "just a prompt\nSteps: 20")
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=info)
        buf.seek(0)

        resp = client.post(
            "/api/v1/gen-info", files={"file": ("t.png", buf, "image/png")}
        )
        assert resp.status_code == 200
        body = resp.json()
        # 无 "Negative prompt:" 段时，prompt 是整个 parameters 串（含后续元数据行）
        assert body["prompt"] == "just a prompt\nSteps: 20"
        assert body["negative_prompt"] == ""


class TestSaveImageEdges:
    def test_no_output_dir_is_400(self, tmp_path):
        _, client = _client(tmp_path, output_dir=None)
        buf = io.BytesIO(b"fakepng")
        resp = client.post(
            "/api/v1/save_image", files={"file": ("a.png", buf, "image/png")}
        )
        assert resp.status_code == 400

    def test_dotdot_filename_is_400(self, tmp_path):
        _, client = _client(tmp_path)
        buf = io.BytesIO(b"x")
        resp = client.post(
            "/api/v1/save_image", files={"file": ("..", buf, "image/png")}
        )
        assert resp.status_code == 400

    def test_empty_filename_is_400(self, tmp_path):
        """HTTP 层空文件名会被 FastAPI 拒成 422（multipart 无 filename 属性），
        代码内的 `not safe_filename` 守卫只能函数级触发，这里直接调 endpoint。"""
        from starlette.datastructures import UploadFile as StarletteUploadFile

        api, _ = _client(tmp_path)
        fake = StarletteUploadFile(file=io.BytesIO(b"x"), filename="")
        with pytest.raises(Exception) as ei:
            api.api_save_image(fake)
        assert ei.value.status_code == 400

    def test_path_traversal_is_sanitized(self, tmp_path):
        """../../evil.png 只保留文件名部分，落在 output_dir 内。"""
        out = tmp_path / "out"
        out.mkdir()
        _, client = _client(tmp_path, output_dir=out)
        buf = io.BytesIO(b"content")
        resp = client.post(
            "/api/v1/save_image",
            files={"file": ("../../evil.png", buf, "image/png")},
        )
        assert resp.status_code == 200
        assert (out / "evil.png").read_bytes() == b"content"
        assert not (tmp_path.parent / "evil.png").exists()


class TestServerConfigWithPlugins:
    def test_plugins_listed(self, tmp_path):
        api, client = _client(tmp_path)
        api.plugins["FakePlugin"] = FakePlugin(
            support_gen_image=True, support_gen_mask=False
        )
        body = client.get("/api/v1/server-config").json()
        infos = {p["name"]: p for p in body["plugins"]}
        assert infos["FakePlugin"]["support_gen_image"] is True
        assert infos["FakePlugin"]["support_gen_mask"] is False


class TestSwitchModelSameName:
    def test_same_name_is_noop(self, tmp_path):
        """POST 同名模型不应触发 reload（否则会白扔一次几十秒的装载）。"""
        api, client = _client(tmp_path)
        current = client.get("/api/v1/model").json()["name"]
        model_before = api.model_manager.model

        resp = client.post(
            "/api/v1/model",
            content=f'{{"name": "{current}"}}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == current
        assert api.model_manager.model is model_before  # 同一个对象 = 没 reload


class TestAdjustMaskImageEndpoint:
    def test_adjust_mask_returns_rgba_brush(self, tmp_path):
        """adjust_mask 输出也应是笔刷色 RGBA（与 gen_frontend_mask 一致）。"""
        _, client = _client(tmp_path)
        req = AdjustMaskRequest(mask=_b64_mask(), operate="expand", kernel_size=3)
        resp = client.post(
            "/api/v1/adjust_mask",
            content=req.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        out = np.array(Image.open(io.BytesIO(resp.content)))
        assert out.shape == (64, 64, 4)
        center = out[32, 32]
        assert tuple(center[:3]) == (255, 203, 0)


# ---------------------------------------------------------------------------
# 剩余单行缺口：路由委托、404、launch、emit 降级、rich 导入失败
# ---------------------------------------------------------------------------


class TestSwitchToDifferentModel:
    def test_route_delegates_to_switch(self, tmp_path, monkeypatch):
        """不同名模型：路由必须委托 model_manager.switch 并返回切换后的 current_model。

        真实 switch 需要权重（slow 集已覆盖），这里只验证路由层委托关系。
        """
        api, client = _client(tmp_path)
        calls = []
        monkeypatch.setattr(
            api.model_manager, "switch", lambda name: calls.append(name)
        )

        resp = client.post(
            "/api/v1/model",
            content='{"name": "some-other-model"}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert calls == ["some-other-model"]
        # fake switch 不改状态 → 仍是原模型
        assert resp.json()["name"] == "cv2"


class TestInputImageNotAFile:
    def test_missing_input_file_is_404(self, tmp_path):
        """input 配置了路径但不存在（非目录，不建 FileManager）→ 404。"""
        _, client = _client(tmp_path, input=tmp_path / "missing.png")
        resp = client.get("/api/v1/inputimage")
        assert resp.status_code == 404


class TestLaunch:
    def test_launch_runs_uvicorn_on_combined_asgi(self, tmp_path, monkeypatch):
        """launch 必须跑 combined_asgi_app（socketio 才能收到事件），且用 config 的 host/port。"""
        captured = {}
        monkeypatch.setattr(
            api_mod.uvicorn,
            "run",
            lambda app, **kw: captured.update(app=app, **kw),
        )
        api, _ = _client(tmp_path)
        api.config.host = "0.0.0.0"
        api.config.port = 12345
        api.launch()
        assert captured["app"] is api.combined_asgi_app
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 12345


class TestEmitOnClosedLoop:
    def test_closed_loop_logs_and_returns(self):
        """主循环已关闭（服务关停竞态）时 run_coroutine_threadsafe 会同步抛，
        必须被接住记 warning，而不是把 worker 线程打挂。"""

        async def make_closed_loop():
            loop = asyncio.new_event_loop()
            loop.close()
            return loop

        api_mod.main_loop = asyncio.run(make_closed_loop())
        try:
            assert emit_from_thread(object(), "evt", None) is None
        finally:
            api_mod.main_loop = None


class TestRichImportFailureFallback:
    def test_rich_import_failure_still_500(self, monkeypatch):
        """rich.console 导入失败（依赖损坏）时应静默回退到 traceback 打印，响应不变。"""
        monkeypatch.setenv("WEBUI_RICH_EXCEPTIONS", "1")
        # sys.modules 里放 None 会让 `from rich.console import Console` 抛 ImportError
        monkeypatch.setitem(sys.modules, "rich.console", None)

        app = FastAPI()
        api_middleware(app)

        @app.get("/boom")
        def boom():
            raise ValueError("rich broken")

        resp = TestClient(app, raise_server_exceptions=False).get("/boom")
        assert resp.status_code == 500
        assert resp.json()["error"] == "ValueError"
