"""D-10 插件并发锁（CON-1）+ interactive_seg 的 check-then-act 顺序。

另含 GFPGAN / RestoreFormer 复用同一个 RealESRGAN upscaler 时的锁：两者各持
各的 plugin.lock，共享的 RealESRGANer 实例状态必须靠 upscaler 自己的锁保护。
"""
import base64
import io
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image

from iopaint.api import Api
from iopaint.plugins import gfpgan_plugin, gfpganer, restoreformer
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.gfpgan_plugin import GFPGANPlugin
from iopaint.plugins.restoreformer import RestoreFormerPlugin
from iopaint.schema import RunPluginRequest
from iopaint.tests.test_api_phase1 import _minimal_config

SLEEP_SECONDS = 0.2
THREADS = 4


def _b64_image(size=(32, 32)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OverlapProbePlugin(BasePlugin):
    """记录 gen_image/gen_mask 的并发重入情况。"""

    name = "overlap_probe"
    support_gen_image = True
    support_gen_mask = True

    def __init__(self):
        super().__init__()
        self._guard = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def check_dep(self):
        return None

    def _enter(self):
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def _leave(self):
        with self._guard:
            self.in_flight -= 1

    def gen_image(self, rgb_np_img, req):
        self._enter()
        time.sleep(SLEEP_SECONDS)
        self._leave()
        return rgb_np_img[:, :, ::-1]  # BGR

    def gen_mask(self, rgb_np_img, req):
        self._enter()
        time.sleep(SLEEP_SECONDS)
        self._leave()
        return np.zeros(rgb_np_img.shape[:2], np.uint8)


def _api_with_probe(tmp_path):
    app = FastAPI()
    api = Api(app, _minimal_config(tmp_path))
    api.plugins[OverlapProbePlugin.name] = OverlapProbePlugin()
    return api


def _run_concurrently(fn, n=THREADS):
    results, errors = [], []

    def worker():
        try:
            results.append(fn())
        except Exception as e:  # noqa: BLE001 - 测试里要把异常带出来
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results, errors, time.time() - start


class TestPluginLock:
    def test_gen_image_is_serialized(self, tmp_path):
        api = _api_with_probe(tmp_path)
        plugin = api.plugins[OverlapProbePlugin.name]
        req = RunPluginRequest(name=plugin.name, image=_b64_image())

        results, errors, elapsed = _run_concurrently(
            lambda: api._plugin_gen_image_blocking(req)
        )

        assert errors == []
        assert len(results) == THREADS
        assert all(len(b) > 0 for b in results)
        # 关键断言：锁生效 → 任何时刻只有一个请求在插件内部
        assert plugin.max_in_flight == 1, f"并发重入：{plugin.max_in_flight}"
        # 而且它们是真的串行跑完的（不是大家都没跑到 sleep）
        assert elapsed >= SLEEP_SECONDS * (THREADS - 1) * 0.9, f"耗时 {elapsed:.2f}s"

    def test_gen_mask_is_serialized(self, tmp_path):
        api = _api_with_probe(tmp_path)
        plugin = api.plugins[OverlapProbePlugin.name]
        req = RunPluginRequest(name=plugin.name, image=_b64_image())

        results, errors, _ = _run_concurrently(
            lambda: api._plugin_gen_mask_blocking(req)
        )

        assert errors == []
        assert len(results) == THREADS
        assert plugin.max_in_flight == 1, f"并发重入：{plugin.max_in_flight}"

    def test_probe_would_detect_overlap_without_lock(self):
        """反向验证：不走锁的路径确实能测出重入，否则上面两条测试形同虚设。"""
        plugin = OverlapProbePlugin()
        _run_concurrently(lambda: plugin.gen_image(None, None))
        # 直接并发调用（没有 api 层的锁）→ 必须能观察到重入
        assert plugin.max_in_flight > 1


class TestInteractiveSegCacheOrder:
    """prev_img_md5 必须在 set_image 成功之后才更新（check-then-act）。"""

    def test_failed_set_image_does_not_mark_cache(self):
        from iopaint.plugins.interactive_seg import InteractiveSeg

        class FakePredictor:
            def __init__(self):
                self.fail = True
                self.set_calls = 0

            def set_image(self, img):
                self.set_calls += 1
                if self.fail:
                    raise RuntimeError("set_image boom")

            def predict(self, **kwargs):
                return np.zeros((1, 4, 4), dtype=bool), None, None

        seg = InteractiveSeg.__new__(InteractiveSeg)  # 跳过 __init__（那会下载权重）
        seg.predictor = FakePredictor()
        seg.prev_img_md5 = None
        img = np.zeros((4, 4, 3), np.uint8)

        with pytest.raises(RuntimeError):
            seg.forward(img, [[1, 1, 1]], "md5-1")
        # 失败后不能标成"已缓存"，否则下次请求会跳过 set_image 拿旧图去 predict
        assert seg.prev_img_md5 is None

        seg.predictor.fail = False
        seg.forward(img, [[1, 1, 1]], "md5-1")
        assert seg.prev_img_md5 == "md5-1"
        assert seg.predictor.set_calls == 2


class _AcceptsDevice:
    """只实现 .to(device)：真实现会把权重搬上 device，测试里搬不搬无所谓。"""

    def to(self, device: Any) -> "_AcceptsDevice":
        return self


class FakeFaceDet(_AcceptsDevice):
    """.mean_tensor 与自身被 GFPGANPlugin.__init__ 末尾各 .to(device) 一次。"""

    def __init__(self) -> None:
        self.mean_tensor = _AcceptsDevice()


class FakeUpscalerModel:
    """冒充 RealESRGANer：真身把 img/output 挂实例上，测试只需能拿回 upscaler 的锁。"""

    def __init__(self, upscaler: "FakeUpscaler") -> None:
        self.lock = upscaler.lock


class FakeUpscaler:
    """只提供 face 插件需要的那两样东西：.model（当 bg_upsampler）和 .lock。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = FakeUpscalerModel(self)


class RecordingEnhancer:
    """假 MyGFPGANer，记录被调用时 upscaler 锁是否已由调用方持有。

    threading.Lock 不可重入：同一线程里再 acquire(blocking=False) 失败，
    恰好证明调用方正持锁，而不是「别的线程碰巧持有」。
    """

    def __init__(
        self, lock: Optional[Any] = None, bg_upsampler: Optional[Any] = None
    ) -> None:
        self.lock = lock
        self.bg_upsampler = bg_upsampler
        self.lock_held: Optional[bool] = None
        self.calls = 0
        # GFPGANPlugin.__init__ 会对 face_helper.face_det 调 .to(device)
        self.face_helper = SimpleNamespace(face_det=FakeFaceDet())

    def enhance(self, img: np.ndarray, **kwargs: Any) -> tuple:
        self.calls += 1
        if self.lock is not None:
            acquired = self.lock.acquire(blocking=False)
            if acquired:  # 没持锁就进来了，立刻归还以免污染后续断言
                self.lock.release()
            self.lock_held = not acquired
        return [], [], np.zeros_like(img)


@pytest.fixture
def make_face_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any, Optional[FakeUpscaler]], Any]:
    """用真的 __init__ 造插件，只打桩两处会碰权重/网络的边。

    download_model 只负责返回权重路径，MyGFPGANer 负责建网络并加载权重；
    两者都换成假的，BasePlugin.__init__（含 check_dep）保持原样。
    """

    def fake_my_gfpganer(
        model_path: Optional[str] = None,
        upscale: float = 2,
        arch: str = "clean",
        channel_multiplier: int = 2,
        bg_upsampler: Optional[Any] = None,
        device: Any = None,
    ) -> RecordingEnhancer:
        return RecordingEnhancer(
            bg_upsampler.lock if bg_upsampler is not None else None,
            bg_upsampler=bg_upsampler,
        )

    # 两个插件模块各自 from iopaint.helper import download_model，必须分别打桩
    for module in (gfpgan_plugin, restoreformer):
        monkeypatch.setattr(module, "download_model", lambda *args, **kwargs: "dummy")
    # 插件在 __init__ 里 from .gfpganer import MyGFPGANer，打桩模块属性即可生效
    monkeypatch.setattr(gfpganer, "MyGFPGANer", fake_my_gfpganer)

    return lambda plugin_cls, upscaler: plugin_cls(device="cpu", upscaler=upscaler)


class TestSharedUpscalerLock:
    """GFPGAN / RestoreFormer 的 bg_upsampler 是同一个 RealESRGANer。"""

    @pytest.mark.parametrize("plugin_cls", [GFPGANPlugin, RestoreFormerPlugin])
    def test_enhance_holds_upscaler_lock(
        self,
        plugin_cls: Any,
        make_face_plugin: Callable[[Any, Optional[FakeUpscaler]], Any],
    ) -> None:
        upscaler = FakeUpscaler()
        plugin = make_face_plugin(plugin_cls, upscaler)
        rgb = np.full((8, 8, 3), 127, dtype=np.uint8)

        # __init__ 必须自己把共享 upscaler 存下来，否则 _enhance 拿不到那把锁
        assert plugin.upscaler is upscaler
        # 真正共用给 bg_upsampler 的也是它内部的 RealESRGANer
        assert plugin.face_enhancer.bg_upsampler is upscaler.model

        out = plugin.gen_image(rgb, RunPluginRequest(name=plugin_cls.name, image=""))

        assert plugin.face_enhancer.calls == 1
        assert plugin.face_enhancer.lock_held is True
        assert out.shape == (8, 8, 3)
        # enhance 返回后锁必须归还，不能把共享的 upscaler 永久锁死
        assert upscaler.lock.acquire(timeout=0.5)
        upscaler.lock.release()

    @pytest.mark.parametrize("plugin_cls", [GFPGANPlugin, RestoreFormerPlugin])
    def test_enhance_without_upscaler(
        self,
        plugin_cls: Any,
        make_face_plugin: Callable[[Any, Optional[FakeUpscaler]], Any],
    ) -> None:
        plugin = make_face_plugin(plugin_cls, None)
        rgb = np.full((8, 8, 3), 127, dtype=np.uint8)

        assert plugin.upscaler is None
        assert plugin.face_enhancer.bg_upsampler is None

        out = plugin.gen_image(rgb, RunPluginRequest(name=plugin_cls.name, image=""))

        assert plugin.face_enhancer.calls == 1
        assert plugin.face_enhancer.lock_held is None
        assert out.shape == (8, 8, 3)
