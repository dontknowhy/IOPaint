"""D-10 插件并发锁（CON-1）+ interactive_seg 的 check-then-act 顺序。"""
import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image

from iopaint.api import Api
from iopaint.plugins.base_plugin import BasePlugin
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
