"""`/api/v1/inpaint` 与 `/api/v1/export` 的纯 CPU 路径基准。

本地跑，**不进 CI、不进 fast suite**（`pytest.ini` 的 `testpaths = iopaint/tests`）：

    python -m pytest benchmarks -m bench -s
    python -m pytest benchmarks/test_inpaint_cpu_path.py -m bench -s

用 `--model cv2` 建服务（无权重、瞬时），但把 `model_manager` 换成一个固定开销的
假模型：真实 `cv2.inpaint` 在 12MP 上单次要 ~360ms，会把要观测的 CPU 段淹没掉
（改前 1144ms / 改后约 1000ms 看不出比例）。假模型只做真实路径同样会做的两件事——
BGR 全帧拷贝 + 在笔迹范围内改一块像素——于是这里量到的就是
「base64 解码 → sha256 → 栅格化 → commit diff → 编码 → alias hash」本身。

两条路径分开测，因为它们的输入完全不同：
- **cold**：前端首轮，上传 12MP 图片字节 + 笔迹。走 base64 解码、sha256、PIL 解码、
  建 revision。
- **warm**：后续每一笔，只带 `revision` + 笔迹。走 canonical 像素、commit diff、
  编码、响应字节 alias hash。

`BASELINE_*` 是 2026-11-01 本机（Xeon E3-1231 v3，8 线程，Python 3.12，cv2 5.0.0，
numpy 2.5.4）实测值；改动前后用同一命令复跑对比比值，不要跨机器比绝对值。
"""
import base64
import statistics
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api
from iopaint.tests.test_api import _minimal_config

pytestmark = pytest.mark.bench

WIDTH, HEIGHT = 4000, 3000  # 12.0 MP，手机/相机出图的常见量级
ROUNDS = 5

# 假模型改写的矩形：落在下面那条笔迹的中段，模拟“只有笔迹覆盖区变了”
PAINT_BOX = (slice(1400, 1500), slice(1000, 1100))

# 当前实现的实测中位数（含 P7 的 Z_RLE；P1~P5 前后 cold 988.2→834.1 / warm 827.7→675.3
# / export 566.5→536.2）
BASELINE_COLD_MS = 793.0
BASELINE_WARM_MS = 620.8
BASELINE_EXPORT_MS = 472.2


class _FixedCostModel:
    """代替真实模型：只保留真实路径必然付出的全帧拷贝与局部改写。"""

    def __call__(self, image, mask, config):
        out = image[:, :, ::-1].copy()  # RGB → BGR，和 InpaintModel.__call__ 一样
        out[PAINT_BOX] = 0
        return out


def _photo_png(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """一张去掉压缩噪声的合成照片：小图随机再放大，PNG 体积才接近真实照片。

    纯 `np.random` 的 12MP 图几乎不可压缩（PNG 会到 36MB），base64 后触到
    schema 的 64MB 字段上限；这里放大 32× 的小图，体积与真实照片同量级。
    """
    rng = np.random.default_rng(20261101)
    base = rng.integers(0, 255, (height // 32, width // 32, 3), dtype=np.uint8)
    arr = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert ok
    return buf.tobytes()


def _strokes(n: int = 300) -> list:
    """一条横跨中部的笔迹，点数接近真实涂刷（前端按帧采样几百个点）。"""
    pts = [[600 + i * 9.0, 1400 + 180 * float(np.sin(i / 40))] for i in range(n)]
    return [{"pts": pts, "size": 48}]


@pytest.fixture(scope="module")
def photo_png() -> bytes:
    png = _photo_png()
    npy = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert npy.shape == (HEIGHT, WIDTH, 3)
    return png


@pytest.fixture(scope="module")
def payload(photo_png) -> str:
    return base64.b64encode(photo_png).decode()


def _client(tmp_path):
    app = FastAPI()
    api = Api(app, _minimal_config(tmp_path))
    api.model_manager = _FixedCostModel()
    return TestClient(app)


def _inpaint(client, body: dict):
    resp = client.post("/api/v1/inpaint", json=body)
    assert resp.status_code == 200, resp.text[:400]
    return resp


def _median_rounds(fn, rounds: int = ROUNDS) -> float:
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000


def test_cold_round_latency(tmp_path, payload):
    """首轮：上传 12MP 图片 + 笔迹（base64 解码、sha256、PIL 解码、建 revision）。"""
    with _client(tmp_path) as client:
        _inpaint(  # 预热一次，避免首请求的懒加载算进 median
            client, {"image": payload, "mask_strokes": _strokes()}
        )
        ms = _median_rounds(
            lambda: _inpaint(client, {"image": payload, "mask_strokes": _strokes()})
        )
    print(f"\ninpaint cold 12MP: median {ms:.1f} ms（基线 {BASELINE_COLD_MS:.1f}）")
    assert ms > 0


def test_warm_round_latency(tmp_path, payload):
    """后续每一笔：只带 revision + 笔迹（canonical 像素、commit diff、编码、alias）。"""
    with _client(tmp_path) as client:
        resp = _inpaint(client, {"image": payload, "mask_strokes": _strokes()})
        revision = resp.headers["X-Revision"]
        assert resp.headers.get("X-Image-Preview") is None  # 12MP 不该走预览

        def one_round():
            nonlocal revision
            resp2 = _inpaint(
                client, {"revision": revision, "mask_strokes": _strokes()}
            )
            revision = resp2.headers["X-Revision"]

        one_round()  # 预热
        ms = _median_rounds(one_round)
    print(f"\ninpaint warm 12MP: median {ms:.1f} ms（基线 {BASELINE_WARM_MS:.1f}）")
    assert ms > 0


def test_export_latency(tmp_path, payload):
    """导出：revision + 源格式（PNG）整图重编码。"""
    with _client(tmp_path) as client:
        resp = _inpaint(client, {"image": payload, "mask_strokes": _strokes()})
        revision = resp.headers["X-Revision"]

        def one_export():
            r = client.post("/api/v1/export", json={"revision": revision})
            assert r.status_code == 200, r.text[:400]

        one_export()
        ms = _median_rounds(one_export)
    print(f"\nexport png 12MP: median {ms:.1f} ms（基线 {BASELINE_EXPORT_MS:.1f}）")
    assert ms > 0


# ---------------------------------------------------------------------------
# 单项 A/B：把每处改动的收益从端到端噪声里单独拎出来
# ---------------------------------------------------------------------------


def _time_one(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def _median_ms(fn, rounds: int = ROUNDS) -> float:
    return statistics.median(_time_one(fn) for _ in range(rounds))


def test_decode_uploaded_bytes_avoids_second_base64(payload, photo_png):
    """P1：调用方已经解好的字节不必再解一遍。

    左边是旧签名（传入 base64 字符串，内部自己 `b64decode`），右边是新签名
    （`image_bytes=`）。两者解出的像素必须完全相同，差值就是省下的那次解码。
    """
    from iopaint.helper import decode_base64_to_image

    legacy = _median_ms(lambda: decode_base64_to_image(payload, max_pixels=None))
    fast = _median_ms(
        lambda: decode_base64_to_image(image_bytes=photo_png, max_pixels=None)
    )

    a = decode_base64_to_image(payload, max_pixels=None)
    b = decode_base64_to_image(image_bytes=photo_png, max_pixels=None)
    assert a[3] == b[3]  # ext
    assert np.array_equal(a[0], b[0])
    assert a[1] is None and b[1] is None

    print(
        f"\ndecode 12MP: base64 字符串 {legacy:.1f} ms / 已解字节 {fast:.1f} ms"
        f"（省 {legacy - fast:.1f} ms）"
    )
    assert fast < legacy


def test_png_encode_skips_pil_roundtrip(photo_png):
    """P2：ndarray 直接编码出的 PNG 与 `Image.fromarray` 往返后逐字节相同。"""
    from iopaint.helper import pil_to_bytes

    arr = cv2.imdecode(np.frombuffer(photo_png, np.uint8), cv2.IMREAD_COLOR)
    old_bytes = pil_to_bytes(Image.fromarray(arr), ext="png")
    new_bytes = pil_to_bytes(arr, ext="png")
    assert old_bytes == new_bytes

    old_ms = _median_ms(lambda: pil_to_bytes(Image.fromarray(arr), ext="png"))
    new_ms = _median_ms(lambda: pil_to_bytes(arr, ext="png"))
    print(
        f"\npng encode 12MP: PIL 往返 {old_ms:.1f} ms / ndarray 直编 {new_ms:.1f} ms"
        f"（省 {old_ms - new_ms:.1f} ms）"
    )
    assert new_ms < old_ms


def test_png_strategy_rle_vs_default(photo_png):
    """P7：同一编码器只换 zlib 策略——RGB 上 Z_RLE 快约 10%，体积 +8%。

    `png_bytes_from_numpy` 已按输入形状选策略（RGB 默认级别 → Z_RLE），这里把
    `compressobj` 的 strategy 强制回旧值，复现改动前那一侧。
    """
    import zlib

    from iopaint import helper as helper_mod

    arr = cv2.imdecode(np.frombuffer(photo_png, np.uint8), cv2.IMREAD_COLOR)

    def encode(strategy):
        if strategy is None:
            return helper_mod.png_bytes_from_numpy(arr)
        real = zlib.compressobj

        def forced(level, method, wbits, memlevel, _strategy):
            return real(level, method, wbits, memlevel, strategy)

        zlib.compressobj = forced
        try:
            return helper_mod.png_bytes_from_numpy(arr)
        finally:
            zlib.compressobj = real

    old_ms = _median_ms(lambda: encode(zlib.Z_DEFAULT_STRATEGY))
    new_ms = _median_ms(lambda: encode(strategy=None))
    old_bytes = encode(zlib.Z_DEFAULT_STRATEGY)
    new_bytes = encode(strategy=None)
    assert np.array_equal(
        cv2.imdecode(np.frombuffer(old_bytes, np.uint8), cv2.IMREAD_COLOR),
        cv2.imdecode(np.frombuffer(new_bytes, np.uint8), cv2.IMREAD_COLOR),
    )

    print(
        f"\npng strategy 12MP RGB: default {old_ms:.1f} ms / {len(old_bytes)/1e6:.2f} MB"
        f" / Z_RLE {new_ms:.1f} ms / {len(new_bytes)/1e6:.2f} MB"
    )
    assert new_ms < old_ms


def test_display_preview_opaque_alpha():
    """P4：alpha 全 255 时跳过全分辨率 float32 合成。

    预览只在 >16Mi px 时触发（`PREVIEW_MAX_PIXELS`），所以这里用 25MP。
    """
    from iopaint.api import _display_preview_jpeg

    arr = np.full((5000, 5000, 3), 90, dtype=np.uint8)
    alpha = np.full((5000, 5000), 255, dtype=np.uint8)

    preview = _display_preview_jpeg(arr, alpha)
    assert preview is not None
    # 不透明 alpha 的合成是恒等变换：结果必须与「不带 alpha」逐字节相同
    assert preview == _display_preview_jpeg(arr, None)

    ms = _median_ms(lambda: _display_preview_jpeg(arr, alpha))
    print(f"\npreview 25MP 全不透明 alpha: median {ms:.1f} ms")
    assert ms > 0


def test_display_preview_translucent_alpha_still_composites():
    """非全不透明 alpha 仍走垫白合成（只是不被快路径吃掉）。"""
    from iopaint.api import _display_preview_jpeg

    arr = np.zeros((4200, 4200, 3), dtype=np.uint8)
    alpha = np.zeros((4200, 4200), dtype=np.uint8)  # 全透明 → 应垫成白底
    preview = _display_preview_jpeg(arr, alpha)
    assert preview is not None
    assert preview != _display_preview_jpeg(arr, None)
