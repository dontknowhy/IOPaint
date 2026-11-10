"""`/api/v1/inpaint` 与 `/api/v1/export` 的纯 CPU 路径跑分。

本地跑，**不进 CI、不进 fast suite**（`pytest.ini` 的 `testpaths = iopaint/tests`）：

    python benchmarks/run.py
    python benchmarks/run.py -k "cold or warm or export"

用 `--model cv2` 建服务（无权重、瞬时），但把 `model_manager` 换成一个固定开销的
假模型：真实 `cv2.inpaint` 在 12MP 上单次要 ~360ms，会把要观测的 CPU 段淹没掉。
假模型只做真实路径同样会做的两件事——BGR 全帧拷贝 + 在笔迹范围内改一块像素——
于是这里量到的就是「解码 → sha256 → 栅格化 → commit diff → 编码 → alias hash」本身。

两条路径分开测，因为它们的输入完全不同：
- **cold**：前端首轮，上传 12MP 图片字节 + 笔迹。走图片解码、sha256、PIL 解码、
  建 revision。
- **warm**：后续每一笔，只带 `revision` + 笔迹。走 canonical 像素、commit diff、
  编码、响应字节 alias hash。

单项 A/B 用例（`*.speedup`）在同一个进程里对照两套实现，比值与机器无关；
墙钟绝对值只用于同机前后对比，不要跨机器比。
"""
import base64

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from harness import measure, record
from iopaint.api import Api
from iopaint.tests.test_api import _minimal_config
from iopaint.tests.utils import multipart

pytestmark = pytest.mark.bench

WIDTH, HEIGHT = 4000, 3000  # 12.0 MP，手机/相机出图的常见量级
ROUNDS = 7

# 假模型改写的矩形：落在下面那条笔迹的中段，模拟“只有笔迹覆盖区变了”
PAINT_BOX = (slice(1400, 1500), slice(1000, 1100))


class _FixedCostModel:
    """代替真实模型：只保留真实路径必然付出的全帧拷贝与局部改写。"""

    def __call__(self, image, mask, config):
        out = image[:, :, ::-1].copy()  # RGB → BGR，和 InpaintModel.__call__ 一样
        out[PAINT_BOX] = 0
        return out


def _photo_png(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """一张去掉压缩噪声的合成照片：小图随机再放大，PNG 体积才接近真实照片。

    纯 `np.random` 的 12MP 图几乎不可压缩（PNG 会到 36MB）；这里放大 32× 的小图，
    体积与真实照片同量级。
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
    fields = dict(body)
    image = fields.pop("image", None)
    resp = client.post("/api/v1/inpaint", **multipart(fields, image=image))
    assert resp.status_code == 200, resp.text[:400]
    return resp


def test_cold_round_latency(tmp_path, photo_png):
    """首轮：上传 12MP 图片 + 笔迹（sha256、PIL 解码、建 revision、编码）。"""
    with _client(tmp_path) as client:
        measure(
            "inpaint.cold.12mp",
            lambda: _inpaint(client, {"image": photo_png, "mask_strokes": _strokes()}),
            rounds=ROUNDS,
        )


def test_warm_round_latency(tmp_path, photo_png):
    """后续每一笔：只带 revision + 笔迹（canonical 像素、commit diff、编码、alias）。"""
    with _client(tmp_path) as client:
        resp = _inpaint(client, {"image": photo_png, "mask_strokes": _strokes()})
        revision = resp.headers["X-Revision"]
        # 12MP 不透明 PNG 落在 4–16Mi 区间：响应是全分辨率 JPEG 显示预览，
        # canonical 像素仍在 RevisionCache 里（导出/后续编辑不受影响）。
        assert resp.headers.get("X-Image-Preview") == "1"

        def one_round():
            nonlocal revision
            resp2 = _inpaint(
                client, {"revision": revision, "mask_strokes": _strokes()}
            )
            revision = resp2.headers["X-Revision"]

        measure("inpaint.warm.12mp", one_round, rounds=ROUNDS)


def test_export_latency(tmp_path, photo_png):
    """导出：revision + 源格式（PNG）整图重编码。"""
    with _client(tmp_path) as client:
        resp = _inpaint(client, {"image": photo_png, "mask_strokes": _strokes()})
        revision = resp.headers["X-Revision"]

        def one_export():
            r = client.post("/api/v1/export", data={"revision": revision})
            assert r.status_code == 200, r.text[:400]

        measure("export.png.12mp", one_export, rounds=ROUNDS)


# ---------------------------------------------------------------------------
# 单项 A/B：把每处改动的收益从端到端噪声里单独拎出来
# ---------------------------------------------------------------------------


def test_decode_uploaded_bytes_avoids_second_base64(payload, photo_png):
    """已解好的字节不必再解一遍（插件/调整掩膜端点仍是 base64 入口）。

    左边是字符串签名（内部自己 `b64decode`），右边是 `image_bytes=`。
    两者解出的像素必须完全相同，差值就是省下的那次解码。
    """
    from iopaint.helper import decode_base64_to_image

    legacy = measure(
        "decode.12mp.base64",
        lambda: decode_base64_to_image(payload, max_pixels=None),
        rounds=ROUNDS,
    )
    fast = measure(
        "decode.12mp.bytes",
        lambda: decode_base64_to_image(image_bytes=photo_png, max_pixels=None),
        rounds=ROUNDS,
    )

    a = decode_base64_to_image(payload, max_pixels=None)
    b = decode_base64_to_image(image_bytes=photo_png, max_pixels=None)
    assert a[3] == b[3]  # ext
    assert np.array_equal(a[0], b[0])
    assert a[1] is None and b[1] is None

    record(
        "decode.12mp.speedup",
        legacy.median / fast.median,
        "x",
        direction="higher",
    )


def test_png_encode_skips_pil_roundtrip(photo_png):
    """ndarray 直接编码出的 PNG 与 `Image.fromarray` 往返后逐字节相同。"""
    from iopaint.helper import pil_to_bytes

    arr = cv2.imdecode(np.frombuffer(photo_png, np.uint8), cv2.IMREAD_COLOR)
    old_bytes = pil_to_bytes(Image.fromarray(arr), ext="png")
    new_bytes = pil_to_bytes(arr, ext="png")
    assert old_bytes == new_bytes

    old = measure(
        "png.encode.12mp.pil_roundtrip",
        lambda: pil_to_bytes(Image.fromarray(arr), ext="png"),
        rounds=ROUNDS,
    )
    new = measure(
        "png.encode.12mp.ndarray",
        lambda: pil_to_bytes(arr, ext="png"),
        rounds=ROUNDS,
    )
    record(
        "png.encode.12mp.speedup",
        old.median / new.median,
        "x",
        direction="higher",
    )


def test_png_strategy_rle_vs_default(photo_png):
    """同一编码器只换 zlib 策略：RGB 默认级别下 Z_RLE 更快、体积略大。

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

    old = measure(
        "png.zlib.12mp.default",
        lambda: encode(zlib.Z_DEFAULT_STRATEGY),
        rounds=ROUNDS,
    )
    new = measure(
        "png.zlib.12mp.rle",
        lambda: encode(strategy=None),
        rounds=ROUNDS,
    )
    old_bytes = encode(zlib.Z_DEFAULT_STRATEGY)
    new_bytes = encode(strategy=None)
    assert np.array_equal(
        cv2.imdecode(np.frombuffer(old_bytes, np.uint8), cv2.IMREAD_COLOR),
        cv2.imdecode(np.frombuffer(new_bytes, np.uint8), cv2.IMREAD_COLOR),
    )

    record("png.size.12mp.default", len(old_bytes) / 1e6, "MB")
    record("png.size.12mp.rle", len(new_bytes) / 1e6, "MB")
    record("png.zlib.12mp.speedup", old.median / new.median, "x", direction="higher")


def test_display_preview_opaque_alpha():
    """alpha 全 255 时跳过全分辨率 float32 合成。

    预览只在 >16Mi px 时触发（`PREVIEW_MAX_PIXELS`），所以这里用 25MP。
    """
    from iopaint.api import _display_preview_jpeg

    arr = np.full((5000, 5000, 3), 90, dtype=np.uint8)
    alpha = np.full((5000, 5000), 255, dtype=np.uint8)

    preview = _display_preview_jpeg(arr, alpha)
    assert preview is not None
    # 不透明 alpha 的合成是恒等变换：结果必须与「不带 alpha」逐字节相同
    assert preview == _display_preview_jpeg(arr, None)

    measure(
        "preview.25mp.opaque_alpha",
        lambda: _display_preview_jpeg(arr, alpha),
        rounds=ROUNDS,
    )


def test_display_preview_translucent_alpha_still_composites():
    """非全不透明 alpha 仍走垫白合成（只是不被快路径吃掉）。"""
    from iopaint.api import _display_preview_jpeg

    arr = np.zeros((4200, 4200, 3), dtype=np.uint8)
    alpha = np.zeros((4200, 4200), dtype=np.uint8)  # 全透明 → 应垫成白底
    preview = _display_preview_jpeg(arr, alpha)
    assert preview is not None
    assert preview != _display_preview_jpeg(arr, None)
