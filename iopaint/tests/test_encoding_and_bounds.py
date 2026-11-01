"""编码参数与输入边界（PERF-2、DOS-1、REL-1）。
"""
import base64
import io
import math
import zlib
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image, JpegImagePlugin, PngImagePlugin
from pydantic import ValidationError

import iopaint.api as api_mod
from iopaint.exceptions import ModelLoadError
from iopaint.helper import (
    jpeg_sampling,
    jpeg_sampling_from_bytes,
    numpy_to_bytes,
    pil_to_bytes,
)
from iopaint.schema import AdjustMaskRequest, InpaintRequest, RunPluginRequest


# ---------------------------------------------------------------------------
# D-5 numpy_to_bytes: 按格式拆参数
# ---------------------------------------------------------------------------

class TestNumpyToBytesParams:
    def test_png_is_compressed(self):
        """平坦 mask 在 compression=6 下体积远小于 compression=0。"""
        mask = np.zeros((1024, 1024, 4), dtype=np.uint8)
        mask[100:900, 100:900] = [255, 203, 0, 186]

        compressed = numpy_to_bytes(mask, "png", png_compression=6)
        raw = numpy_to_bytes(mask, "png", png_compression=0)

        assert len(compressed) < len(raw) / 10

    def test_png_roundtrip_pixels(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        data = numpy_to_bytes(img, "png")
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == img.shape
        # PNG 无损
        assert np.array_equal(decoded, img)

    def test_jpeg_quality_is_honored(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        low = numpy_to_bytes(img, "jpg", quality=20)
        high = numpy_to_bytes(img, "jpg", quality=95)
        assert len(low) < len(high)

    def test_jpg_extension_accepted(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert len(numpy_to_bytes(img, "jpg")) > 0
        assert len(numpy_to_bytes(img, "jpeg")) > 0

    def test_dot_prefix_accepted(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert len(numpy_to_bytes(img, ".png")) > 0


# ---------------------------------------------------------------------------
# D-5 pil_to_bytes: png_compress_level
# ---------------------------------------------------------------------------

class TestPilCompressLevel:
    def _big_image(self):
        # 用平滑渐变图（可压缩）；纯随机噪声不可压缩，测不出压缩级别差异
        x = np.linspace(0, 255, 2000, dtype=np.uint8)
        grad = np.tile(x, (2000, 1))
        return Image.fromarray(np.stack([grad, grad.T, grad], axis=-1))

    def test_default_is_fast_level(self):
        img = self._big_image()
        # 默认 compress_level=1（快）与显式 1 结果一致
        default_bytes = pil_to_bytes(img, "png")
        level1 = pil_to_bytes(img, "png", png_compress_level=1)
        assert default_bytes == level1

    def test_level1_smaller_than_level0(self):
        img = self._big_image()
        l0 = pil_to_bytes(img, "png", png_compress_level=0)
        l1 = pil_to_bytes(img, "png", png_compress_level=1)
        assert len(l1) < len(l0)

    def test_lossless_at_any_level(self):
        arr = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        for level in (1, 6, 9):
            out = pil_to_bytes(img, "png", png_compress_level=level)
            decoded = np.array(Image.open(io.BytesIO(out)))
            assert np.array_equal(decoded, arr), f"level={level} not lossless"

    def test_jpg_quality_default_95(self):
        arr = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        default_bytes = pil_to_bytes(img, "jpg")
        q95 = pil_to_bytes(img, "jpg", quality=95)
        assert default_bytes == q95


# ---------------------------------------------------------------------------
# pil_to_bytes: RGB/RGBA 的 PNG 快速编码器
# ---------------------------------------------------------------------------

def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, out = 8, []
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        tag = data[pos + 4 : pos + 8]
        out.append((tag, data[pos + 8 : pos + 8 + length]))
        pos += 12 + length
    return out


class TestPngFastEncoder:
    """自定义 PNG 写出器必须与 Pillow 逐像素一致，元数据块逐字节一致。"""

    def test_rgb_rgba_roundtrip_is_exact(self) -> None:
        rng = np.random.default_rng(3)
        # 257 行跨越 256 行的滤波分块边界，覆盖 prior 行的接缝
        for shape in ((1, 1, 3), (64, 64, 3), (257, 300, 3), (300, 257, 4)):
            arr = rng.integers(0, 256, shape, dtype=np.uint8)
            img = Image.fromarray(arr)
            for level in (0, 1, 6, 9):
                out = pil_to_bytes(img, "png", png_compress_level=level)
                decoded = np.array(Image.open(io.BytesIO(out)))
                assert np.array_equal(decoded, arr), (shape, level)

    def test_compress_strategy_follows_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RGB 默认级别用 Z_RLE，RGBA 与高压缩级别仍用默认策略。

        Z_RLE 只发距离 1 的匹配：RGB 的 Up 滤波输出吃这个，RGBA 交错 alpha
        吃不到（体积 +52%）。它忽略级别，所以 level 6/9 不能让给它。
        """
        seen: list[tuple] = []
        real = zlib.compressobj

        def spy(*args: Any) -> Any:
            seen.append(args)
            return real(*args)

        monkeypatch.setattr(zlib, "compressobj", spy)
        pil_to_bytes(np.zeros((8, 8, 3), np.uint8), "png")
        pil_to_bytes(np.zeros((8, 8, 4), np.uint8), "png")
        pil_to_bytes(np.zeros((8, 8, 3), np.uint8), "png", png_compress_level=6)
        assert [args[-1] for args in seen] == [
            zlib.Z_RLE,
            zlib.Z_DEFAULT_STRATEGY,
            zlib.Z_DEFAULT_STRATEGY,
        ]

    def test_metadata_chunks_match_pillow(self) -> None:
        arr = np.zeros((5, 7, 3), dtype=np.uint8)
        arr[:] = (10, 20, 30)
        parameters = "prompt 参数"
        infos = {
            "icc_profile": b"FAKEICC" * 20,
            "dpi": (300, 300),
            "exif": b"Exif\x00\x00II*\x00" + b"\x00" * 10,
            "parameters": parameters,
        }
        fast = pil_to_bytes(Image.fromarray(arr), "png", infos=infos)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", parameters)
        buf = io.BytesIO()
        Image.fromarray(arr).save(
            buf,
            "PNG",
            compress_level=1,
            icc_profile=infos["icc_profile"],
            dpi=infos["dpi"],
            exif=infos["exif"],
            pnginfo=pnginfo,
        )
        # IDAT 的 zlib 参数与滤波选择不同，其余块必须完全一致
        fast_chunks = [c for c in _png_chunks(fast) if c[0] != b"IDAT"]
        pil_chunks = [c for c in _png_chunks(buf.getvalue()) if c[0] != b"IDAT"]
        assert [tag for tag, _ in fast_chunks] == [
            b"IHDR",
            b"iCCP",
            b"iTXt",
            b"pHYs",
            b"eXIf",
            b"IEND",
        ]
        assert fast_chunks == pil_chunks

        reopened = Image.open(io.BytesIO(fast))
        assert reopened.info["icc_profile"] == infos["icc_profile"]
        assert reopened.info["parameters"] == parameters
        assert reopened.info["exif"] == infos["exif"]
        assert reopened.info["dpi"][0] == pytest.approx(300, abs=0.01)

    def test_non_rgb_modes_still_use_pillow(self) -> None:
        arr = np.arange(32 * 16, dtype=np.uint8).reshape(32, 16)
        out = pil_to_bytes(Image.fromarray(arr, mode="L"), "png")
        assert np.array_equal(np.array(Image.open(io.BytesIO(out))), arr)


class TestJpegSubsampling:
    """导出必须显式传 subsampling，Pillow 默认会落 4:2:0。"""

    def _photo_like(self) -> np.ndarray:
        rng = np.random.default_rng(11)
        arr = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        # 平滑渐变 + 少量饱和色块：既不像纯噪声，也保留足够的色度细节
        x = np.linspace(0, 255, 128, dtype=np.uint8)
        arr[:, :, 0] = np.tile(x, (96, 1))
        arr[:, :, 1] = np.tile(x[::-1], (96, 1))
        arr[24:72, 32:96] = (220, 30, 180)
        return arr

    def _psnr(self, a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

    def test_pil_to_bytes_honors_subsampling(self) -> None:
        img = Image.fromarray(self._photo_like())
        assert JpegImagePlugin.get_sampling(
            Image.open(io.BytesIO(pil_to_bytes(img, "jpeg", quality=100, subsampling=0)))
        ) == 0
        assert JpegImagePlugin.get_sampling(
            Image.open(io.BytesIO(pil_to_bytes(img, "jpeg", quality=100, subsampling=2)))
        ) == 2
        # 不传时保持旧行为：libjpeg 默认 4:2:0
        assert JpegImagePlugin.get_sampling(
            Image.open(io.BytesIO(pil_to_bytes(img, "jpeg", quality=100)))
        ) == 2

    def test_q100_444_beats_q100_420(self) -> None:
        arr = self._photo_like()
        img = Image.fromarray(arr)
        decoded = {}
        for sampling in (0, 2):
            data = pil_to_bytes(img, "jpeg", quality=100, subsampling=sampling)
            decoded[sampling] = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        psnr_444 = self._psnr(decoded[0], arr)
        psnr_420 = self._psnr(decoded[2], arr)
        # 计划实测 12MP 照片 50.32 vs 36.45dB；这里只锁方向与量级（同图比较，不引用外部数字）
        assert psnr_444 > psnr_420 + 8.0, (psnr_444, psnr_420)

    def test_jpeg_sampling_helpers(self) -> None:
        img = Image.fromarray(self._photo_like())
        jpeg = pil_to_bytes(img, "jpeg", quality=90, subsampling=1)
        png = pil_to_bytes(img, "png")
        assert jpeg_sampling_from_bytes(jpeg) == 1
        assert jpeg_sampling_from_bytes(png) is None
        with Image.open(io.BytesIO(jpeg)) as opened:
            assert jpeg_sampling(opened) == 1


# ---------------------------------------------------------------------------
# D-2 输入边界
# ---------------------------------------------------------------------------

class TestInputBounds:
    def test_sd_steps_zero_rejected(self):
        with pytest.raises(ValidationError) as ei:
            InpaintRequest(sd_steps=0)
        assert ei.value.errors()[0]["loc"] == ("sd_steps",)
        # ge=1 的下边界：1 必须放行，否则合法请求会被 422
        assert InpaintRequest(sd_steps=1).sd_steps == 1

    def test_sd_steps_huge_rejected(self):
        with pytest.raises(ValidationError) as ei:
            InpaintRequest(sd_steps=1_000_000)
        assert ei.value.errors()[0]["loc"] == ("sd_steps",)

    def test_ldm_steps_zero_rejected(self):
        with pytest.raises(ValidationError) as ei:
            InpaintRequest(ldm_steps=0)
        assert ei.value.errors()[0]["loc"] == ("ldm_steps",)
        assert InpaintRequest(ldm_steps=1).ldm_steps == 1

    def test_ldm_steps_huge_rejected(self):
        with pytest.raises(ValidationError) as ei:
            InpaintRequest(ldm_steps=1_000_000)
        assert ei.value.errors()[0]["loc"] == ("ldm_steps",)

    def test_steps_in_range_ok(self):
        assert InpaintRequest(sd_steps=100).sd_steps == 100
        assert InpaintRequest(ldm_steps=50).ldm_steps == 50

    def test_upload_part_limit_replaces_base64_field_cap(self) -> None:
        """图片/掩膜改走 multipart 文件部件后，原来 64M 字符的 base64 字段上限
        由 `MAX_UPLOAD_BYTES` 承担：64M 字符解码后正好约 48MB 原始字节。"""
        assert api_mod.MAX_UPLOAD_BYTES == 48 * 1024 * 1024

    def test_plugin_image_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            RunPluginRequest(name="x", image="A" * 64_000_001)
        assert ei.value.errors()[0]["loc"] == ("image",)
        assert RunPluginRequest(name="x", image="A" * 64_000_000).image

    def test_adjust_mask_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            AdjustMaskRequest(mask="A" * 64_000_001, operate="expand")
        assert ei.value.errors()[0]["loc"] == ("mask",)
        assert AdjustMaskRequest(mask="A" * 64_000_000, operate="expand").mask

    def test_plugin_scale_bounds(self) -> None:
        """scale 会作为 outscale 乘到输出尺寸上（12MP × 1000 ≈ 1.2e13 像素）。"""
        for bad in (0, -1, 1000):
            with pytest.raises(ValidationError) as ei:
                RunPluginRequest(name="x", image="aGk=", scale=bad)
            assert ei.value.errors()[0]["loc"] == ("scale",)
        for ok in (2, 4):
            assert RunPluginRequest(name="x", image="aGk=", scale=ok).scale == ok

    def test_kernel_size_bounds(self):
        for bad in (0, 1_000_000):
            with pytest.raises(ValidationError) as ei:
                AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=bad)
            assert ei.value.errors()[0]["loc"] == ("kernel_size",)
        assert (
            AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=12).kernel_size
            == 12
        )

    def test_kernel_size_capped_at_ui_max(self) -> None:
        """kernel_size 代价随尺寸平方增长（k=1000 实测 50.8s/次），
        上限与 WebUI 滑块 max=100 对齐。"""
        with pytest.raises(ValidationError):
            AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=101)
        assert (
            AdjustMaskRequest(
                mask="aGk=", operate="expand", kernel_size=100
            ).kernel_size
            == 100
        )

    def test_mask_strokes_total_points_bounded(self) -> None:
        """单条 20 万点、最多 1 万条都不限总量：2M 点光栅化实测 5.3s CPU，
        因此额外限制所有笔迹的总点数。"""

        def stroke(n: int) -> dict:
            return {"pts": [[1.0, 1.0]] * n, "size": 8}

        with pytest.raises(ValidationError):
            InpaintRequest(mask_strokes=[stroke(50_000), stroke(50_001)])
        req = InpaintRequest(mask_strokes=[stroke(50_000), stroke(50_000)])
        assert req.mask_strokes is not None
        assert sum(len(s.pts) for s in req.mask_strokes) == 100_000
        # 正常小请求不受影响
        normal = InpaintRequest(
            mask_strokes=[
                {"pts": [[0, 0], [1, 1]], "size": 5},
                {"pts": [[2, 2]], "size": 5},
            ]
        )
        assert normal.mask_strokes is not None
        assert len(normal.mask_strokes) == 2

    def test_mask_stroke_size_capped_at_cv2_max_thickness(self) -> None:
        """size 直接当作 cv2.polylines/circle 的 thickness，超过 OpenCV 的
        MAX_THICKNESS(32767) 会断言失败变成 500，所以在 schema 边界就对不齐。"""
        with pytest.raises(ValidationError):
            InpaintRequest(mask_strokes=[{"pts": [[0, 0]], "size": 32_768}])
        req = InpaintRequest(mask_strokes=[{"pts": [[0, 0]], "size": 32_767}])
        assert req.mask_strokes is not None
        assert req.mask_strokes[0].size == 32_767

    def test_croper_coord_bounds(self):
        with pytest.raises(ValidationError):
            InpaintRequest(croper_x=999_999)
        with pytest.raises(ValidationError):
            InpaintRequest(extender_x=-999_999)
        # extender 坐标可为负（base.py 注释），小幅负值要放行
        assert InpaintRequest(extender_x=-100).extender_x == -100

    def test_crop_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            InpaintRequest(croper_width=0)
        with pytest.raises(ValidationError):
            InpaintRequest(extender_height=-1)

    def test_hd_strategy_limits_bounded(self):
        with pytest.raises(ValidationError):
            InpaintRequest(hd_strategy_resize_limit=0)
        with pytest.raises(ValidationError):
            InpaintRequest(hd_strategy_crop_trigger_size=10_000_000)

    def test_float_clicks_rejected_by_schema(self):
        """待验证 #1 的结论：pydantic lax 模式**拒绝**小数而非截断。

        所以前端必须在发送前取整（D-12），否则触屏浮点坐标会打到 422。
        """
        with pytest.raises(ValidationError):
            RunPluginRequest(name="x", image="aGk=", clicks=[[1.7, 2.9, 1]])

    def test_int_clicks_ok(self):
        req = RunPluginRequest(name="x", image="aGk=", clicks=[[10, 20, 1]])
        assert req.clicks == [[10, 20, 1]]


# ---------------------------------------------------------------------------
# D-7 ModelLoadError
# ---------------------------------------------------------------------------

class TestModelLoadError:
    def test_is_runtime_error(self):
        """必须是 Exception 子类，否则会像 SystemExit 一样绕过 FastAPI 异常处理。"""
        assert issubclass(ModelLoadError, RuntimeError)

    def test_not_base_exception_only(self):
        assert not issubclass(ModelLoadError, BaseException) or issubclass(
            ModelLoadError, Exception
        )

    def test_helper_download_raises_model_load_error(self, monkeypatch, tmp_path):
        from iopaint import helper

        # 下载到一个 md5 不匹配的文件 → 必须抛 ModelLoadError（原为 SystemExit）
        cached = tmp_path / "model.pt"
        assert not cached.exists()

        def fake_get_cache_path(url):
            return str(cached)

        def fake_download(url, cached_file, hash_prefix, progress=True):
            with open(cached_file, "wb") as fw:
                fw.write(b"not a real model")

        monkeypatch.setattr(helper, "get_cache_path_by_url", fake_get_cache_path)
        monkeypatch.setattr(helper, "download_url_to_file", fake_download)

        with pytest.raises(ModelLoadError):
            helper.download_model(
                "http://example.com/model.pt", model_md5="deadbeef"
            )
        # 错误的模型文件应当被删掉
        assert not cached.exists()

    def test_handle_error_raises_model_load_error(self, tmp_path):
        from iopaint import helper

        p = tmp_path / "m.bin"
        p.write_bytes(b"abc")
        with pytest.raises(ModelLoadError):
            helper.handle_error(str(p), "0000", Exception("boom"))

    def test_base_plugin_raises_on_missing_dep(self, monkeypatch):
        from iopaint.plugins.base_plugin import BasePlugin

        class Fake(BasePlugin):
            name = "fake"

            def check_dep(self):
                return "missing dependency foo"

        with pytest.raises(ModelLoadError):
            Fake()

    def test_handle_from_pretrained_raises_model_load_error(
        self, monkeypatch, tmp_path
    ):
        from iopaint.model.utils import handle_from_pretrained_exceptions

        def boom(**kwargs):
            raise OSError("Max retries exceeded with host: example.com")

        with pytest.raises(ModelLoadError):
            handle_from_pretrained_exceptions(boom)
