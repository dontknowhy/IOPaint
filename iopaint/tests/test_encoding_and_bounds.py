"""编码参数与输入边界（PERF-2、DOS-1、REL-1）。
"""
import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from iopaint.exceptions import ModelLoadError
from iopaint.helper import numpy_to_bytes, pil_to_bytes
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
# D-2 输入边界
# ---------------------------------------------------------------------------

class TestInputBounds:
    def test_sd_steps_zero_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(sd_steps=0)

    def test_sd_steps_huge_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(sd_steps=1_000_000)

    def test_ldm_steps_zero_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(ldm_steps=0)

    def test_ldm_steps_huge_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(ldm_steps=1_000_000)

    def test_steps_in_range_ok(self):
        assert InpaintRequest(sd_steps=100).sd_steps == 100
        assert InpaintRequest(ldm_steps=50).ldm_steps == 50

    def test_oversize_image_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(image="x" * 64_000_001)

    def test_oversize_mask_rejected(self):
        with pytest.raises(ValidationError):
            InpaintRequest(mask="x" * 64_000_001)

    def test_normal_size_image_ok(self):
        req = InpaintRequest(image="a" * 1000, mask="b" * 1000)
        assert req.image is not None

    def test_kernel_size_bounds(self):
        with pytest.raises(ValidationError):
            AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=0)
        with pytest.raises(ValidationError):
            AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=1_000_000)
        assert (
            AdjustMaskRequest(mask="aGk=", operate="expand", kernel_size=12).kernel_size
            == 12
        )

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
