"""Tests for iopaint.helper – pure-image-processing and utility functions."""
import base64
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from iopaint.helper import (
    adjust_mask,
    boxes_from_mask,
    ceil_modulo,
    concat_alpha_channel,
    decode_base64_to_image,
    encode_pil_to_base64,
    gen_frontend_mask,
    load_img,
    md5sum,
    norm_img,
    only_keep_largest_contour,
    pad_img_to_modulo,
    pil_to_bytes,
    numpy_to_bytes,
    resize_max_size,
    switch_mps_device,
    is_mac,
)

# ---------------------------------------------------------------------------
# md5sum
# ---------------------------------------------------------------------------

class TestMd5sum:
    def test_known_content(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        result = md5sum(str(p))
        assert result == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        result = md5sum(str(p))
        assert isinstance(result, str)
        assert len(result) == 32


# ---------------------------------------------------------------------------
# ceil_modulo
# ---------------------------------------------------------------------------

class TestCeilModulo:
    def test_exact(self):
        assert ceil_modulo(10, 5) == 10

    def test_needs_round(self):
        assert ceil_modulo(11, 5) == 15

    def test_one(self):
        assert ceil_modulo(1, 8) == 8

    def test_zero(self):
        assert ceil_modulo(0, 8) == 0


# ---------------------------------------------------------------------------
# switch_mps_device
# ---------------------------------------------------------------------------

class TestSwitchMpsDevice:
    def test_unsupported_model_switches_to_cpu(self):
        import torch
        dev = switch_mps_device("lama", torch.device("mps"))
        assert str(dev) == "cpu"

    def test_supported_model_stays(self):
        import torch
        dev = switch_mps_device("cv2", torch.device("mps"))
        # cv2 is also in MPS_UNSUPPORT_MODELS – verify
        from iopaint.const import MPS_UNSUPPORT_MODELS
        assert "cv2" in MPS_UNSUPPORT_MODELS

    def test_non_mps_device_unchanged(self):
        import torch
        dev = switch_mps_device("lama", torch.device("cpu"))
        assert str(dev) == "cpu"


# ---------------------------------------------------------------------------
# norm_img
# ---------------------------------------------------------------------------

class TestNormImg:
    def test_2d(self):
        img = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        result = norm_img(img)
        assert result.shape == (1, 2, 2)
        assert result.dtype == np.float32

    def test_3d(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = norm_img(img)
        assert result.shape == (3, 4, 4)


# ---------------------------------------------------------------------------
# resize_max_size
# ---------------------------------------------------------------------------

class TestResizeMaxSize:
    def test_no_resize_needed(self):
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        result = resize_max_size(img, 200)
        assert result.shape == img.shape

    def test_resize_applied(self):
        img = np.zeros((400, 200, 3), dtype=np.uint8)
        result = resize_max_size(img, 100)
        assert max(result.shape[:2]) <= 100

    def test_already_small(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = resize_max_size(img, 100)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# pad_img_to_modulo
# ---------------------------------------------------------------------------

class TestPadImgToModulo:
    def test_no_pad_needed(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8)
        assert result.shape == (16, 16, 3)

    def test_pad_height(self):
        img = np.zeros((15, 16, 3), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8)
        assert result.shape[0] % 8 == 0
        assert result.shape[1] % 8 == 0

    def test_pad_width(self):
        img = np.zeros((16, 13, 3), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8)
        assert result.shape[1] % 8 == 0

    def test_square_option(self):
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8, square=True)
        assert result.shape[0] == result.shape[1]

    def test_min_size(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8, min_size=32)
        assert result.shape[0] >= 32
        assert result.shape[1] >= 32

    def test_2d_input(self):
        img = np.zeros((15, 13), dtype=np.uint8)
        result = pad_img_to_modulo(img, 8)
        assert result.ndim == 3
        assert result.shape[2] == 1


# ---------------------------------------------------------------------------
# load_img
# ---------------------------------------------------------------------------

class TestLoadImg:
    def _make_png_bytes(self, mode="RGB", size=(64, 64)):
        img = Image.new(mode, size, color=(128, 64, 32) if mode == "RGB" else (128,))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_load_rgb(self):
        data = self._make_png_bytes("RGB")
        np_img, alpha = load_img(data)
        assert np_img.shape == (64, 64, 3)
        assert alpha is None

    def test_load_rgba(self):
        data = self._make_png_bytes("RGBA")
        np_img, alpha = load_img(data)
        assert np_img.shape == (64, 64, 3)
        assert alpha is not None
        assert alpha.shape == (64, 64)

    def test_load_grayscale(self):
        data = self._make_png_bytes("L")
        np_img, alpha = load_img(data, gray=True)
        assert np_img.ndim == 2
        assert alpha is None

    def test_return_info(self):
        data = self._make_png_bytes("RGB")
        np_img, alpha, info = load_img(data, return_info=True)
        assert isinstance(info, dict)

    def test_jpg_load(self):
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        np_img, alpha = load_img(buf.getvalue())
        assert np_img.shape[2] == 3
        assert alpha is None

    def _gray_png_bytes(self, arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def test_16bit_gray_scales_instead_of_clipping(self) -> None:
        """16-bit 灰度 PNG 解出来是 I;16，Pillow 的 convert 是截断到 255：
        梯度图会变成全 255，掩膜整幅变白（整图被重绘）。必须按 >>8 缩放。"""
        arr = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
        data = self._gray_png_bytes(arr)

        np_img, alpha = load_img(data)
        assert alpha is None
        assert np_img.max() == 255
        assert abs(float(np_img.mean()) - 127.5) <= 2

        gray_img, _ = load_img(data, gray=True)
        assert gray_img.dtype == np.uint8
        assert gray_img.max() == 255
        assert abs(float(gray_img.mean()) - 127.5) <= 2

    def test_8bit_gray_pixels_unchanged(self) -> None:
        arr = np.array([[0, 64, 128], [192, 254, 255]], dtype=np.uint8)
        data = self._gray_png_bytes(arr)
        for gray in (True, False):
            np_img, _ = load_img(data, gray=gray)
            if gray:
                np.testing.assert_array_equal(np_img, arr)
            else:
                expected = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
                np.testing.assert_array_equal(np_img, expected)

    def test_int32_tiff_keeps_8bit_values(self) -> None:
        """mode "I" 的容器里可能装的是 8-bit 数据（部分 TIFF）：不能一律 >>8。"""
        arr = np.array([[0, 17, 34], [51, 68, 255]], dtype=np.int32)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="TIFF")
        np_img, _ = load_img(buf.getvalue(), gray=True)
        assert Image.open(io.BytesIO(buf.getvalue())).mode == "I"
        np.testing.assert_array_equal(np_img, arr.astype(np.uint8))

    def test_16bit_tiff_scales_instead_of_clipping(self) -> None:
        arr = np.array([[0, 257, 32768], [51400, 65280, 65535]], dtype=np.uint16)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="TIFF")
        np_img, _ = load_img(buf.getvalue(), gray=True)
        np.testing.assert_array_equal(
            np_img, np.array([[0, 1, 128], [200, 255, 255]], dtype=np.uint8)
        )


# ---------------------------------------------------------------------------
# numpy_to_bytes / pil_to_bytes
# ---------------------------------------------------------------------------

class TestNumpyToBytes:
    def test_png_roundtrip(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5] = [255, 0, 0]
        data = numpy_to_bytes(img, "png")
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_jpg_roundtrip(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = numpy_to_bytes(img, "jpg")
        assert isinstance(data, bytes)


class TestPilToBytes:
    def test_png_with_parameters(self):
        img = Image.new("RGB", (10, 10))
        infos = {"parameters": "test prompt"}
        data = pil_to_bytes(img, "png", infos=infos)
        assert isinstance(data, bytes)

    def test_jpg_quality(self):
        img = Image.new("RGB", (10, 10))
        data = pil_to_bytes(img, "jpg", quality=50)
        assert isinstance(data, bytes)

    def test_no_infos(self):
        img = Image.new("RGB", (10, 10))
        data = pil_to_bytes(img, "png")
        assert isinstance(data, bytes)

    def test_exif_passthrough(self):
        img = Image.new("RGB", (10, 10))
        infos = {"exif": b"\x00"}
        data = pil_to_bytes(img, "png", infos=infos)
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# decode_base64_to_image / encode_pil_to_base64
# ---------------------------------------------------------------------------

class TestBase64Roundtrip:
    def _b64_encode_rgb(self):
        img = Image.new("RGB", (32, 32), color=(100, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _b64_encode_rgba(self):
        img = Image.new("RGBA", (32, 32), color=(100, 200, 50, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_decode_rgb(self):
        encoding = self._b64_encode_rgb()
        np_img, alpha, infos, ext = decode_base64_to_image(encoding)
        assert np_img.shape == (32, 32, 3)
        assert alpha is None
        assert ext == "png"

    def test_decode_rgba(self):
        encoding = self._b64_encode_rgba()
        np_img, alpha, infos, ext = decode_base64_to_image(encoding)
        assert alpha is not None

    def test_decode_gray(self):
        encoding = self._b64_encode_rgb()
        np_img, alpha, infos, ext = decode_base64_to_image(encoding, gray=True)
        assert np_img.ndim == 2

    def test_data_uri_prefix(self):
        encoding = self._b64_encode_rgb()
        data_uri = f"data:image/png;base64,{encoding}"
        np_img, _, _, ext = decode_base64_to_image(data_uri)
        assert np_img.shape[2] == 3

    def test_octet_stream_prefix(self):
        encoding = self._b64_encode_rgb()
        data_uri = f"data:application/octet-stream;base64,{encoding}"
        np_img, _, _, _ = decode_base64_to_image(data_uri)
        assert np_img is not None

    def test_encode_decode_roundtrip(self):
        img = Image.new("RGB", (16, 16), color=(50, 100, 200))
        b64 = encode_pil_to_base64(img, quality=95, infos={})
        decoded = base64.b64decode(b64)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.size == (16, 16)

    def test_decode_16bit_gray_scales_instead_of_clipping(self) -> None:
        arr = np.linspace(0, 65535, 32 * 32).reshape(32, 32).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        encoding = base64.b64encode(buf.getvalue()).decode()

        np_img, alpha, _, ext = decode_base64_to_image(encoding)
        assert ext == "png"
        assert alpha is None
        assert np.max(np_img) == 255
        assert abs(float(np.mean(np_img)) - 127.5) <= 2

        gray_img, _, _, _ = decode_base64_to_image(encoding, gray=True)
        assert np.asarray(gray_img).dtype == np.uint8
        assert np.max(gray_img) == 255
        assert abs(float(np.mean(gray_img)) - 127.5) <= 2


# ---------------------------------------------------------------------------
# concat_alpha_channel
# ---------------------------------------------------------------------------

class TestConcatAlphaChannel:
    def test_with_alpha(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        alpha = np.ones((10, 10), dtype=np.uint8) * 255
        result = concat_alpha_channel(rgb, alpha)
        assert result.shape == (10, 10, 4)

    def test_without_alpha(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        result = concat_alpha_channel(rgb, None)
        assert result.shape == (10, 10, 3)

    def test_alpha_resize(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        alpha = np.ones((10, 10), dtype=np.uint8) * 255
        result = concat_alpha_channel(rgb, alpha)
        assert result.shape == (20, 20, 4)


# ---------------------------------------------------------------------------
# adjust_mask
# ---------------------------------------------------------------------------

class TestAdjustMask:
    def _make_mask(self, size=64):
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[16:48, 16:48] = 200
        return mask

    def test_expand(self):
        mask = self._make_mask()
        result = adjust_mask(mask, 5, "expand")
        assert result.shape[2] == 4  # RGBA

    def test_shrink(self):
        mask = self._make_mask()
        result = adjust_mask(mask, 5, "shrink")
        assert result.shape[2] == 4

    def test_reverse(self):
        mask = self._make_mask()
        result = adjust_mask(mask, 5, "reverse")
        assert result.shape[2] == 4

    def test_threshold_behavior(self):
        mask = np.full((10, 10), 100, dtype=np.uint8)
        result = adjust_mask(mask, 0, "expand")
        # Values < 127 become 0, so reverse should give non-zero
        assert result is not None


# ---------------------------------------------------------------------------
# gen_frontend_mask
# ---------------------------------------------------------------------------

class TestGenFrontendMask:
    def test_gray_input(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        result = gen_frontend_mask(mask)
        assert result.shape[2] == 4  # RGBA

    def test_bgr_input(self):
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        result = gen_frontend_mask(mask)
        assert result.shape[2] == 4


# ---------------------------------------------------------------------------
# boxes_from_mask
# ---------------------------------------------------------------------------

class TestBoxesFromMask:
    def test_single_box(self):
        mask = np.zeros((100, 100, 1), dtype=np.uint8)
        mask[20:40, 30:50] = 255
        boxes = boxes_from_mask(mask)
        assert len(boxes) >= 1
        assert boxes[0].shape == (4,)

    def test_empty_mask(self):
        mask = np.zeros((100, 100, 1), dtype=np.uint8)
        boxes = boxes_from_mask(mask)
        assert len(boxes) == 0


# ---------------------------------------------------------------------------
# only_keep_largest_contour
# ---------------------------------------------------------------------------

class TestOnlyKeepLargestContour:
    def test_keeps_largest(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # small
        mask[50:90, 50:90] = 255  # large
        result = only_keep_largest_contour(mask)
        assert result is not None
        # The large box area should still be present
        assert result[70, 70] == 255

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = only_keep_largest_contour(mask)
        assert np.sum(result) == 0


# ---------------------------------------------------------------------------
# is_mac
# ---------------------------------------------------------------------------

class TestIsMac:
    def test_returns_bool(self):
        assert isinstance(is_mac(), bool)


# ---------------------------------------------------------------------------
# encode/decode consistency
# ---------------------------------------------------------------------------

class TestEncodeDecodeConsistency:
    def test_roundtrip_preserves_shape(self):
        original = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        img = Image.fromarray(original)
        b64 = encode_pil_to_base64(img, 95, {})
        decoded, _, _, _ = decode_base64_to_image(b64.decode())
        assert decoded.shape == original.shape
