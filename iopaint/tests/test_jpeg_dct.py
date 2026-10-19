"""baseline JPEG 系数域局部替换的可执行回归。"""
import io
from typing import Any

import numpy as np
from PIL import Image

from iopaint.jpeg_dct import patch_jpeg


def _jpeg(arr: np.ndarray, **kwargs: Any) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def _decode(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def _block_mask(changed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.repeat(np.repeat(changed, 8, axis=0), 8, axis=1)
    return mask[: shape[0], : shape[1]]


def test_unchanged_file_roundtrips_pixel_exact() -> None:
    arr = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = _jpeg(arr, quality=95)
    out = patch_jpeg(source, arr, np.zeros((8, 8), dtype=bool))
    assert out is not None
    assert np.array_equal(_decode(out), _decode(source))


def test_444_outside_changed_blocks_is_pixel_exact() -> None:
    arr = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = _jpeg(arr, quality=95, subsampling=0)
    canonical = arr.copy()
    canonical[16:32, 16:32] = [0, 255, 0]
    changed = np.zeros((8, 8), dtype=bool)
    changed[2:4, 2:4] = True

    out = patch_jpeg(source, canonical, changed)
    assert out is not None
    outside = ~_block_mask(changed, arr.shape)
    assert np.abs(_decode(out) - _decode(source))[outside].max() == 0


def test_420_keeps_outside_coefficients_but_chroma_filter_spreads_boundary() -> None:
    arr = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = _jpeg(arr, quality=95, subsampling=2)
    canonical = arr.copy()
    canonical[16:32, 16:32] = [0, 255, 0]
    changed = np.zeros((8, 8), dtype=bool)
    changed[2:4, 2:4] = True
    out = patch_jpeg(source, canonical, changed)
    assert out is not None
    # 4:2:0 的 fancy upsampling 会把改动的色度块混到相邻像素，因此像素级
    # 只能保证“未改动 DCT 块系数一致”。
    outside = ~_block_mask(changed, arr.shape)
    assert np.abs(_decode(out) - _decode(source))[outside].max() > 0


def test_progressive_is_unsupported() -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    source = _jpeg(arr, quality=90, progressive=True)
    assert patch_jpeg(source, arr, np.ones((4, 4), dtype=bool)) is None
