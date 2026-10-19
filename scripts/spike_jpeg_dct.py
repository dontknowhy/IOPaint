"""纯 Python JPEG DCT 系数域局部替换的实测脚本。

只做测量，不参与产品路径。输出可复跑的数字：

- unchanged roundtrip：空改动时输出解码像素与源解码逐像素一致；
- 4:4:4 局部替换：改动区域外像素逐像素一致；
- 4:2:0 局部替换：改动区域外**系数**一致，像素只在边界受 chroma 上采样扩散影响；
- progressive fallback：返回 None；
- 12MP 图像上的编解码耗时（3 次取中位）。

运行：
    /home/dontknowhy/miniconda3/envs/iopaint/bin/python scripts/spike_jpeg_dct.py
"""
from __future__ import annotations

import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iopaint.jpeg_dct import _decode_coefficients, _parse, patch_jpeg  # noqa: E402


WIDTH, HEIGHT = 4000, 3000  # 12MP


def _image() -> np.ndarray:
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    arr = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    arr[:, :, 0] = (xx * 255 // WIDTH).astype(np.uint8)
    arr[:, :, 1] = (yy * 255 // HEIGHT).astype(np.uint8)
    arr[:, :, 2] = ((xx + yy) * 255 // (WIDTH + HEIGHT)).astype(np.uint8)
    # 加一点纹理，避免平滑图把耗时测低
    noise = rng.integers(-12, 13, (HEIGHT, WIDTH, 3), dtype=np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _jpeg(arr: np.ndarray, subsampling: int, progressive: bool = False) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf,
        format="JPEG",
        quality=95,
        subsampling=subsampling,
        progressive=progressive,
    )
    return buf.getvalue()


def _decode(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _changed_blocks(shape, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros((shape[0] // 8, shape[1] // 8), dtype=bool)
    mask[y1 // 8 : (y2 + 7) // 8, x1 // 8 : (x2 + 7) // 8] = True
    return mask


def _coefficients(data: bytes):
    j = _parse(data)
    decoded = _decode_coefficients(j)
    return j, decoded["coeffs"], decoded["mcu_cols"], decoded["mcu_rows"]


def _outside_blocks(changed: np.ndarray, comp_h: int, comp_v: int) -> np.ndarray:
    """按分量块展开 luma changed mask：任一所覆盖 luma 块改动即视为改动。"""
    bh, bw = changed.shape
    out = np.zeros((bh * comp_v, bw * comp_h), dtype=bool)
    for vy in range(comp_v):
        for hx in range(comp_h):
            out[vy::comp_v, hx::comp_h] = changed
    return out


def _median_ms(fn, repeats=3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _report_common(name: str, source: bytes, out: bytes, changed: np.ndarray, box) -> None:
    src = _decode(source)
    dec = _decode(out)
    x1, y1, x2, y2 = box
    outside = np.ones(src.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    diff = np.abs(dec.astype(np.int16) - src.astype(np.int16)).max(axis=2)
    print(f"  {name}: bytes {len(source)} -> {len(out)}")
    print(
        f"  {name}: outside maxdiff={diff[outside].max()} "
        f"changed_pixels={(diff[outside] > 0).mean() * 100:.4f}%"
    )


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    print(f"repo={root} image={WIDTH}x{HEIGHT} ({WIDTH * HEIGHT / 1e6:.1f}MP)")

    arr = _image()
    box = (1400, 904, 2600, 2104)  # 1200x1200 ≈ 9.5% 面积，且与 8x8 块对齐
    changed = _changed_blocks(arr.shape, box)

    src444 = _jpeg(arr, subsampling=0)
    src420 = _jpeg(arr, subsampling=2)

    print("\n[encode/decode baselines, median ms]")
    print(f"  source encode q95 4:4:4: {_median_ms(lambda: _jpeg(arr, 0)):.1f}")
    print(f"  source decode 4:4:4:   {_median_ms(lambda: _decode(src444)):.1f}")
    print(f"  source encode q95 4:2:0: {_median_ms(lambda: _jpeg(arr, 2)):.1f}")
    print(f"  source decode 4:2:0:   {_median_ms(lambda: _decode(src420)):.1f}")

    print("\n[unchanged roundtrip]")
    empty = np.zeros_like(changed)
    t0 = time.perf_counter()
    out = patch_jpeg(src444, _decode(src444), empty)
    elapsed = (time.perf_counter() - t0) * 1000
    assert out is not None
    print(f"  patch time={elapsed:.1f}ms  output={len(out)}B")
    _report_common("unchanged", src444, out, empty, (0, 0, 0, 0))

    print("\n[4:4:4 local patch]")
    canonical = _decode(src444).copy()
    canonical[box[1] : box[3], box[0] : box[2]] = (0, 255, 0)
    t0 = time.perf_counter()
    out444 = patch_jpeg(src444, canonical, changed)
    print(f"  patch time={(time.perf_counter() - t0) * 1000:.1f}ms")
    assert out444 is not None
    _report_common("444", src444, out444, changed, box)
    j_src, c_src, mc, mr = _coefficients(src444)
    j_out, c_out, _, _ = _coefficients(out444)
    outside_blocks = ~_outside_blocks(changed, j_src.components[0].h, j_src.components[0].v)
    same = all(
        np.array_equal(c_src[c.id][outside_blocks], c_out[c.id][outside_blocks])
        for c in j_src.components
    )
    print(f"  444: outside DCT coefficients identical={same}")

    print("\n[4:2:0 local patch]")
    canonical = _decode(src420).copy()
    canonical[box[1] : box[3], box[0] : box[2]] = (0, 255, 0)
    t0 = time.perf_counter()
    out420 = patch_jpeg(src420, canonical, changed)
    print(f"  patch time={(time.perf_counter() - t0) * 1000:.1f}ms")
    assert out420 is not None
    _report_common("420", src420, out420, changed, box)
    j_src, c_src, _, _ = _coefficients(src420)
    _, c_out, _, _ = _coefficients(out420)
    hmax = max(x.h for x in j_src.components)
    vmax = max(x.v for x in j_src.components)
    same = {}
    for c in j_src.components:
        sx = hmax // c.h
        sy = vmax // c.v
        comp_outside = np.zeros(c_src[c.id].shape[:2], dtype=bool)
        for by in range(comp_outside.shape[0]):
            for bx in range(comp_outside.shape[1]):
                x0, y0 = bx * 8 * sx, by * 8 * sy
                comp_outside[by, bx] = not changed[
                    y0 // 8 : (y0 + 8 * sy + 7) // 8,
                    x0 // 8 : (x0 + 8 * sx + 7) // 8,
                ].any()
        same[c.id] = bool(
            np.array_equal(c_src[c.id][comp_outside], c_out[c.id][comp_outside])
        )
    print(f"  420: outside DCT coefficients identical={same}")

    print("\n[progressive fallback]")
    src_prog = _jpeg(arr, subsampling=2, progressive=True)
    print(f"  patch_jpeg(progressive) -> {patch_jpeg(src_prog, _decode(src_prog), changed)}")

    print("\n[boundary artifact ring, 444 vs 420]")
    for name, source, out in (("444", src444, out444), ("420", src420, out420)):
        src = _decode(source)
        dec = _decode(out)
        diff = np.abs(dec.astype(np.int16) - src.astype(np.int16)).max(axis=2)
        x1, y1, x2, y2 = box
        ring = np.zeros(src.shape[:2], dtype=bool)
        ring[max(0, y1 - 8) : y2 + 8, max(0, x1 - 8) : x2 + 8] = True
        ring[y1:y2, x1:x2] = False
        print(
            f"  {name}: ring mean={diff[ring].mean():.3f} max={diff[ring].max()} "
            f"changed={100 * (diff[ring] > 0).mean():.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
