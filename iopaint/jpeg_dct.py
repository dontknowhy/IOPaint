"""Baseline JPEG 的 DCT 系数域局部替换。

目标：导出 JPEG 时，掩膜**外**的 8x8 块直接复用源文件的量化系数，只对
``changed_blocks`` 覆盖的块用 canonical 像素重新做 DCT + 量化。这样解码后
未改动的块逐像素等于源文件（同一张量化表、同一份 Huffman 表、同一套采样），
不依赖 jpegio / jpegtran / 系统 libjpeg 工具。

约束：
- 只支持 baseline / extended-sequential 8-bit JPEG（SOF0/SOF1）；
  progressive（SOF2）、restart marker、CMYK/Adobe RGB 一律返回 None，
  调用方回退到整图 full 编码。
- 量化表与采样方式沿用源文件，因此掩膜内质量 = 源质量；需要 q100/4:4:4
  时由 ``/export`` 显式走 full 模式。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]


def _dct_matrix() -> np.ndarray:
    mat = np.zeros((8, 8), dtype=np.float64)
    for k in range(8):
        c = 1.0 / math.sqrt(2.0) if k == 0 else 1.0
        for n in range(8):
            mat[k, n] = 0.5 * c * math.cos((2 * n + 1) * k * math.pi / 16.0)
    return mat


_DCT = _dct_matrix()


class _Unsupported(Exception):
    pass


class _BitReader:
    """按字节填充的位读取器，配合 ``_Huffman.lut`` 做 O(1) Huffman 解码。

    纯 Python 逐 bit 读取在 12MP 图像上要几十秒；这里一次 peek 16 bit，
    查表得到 ``(symbol << 5) | length``，只对实际用到的长度做 skip。
    """

    __slots__ = ("data", "pos", "buf", "n")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.n = 0

    def _fill(self, need: int) -> None:
        while self.n < need and self.pos < len(self.data):
            self.buf = (self.buf << 8) | self.data[self.pos]
            self.pos += 1
            self.n += 8

    def peek(self, n: int) -> int:
        if self.n < n:
            self._fill(n)
        if self.n < n:
            # 数据末尾不足 n bit：低位补 0（解码循环在预期块数处停止）
            self.buf <<= n - self.n
            self.n = n
        return (self.buf >> (self.n - n)) & ((1 << n) - 1)

    def skip(self, n: int) -> None:
        self.n -= n
        self.buf = self.buf & ((1 << self.n) - 1) if self.n else 0

    def bits(self, n: int) -> int:
        if n <= 0:
            return 0
        value = self.peek(n)
        self.skip(n)
        return value

    def bit(self) -> int:
        return self.bits(1)

    def extend(self, n: int) -> int:
        if n == 0:
            return 0
        value = self.bits(n)
        if value < (1 << (n - 1)):
            value -= (1 << n) - 1
        return value


class _BitWriter:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, value: int, n: int) -> None:
        if n <= 0:
            return
        value &= (1 << n) - 1
        self.acc = (self.acc << n) | value
        self.nbits += n
        while self.nbits >= 8:
            self.nbits -= 8
            byte = (self.acc >> self.nbits) & 0xFF
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0x00)
        self.acc &= (1 << self.nbits) - 1 if self.nbits else 0

    def flush(self) -> bytes:
        if self.nbits:
            self.write((1 << (8 - self.nbits)) - 1, 8 - self.nbits)
        return bytes(self.out)


class _Huffman:
    def __init__(self, counts: List[int], symbols: List[int]) -> None:
        self.counts = counts
        self.symbols = symbols
        # 解码：mincode/maxcode/valptr（JPEG 规范 F.2.2.3）
        self.mincode = [0] * 17
        self.maxcode = [-1] * 18
        self.valptr = [0] * 17
        code = 0
        k = 0
        for length in range(1, 17):
            n = counts[length - 1]
            self.valptr[length] = k
            self.mincode[length] = code
            code += n
            k += n
            self.maxcode[length] = code - 1 if n else -1
            if n:
                self.maxcode[length] = code - 1
            code <<= 1
        # 编码表
        self.encode_map: Dict[int, Tuple[int, int]] = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                self.encode_map[symbols[k]] = (code, length)
                code += 1
                k += 1
            code <<= 1
        self.lut: Optional[List[int]] = None

    def _build_lut(self) -> None:
        # 16-bit 前缀表：entry = (symbol << 5) | code_length
        lut: List[int] = [-1] * (1 << 16)
        code = 0
        k = 0
        for length in range(1, 17):
            n = self.counts[length - 1]
            shift = 16 - length
            for _ in range(n):
                entry = (self.symbols[k] << 5) | length
                start = code << shift
                end = (code + 1) << shift
                lut[start:end] = [entry] * (end - start)
                code += 1
                k += 1
            code <<= 1
        self.lut = lut

    def decode(self, reader: _BitReader) -> int:
        if self.lut is None:
            self._build_lut()
        lut = self.lut
        assert lut is not None
        entry = lut[reader.peek(16)]
        if entry < 0:
            raise _Unsupported("bad huffman code")
        reader.skip(entry & 0x1F)
        return entry >> 5


class _Component:
    def __init__(self, cid: int, h: int, v: int, tq: int) -> None:
        self.id = cid
        self.h = h
        self.v = v
        self.tq = tq


class _Jpeg:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.precision = 8
        self.components: List[_Component] = []
        self.qtables: Dict[int, np.ndarray] = {}
        self.dc_tables: Dict[int, _Huffman] = {}
        self.ac_tables: Dict[int, _Huffman] = {}
        self.scan: Optional[List[Tuple[int, int, int]]] = None
        self.prefix = b""
        self.entropy = b""
        self.adobe_transform: Optional[int] = None
        self.has_restart = False


def _parse(data: bytes) -> _Jpeg:
    j = _Jpeg()
    if not data.startswith(b"\xff\xd8"):
        raise _Unsupported("not a JPEG")
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            raise _Unsupported("expected marker")
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            raise _Unsupported("truncated marker")
        marker = data[i]
        i += 1
        if marker == 0xD9:  # EOI
            break
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            raise _Unsupported("truncated segment")
        length = (data[i] << 8) | data[i + 1]
        if length < 2 or i + length > n:
            raise _Unsupported("bad segment length")
        payload = data[i + 2 : i + length]
        if marker == 0xDB:  # DQT
            p = 0
            while p < len(payload):
                pq = payload[p] >> 4
                tq = payload[p] & 0x0F
                p += 1
                if pq == 0:
                    qtable = np.frombuffer(payload[p : p + 64], dtype=np.uint8)
                    p += 64
                else:
                    qtable = np.frombuffer(
                        payload[p : p + 128], dtype=">u2"
                    ).astype(np.int32)
                    p += 128
                if qtable.size != 64:
                    raise _Unsupported("bad DQT")
                j.qtables[tq] = qtable.astype(np.int32)
        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(payload):
                tc = payload[p] >> 4
                th = payload[p] & 0x0F
                p += 1
                counts = list(payload[p : p + 16])
                p += 16
                total = sum(counts)
                symbols = list(payload[p : p + total])
                p += total
                table = _Huffman(counts, symbols)
                if tc == 0:
                    j.dc_tables[th] = table
                else:
                    j.ac_tables[th] = table
        elif marker in (0xC0, 0xC1):  # baseline / extended sequential
            j.precision = payload[0]
            j.height = (payload[1] << 8) | payload[2]
            j.width = (payload[3] << 8) | payload[4]
            ncomp = payload[5]
            if j.precision != 8:
                raise _Unsupported("only 8-bit JPEG")
            p = 6
            for _ in range(ncomp):
                cid = payload[p]
                hv = payload[p + 1]
                tq = payload[p + 2]
                p += 3
                j.components.append(_Component(cid, hv >> 4, hv & 0x0F, tq))
        elif marker in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise _Unsupported(f"unsupported SOF marker 0x{marker:02x}")
        elif marker == 0xDD:  # DRI
            j.has_restart = True
        elif marker == 0xEE:  # APP14
            if payload.startswith(b"Adobe") and len(payload) >= 12:
                j.adobe_transform = payload[11]
        elif marker == 0xDA:  # SOS
            ncomp = payload[0]
            p = 1
            scan = []
            for _ in range(ncomp):
                cid = payload[p]
                td = payload[p + 1] >> 4
                ta = payload[p + 1] & 0x0F
                p += 2
                scan.append((cid, td, ta))
            if payload[p] != 0 or payload[p + 1] != 63 or payload[p + 2] != 0:
                raise _Unsupported("not a baseline sequential scan")
            j.scan = scan
            entropy_start = i + length
            j.prefix = data[:entropy_start]
            # 找到 EOI / 下一个 marker（baseline 只有一个 scan）
            q = entropy_start
            while q + 1 < n:
                if data[q] != 0xFF:
                    q += 1
                    continue
                nxt = data[q + 1]
                if nxt == 0x00:
                    q += 2
                    continue
                if 0xD0 <= nxt <= 0xD7:
                    raise _Unsupported("restart markers not supported")
                if nxt == 0xD9:
                    j.entropy = data[entropy_start:q]
                    i = n
                    break
                raise _Unsupported("multiple scans not supported")
            else:
                raise _Unsupported("missing EOI")
            break
        i += length

    if j.scan is None or not j.components or not j.qtables:
        raise _Unsupported("incomplete JPEG")
    if j.has_restart:
        raise _Unsupported("restart interval not supported")
    if len(j.components) == 4:
        raise _Unsupported("CMYK JPEG not supported")
    if j.adobe_transform not in (None, 1):
        raise _Unsupported("Adobe color transform not supported")
    return j


def _unstuff(data: bytes) -> bytes:
    if b"\xff" not in data:
        return data
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0xFF:
            if i + 1 >= n or data[i + 1] != 0x00:
                raise _Unsupported("unexpected marker in entropy data")
            out.append(0xFF)
            i += 2
        else:
            out.append(b)
            i += 1
    return bytes(out)


def _decode_block(
    reader: _BitReader, dc: _Huffman, ac: _Huffman, pred: int
) -> Tuple[List[int], int]:
    coeff = [0] * 64
    size = dc.decode(reader)
    diff = reader.extend(size)
    pred += diff
    coeff[0] = pred
    k = 1
    while k < 64:
        rs = ac.decode(reader)
        run = rs >> 4
        size = rs & 0x0F
        if size == 0:
            if run == 15:
                k += 16
                continue
            break
        k += run
        if k >= 64:
            raise _Unsupported("bad AC run")
        # 系数按 zigzag 顺序保存，和 _quantize 的输出、熵编码循环保持一致
        coeff[k] = reader.extend(size)
        k += 1
    return coeff, pred


def _decode_coefficients(j: _Jpeg) -> Dict[str, Any]:
    scan = j.scan
    if scan is None:
        raise _Unsupported("missing scan")
    reader = _BitReader(_unstuff(j.entropy))
    hmax = max(c.h for c in j.components)
    vmax = max(c.v for c in j.components)
    mcu_cols = (j.width + 8 * hmax - 1) // (8 * hmax)
    mcu_rows = (j.height + 8 * vmax - 1) // (8 * vmax)
    comps = {c.id: c for c in j.components}
    coeffs = {}
    for c in j.components:
        cb_cols = mcu_cols * c.h
        cb_rows = mcu_rows * c.v
        coeffs[c.id] = np.zeros((cb_rows, cb_cols, 64), dtype=np.int32)
    preds = {c.id: 0 for c in j.components}
    for my in range(mcu_rows):
        for mx in range(mcu_cols):
            for cid, td, ta in scan:
                c = comps[cid]
                dc = j.dc_tables[td]
                ac = j.ac_tables[ta]
                for vy in range(c.v):
                    for hx in range(c.h):
                        block, pred = _decode_block(reader, dc, ac, preds[cid])
                        preds[cid] = pred
                        by = my * c.v + vy
                        bx = mx * c.h + hx
                        coeffs[cid][by, bx] = block
    return {
        "coeffs": coeffs,
        "mcu_cols": mcu_cols,
        "mcu_rows": mcu_rows,
        "hmax": hmax,
        "vmax": vmax,
    }


def _rgb_to_ycbcr(rgb: np.ndarray) -> List[np.ndarray]:
    arr = rgb.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return [y, cb, cr]


def _downsample(
    plane: np.ndarray, sx: int, sy: int, out_w: int, out_h: int
) -> np.ndarray:
    h, w = plane.shape[:2]
    out: np.ndarray
    if sx == 1 and sy == 1:
        out = np.empty((out_h, out_w), dtype=np.float32)
        out[: min(h, out_h), : min(w, out_w)] = plane[: min(h, out_h), : min(w, out_w)]
        if out_h > h or out_w > w:
            out = np.pad(
                out,
                ((0, max(0, out_h - h)), (0, max(0, out_w - w))),
                mode="edge",
            )
        return out
    ph = ((h + sy - 1) // sy) * sy
    pw = ((w + sx - 1) // sx) * sx
    padded = np.pad(plane, ((0, ph - h), (0, pw - w)), mode="edge")
    out = padded.reshape(ph // sy, sy, pw // sx, sx).mean(axis=(1, 3))
    out = out.astype(np.float32)
    if out.shape[0] < out_h or out.shape[1] < out_w:
        out = np.pad(
            out,
            ((0, max(0, out_h - out.shape[0])), (0, max(0, out_w - out.shape[1]))),
            mode="edge",
        )
    return out[:out_h, :out_w]


def _quantize(block: np.ndarray, qtable_zigzag: np.ndarray) -> np.ndarray:
    # qtable_zigzag 是 DQT 里的 zigzag 顺序；换成自然顺序后与 DCT 输出对齐
    q_flat = np.empty(64, dtype=np.float32)
    q_flat[ZIGZAG] = qtable_zigzag.astype(np.float32)
    q_nat = q_flat.reshape(8, 8)
    shifted = block.astype(np.float64) - 128.0
    coef = _DCT @ shifted @ _DCT.T
    quant = np.round(coef / np.maximum(q_nat, 1.0)).astype(np.int32)
    return quant.ravel()[ZIGZAG].astype(np.int32)


def patch_jpeg(
    source: bytes,
    canonical_rgb: np.ndarray,
    changed_blocks: np.ndarray,
) -> Optional[bytes]:
    """把 canonical 像素的改动块写回源 JPEG 的 DCT 系数。

    Returns None when the source is not supported by this module (progressive,
    restart markers, CMYK, size mismatch...). Callers must fall back to a full
    re-encode.
    """
    try:
        j = _parse(source)
    except _Unsupported:
        return None
    if (
        canonical_rgb.ndim != 3
        or canonical_rgb.shape[2] < 3
        or canonical_rgb.shape[0] != j.height
        or canonical_rgb.shape[1] != j.width
    ):
        return None

    decoded = _decode_coefficients(j)
    coeffs = decoded["coeffs"]
    hmax, vmax = decoded["hmax"], decoded["vmax"]
    mcu_cols, mcu_rows = decoded["mcu_cols"], decoded["mcu_rows"]
    rgb = canonical_rgb[:, :, :3]
    ycc = _rgb_to_ycbcr(rgb)
    if changed_blocks.ndim < 2:
        return None
    bh, bw = changed_blocks.shape[:2]

    for ci, c in enumerate(j.components):
        sx = hmax // c.h
        sy = vmax // c.v
        cb_cols = mcu_cols * c.h
        cb_rows = mcu_rows * c.v
        plane = _downsample(ycc[ci], sx, sy, cb_cols * 8, cb_rows * 8)
        q = j.qtables.get(c.tq)
        if q is None:
            return None
        # 把 luma 改动块池化到该分量的块网格：只有受影响的块才重新 DCT。
        ph = ((bh + sy - 1) // sy) * sy
        pw = ((bw + sx - 1) // sx) * sx
        padded = (
            changed_blocks
            if (ph == bh and pw == bw)
            else np.pad(changed_blocks, ((0, ph - bh), (0, pw - bw)), mode="constant")
        )
        comp_changed = padded.reshape(ph // sy, sy, pw // sx, sx).any(axis=(1, 3))
        for by, bx in np.argwhere(comp_changed[:cb_rows, :cb_cols]):
            block = plane[by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8]
            if block.shape != (8, 8):
                pad = np.pad(
                    block,
                    ((0, 8 - block.shape[0]), (0, 8 - block.shape[1])),
                    mode="edge",
                )
                block = pad
            coeffs[c.id][by, bx] = _quantize(block, q)

    # 重新熵编码整个 scan（未改动块的系数原样保留）
    scan = j.scan
    if scan is None:
        return None
    writer = _BitWriter()
    comps = {c.id: c for c in j.components}
    preds = {c.id: 0 for c in j.components}
    for my in range(mcu_rows):
        for mx in range(mcu_cols):
            for cid, td, ta in scan:
                c = comps[cid]
                dc = j.dc_tables[td]
                ac = j.ac_tables[ta]
                for vy in range(c.v):
                    for hx in range(c.h):
                        by = my * c.v + vy
                        bx = mx * c.h + hx
                        coeff = coeffs[cid][by, bx]
                        diff = int(coeff[0]) - preds[cid]
                        preds[cid] = int(coeff[0])
                        _write_value(writer, dc, diff)
                        run = 0
                        for k in range(1, 64):
                            value = int(coeff[k])
                            if value == 0:
                                run += 1
                                continue
                            while run > 15:
                                _write_symbol(writer, ac, 0xF0)
                                run -= 16
                            size = abs(value).bit_length()
                            _write_symbol(writer, ac, (run << 4) | size)
                            writer.write(_amplitude(value, size), size)
                            run = 0
                        if run:
                            _write_symbol(writer, ac, 0x00)
    entropy = writer.flush()
    return j.prefix + entropy + b"\xff\xd9"


def _write_symbol(writer: _BitWriter, table: _Huffman, symbol: int) -> None:
    code, length = table.encode_map[symbol]
    writer.write(code, length)


def _amplitude(value: int, size: int) -> int:
    if value < 0:
        return value + (1 << size) - 1
    return value


def _write_value(writer: _BitWriter, table: _Huffman, value: int) -> None:
    size = abs(value).bit_length()
    _write_symbol(writer, table, size)
    if size:
        writer.write(_amplitude(value, size), size)
