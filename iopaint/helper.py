import base64
import io
import ipaddress
import os
import struct
import sys
import warnings
import zlib
from typing import Any, List, Optional, Dict, Tuple

from urllib.parse import urlparse
import cv2
from PIL import Image, ImageOps, PngImagePlugin, JpegImagePlugin
import numpy as np
import torch
from iopaint.const import MPS_UNSUPPORT_MODELS
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib

from iopaint.exceptions import ModelLoadError
from iopaint.i18n import cli_text
from iopaint.runtime import device_str


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def switch_mps_device(model_name, device):
    if model_name in MPS_UNSUPPORT_MODELS and device_str(device) == "mps":
        logger.info(f"{model_name} not support mps, switch to cpu")
        return torch.device("cpu")
    return device


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_md5: str = None):
    if os.path.exists(url):
        cached_file = url
    else:
        cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logger.error(
                        cli_text(
                            "cli.error.md5_mismatch_deleted",
                            md5=_md5,
                            expected=model_md5,
                        )
                    )
                except OSError:
                    logger.error(
                        cli_text(
                            "cli.error.md5_mismatch_delete_hint",
                            md5=_md5,
                            expected=model_md5,
                            path=cached_file,
                        )
                    )
                raise ModelLoadError(
                    cli_text("cli.error.md5_mismatch", path=cached_file)
                )

    return cached_file


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def handle_error(model_path, model_md5, e):
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logger.error(
                cli_text(
                    "cli.error.md5_mismatch_deleted",
                    md5=_md5,
                    expected=model_md5,
                )
            )
        except OSError:
            logger.error(
                cli_text(
                    "cli.error.md5_mismatch_delete_hint",
                    md5=_md5,
                    expected=model_md5,
                    path=model_path,
                )
            )
    else:
        logger.error(
            cli_text("cli.error.load_failed_issue", path=model_path, error=e)
        )
    raise ModelLoadError(cli_text("cli.error.load_failed", path=model_path, error=e))


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logger.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def load_model(model: torch.nn.Module, url_or_path, device, model_md5):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    try:
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def numpy_to_bytes(
    image_numpy: np.ndarray, ext: str, quality: int = 95, png_compression: int = 6
) -> bytes:
    """Encode a numpy image with format-appropriate encoder params.

    原实现把 JPEG 和 PNG 的参数混在同一个列表里传给 cv2.imencode：
    `[JPEG_QUALITY, 100, PNG_COMPRESSION, 0]`。对 PNG 而言 compression=0 让
    12MP RGBA mask 编出 48.08MB（实测），而 mask 这种 0/255 平坦图在
    compression=6 下只有 0.05MB，耗时 418ms → 572ms（+154ms 换 960 倍体积）。
    对 JPEG 而言 quality 硬编码 100，与 `--quality` 默认 95 不一致。
    """
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    else:
        params = []
    data = cv2.imencode(f".{ext}", image_numpy, params)[1]
    image_bytes = data.tobytes()
    return image_bytes


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_ROWS_PER_BLOCK = 256
_PNG_IDAT_BLOCK = 1 << 20


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _png_metadata_chunks(infos: Dict) -> bytes:
    """iCCP / text / pHYs / eXIf，按 Pillow 的写出顺序与字节格式。

    只覆盖 `pil_to_bytes` 原来会写的那几项（exif、icc_profile、dpi、
    parameters）；`comment` 在 Pillow 的 PNG 编码器里本来就被忽略。
    """
    chunks = []
    icc = infos.get("icc_profile")
    if icc:
        chunks.append(_png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc)))
    parameters = infos.get("parameters")
    if parameters is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", parameters)
        for cid, data in (c[:2] for c in pnginfo.chunks):
            chunks.append(_png_chunk(cid, data))
    dpi = infos.get("dpi")
    if dpi:
        chunks.append(
            _png_chunk(
                b"pHYs",
                struct.pack(
                    ">IIB",
                    int(dpi[0] / 0.0254 + 0.5),
                    int(dpi[1] / 0.0254 + 0.5),
                    1,
                ),
            )
        )
    exif = infos.get("exif")
    if exif:
        if isinstance(exif, Image.Exif):
            exif = exif.tobytes(8)
        if exif.startswith(b"Exif\x00\x00"):
            exif = exif[6:]
        chunks.append(_png_chunk(b"eXIf", exif))
    return b"".join(chunks)


def png_bytes_from_numpy(
    arr: np.ndarray, compress_level: int = 1, infos: Optional[Dict] = None
) -> bytes:
    """RGB/RGBA uint8 → PNG 字节，固定 Up 滤波。

    Pillow 的 PNG 编码器逐行在 5 种滤波里挑代价最小的一种，这部分占了大图
    编码时间的大头。固定用 Up（每一行减去上一行）后，zlib 的输入几乎一样
    好压：三张真实照片（24.0/24.4/45.4MP）与 Pillow level 1 相比体积差
    -3.1%~+1.4%，编码耗时从 1.90/2.74/4.08s 降到 1.08/1.82/2.27s（1.5~1.8×）。

    只有 RGB/RGBA 走这里；其它模式（P/L/I/16 位）由 `pil_to_bytes` 回退
    Pillow，避免为冷门模式重写位深与调色板逻辑。
    """
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported array for png: {arr.shape} {arr.dtype}")
    level = max(-1, min(9, int(compress_level)))
    h, w, channels = arr.shape
    rowbytes = w * channels
    rows = np.ascontiguousarray(arr).reshape(h, rowbytes)

    compressor = zlib.compressobj(
        level, zlib.DEFLATED, zlib.MAX_WBITS, 9, zlib.Z_DEFAULT_STRATEGY
    )
    idat = bytearray()
    # 一块 256 行：既让滤波的临时数组保持在几 MB，又给 zlib 足够长的连续输入
    block = np.empty((min(_PNG_ROWS_PER_BLOCK, h), rowbytes + 1), dtype=np.uint8)
    for y0 in range(0, h, _PNG_ROWS_PER_BLOCK):
        y1 = min(y0 + _PNG_ROWS_PER_BLOCK, h)
        n = y1 - y0
        pixels = rows[y0:y1]
        prior = np.empty_like(pixels)
        if y0 == 0:
            prior[0] = 0
        else:
            prior[0] = rows[y0 - 1]
        prior[1:] = pixels[:-1]
        out = block[:n]
        out[:, 0] = 2  # filter type: Up
        np.subtract(pixels, prior, out=out[:, 1:], dtype=np.uint8)
        idat += compressor.compress(out)
    idat += compressor.flush()

    color_type = 2 if channels == 3 else 6
    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    parts = [
        _PNG_SIGNATURE,
        _png_chunk(b"IHDR", header),
        _png_metadata_chunks(infos or {}),
    ]
    for i in range(0, len(idat), _PNG_IDAT_BLOCK):
        parts.append(_png_chunk(b"IDAT", bytes(idat[i : i + _PNG_IDAT_BLOCK])))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


# save() kwargs that are safe to pass through from input image metadata.
# `parameters` is handled separately (embedded as a pnginfo text chunk).
_PIL_SAVE_WHITELIST = ("exif", "icc_profile", "dpi", "comment")


def pil_to_bytes(
    pil_img,
    ext: str,
    quality: int = 95,
    infos=None,
    png_compress_level: int = 1,
    subsampling: Optional[int] = None,
    webp_method: int = 4,
    webp_lossless: bool = True,
) -> bytes:
    """Encode a PIL image to bytes.

    `png_compress_level` 默认 1：PIL PNG 编码是纯 CPU 的 zlib 压缩，4000x3000
    （12MP）实测 level0=506ms/36MB、level1=979ms/13.0MB、level6=6732ms/11.0MB、
    level9=35317ms——降级只损失 ~18% 体积，却省下数秒延迟。PNG 本身无损，
    降低压缩级别不影响像素，只影响响应延迟与字节数（实测 level1/6 均逐像素相等）。
    响应路径（api.py / batch）都走默认值；需要更小体积时显式传 6/9。

    `subsampling` 只对 JPEG 生效：Pillow 不显式传时落 libjpeg 默认 4:2:0。
    导出边界要显式传 0 才是 4:4:4；`None` 保持旧行为。

    RGB/RGBA 的 PNG 编码走 `png_bytes_from_numpy`（固定 Up 滤波），大约比
    Pillow 快一倍；其它模式仍交给 Pillow。

    `ext="webp"` 默认走 lossless（`method=4`）：无状态无损线格式体积比 PNG
    level1 小约 15%，但 12MP 编码慢 3~4 倍；默认不启用。导出边界需要按
    `quality` 有损编码时显式传 `webp_lossless=False`。
    """
    if infos is None:
        infos = {}
    if ext.lower() == "png" and pil_img.mode in ("RGB", "RGBA"):
        return png_bytes_from_numpy(
            np.asarray(pil_img), compress_level=png_compress_level, infos=infos
        )
    with io.BytesIO() as output:
        if ext == "jpg":
            ext = "jpeg"

        save_kwargs = {}
        is_jpeg = ext.lower() in ("jpg", "jpeg")
        if ext.lower() == "png":
            save_kwargs["compress_level"] = png_compress_level
        elif ext.lower() == "webp":
            save_kwargs["lossless"] = bool(webp_lossless)
            save_kwargs["method"] = int(webp_method)
        elif is_jpeg and subsampling is not None:
            save_kwargs["subsampling"] = int(subsampling)
        parameters = infos.get("parameters")
        if parameters is not None and ext.lower() == "png":
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", parameters)
            save_kwargs["pnginfo"] = pnginfo_data

        for key in _PIL_SAVE_WHITELIST:
            value = infos.get(key)
            if value is not None:
                save_kwargs[key] = value

        pil_img.save(output, format=ext, quality=quality, **save_kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def jpeg_sampling(image: Image.Image) -> Optional[int]:
    """读取源 JPEG 的色度抽样（0=4:4:4, 1=4:2:2, 2=4:2:0）。

    非 JPEG / Pillow 不支持时返回 None。必须在 convert()/copy() 之前调用。
    """
    try:
        if (image.format or "").upper() not in ("JPEG", "MPO"):
            return None
        return int(JpegImagePlugin.get_sampling(image))
    except Exception:
        return None


def jpeg_sampling_from_bytes(data: bytes) -> Optional[int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return jpeg_sampling(image)
    except Exception:
        return None



# 单张图片的像素上限，按请求来源区分（见 api.py:_max_pixels_for_request）：
# 外网来源沿用 1 亿像素上限；内网（回环 / RFC1918 私有网段 / 链路本地）来源
# 传 None，表示不限制。
#
# Pillow 的 `Image.MAX_IMAGE_PIXELS` 是进程级全局常量，无法按请求区分来源，
# 而解码跑在线程池里并发执行，逐请求改写全局会互相踩。因此 iopaint/__init__.py
# 把该全局置为 None，改由下面的 check_image_pixels 按 `max_pixels` 显式检查。
EXTERNAL_MAX_IMAGE_PIXELS = 100_000_000
PRIVATE_MAX_IMAGE_PIXELS: Optional[int] = None

# 视为内网的来源网段：回环 + RFC1918 私有网段 + 链路本地 + IPv6 ULA。
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def is_private_host(host: Optional[str]) -> bool:
    """判断请求来源是否属于内网。

    命中回环（127.0.0.0/8、::1）、RFC1918 私有网段（10/8、172.16/12、
    192.168/16）、链路本地（169.254/16、fe80::/10）以及 IPv6 ULA（fc00::/7）
    即视为内网；字面量 "localhost" 也算，双栈下的 IPv4-mapped IPv6
    （::ffff:192.168.x.x）先还原成 IPv4 再判断。这里刻意不直接用
    ipaddress.is_private——它还会把文档/保留网段（如 203.0.113.0/24）也算作
    private，那样会给非内网来源放开限制。无法解析的主机名（如测试客户端的
    "testclient"）与 None 一律按外网处理，宁可多限制。
    """
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # 双栈监听时对端可能以 IPv4-mapped IPv6（::ffff:192.168.x.x）出现
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip in network for network in _PRIVATE_NETWORKS)


def check_image_pixels(size: Tuple[int, int], max_pixels: Optional[int]) -> None:
    """复刻 Pillow 的 DecompressionBomb 两档语义，但上限可按请求来源指定。

    `max_pixels=None`（内网）不检查；超过 `max_pixels` 只告警，超过 2×上限
    视为解压炸弹抛 `DecompressionBombError`——与 Pillow 原生行为一致。
    """
    if max_pixels is None:
        return

    pixels = max(1, size[0]) * max(1, size[1])
    if pixels > 2 * max_pixels:
        raise Image.DecompressionBombError(
            f"Image size ({pixels} pixels) exceeds limit of {2 * max_pixels} "
            "pixels, could be decompression bomb DOS attack."
        )
    if pixels > max_pixels:
        warnings.warn(
            f"Image size ({pixels} pixels) exceeds limit of {max_pixels} pixels, "
            "could be decompression bomb DOS attack.",
            Image.DecompressionBombWarning,
        )


def load_img(
    img_bytes: bytes,
    gray: bool = False,
    return_info: bool = False,
    max_pixels: Optional[int] = EXTERNAL_MAX_IMAGE_PIXELS,
) -> Any:
    alpha_channel = None
    image = Image.open(io.BytesIO(img_bytes))
    check_image_pixels(image.size, max_pixels)

    if return_info:
        infos = image.info

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_info:
        return np_img, alpha_channel, infos
    return np_img, alpha_channel


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_AREA
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    if out_height == height and out_width == width:
        return img

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def rasterize_mask_strokes(
    shape: Tuple[int, ...], strokes: List[Any]
) -> np.ndarray:
    """把前端笔迹栅格化成 HxW 的 0/255 掩膜。

    前端原来按原图分辨率建画布再 toBlob 成 PNG，大图上这一步在 WebKit 要
    0.7~1.8s（12/30MP），而掩膜本身只有 0.2~0.8MB。笔迹本来就是矢量数据，
    直接在这里画到 `np.zeros((H, W), uint8)` 上，几毫秒即可。

    语义对齐 canvas 的 round cap / round join：`LINE_8` 折线负责线段，每个顶点
    再补一个实心圆。不要换成 `LINE_AA` —— 抗锯齿灰度再按 127 二值化会让边缘
    缩进约 0.5px，1px 笔刷会被吃掉。
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not strokes:
        return mask

    for stroke in strokes:
        pts = np.asarray(getattr(stroke, "pts", stroke), dtype=np.float64).reshape(
            -1, 2
        )
        if pts.size == 0:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
        int_pts = np.round(pts).astype(np.int32)
        thickness = max(1, int(round(float(getattr(stroke, "size", 1)))))
        if len(int_pts) >= 2:
            cv2.polylines(
                mask,
                [int_pts.reshape(-1, 1, 2)],
                isClosed=False,
                color=255,
                thickness=thickness,
                lineType=cv2.LINE_8,
            )
        # 圆头端点 + 圆角接点。半径取 thickness//2，单点笔迹就是这里画的圆点。
        radius = thickness // 2
        for x, y in int_pts:
            cv2.circle(
                mask,
                (int(x), int(y)),
                radius,
                255,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
    return mask


def only_keep_largest_contour(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w)  0~255

    Returns:

    """
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = i

    if max_index != -1:
        new_mask = np.zeros_like(mask)
        return cv2.drawContours(new_mask, contours, max_index, 255, -1)
    else:
        return mask


def is_mac():
    return sys.platform == "darwin"


def decode_base64_to_image(
    encoding: str,
    gray: bool = False,
    max_pixels: Optional[int] = EXTERNAL_MAX_IMAGE_PIXELS,
) -> Tuple[np.array, Optional[np.array], Dict, str]:
    if encoding.startswith("data:image/") or encoding.startswith(
        "data:application/octet-stream;base64,"
    ):
        encoding = encoding.split(";")[1].split(",")[1]
    image_bytes = base64.b64decode(encoding)
    image = Image.open(io.BytesIO(image_bytes))
    check_image_pixels(image.size, max_pixels)
    # image.format must be read before exif_transpose, which returns a new
    # image without the format attribute set.
    ext = (image.format or "jpeg").lower()

    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    # exif_transpose will remove exif rotate info，we must call image.info after exif_transpose
    infos = image.info

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    return np_img, alpha_channel, infos, ext


def encode_pil_to_base64(image: Image, quality: int, infos: Dict) -> bytes:
    img_bytes = pil_to_bytes(
        image,
        "png",
        quality=quality,
        infos=infos,
    )
    return base64.b64encode(img_bytes)


def concat_alpha_channel(rgb_np_img, alpha_channel) -> np.ndarray:
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(rgb_np_img.shape[1], rgb_np_img.shape[0])
            )
        rgb_np_img = np.concatenate(
            (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )
    return rgb_np_img


def adjust_mask(mask: np.ndarray, kernel_size: int, operate):
    # fronted brush color "ffcc00bb"
    # kernel_size = kernel_size*2+1
    mask[mask >= 127] = 255
    mask[mask < 127] = 0

    if operate == "reverse":
        mask = 255 - mask
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
        )
        if operate == "expand":
            mask = cv2.dilate(
                mask,
                kernel,
                iterations=1,
            )
        else:
            mask = cv2.erode(
                mask,
                kernel,
                iterations=1,
            )
    res_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    res_mask[mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask


def gen_frontend_mask(bgr_or_gray_mask):
    if len(bgr_or_gray_mask.shape) == 3 and bgr_or_gray_mask.shape[2] != 1:
        bgr_or_gray_mask = cv2.cvtColor(bgr_or_gray_mask, cv2.COLOR_BGR2GRAY)

    # fronted brush color "ffcc00bb"
    # TODO: how to set kernel size?
    kernel_size = 9
    bgr_or_gray_mask = cv2.dilate(
        bgr_or_gray_mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )
    res_mask = np.zeros(
        (bgr_or_gray_mask.shape[0], bgr_or_gray_mask.shape[1], 4), dtype=np.uint8
    )
    res_mask[bgr_or_gray_mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask
