"""动态上下文扩展：用便宜指标决定每个掩膜裁块要带多少上下文。

`HDStrategy.Crop` 的边距固定 128：平坦背景上的水印多给上下文没坏处，复杂纹理上
却可能不够，而大掩膜又会被这 128 拖成整图前向。这里按 solution 规格分两层决策：

1. 几何规则（零成本）：mask 面积占比、bbox 到图像四边的距离 —— 给出基础 pad。
2. ring 复杂度（只在缩略 patch 上算）：局部熵、方向一致性、块匹配最小距离 ——
   复杂纹理才额外扩大，平坦区和高规则纹理保持 base_pad。

决策完全确定：同一 image/mask/配置走同一裁剪、同一 padding。所有参考值和阈值都
来自 `DynamicContextConfig`，本模块不含随机数，也不读当前时间。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from iopaint.helper import crop_rect
from iopaint.schema import DynamicContextConfig


@dataclass(frozen=True)
class RingMetrics:
    """ring 上的复杂度指标原始值。"""

    entropy_mean: float
    entropy_p75: float
    entropy_blocks: int
    coherence: float
    grad_mag: float
    match_dist: float
    match_queries: int
    interaction: float
    ring_pixels: int


@dataclass(frozen=True)
class ContextDecision:
    """一个掩膜轮廓的完整决策记录（可直接写进决策日志）。"""

    bbox: Tuple[int, int, int, int]
    rect: Tuple[int, int, int, int]
    pad: int
    geometry_pad: int
    dynamic_pad: int
    complexity: float
    area_ratio: float
    edge_distance: int
    metrics: Optional[RingMetrics] = None
    normalized: Dict[str, float] = field(default_factory=dict)
    flags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "bbox": list(self.bbox),
            "rect": list(self.rect),
            "pad": self.pad,
            "geometry_pad": self.geometry_pad,
            "dynamic_pad": self.dynamic_pad,
            "complexity": round(self.complexity, 4),
            "area_ratio": round(self.area_ratio, 6),
            "edge_distance": self.edge_distance,
            "normalized": {k: round(v, 4) for k, v in self.normalized.items()},
            "flags": list(self.flags),
        }
        if self.metrics is not None:
            data["metrics"] = {
                "entropy_mean": round(self.metrics.entropy_mean, 4),
                "entropy_p75": round(self.metrics.entropy_p75, 4),
                "entropy_blocks": self.metrics.entropy_blocks,
                "coherence": round(self.metrics.coherence, 4),
                "grad_mag": round(self.metrics.grad_mag, 2),
                "match_dist": round(self.metrics.match_dist, 6),
                "match_queries": self.metrics.match_queries,
                "interaction": round(self.metrics.interaction, 4),
                "ring_pixels": self.metrics.ring_pixels,
            }
        return data


def _clamp(value: int, cfg: DynamicContextConfig) -> int:
    return max(cfg.min_pad, min(cfg.max_pad, int(value)))


def mask_area_ratio(mask: np.ndarray) -> float:
    """掩膜像素数 / 原图像素数。"""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask >= 127)) / float(mask.shape[0] * mask.shape[1])


def edge_distance(box: Tuple[int, int, int, int], shape: Tuple[int, ...]) -> int:
    """掩膜 bbox 到图像四边的最小距离。"""
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    return int(min(x1, y1, width - x2, height - y2))


def geometry_pad(area_ratio: float, cfg: DynamicContextConfig) -> int:
    """只用零成本几何信息给基础 padding（spec 第 4 节）。"""
    pad = cfg.base_pad
    if area_ratio > cfg.area_ratio_mid:
        pad = int(pad * cfg.area_pad_scale + 0.5)
    if area_ratio > cfg.area_ratio_high:
        pad = max(pad, cfg.area_high_pad)
    return pad


def _thumbnail(
    image: np.ndarray, mask: np.ndarray, thumb_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """把局部 patch 等比缩到 thumb_size 以内；不放大，保证小掩膜不被插值糊掉。"""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= thumb_size:
        return image, mask
    scale = thumb_size / longest
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _ring_from_mask(mask: np.ndarray, ring_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (binary_mask, ring)：ring = 以 ring_width 为半径膨胀的 mask 减原 mask。"""
    binary = (mask >= 127).astype(np.uint8)
    if not binary.any() or binary.all():
        return binary, np.zeros_like(binary, dtype=bool)
    radius = max(1, int(ring_width))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(binary, kernel)
    ring = (dilated > 0) & (binary == 0)
    return binary, ring


def _ring_entropy(
    gray: np.ndarray, ring: np.ndarray, block: int, min_pixels: int
) -> Tuple[float, float, int]:
    """ring 上分块灰度直方图熵，返回 (均值, 75 分位, 有效块数)。"""
    height, width = gray.shape
    values: List[float] = []
    for y in range(0, height, block):
        for x in range(0, width, block):
            block_ring = ring[y : y + block, x : x + block]
            count = int(block_ring.sum())
            if count < min_pixels:
                continue
            pixels = gray[y : y + block, x : x + block][block_ring]
            hist = np.bincount(pixels.astype(np.int64), minlength=256).astype(float)
            prob = hist[hist > 0] / hist.sum()
            values.append(float(-(prob * np.log2(prob)).sum()))
    if not values:
        return 0.0, 0.0, 0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 75)), int(arr.size)


def _structure_tensor(
    gray: np.ndarray, ring: np.ndarray, window: int
) -> Tuple[float, float]:
    """ring 上的平均方向一致性 coherence 与平均梯度幅值。

    结构张量 J = [[Jxx, Jxy], [Jxy, Jyy]]，λ1≥λ2，
    coherence = (λ1-λ2)/(λ1+λ2)：条纹/边缘接近 1，杂乱纹理接近 0。
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ksize = (window, window)
    jxx = cv2.GaussianBlur(gx * gx, ksize, 0)
    jyy = cv2.GaussianBlur(gy * gy, ksize, 0)
    jxy = cv2.GaussianBlur(gx * gy, ksize, 0)

    trace = jxx + jyy
    diff = jxx - jyy
    root = np.sqrt(diff * diff + 4.0 * jxy * jxy)
    l1 = 0.5 * (trace + root)
    l2 = 0.5 * (trace - root)
    coherence_map = (l1 - l2) / (l1 + l2 + 1e-6)
    magnitude = np.sqrt(gx * gx + gy * gy)

    mask = ring
    if not mask.any():
        return 0.0, 0.0
    coherence = float(coherence_map[mask].mean())
    grad_mag = float(magnitude[mask].mean())
    return coherence, grad_mag


def _block_candidates(
    ring: np.ndarray, block: int, min_pixels: int
) -> List[Tuple[int, int]]:
    """ring 覆盖足够多的块（整块在 patch 内），按 (y, x) 排序保证确定性。"""
    height, width = ring.shape
    need = max(min_pixels, (block * block) // 2)
    candidates: List[Tuple[int, int]] = []
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            if int(ring[y : y + block, x : x + block].sum()) >= need:
                candidates.append((y, x))
    return candidates


def _boundary_queries(
    mask: np.ndarray, ring: np.ndarray, block: int, max_queries: int
) -> List[Tuple[int, int]]:
    """取贴着 mask 边界的 ring 块作为查询块；超出上限时等距抽样（确定性）。"""
    binary = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = (cv2.dilate(binary, kernel) > 0) & (binary == 0)

    height, width = ring.shape
    queries: List[Tuple[int, int]] = []
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            if not boundary[y : y + block, x : x + block].any():
                continue
            if ring[y : y + block, x : x + block].sum() == 0:
                continue
            queries.append((y, x))
    if len(queries) > max_queries:
        step = len(queries) / float(max_queries)
        queries = [queries[int(i * step)] for i in range(max_queries)]
    return queries


def _match_distance(
    gray: np.ndarray,
    mask: np.ndarray,
    ring: np.ndarray,
    cfg: DynamicContextConfig,
) -> Tuple[float, int]:
    """mask 边界附近查询块到 ring 内最相似块的归一化 SSD 中位数。"""
    block = cfg.block_size
    candidates = _block_candidates(ring, block, cfg.min_block_pixels)
    if len(candidates) < 2:
        return 0.0, 0

    queries = _boundary_queries(mask, ring, block, cfg.max_queries)
    if not queries:
        return 0.0, 0

    cand_pixels = np.stack(
        [gray[y : y + block, x : x + block].astype(np.float32) for y, x in candidates]
    ).reshape(len(candidates), -1)
    cand_ring = np.stack(
        [ring[y : y + block, x : x + block] for y, x in candidates]
    ).reshape(len(candidates), -1).astype(np.float32)

    distances: List[float] = []
    for qy, qx in queries:
        query = gray[qy : qy + block, qx : qx + block].astype(np.float32).reshape(-1)
        query_ring = ring[qy : qy + block, qx : qx + block].reshape(-1)
        both = cand_ring * query_ring[np.newaxis, :]
        counts = both.sum(axis=1)
        # 排除自身（同一位置）与重叠过多的邻居块
        center = np.asarray(candidates, dtype=np.float64)
        too_close = (np.abs(center[:, 0] - qy) < block) & (
            np.abs(center[:, 1] - qx) < block
        )
        valid = (counts >= cfg.min_block_pixels) & (~too_close)
        if not valid.any():
            continue
        squared = (cand_pixels - query[np.newaxis, :]) ** 2
        denom = np.maximum(counts, 1.0) * 255.0 * 255.0
        ssd = (squared * both).sum(axis=1) / denom
        distances.append(float(ssd[valid].min()))

    if not distances:
        return 0.0, 0
    return float(np.median(np.asarray(distances))), len(distances)


def ring_metrics(
    image: np.ndarray, mask: np.ndarray, cfg: DynamicContextConfig
) -> RingMetrics:
    """在局部 patch（image RGB + 同尺寸 mask）上算全部 ring 指标。

    调用方负责把 patch 限制在 bbox 外扩 2*ring_width 的范围内；这里再按
    thumb_size 缩放，保证指标开销与掩膜大小无关。
    """
    image, mask = _thumbnail(image, mask, cfg.thumb_size)
    binary, ring = _ring_from_mask(mask, cfg.ring_width)
    ring_pixels = int(ring.sum())
    if ring_pixels == 0:
        return RingMetrics(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    entropy_mean, entropy_p75, blocks = _ring_entropy(
        gray, ring, cfg.block_size, cfg.min_block_pixels
    )
    coherence, grad_mag = _structure_tensor(gray, ring, cfg.structure_window)
    match_dist, queries = _match_distance(gray, binary, ring, cfg)

    grad_n = min(1.0, grad_mag / cfg.grad_ref)
    interaction = (1.0 - coherence) * grad_n
    return RingMetrics(
        entropy_mean=entropy_mean,
        entropy_p75=entropy_p75,
        entropy_blocks=blocks,
        coherence=coherence,
        grad_mag=grad_mag,
        match_dist=match_dist,
        match_queries=queries,
        interaction=interaction,
        ring_pixels=ring_pixels,
    )


def normalize_metrics(
    metrics: RingMetrics, cfg: DynamicContextConfig
) -> Dict[str, float]:
    """各项指标的归一化值（写进日志，便于标定阈值）。"""
    return {
        "entropy": min(1.0, metrics.entropy_mean / cfg.entropy_ref),
        "match": min(1.0, metrics.match_dist / cfg.match_dist_ref),
        "grad": min(1.0, metrics.grad_mag / cfg.grad_ref),
        "interaction": metrics.interaction,
    }


def complexity_score(metrics: RingMetrics, cfg: DynamicContextConfig) -> float:
    """加权和复杂度，钳制到 0..1。"""
    norm = normalize_metrics(metrics, cfg)
    value = (
        cfg.w_entropy * norm["entropy"]
        + cfg.w_match * norm["match"]
        + cfg.w_interaction * norm["interaction"]
    )
    return float(min(1.0, max(0.0, value)))


def dynamic_pad(complexity: float, cfg: DynamicContextConfig) -> int:
    """复杂度 → padding（spec 第 7 节）。"""
    if complexity < cfg.complexity_low:
        return cfg.base_pad
    if complexity > cfg.complexity_high:
        return max(cfg.base_pad, cfg.high_pad)
    return int(cfg.base_pad * (1.0 + cfg.alpha * complexity) + 0.5)


def _local_patch(
    image: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int], margin: int
) -> Tuple[np.ndarray, np.ndarray]:
    """bbox 外扩 margin 的局部 patch（clamp 到图内）。"""
    height, width = mask.shape[:2]
    x1 = max(0, box[0] - margin)
    y1 = max(0, box[1] - margin)
    x2 = min(width, box[2] + margin)
    y2 = min(height, box[3] + margin)
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def decide_context(
    image: np.ndarray,
    mask: np.ndarray,
    box: Union[np.ndarray, Tuple[int, int, int, int]],
    cfg: DynamicContextConfig,
) -> ContextDecision:
    """一个掩膜轮廓的最终裁剪窗口与决策记录。"""
    height, width = mask.shape[:2]
    bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    area_ratio = mask_area_ratio(mask)
    geom = geometry_pad(area_ratio, cfg)

    flags: List[str] = []
    metrics: Optional[RingMetrics] = None
    complexity = 0.0
    dyn = geom

    if cfg.force_pad is not None:
        pad = cfg.force_pad
        flags.append("forced_pad")
    else:
        if cfg.enable_complexity and geom < cfg.skip_complexity_at_pad:
            patch, patch_mask = _local_patch(
                image, mask, bbox, 2 * int(cfg.ring_width)
            )
            metrics = ring_metrics(patch, patch_mask, cfg)
            if metrics.ring_pixels == 0:
                flags.append("ring_empty")
            else:
                complexity = complexity_score(metrics, cfg)
                dyn = dynamic_pad(complexity, cfg)
        else:
            flags.append("complexity_skipped")
        pad = max(geom, dyn)
        if pad > cfg.max_pad:
            flags.append("pad_capped")

    pad = _clamp(pad, cfg)
    rect = crop_rect(np.asarray(bbox), pad, mask.shape)

    margins = (bbox[0] - rect[0], bbox[1] - rect[1], rect[2] - bbox[2], rect[3] - bbox[3])
    if min(margins) < pad:
        flags.append("edge_clamped")
    if max(rect[2] - rect[0], rect[3] - rect[1]) > cfg.max_input_side:
        flags.append("downscaled")

    normalized: Dict[str, float] = {}
    if metrics is not None:
        normalized = normalize_metrics(metrics, cfg)

    return ContextDecision(
        bbox=bbox,
        rect=rect,
        pad=pad,
        geometry_pad=geom,
        dynamic_pad=dyn,
        complexity=complexity,
        area_ratio=area_ratio,
        edge_distance=edge_distance(bbox, (height, width)),
        metrics=metrics,
        normalized=normalized,
        flags=tuple(flags),
    )


def seam_score(
    result_bgr: np.ndarray, mask: np.ndarray, cfg: DynamicContextConfig
) -> Tuple[float, float, float]:
    """接缝分数 = 边界内侧窄带平均梯度 / ring 平均梯度。

    输入是裁剪窗口内的结果（BGR）与同尺寸 mask。任一侧没有像素时返回 0
    （不触发重跑）。返回 (score, band_grad, ring_grad)。
    """
    if result_bgr.ndim == 3:
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = result_bgr
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)

    binary = (mask >= 127).astype(np.uint8)
    if not binary.any():
        return 0.0, 0.0, 0.0

    band_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * cfg.seam_band + 1, 2 * cfg.seam_band + 1)
    )
    ring_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * cfg.ring_width + 1, 2 * cfg.ring_width + 1)
    )
    band = (binary - cv2.erode(binary, band_k)) > 0
    ring = ((cv2.dilate(binary, ring_k) > 0) & ~(cv2.dilate(binary, band_k) > 0))
    if not band.any() or not ring.any():
        return 0.0, 0.0, 0.0

    band_grad = float(magnitude[band].mean())
    ring_grad = float(magnitude[ring].mean())
    if ring_grad <= 1e-6:
        return 0.0, band_grad, ring_grad
    return band_grad / ring_grad, band_grad, ring_grad


def seam_rerun_pad(pad: int, cfg: DynamicContextConfig) -> int:
    """接缝兜底重跑用的 padding：当前 pad 的 1.5 倍，受 max_pad 限制。"""
    return _clamp(int(pad * 1.5 + 0.5), cfg)
