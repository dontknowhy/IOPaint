"""多盒合并：互相重叠的掩膜裁块共用一次前向，并给合并面积设上限。

多笔掩膜（比如一段文字）本来每个轮廓单独前向，固定开销乘上笔画数；掩膜边距
128 起步，密集笔画各自的裁剪框本来就大面积重叠，并成一块既少跑几次、总面积也
更小。散开的笔画不合并：并集大于各块面积之和就是多算像素，实测 12MP 上 30 条
620px 间距的笔画合成一行宽裁剪反而慢 11%。合并会改变掩膜内的像素，所以默认
关闭（`hd_strategy_crop_merge=False`）。

显存方面不能只看距离：挨着的笔迹一路并下去可以覆盖整张图，LaMa 这类 fp32 模型
峰值约 1GB/百万像素，所以并集面积超过 `hd_strategy_crop_merge_max_pixels` 就
不合并。

这里守五件事：`crop_rect` 的几何与旧 `_crop_box` 逐像素一致；关闭时裁剪框逐个
不变；开启时只合并重叠的块、不超面积上限；掩膜外像素仍严格等于原图；定尺寸
瓦片路径同样受益。不加载权重。
"""
import numpy as np
import pytest

from iopaint.helper import (
    boxes_from_mask,
    crop_rect,
    merge_overlapping_rects,
)
from iopaint.model.base import InpaintModel
from iopaint.schema import HDStrategy, InpaintRequest


class StubFlat(InpaintModel):
    """普通擦除模型的假模型：forward 恒等回 BGR，只记录被送进网络的尺寸。"""

    name = "stub_flat"
    pad_mod = 8

    def __init__(self, device: str = "cpu", **kwargs: object) -> None:
        self.forward_shapes: list[tuple[int, ...]] = []
        super().__init__(device, **kwargs)

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def init_model(self, device: str, **kwargs: object) -> None:
        pass

    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        self.forward_shapes.append(tuple(image.shape[:2]))
        return image[:, :, ::-1].copy()


class StubTile(StubFlat):
    """定尺寸瓦片的假模型（migan/fcf 那一档）。"""

    name = "stub_tile"
    min_size = 512
    pad_mod = 512
    pad_to_square = True
    fixed_size = 512


def make_case(
    height: int, width: int, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(height * width)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.uint8)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = 255
    return image, mask


def cfg(
    merge: bool = False, max_pixels: int = 4 * 1024 * 1024, margin: int = 128
) -> InpaintRequest:
    return InpaintRequest(
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_crop_margin=margin,
        hd_strategy_crop_merge=merge,
        hd_strategy_crop_merge_max_pixels=max_pixels,
    )


def legacy_crop_rect(
    box: np.ndarray, margin: int, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """`_crop_box` 抽出 `crop_rect` 之前的几何实现，逐行照抄用来比对。"""
    img_h, img_w = image_shape[:2]
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    w = box_w + margin * 2
    h = box_h + margin * 2

    _l = cx - w // 2
    _r = cx + w // 2
    _t = cy - h // 2
    _b = cy + h // 2

    l = max(_l, 0)
    r = min(_r, img_w)
    t = max(_t, 0)
    b = min(_b, img_h)

    if _l < 0:
        r += abs(_l)
    if _r > img_w:
        l -= _r - img_w
    if _t < 0:
        b += abs(_t)
    if _b > img_h:
        t -= _b - img_h

    return max(l, 0), max(t, 0), min(r, img_w), min(b, img_h)


def glyph_row(count: int, top: int = 500, left: int = 200, step: int = 50) -> list:
    """一排挨着的小笔画：掩膜 40×40（间距 10），外扩 128 后裁块大面积重叠。"""
    return [(top, left + i * step, top + 40, left + i * step + 40) for i in range(count)]


@pytest.mark.parametrize(
    "shape, box, margin",
    [
        ((1200, 1600), [400, 500, 480, 560], 128),
        ((1200, 1600), [0, 0, 60, 60], 128),  # 左上角，两条边都贴边
        ((1200, 1600), [1540, 1140, 1600, 1200], 128),  # 右下角
        ((900, 3000), [2900, 100, 3000, 800], 128),  # 右侧竖条
        ((1200, 1600), [700, 500, 900, 700], 0),  # 无边距
        ((1200, 1600), [700, 500, 900, 700], 321),
    ],
)
def test_crop_rect_matches_legacy_geometry(
    shape: tuple[int, int], box: list[int], margin: int
) -> None:
    """把几何从 `_crop_box` 抽出来不能改任何一个像素的取窗范围。"""
    assert crop_rect(np.array(box), margin, shape) == legacy_crop_rect(
        np.array(box), margin, shape
    )


def test_merge_overlapping_rects_merges_only_overlapping() -> None:
    left = (0, 0, 100, 100)
    overlapping = (60, 0, 160, 100)
    touching = (100, 0, 200, 100)
    apart = (110, 0, 210, 100)

    assert merge_overlapping_rects([left, overlapping], max_pixels=10**9) == [
        (0, 0, 160, 100)
    ]
    # 相接：并集面积等于两块之和，合并不多算像素
    assert merge_overlapping_rects([left, touching], max_pixels=10**9) == [
        (0, 0, 200, 100)
    ]
    # 分开 10px：并集比两块之和更大，合并只会多算像素
    assert merge_overlapping_rects([left, apart], max_pixels=10**9) == [left, apart]


def test_merge_overlapping_rects_respects_max_pixels() -> None:
    """重叠也要看面积上限：显存跟着并集面积涨。"""
    a = (0, 0, 1000, 1000)
    b = (100, 100, 1100, 1100)

    assert len(merge_overlapping_rects([a, b], max_pixels=1_000_000)) == 2
    assert merge_overlapping_rects([a, b], max_pixels=1_300_000) == [
        (0, 0, 1100, 1100)
    ]


def test_merge_overlapping_rects_is_greedy_but_bounded() -> None:
    """第三步会超限：并到上限为止，剩下的保持独立。"""
    rects = [(0, 0, 100, 100), (90, 0, 190, 100), (180, 0, 280, 100)]

    merged = merge_overlapping_rects(rects, max_pixels=190 * 100)

    assert merged == [(0, 0, 190, 100), (180, 0, 280, 100)]


def test_merge_overlapping_rects_keeps_first_seen_order() -> None:
    rects = [(500, 0, 600, 100), (0, 0, 100, 100), (550, 0, 650, 100)]

    merged = merge_overlapping_rects(rects, max_pixels=10**9)

    assert merged == [(500, 0, 650, 100), (0, 0, 100, 100)]
    assert merged == merge_overlapping_rects(rects, max_pixels=10**9)


def test_crop_rects_without_merge_is_one_per_contour() -> None:
    image, mask = make_case(
        1200, 1600, [(100, 100, 200, 300), (400, 500, 520, 560), (900, 1200, 1000, 1300)]
    )
    model = StubFlat()
    config = cfg()

    rects = model._crop_rects(mask, config)

    assert rects == [
        crop_rect(box, config.hd_strategy_crop_margin, mask.shape)
        for box in boxes_from_mask(mask)
    ]
    assert len(rects) == 3


def test_crop_rects_accepts_3d_mask() -> None:
    """api 传给模型的是 2D mask，但 `_pad_forward` 里可能是 [H,W,1]，两种都要认。"""
    _, mask = make_case(1200, 1600, glyph_row(3))
    model = StubFlat()

    assert model._crop_rects(mask, cfg(merge=True)) == model._crop_rects(
        mask[:, :, None], cfg(merge=True)
    )


def test_dense_glyphs_merge_into_fewer_forwards() -> None:
    """一排挨着的小笔画：裁块互相重叠，合并后前向次数和总面积都下降。"""
    image, mask = make_case(1200, 1600, glyph_row(10))

    separate = StubFlat()
    separate(image, mask, cfg())
    merged = StubFlat()
    merged(image, mask, cfg(merge=True))

    assert len(separate.forward_shapes) == 10
    assert len(merged.forward_shapes) < len(separate.forward_shapes)
    assert sum(h * w for h, w in merged.forward_shapes) < sum(
        h * w for h, w in separate.forward_shapes
    )


def test_scattered_strokes_are_not_merged() -> None:
    """620px 间距的笔画合成一行宽裁剪是多算像素，实测更慢，所以不合并。"""
    boxes = [(500, 300 + i * 620, 540, 540 + i * 620) for i in range(6)]
    _, mask = make_case(3000, 4000, boxes)
    model = StubFlat()

    assert len(model._crop_rects(mask, cfg(merge=True))) == 6


def test_merged_crop_keeps_pixels_outside_the_mask() -> None:
    image, mask = make_case(1200, 1600, glyph_row(6))
    model = StubFlat()

    out = model(image, mask, cfg(merge=True))

    outside = mask < 127
    assert np.array_equal(out[outside], image[:, :, ::-1][outside])


def test_merged_crop_respects_max_pixels_end_to_end() -> None:
    """上限设得再小，每次前向的面积也都留在上限内。"""
    boxes = glyph_row(10) + [(900, 200, 940, 440), (900, 800, 940, 1040)]
    image, mask = make_case(1200, 1600, boxes)
    model = StubFlat()

    model(image, mask, cfg(merge=True, max_pixels=200_000))

    # `_pad_forward` 会 pad 到 pad_mod，留出边框的余量
    assert all(h * w <= 200_000 + 2 * 8 * (h + w) for h, w in model.forward_shapes)
    assert len(model.forward_shapes) > 1


def test_fixed_size_path_merges_too() -> None:
    """定尺寸瓦片（migan/fcf）非 512 输入也走合并，每次前向仍是 512。"""
    image, mask = make_case(1200, 1600, glyph_row(4))

    separate = StubTile()
    separate(image, mask, cfg())
    merged = StubTile()
    merged(image, mask, cfg(merge=True))

    assert len(separate.forward_shapes) == 4
    assert len(merged.forward_shapes) < len(separate.forward_shapes)
    assert all(shape == (512, 512) for shape in merged.forward_shapes)
