"""动态上下文扩展的分档 A/B 验证（本地，不进 CI）。

对样本目录里的每组 `image + *_mask.png (+ *_cleanup.png)`：

1. 用 `decide_context` 算每个轮廓的复杂度与动态 padding；
2. 用 Dynamic 策略 + 若干固定 padding（`--pads`）各推理一次，记录耗时；
3. 与 `*_cleanup.png`（若存在）比较 mask 内 PSNR / MAE，并记录接缝分数。

结论是「同一台机器上的相对比较」，不是质量金标准：`_cleanup.png` 通常是之前
某一版参数（多为 pad 128）的人工认可结果，PSNR 只衡量与它的接近程度。真正的
判断看 mask 内 MAE、接缝分数和人工盲评。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iopaint.helper import boxes_from_mask  # noqa: E402
from iopaint.model import models  # noqa: E402
from iopaint.model.helper.dynamic_context import decide_context  # noqa: E402
from iopaint.schema import (  # noqa: E402
    DynamicContextConfig,
    HDStrategy,
    InpaintRequest,
)

IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def find_pairs(sample_dir: Path) -> List[Dict[str, Optional[Path]]]:
    pairs = []
    for mask_path in sorted(sample_dir.glob("*_mask.png")):
        stem = mask_path.name[: -len("_mask.png")]
        image = next(
            (sample_dir / (stem + ext) for ext in IMAGE_EXTS if (sample_dir / (stem + ext)).exists()),
            None,
        )
        if image is None:
            continue
        pairs.append(
            {
                "stem": stem,
                "image": image,
                "mask": mask_path,
                "cleanup": sample_dir / (stem + "_cleanup.png"),
            }
        )
    return pairs


def load_pair(pair) -> tuple:
    image = cv2.imread(str(pair["image"]))
    mask = cv2.imread(str(pair["mask"]), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"cannot read {pair['image']} / {pair['mask']}")
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    mask = (mask >= 127).astype(np.uint8) * 255
    clean = cv2.imread(str(pair["cleanup"])) if pair["cleanup"].exists() else None
    if clean is not None and clean.shape != image.shape:
        clean = None
    return image, mask, clean


def psnr(reference: np.ndarray, result: np.ndarray) -> float:
    diff = reference.astype(np.float64) - result.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse == 0 else round(10 * np.log10(255.0 * 255.0 / mse), 2)


def tier_of(complexity: float, cfg: DynamicContextConfig) -> str:
    if complexity < cfg.complexity_low:
        return "low"
    if complexity > cfg.complexity_high:
        return "high"
    return "mid"


def run_once(model, image, mask, cfg: DynamicContextConfig):
    request = InpaintRequest(hd_strategy=HDStrategy.DYNAMIC, dynamic_context=cfg)
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    records: List[dict] = []

    def sink(message):
        text = message.record["message"]
        if text.startswith("dynamic_context "):
            records.append(json.loads(text[len("dynamic_context ") :]))

    handler = logger.add(sink, level="INFO")
    try:
        t0 = time.perf_counter()
        result = model(rgb, mask, request)
        elapsed = time.perf_counter() - t0
    finally:
        logger.remove(handler)
    return elapsed, records, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--model", default="lama")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pads", default="64,128,256")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    pairs = find_pairs(args.sample_dir)
    if not pairs:
        print(f"no image/*_mask.png pairs in {args.sample_dir}")
        return 1

    cfg = DynamicContextConfig()
    fixed_pads = [int(v) for v in args.pads.split(",") if v.strip()]
    model = models[args.model](args.device)
    # 预热：必须带一个真实掩膜轮廓，否则 boxes_from_mask 为空、根本不进前向
    warm_image = np.zeros((512, 512, 3), np.uint8)
    warm_mask = np.zeros((512, 512), np.uint8)
    warm_mask[200:312, 200:312] = 255
    model(
        warm_image,
        warm_mask,
        InpaintRequest(
            hd_strategy=HDStrategy.DYNAMIC,
            dynamic_context=DynamicContextConfig(force_pad=128),
        ),
    )

    rows = []
    for pair in pairs:
        image, mask, clean = load_pair(pair)
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        decisions = [
            decide_context(rgb, mask, box, cfg) for box in boxes_from_mask(mask)
        ]
        complexity = max((d.complexity for d in decisions), default=0.0)
        tier = tier_of(complexity, cfg)
        dynamic_pad = max((d.pad for d in decisions), default=cfg.base_pad)
        mask_pixels = mask >= 127

        for label, pad_cfg in [("dynamic", cfg)] + [
            (f"pad{pad}", DynamicContextConfig(force_pad=pad)) for pad in fixed_pads
        ]:
            times = []
            records: List[dict] = []
            result = None
            rounds = max(3, args.rounds)
            for _ in range(rounds):
                elapsed, records, result = run_once(model, image, mask, pad_cfg)
                times.append(elapsed)
            # TorchScript 的 ProfilingExecutor 前两次执行在采集 profile，之后才用
            # 优化后的图；只统计第 3 次起的耗时，首调单独记录
            measured = times[2:]
            row = {
                "stem": pair["stem"],
                "size": list(image.shape[:2]),
                "complexity": round(complexity, 4),
                "tier": tier,
                "mode": label,
                "pad": dynamic_pad if label == "dynamic" else int(label[3:]),
                "time_s": round(statistics.median(measured), 3),
                "time_first": round(times[0], 3),
                "seams": [r.get("seam_score") for r in records],
                "seam_reruns": sum(1 for r in records if r.get("seam_rerun")),
            }
            if clean is not None:
                diff = np.abs(result.astype(np.int16) - clean.astype(np.int16))
                row["psnr"] = psnr(clean, result)
                row["mask_mae"] = round(float(diff[mask_pixels].mean()), 2)
                row["outside_mae"] = round(float(diff[~mask_pixels].mean()), 3)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    if args.markdown:
        args.markdown.write_text(render_markdown(rows))
    return 0


def render_markdown(rows) -> str:
    lines = [
        "| image | tier | complexity | mode | pad | time(s) | seam | PSNR | mask MAE | outside MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        values = dict(row)
        values["seams"] = ", ".join(
            "0" if s is None else str(s) for s in row["seams"]
        )
        values.setdefault("psnr", "-")
        values.setdefault("mask_mae", "-")
        values.setdefault("outside_mae", "-")
        lines.append(
            "| {stem} | {tier} | {complexity} | {mode} | {pad} | {time_s} | "
            "{seams} | {psnr} | {mask_mae} | {outside_mae} |".format(**values)
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
