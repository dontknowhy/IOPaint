import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from iopaint.helper import pil_to_bytes
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest


def glob_images(path: Path) -> Dict[str, Path]:
    # png/jpg/jpeg
    if path.is_file():
        return {path.stem: path}
    elif path.is_dir():
        res: Dict[str, Path] = {}
        # 排序：glob 顺序不保证，排序后同名文件的 _1/_2 后缀才可复现
        for it in sorted(path.glob("*.*")):
            if it.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            key = it.stem
            if key in res:
                # D-15：a.jpg + a.png 撞同一个 stem，后扫描到的会把先前的
                # 静默覆盖 → 少跑一张图。改成 a_1 并告警。
                n = 1
                while f"{key}_{n}" in res:
                    n += 1
                new_key = f"{key}_{n}"
                logger.warning(
                    f"Duplicate stem '{key}': {it.name} will be processed as '{new_key}'"
                )
                key = new_key
            res[key] = it
        return res


def batch_inpaint(
    model: str,
    device,
    image: Path,
    mask: Path,
    output: Path,
    config: Optional[Path] = None,
    concat: bool = False,
):
    if image.is_dir() and output.is_file():
        logger.error(
            "invalid --output: when image is a directory, output should be a directory"
        )
        raise SystemExit(1)
    output.mkdir(parents=True, exist_ok=True)

    image_paths = glob_images(image)
    mask_paths = glob_images(mask)
    if len(image_paths) == 0:
        logger.error("invalid --image: empty image folder")
        exit(-1)
    if len(mask_paths) == 0:
        logger.error("invalid --mask: empty mask folder")
        exit(-1)

    if config is None:
        inpaint_request = InpaintRequest()
        logger.info(f"Using default config: {inpaint_request}")
    else:
        with open(config, "r", encoding="utf-8") as f:
            inpaint_request = InpaintRequest(**json.load(f))
        logger.info(f"Using config: {inpaint_request}")

    model_manager = ModelManager(name=model, device=device)
    first_mask = list(mask_paths.values())[0]

    # D-15：判断输出会不会覆盖输入时按解析后的绝对路径比对
    # （--image 与 --output 可能一个是相对路径一个是绝对路径）
    input_files = {it.resolve() for it in image_paths.values()}
    input_files |= {it.resolve() for it in mask_paths.values()}

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Batch processing...", total=len(image_paths))
        for stem, image_p in image_paths.items():
            if mask.is_dir():
                mask_p = mask_paths.get(stem)
                if mask_p is None:
                    # D-15：同名文件被 glob_images 改成 stem_1 后，仍要能回退到
                    # 基础 stem，保证 a.jpg + a.png 共用 a.png 的 mask
                    mask_p = mask_paths.get(stem.rsplit("_", 1)[0])
                if mask_p is None:
                    progress.log(f"mask for {image_p} not found")
                    progress.update(task, advance=1)
                    continue
            else:
                # 单个 mask 应用到所有图片
                mask_p = mask_paths.get(stem, first_mask)

            infos = Image.open(image_p).info

            img = np.array(Image.open(image_p).convert("RGB"))
            mask_img = np.array(Image.open(mask_p).convert("L"))

            if mask_img.shape[:2] != img.shape[:2]:
                progress.log(
                    f"resize mask {mask_p.name} to image {image_p.name} size: {img.shape[:2]}"
                )
                mask_img = cv2.resize(
                    mask_img,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_img[mask_img >= 127] = 255
            mask_img[mask_img < 127] = 0

            # bgr
            inpaint_result = model_manager(img, mask_img, inpaint_request)
            inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
            if concat:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
                inpaint_result = cv2.hconcat([img, mask_img, inpaint_result])

            img_bytes = pil_to_bytes(Image.fromarray(inpaint_result), "png", 100, infos)
            save_p = output / f"{stem}.png"
            # D-15：--output-dir 指向输入目录时，会把输入原图覆盖掉。
            # 只有"文件已存在且属于输入集合"才改名，避免影响重复跑同一输出目录。
            if save_p.exists() and save_p.resolve() in input_files:
                n = 1
                while (candidate := output / f"{stem}_{n}.png").resolve() in input_files:
                    n += 1
                logger.warning(
                    f"Output {save_p.name} is an input file, write to {candidate.name} instead"
                )
                save_p = candidate
            with open(save_p, "wb") as fw:
                fw.write(img_bytes)

            progress.update(task, advance=1)
            # pid = psutil.Process().pid
            # memory_info = psutil.Process(pid).memory_info()
            # memory_in_mb = memory_info.rss / (1024 * 1024)
            # print(f"原图大小：{img.shape},当前进程的内存占用：{memory_in_mb}MB")
