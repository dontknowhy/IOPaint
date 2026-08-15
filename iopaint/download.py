import glob
import json
import os
from functools import lru_cache
from typing import List, Optional

import torch
from iopaint.schema import ModelType, ModelInfo
from loguru import logger
from pathlib import Path

from iopaint.const import (
    DEFAULT_MODEL_DIR,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
    ANYTEXT_NAME,
)


def cli_download_model(model: str):
    from iopaint.model import models
    from iopaint.model.utils import handle_from_pretrained_exceptions

    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    elif model == ANYTEXT_NAME:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from diffusers import DiffusionPipeline

        downloaded_path = handle_from_pretrained_exceptions(
            DiffusionPipeline.download, pretrained_model_name=model, variant="fp16"
        )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


_UNET_FIRST_CONV_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "unet.conv_in.weight",
)
_SDXL_MARKER_PREFIX = "conditioner."


def _inspect_single_file(model_abs_path: str) -> Optional[tuple]:
    """Inspect a single-file checkpoint without loading its weights.

    Returns (unet_first_conv_in_channels, is_sdxl_layout) or None on failure.
    For .safetensors only the header is read; for .ckpt the pickle must be
    loaded once and both values are derived from the same load.
    """
    if model_abs_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(model_abs_path, framework="pt") as f:
                keys = f.keys()
                in_channels = None
                for key in _UNET_FIRST_CONV_KEYS:
                    if key in keys:
                        in_channels = f.get_slice(key).get_shape()[1]
                        break
                is_sdxl = any(k.startswith(_SDXL_MARKER_PREFIX) for k in keys)
                return in_channels, is_sdxl
        except Exception as e:
            logger.error(f"Failed to inspect {model_abs_path}: {e}")
            return None

    # .ckpt fallback: pickle must be fully loaded to inspect the keys
    try:
        checkpoint = torch.load(model_abs_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.error(f"Failed to load {model_abs_path}: {e}")
        return None
    in_channels = None
    for key in _UNET_FIRST_CONV_KEYS:
        if key in checkpoint:
            in_channels = checkpoint[key].shape[1]
            break
    is_sdxl = any(k.startswith(_SDXL_MARKER_PREFIX) for k in checkpoint.keys())
    return in_channels, is_sdxl


@lru_cache(maxsize=512)
def get_sd_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in Path(model_abs_path).name.lower():
        return ModelType.DIFFUSERS_SD_INPAINT

    inspected = _inspect_single_file(model_abs_path)
    if inspected is None:
        logger.info(f"Ignore non sdxl file: {model_abs_path}")
        return
    in_channels, is_sdxl = inspected
    if is_sdxl:
        logger.info(f"Ignore non sdxl file: {model_abs_path}")
        return
    if in_channels == 9:
        return ModelType.DIFFUSERS_SD_INPAINT
    if in_channels == 4:
        return ModelType.DIFFUSERS_SD
    logger.info(f"Ignore non sdxl file: {model_abs_path}")
    return


@lru_cache()
def get_sdxl_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in model_abs_path:
        return ModelType.DIFFUSERS_SDXL_INPAINT

    inspected = _inspect_single_file(model_abs_path)
    if inspected is None:
        logger.info(f"Ignore non sdxl file: {model_abs_path}")
        return
    in_channels, is_sdxl = inspected
    if not is_sdxl:
        logger.info(f"Ignore non sdxl file: {model_abs_path}")
        return
    if in_channels == 9:
        # https://github.com/huggingface/diffusers/issues/6610
        return ModelType.DIFFUSERS_SDXL_INPAINT
    if in_channels == 4:
        return ModelType.DIFFUSERS_SDXL
    logger.info(f"Ignore non sdxl file: {model_abs_path}")
    return


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    cache_file = stable_diffusion_dir / "iopaint_cache.json"
    model_type_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                model_type_cache = json.load(f)
                assert isinstance(model_type_cache, dict)
        except:
            pass

    res = []
    for it in stable_diffusion_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sd_model_type(model_abs_path)
        if model_type is None:
            continue

        model_type_cache[it.name] = model_type
        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    if stable_diffusion_dir.exists():
        with open(cache_file, "w", encoding="utf-8") as fw:
            json.dump(model_type_cache, fw, indent=2, ensure_ascii=False)

    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    sdxl_cache_file = stable_diffusion_xl_dir / "iopaint_cache.json"
    sdxl_model_type_cache = {}
    if sdxl_cache_file.exists():
        try:
            with open(sdxl_cache_file, "r", encoding="utf-8") as f:
                sdxl_model_type_cache = json.load(f)
                assert isinstance(sdxl_model_type_cache, dict)
        except:
            pass

    for it in stable_diffusion_xl_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = sdxl_model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sdxl_model_type(model_abs_path)
        if model_type is None:
            continue

        sdxl_model_type_cache[it.name] = model_type

        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    if stable_diffusion_xl_dir.exists():
        with open(sdxl_cache_file, "w", encoding="utf-8") as fw:
            json.dump(sdxl_model_type_cache, fw, indent=2, ensure_ascii=False)
    return res


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from iopaint.model import models

    # logger.info(f"Scanning inpaint models in {model_dir}")

    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
    return res


def scan_diffusers_models() -> List[ModelInfo]:
    from huggingface_hub.constants import HF_HUB_CACHE

    available_models = []
    cache_dir = Path(HF_HUB_CACHE)
    # logger.info(f"Scanning diffusers models in {cache_dir}")
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        try:
            with open(it, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        _class_name = data["_class_name"]
        name = folder_name_to_show_name(it.parent.parent.parent.name)
        if name in diffusers_model_names:
            continue
        if "PowerPaint" in name:
            model_type = ModelType.DIFFUSERS_OTHER
        elif _class_name == DIFFUSERS_SD_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD
        elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL
        elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
            model_type = ModelType.DIFFUSERS_SDXL_INPAINT
        elif _class_name in [
            "StableDiffusionInstructPix2PixPipeline",
            "PaintByExamplePipeline",
            "KandinskyV22InpaintPipeline",
            "AnyText",
        ]:
            model_type = ModelType.DIFFUSERS_OTHER
        else:
            continue

        diffusers_model_names.append(name)
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=model_type,
            )
        )
    return available_models


def _scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        with open(it, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                logger.error(
                    f"Failed to load {it}, please try revert from original model or fix model_index.json by hand."
                )
                continue

            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.name)
            if name in diffusers_model_names:
                continue
            elif _class_name == DIFFUSERS_SD_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD
            elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
            elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL
            elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                continue

            diffusers_model_names.append(name)
            available_models.append(
                ModelInfo(
                    name=name,
                    path=str(it.parent.absolute()),
                    model_type=model_type,
                )
            )
    return available_models


def scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_dir))
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_xl_dir))
    return available_models


def scan_models() -> List[ModelInfo]:
    model_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)
    available_models = []
    available_models.extend(scan_inpaint_models(model_dir))
    available_models.extend(scan_single_file_diffusion_models(model_dir))
    available_models.extend(scan_diffusers_models())
    available_models.extend(scan_converted_diffusers_models(model_dir))
    return available_models
