"""Python 侧的 i18n 基建：可扩展的语言包加载 + CLI 文案 + `gr.I18n` 组装。

翻译文件放在 `iopaint/locales/<locale>.json`（BCP 47，例如 `zh-CN.json`）：放进一个
文件即自动注册成一门语言。目录里有两个命名空间：

- `web_config.*`：Gradio 配置页，`build_i18n()` 交给 Gradio 前端按浏览器语言选择；
- `cli.*`：命令行帮助与报错，`cli_text()` 按 `IOPAINT_LOCALE` / `LC_ALL` /
  `LC_MESSAGES` / `LANG` 选择（见 `resolve_cli_locale()`）。

英文目录由 `const.py` 的 `*_HELP` 常量 + web_config / CLI 自有文案在 `build_en_catalog()`
/ `build_en_cli_catalog()` 里组装——同一段帮助只有一种英文写法；`iopaint/locales/en.json`
若存在会覆盖它。

本模块不导入 gradio（`build_i18n()` 里懒导入），所以校验脚本 / 单测可以在任意环境
里导入它，`const.py` 也同样不依赖 gradio。
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    import gradio as gr

from iopaint.const import (
    ANIMESEG_HELP,
    DEFAULT_MODEL_DIR,
    CPU_OFFLOAD_HELP,
    CPU_TEXTENCODER_HELP,
    DISABLE_NSFW_HELP,
    EXPORT_FORCE_HELP,
    EXPORT_FORMAT_HELP,
    EXPORT_MODE_HELP,
    EXPORT_PNG_COMPRESS_LEVEL_HELP,
    EXPORT_QUALITY_HELP,
    EXPORT_SUBSAMPLING_HELP,
    EXPORT_WEBP_LOSSLESS_HELP,
    FORCE_FP16_HELP,
    GFPGAN_HELP,
    GPU_ID_HELP,
    HD_STRATEGY_CROP_MARGIN_HELP,
    INBROWSER_HELP,
    INPUT_HELP,
    INTERACTIVE_SEG_HELP,
    INTERACTIVE_SEG_MODEL_HELP,
    LOCAL_FILES_ONLY_HELP,
    LOW_MEM_HELP,
    MASK_DIR_HELP,
    MODEL_DIR_HELP,
    NO_HALF_HELP,
    OUTPUT_DIR_HELP,
    QUALITY_HELP,
    REALESRGAN_HELP,
    REMOVE_BG_DEVICE_HELP,
    REMOVE_BG_HELP,
    RESTOREFORMER_HELP,
)

DEFAULT_LOCALE = "en"
LOCALES_DIR = Path(__file__).parent / "locales"


def build_en_catalog() -> Dict[str, Any]:
    """英文目录 = 规范源。key 用 Gradio/svelte-i18n 的点分命名空间（会嵌套查找）。"""
    return {
        "web_config": {
            "config_file": "Config file",
            "save": "Save configurations",
            "tab_common": "Common",
            "tab_plugins": "Plugins",
            "host": "Host",
            "port": "Port",
            "inbrowser": INBROWSER_HELP,
            "recommend_models": "Recommended Models",
            "downloaded_models": "Downloaded Models",
            "model": (
                "Current Model. Model will be automatically downloaded. "
                "You can select a model in Recommended Models or Downloaded Models "
                "or manually enter the SD/SDXL model ID from HuggingFace, for "
                "example, runwayml/stable-diffusion-inpainting."
            ),
            "device": "Device",
            "gpu_id": f"GPU index {GPU_ID_HELP}",
            "quality": f"Image Quality ({QUALITY_HELP})",
            "export_force": EXPORT_FORCE_HELP,
            "export_format": EXPORT_FORMAT_HELP,
            "export_quality": EXPORT_QUALITY_HELP,
            "export_subsampling": EXPORT_SUBSAMPLING_HELP,
            "export_mode": EXPORT_MODE_HELP,
            "export_png_compress_level": EXPORT_PNG_COMPRESS_LEVEL_HELP,
            "export_webp_lossless": EXPORT_WEBP_LOSSLESS_HELP,
            "empty_cache_after_inpaint": (
                "Empty CUDA cache after each inpainting. "
                "Disabled by default for faster repeated inference."
            ),
            "hd_strategy_crop_margin": (
                "Crop margin (px) for hd_strategy=Crop. "
                "Applies to erase and diffusion models."
            ),
            "no_half": NO_HALF_HELP,
            "force_fp16": FORCE_FP16_HELP,
            "cpu_offload": CPU_OFFLOAD_HELP,
            "low_mem": LOW_MEM_HELP,
            "cpu_textencoder": CPU_TEXTENCODER_HELP,
            "disable_nsfw_checker": DISABLE_NSFW_HELP,
            "local_files_only": LOCAL_FILES_ONLY_HELP,
            "model_dir": MODEL_DIR_HELP,
            "input": f"Input file or directory. {INPUT_HELP}",
            "output_dir": f"Output directory. {OUTPUT_DIR_HELP}",
            "mask_dir": f"Mask directory. {MASK_DIR_HELP}",
            "enable_interactive_seg": INTERACTIVE_SEG_HELP,
            "interactive_seg_model": f"Segment Anything models. {INTERACTIVE_SEG_MODEL_HELP}",
            "interactive_seg_device": "Segment Anything Device",
            "enable_remove_bg": REMOVE_BG_HELP,
            "remove_bg_device": REMOVE_BG_DEVICE_HELP,
            "remove_bg_model": "Remove bg model",
            "enable_anime_seg": ANIMESEG_HELP,
            "enable_realesrgan": REALESRGAN_HELP,
            "realesrgan_device": "RealESRGAN Device",
            "realesrgan_model": "RealESRGAN model",
            "enable_gfpgan": GFPGAN_HELP,
            "gfpgan_device": "GFPGAN Device",
            "enable_restoreformer": RESTOREFORMER_HELP,
            "restoreformer_device": "RestoreFormer Device",
            # 后端 save_config 的动态消息：服务端按请求 locale 用 translate() 取用，
            # 英文串与迁移前的返回值逐字一致（老测试/老日志不受影响）。
            "msg_save_success": "[{time}] Successful save config to: {path}",
            "msg_input_missing": "[Error] Input file or directory does not exist",
            "msg_save_failed": "Save configure file failed: {error}",
        },
        **build_en_cli_catalog(),
    }


def build_en_cli_catalog() -> Dict[str, Any]:
    """CLI（typer）的英文目录 = 规范源。

    与 `web_config` 共用 `const.py` 的 `*_HELP` 常量，所以同一段帮助不会出现两种英文
    写法；CLI 独有的文案（命令描述、掩膜说明、报错）直接写在这里。其他语言在
    `iopaint/locales/<locale>.json` 的 `cli.*` 下补齐相同 key 即可。

    `{...}` 是 `translate()` 手工替换的占位符，调用方传同名参数。
    """
    return {
        "cli": {
            "command": {
                "install_plugins_packages": "Install all plugins dependencies",
                "download": "Download SD/SDXL normal/inpainting model from HuggingFace",
                "list": "List downloaded models",
                "run": "Batch processing images",
                "start": "Start IOPaint server",
                "start_web_config": "Start IOPaint web config page",
            },
            "help": {
                # Typer 用 Rich 渲染 help，`[...]` 会被当成 markup 吞掉；转义开括号
                # 才能显示字面量方括号（同时转义闭括号会留下反斜杠）。
                "model": "Erase models: \\[{erase_models}].\n"
                "Diffusion models: \\[{diffusion_models}] or any SD/SDXL normal/inpainting "
                "models on HuggingFace.",
                "download_model": (
                    "Model id on HuggingFace e.g: runwayml/stable-diffusion-inpainting"
                ),
                "device": "Device. One of: {values}",
                "device_values": "cpu | cuda | cuda:N (use GPU N) | mps",
                "image": "Image folders or file path",
                "mask": "Mask folders or file path. "
                "If it is a directory, the mask images in the directory should have the "
                "same name as the original image."
                "If it is a file, all images will use this mask."
                "Mask will automatically resize to the same size as the original image.",
                "output": "Output directory or file path",
                "config": (
                    "Config file path. You can use dump command to create a base config."
                ),
                "concat": (
                    "Concat original image, mask and output images into one image"
                ),
                "revision_cache_mb": "In-process canonical revision cache budget in MB. "
                "0 disables the lossless revision path. Single-entry and disabled "
                "thresholds derive from this value (half / full).",
                "delta_compress": "Compress undo patches with zstd (optional dependency; "
                "falls back to raw patches when zstandard is absent)",
                "stateless_lossless": "Stateless lossless: use PNG as the /inpaint wire "
                "format so every round is lossless even without the revision cache "
                "(costs ~1.5s per 12MP round)",
                "wire_format": "Wire format for stateless lossless: png (fast) or "
                "webp (smaller, ~3x slower)",
                "empty_cache_after_inpaint": "Call torch.cuda.empty_cache() after each "
                "inpainting. Disabled by default for faster repeated inference, "
                "enable it if you run into CUDA out of memory errors.",
                # 与 web_config 共用 const.py 的常量；这里把常量里拼好的默认路径
                # 参数化，翻译串才能复用同一个默认路径（见 cli.py 的 model_dir 传参）
                "model_dir": MODEL_DIR_HELP.replace(DEFAULT_MODEL_DIR, "{model_dir}"),
                "gpu_id": GPU_ID_HELP,
                "inbrowser": INBROWSER_HELP,
                "low_mem": LOW_MEM_HELP,
                "no_half": NO_HALF_HELP,
                "force_fp16": FORCE_FP16_HELP,
                "cpu_offload": CPU_OFFLOAD_HELP,
                "disable_nsfw_checker": DISABLE_NSFW_HELP,
                "cpu_textencoder": CPU_TEXTENCODER_HELP,
                "local_files_only": LOCAL_FILES_ONLY_HELP,
                "input": INPUT_HELP,
                "mask_dir": MASK_DIR_HELP,
                "output_dir": OUTPUT_DIR_HELP,
                "quality": QUALITY_HELP,
                "export_quality": EXPORT_QUALITY_HELP,
                "export_subsampling": EXPORT_SUBSAMPLING_HELP,
                "export_mode": EXPORT_MODE_HELP,
                "export_force": EXPORT_FORCE_HELP,
                "export_format": EXPORT_FORMAT_HELP,
                "export_png_compress_level": EXPORT_PNG_COMPRESS_LEVEL_HELP,
                "export_webp_lossless": EXPORT_WEBP_LOSSLESS_HELP,
                "hd_strategy_crop_margin": HD_STRATEGY_CROP_MARGIN_HELP,
                "enable_interactive_seg": INTERACTIVE_SEG_HELP,
                "interactive_seg_model": INTERACTIVE_SEG_MODEL_HELP,
                "enable_remove_bg": REMOVE_BG_HELP,
                "remove_bg_device": REMOVE_BG_DEVICE_HELP,
                "enable_anime_seg": ANIMESEG_HELP,
            },
            "error": {
                "export_format": "Invalid --export-format '{value}'. "
                "Choose one of: {choices}",
                "model_download": "Failed to download model {model}: {error}",
                "input_missing": "invalid --input: {input} not exists",
                "mask_dir_missing": "invalid --mask-dir: {mask_dir} not exists",
                "output_dir_required": "invalid --output-dir: --output-dir must be set "
                "when --input is a directory",
                "start_failed": "Failed to start IOPaint: {error}",
                "device_index": "Invalid --device '{spec}': index must be a "
                "non-negative integer, e.g. cuda:1. Valid values: {values}",
                "device": "Invalid --device '{spec}'. Valid values: {values}",
                "device_conflict": "--device '{spec}' conflicts with --gpu-id {gpu_id}, "
                "keep only one",
                "gpu_needs_cuda": "GPU index is only valid with device=cuda, but current "
                "device is '{device}'",
                "gpu_env": "Environment variable {env} must be a non-negative integer, "
                "got '{value}'",
                "gpu_out_of_range": "GPU index {gpu_id} is out of range, only {count} "
                "CUDA device(s) are visible: {gpus}",
                "batch_output_not_dir": "invalid --output: when image is a directory, "
                "output should be a directory",
                "batch_empty_image": "invalid --image: empty image folder",
                "batch_empty_mask": "invalid --mask: empty mask folder",
                "inspect_failed": "Failed to inspect {path}: {error}",
                "model_file_load_failed": "Failed to load {path}: {error}",
                "md5_mismatch_deleted": "Model md5: {md5}, expected md5: {expected}, "
                "wrong model deleted. Please restart iopaint."
                "If you still have errors, please try download model manually first "
                "https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n",
                "md5_mismatch_delete_hint": "Model md5: {md5}, expected md5: {expected}, "
                "please delete {path} and restart iopaint.",
                "md5_mismatch": "Model md5 mismatch for {path}, please delete it and "
                "restart iopaint.",
                "load_failed_issue": "Failed to load model {path},"
                "please submit an issue at https://github.com/Sanster/IOPaint/issues and "
                "include a screenshot of the error:\n{error}",
                "load_failed": "Failed to load model {path}: {error}",
                "hf_fetch": "Fetching model from HuggingFace failed. "
                "If this is your first time downloading the model, you may need to set up "
                "proxy in terminal."
                "If the model has already been downloaded, you can add --local-files-only "
                "when starting.",
                "realesrgan_model": "Unknown RealESRGAN model name: {name}",
                "remove_bg_upgrade": "To use birefnet models, please upgrade rembg to "
                ">= 2.0.59. pip install -U rembg",
            },
        }
    }


def load_catalogs() -> Dict[str, Dict[str, Any]]:
    """`{locale: 目录}`：内置 en + `iopaint/locales/*.json`（文件名即 locale 码）。"""
    catalogs: Dict[str, Dict[str, Any]] = {DEFAULT_LOCALE: build_en_catalog()}
    if not LOCALES_DIR.is_dir():
        return catalogs
    for path in sorted(LOCALES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping invalid locale file {path}: {e}")
            continue
        if not isinstance(data, dict):
            logger.warning(f"Skipping locale file {path}: top level must be an object")
            continue
        catalogs[path.stem] = data
    return catalogs


def _parse_accept_language(accept_language: str) -> List[Tuple[float, str]]:
    """把 `Accept-Language` 解析成 `(q, tag)` 列表，按 q 从大到小排序。"""
    parsed: List[Tuple[float, str]] = []
    for raw in accept_language.split(","):
        part = raw.strip()
        if not part:
            continue
        tag, _, params = part.partition(";")
        tag = tag.strip()
        if not tag or tag == "*":
            continue
        quality = 1.0
        for param in params.split(";"):
            param = param.strip()
            if param.startswith("q="):
                try:
                    quality = float(param[2:])
                except ValueError:
                    quality = 0.0
        if quality > 0:  # q=0 表示不接受该语言
            parsed.append((quality, tag))
    parsed.sort(key=lambda item: item[0], reverse=True)
    return parsed


def pick_locale(accept_language: Optional[str]) -> str:
    """从 `Accept-Language` 里挑一个已有目录的 locale，找不到回退 `en`。

    只做**精确匹配**（大小写不敏感），不做 `zh` → `zh-CN` 这种子标签回退：Gradio
    前端对 `web_config.*` 这类自定义 key 也只在 locale 精确命中时才用该语言的目录，
    否则回退 `en`；服务端跟随同一规则，返回的消息语言才和界面文案一致。
    """
    by_lower = {code.lower(): code for code in load_catalogs()}
    for _quality, tag in _parse_accept_language(accept_language or ""):
        matched = by_lower.get(tag.lower())
        if matched is not None:
            return matched
    return DEFAULT_LOCALE


def _lookup(catalog: Dict[str, Any], key: str) -> Optional[str]:
    node: Any = catalog
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def translate(key: str, locale: str = DEFAULT_LOCALE, **params: Any) -> str:
    """取 `locale` 目录里的点分 `key`，缺失时回退 `en`，再缺失就原样返回 key。

    `{name}` 占位符用 `params` 里的同名值手工替换（不用 `str.format`：翻译串里出现
    字面量花括号时 `format` 会抛，手工替换不会；漏配的占位符会原样留在结果里，由
    key/占位符对齐测试兜底）。
    """
    text = _lookup(load_catalogs().get(locale, {}), key)
    if text is None and locale != DEFAULT_LOCALE:
        text = _lookup(build_en_catalog(), key)
    if text is None:
        return key
    for name, value in params.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def build_i18n() -> "gr.I18n":
    """构造 `gr.I18n` 实例，交给 `demo.launch(i18n=...)`。"""
    import gradio as gr

    return gr.I18n(**load_catalogs())


CLI_LOCALE_ENV = "IOPAINT_LOCALE"
# CLI 没有浏览器语言可用，按 gettext / POSIX 的惯例读环境变量
_CLI_LOCALE_ENV_VARS = ("LC_ALL", "LC_MESSAGES", "LANG")


def _normalize_locale_tag(raw: Optional[str]) -> Optional[str]:
    """`zh_CN.UTF-8` / `zh-CN` / `zh_CN@euro` → `zh-CN`；`C` / `POSIX` / 空 → None。"""
    tag = (raw or "").strip()
    if not tag:
        return None
    tag = tag.split("@")[0].split(".")[0].replace("_", "-")
    if not tag or tag.lower() in ("c", "posix"):
        return None
    return tag


def _match_locale_tag(tag: str) -> Optional[str]:
    """精确匹配已有目录（大小写不敏感）；只写语言时命中唯一一门该语言的目录。"""
    by_lower = {code.lower(): code for code in load_catalogs()}
    matched = by_lower.get(tag.lower())
    if matched is not None:
        return matched
    language = tag.lower().split("-")[0]
    same_language = sorted(
        code for code in by_lower.values() if code.lower().split("-")[0] == language
    )
    return same_language[0] if len(same_language) == 1 else None


def resolve_cli_locale() -> str:
    """CLI 语言：`IOPAINT_LOCALE`（显式指定，认不出就回退 `en`）→ `LC_ALL` →
    `LC_MESSAGES` → `LANG` → `en`。

    `IOPAINT_LOCALE` 和系统变量里的空值都视为未设置。`C` / `POSIX` 直接定成 `en`
    并且不再往下找（与 POSIX 一致：`LC_ALL=C` 会压过后面的 `LANG`）；系统变量里
    `zh_CN.UTF-8` 归一化成 `zh-CN` 精确命中，其余认不出的 locale 继续看下一个变量。
    """
    explicit = os.environ.get(CLI_LOCALE_ENV, "").strip()
    if explicit:
        tag = _normalize_locale_tag(explicit)
        return (_match_locale_tag(tag) if tag else None) or DEFAULT_LOCALE
    for env in _CLI_LOCALE_ENV_VARS:
        raw = os.environ.get(env, "").strip()
        if not raw:
            continue
        tag = _normalize_locale_tag(raw)
        if tag is None:  # C / POSIX：显式要求 C locale，输出英文
            return DEFAULT_LOCALE
        matched = _match_locale_tag(tag)
        if matched is not None:
            return matched
    return DEFAULT_LOCALE


def cli_text(key: str, **params: Any) -> str:
    """按当前 CLI 语言取 `cli.*` 文案（帮助 / 报错）。"""
    return translate(key, resolve_cli_locale(), **params)
