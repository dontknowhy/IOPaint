"""web_config（Gradio 配置页）的 i18n 基建：可扩展的语言包加载 + `gr.I18n` 组装。

翻译文件放在 `iopaint/locales/<locale>.json`（BCP 47，例如 `zh-CN.json`）：放进一个
文件即自动注册成一门语言，`build_i18n()` 会把它交给 Gradio 前端按浏览器语言选择。

英文目录由 `const.py` 的 `*_HELP` 常量 + web_config 自有文案在 `build_en_catalog()`
里组装——单一来源，不会和 CLI help 漂移；`iopaint/locales/en.json` 若存在会覆盖它。

本模块不导入 gradio（`build_i18n()` 里懒导入），所以校验脚本 / 单测可以在任意环境
里导入它，`const.py` 也同样不依赖 gradio。
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    import gradio as gr

from iopaint.const import (
    ANIMESEG_HELP,
    CPU_OFFLOAD_HELP,
    CPU_TEXTENCODER_HELP,
    DISABLE_NSFW_HELP,
    FORCE_FP16_HELP,
    GFPGAN_HELP,
    GPU_ID_HELP,
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
