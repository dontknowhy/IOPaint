import json
import os
from pathlib import Path
from typing import Optional

import mimetypes

# fix for windows mimetypes registry entries being borked
# see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from iopaint.schema import (
    Device,
    InteractiveSegModel,
    RemoveBGModel,
    RealESRGANModel,
    ApiConfig,
    EXPORT_FORMATS,
)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from datetime import datetime
from json import JSONDecodeError

import gradio as gr
from iopaint.download import scan_models
from loguru import logger

from iopaint.const import *
from iopaint.i18n import build_i18n, pick_locale, translate

_config_file: Path = None

default_configs = dict(
    host="127.0.0.1",
    port=8080,
    inbrowser=True,
    model=DEFAULT_MODEL,
    model_dir=DEFAULT_MODEL_DIR,
    no_half=False,
    force_fp16=False,
    low_mem=False,
    cpu_offload=False,
    disable_nsfw_checker=False,
    local_files_only=False,
    cpu_textencoder=False,
    device=Device.cuda,
    gpu_id=None,
    input=None,
    mask_dir=None,
    output_dir=None,
    quality=95,
    export_force=False,
    export_format=None,
    export_quality=100,
    export_subsampling="444",
    export_mode="dct_patch",
    export_png_compress_level=1,
    export_webp_lossless=True,
    empty_cache_after_inpaint=False,
    hd_strategy_crop_margin=128,
    enable_interactive_seg=False,
    interactive_seg_model=InteractiveSegModel.sam2_1_tiny,
    interactive_seg_device=Device.cpu,
    enable_remove_bg=False,
    remove_bg_device=Device.cpu,
    remove_bg_model=RemoveBGModel.briaai_rmbg_1_4,
    enable_anime_seg=False,
    enable_realesrgan=False,
    realesrgan_device=Device.cpu,
    realesrgan_model=RealESRGANModel.realesr_general_x4v3,
    enable_gfpgan=False,
    gfpgan_device=Device.cpu,
    enable_restoreformer=False,
    restoreformer_device=Device.cpu,
)


class WebConfig(ApiConfig):
    model_dir: str = DEFAULT_MODEL_DIR


def load_config(p: Path) -> WebConfig:
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                return WebConfig(**{**default_configs, **json.load(f)})
            except JSONDecodeError:
                print("Load config file failed, using default configs")
                return WebConfig(**default_configs)
    else:
        return WebConfig(**default_configs)


def save_config(
    host,
    port,
    model,
    model_dir,
    no_half,
    force_fp16,
    low_mem,
    cpu_offload,
    disable_nsfw_checker,
    local_files_only,
    cpu_textencoder,
    device,
    gpu_id,
    input,
    mask_dir,
    output_dir,
    quality,
    export_force,
    export_format,
    export_quality,
    export_subsampling,
    export_mode,
    export_png_compress_level,
    export_webp_lossless,
    empty_cache_after_inpaint,
    hd_strategy_crop_margin,
    enable_interactive_seg,
    interactive_seg_model,
    interactive_seg_device,
    enable_remove_bg,
    remove_bg_device,
    remove_bg_model,
    enable_anime_seg,
    enable_realesrgan,
    realesrgan_device,
    realesrgan_model,
    enable_gfpgan,
    gfpgan_device,
    enable_restoreformer,
    restoreformer_device,
    inbrowser,
    request: Optional[gr.Request] = None,
):
    # `request` 由 Gradio 注入（不在 save_btn.click 的组件列表里），构造
    # WebConfig 时先剔除，其余参数与组件列表仍是位置绑定。
    params = dict(locals())
    params.pop("request", None)
    # Gradio 的下拉框用空串表示“跟随源图格式”，pydantic 的 Optional[Literal]
    # 只接受 None，所以构造前先归一化。
    params["export_format"] = params.get("export_format") or None
    config = WebConfig(**params)
    locale = pick_locale(request.headers.get("accept-language") if request else None)
    if str(config.input) == ".":
        config.input = None
    if str(config.output_dir) == ".":
        config.output_dir = None
    if str(config.mask_dir) == ".":
        config.mask_dir = None
    config.model = config.model.strip()
    print(config.model_dump_json(indent=4))
    if config.input and not os.path.exists(config.input):
        return translate("web_config.msg_input_missing", locale)

    current_time = datetime.now().strftime("%H:%M:%S")
    msg = translate(
        "web_config.msg_save_success",
        locale,
        time=current_time,
        path=str(_config_file.absolute()),
    )
    logger.info(msg)
    try:
        with open(_config_file, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=4))
    except Exception as e:
        return translate("web_config.msg_save_failed", locale, error=str(e))
    return msg


def change_current_model(new_model):
    return new_model


def build_demo(config_file: Path, i18n: "gr.I18n") -> "gr.Blocks":
    """组装配置页 UI（不 launch）：抽出后测试能用空闲端口起真服务驱动。"""
    global _config_file
    _config_file = config_file

    init_config = load_config(config_file)
    downloaded_models = [it.name for it in scan_models()]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Textbox(
                    str(config_file),
                    label=i18n("web_config.config_file"),
                    interactive=False,
                )
            with gr.Column():
                save_btn = gr.Button(value=i18n("web_config.save"))
                message = gr.HTML()

        with gr.Tabs():
            with gr.Tab(i18n("web_config.tab_common")):
                with gr.Row():
                    host = gr.Textbox(init_config.host, label=i18n("web_config.host"))
                    port = gr.Number(init_config.port, label=i18n("web_config.port"), precision=0)
                    inbrowser = gr.Checkbox(
                        init_config.inbrowser, label=i18n("web_config.inbrowser")
                    )

                with gr.Row():
                    recommend_model = gr.Dropdown(
                        ["lama", "mat", "migan"] + DIFFUSION_MODELS,
                        label=i18n("web_config.recommend_models"),
                    )
                    downloaded_model = gr.Dropdown(
                        downloaded_models, label=i18n("web_config.downloaded_models")
                    )
                with gr.Column():
                    model = gr.Textbox(
                        init_config.model,
                        label=i18n("web_config.model"),
                    )

                device = gr.Radio(
                    Device.values(), label=i18n("web_config.device"), value=init_config.device
                )
                gpu_id = gr.Number(
                    value=init_config.gpu_id,
                    precision=0,
                    minimum=0,
                    label=i18n("web_config.gpu_id"),
                )
                quality = gr.Slider(
                    value=95,
                    label=i18n("web_config.quality"),
                    minimum=75,
                    maximum=100,
                    step=1,
                )

                with gr.Row():
                    export_force = gr.Checkbox(
                        init_config.export_force,
                        label=i18n("web_config.export_force"),
                    )
                    export_format = gr.Dropdown(
                        [""] + list(EXPORT_FORMATS),
                        value=init_config.export_format or "",
                        label=i18n("web_config.export_format"),
                    )
                    export_quality = gr.Slider(
                        value=init_config.export_quality,
                        label=i18n("web_config.export_quality"),
                        minimum=1,
                        maximum=100,
                        step=1,
                    )
                with gr.Row():
                    export_subsampling = gr.Dropdown(
                        ["444", "420", "keep"],
                        value=init_config.export_subsampling,
                        label=i18n("web_config.export_subsampling"),
                    )
                    export_mode = gr.Dropdown(
                        ["dct_patch", "full"],
                        value=init_config.export_mode,
                        label=i18n("web_config.export_mode"),
                    )
                    export_png_compress_level = gr.Slider(
                        value=init_config.export_png_compress_level,
                        label=i18n("web_config.export_png_compress_level"),
                        minimum=0,
                        maximum=9,
                        step=1,
                    )
                    export_webp_lossless = gr.Checkbox(
                        init_config.export_webp_lossless,
                        label=i18n("web_config.export_webp_lossless"),
                    )

                empty_cache_after_inpaint = gr.Checkbox(
                    init_config.empty_cache_after_inpaint,
                    label=i18n("web_config.empty_cache_after_inpaint"),
                )
                hd_strategy_crop_margin = gr.Number(
                    value=init_config.hd_strategy_crop_margin,
                    precision=0,
                    minimum=0,
                    label=i18n("web_config.hd_strategy_crop_margin"),
                )

                no_half = gr.Checkbox(init_config.no_half, label=i18n("web_config.no_half"))
                force_fp16 = gr.Checkbox(
                    init_config.force_fp16, label=i18n("web_config.force_fp16")
                )
                cpu_offload = gr.Checkbox(
                    init_config.cpu_offload, label=i18n("web_config.cpu_offload")
                )
                low_mem = gr.Checkbox(
                    init_config.low_mem, label=i18n("web_config.low_mem")
                )
                cpu_textencoder = gr.Checkbox(
                    init_config.cpu_textencoder, label=i18n("web_config.cpu_textencoder")
                )
                disable_nsfw_checker = gr.Checkbox(
                    init_config.disable_nsfw_checker,
                    label=i18n("web_config.disable_nsfw_checker"),
                )
                local_files_only = gr.Checkbox(
                    init_config.local_files_only, label=i18n("web_config.local_files_only")
                )

                with gr.Column():
                    model_dir = gr.Textbox(
                        init_config.model_dir, label=i18n("web_config.model_dir")
                    )
                    input = gr.Textbox(
                        init_config.input,
                        label=i18n("web_config.input"),
                    )
                    output_dir = gr.Textbox(
                        init_config.output_dir,
                        label=i18n("web_config.output_dir"),
                    )
                    mask_dir = gr.Textbox(
                        init_config.mask_dir,
                        label=i18n("web_config.mask_dir"),
                    )

            with gr.Tab(i18n("web_config.tab_plugins")):
                with gr.Row():
                    enable_interactive_seg = gr.Checkbox(
                        init_config.enable_interactive_seg,
                        label=i18n("web_config.enable_interactive_seg"),
                    )
                    interactive_seg_model = gr.Radio(
                        InteractiveSegModel.values(),
                        label=i18n("web_config.interactive_seg_model"),
                        value=init_config.interactive_seg_model,
                    )
                    interactive_seg_device = gr.Radio(
                        Device.values(),
                        label=i18n("web_config.interactive_seg_device"),
                        value=init_config.interactive_seg_device,
                    )
                with gr.Row():
                    enable_remove_bg = gr.Checkbox(
                        init_config.enable_remove_bg,
                        label=i18n("web_config.enable_remove_bg"),
                    )
                    remove_bg_device = gr.Radio(
                        Device.values(),
                        label=i18n("web_config.remove_bg_device"),
                        value=init_config.remove_bg_device,
                    )
                    remove_bg_model = gr.Radio(
                        RemoveBGModel.values(),
                        label=i18n("web_config.remove_bg_model"),
                        value=init_config.remove_bg_model,
                    )
                with gr.Row():
                    enable_anime_seg = gr.Checkbox(
                        init_config.enable_anime_seg,
                        label=i18n("web_config.enable_anime_seg"),
                    )

                with gr.Row():
                    enable_realesrgan = gr.Checkbox(
                        init_config.enable_realesrgan,
                        label=i18n("web_config.enable_realesrgan"),
                    )
                    realesrgan_device = gr.Radio(
                        Device.values(),
                        label=i18n("web_config.realesrgan_device"),
                        value=init_config.realesrgan_device,
                    )
                    realesrgan_model = gr.Radio(
                        RealESRGANModel.values(),
                        label=i18n("web_config.realesrgan_model"),
                        value=init_config.realesrgan_model,
                    )
                with gr.Row():
                    enable_gfpgan = gr.Checkbox(
                        init_config.enable_gfpgan, label=i18n("web_config.enable_gfpgan")
                    )
                    gfpgan_device = gr.Radio(
                        Device.values(),
                        label=i18n("web_config.gfpgan_device"),
                        value=init_config.gfpgan_device,
                    )
                with gr.Row():
                    enable_restoreformer = gr.Checkbox(
                        init_config.enable_restoreformer,
                        label=i18n("web_config.enable_restoreformer"),
                    )
                    restoreformer_device = gr.Radio(
                        Device.values(),
                        label=i18n("web_config.restoreformer_device"),
                        value=init_config.restoreformer_device,
                    )

        downloaded_model.change(change_current_model, [downloaded_model], model)
        recommend_model.change(change_current_model, [recommend_model], model)

        save_btn.click(
            save_config,
            [
                host,
                port,
                model,
                model_dir,
                no_half,
                force_fp16,
                low_mem,
                cpu_offload,
                disable_nsfw_checker,
                local_files_only,
                cpu_textencoder,
                device,
                gpu_id,
                input,
                mask_dir,
                output_dir,
                quality,
                export_force,
                export_format,
                export_quality,
                export_subsampling,
                export_mode,
                export_png_compress_level,
                export_webp_lossless,
                empty_cache_after_inpaint,
                hd_strategy_crop_margin,
                enable_interactive_seg,
                interactive_seg_model,
                interactive_seg_device,
                enable_remove_bg,
                remove_bg_device,
                remove_bg_model,
                enable_anime_seg,
                enable_realesrgan,
                realesrgan_device,
                realesrgan_model,
                enable_gfpgan,
                gfpgan_device,
                enable_restoreformer,
                restoreformer_device,
                inbrowser,
            ],
            message,
        )
    return demo


def main(config_file: Path) -> None:
    i18n = build_i18n()
    demo = build_demo(config_file, i18n)
    demo.launch(inbrowser=True, show_api=False, i18n=i18n)
