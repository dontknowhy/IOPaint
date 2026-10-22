import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional, cast

import typer
from fastapi import FastAPI
from loguru import logger
from typer import Option
from typer_config import use_json_config

from iopaint.const import *
from iopaint.i18n import cli_text
from iopaint.runtime import (
    setup_model_dir,
    dump_environment_info,
    dump_gpu_info,
    check_device,
    parse_device_spec,
    resolve_gpu_id,
    to_torch_device,
    hub_is_reachable,
)
from iopaint.schema import (
    InteractiveSegModel,
    Device,
    RealESRGANModel,
    RemoveBGModel,
    ExportFormat,
    EXPORT_FORMATS,
)


def _resolve_cli_device(device: str, gpu_id: Optional[int]):
    """`--device`(支持 cuda:N) + `--gpu-id` → (Device, GPU序号)，失败直接退出。"""
    try:
        parsed_device, parsed_gpu_id = parse_device_spec(device, gpu_id)
        parsed_device = check_device(parsed_device)
        parsed_gpu_id = resolve_gpu_id(parsed_device, parsed_gpu_id)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if parsed_gpu_id is not None:
        logger.info(f"Using GPU {parsed_gpu_id}")
    return parsed_device, parsed_gpu_id


_EXPORT_FORMAT_ALIASES = {"tif": "tiff"}


def _resolve_export_format(value: str) -> Optional[ExportFormat]:
    """`--export-format` → ApiConfig.export_format；空串表示不强制。

    config.json 里的值也走这里（typer_config 把它当命令行参数注入），
    所以非法格式在启动期就报错，而不是等第一次导出才 500。
    """
    fmt = value.strip().lower().lstrip(".")
    if not fmt:
        return None
    fmt = _EXPORT_FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in EXPORT_FORMATS:
        raise ValueError(
            cli_text(
                "cli.error.export_format",
                value=value,
                choices=", ".join(EXPORT_FORMATS),
            )
        )
    return cast(ExportFormat, fmt)


typer_app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@typer_app.command(help=cli_text("cli.command.install_plugins_packages"))
def install_plugins_packages():
    from iopaint.installer import install_plugins_package

    install_plugins_package()


@typer_app.command(help=cli_text("cli.command.download"))
def download(
    model: str = Option(..., help=cli_text("cli.help.download_model")),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=cli_text("cli.help.model_dir", model_dir=DEFAULT_MODEL_DIR),
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model
    from iopaint.exceptions import ModelLoadError

    try:
        cli_download_model(model)
    except ModelLoadError as e:
        logger.error(cli_text("cli.error.model_download", model=model, error=e))
        raise SystemExit(1)


@typer_app.command(name="list", help=cli_text("cli.command.list"))
def list_model(
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=cli_text("cli.help.model_dir", model_dir=DEFAULT_MODEL_DIR),
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import scan_models

    scanned_models = scan_models()
    for it in scanned_models:
        print(it.name)


@typer_app.command(help=cli_text("cli.command.run"))
def run(
    model: str = Option("lama"),
    device: str = Option(
        "cpu",
        help=cli_text("cli.help.device", values=cli_text("cli.help.device_values")),
    ),
    gpu_id: Optional[int] = Option(None, help=cli_text("cli.help.gpu_id")),
    image: Path = Option(..., help=cli_text("cli.help.image")),
    mask: Path = Option(..., help=cli_text("cli.help.mask")),
    output: Path = Option(..., help=cli_text("cli.help.output")),
    config: Path = Option(None, help=cli_text("cli.help.config")),
    concat: bool = Option(False, help=cli_text("cli.help.concat")),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=cli_text("cli.help.model_dir", model_dir=DEFAULT_MODEL_DIR),
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model, scan_models
    from iopaint.exceptions import ModelLoadError

    dump_environment_info()
    device, gpu_id = _resolve_cli_device(device, gpu_id)
    # 放在 resolve 之后：这时才知道目标卡，才能只给它查空闲显存（少建上下文）
    dump_gpu_info(device, gpu_id)

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        try:
            cli_download_model(model)
        except ModelLoadError as e:
            logger.error(cli_text("cli.error.model_download", model=model, error=e))
            raise SystemExit(1)

    from iopaint.batch_processing import batch_inpaint

    batch_inpaint(
        model, to_torch_device(device, gpu_id), image, mask, output, config, concat
    )


@typer_app.command(help=cli_text("cli.command.start"))
@use_json_config()
def start(
    host: str = Option("127.0.0.1"),
    port: int = Option(8080),
    inbrowser: bool = Option(False, help=cli_text("cli.help.inbrowser")),
    model: str = Option(
        DEFAULT_MODEL,
        help=cli_text(
            "cli.help.model",
            erase_models=", ".join(AVAILABLE_MODELS),
            diffusion_models=", ".join(DIFFUSION_MODELS),
        ),
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=cli_text("cli.help.model_dir", model_dir=DEFAULT_MODEL_DIR),
        dir_okay=True,
        file_okay=False,
        callback=setup_model_dir,
    ),
    low_mem: bool = Option(False, help=cli_text("cli.help.low_mem")),
    no_half: bool = Option(False, help=cli_text("cli.help.no_half")),
    force_fp16: bool = Option(False, help=cli_text("cli.help.force_fp16")),
    cpu_offload: bool = Option(False, help=cli_text("cli.help.cpu_offload")),
    disable_nsfw_checker: bool = Option(
        False, help=cli_text("cli.help.disable_nsfw_checker")
    ),
    cpu_textencoder: bool = Option(False, help=cli_text("cli.help.cpu_textencoder")),
    local_files_only: bool = Option(False, help=cli_text("cli.help.local_files_only")),
    device: str = Option(
        "cpu",
        help=cli_text("cli.help.device", values=cli_text("cli.help.device_values")),
    ),
    gpu_id: Optional[int] = Option(None, help=cli_text("cli.help.gpu_id")),
    input: Optional[Path] = Option(None, help=cli_text("cli.help.input")),
    mask_dir: Optional[Path] = Option(
        None, help=cli_text("cli.help.mask_dir"), dir_okay=True, file_okay=False
    ),
    output_dir: Optional[Path] = Option(
        None, help=cli_text("cli.help.output_dir"), dir_okay=True, file_okay=False
    ),
    # 默认 95 与 const.QUALITY_HELP、helper.pil_to_bytes 的默认值保持一致
    quality: int = Option(95, help=cli_text("cli.help.quality")),
    revision_cache_mb: int = Option(
        512, help=cli_text("cli.help.revision_cache_mb")
    ),
    delta_compress: bool = Option(False, help=cli_text("cli.help.delta_compress")),
    export_quality: int = Option(100, help=cli_text("cli.help.export_quality")),
    export_subsampling: str = Option(
        "444", help=cli_text("cli.help.export_subsampling")
    ),
    export_mode: str = Option("dct_patch", help=cli_text("cli.help.export_mode")),
    export_force: bool = Option(False, help=cli_text("cli.help.export_force")),
    export_format: str = Option("", help=cli_text("cli.help.export_format")),
    export_png_compress_level: int = Option(
        1, help=cli_text("cli.help.export_png_compress_level")
    ),
    export_webp_lossless: bool = Option(
        True, help=cli_text("cli.help.export_webp_lossless")
    ),
    stateless_lossless: bool = Option(
        False, help=cli_text("cli.help.stateless_lossless")
    ),
    wire_format: str = Option("png", help=cli_text("cli.help.wire_format")),
    empty_cache_after_inpaint: bool = Option(
        False, help=cli_text("cli.help.empty_cache_after_inpaint")
    ),
    hd_strategy_crop_margin: int = Option(
        128, help=cli_text("cli.help.hd_strategy_crop_margin")
    ),
    enable_interactive_seg: bool = Option(
        False, help=cli_text("cli.help.enable_interactive_seg")
    ),
    interactive_seg_model: InteractiveSegModel = Option(
        InteractiveSegModel.sam2_1_tiny, help=cli_text("cli.help.interactive_seg_model")
    ),
    interactive_seg_device: Device = Option(Device.cpu),
    enable_remove_bg: bool = Option(False, help=cli_text("cli.help.enable_remove_bg")),
    remove_bg_device: Device = Option(
        Device.cpu, help=cli_text("cli.help.remove_bg_device")
    ),
    remove_bg_model: RemoveBGModel = Option(RemoveBGModel.briaai_rmbg_1_4),
    enable_anime_seg: bool = Option(False, help=cli_text("cli.help.enable_anime_seg")),
    enable_realesrgan: bool = Option(False),
    realesrgan_device: Device = Option(Device.cpu),
    realesrgan_model: RealESRGANModel = Option(RealESRGANModel.realesr_general_x4v3),
    enable_gfpgan: bool = Option(False),
    gfpgan_device: Device = Option(Device.cpu),
    enable_restoreformer: bool = Option(False),
    restoreformer_device: Device = Option(Device.cpu),
):
    dump_environment_info()
    try:
        parsed_export_format = _resolve_export_format(export_format)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    device, gpu_id = _resolve_cli_device(device, gpu_id)
    if force_fp16 and no_half:
        logger.warning("--force-fp16 is ignored because --no-half (fp32) is set")
    dump_gpu_info(device, gpu_id)
    remove_bg_device = check_device(remove_bg_device)
    realesrgan_device = check_device(realesrgan_device)
    gfpgan_device = check_device(gfpgan_device)

    if input and not input.exists():
        logger.error(cli_text("cli.error.input_missing", input=input))
        raise SystemExit(1)
    if mask_dir and not mask_dir.exists():
        logger.error(cli_text("cli.error.mask_dir_missing", mask_dir=mask_dir))
        raise SystemExit(1)
    if input and input.is_dir() and not output_dir:
        logger.error(cli_text("cli.error.output_dir_required"))
        raise SystemExit(1)
    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will be saved to {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output directory {output_dir}")
            output_dir.mkdir(parents=True)
    if mask_dir:
        mask_dir = mask_dir.expanduser().absolute()

    model_dir = model_dir.expanduser().absolute()

    if not local_files_only and not hub_is_reachable():
        # 断网 / DNS 污染时 huggingface_hub.model_info 会在 TCP connect 上挂死
        # （它那个 GET 没有超时），from_pretrained 就永远卡住。探不到就用本地缓存。
        logger.warning(
            "HuggingFace hub is unreachable (offline / DNS blocked?), "
            "use local model cache only"
        )
        local_files_only = True
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from iopaint.download import cli_download_model, scan_models
    from iopaint.exceptions import ModelLoadError

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        try:
            cli_download_model(model)
        except ModelLoadError as e:
            # 启动期下载失败：打印错误并退出，语义与原 SystemExit(1) 一致
            logger.error(cli_text("cli.error.model_download", model=model, error=e))
            raise SystemExit(1)

    from iopaint.api import Api
    from iopaint.schema import ApiConfig

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inbrowser:
            webbrowser.open(f"http://localhost:{port}", new=0, autoraise=True)
        yield

    app = FastAPI(lifespan=lifespan)

    api_config = ApiConfig(
        host=host,
        port=port,
        inbrowser=inbrowser,
        model=model,
        no_half=no_half,
        force_fp16=force_fp16,
        low_mem=low_mem,
        cpu_offload=cpu_offload,
        disable_nsfw_checker=disable_nsfw_checker,
        local_files_only=local_files_only,
        cpu_textencoder=cpu_textencoder if device == Device.cuda else False,
        device=device,
        gpu_id=gpu_id,
        input=input,
        mask_dir=mask_dir,
        output_dir=output_dir,
        quality=quality,
        revision_cache_mb=revision_cache_mb,
        delta_compress=delta_compress,
        export_quality=export_quality,
        export_subsampling=cast(
            Literal["444", "420", "keep"], export_subsampling
        ),
        export_mode=cast(Literal["dct_patch", "full"], export_mode),
        export_force=export_force,
        export_format=parsed_export_format,
        export_png_compress_level=export_png_compress_level,
        export_webp_lossless=export_webp_lossless,
        stateless_lossless=stateless_lossless,
        wire_format=cast(Literal["png", "webp"], wire_format),
        empty_cache_after_inpaint=empty_cache_after_inpaint,
        hd_strategy_crop_margin=hd_strategy_crop_margin,
        enable_interactive_seg=enable_interactive_seg,
        interactive_seg_model=interactive_seg_model,
        interactive_seg_device=interactive_seg_device,
        enable_remove_bg=enable_remove_bg,
        remove_bg_device=remove_bg_device,
        remove_bg_model=remove_bg_model,
        enable_anime_seg=enable_anime_seg,
        enable_realesrgan=enable_realesrgan,
        realesrgan_device=realesrgan_device,
        realesrgan_model=realesrgan_model,
        enable_gfpgan=enable_gfpgan,
        gfpgan_device=gfpgan_device,
        enable_restoreformer=enable_restoreformer,
        restoreformer_device=restoreformer_device,
    )
    print(api_config.model_dump_json(indent=4))
    try:
        api = Api(app, api_config)
    except ModelLoadError as e:
        # 插件依赖缺失 / 模型加载失败：启动期语义保持“打印错误并退出”
        logger.error(cli_text("cli.error.start_failed", error=e))
        raise SystemExit(1)
    api.launch()


@typer_app.command(help=cli_text("cli.command.start_web_config"))
def start_web_config(
    config_file: Path = Option("config.json"),
):
    dump_environment_info()
    dump_gpu_info()
    from iopaint.web_config import main

    main(config_file)
