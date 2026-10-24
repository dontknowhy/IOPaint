import random
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, List, Tuple, get_args

from loguru import logger

from iopaint.const import (
    INSTRUCT_PIX2PIX_NAME,
    KANDINSKY22_NAME,
    POWERPAINT_NAME,
    ANYTEXT_NAME,
    SDXL_CONTROLNET_CHOICES,
    SD2_CONTROLNET_CHOICES,
    SD_CONTROLNET_CHOICES,
    SD_BRUSHNET_CHOICES,
    SDXL_BRUSHNET_CHOICES
)
from pydantic import BaseModel, Field, computed_field, model_validator


class ModelType(str, Enum):
    INPAINT = "inpaint"  # LaMa, MAT...
    DIFFUSERS_SD = "diffusers_sd"
    DIFFUSERS_SD_INPAINT = "diffusers_sd_inpaint"
    DIFFUSERS_SDXL = "diffusers_sdxl"
    DIFFUSERS_SDXL_INPAINT = "diffusers_sdxl_inpaint"
    DIFFUSERS_OTHER = "diffusers_other"


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType
    is_single_file_diffusers: bool = False

    @computed_field
    @property
    def need_prompt(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [
            INSTRUCT_PIX2PIX_NAME,
            KANDINSKY22_NAME,
            POWERPAINT_NAME,
            ANYTEXT_NAME,
        ]

    @computed_field
    @property
    def controlnets(self) -> List[str]:
        if self.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            return SDXL_CONTROLNET_CHOICES
        if self.model_type in [ModelType.DIFFUSERS_SD, ModelType.DIFFUSERS_SD_INPAINT]:
            if "sd2" in self.name.lower():
                return SD2_CONTROLNET_CHOICES
            else:
                return SD_CONTROLNET_CHOICES
        if self.name == POWERPAINT_NAME:
            return SD_CONTROLNET_CHOICES
        return []

    @computed_field
    @property
    def brushnets(self) -> List[str]:
        if self.model_type in [ModelType.DIFFUSERS_SD]:
            return SD_BRUSHNET_CHOICES
        if self.model_type in [ModelType.DIFFUSERS_SDXL]:
            return SDXL_BRUSHNET_CHOICES
        return []

    @computed_field
    @property
    def support_strength(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [POWERPAINT_NAME, ANYTEXT_NAME]

    @computed_field
    @property
    def support_outpainting(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [KANDINSKY22_NAME, POWERPAINT_NAME]

    @computed_field
    @property
    def support_lcm_lora(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_controlnet(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_brushnet(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
        ]

    @computed_field
    @property
    def support_powerpaint_v2(self) -> bool:
        return (
            self.model_type
            in [
                ModelType.DIFFUSERS_SD,
            ]
            and self.name != POWERPAINT_NAME
        )


class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


ExportFormat = Literal["jpeg", "jpg", "png", "webp", "bmp", "tiff"]
# CLI / web-config 的下拉框都从这里派生，新增格式只需改上面的 Literal
EXPORT_FORMATS = get_args(ExportFormat)


class RealESRGANModel(Choices):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


class RemoveBGModel(Choices):
    briaai_rmbg_1_4 = "briaai/RMBG-1.4"
    briaai_rmbg_2_0 = "briaai/RMBG-2.0"
    # models from https://github.com/danielgatis/rembg
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    birefnet_general = "birefnet-general"
    birefnet_general_lite = "birefnet-general-lite"
    birefnet_portrait = "birefnet-portrait"
    birefnet_dis = "birefnet-dis"
    birefnet_hrsod = "birefnet-hrsod"
    birefnet_cod = "birefnet-cod"
    birefnet_massive = "birefnet-massive"


class Device(Choices):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class InteractiveSegModel(Choices):
    vit_b = "vit_b"
    vit_l = "vit_l"
    vit_h = "vit_h"
    sam_hq_vit_b = "sam_hq_vit_b"
    sam_hq_vit_l = "sam_hq_vit_l"
    sam_hq_vit_h = "sam_hq_vit_h"
    mobile_sam = "mobile_sam"

    sam2_tiny = "sam2_tiny"
    sam2_small = "sam2_small"
    sam2_base = "sam2_base"
    sam2_large = "sam2_large"

    sam2_1_tiny = "sam2_1_tiny"
    sam2_1_small = "sam2_1_small"
    sam2_1_base = "sam2_1_base"
    sam2_1_large = "sam2_1_large"


class PluginInfo(BaseModel):
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False


class CV2Flag(str, Enum):
    INPAINT_NS = "INPAINT_NS"
    INPAINT_TELEA = "INPAINT_TELEA"


class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # Resize the longer side of the image to a specific size(hd_strategy_resize_limit),
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    dpm_plus_plus_2m = "DPM++ 2M"
    dpm_plus_plus_2m_karras = "DPM++ 2M Karras"
    dpm_plus_plus_2m_sde = "DPM++ 2M SDE"
    dpm_plus_plus_2m_sde_karras = "DPM++ 2M SDE Karras"
    dpm_plus_plus_sde = "DPM++ SDE"
    dpm_plus_plus_sde_karras = "DPM++ SDE Karras"
    dpm2 = "DPM2"
    dpm2_karras = "DPM2 Karras"
    dpm2_a = "DPM2 a"
    dpm2_a_karras = "DPM2 a Karras"
    euler = "Euler"
    euler_a = "Euler a"
    heun = "Heun"
    lms = "LMS"
    lms_karras = "LMS Karras"

    ddim = "DDIM"
    pndm = "PNDM"
    uni_pc = "UniPC"
    lcm = "LCM"


class PowerPaintTask(Choices):
    text_guided = "text-guided"
    context_aware = "context-aware"
    shape_guided = "shape-guided"
    object_remove = "object-remove"
    outpainting = "outpainting"


class ApiConfig(BaseModel):
    host: str
    port: int
    inbrowser: bool
    model: str
    no_half: bool
    # 强制 diffusion 权重用 fp16 加载（Pascal 上 fp16 更慢，但大模型 fp32 装不下）
    force_fp16: bool = False
    low_mem: bool
    cpu_offload: bool
    disable_nsfw_checker: bool
    local_files_only: bool
    cpu_textencoder: bool
    device: Device
    # 指定用哪块 GPU（仅 device=cuda 时有效）。None = 默认卡
    gpu_id: Optional[int] = None
    input: Optional[Path]
    mask_dir: Optional[Path]
    output_dir: Optional[Path]
    quality: int
    empty_cache_after_inpaint: bool = False
    # 无损盒子：进程内 canonical revision 缓存上限（MB，0=关闭）。
    # 单条上限 = 该值 / 2，禁用阈值 = 该值全额：
    # 超过单条上限只保留最新 canonical（无撤销历史），超过全额则不缓存；
    # 两个阈值由这一个参数派生，不引入第二个尺寸参数。
    revision_cache_mb: int = Field(512, ge=0)
    # 撤销补丁用 zstd 压缩，降低反复编辑的内存占用（需要可选依赖 zstandard；
    # 未安装时回退为原始补丁并打印告警）。压缩只影响历史补丁，不影响像素。
    delta_compress: bool = False
    # 导出默认 quality=100 + 4:4:4；inpaint 响应仍用 quality（保持旧行为/延迟）
    export_quality: int = Field(100, ge=1, le=100)
    # 444 / 420 / keep（keep=沿用源 JPEG 的 get_sampling()）
    export_subsampling: Literal["444", "420", "keep"] = "444"
    # jpeg 源优先走 DCT 系数域局部导出；失败自动回退 full
    export_mode: Literal["dct_patch", "full"] = "dct_patch"
    # 强制覆盖 /api/v1/export 的编码参数：True 时忽略请求里的
    # format/quality/subsampling/export_mode，统一使用下面 export_* 配置；
    # 同时禁用 DCT 局部导出（源量化表不再匹配强制指定的质量）。
    export_force: bool = False
    # 强制导出格式；None = 跟随请求 / 源图格式。jpg 等价 jpeg
    export_format: Optional[ExportFormat] = None
    # PNG zlib 压缩级别 0-9：只影响体积/编码耗时，不影响像素（无损）
    export_png_compress_level: int = Field(1, ge=0, le=9)
    # WebP 导出是否无损（默认无损，保持导出边界旧行为）。无损时
    # export_quality 不生效；只有 export_force=True 时才读取该值，
    # 未强制时沿用 wire 格式的 lossless 行为
    export_webp_lossless: bool = True
    # 无状态兜底（默认关）。开启后 /inpaint 响应改为 PNG level1，
    # 前端把 PNG 字节当作下一轮输入也不会累积损失，代价是每轮 +~1.5s。
    stateless_lossless: bool = False
    # 无状态线格式：png（默认，CPU 最省）或 webp（lossless method=4，体积更小但更慢）
    wire_format: Literal["png", "webp"] = "png"
    # hd_strategy=Crop 时掩膜外扩的边距；同时作用于 erase 与 diffusion 模型
    hd_strategy_crop_margin: int = 128
    enable_interactive_seg: bool
    interactive_seg_model: InteractiveSegModel
    interactive_seg_device: Device
    enable_remove_bg: bool
    remove_bg_device: Device
    remove_bg_model: str
    enable_anime_seg: bool
    enable_realesrgan: bool
    realesrgan_device: Device
    realesrgan_model: RealESRGANModel
    enable_gfpgan: bool
    gfpgan_device: Device
    enable_restoreformer: bool
    restoreformer_device: Device


class MaskStroke(BaseModel):
    """一条前端笔迹：原图坐标下的折线 + 笔宽（px）。"""

    pts: List[Tuple[float, float]] = Field(
        ..., min_length=1, max_length=200_000, description="[[x, y], ...]"
    )
    size: float = Field(..., gt=0, le=65536, description="brush size in pixels")


class InpaintRequest(BaseModel):
    image: Optional[str] = Field(
        None, description="base64 encoded image", max_length=64_000_000
    )
    mask: Optional[str] = Field(
        None, description="base64 encoded mask", max_length=64_000_000
    )
    # 交互编辑的主路径：传笔迹，由服务端按原图尺寸栅格化，省掉整图 PNG 的
    # 生成/编解码。mask 仍是 fallback（交互分割、/adjust_mask、旧客户端）。
    mask_strokes: Optional[List[MaskStroke]] = Field(
        None, max_length=10_000, description="vector mask strokes in image coords"
    )
    # 服务端 canonical 的 revision。命中时 image 可省略（服务端忽略上传字节）。
    revision: Optional[str] = Field(
        None, max_length=128, description="Canonical revision from X-Revision"
    )

    ldm_steps: int = Field(20, ge=1, le=150, description="Steps for ldm model.")
    ldm_sampler: str = Field(LDMSampler.plms, description="Sampler for ldm model.")
    zits_wireframe: bool = Field(True, description="Enable wireframe for zits model.")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, used by erase models(e.g. lama/mat) and diffusion models(e.g. SD/SDXL)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        ge=1,
        le=65536,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, ge=0, le=65536, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, ge=1, le=65536, description="Resize limit for hd_strategy=RESIZE"
    )

    prompt: str = Field("", description="Prompt for diffusion models.")
    negative_prompt: str = Field(
        "", description="Negative prompt for diffusion models."
    )
    use_croper: bool = Field(
        False, description="Crop image before doing diffusion inpainting"
    )
    croper_x: int = Field(0, ge=-16384, le=16384, description="Crop x for croper")
    croper_y: int = Field(0, ge=-16384, le=16384, description="Crop y for croper")
    croper_height: int = Field(
        512, ge=1, le=16384, description="Crop height for croper"
    )
    croper_width: int = Field(
        512, ge=1, le=16384, description="Crop width for croper"
    )

    use_extender: bool = Field(
        False, description="Extend image before doing sd outpainting"
    )
    # extender 坐标可能为负数，见 base.py:_do_outpainting 中 cropper_x/y 注释
    extender_x: int = Field(0, ge=-16384, le=16384, description="Extend x for extender")
    extender_y: int = Field(0, ge=-16384, le=16384, description="Extend y for extender")
    extender_height: int = Field(
        640, ge=1, le=16384, description="Extend height for extender"
    )
    extender_width: int = Field(
        640, ge=1, le=16384, description="Extend width for extender"
    )

    sd_scale: float = Field(
        1.0,
        description="Resize the image before doing sd inpainting, the area outside the mask will not lose quality.",
        gt=0.0,
        le=1.0,
    )
    # D-17：sd_mask_blur 已删除（合成羽化固定为 DiffusionInpaintModel.COMPOSITE_MASK_BLUR，
    # 前端滑块与 payload 同步移除）。pydantic 默认忽略未知字段，旧客户端仍可正常提交。
    sd_strength: float = Field(
        1.0,
        description="Strength is a measure of how much noise is added to the base image, which influences how similar the output is to the base image. Higher value means more noise and more different from the base image",
        le=1.0,
    )
    sd_steps: int = Field(
        50,
        ge=1,
        le=150,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    sd_guidance_scale: float = Field(
        7.5,
        description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
    )
    sd_sampler: str = Field(
        SDSampler.uni_pc, description="Sampler for diffusion model."
    )
    sd_seed: int = Field(
        42,
        description="Seed for diffusion model. -1 mean random seed",
        validate_default=True,
    )
    sd_match_histograms: bool = Field(
        False,
        description="Match histograms between inpainting area and original image.",
    )

    sd_outpainting_softness: float = Field(20.0)
    sd_outpainting_space: float = Field(20.0)

    sd_lcm_lora: bool = Field(
        False,
        description="Enable lcm-lora mode. https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm#texttoimage",
    )

    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )

    cv2_flag: CV2Flag = Field(
        CV2Flag.INPAINT_NS,
        description="Flag for opencv inpainting: https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07",
    )
    cv2_radius: int = Field(
        4,
        description="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm",
    )

    # Paint by Example
    paint_by_example_example_image: Optional[str] = Field(
        None, description="Base64 encoded example image for paint by example model"
    )

    # InstructPix2Pix
    p2p_image_guidance_scale: float = Field(1.5, description="Image guidance scale")

    # ControlNet
    enable_controlnet: bool = Field(False, description="Enable controlnet")
    controlnet_conditioning_scale: float = Field(
        0.4, description="Conditioning scale", ge=0.0, le=1.0
    )
    controlnet_method: str = Field(
        "lllyasviel/control_v11p_sd15_canny", description="Controlnet method"
    )

    # BrushNet
    enable_brushnet: bool = Field(False, description="Enable brushnet")
    brushnet_method: str = Field(SD_BRUSHNET_CHOICES[0], description="Brushnet method")
    brushnet_conditioning_scale: float = Field(
        1.0, description="brushnet conditioning scale", ge=0.0, le=1.0
    )

    # PowerPaint
    enable_powerpaint_v2: bool = Field(False, description="Enable PowerPaint v2")
    powerpaint_task: PowerPaintTask = Field(
        PowerPaintTask.text_guided, description="PowerPaint task"
    )
    fitting_degree: float = Field(
        1.0,
        description="Control the fitting degree of the generated objects to the mask shape.",
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_field(self) -> "InpaintRequest":
        if self.sd_seed == -1:
            self.sd_seed = random.randint(1, 99999999)
            logger.info(f"Generate random seed: {self.sd_seed}")

        if self.use_extender and self.enable_controlnet:
            logger.info("Extender is enabled, set controlnet_conditioning_scale=0")
            self.controlnet_conditioning_scale = 0

        if self.use_extender:
            logger.info("Extender is enabled, set sd_strength=1")
            self.sd_strength = 1.0

        if self.enable_brushnet:
            logger.info("BrushNet is enabled, set enable_controlnet=False")
            if self.enable_controlnet:
                self.enable_controlnet = False
            if self.sd_lcm_lora:
                logger.info("BrushNet is enabled, set sd_lcm_lora=False")
                self.sd_lcm_lora = False

        if self.enable_controlnet:
            logger.info("ControlNet is enabled, set enable_brushnet=False")

        return self


class ExportRequest(BaseModel):
    """导出边界：唯一编码出口。

    revision 命中时使用服务端 canonical 像素；否则用上传的 base64 图片。
    未命中且没有 image 时返回 410，前端回传图片字节重试。
    """

    revision: Optional[str] = Field(None, max_length=128)
    image: Optional[str] = Field(None, max_length=64_000_000)
    # 目标格式；缺省跟随 revision 的源格式 / 上传图片的真实格式
    format: Optional[str] = Field(None, max_length=16)
    quality: Optional[int] = Field(None, ge=1, le=100)
    subsampling: Optional[Literal["444", "420", "keep"]] = None
    export_mode: Optional[Literal["dct_patch", "full"]] = None


class RunPluginRequest(BaseModel):
    name: str
    image: str = Field(..., description="base64 encoded image")
    clicks: List[List[int]] = Field(
        [], description="Clicks for interactive seg, [[x,y,0/1], [x2,y2,0/1]]"
    )
    scale: float = Field(2.0, description="Scale for upscaling")


MediaTab = Literal["input", "output", "mask"]


class MediasResponse(BaseModel):
    name: str
    height: int
    width: int
    ctime: float
    mtime: float


class GenInfoResponse(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""


class ServerConfigResponse(BaseModel):
    plugins: List[PluginInfo]
    modelInfos: List[ModelInfo]
    removeBGModel: RemoveBGModel
    removeBGModels: List[RemoveBGModel]
    realesrganModel: RealESRGANModel
    realesrganModels: List[RealESRGANModel]
    interactiveSegModel: InteractiveSegModel
    interactiveSegModels: List[InteractiveSegModel]
    enableFileManager: bool
    enableAutoSaving: bool
    enableControlnet: bool
    controlnetMethod: Optional[str]
    disableModelSwitch: bool
    isDesktop: bool
    samplers: List[str]
    # 显存不足触发自动回退时的提示，WebUI 拿到后 toast 给用户
    deviceNotice: Optional[str] = None


class SwitchModelRequest(BaseModel):
    name: str


class SwitchPluginModelRequest(BaseModel):
    plugin_name: str
    model_name: str


AdjustMaskOperate = Literal["expand", "shrink", "reverse"]


class AdjustMaskRequest(BaseModel):
    mask: str = Field(
        ..., description="base64 encoded mask. 255 means area to do inpaint"
    )
    operate: AdjustMaskOperate = Field(..., description="expand/shrink/reverse")
    kernel_size: int = Field(
        5, ge=1, le=1000, description="Kernel size for expanding mask"
    )
