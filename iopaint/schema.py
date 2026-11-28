import random
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, List, Tuple, get_args

from loguru import logger
from fastapi import UploadFile

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
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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
    # 每个掩膜轮廓单独裁一块送进网络，边距由几何规则与 ring 复杂度决定
    # （见 model/helper/dynamic_context.py），输入窗口不再固定 128。
    DYNAMIC = "Dynamic"


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


class DynamicContextConfig(BaseModel):
    """`hd_strategy=Dynamic` 的全部可调参数。

    决策流程：几何规则先给一个 pad，再在 ring 上算复杂度得到第二个 pad，取较大者
    并钳制。所有参考值与阈值都在这里，运行时不再有魔法数；同一输入、同一配置得到
    同一裁剪、同一 padding、同一推理路径。
    """

    # 基础上下文边距（像素）；复杂度低时保持该值不扩大
    base_pad: int = Field(128, ge=0, le=65536)
    min_pad: int = Field(64, ge=0, le=65536)
    max_pad: int = Field(512, ge=0, le=65536)
    # ring 宽度：膨胀 mask 得到 ring，复杂度指标只在 ring 上算
    ring_width: int = Field(48, ge=1, le=4096)
    # 指标计算前的缩略图最长边（局部 patch 超过该值才缩放）
    thumb_size: int = Field(384, ge=32, le=8192)
    # 几何规则：面积占比超过 mid 时 pad×scale；超过 high 时 pad 至少 high_pad
    area_ratio_mid: float = Field(0.05, gt=0, le=1)
    area_ratio_high: float = Field(0.15, gt=0, le=1)
    area_pad_scale: float = Field(1.5, ge=1.0, le=8.0)
    area_high_pad: int = Field(256, ge=0, le=65536)
    # 熵：16×16 块内 ring 像素的灰度直方图熵
    block_size: int = Field(16, ge=4, le=256)
    min_block_pixels: int = Field(32, ge=1)
    entropy_ref: float = Field(6.0, gt=0)  # 归一化参考（bit）
    # 结构张量：高斯窗边长，coherence=(λ1-λ2)/(λ1+λ2)
    structure_window: int = Field(7, ge=3, le=31)
    # 梯度幅值归一化参考（Sobel 口径，0..1020）
    grad_ref: float = Field(64.0, gt=0)
    # 块匹配：归一化 SSD（除以 255²）的参考值。实测非规则纹理约 1e-3，
    # 0.01 会让这一项几乎恒为 0，取 0.002 让中高纹理能被区分出来
    match_dist_ref: float = Field(0.002, gt=0)
    # 每块最多取多少个边界查询块（等距抽样，确定性）
    max_queries: int = Field(24, ge=1, le=256)
    # complexity = w1*H + w2*match + w3*interaction
    w_entropy: float = Field(0.3, ge=0, le=1)
    w_match: float = Field(0.4, ge=0, le=1)
    w_interaction: float = Field(0.3, ge=0, le=1)
    # pad_dynamic = base_pad * (1 + alpha * complexity)。默认 0：156 个
    # 64~160px 干净块的 GT 标定里，pad 192/256 相对 128 的 PSNR 差在 ±0.03dB
    # 内、胜负各半，抬高 pad 只增加耗时（256 是 128 的 1.42 倍）。换内容类别
    # 想要更大窗口时再调大。
    alpha: float = Field(0.0, ge=0, le=8.0)
    # 低于 low 保持 base_pad；高于 high 至少取 high_pad。复杂度仍会算并写进
    # 日志，默认配置下不再改变 pad。样本实测分布：平坦 0.04~0.05，规则纹理
    # 0.10~0.21，复杂非规则 0.33~0.46。
    complexity_low: float = Field(0.2, ge=0, le=1)
    complexity_high: float = Field(0.45, ge=0, le=1)
    high_pad: int = Field(128, ge=0, le=65536)
    # 几何 pad 达到该值时跳过复杂度计算（已经足够大）
    skip_complexity_at_pad: int = Field(384, ge=0, le=65536)
    enable_complexity: bool = True
    # 接缝兜底：seam_score=边界内侧窄带平均梯度/ring 平均梯度，超阈值重跑一次
    enable_seam_fallback: bool = True
    seam_ratio: float = Field(1.5, gt=0)
    # 比值单独用会在平坦背景上误报（ring 梯度本身接近 0）；窄带梯度
    # 低于该绝对值（Sobel 均值，0..1020）时视为无可视接缝
    seam_min_grad: float = Field(10.0, ge=0)
    seam_band: int = Field(2, ge=1, le=64)
    # 单次前向的输入最长边上限；超过则先缩小推理、结果放大回贴
    max_input_side: int = Field(2048, ge=64, le=65536)
    # 回贴羽化半径（像素，0=硬边）；只影响 mask 边界 ±feather 的范围
    paste_feather: int = Field(3, ge=0, le=64)
    # 调试/分档验证用：非空时跳过决策，所有裁块强制用该 padding
    force_pad: Optional[int] = Field(None, ge=0, le=65536)

    # 这份配置常从手写 JSON 读入：拼错的键要报错，而不是静默沿用默认值
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_order(self) -> "DynamicContextConfig":
        if self.min_pad > self.max_pad:
            raise ValueError(
                f"min_pad ({self.min_pad}) must not exceed max_pad ({self.max_pad})"
            )
        if self.complexity_low > self.complexity_high:
            raise ValueError(
                "complexity_low must not exceed complexity_high "
                f"({self.complexity_low} > {self.complexity_high})"
            )
        if self.area_ratio_mid >= self.area_ratio_high:
            raise ValueError(
                "area_ratio_mid must be smaller than area_ratio_high "
                f"({self.area_ratio_mid} >= {self.area_ratio_high})"
            )
        return self


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
    # 只对纯 SD/SDXL 的 UNet 生效；ControlNet/BrushNet/PowerPaint/LCM LoRA 会回退 eager
    torch_compile: bool = False
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
    # hd_strategy=Crop 时把相互重叠的裁块并成一次前向（默认关：会改变掩膜内像素，
    # 掩膜外仍逐像素保持原图）
    hd_strategy_crop_merge: bool = False
    # 合并后单个裁块的面积上限，防止相邻笔迹一路并成整张图把显存吃穿
    hd_strategy_crop_merge_max_pixels: int = 4 * 1024 * 1024
    # hd_strategy=Dynamic 的全部参数（见 DynamicContextConfig）。启动时用
    # --dynamic-context <json> 覆盖；exclude=True 让启动时打印的 config.json
    # 保持扁平（外部 typer-config wrapper 只认标量字段）。
    dynamic_context: DynamicContextConfig = Field(
        default_factory=lambda: DynamicContextConfig(), exclude=True
    )
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
    # OpenCV 的厚度上限是 MAX_THICKNESS(32767)，超过会让 polylines/circle 断言失败
    size: float = Field(..., gt=0, le=32767, description="brush size in pixels")


class InpaintParams(BaseModel):
    """`/inpaint` 的全部标量参数，JSON 与 multipart 两种请求体共用。

    图像/掩膜不走这里：`/inpaint` 是 multipart，文件部件由 handler 读成原始
    字节直接喂解码器（不再 base64）；`InpaintRequest` 在其上补 `mask_strokes`。
    """

    # 服务端 canonical 的 revision。命中时不用上传图片字节。
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
    hd_strategy_crop_merge: bool = Field(
        False,
        description=(
            "Merge overlapping crops so a multi-stroke mask needs fewer forward "
            "passes; merging changes the pixels inside the mask, the area outside "
            "it stays untouched"
        ),
    )
    hd_strategy_crop_merge_max_pixels: int = Field(
        4 * 1024 * 1024,
        ge=1,
        description=(
            "Upper bound on a merged crop's area in pixels; bounds peak VRAM "
            "(LaMa-class fp32 models need about 1GB per megapixel)"
        ),
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
    def validate_field(self) -> "InpaintParams":
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


class InpaintRequest(InpaintParams):
    """推理侧请求：`_inpaint_blocking` 及其模型拿到的那份参数。

    # 交互编辑的主路径：传笔迹，由服务端按原图尺寸栅格化，省掉整图 PNG 的
    # 生成/编解码。整图掩膜（`mask`）仍是 fallback，来自交互分割或 /adjust_mask。
    """

    mask_strokes: Optional[List[MaskStroke]] = Field(
        None, max_length=10_000, description="vector mask strokes in image coords"
    )
    # hd_strategy=Dynamic 的参数。HTTP 请求由服务端用 ApiConfig 覆盖（多部件
    # 表单不支持嵌套模型），批量模式可直接在 --config JSON 里给这一段。
    dynamic_context: DynamicContextConfig = Field(
        default_factory=lambda: DynamicContextConfig()
    )
    # mask 外扩像素数：覆盖水印/文字的抗锯齿边和 JPEG 振铃。这些像素不属于
    # mask，推理后原样保留，会像孤立的小点。0 = 关闭。
    mask_dilate: int = Field(2, ge=0, le=64)

    @model_validator(mode="after")
    def validate_mask_strokes_total(self) -> "InpaintRequest":
        # 单条笔迹和条数都有上限，但光栅化是逐点 cv2.circle：总点数也必须
        # 有界，否则几百条大笔迹就能占住线程池很久。
        if self.mask_strokes:
            total = sum(len(s.pts) for s in self.mask_strokes)
            if total > 100_000:
                raise ValueError(
                    f"mask_strokes total points ({total}) exceeds limit of 100000"
                )
        return self


class InpaintForm(InpaintParams):
    """`POST /api/v1/inpaint` 的 multipart 表单。

    `image` / `mask` 是文件部件，handler 读成原始字节直接解码，省掉 33% 的
    base64 膨胀和一次 17MB 的 JSON 解析。`mask_strokes` 只能是文本部件，所以
    是 JSON 字符串，handler 解析后交给 `InpaintRequest` 校验。
    """

    mask_strokes: Optional[str] = Field(None, description="JSON: mask_strokes")
    image: Optional[UploadFile] = None
    mask: Optional[UploadFile] = None
    # FileManager 打开的文件：首次编辑直接让服务端从 input/output 目录读，
    # 不再把原图上传一遍。后续编辑仍然走 revision。
    media_tab: Optional[Literal["input", "output"]] = None
    media_name: Optional[str] = None


class ExportRequest(BaseModel):
    """导出边界：唯一编码出口。

    revision 命中时使用服务端 canonical 像素；否则用本次上传的图片字节。
    未命中且没有图片时返回 410，前端回传图片字节重试。
    """

    revision: Optional[str] = Field(None, max_length=128)
    # 目标格式；缺省跟随 revision 的源格式 / 上传图片的真实格式
    format: Optional[ExportFormat] = None
    quality: Optional[int] = Field(None, ge=1, le=100)
    subsampling: Optional[Literal["444", "420", "keep"]] = None
    export_mode: Optional[Literal["dct_patch", "full"]] = None


class ExportForm(ExportRequest):
    """`POST /api/v1/export` 的 multipart 表单：图片是文件部件，不是 base64。"""

    image: Optional[UploadFile] = None


class RunPluginRequest(BaseModel):
    name: str
    image: str = Field(
        ...,
        max_length=64_000_000,
        description="base64 encoded image",
    )
    clicks: List[List[int]] = Field(
        [], description="Clicks for interactive seg, [[x,y,0/1], [x2,y2,0/1]]"
    )
    # RealESRGANUpscaler 把它当 outscale 传给 RealESRGANer.enhance（输出尺寸 ×scale）
    scale: float = Field(2.0, gt=0, le=8, description="Scale for upscaling")


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


class GenInfoForm(BaseModel):
    """`POST /api/v1/gen-info` 的 multipart 表单。

    FileManager 打开的文件可以只给 tab+filename，让服务端直接读盘，避免为了
    抽取 prompt 再把整张原图上传一次。
    """

    file: Optional[UploadFile] = None
    media_tab: Optional[Literal["input", "output"]] = None
    media_name: Optional[str] = None


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
        ...,
        max_length=64_000_000,
        description="base64 encoded mask. 255 means area to do inpaint",
    )
    operate: AdjustMaskOperate = Field(..., description="expand/shrink/reverse")
    kernel_size: int = Field(
        5, ge=1, le=100, description="Kernel size for expanding mask"
    )
