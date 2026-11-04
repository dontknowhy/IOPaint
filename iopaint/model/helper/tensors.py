"""erase / diffusion 模型共用的「上卡—量化」助手。

每请求的主机↔设备传输是固定开销：512² 的图用 float32 上卡是 3MB，用 uint8 只有
768KB；在设备上除以 255 得到的 float32 与 CPU 端 norm_img() 的逐位相同。回传同理：
先在设备上量化成 uint8 再回传，字节数只有 float32 的 1/4。
"""

import cv2
import numpy as np
import torch


def upload_uint8(image: np.ndarray, device: str) -> torch.Tensor:
    """[H, W, C] uint8 RGB -> [1, C, H, W] 设备端 float32 [0, 1]。"""
    tensor = torch.from_numpy(image)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    # 按 HWC 逐字节上卡（比 float32 少 3/4 字节），排成 CHW 后再转 float，避免让
    # 逐元素拷贝去处理转置后的 stride。
    tensor = tensor.to(device).permute(2, 0, 1).unsqueeze(0)
    # 除数必须是张量：CUDA 上除标量走的是近似除法，126/256 个取值会差 1 ULP，
    # 而按张量除走的是 IEEE 除法，与 CPU 端 norm_img() 逐位相同。
    divisor = torch.tensor(255.0, dtype=torch.float32, device=tensor.device)
    return tensor.float().div_(divisor)


def binary_mask(
    mask: np.ndarray, device: str, threshold: int = 0
) -> torch.Tensor:
    """[H, W] 或 [H, W, 1] uint8 -> [1, 1, H, W] 设备端 float32 0/1。

    `threshold` 是 0-255 口径（各模型原本的 `> 0` / `> 120` / `> 127`），结果与先
    norm_img() 再用同一阈值二值化逐位相同，但上卡只有 H×W 字节。
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = mask > threshold
    return (
        torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).to(device).float()
    )


def to_bgr_uint8(
    output: torch.Tensor,
    scale: float,
    offset: float = 0.0,
    rounding: bool = False,
) -> np.ndarray:
    """[1, C, H, W] 浮点输出 -> [H, W, C] BGR uint8。

    量化在设备上完成，只回传 H×W×C 字节。`rounding=False` 是截断语义，等价于
    `np.clip(x * scale + offset, 0, 255).astype(np.uint8)`；`rounding=True` 是 GAN
    系的 `.round()` 语义。
    """
    tensor = output * scale
    if offset:
        tensor = tensor + offset
    if rounding:
        tensor = tensor.round()
    # 夹紧只在模型输出越界时生效：那种情况下 numpy 的浮点→uint8 转换本身就是未定义行为。
    tensor = tensor.clamp_(0, 255).to(torch.uint8)
    image = tensor[0].permute(1, 2, 0).contiguous().cpu().numpy()
    if image.shape[2] == 1:  # manga 的灰度输出
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
