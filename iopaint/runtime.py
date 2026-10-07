# https://github.com/huggingface/huggingface_hub/blob/5a12851f54bf614be39614034ed3a9031922d297/src/huggingface_hub/utils/_runtime.py
import os
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import packaging.version
from iopaint.schema import Device
from loguru import logger
from rich import print
from typing import Dict, Any, List, Optional, Tuple

# 用户可用 `--device cuda:N` / `--gpu-id N` / `IOPaint_GPU=N` 指定用哪块 GPU
DEVICE_SPEC_HELP = "cpu | cuda | cuda:N (使用第 N 块 GPU) | mps"
GPU_ENV_VAR = "IOPaint_GPU"


_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata  # type: ignore
else:
    import importlib.metadata as importlib_metadata  # type: ignore

_package_versions = {}

_CANDIDATES = [
    "torch",
    "torchvision",
    "Pillow",
    "diffusers",
    "transformers",
    "opencv-python",
    "accelerate",
    "iopaint",
    "rembg",
    "onnxruntime",
]
# Check once at runtime
for name in _CANDIDATES:
    _package_versions[name] = "N/A"
    try:
        _package_versions[name] = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        pass


def dump_environment_info() -> Dict[str, str]:
    """Dump information about the machine to help debugging issues."""

    # Generic machine info
    info: Dict[str, Any] = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
    }
    info.update(_package_versions)
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")
    return info


def dump_gpu_info(device=None, gpu_id: Optional[int] = None) -> None:
    """启动时把显卡列出来，方便确认「到底跑在哪块卡上 / 有没有认出卡」。

    只对将要使用的那张卡查空闲显存，其余卡只报名字和总显存 —— 查空闲会在这张卡上
    建 CUDA 上下文，不该为了打印一行日志就在没打算用的卡上占显存。
    """
    gpus = list_gpus()
    if not gpus:
        print("- GPU: no CUDA device detected\n")
        return

    target = None
    if device is not None and device_str(device) == "cuda":
        target = 0 if gpu_id is None else gpu_id
    if target is not None and 0 <= target < len(gpus):
        free_mb = query_free_mb(gpus[target].index)
        if free_mb is not None:
            gpus[target] = replace(gpus[target], free_mb=free_mb)

    print("\n".join([f"- {it.describe()}" for it in gpus]) + "\n")


def check_device(device: Device) -> Device:
    if device == Device.cuda:
        import platform

        if platform.system() == "Darwin":
            logger.warning("MacOS does not support cuda, use cpu instead")
            return Device.cpu
        else:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA is not available, use cpu instead")
                return Device.cpu
    elif device == Device.mps:
        import torch

        if not torch.backends.mps.is_available():
            logger.warning("mps is not available, use cpu instead")
            return Device.cpu
    return device


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mb: int
    # None = 没查过。查它必须调 mem_get_info，而那会在卡上建 CUDA 上下文
    free_mb: Optional[int] = None

    @property
    def room(self) -> Tuple[int, int]:
        """判断「这块卡是否比另一块更有富余」的排序键：先比总显存，再比空闲显存。"""
        return (self.total_mb, -1 if self.free_mb is None else self.free_mb)

    def describe(self) -> str:
        free = "" if self.free_mb is None else f", free {self.free_mb} MB"
        return f"GPU{self.index} {self.name} (total {self.total_mb} MB{free})"


def query_free_mb(index: int) -> Optional[int]:
    """查某张卡的空闲显存。

    ⚠ `torch.cuda.mem_get_info` 会在这张卡上创建一个 CUDA 上下文（实测 ~40MB，
    nvidia-smi 里会凭空多出一条 compute 进程记录）。总显存可以走
    `get_device_properties`（不建上下文），空闲显存只能这样拿，
    所以只对「马上要动的卡」调用，别拿它来给没打算用的卡做展示。
    """
    try:
        import torch

        free_b, _total_b = torch.cuda.mem_get_info(index)
        return int(free_b >> 20)
    except Exception:
        return None


def list_gpus(query_free: bool = False) -> List[GpuInfo]:
    """枚举当前进程能看到的 CUDA 设备。任何失败都只返回已枚举到的部分，不抛异常。

    默认只拿 index/name/total（这些都不建 CUDA 上下文）；`query_free=True` 才查空闲
    显存，用在 OOM 之后挑下一块卡——那时反正已经要碰这些卡了。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus: List[GpuInfo] = []
        for i in range(torch.cuda.device_count()):
            total_mb = 0
            free_mb = query_free_mb(i) if query_free else None
            try:
                # total 走 properties，不建上下文
                total_mb = int(torch.cuda.get_device_properties(i).total_memory >> 20)
            except Exception:
                # 拿不到总显存不影响选卡，退化成 0 即可
                pass
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"cuda:{i}"
            gpus.append(GpuInfo(i, name, total_mb, free_mb))
        return gpus
    except Exception as e:
        logger.debug(f"Enumerate GPUs failed: {e}")
        return []


def describe_gpus() -> str:
    gpus = list_gpus()
    if not gpus:
        return "no CUDA device detected"
    return "; ".join(it.describe() for it in gpus)


def device_str(device) -> str:
    """把 torch.device / str / Device 枚举统一成 `cuda`、`cuda:1`、`cpu` 这种裸字符串。

    Python 3.11+ 里 `str(Device.cuda)` 是 `"Device.cuda"`（而不是 `"cuda"`），
    直接 `str(device) == "cuda"` 会永远为 False，所以枚举要取 `.value`。
    """
    value = getattr(device, "value", None)
    if isinstance(value, str):
        return value
    return str(device)


def parse_device_spec(
    spec: Optional[str], gpu_id: Optional[int] = None
) -> Tuple[Device, Optional[int]]:
    """解析 `--device`（允许 `cuda:N`）与 `--gpu-id`，返回 (Device, GPU序号)。

    解析/校验失败抛 ValueError，由调用方打印友好错误并退出。
    """
    raw = ("" if spec is None else str(spec)).strip().lower()
    parsed_id: Optional[int] = None
    if ":" in raw:
        name, _, index = raw.partition(":")
        if not index.isdigit():
            raise ValueError(
                f"Invalid --device '{spec}': index must be a non-negative integer, "
                f"e.g. cuda:1. Valid values: {DEVICE_SPEC_HELP}"
            )
        parsed_id = int(index)
        raw = name
    if raw not in Device.values():
        raise ValueError(
            f"Invalid --device '{spec}'. Valid values: {DEVICE_SPEC_HELP}"
        )

    if gpu_id is not None:
        if parsed_id is not None and parsed_id != gpu_id:
            raise ValueError(
                f"--device '{spec}' conflicts with --gpu-id {gpu_id}, keep only one"
            )
        parsed_id = gpu_id

    device = Device(raw)
    if device != Device.cuda:
        if parsed_id is not None:
            raise ValueError(
                f"GPU index is only valid with device=cuda, but current device is '{raw}'"
            )
        env_id = _env_gpu_id(strict=False)
        if env_id is not None:
            logger.warning(
                f"{GPU_ENV_VAR}={env_id} is ignored because device is '{raw}', not cuda"
            )
        return device, None

    if parsed_id is None:
        parsed_id = _env_gpu_id()
    return device, parsed_id


def _env_gpu_id(strict: bool = True) -> Optional[int]:
    raw = os.environ.get(GPU_ENV_VAR, "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        if strict:
            raise ValueError(
                f"Environment variable {GPU_ENV_VAR} must be a non-negative integer, "
                f"got '{raw}'"
            )
        return None
    return int(raw)


def resolve_gpu_id(device: Device, gpu_id: Optional[int]) -> Optional[int]:
    """校验 GPU 序号在当前机器上确实存在；不存在则抛 ValueError。"""
    if device != Device.cuda or gpu_id is None:
        return None
    gpus = list_gpus()
    if not gpus:
        # 没有可用 CUDA（check_device 通常已经先把 device 换成 cpu 了）
        return None
    if gpu_id >= len(gpus):
        raise ValueError(
            f"GPU index {gpu_id} is out of range, only {len(gpus)} CUDA device(s) "
            f"are visible: {describe_gpus()}"
        )
    return gpu_id


def to_torch_device(device: Device, gpu_id: Optional[int] = None):
    """Device 枚举 + GPU 序号 → torch.device。"""
    import torch

    if device == Device.cuda:
        return torch.device("cuda" if gpu_id is None else f"cuda:{gpu_id}")
    return torch.device(device)


def cuda_device_index(device) -> int:
    """把任意 device 表示里的 CUDA 序号取出来（cpu/mps/枚举都返回 0）。"""
    s = device_str(device)
    if not s.startswith("cuda"):
        return 0
    _, _, index = s.partition(":")
    try:
        return int(index) if index else 0
    except ValueError:
        return 0


def _release_memory(device=None) -> None:
    """清掉半路失败的模型占着的显存，再尝试下一个设备。

    两步缺一不可：
    1. `gc.collect()` —— 扩散模型的 pipeline 模块间自引用成环，只靠引用计数回收不掉；
    2. `empty_cache()` —— 回收后显存只是还给 torch 的缓存分配器，nvidia-smi 里仍然算
       「已占用」，必须显式 empty 才会真正释放给驱动。

    `empty_cache()` 只作用于 current device：失败的是 cuda:0、而之后 current 已经切到
    cuda:1，不显式切回去的话 GPU0 上那一两个 G 会永远留在缓存里。
    """
    import gc

    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if device is not None and device_str(device).split(":")[0] == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_out_of_memory(e: BaseException) -> bool:
    """判断一次加载失败是不是「显存不够」，只有这类错误才值得换卡重试。"""
    import torch

    for cls in (
        getattr(torch, "OutOfMemoryError", None),
        getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
    ):
        if cls is not None and isinstance(e, cls):
            return True
    msg = str(e).lower()
    return "out of memory" in msg or "out_of_memory" in msg


def next_bigger_gpu(tried: list) -> Optional[int]:
    """在「没试过」的 GPU 里挑一块比失败那块更有富余的（总显存优先，空闲显存次之）。

    用户期望的顺序是：默认卡 → 显存更大的卡 → CPU。这里只在失败后被调用一次，
    `tried` 是已经因为 OOM 失败过的设备列表。
    """
    if not tried:
        return None
    # 这里要拿空闲显存做同型号多卡的区分，query_free=True 会在候选卡上建 CUDA 上下文 ——
    # 但此刻刚 OOM 完，反正要换卡重试了，代价可以接受
    gpus = list_gpus(query_free=True)
    if not gpus:
        return None

    tried_index = {cuda_device_index(it) for it in tried}
    last_index = cuda_device_index(tried[-1])
    last = next((it for it in gpus if it.index == last_index), None)
    if last is None:
        return None

    # 「显存更大」既包括总显存更大，也包括总显存相同但当前更空闲，
    # 否则同型号多卡机器上永远选不出第二块卡。
    candidates = [
        it
        for it in gpus
        if it.index not in tried_index and it.room > last.room
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda it: it.room, reverse=True)
    return candidates[0].index


def load_with_fallback(loader, preferred):
    """按「首选设备 → 显存更有富余的 GPU → CPU」的顺序重试模型加载。

    - 只有显存不足(OOM)才换设备；其他异常原样抛出，避免掩盖真正的 bug。
    - 返回 (加载结果, 实际生效的 torch.device, 需要通知用户的提示语列表)。
      提示语同时给终端(loguru)和 WebUI(server-config 的 deviceNotice)用。
    """
    import torch

    notices: List[str] = []
    tried: List = []
    device = preferred
    preferred_str = device_str(preferred)

    while True:
        try:
            return loader(device), device, notices
        except Exception as e:
            if device_str(device).split(":")[0] != "cuda" or not is_out_of_memory(e):
                raise
            tried.append(device)
            failed = device_str(device)
            err = str(e)
            # 关键：必须在 e 还活着的时候切断 traceback 链。
            # e.__traceback__ 一路挂着 loader → SD.init_model 的帧 → self →
            # 只挪了一半的 pipeline（模块之间还互相引用成环）。不切断的话，
            # 下面的 gc.collect() 只是空转，那一个多 G 会一直留在失败的卡上。
            # __context__/__cause__ 也可能各带一条 traceback，一并清掉。
            e.__traceback__ = None
            e.__context__ = None
            e.__cause__ = None
        # 走到这里 except 已经结束：e 被 Python 自动删掉，半成品模型再没有任何活引用，
        # 此时清理才真的有效
        logger.warning(f"Model loading on {failed} ran out of VRAM: {err}")
        _release_memory(device)

        bigger = next_bigger_gpu(tried)
        if bigger is None:
            msg = (
                f"{preferred_str} out of VRAM and no other GPU has more headroom, "
                f"fell back to CPU (inference will be much slower)"
            )
            logger.warning(msg)
            notices.append(msg)
            device = torch.device("cpu")
            continue

        target = f"cuda:{bigger}"
        msg = f"{failed} out of VRAM, switched to {target}"
        logger.warning(msg)
        notices.append(msg)
        device = torch.device(target)


def setup_model_dir(model_dir: Path):
    model_dir = model_dir.expanduser().absolute()
    logger.info(f"Model directory: {model_dir}")
    os.environ["U2NET_HOME"] = str(model_dir)
    os.environ["XDG_CACHE_HOME"] = str(model_dir)
    if not model_dir.exists():
        logger.info(f"Create model directory: {model_dir}")
        model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir
