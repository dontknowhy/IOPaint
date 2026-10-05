import os
import importlib.util
import shutil
import ctypes
import logging

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# https://github.com/pytorch/pytorch/issues/27971#issuecomment-1768868068
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"
os.environ["LRU_CACHE_CAPACITY"] = "1"
# prevent CPU memory leak when run model on GPU
# https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869288431
# https://github.com/pytorch/pytorch/issues/108334#issuecomment-1752763633
os.environ["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = "1"

import warnings

from PIL import Image

warnings.simplefilter("ignore", UserWarning)

# 防解压炸弹：显式钉住单张图片的最大像素数（约 1 亿像素）。
# PIL 默认值是 89,478,485（≈89.5MP），本仓库此前未显式设置；这里把它写成
# 显式常量，让限制可见。超过 2× 上限时 Pillow 抛 DecompressionBombError，
# 超过上限时抛 DecompressionBombWarning（RuntimeWarning，不受上面的
# simplefilter("ignore", UserWarning) 影响）。
Image.MAX_IMAGE_PIXELS = 100_000_000


def fix_window_pytorch():
    # copy from: https://github.com/comfyanonymous/ComfyUI/blob/5cbaa9e07c97296b536f240688f5a19300ecf30d/fix_torch.py#L4
    import platform

    try:
        if platform.system() != "Windows":
            return
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            lib_folder = os.path.join(folder, "lib")
            test_file = os.path.join(lib_folder, "fbgemm.dll")
            dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
            if os.path.exists(dest):
                break

            with open(test_file, "rb") as f:
                contents = f.read()
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                mydll = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                logging.warning("Detected pytorch version with libomp issue, patching.")
                shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
    except Exception:
        pass


def entry_point():
    # To make os.environ["XDG_CACHE_HOME"] = args.model_cache_dir works for diffusers
    # https://github.com/huggingface/diffusers/blob/be99201a567c1ccd841dc16fb24e88f7f239c187/src/diffusers/utils/constants.py#L18
    from iopaint.cli import typer_app

    fix_window_pytorch()

    typer_app()
