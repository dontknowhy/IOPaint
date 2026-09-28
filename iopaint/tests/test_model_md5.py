import pytest

# 整个文件都遍历加载全部 erase 模型 + vit_l（需要下载权重）
pytestmark = pytest.mark.slow


def test_load_model():
    from iopaint.plugins import InteractiveSeg
    from iopaint.model_manager import ModelManager

    interactive_seg_model = InteractiveSeg("vit_l", "cpu")

    models = ["lama", "ldm", "zits", "mat", "fcf", "manga", "migan"]
    for m in models:
        ModelManager(
            name=m,
            device="cpu",
            no_half=False,
            disable_nsfw=False,
            sd_cpu_textencoder=True,
            cpu_offload=True,
        )
