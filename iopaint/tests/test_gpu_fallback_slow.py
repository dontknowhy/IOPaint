"""真实 CUDA OOM 的端到端回退 —— 只有本地能跑的 slow smoke。

为什么放进 slow：
- 需要至少 2 块可见 CUDA 设备，GitHub-hosted runner 没有 GPU，
  `pytest -m slow` 在 `.github/workflows/slow-smoke.yml` 里只会拿到 skip，不会变红。
- `scripts/check.sh` / CI 的 backend job 跑的是 `-m "not slow"`，也不会碰到它。

本地跑：
    python -m pytest iopaint/tests/test_gpu_fallback_slow.py -m slow -v

怎么造 OOM：
    不去把显存榨干（那会拖垮桌面那块卡，也可能连累同卡的别的进程），而是用
    `torch.cuda.set_per_process_memory_fraction` 把本进程在**首选卡**上的预算压到 2%，
    再申请超过预算的块 —— 分配器会抛出货真价实的 `torch.OutOfMemoryError`，
    但**一个字节都不会真的落到卡上**（实测 allocated 始终为 0）。
    这样回退链路走的是和真实事故完全一样的错误类型/消息，风险却是零。
"""
import pytest
import torch

from iopaint.runtime import list_gpus, load_with_fallback

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 块可见 CUDA 设备",
    ),
]


def _pick_cards():
    """首选卡挑小的，保证存在一块「更有富余」的下家（挑不出就跳过）。"""
    gpus = list_gpus()
    if len(gpus) < 2:
        pytest.skip(f"只看到 {len(gpus)} 块卡")
    totals = {it.total_mb for it in gpus}
    if len(totals) < 2:
        pytest.skip(f"两块卡同型号（{totals}），按总显存挑不出更有富余的下家")
    small = min(gpus, key=lambda it: it.total_mb)
    big = max(gpus, key=lambda it: it.total_mb)
    return small, big


class TestRealOomFallback:
    def test_oom_on_preferred_gpu_switches_to_bigger_one_and_frees_it(self):
        small, big = _pick_cards()
        preferred = torch.device(f"cuda:{small.index}")
        fallback = torch.device(f"cuda:{big.index}")
        budget_bytes = int(small.total_mb * 1024 * 1024 * 0.02)

        torch.cuda.set_per_process_memory_fraction(0.02, preferred)
        try:
            def loader(device):
                idx = device.index or 0
                if device.type == "cuda" and idx == small.index:
                    # 超出 2% 预算 → 真 OutOfMemoryError，且不真的占卡
                    torch.empty(
                        budget_bytes + (64 << 20), dtype=torch.uint8, device=device
                    )
                    raise AssertionError("申请超预算却没抛 OOM，测试前提失效")
                return f"model@{device}"

            baseline_other = torch.cuda.memory_allocated(big.index)
            model, device, notices = load_with_fallback(loader, preferred)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, preferred)

        # 1. 确实换到了显存更有富余的那块卡
        assert str(device) == str(fallback), f"回退到了 {device}"
        assert model == f"model@{fallback}"

        # 2. 提示语同时给终端和 WebUI 用，且指明了从哪块换到哪块
        assert len(notices) == 1
        assert f"cuda:{small.index}" in notices[0]
        assert f"cuda:{big.index}" in notices[0]

        # 3. 关键回归点：失败那块卡必须干干净净
        #    （修复前 traceback 拽着半成品 pipeline，allocated + reserved 都下不去，
        #      nvidia-smi 里就是「失败的卡莫名其妙被占着 1 个多 G」）
        assert torch.cuda.memory_allocated(small.index) == 0, "失败的卡还有活引用没释放"
        assert torch.cuda.memory_reserved(small.index) == 0, "失败的卡的缓存没 empty 掉"

        # 4. 目标卡没有被误伤（loader 在它上面没分配过东西）
        assert torch.cuda.memory_allocated(big.index) == baseline_other

    def test_non_oom_error_does_not_switch_device(self):
        """只有 OOM 才换卡，别的异常必须原样抛（否则会掩盖真正的 bug）。"""
        small, big = _pick_cards()

        def loader(device):
            raise ValueError("权重文件损坏")

        with pytest.raises(ValueError, match="权重文件损坏"):
            load_with_fallback(loader, torch.device(f"cuda:{small.index}"))
