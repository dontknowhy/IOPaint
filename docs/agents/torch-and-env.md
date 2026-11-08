# Torch pins, env files, and import-time env setup

`scripts/environment.yaml` is a stale lama-cleaner leftover and is not the env for this project; the real, complete env
file is the repo-root `environment.yml` (python 3.12, pytest), which `scripts/setup_conda.sh` uses. conda's `pytorch`
channel has no 2.14.1 build, so the yml installs torch from its `pip:` section with
`--extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu126` (conda writes that list to a temporary requirements
file, so pip option lines are honoured).
`requirements.txt` pins `torch==2.14.1+cu126` / `torchvision==0.29.1+cu126`: the `+cu126` local version is not on PyPI so
it needs that CUDA index, and the cu126 wheel pins `nvidia-cudnn-cu12==9.10.2.21`, the last cuDNN release that still
runs on SM < 7.5 (the Pascal P104-100 / P620 on this machine). The default PyPI wheel of the same torch is cu13 + cuDNN
9.24, which does **not** run on Pascal, so that index is load-bearing — don't flatten these pins to plain
`torch==2.14.1`. Measured in the cu126 env: `arch_list` still carries sm_50/sm_60, cv2/lama/fcf infer on both Pascal
GPUs, `lama` output differs from the cu121 baseline by max 1/255 (mean ~0), and conv2d / lama timings match (3.52 vs 3.53
ms; 307 vs 311 ms). The code itself is not CUDA-version sensitive, so don't add CUDA-version gates; the one
torch-version gate is the nvfuser call described below. CI filters those two lines out and installs
`torch==2.14.1+cpu` / `torchvision==0.29.1+cpu` instead.
The deprecated `torch.cuda.amp.autocast` wrapper is gone at the source: `iopaint/model/ldm.py`'s `_autocast_on_device` opens
`torch.autocast(self.device.type)` per call, so the device type follows the model's backend — ROCm also reports `cuda`,
Intel XPU reports `xpu`, and hard-coding `"cuda"` would silently disable autocast there. cpu/mps keep the old
no-autocast behavior via `enabled=False`.
`torch._C._jit_set_nvfuser_enabled` has been a recorded no-op since torch 2.2, so `api.py:_torch_major_minor()` (parses
`torch.__version__`, `ValueError` → treated as new) only calls it on torch < 2.2, which keeps the 2.1.2 floor working.
`torch.jit.load` keeps printing its `FutureWarning` on 2.14; it is still the only way to run the lama/zits/manga/migan
TorchScript archives, so the warning is expected noise and is deliberately not filtered from logs.
`iopaint/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK` and the torch cache env vars **before** any `import torch`
(they must be in place before torch reads them), while `api.py` calls `torch._C._jit_override_can_fuse_on_*` /
`_jit_set_*_fuser_enabled` / `_jit_set_profiling_mode` right after its own `import torch` (those calls need torch, so
they cannot sit above it) and before the first model import.
