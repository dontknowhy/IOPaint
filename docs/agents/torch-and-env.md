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
ms; 307 vs 311 ms). The code itself is not CUDA-version sensitive, so don't add version gates. CI filters those two lines
out and installs `torch==2.14.1+cpu` / `torchvision==0.29.1+cpu` instead.
Two 2.14 warnings are expected noise, not regressions: `torch.jit.load` is deprecated (lama/zits/manga/migan still use
it) and `torch._C._jit_set_nvfuser_enabled` is now a no-op.
`iopaint/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK` and torch cache env vars, and `api.py` disables the torch JIT
fusers at import, so keep those statements above `import torch`.
