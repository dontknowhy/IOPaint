# Torch pins, env files, and import-time env setup

`scripts/environment.yaml` is a stale lama-cleaner leftover and is not the env for this project; the real, complete env
file is the repo-root `environment.yml` (python 3.12, cu121 torch, pytest), which `scripts/setup_conda.sh` uses.
`requirements.txt` pins `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` (that's what this machine was built with, and
the `+cu121` local version is not on PyPI so it needs a CUDA index — see the comment at the top of the file); the code
itself is not CUDA-version sensitive and other CUDA builds (e.g. cu129) run fine, so don't add version gates, don't
"fix" these pins, and don't reinstall a working env unprompted — CI filters those two lines out and installs CPU wheels.
`iopaint/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK` and torch cache env vars, and `api.py` disables the torch JIT
fusers at import, so keep those statements above `import torch`.
