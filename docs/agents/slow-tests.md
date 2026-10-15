# Slow tests (run locally — CI only spot-checks)

`pytest -m slow` pulls tens of GB of weights (SD1.5, ControlNet, erase models) and is only practical on a GPU box.
GitHub-hosted runners (16GB RAM, CPU-only, no persistent weight cache — 10GB cap / 7-day expiry, so ~20GB would be
re-downloaded every run) cannot carry the SD-level set, so CI only spot-checks erase models (see **Health check** in
AGENTS.md) and the full set belongs on this machine.

One-time downloads:

```bash
for m in lama ldm zits mat fcf manga migan; do python -m iopaint download --model "$m"; done
python -m iopaint download --model runwayml/stable-diffusion-v1-5
python -m iopaint download --model runwayml/stable-diffusion-inpainting
# ControlNet / plugin weights download lazily inside the tests (nothing to run)
```

Then run — the full slow set, or this subset, which drops the tests whose weights the repo has no download entry for:

```bash
python -m pytest -m slow -v --tb=short --durations=10
```

```bash
python -m pytest -m slow -v --tb=short \
  --ignore=iopaint/tests/test_instruct_pix2pix.py \
  --ignore=iopaint/tests/test_sdxl.py \
  --ignore=iopaint/tests/test_paint_by_example.py \
  --ignore=iopaint/tests/test_brushnet.py \
  --deselect=iopaint/tests/test_sd_model.py::test_local_file_path \
  --deselect=iopaint/tests/test_controlnet.py::test_local_file_path
```

- The `--ignore`d files need single-file ckpts / repos the repo has no download entry for (5-21GB each; 19.4GB for
  the two deselected `test_local_file_path` params).
- Don't add `--forked` on this GPU box: pytest-forked forks the runtest protocol, torch then raises
  `Cannot re-initialize CUDA in forked subprocess` (measured locally). CI has no GPU, so it *does* pass `--forked`
  to bound memory; if you ever need forked locally, set `CUDA_VISIBLE_DEVICES=""`.
- Tests never bind ports (in-process `TestClient` / `ModelManager`), but the single GPU is shared with the production
  service on :8080 — run when it is idle.
- Fast regression point for the depth-controlnet fix (controlnet-aux 0.0.10 returns a 3-channel depth map):
  `python -m pytest iopaint/tests/test_controlnet.py::test_controlnet_switch -v`.
- `iopaint/tests/test_gpu_fallback_slow.py` is a **local-only** slow smoke test (real CUDA OOM → automatic fallback →
  the failed card's memory must be back to zero). It is only collected under `pytest -m slow` and needs ≥2 visible CUDA
  devices; CI's `slow-smoke.yml` has no GPU, so it gets a skip instead of a failure — it **can never go red and never
  actually runs**. To verify it, do it on this machine: `python -m pytest iopaint/tests/test_gpu_fallback_slow.py -m slow
  -v` (~1.5s, no weight downloads). The OOM is produced by squeezing this process's budget to 2% via
  `set_per_process_memory_fraction`, so it **does not really occupy the GPU** and is safe even on a desktop card.
