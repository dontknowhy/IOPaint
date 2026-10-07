# AGENTS.md

IOPaint = Python backend (`iopaint/`, FastAPI + socket.io, packaged as the `iopaint` CLI) + React/Vite/TS
frontend (`web_app/`) that is compiled into the Python package and served by the backend.

## TOC

- [Commands](#commands)
- [Health check](#health-check)
- [The backend serves the built frontend](#the-backend-serves-the-built-frontend)
- [Tests](#tests)
  - [Slow tests (run locally — CI only spot-checks)](#slow-tests-run-locally--ci-only-spot-checks)
- [Architecture](#architecture)
- [Frontend notes](#frontend-notes)
- [Commit messages](#commit-messages)
- [Other gotchas](#other-gotchas)

## Commands

Backend (conda env `iopaint`, or `pip install -r requirements.txt`):
- `python -m iopaint start --model lama --port 8080` (also `iopaint start ...` or `python main.py start ...` — all hit the
  same typer app). Use `--model cv2` for a weightless, instant model.
- `iopaint run --model lama --device cpu --image DIR --mask DIR --output OUT` for batch mode; `--mask` may be a single file
  applied to every image.
- Others: `iopaint list`, `iopaint download --model <hf-id>` (the id is a **required option**, not a positional arg —
  bare `iopaint download <hf-id>` dies with `Missing option '--model'`), `iopaint start-web-config`,
  `iopaint install-plugins-packages`.
- Model cache = `XDG_CACHE_HOME` (default `~/.cache`). `--model-dir` is a typer callback (`setup_model_dir`) that also sets
  `U2NET_HOME`; it must stay a directory.
- `bash publish.sh [upload]` = build frontend, copy `web_app/dist` → `iopaint/web_app`, `python3 -m build`, optional twine.

Frontend (`cd web_app`):
- `npm run dev` (Vite :5173). `web_app/.env.local` is committed with `VITE_BACKEND=http://127.0.0.1:8080`; there is
  **no** Vite proxy, so dev requests are cross-origin to the backend (CORS is `*`). Production builds ignore it
  (`api.ts` uses `/api/v1` unless `import.meta.env.DEV`).
- `npm run build` = `tsc && vite build`; `npm run lint` = ESLint 9 flat config with `--max-warnings 0` (warnings fail).
- `npm run preview`. Node version pinned in `web_app/.nvmrc` (Vite 7 needs >=22.12).

## Health check

`bash scripts/check.sh` = `pytest -m "not slow"` + `npm run lint` + `npm run build` + copy `web_app/dist` into
`iopaint/web_app`. Runs the same commands as `.github/workflows/ci.yml`; keep them in sync. Three CI jobs: `backend`
(`pytest -m "not slow"`), `frontend` (lint + build), `webui` (playwright against the built UI).

A **separate dispatch-only workflow**, `.github/workflows/slow-smoke.yml`, is deliberately **not** wired to push/PR, so
the PR-required set stays fast and offline. It triggers only on `workflow_dispatch` (no weekly schedule): a clean-room
`pip install` → `iopaint download` → erase-model inference spot check that pre-downloads only lama/fcf and runs
`iopaint/tests/test_local_models.py` (~19 cases) under a 60-minute job timeout. There is **no** persistent weight cache
(GitHub's 10GB / 7-day cap cannot hold the SD-level assets), so `check.sh` does **not** cover it.

## The backend serves the built frontend

`iopaint/api.py` mounts the UI at `/` (`WEB_APP_DIR` = `iopaint/web_app`, a gitignored build output). If that dir is
missing the backend **still starts**: the mount is skipped and `/` answers 503 with build instructions, while every
`/api/v1/*` route works normally. So a fresh clone is testable without touching Node. Copy the *contents* of dist, not
the dist dir itself (the latter nests a level too deep and makes `/` 404):

```bash
cd web_app && npm run build && cd ..
rm -rf iopaint/web_app && cp -r web_app/dist iopaint/web_app
```

`publish.sh` already does this correctly — mirror it.

## Tests

- `pytest -m "not slow" -q` from the repo root is the fast/offline suite (~30s, no weights, no GPU, no frontend build).
  `pytest.ini` declares the `slow` marker; weight-heavy files opt out with a module-level
  `pytestmark = pytest.mark.slow`, so new test files are never missed by hand-maintained lists. Drop that line only if
  the file is genuinely weightless.
- `iopaint/tests/conftest.py` exposes a session-scoped `iopaint_server` that spawns `python -m iopaint start --model cv2
  --device cpu` on a **free port** (`bind(("", 0))`, never 8080) and polls until it accepts connections.
- Tests that construct `ApiConfig(port=8080)` (`test_api.py`, `test_local_models.py`) drive the app through in-process
  `TestClient`, so they do not bind a port at all.
- `cv2` has no weights (`OpenCV2.is_downloaded()` returns `True`). The `slow` set downloads diffusion/erase/plugin
  checkpoints into the model dir; `iopaint/tests/utils.py:check_device()` skips cuda/mps when unavailable; cpu runs use
  2 steps.
- `test_cli.py:test_valid_input_dir` patches `iopaint.api.Api.launch` — `cli.py:start` really calls
  `api.launch()` (uvicorn, blocking), so un-mocking it hangs the test and grabs a port.
- `test_webui.py` is Playwright and `importorskip`s itself, so it silently never runs unless you
  `pip install playwright && playwright install chromium`. It also skips itself when `iopaint/web_app` is missing (it
  drives the real built frontend), so a fresh clone without Node still passes. The `webui` CI job installs chromium and
  builds the frontend first, so it actually runs there — and it fails for real if the frontend regresses (verified).
  It is a **fast** test (not under `-m slow`), and `scripts/check.sh` runs it too.

### Slow tests (run locally — CI only spot-checks)

`pytest -m slow` pulls tens of GB of weights (SD1.5, ControlNet, erase models) and is only practical on a GPU box.
GitHub-hosted runners (16GB RAM, CPU-only, no persistent weight cache — 10GB cap / 7-day expiry, so ~20GB would be
re-downloaded every run) cannot carry the SD-level set, so CI only spot-checks erase models (see **Health check**) and
the full set belongs on this machine.

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

## Architecture

- Entry: `iopaint/__init__.py:entry_point` → `iopaint/cli.py:typer_app` (typer). `cli.py:start` validates paths, dumps the
  env, builds a pydantic `ApiConfig` (`iopaint/schema.py`) and constructs `iopaint/api.py:Api`.
- `Api` owns the routes (`/api/v1/*`, all registered in `api.py:__init__`), exception middleware, the FileManager, the
  plugins, and the static + socket.io mounts. New endpoints go through `self.add_api_route` in `__init__`.
- Model registry: `iopaint/model/__init__.py:models` maps name → class; `ModelManager` (`iopaint/model_manager.py`) scans
  the model dir, downloads lazily, and **serializes all inference behind a `threading.Lock`** (added to fix races on
  shared scheduler state). Call sites therefore use `await loop.run_in_executor(None, self.model_manager, ...)`; keep
  that pattern for any new blocking work so the event loop stays responsive.
- Model contract: each model implements `forward(image, mask, config: InpaintRequest)` and returns a **BGR** numpy array.
  Input image is RGB, mask is single-channel 0/255 with the same HxW as the image (API returns 400 on mismatch). Respect
  each class's `pad_mod` and `is_erase_model` flags.
- Plugins are built by `iopaint/plugins/__init__.py:build_plugins` from `--enable-*` flags. Their deps
  (`onnxruntime<=1.19.2`, `rembg[cpu]`) are **not** in `requirements.txt` — use `iopaint install-plugins-packages`.
- Perf regressions to avoid: don't call `scan_models()` per request (`/api/v1/server-config` uses the cached
  `ModelManager.get_available_models()`; scanning costs ~2s), don't rebuild diffusion schedulers per call (cached by
  `(sampler, lcm_lora)`), don't reload ControlNet preprocessors per request (lazy singletons), and keep
  `--empty-cache-after-inpaint` default-off.

## Frontend notes

- One zustand store in `web_app/src/lib/states.ts` (`persist` + `immer` + `createWithEqualityFn`); the `Settings` type
  lives there too, magic numbers in `web_app/src/lib/const.ts`. `model` is deliberately not persisted.
- `web_app/src/lib/api.ts` is a single axios instance whose response interceptor converts errors into the user-facing
  message (`getErrorMessage`). `API_ENDPOINT` is `VITE_BACKEND + "/api/v1"` in dev and `/api/v1` in production builds.
- socket.io: the server mounts `/ws` **before** the `/` static mount and requires `socketio_path="/ws/socket.io"`
  (`api.py:Api.__init__`); the client hardcodes the same path in `components/DiffusionProgress.tsx`. The `/ws` mount must stay
  ahead of `/`, or the static mount shadows it and progress events 404.
- `components/Editor.tsx` (~1900 lines) paints the mask into a separate offscreen canvas: committed strokes live in
  `lineGroups` while the in-progress stroke is in `strokeRef`, and the canvas is only redrawn as "committed + current
  stroke". The mask is scaled by `maskScale`, re-expanded to full image size in `generateMask` (`lib/utils.ts`) and
  encoded with `canvasToBlob` (not `toDataURL`). Backend `gen_frontend_mask` mirrors the `ffcc00bb` brush color.
- Shared hooks: `useDragResize`, `useImage`, `useInputImage`, `useHotkey`, `useResolution` (viewport breakpoints, not
  model resolution).

## Commit messages

House rule: **statement, not changelog** — a commit message states what *is*, not what *changed*. The diff already records
the change; the message should read like a small piece of documentation that stays true forever, in the same
statement-of-current-state voice as "Other gotchas" below. Never write it as a changelog or a debug diary.

- Subject: `[<model>老师]主题` — the bracketed tag is the writing model's own name plus 老师; MiMo wrote
  `[MiMo老师]...` and a DeepSeek-written commit uses `[DeepSeek老师]`. When more than one model contributes, join
  the tags with `&` (e.g. `[MiMo老师&DeepSeek老师]`). The tag marks AI-authored subjects so they stay
  distinguishable from human ones. Keep the repo's CN/EN mix.
- Body = evidence-first facts a future reader needs. Two shapes:
  - fix → root cause (`file:line`) → impact (who hits it, what symptom) → fix (why it is equivalent/safe) →
    measurement (numbers + a re-runnable regression command);
  - feature / CI / docs → current behavior, key trade-offs and supporting evidence (measured numbers, test counts,
    time/memory), and the pitfalls not to undo.
- Every claim is measured: cite the real numbers and their scope (e.g. "fast test suite: 256 passed"); anything not
  verified is marked `待验证` — never fabricate a result.
- Cut session narrative: CI run IDs, "grabbed a failure / checked frame by frame / walked the call stack", "compared
  with the previous version" phrasing. Those describe the working session, not the code, and go stale the moment they
  are written.
- The message must match the tree: `git show --stat HEAD` before pushing — a multi-path `git add` aborts wholesale
  when one pathspec matches nothing, and that once shipped a message describing files the commit didn't contain.
- Cite only documents the tree actually carries; deleting a doc takes its explicit references with it (decision.md,
  `37f3b22`). Commits written while a doc existed keep their historical references — but rewriting already-pushed
  history (rebase reword + `--force-with-lease`, with `git diff <old> HEAD` empty) only ever happens on explicit
  request.
- The same rule governs comments: yaml/code comments describe current state too, not the debugging history.

## Other gotchas

`scripts/environment.yaml` is a stale lama-cleaner leftover and is not the env for this project; the real, complete env
file is the repo-root `environment.yml` (python 3.12, cu121 torch, pytest), which `scripts/setup_conda.sh` uses.
`requirements.txt` pins `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` (that's what this machine was built with, and
the `+cu121` local version is not on PyPI so it needs a CUDA index — see the comment at the top of the file); the code
itself is not CUDA-version sensitive and other CUDA builds (e.g. cu129) run fine, so don't add version gates, don't
"fix" these pins, and don't reinstall a working env unprompted — CI filters those two lines out and installs CPU wheels.
`iopaint/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK` and torch cache env vars, and `api.py` disables the torch JIT
fusers at import, so keep those statements above `import torch`. `iopaint/web_config.py:save_config` builds the pydantic
model from `locals()` and is wired to a long positional list of Gradio components, so adding a config field means editing
the signature, the `save_btn.click` input list, and `default_configs` together; `web_config.py` itself is a standalone
Gradio config editor behind `iopaint start-web-config` that shares nothing with the `web_app/` frontend, which is the UI
actually served at `/`. Ruff is deliberately **not** a dependency any more (it was in `environment.yml` with no config
committed, and a repo-wide `ruff check` reports ~1700 pre-existing findings), so don't reintroduce it as a gate; if you
want it locally, scope checks to the files you touched and never mass-fix. On language: code comments, UI strings and
commit subjects are deliberately CN/EN mixed (subjects are prefixed `[<model>老师]`, e.g. `[MiMo老师]`), so when you edit a file
keep the surrounding style and never translate, delete, or "normalize" existing Chinese text — the mix rule is scoped to
code / UI / commit text, while AGENTS.md itself stays English because agents read it and it descends from the
English-speaking upstream.
