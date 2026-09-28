# AGENTS.md

IOPaint = Python backend (`iopaint/`, FastAPI + socket.io, packaged as the `iopaint` CLI) + React/Vite/TS
frontend (`web_app/`) that is compiled into the Python package and served by the backend.

## Commands

Backend (conda env `iopaint`, or `pip install -r requirements.txt`):
- `python -m iopaint start --model lama --port 8080` (also `iopaint start ...` or `python main.py start ...` — all hit the
  same typer app). Use `--model cv2` for a weightless, instant model.
- `iopaint run --model lama --device cpu --image DIR --mask DIR --output OUT` for batch mode; `--mask` may be a single file
  applied to every image.
- Others: `iopaint list`, `iopaint download <hf-id>`, `iopaint start-web-config`, `iopaint install-plugins-packages`.
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
  (`api.py:210-218`); the client hardcodes the same path in `components/DiffusionProgress.tsx`. The `/ws` mount must stay
  ahead of `/`, or the static mount shadows it and progress events 404.
- `components/Editor.tsx` (~1900 lines) paints the mask into a separate offscreen canvas: committed strokes live in
  `lineGroups` while the in-progress stroke is in `strokeRef`, and the canvas is only redrawn as "committed + current
  stroke". The mask is scaled by `maskScale`, re-expanded to full image size in `generateMask` (`lib/utils.ts`) and
  encoded with `canvasToBlob` (not `toDataURL`). Backend `gen_frontend_mask` mirrors the `ffcc00bb` brush color.
- Shared hooks: `useDragResize`, `useImage`, `useInputImage`, `useHotkey`, `useResolution` (viewport breakpoints, not
  model resolution).

## Other gotchas

- `scripts/environment.yaml` is a stale lama-cleaner leftover and is not the env for this project. The real, complete env
  file is the repo-root `environment.yml` (python 3.12, cu121 torch, pytest) — `scripts/setup_conda.sh` uses that one.
- `requirements.txt` pins `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` (that's what this machine was built with, and
  the `+cu121` local version is not on PyPI so it needs a CUDA index — see the comment at the top of the file). The code
  itself is not CUDA-version sensitive; other CUDA builds (e.g. cu129) run fine. Don't add version gates, don't
  "fix" these pins, and don't reinstall a working env unprompted. CI filters those two lines out and installs CPU wheels.
- `iopaint/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK` and torch cache env vars, and `api.py` disables the torch JIT
  fusers at import. Keep those statements above `import torch`.
- `iopaint/web_config.py:save_config` builds the pydantic model from `locals()` and is wired to a long positional list of
  Gradio components — adding a config field means editing the signature, the `save_btn.click` input list, and
  `default_configs` together.
- `web_config.py` is a standalone Gradio config editor behind `iopaint start-web-config`; it shares nothing with the
  `web_app/` frontend, which is the UI actually served at `/`.
- Ruff is deliberately **not** a dependency any more (it was in `environment.yml` with no config committed, and a
  repo-wide `ruff check` reports ~1700 pre-existing findings). Don't reintroduce it as a gate; if you want it locally,
  scope checks to the files you touched and never mass-fix.
- Chinese + English mixed text is the house style, not a defect: code comments, UI strings and commit subjects are all
  mixed, and commit subjects are often prefixed `[作者]主题`. When you edit a file, keep the surrounding style, and don't
  translate, delete, or "normalize" existing Chinese text. Commit subjects follow the same convention.
