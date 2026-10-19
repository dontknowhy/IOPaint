# AGENTS.md

IOPaint = Python backend (`iopaint/`, FastAPI + socket.io, packaged as the `iopaint` CLI) + React/Vite/TS
frontend (`web_app/`) that is compiled into the Python package and served by the backend.

## TOC

- [Commands](#commands)
- [Health check](#health-check)
- [The backend serves the built frontend](#the-backend-serves-the-built-frontend)
- [Tests](#tests)
- [Architecture](#architecture)
- [Commit messages](#commit-messages)
- [Other gotchas](#other-gotchas)
- [Agent docs index](#agent-docs-index)
- [Multi-batch agent tasks](#multi-batch-agent-tasks)

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
- `npm run test` = `vitest run` (jsdom; `src/**/*.test.ts`, config in `web_app/vitest.config.ts`); `npm run test:watch`
  for watch mode. Vitest lives in `web_app/node_modules` like every other dev dep; its coverage output dir
  (`web_app/coverage`) is gitignored.
- `npm run preview`. Node version pinned in `web_app/.nvmrc` (Vite 7 needs >=22.12).

## Health check

`bash scripts/check.sh` = `pytest -m "not slow"` + `npm run lint` + `npm run test` + `npm run build` + copy
`web_app/dist` into `iopaint/web_app` + `python scripts/check_types.py` (new-code-only mypy; skipped with a notice
locally when mypy isn't installed, always installed in CI). Runs the same commands as `.github/workflows/ci.yml`;
keep them in sync. Three CI jobs: `backend`
(`pytest -m "not slow"` + "Type check (new code only)"), `frontend` (lint + test + build), `webui` (playwright against the built UI).

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

Slow tests: see docs/agents/slow-tests.md (trigger: running or extending the weight-heavy `pytest -m slow` suite, or
downloading model weights). CI only spot-checks erase models via the dispatch-only `slow-smoke.yml`; the full set
belongs on a local GPU box.

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

Frontend details (zustand store, api.ts, socket.io `/ws` mount order, Editor mask canvas, shared hooks): see
docs/agents/frontend.md (trigger: working in `web_app/src/`, or touching the backend's static/socket.io mounts).

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
- Executable criterion for the subject: it must name at least one concrete symbol (class / method / file, e.g.
  `ModelManager`, `_wait_interactive`, `Editor.tsx`) **and** one concrete mechanism noun (e.g. `threading.Lock`,
  `run_in_executor`, `pointer-events`); a subject failing either half is rewritten before committing. Two
  negative → positive pairs:
  - ✗ `[X老师]ModelManager 通过锁机制保证了并发安全性` → ✓ `[X老师]ModelManager 用 threading.Lock 序列化推理，消除共享 scheduler 状态竞态`
  - ✗ `[X老师]修复了 webui 测试不稳定的问题` → ✓ `[X老师]test_webui 在 Radix 关闭动画后等 pointerEvents 恢复再绘图，消除偶发丢笔画`

## Other gotchas

Torch pins, env files (`environment.yml` vs the stale `scripts/environment.yaml`), and the MPS-fallback / JIT-fuser
import order: see docs/agents/torch-and-env.md (trigger: touching `requirements.txt` / `environment.yml`, setting up an
env, or moving statements around `import torch`).
`web_config.py` (the standalone Gradio config editor behind `iopaint start-web-config`, unrelated to the `web_app/`
frontend): see docs/agents/web-config.md (trigger: adding a config field or editing `iopaint/web_config.py`).
Ruff is deliberately **not** a dependency any more (it was in `environment.yml` with no config
committed, and a repo-wide `ruff check` reports ~1700 pre-existing findings), so don't reintroduce it as a gate; if you
want it locally, scope checks to the files you touched and never mass-fix. On language: code comments, UI strings and
commit subjects are deliberately CN/EN mixed (subjects are prefixed `[<model>老师]`, e.g. `[MiMo老师]`), so when you edit a file
keep the surrounding style and never translate, delete, or "normalize" existing Chinese text — the mix rule is scoped to
code / UI / commit text, while AGENTS.md itself stays English because agents read it and it descends from the
English-speaking upstream.

Playwright webui test details (Radix Select/Dialog `pointer-events: none` close animation, `_wait_interactive()`,
brush-slider `aria-valuemax`): see docs/agents/playwright.md (trigger: writing or debugging `iopaint/tests/test_webui.py`
cases that draw on the canvas or assert on UI controls).

## Agent docs index

Long-form topics live in `docs/agents/*.md` (all English, like this file). Each entry has the condition for when to
read it:

- `docs/agents/slow-tests.md` — running/extending the weight-heavy `pytest -m slow` suite; one-time weight downloads;
  `--forked`/GPU/shared-:8080 gotchas (trigger: about to run or modify slow tests).
- `docs/agents/playwright.md` — webui test details: Radix close-animation `pointer-events`, `_wait_interactive()`,
  brush-slider `aria-valuemax` (trigger: writing or debugging `test_webui.py`).
- `docs/agents/torch-and-env.md` — torch cu121 pins, `environment.yml` vs stale `scripts/environment.yaml`, MPS
  fallback / JIT fuser import order (trigger: touching `requirements.txt`/`environment.yml` or `import torch` order).
- `docs/agents/web-config.md` — `web_config.py` Gradio wiring: `save_config`'s positional component list,
  `save_btn.click` inputs, `default_configs` move together (trigger: adding a config field or editing web_config.py).
- `docs/agents/frontend.md` — zustand store, `api.ts`, socket.io `/ws`-before-`/` mount order, Editor mask canvas,
  shared hooks (trigger: working in `web_app/src/` or the backend static/socket.io mounts).
- `docs/agents/i18n.md` — the two language catalogs (`web_app/src/locales/*.json`, `iopaint/locales/*.json`), how to
  add a language (drop one JSON), key conventions, and the parity tests (trigger: adding/editing UI strings or a
  locale).
- `docs/agents/type-check.md` — new-code-only mypy gate: `python scripts/check_types.py` runs mypy on
  `iopaint/**/*.py` changed vs `origin/main` and fails only on errors anchored to newly added lines (trigger: before
  committing backend Python changes; it also gates CI — the `backend` job runs it as "Type check (new code only)",
  see the doc).
- `docs/agents/perf-guards.md` — benchmark designs for the four perf regressions listed under Architecture (trigger:
  before implementing a perf benchmark or touching `scan_models` / schedulers / ControlNet preprocessors /
  `--empty-cache-after-inpaint`).
- `docs/agents/review-prompts.md` — adversarial-review, test-quality-review, and mutation-testing prompt templates
  (trigger: preparing a code review session or judging test suite quality).

## Multi-batch agent tasks

Any task that spans more than one batch/phase must keep a plan file at `<harness-tmp-dir>/<task>-plan.md` (e.g.
`/tmp/opencode/<task>-plan.md`) tracking status, goals, todo, done, open questions, and pitfalls hit. It must **never**
be committed to the repo and must **never** be mentioned in commit messages. Minimal template:

```markdown
# <task> — plan
- Status / goal: <one line each>
- Done / Todo: <bullets>
- Open questions / pitfalls: <bullets> (每项改动的自验证命令与结果附在其下)
```
