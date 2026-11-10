# Perf guard benchmark designs (guard #1 implemented — local `python benchmarks/run.py`, not CI)

The four perf regressions listed under **Architecture** in AGENTS.md each get a minimal benchmark design here.
**#1 is implemented**, in `benchmarks/test_server_config_scan_cache.py`; 2–4 remain design only, waiting for a human
to pick. Every design names: what to measure, the hook point, how to set the threshold, and whether it belongs in the
fast suite or the slow/local set. A new agent should be able to implement one from this doc alone.

General rules for all four:

- All timing goes through `measure` in `benchmarks/harness.py` (warmup + N rounds, median/p90/stddev/cv); do not
  write your own `perf_counter` loop in a case. Use `record` / `add_extra` for non-time scalars (bytes, peak VRAM,
  A/B ratios).
- No new dependencies: no pytest-benchmark; plain pytest + stdlib timing is enough.
- Regression gating belongs to a same-machine baseline: `python benchmarks/run.py --bench-save <json>` stores one;
  after a change, `--bench-compare <json>` compares ratios, default `--bench-max-regression 1.5` (50% slower).
  Never hard-code a machine's millisecond constant in a case; keep measured numbers in the JSON or a comment as
  reference.
- The call/count assertion is the guard; the timing number is documentation with a loose backstop — wall time is
  machine-dependent noise, a call count is not.

## 1. `scan_models()` must not run per request — **implemented**

- **What**: `GET /api/v1/server-config` must be served from the cached `ModelManager.get_available_models()`
  (`iopaint/model_manager.py`), not by re-scanning the model dir (a scan costs ~2s).
- **How**: in-process `TestClient` (same pattern as `iopaint/tests/test_api.py`, no port bound). Monkeypatch
  `iopaint.model_manager.ModelManager.scan_models` with a counting wrapper, then hit `/api/v1/server-config` 3 times;
  assert the counter does not increase (cache is filled by `ModelManager.__init__`, before any request). A second
  test records the median of 5 rounds × 3 calls through `measure`; compare it against a stored baseline, never
  against a fixed millisecond constant. The counter assertion is the primary one.
- **Threshold**: call count == 0 after startup; the latency number is recorded, not asserted.
- **Suite**: fast (`--model cv2`, no weights) — but it lives **outside** CI, see below.

**Landed** as `benchmarks/test_server_config_scan_cache.py`; run it with

```bash
python benchmarks/run.py -k server_config      # 2 tests, ~5s
```

- Location is deliberate: `pytest.ini` has `testpaths = iopaint/tests`, so plain `pytest`,
  `pytest -m "not slow"` (CI and `scripts/check.sh`) never collect `benchmarks/` — the guard does not gate anything
  until someone decides to promote it. `bench` is registered as a marker in `pytest.ini` so `-m bench` is warning-free.
- Two tests: `test_server_config_is_served_from_cache` (primary — counting wrapper installed *after* `Api(...)`
  construction, 0 `scan_models` calls across 3 requests, plus a non-empty `modelInfos` sanity check) and
  `test_server_config_latency_recorded` (median of 5 rounds × 3 calls through `measure`, no hardcoded threshold).
- **Measured reference** (2026-11-09, Xeon E3-1231 v3, `--model cv2` on cpu): `server_config.3x` median
  **5.5–5.7 ms** across two processes. This is a recorded number, not a gate.
- **Proven to bite**: pointing `api_server_config` at `self.model_manager.scan_models()` instead of
  `get_available_models()` trips the call-count assertion immediately (previously measured at 12.1 ms per 3 calls,
  so a stored baseline + `--bench-compare` flags the latency as well). Reverting restores 2 passed.

## 2. Diffusion schedulers must not be rebuilt per call (fast if stubbed)

- **What**: `DiffusionInpaintModel.set_scheduler` (`iopaint/model/base.py:408`) must reuse `self._scheduler_cache`
  keyed by `(sd_sampler, sd_lcm_lora)`; a rebuild per inpaint request is the regression.
- **How**: unit-level, no HTTP. Construct (or fake) a model with a stub `self.model.scheduler.config` and a stub
  `get_scheduler`; call `set_scheduler` twice with the same `(sd_sampler, lcm_lora)` and assert the stub factory ran
  once (counting spy). Then vary the sampler and assert the factory runs again. This tests the cache logic without any
  weights, so it can actually live in the **fast** suite if the stubbing stays hermetic; if you end up needing a real
  pipeline, move it to slow.
- **Threshold**: factory call count, not time.
- **Suite**: fast if fully stubbed (preferred); slow only if a real scheduler object is required.

## 3. ControlNet preprocessors must not reload per request (slow)

- **What**: `_get_openpose_processor()` / `_get_midas_processor()` (`iopaint/model/helper/controlnet_preprocess.py`)
  are lazy singletons stored in module globals; reloading them per request wastes seconds and VRAM.
- **How**: on a GPU box, time two consecutive `make_openpose_control_image(image)` calls on the same small image
  (e.g. 512x512). First call includes model load; the second must be much faster. Assert `t2 < t1 / 2` as the shape
  check, and additionally monkeypatch/count `MidasDetector.from_pretrained` (and the openpose equivalent) to assert
  exactly one load across both calls. The load-count assertion is the real guard; the timing is documentation.
- **Threshold**: exactly 1 `from_pretrained` call across 2 invocations.
- **Suite**: slow (downloads the Annotators weights on first run; needs the machine described in
  `docs/agents/slow-tests.md`).

## 4. `--empty-cache-after-inpaint` stays default-off and off the hot path (fast)

- **What**: `empty_cache_after_inpaint` (`iopaint/schema.py:278`, used at `iopaint/api.py:618/804/844`) defaults to
  `False`, and when off the request path must not call `torch.cuda.empty_cache` at all.
- **How**: two assertions in one fast test: (a) `ApiConfig().empty_cache_after_inpaint is False` — the default-drift
  guard; (b) monkeypatch `torch.cuda.empty_cache` with a raising stub, run one cv2 inpaint through the in-process
  `TestClient` with the default config, and assert no call happened. No GPU needed (cv2 model, cpu).
- **Threshold**: boolean / call-count.
- **Suite**: fast.

## 5. Inference latency benchmark — `benchmarks/test_inpaint_latency.py` **implemented**

- **What**: median/p90/min/max and peak VRAM for one LaMa forward (256/512/1024) and one SD1.5 inpainting UNet step
  (512 output, batch=2, latent 64×64). Used to judge whether `torch.compile`/AOTInductor is worth it, and for
  before/after regression comparisons.
- **How**: `ModelManager(name="lama", device=cuda:0)` goes through the real load path; the UNet is read from the
  local HF cache (`local_files_only=True`). Missing weights or no CUDA → `pytest.skip`; nothing is downloaded.
- **Which GPU**: defaults to `cuda:0`; `IOPAINT_BENCH_GPU=N` picks another card. Every row reports
  `name cuda:N <MiB> sm_XY`, because `cuda:0` is not necessarily the fast card (and the number is torch's
  enumeration, not `nvidia-smi`'s). The UNet case needs ≥5120 MiB; smaller cards skip with a reason instead of OOM.
- **Run**: `IOPAINT_BENCH_GPU=N python benchmarks/run.py -k "lama or sd15"` (outside `testpaths`; CI never collects it).
- **Suite**: local/slow; numbers only, no universal threshold. For a regression on a specific machine, use
  `--bench-save` for the baseline and `--bench-compare` for the ratio (`--bench-max-regression 1.5`).
- **Measured reference** (2026-11-09, `cuda:0` = NVIDIA P104-100 8109MiB sm_61, torch 2.14.1+cu126, fp32):
  - LaMa end to end (host↔GPU copies and numpy conversion included): 256×256 **52.7 ms**, 512×512 **143.0 ms**,
    1024×1024 **537.3 ms** (peak VRAM allocated/reserved 230/262, 330/456, 729/1486 MiB).
  - SD1.5 UNet step **411.6 ms** (fp32, batch=2, latent 64×64, peak VRAM 3875/4278 MiB).
  - Historical A/B (different card): `cudnn.benchmark=True` helped neither; `channels_last` slowed the UNet to
    449.4 ms.

## Pick order (suggestion for the human)

**#1 is already landed** (see its section). Of the remaining three: 4 is the cheapest to land; 2 is fast if the
stubbing stays clean; 3 is the most valuable regression-wise but needs the GPU box, so it goes last — it is also the
only one of the four with no known cost attached today.

## 6. Request CPU path: no duplicated decode/encode/commit work — **implemented**

Four points are guarded together: structural guards live in `iopaint/tests/test_webui_perf.py` (fast suite),
numbers in `benchmarks/test_inpaint_cpu_path.py` (`python benchmarks/run.py`, not CI).

- **Uploaded bytes are not base64 and are decoded once**: `/inpaint` and `/export` are multipart; `image`/`mask` are
  file parts, and the handler decodes the raw bytes once (`_decode_uploaded_image`). One base64 decode of a 12MP PNG
  costs 39 ms, JSON parsing 16 ms, and 33% more bytes. Guard: `test_inpaint_decodes_uploaded_bytes_once` counts
  `_decode_uploaded_image` calls — exactly 1 for a cold inpaint, 0 for a revision round. A file part over
  `MAX_UPLOAD_BYTES` (48MB) → 413; non-image bytes → 400.
- **Encoding does not round-trip through PIL**: `pil_to_bytes` now accepts a uint8 RGB/RGBA ndarray; the PNG path
  goes straight to `png_bytes_from_numpy`, dropping the `Image.fromarray` (11 ms) + `np.asarray` (21 ms) full-image
  round trip. Output bytes are byte-identical to passing a PIL image (asserted in the benchmark). Guard:
  `test_png_response_skips_pil_roundtrip` turns `Image.fromarray` into a bomb and still requires a 200 PNG response.
- **Do not hash the source on a revision hit**: `sha256_bytes(12MP PNG) = 27 ms` is only needed for the
  `begin_by_hash` fallback. Guard: `test_revision_hit_skips_source_hash` — 2 calls on the first round
  (seed + response alias), only the alias call on a resend with `revision`.
- **Commit compares only the rectangle that can change**: `revision_cache._diff_hinted` takes the caller's `region`
  (mask bbox + `hd_strategy_crop_margin`, plus the crop rect when a cropper is used), compares pixels inside that
  box, and byte-checks the rest with `tobytes()`; if the outside check fails it falls back to the full frame
  (12MP 158 ms → ~25 ms). **Correctness does not depend on the hint being accurate**: a mismatch outside the box
  falls back. Guard: `iopaint/tests/test_revision_cache_diff.py` (random / out-of-bounds / too-small hint paths match
  the full-frame result pixel for pixel; end-to-end "hint vs forced None" export bytes are equal).
- **Preview alpha compositing**: `_display_preview_jpeg` skips the full-resolution float32 composite when
  `(alpha == 255).all()` (25MP 532 ms → 123 ms, bit-identical result). Guards:
  `test_display_preview_skips_composite_for_opaque_alpha` (`np.clip` bomb + byte comparison with the no-alpha
  result) and `test_display_preview_composites_translucent_alpha`.
- **PNG compression strategy follows the shape**: `png_bytes_from_numpy` uses `zlib.Z_RLE` for RGB at the default
  level (≤1) — it only emits distance-1 matches, exactly what the Up-filtered runs of repeated bytes need; RGBA and
  level 6/9 keep the default strategy. RGBA's alpha plane becomes a run of zeros after Up filtering, so once
  interleaved with RGB only distance-4 matches help (measured: +52% size, 14% slower), and Z_RLE ignores the level
  (level 6 and level 1 produce the same size), so putting it in the small-file tier would only make files bigger.
  Guard: `test_compress_strategy_follows_shape` inspects the `strategy` argument to `zlib.compressobj`
  (RGB→`Z_RLE`, RGBA→default, level 6→default).
- **Measured (2026-11-01, Xeon E3-1231 v3 / 8 threads, 12MP PNG, cv2 model, fixed-cost fake model)**: the same
  benchmark alternated before/after three rounds each,
  `python -m pytest benchmarks/test_inpaint_cpu_path.py -m bench -s -k "cold or warm or export"`:
  cold **988.2 → 834.1 → 793.0 → 696.4 ms** (−29.5%), warm **827.7 → 675.3 → 620.8 → 617.8 ms** (−25.4%),
  export **566.5 → 536.2 → 472.2 → 473.2 ms** (−16.5%); single-point A/B: base64 325.0 vs 284.8 ms (−40 ms),
  PNG encoding 542.5 vs 519.7 ms (−23 ms), preview compositing 532.7 vs 123.4 ms (−409 ms), PNG strategy
  539.3 vs 462.7 ms (−77 ms, same encoder, only the zlib strategy changed; size 13.80 → 15.49 MB). The multipart
  step only affects the cold request (a warm request never uploads an image): 793.0 → 696.4 ms (−96 ms: base64
  46 ms + JSON 21 ms + transfer copies), and uploaded bytes drop 33% (17.3MB → 13.0MB).

  **Current reference values (2026-11-09, same machine, `python benchmarks/run.py -k "not lama and not sd15"`)**:
  cold 256–285 ms, warm 183–212 ms, export 466–469 ms. The warm round now takes the full-resolution JPEG preview
  path for opaque PNGs in the 4–16Mi range, which is not the PNG-response path behind the 617.8 ms above — do not
  compare them directly.
