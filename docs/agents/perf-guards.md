# Perf guard benchmark designs (design only — nothing implemented yet)

The four perf regressions listed under **Architecture** in AGENTS.md each get a minimal benchmark design here. None is
implemented: a human picks which one to land first. Every design names: what to measure, the hook point, how to set
the threshold, and whether it belongs in the fast suite or the slow/local set. A new agent should be able to implement
one from this doc alone.

General rules for all four:

- Time with `time.perf_counter`, report the median of N runs (min 5), never a single sample.
- No new dependencies: no pytest-benchmark; plain pytest + stdlib timing is enough.
- Thresholds are set from a measured baseline on the target machine, then encoded as `baseline * 1.5` (50% headroom);
  record the measured baseline numbers in the test file's comment with the re-runnable command (house rule: every
  claim is measured).

## 1. `scan_models()` must not run per request (fast)

- **What**: `GET /api/v1/server-config` must be served from the cached `ModelManager.get_available_models()`
  (`iopaint/model_manager.py`), not by re-scanning the model dir (a scan costs ~2s).
- **How**: in-process `TestClient` (same pattern as `iopaint/tests/test_api.py`, no port bound). Monkeypatch
  `iopaint.model_manager.ModelManager.scan_models` with a counting wrapper, then hit `/api/v1/server-config` 3 times;
  assert the counter does not increase (cache populated once at startup). Timing variant: assert wall time of the 3
  calls < 1.5s; the counter assertion is the primary one (timing is machine-dependent noise, the call count is not).
- **Threshold**: call count == 0 after startup; wall-time guard as a loose backstop.
- **Suite**: fast (`pytest -m "not slow"`; no weights — use `--model cv2` via the existing test fixtures).

## 2. Diffusion schedulers must not be rebuilt per call (slow)

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

- **What**: `empty_cache_after_inpaint` (`iopaint/schema.py:273`, used at `iopaint/api.py:463/499/533`) defaults to
  `False`, and when off the request path must not call `torch.cuda.empty_cache` at all.
- **How**: two assertions in one fast test: (a) `ApiConfig().empty_cache_after_inpaint is False` — the default-drift
  guard; (b) monkeypatch `torch.cuda.empty_cache` with a raising stub, run one cv2 inpaint through the in-process
  `TestClient` with the default config, and assert no call happened. No GPU needed (cv2 model, cpu).
- **Threshold**: boolean / call-count.
- **Suite**: fast.

## Pick order (suggestion for the human)

1 wins the most (it is the only one of the four that already has a known ~2s cost attached and is fully fast/hermetic);
4 is the cheapest to land; 2 is fast if the stubbing stays clean; 3 is the most valuable regression-wise but needs the
GPU box, so it goes last.
