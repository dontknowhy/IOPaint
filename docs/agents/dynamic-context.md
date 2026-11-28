# Dynamic context strategy

`--hd-strategy Dynamic` (`HDStrategy.DYNAMIC`) is a third crop strategy beside
`Original` / `Resize` / `Crop`. Crop uses one fixed margin — the web UI and
`web_app/src/lib/const.ts` hard-code `Crop` with `hd_strategy_crop_margin=128`.
Dynamic keeps the per-contour crop, but decides each contour's margin from the
mask geometry plus what the pixels around the hole look like, so a small mask
on a flat wall does not pay for a large window and a textured region gets more
context than 128px.

Trigger: touching `iopaint/model/helper/dynamic_context.py`,
`InpaintModel._dynamic_context_call` in `iopaint/model/base.py`, the
`DynamicContextConfig` fields in `iopaint/schema.py`, or running the A/B script.

## Pipeline

`InpaintModel.__call__` dispatches `HDStrategy.DYNAMIC` to
`_dynamic_context_call` before the CROP/RESIZE branches (both `fixed_size`
models keep their own path). For every box from `boxes_from_mask(mask)`:

1. `decide_context(image, mask, box, cfg)`:
   - `force_pad` short-circuits the whole decision (debug / A-B only).
   - `geometry_pad`: `pad = base_pad`; area ratio > `area_ratio_mid` → `pad *
     area_pad_scale`; > `area_ratio_high` → at least `area_high_pad`.
   - Complexity is skipped when the geometry pad already reaches
     `skip_complexity_at_pad` or `enable_complexity=False`.
   - Otherwise a local patch (`box` grown by `2*ring_width`, longest side capped
     at `thumb_size`) gets a ring = `dilate(mask) - mask`, and four metrics:
     - `entropy`: per-`block_size` histogram entropy of ring pixels (mean, p75);
     - `coherence`: structure-tensor `(λ1-λ2)/(λ1+λ2)` averaged on the ring;
     - `grad_mag`: Sobel magnitude on the ring;
     - `match_dist`: each boundary query block's nearest same-size block inside
       the ring, normalized SSD ÷ 255², median over queries — low means the
       surroundings repeat, so the model has plenty of usable context.
   - `complexity = w_entropy*Hn + w_match*match + w_interaction*interaction`,
     each term clipped to 0..1 by `entropy_ref` / `match_dist_ref` / `grad_ref`.
     Below `complexity_low` keep `base_pad`, above `complexity_high` take
     `max(base_pad, high_pad)`, in between `base_pad * (1 + alpha*c)`.
   - `pad = max(geometry, dynamic)`, clamped to `[min_pad, max_pad]`; the rect is
     `crop_rect(bbox, pad, mask.shape)` (centered where possible, shifted at the
     border, asymmetric allowed). Flags record `forced_pad`,
     `complexity_skipped`, `ring_empty`, `pad_capped`, `edge_clamped`,
     `downscaled`.
2. `_dynamic_region_forward`: if the window's longest side exceeds
   `max_input_side` (2048), resize to that limit, infer, then `INTER_CUBIC`
   back to the window. Whole-image inference is never used for a dynamic crop.
3. `seam_score` on the result: mean Sobel magnitude in a `seam_band` stripe just
   inside the mask boundary divided by the mean in the neighbouring ring. If
   `score > seam_ratio` **and** the band gradient is at least `seam_min_grad`
   (10.0), the crop is re-run once with `pad*1.5` (still ≤ `max_pad`) and the
   score is re-measured. The ratio alone fires on flat backgrounds where the
   ring gradient is ~0; the absolute floor is what keeps that from wasting a
   forward pass.
4. `_paste_mask_region` copies back only the mask plus `paste_feather` (3px,
   Gaussian) — outside that region every pixel equals the input, so a dynamic
   run never re-encodes untouched image content.

Each contour logs one INFO line:

```
dynamic_context {"bbox": [...], "rect": [...], "pad": 142, "complexity": 0.216,
 "area_ratio": ..., "entropy": ..., "match_dist": ..., "interaction": ...,
 "flags": [...], "seam_score": 1.28, "seam_band_grad": ..., "seam_ring_grad": ...,
 "seam_rerun": false, "seam_failed": false}
```

`ContextDecision.to_dict()` is the full field list; the line is JSON so a
benchmark or a log reader can parse it without importing the module.

## Configuration

All knobs live in `DynamicContextConfig` (`iopaint/schema.py`), default-off
everywhere except when the strategy is selected. `ApiConfig.dynamic_context` is
`exclude=True`, so the config dumped at startup stays flat and older
`typer-config` wrappers keep working; `--dynamic-context path.json` loads a
partial JSON (`DynamicContextConfig(**data)`, unknown keys error out). The same
fields can travel in a `/api/v1/inpaint` request (`InpaintRequest.dynamic_context`)
or in a batch `--config` JSON. `Api._inpaint_blocking` copies the server config
onto each request, so `--dynamic-context` wins over request-level defaults.

Defaults are calibrated on `~/下载/sample`: flat backgrounds score 0.04–0.05,
regular texture 0.10–0.21, complex irregular texture 0.33–0.46. That is why
`complexity_low=0.2`, `complexity_high=0.45` and `match_dist_ref=0.002`; the
comment next to each field records the measured band.

## A/B verification

`benchmarks/dynamic_context_ab.py` is standalone (not collected by pytest, not in
CI). It pairs `image + *_mask.png (+ *_cleanup.png)` in a directory, prints each
contour decision, runs Dynamic plus fixed `--pads`, and writes JSON/markdown:

```bash
conda run -n iopaint python benchmarks/dynamic_context_ab.py \
  --sample-dir ~/下载/sample --device cuda \
  --json /tmp/opencode/ab.json --markdown /tmp/opencode/ab.md
```

`*_cleanup.png` in the sample set is the previous Crop `margin=128` output (its
pixels outside the mask are identical to the source), so PSNR against it only
measures similarity to that reference — a larger pad legitimately scores lower.
Read mask MAE, `seam_score`, and wall time together; do not treat the cleanup as
ground truth.

Measured on the 10 sample pairs (P104-100, LaMa, 3 rounds, outside-mask MAE 0.0
for every row):

- Dynamic picks 128/142/154/256 for complexity 0.04–0.46 and matches `pad128`
  bit-for-bit wherever the decision is 128.
- `pad64` is consistently worse inside the mask (MAE 0.66 vs 0.51, 1.62 vs 0.85,
  1.17 vs 1.0) — that is the case the geometry rule is meant to avoid.
- Seam reruns drop to zero at the default floor; only one `pad64` contour of the
  5790×3860 image fires (ratio 1.57) and its rerun reports `seam_failed=false`.
- Time tracks window area: dynamic 0.12–0.78 s, pad64 0.07–0.57 s, pad128
  0.09–0.63 s, pad256 0.22–1.31 s. The low end of the dynamic range is the same
  cost as Crop; the high end is the price of a bigger window on a complex mask.
  TorchScript's profiling executor is slow for the first two forwards of a graph
  (up to 1.8 s here), so the script discards them (`time_first` keeps the number).

## Pitfalls

- Model contract is RGB in / BGR out (`InpaintModel.__call__`); a harness that
  feeds cv2 BGR blobs sees PSNR collapse and non-zero outside-mask MAE.
- The pasted region is mask ± `paste_feather`, so assertions on synthesized
  output must check interior pixels, not every pixel of the rectangle.
- `seam_score` divides by the ring gradient; keep `seam_min_grad` unless a new
  set of samples shows real seams in flat areas.
- `max_input_side` trades detail for speed; the resize-back uses `INTER_CUBIC`,
  so a maxed-out window is not bit-identical to the un-downscaled run.
