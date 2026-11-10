# benchmarks

本地跑分，**不进 CI**（`pytest.ini` 的 `testpaths = iopaint/tests`，`scripts/check.sh` 也不跑这里）。
数字随机器、驱动、负载变化，绑进 CI 只会让别人的机器莫名其妙变红；这里是「自己量、自己比」的工具箱。

跑分和守卫是两件事：

- **守卫**（调用次数、位级等价）在 `iopaint/tests/` 的 fast suite 里，断言与机器无关。
- **跑分**（墙钟、峰值显存、字节数）在这里记录数字；回归判断靠同一台机器上的
  `--bench-compare` 比值，不写死某台机器的毫秒阈值。

## 快速开始

```bash
python benchmarks/run.py                                  # 全部（等价于 pytest benchmarks -m bench）
python benchmarks/run.py -k "cold or warm or export"      # 只跑匹配的用例
IOPAINT_BENCH_GPU=0 python benchmarks/run.py -k "lama or sd15"
python benchmarks/run.py --bench-rounds 21                # 覆盖采样轮数，数字更稳、更慢
```

session 结束时会打印总表（median / p90 / min / max / cv%），`-s` 不是必需的。
只收集 `-m bench` 标记的用例；`benchmarks/test_inpaint_latency.py` 同时带 `slow` 标记。

## 基线对比

```bash
# 1. 在当前机器上存一份基线（JSON 带机器指纹、git commit、依赖版本）
python benchmarks/run.py --bench-save benchmarks/baselines/local.json

# 2. 改完代码后，用同一条命令对比；任一指标超过阈值就以非 0 退出
python benchmarks/run.py --bench-compare benchmarks/baselines/local.json

# 3. 阈值默认 1.5（慢 50%）；要更严就显式传
python benchmarks/run.py --bench-compare benchmarks/baselines/local.json \
    --bench-max-regression 1.25
```

- **只在同一台机器、同一套环境里比**。JSON 里的 `machine`（CPU/内存/GPU/Python/numpy/cv2/torch/
  git commit）就是给这件事用的；换机器请重新存基线。
- 建议关掉其他占 CPU/GPU 的任务，并用 `--bench-rounds 21` 跑正式对比；日常冒烟用默认 7 轮。
- `--bench-compare` 按 benchmark 名字匹配；这次没跑到的用例标 `missing`，不算回归。
- `extra` 里的数值（峰值显存、字节数）也参与对比，方向都是越小越好；`*.speedup` 这类比值
  方向是越大越好（`direction="higher"`）。
- `benchmarks/baselines/*.json` 默认不提交：它属于某台机器。要留档就把文件贴进 PR 或 issue。

## 用例

| 文件 | 内容 |
| --- | --- |
| `test_inpaint_cpu_path.py` | `/api/v1/inpaint` 与 `/api/v1/export` 的纯 CPU 段；假模型把推理摘成固定开销 |
| `test_inpaint_latency.py` | 真模型推理：LaMa 256/512/1024、SD1.5 inpainting UNet 单步；带峰值显存 |
| `test_server_config_scan_cache.py` | `/server-config` 不许按请求重扫模型目录（调用计数守卫）+ 耗时记录 |

`test_inpaint_cpu_path.py` 同时跑单项 A/B（`decode.*.speedup`、`png.*.speedup`），
两侧在同一个进程里先后测，比值与机器无关。12MP 不透明 PNG 现在落在
4–16Mi 的显示预览区间，warm 轮编码的是全分辨率 JPEG；canonical 像素仍在
`RevisionCache`，`export.png.12mp` 才走 PNG 重编码。

### 参考值（2026-11-09，Xeon E3-1231 v3 / 8 线程 / 31GiB，Python 3.12.15，numpy 2.5.4，cv2 5.0.0）

CPU（两次独立进程的区间，仅作数量级参考）：

| case | median |
| --- | --- |
| `inpaint.cold.12mp` | 256–285 ms |
| `inpaint.warm.12mp` | 183–212 ms |
| `export.png.12mp` | 466–469 ms |
| `decode.12mp.base64` / `bytes` | 332–334 / 291–293 ms（×1.13–1.14） |
| `png.encode.12mp.pil_roundtrip` / `ndarray` | 474–484 / 450–458 ms（×1.02–1.06） |
| `png.zlib.12mp.default` / `rle` | 541–543 / 452–459 ms（×1.18–1.20，体积 13.80 / 15.49 MB） |
| `preview.25mp.opaque_alpha` | 127–128 ms |
| `server_config.3x` | 5.5–5.7 ms |

GPU（`IOPAINT_BENCH_GPU=0` → `NVIDIA P104-100 cuda:0 8109MiB sm_61`，torch 2.14.1+cu126）：

| case | median | 峰值显存（allocated / reserved） |
| --- | --- | --- |
| `lama.256` | 52.7 ms | 230 / 262 MiB |
| `lama.512` | 143.0 ms | 330 / 456 MiB |
| `lama.1024` | 537.3 ms | 729 / 1486 MiB |
| `sd15.unet.step`（fp32，batch=2） | 411.6 ms | 3875 / 4278 MiB |

这些数字只说明当前实现的大致量级；做回归请用 `--bench-save` / `--bench-compare` 在同一台机器上比。

## 框架

`benchmarks/harness.py`（只依赖标准库）：

- `measure(name, fn, warmup=2, rounds=7, sync=...)`：先 warmup 再采样，登记
  median / mean / p90 / min / max / stddev / cv，返回 `Stats`。
- `record(name, value, unit, direction=...)`：登记非耗时标量（字节数、比值）。
- `add_extra(name, ...)`：给已登记的结果补采样结束后才知道的值（峰值显存）。
- `machine_info()`：CPU/内存/GPU/Python/numpy/cv2/torch/git commit。
- `save` / `load` / `compare`：JSON 基线；对比按名字匹配，方向由 `direction` 决定。

CLI 选项（`benchmarks/conftest.py`）：`--bench-save PATH`、`--bench-compare PATH`、
`--bench-max-regression RATIO`（默认 1.5）、`--bench-rounds N`、`--bench-warmup N`、
`--bench-label TEXT`。用例自己的 rounds/warmup 会被 CLI 覆盖。

新增用例时按 `类别.形状.场景` 命名（例如 `inpaint.cold.12mp`），耗时用 `measure`，
非耗时用 `record`；不要自己写 `time.perf_counter` 循环，也不要写 `assert ms > 0` 这种空断言。
机器相关的阈值只放在 `--bench-compare` 里，用例本身只断言正确性与调用次数。

## 注意

- **GPU 编号是 torch 的 `cuda:N`**（受 `CUDA_DEVICE_ORDER` 影响，默认 `FASTEST_FIRST`），
  不一定等于 `nvidia-smi` 的 index。结果里会打印卡名/显存/算力；用
  `IOPAINT_BENCH_GPU=N` 换卡，别把慢卡或小显存卡的数字当成整机结果。
- 同一进程里的用例会互相影响（内存分配器、文件页缓存、GPU 时钟）。对比时保持用例集合、
  顺序和轮数一致；要更干净就在前后各跑一轮取稳定值。
- `cv%`（stddev/mean）超过 ~5% 时，先怀疑负载/降频，再怀疑代码。
