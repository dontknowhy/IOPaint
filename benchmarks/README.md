# benchmarks

本地性能基准，**不进 CI**（`pytest.ini` 的 `testpaths = iopaint/tests`，`scripts/check.sh` 也不跑这里）。
阈值来自某台机器的实测，绑进 CI 只会让别人的机器莫名其妙变红；这里是"自己量、自己比"的工具箱。

```bash
python -m pytest benchmarks -m bench -s              # 全部（-s 才会打印中位数）
python -m pytest benchmarks/test_inpaint_cpu_path.py -m bench -s
python -m pytest benchmarks -m bench -s -k "cold or warm or export"
```

- `test_inpaint_cpu_path.py` — `/api/v1/inpaint` 与 `/api/v1/export` 的纯 CPU 段（假模型把推理摘成固定开销）。
  文件头的 `BASELINE_*` 是最近一次实测中位数，注释里记着历史值；改动前后用**同一台机器、同一条命令**
  各跑几轮比比值，别跨机器比绝对值。单项 A/B 用例（`_vs_`、`_skips_`）在同一个进程里对照新老实现。
- `test_inpaint_latency.py` — 真模型推理延迟（`IOPAINT_BENCH_GPU` 选卡，LaMa / SD1.5）。
- `test_server_config_scan_cache.py` — `/server-config` 不许每次请求重扫模型目录（调用计数守卫）。

结构守卫（调用次数、位级等价）不在这里，它们在 `iopaint/tests/` 的 fast suite 里：墙钟时间是噪音，
调用计数才是能长期复现的断言。设计说明见 `docs/agents/perf-guards.md`。
