"""benchmarks/ 的跑分框架：统一采样、统计、机器指纹、结果文件与同机基线对比。

只依赖标准库（不引入 pytest-benchmark）。用例通过 `measure` / `record` 登记结果，
`benchmarks/conftest.py` 在 session 结束时打印总表，并按需写 JSON / 对比基线。

口径：
- 每个用例先 warmup 再采样 N 轮，报告 median / p90 / min / max / stddev / cv。
  cv（stddev/mean）用来判断这一轮数字是否可信，case 之间不要拿单点比。
- `median` 是主口径；`min` 只用于观察「无干扰时能跑多快」，不作为基线。
- 单位默认 ms；`record` 用来登记非耗时指标（字节数、峰值显存、A/B 比值）。
- 所有结果都带 `direction`：`lower`（越小越好）或 `higher`（越大越好），
  基线对比据此判断回归。

样例：

    from harness import measure, record

    stats = measure("inpaint.cold.12mp", run_cold, rounds=7)
    old = measure("decode.base64", decode_b64)
    new = measure("decode.bytes", decode_bytes)
    record("decode.speedup", old.median / new.median, "x", direction="higher")
"""
from __future__ import annotations

import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

DEFAULT_ROUNDS = 7
DEFAULT_WARMUP = 2

# measure 只对时间做单位换算；其它单位由 record 自行给到目标单位。
_TIME_SCALE = {"ms": 1e3, "s": 1.0, "us": 1e6}

_config: dict[str, Any] = {"rounds": None, "warmup": None, "label": None}
_results: dict[str, dict[str, Any]] = {}


def configure(
    *, rounds: int | None = None, warmup: int | None = None, label: str | None = None
) -> None:
    """由 conftest 调用，把 CLI 覆盖项灌进 harness。"""
    _config.update(rounds=rounds, warmup=warmup, label=label)


@dataclass
class Stats:
    name: str
    unit: str
    n: int
    median: float
    mean: float
    p90: float
    min: float
    max: float
    stddev: float
    cv_pct: float
    direction: str = "lower"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "n": self.n,
            "median": self.median,
            "mean": self.mean,
            "p90": self.p90,
            "min": self.min,
            "max": self.max,
            "stddev": self.stddev,
            "cv_pct": self.cv_pct,
            "direction": self.direction,
            "extra": self.extra,
        }


def _round(value: float, unit: str) -> float:
    # 耗时保留 0.1%（相对精度），字节数/计数保留整数。
    if unit in _TIME_SCALE:
        return round(value, 4)
    return round(value, 2)


def _percentile(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def measure(
    name: str,
    fn: Callable[[], Any],
    *,
    unit: str = "ms",
    warmup: int | None = None,
    rounds: int | None = None,
    sync: Callable[[], None] | None = None,
    direction: str = "lower",
    extra: dict[str, Any] | None = None,
) -> Stats:
    """先跑 `warmup` 次，再采样 `rounds` 轮，登记并返回统计量。

    `sync` 用于 CUDA：每次计时前后调用（通常是 `torch.cuda.synchronize`），
    否则拿到的是 kernel 入队时间而不是执行时间。
    rounds/warmup 的优先级：CLI 覆盖 > 调用参数 > 默认值。
    """
    if _config["rounds"] is not None:
        rounds = _config["rounds"]
    elif rounds is None:
        rounds = DEFAULT_ROUNDS
    if _config["warmup"] is not None:
        warmup = _config["warmup"]
    elif warmup is None:
        warmup = DEFAULT_WARMUP
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        fn()

    scale = _TIME_SCALE.get(unit, 1.0)
    samples: list[float] = []
    for _ in range(rounds):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * scale)

    mean = statistics.fmean(samples)
    stats = Stats(
        name=name,
        unit=unit,
        n=len(samples),
        median=_round(statistics.median(samples), unit),
        mean=_round(mean, unit),
        p90=_round(_percentile(samples, 0.9), unit),
        min=_round(min(samples), unit),
        max=_round(max(samples), unit),
        stddev=_round(statistics.stdev(samples) if len(samples) > 1 else 0.0, unit),
        cv_pct=round(statistics.stdev(samples) / mean * 100, 1)
        if len(samples) > 1 and mean
        else 0.0,
        direction=direction,
        extra=dict(extra or {}),
    )
    _results[name] = stats.as_dict()
    return stats


def record(
    name: str,
    value: float,
    unit: str,
    *,
    direction: str = "lower",
    extra: dict[str, Any] | None = None,
) -> Stats:
    """登记一个非耗时标量（字节数、峰值显存、A/B 比值），n=1。"""
    stats = Stats(
        name=name,
        unit=unit,
        n=1,
        median=_round(value, unit),
        mean=_round(value, unit),
        p90=_round(value, unit),
        min=_round(value, unit),
        max=_round(value, unit),
        stddev=0.0,
        cv_pct=0.0,
        direction=direction,
        extra=dict(extra or {}),
    )
    _results[name] = stats.as_dict()
    return stats


def results() -> dict[str, dict[str, Any]]:
    return _results


def add_extra(name: str, **values: Any) -> None:
    """给已登记的结果补字段（例如采样结束后的峰值显存）。

    `measure` 的 `extra` 参数在计时开始前求值，读不到采样期间的峰值；
    需要这类值就在 `measure` 返回后调用这里。
    """
    if name not in _results:
        raise KeyError(f"unknown benchmark: {name}")
    _results[name]["extra"].update(values)


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gib() -> float | None:
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page * pages / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _gpu_info() -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_mib": round(
                    torch.cuda.get_device_properties(index).total_memory / 2**20
                ),
                "capability": "sm_%d%d"
                % torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    commit = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def machine_info() -> dict[str, Any]:
    """跑分机器指纹：换机器/换版本后别拿旧基线比绝对值。"""
    return {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "cv2": _version("cv2"),
        "torch": _version("torch"),
        "gpus": _gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "iopaint_bench_gpu": os.environ.get("IOPAINT_BENCH_GPU"),
        "git": _git_state(),
    }


def snapshot(label: str | None = None) -> dict[str, Any]:
    """本轮结果的完整 JSON 结构。"""
    return {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label if label is not None else _config["label"],
        "machine": machine_info(),
        "benchmarks": dict(sorted(_results.items())),
    }


def save(path: str | Path, label: str | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(snapshot(label), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_rows(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    max_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, cur: dict[str, Any], base: dict[str, Any] | None) -> None:
        if base is None:
            rows.append(
                {
                    "name": name,
                    "unit": cur.get("unit"),
                    "baseline": None,
                    "current": cur.get("median"),
                    "ratio": None,
                    "verdict": "new",
                }
            )
            return
        if cur.get("unit") != base.get("unit"):
            rows.append(
                {
                    "name": name,
                    "unit": cur.get("unit"),
                    "baseline": base.get("median"),
                    "current": cur.get("median"),
                    "ratio": None,
                    "verdict": "unit mismatch",
                }
            )
            return
        base_value = base.get("median")
        cur_value = cur.get("median")
        if not isinstance(base_value, (int, float)) or base_value == 0:
            ratio = None
        elif isinstance(cur_value, (int, float)):
            ratio = cur_value / base_value
        else:
            ratio = None

        if ratio is None:
            verdict = "?"
        else:
            direction = cur.get("direction", "lower")
            if direction == "higher":
                regressed = ratio * max_ratio < 1
            else:
                regressed = ratio > max_ratio
            verdict = "REGRESSION" if regressed else "ok"
        rows.append(
            {
                "name": name,
                "unit": cur.get("unit"),
                "baseline": base_value,
                "current": cur_value,
                "ratio": ratio,
                "verdict": verdict,
            }
        )

    for name, cur in current.items():
        base = baseline.get(name)
        add(name, cur, base)
        # extra 里的数值（峰值显存、字节数）也按「越小越好」参与对比。
        base_extra = (base or {}).get("extra") or {}
        for key, cur_extra in (cur.get("extra") or {}).items():
            if not isinstance(cur_extra, (int, float)):
                continue
            base_extra_value = base_extra.get(key)
            if not isinstance(base_extra_value, (int, float)) or base_extra_value == 0:
                continue
            add(
                f"{name}/{key}",
                {
                    "unit": "value",
                    "median": cur_extra,
                    "direction": "lower",
                },
                {"unit": "value", "median": base_extra_value},
            )

    for name in baseline:
        if name not in current:
            rows.append(
                {
                    "name": name,
                    "unit": baseline[name].get("unit"),
                    "baseline": baseline[name].get("median"),
                    "current": None,
                    "ratio": None,
                    "verdict": "missing",
                }
            )
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_results(results_map: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'name':<38} {'n':>3} {'median':>10} {'p90':>10} {'min':>10} "
        f"{'max':>10} {'cv%':>6}  unit"
    )
    lines = [header, "-" * len(header)]
    for name, item in results_map.items():
        extra = " ".join(
            f"{key}={_fmt(value)}"
            for key, value in (item.get("extra") or {}).items()
        )
        lines.append(
            f"{name:<38} {item.get('n', 1):>3} {_fmt(item.get('median')):>10} "
            f"{_fmt(item.get('p90')):>10} {_fmt(item.get('min')):>10} "
            f"{_fmt(item.get('max')):>10} {_fmt(item.get('cv_pct')):>6}  "
            f"{item.get('unit', '')}"
            + (f"  {extra}" if extra else "")
        )
    return "\n".join(lines)


def format_comparison(rows: list[dict[str, Any]], max_ratio: float) -> str:
    header = (
        f"{'name':<38} {'baseline':>10} {'current':>10} {'ratio':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        ratio = row["ratio"]
        lines.append(
            f"{row['name']:<38} {_fmt(row['baseline']):>10} "
            f"{_fmt(row['current']):>10} {_fmt(ratio):>7}  {row['verdict']}"
        )
    lines.append(
        f"\n阈值：ratio > {max_ratio:.2f} 记为 REGRESSION（higher 方向取倒数）。"
    )
    return "\n".join(lines)


def compare(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    max_ratio: float = 1.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按名字对比两组 `benchmarks` 字典，返回 (全部行, 回归行)。"""
    rows = _metric_rows(current, baseline, max_ratio)
    regressions = [row for row in rows if row["verdict"] == "REGRESSION"]
    return rows, regressions
