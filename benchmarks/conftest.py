"""benchmarks/ 的 pytest 接线：CLI 覆盖项 + session 结束时的总表/JSON/基线对比。

只在本目录被收集时生效（`pytest benchmarks ...`），不影响 `iopaint/tests` 的 fast suite。
"""
from __future__ import annotations

import pytest

import harness


def pytest_addoption(parser):
    group = parser.getgroup("benchmarks")
    group.addoption(
        "--bench-save",
        metavar="PATH",
        default=None,
        help="把本轮结果写成 JSON（含机器指纹），例如 benchmarks/baselines/xeon-e3.json",
    )
    group.addoption(
        "--bench-compare",
        metavar="PATH",
        default=None,
        help="与本机此前的 JSON 结果对比；任一指标超过 --bench-max-regression 时测试失败",
    )
    group.addoption(
        "--bench-max-regression",
        type=float,
        default=1.5,
        metavar="RATIO",
        help="回归阈值（current/baseline 的中位数比值，默认 1.5 = 慢 50%%）",
    )
    group.addoption(
        "--bench-rounds",
        type=int,
        default=None,
        metavar="N",
        help="覆盖所有用例的采样轮数（默认各用例自带，通常 7）",
    )
    group.addoption(
        "--bench-warmup",
        type=int,
        default=None,
        metavar="N",
        help="覆盖所有用例的 warmup 次数（默认 2）",
    )
    group.addoption(
        "--bench-label",
        default=None,
        metavar="TEXT",
        help="写进 JSON 的自由文本，例如改动说明或 commit",
    )


def pytest_configure(config):
    if config.getoption("bench_rounds") is not None and config.getoption("bench_rounds") < 1:
        raise pytest.UsageError("--bench-rounds must be >= 1")
    if config.getoption("bench_warmup") is not None and config.getoption("bench_warmup") < 0:
        raise pytest.UsageError("--bench-warmup must be >= 0")
    harness.configure(
        rounds=config.getoption("bench_rounds"),
        warmup=config.getoption("bench_warmup"),
        label=config.getoption("bench_label"),
    )


def _write(session, text: str) -> None:
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(text)
    else:  # pragma: no cover - pytest always provides terminalreporter
        print(text)


def pytest_sessionfinish(session, exitstatus):
    collected = harness.results()
    if not collected:
        return

    _write(session, "\n=== benchmarks ===")
    _write(session, harness.format_results(collected))

    save_path = session.config.getoption("bench_save")
    if save_path:
        path = harness.save(save_path)
        _write(session, f"\n结果已写入 {path}")

    compare_path = session.config.getoption("bench_compare")
    if not compare_path:
        return

    max_ratio = session.config.getoption("bench_max_regression")
    baseline = harness.load(compare_path)
    rows, regressions = harness.compare(
        collected, baseline.get("benchmarks", {}), max_ratio=max_ratio
    )

    machine = baseline.get("machine") or {}
    label = baseline.get("label")
    cpu = machine.get("cpu", "?")
    plat = machine.get("platform", "?")
    py = machine.get("python", "?")
    torch = machine.get("torch", "?")
    _write(session, f"\n=== 基线对比：{compare_path} ===")
    _write(
        session,
        f"基线：{cpu} / {plat} / python {py} / torch {torch}"
        + (f" / {label}" if label else ""),
    )
    _write(session, harness.format_comparison(rows, max_ratio))

    if regressions:
        _write(session, f"\n!! {len(regressions)} 项回归超过阈值 {max_ratio:.2f}")
        if session.exitstatus == 0:
            session.exitstatus = 1
