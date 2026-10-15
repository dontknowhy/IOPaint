"""Perf guard #1：`GET /api/v1/server-config` 必须由缓存的
`ModelManager.get_available_models()` 服务，绝不能按请求重扫模型目录。

设计见 `docs/agents/perf-guards.md` §1。本地跑，**不进 CI、不进 fast suite**：

    python -m pytest benchmarks -m bench

（`pytest.ini` 的 `testpaths = iopaint/tests`，所以 CI 与 `scripts/check.sh` 里的
`pytest -m "not slow"` 根本不会收集本目录。）

主断言是"重扫次数"（与机器无关），耗时只是兜底（机器相关，噪声大）。
"""
import statistics
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.api import Api
from iopaint.model_manager import ModelManager
from iopaint.tests.test_api import _minimal_config

pytestmark = pytest.mark.bench

ROUNDS = 5
CALLS_PER_ROUND = 3

# 耗时兜底阈值 = 实测基线 × 1.5（50% 余量）。基线：2026-10、本机 8 核、--model cv2 cpu，
# `ROUNDS=5 × CALLS_PER_ROUND=3` 一轮的中位数连跑 10 次落在 5.1–5.6 ms，取最大 5.6 ms。
# 复跑命令： python -m pytest benchmarks -m bench -s
BASELINE_MEDIAN_S = 5.6e-3
THRESHOLD_S = BASELINE_MEDIAN_S * 1.5  # 8.4 ms


def _client(tmp_path):
    cfg = _minimal_config(tmp_path)
    app = FastAPI()
    api = Api(app, cfg)  # 构造期那次 scan_models 已经把 available_models 填好
    return api, TestClient(app)


def _count_scans(api, monkeypatch):
    """把 scan_models 换成计数包装；构造已完成，此后的调用都算"按请求重扫"。"""
    calls = []
    original = ModelManager.scan_models

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(ModelManager, "scan_models", counting)
    return calls


def test_server_config_is_served_from_cache(tmp_path, monkeypatch):
    """主断言：连续 3 次请求 0 次重扫（`scan_models` 单次 ~2s，见 AGENTS.md）。"""
    api, client = _client(tmp_path)
    calls = _count_scans(api, monkeypatch)

    for _ in range(CALLS_PER_ROUND):
        resp = client.get("/api/v1/server-config")
        assert resp.status_code == 200
        # 确认真的回了缓存里的模型列表，而不是空壳响应
        assert resp.json()["modelInfos"]

    assert calls == [], f"scan_models 被按请求调用了 {len(calls)} 次"


def test_server_config_latency_backstop(tmp_path, monkeypatch):
    """耗时兜底：中位数超过基线×1.5 就报警（真断言是上面的调用次数）。"""
    api, client = _client(tmp_path)
    calls = _count_scans(api, monkeypatch)

    rounds = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(CALLS_PER_ROUND):
            assert client.get("/api/v1/server-config").status_code == 200
        rounds.append(time.perf_counter() - start)

    median = statistics.median(rounds)
    print(
        f"\nserver-config {CALLS_PER_ROUND} 次 × {ROUNDS} 轮："
        f"中位数 {median * 1000:.1f} ms（阈值 {THRESHOLD_S * 1000:.1f} ms）"
    )
    assert calls == []
    assert median < THRESHOLD_S, (
        f"中位数 {median * 1000:.1f} ms 超过阈值 {THRESHOLD_S * 1000:.1f} ms"
        f"（基线 {BASELINE_MEDIAN_S * 1000:.1f} ms × 1.5）"
    )
