"""Perf guard #1：`GET /api/v1/server-config` 必须由缓存的
`ModelManager.get_available_models()` 服务，绝不能按请求重扫模型目录。

设计见 `docs/agents/perf-guards.md` §1。本地跑，**不进 CI、不进 fast suite**：

    python benchmarks/run.py -k server_config

（`pytest.ini` 的 `testpaths = iopaint/tests`，所以 CI 与 `scripts/check.sh` 里的
`pytest -m "not slow"` 根本不会收集本目录。）

主断言是「重扫次数」（与机器无关）；耗时只登记数字，回归由
`--bench-compare` 的比值判断，不写死某台机器的毫秒阈值。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harness import measure
from iopaint.api import Api
from iopaint.model_manager import ModelManager
from iopaint.tests.test_api import _minimal_config

pytestmark = pytest.mark.bench

ROUNDS = 5
CALLS_PER_ROUND = 3


def _client(tmp_path):
    cfg = _minimal_config(tmp_path)
    app = FastAPI()
    api = Api(app, cfg)  # 构造期那次 scan_models 已经把 available_models 填好
    return api, TestClient(app)


def _count_scans(api, monkeypatch):
    """把 scan_models 换成计数包装；构造已完成，此后的调用都算「按请求重扫」。"""
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


def test_server_config_latency_recorded(tmp_path, monkeypatch):
    """耗时只登记：这个数只用于同机 `--bench-compare`，跨机器没有意义。"""
    api, client = _client(tmp_path)
    calls = _count_scans(api, monkeypatch)

    def one_round():
        for _ in range(CALLS_PER_ROUND):
            assert client.get("/api/v1/server-config").status_code == 200

    measure("server_config.3x", one_round, rounds=ROUNDS)
    assert calls == []
