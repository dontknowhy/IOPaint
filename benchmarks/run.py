"""`python benchmarks/run.py` — 跑 benchmarks/ 的薄入口。

等价于 `python -m pytest benchmarks -m bench <参数>`；额外的参数原样交给 pytest，
所以 `-k cold`、`--bench-save ...`、`--bench-compare ...` 都能直接跟在后面。

    python benchmarks/run.py
    python benchmarks/run.py -k "cold or warm"
    python benchmarks/run.py --bench-save benchmarks/baselines/local.json
    python benchmarks/run.py --bench-compare benchmarks/baselines/local.json
"""
from __future__ import annotations

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["benchmarks", "-m", "bench", *sys.argv[1:]])
    )
