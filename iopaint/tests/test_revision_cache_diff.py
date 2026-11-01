"""`_diff_hinted` 的等价性：带范围提示与全帧比较必须给出同一张变化图。

`RevisionCache._commit` 原来对整帧做 `np.any(result != canonical, axis=2)`
（12MP 实测 158ms）。现在先按调用方给的矩形只比盒内，盒外逐字节校验，校验不过
就退回全帧。这里的用例就是那个"不过"与"通过"两条路的等价证明。
"""
import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api
from iopaint.revision_cache import _diff_hinted, _outside_equal
from iopaint.tests.test_api import _minimal_config


def _frame(seed: int, shape: Tuple[int, int, int] = (120, 160, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def _full_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.any(a != b, axis=2)


class TestDiffHinted:
    def test_region_covers_change(self) -> None:
        a = _frame(1)
        b = a.copy()
        b[20:40, 30:70] = 0
        hint = (10, 10, 90, 60)
        assert _outside_equal(a, b, hint)
        assert np.array_equal(_diff_hinted(a, b, hint), _full_diff(a, b))

    def test_region_too_small_falls_back(self) -> None:
        a = _frame(2)
        b = a.copy()
        b[80:90, 5:15] = 7  # 提示框（盒内）根本没覆盖到这块
        hint = (20, 20, 40, 40)
        assert not _outside_equal(a, b, hint)
        assert np.array_equal(_diff_hinted(a, b, hint), _full_diff(a, b))

    def test_region_none_and_out_of_range(self) -> None:
        a = _frame(3)
        b = a.copy()
        b[0:2, 0:2] = 255
        assert np.array_equal(_diff_hinted(a, b, None), _full_diff(a, b))
        # 越界/空矩形一律退回全帧，不能悄悄返回 "无变化"
        assert np.array_equal(_diff_hinted(a, b, (500, 500, 900, 900)), _full_diff(a, b))
        assert np.array_equal(_diff_hinted(a, b, (9, 9, 9, 9)), _full_diff(a, b))

    def test_identical_frames_have_no_diff(self) -> None:
        a = _frame(4)
        assert not _diff_hinted(a, a.copy(), (10, 10, 100, 100)).any()
        assert not _diff_hinted(a, a.copy(), None).any()

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_full_diff_on_random_edits(self, seed: int) -> None:
        rng = np.random.default_rng(100 + seed)
        a = _frame(200 + seed)
        b = a.copy()
        for _ in range(20):
            y, x = int(rng.integers(0, 120)), int(rng.integers(0, 160))
            b[y : y + 3, x : x + 3] = int(rng.integers(0, 255))
        # 一半用例提示框故意框小，覆盖回退分支
        hint = (0, 0, 80, 60) if seed % 2 else (20, 20, 40, 40)
        assert np.array_equal(_diff_hinted(a, b, hint), _full_diff(a, b))


# ---------------------------------------------------------------------------
# 端到端：带提示与强制全帧比较必须产出同一份像素
# ---------------------------------------------------------------------------


def _png_b64(width: int, height: int) -> str:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _run_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_hint: bool
) -> bytes:
    if not with_hint:
        monkeypatch.setattr(Api, "_region_hint", staticmethod(lambda *a, **k: None))

    cfg = _minimal_config(tmp_path)
    app = FastAPI()
    Api(app, cfg)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/inpaint",
            json={
                "image": _png_b64(320, 240),
                "mask_strokes": [{"pts": [[40, 40], [120, 90]], "size": 16}],
            },
        )
        assert resp.status_code == 200, resp.text[:200]
        revision = resp.headers["X-Revision"]
        for i in range(3):
            resp = client.post(
                "/api/v1/inpaint",
                json={
                    "revision": revision,
                    "mask_strokes": [
                        {"pts": [[60 + i * 20, 50 + i * 15], [200, 150]], "size": 12}
                    ],
                },
            )
            assert resp.status_code == 200, resp.text[:200]
            revision = resp.headers["X-Revision"]
        exported = client.post("/api/v1/export", json={"revision": revision})
        assert exported.status_code == 200
        return exported.content


def test_hint_does_not_change_exported_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_hint = _run_edits(tmp_path / "hint", monkeypatch, with_hint=True)
    monkeypatch.undo()
    without_hint = _run_edits(tmp_path / "no_hint", monkeypatch, with_hint=False)
    assert with_hint == without_hint
