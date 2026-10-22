"""scripts/check_types.py：mypy 二进制必须与运行解释器同环境。"""
import sys

import pytest

from scripts import check_types


def test_prefers_module_of_running_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH 里的 /usr/bin/mypy 可能属于另一个 Python（解析 site-packages 会失败）。"""
    monkeypatch.delenv("MYPY", raising=False)
    monkeypatch.setattr(check_types.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(check_types.shutil, "which", lambda name: "/usr/bin/mypy")
    assert check_types.mypy_command() == [sys.executable, "-m", "mypy"]


def test_falls_back_to_path_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MYPY", raising=False)
    monkeypatch.setattr(check_types.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(check_types.shutil, "which", lambda name: "/opt/mypy")
    assert check_types.mypy_command() == ["/opt/mypy"]


def test_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYPY", "/custom/mypy")
    monkeypatch.setattr(check_types.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(check_types.shutil, "which", lambda name: "/usr/bin/mypy")
    assert check_types.mypy_command() == ["/custom/mypy"]


def test_missing_mypy_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MYPY", raising=False)
    monkeypatch.setattr(check_types.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(check_types.shutil, "which", lambda name: None)
    assert check_types.mypy_command() == []
