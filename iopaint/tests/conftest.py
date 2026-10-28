"""Shared fixtures for IOPaint tests."""
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

# CLI 文案按 IOPAINT_LOCALE / LANG 选语言，开发机若是 zh_CN 会让断言英文文案的用例变红
# （与 test_webui 固定 locale="en-US" 同理）。这里在导入任何测试模块之前钉死 en
# —— CLI 帮助串在 `iopaint.cli` 导入时就已经取好；要测中文的用例自己用 monkeypatch
# 覆盖环境变量，或起子进程传入 IOPAINT_LOCALE=zh-CN。
os.environ["IOPAINT_LOCALE"] = "en"


def _free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def iopaint_port():
    """Return a free port for the test server."""
    return _free_port()


def _launch_iopaint(
    tmp_path_factory: pytest.TempPathFactory,
    port: int,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen, str]:
    """Start `iopaint start --model cv2` on `port` and wait until it answers.

    Returns the process and its base URL; the caller owns the teardown (see `_stop`).
    """
    output_dir = tmp_path_factory.mktemp("output")
    cmd = [
        sys.executable, "-m", "iopaint",
        "start",
        "--model", "cv2",
        "--device", "cpu",
        "--port", str(port),
        "--output-dir", str(output_dir),
        *(extra_args or []),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until the server is accepting connections (max 30s).
    # 只要拿到 HTTP 响应就算起来了：前端没 build 时 `/` 会回 503（见 api.py 的
    # WebAppStaticFiles），那是"服务已就绪但 UI 未构建"，不是"没起来"。
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            import urllib.error
            import urllib.request
            try:
                urllib.request.urlopen(base_url, timeout=2)
            except urllib.error.HTTPError:
                pass  # 有响应即就绪（4xx/5xx 都算）
            break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Server did not start within 30s.\nstdout:\n{stdout.decode()}\n"
            f"stderr:\n{stderr.decode()}"
        )
    return proc, base_url


def _stop_iopaint(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def iopaint_server(
    tmp_path_factory: pytest.TempPathFactory, iopaint_port: int
) -> Iterator[str]:
    """Start an IOPaint server with the cv2 model for testing.

    Yields the base URL.  The server process is killed after all tests finish.
    """
    proc, base_url = _launch_iopaint(tmp_path_factory, iopaint_port)
    yield base_url
    _stop_iopaint(proc)


@pytest.fixture(scope="session")
def iopaint_file_manager_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """带 `--input <目录>` 的服务：FileManager 只在 config.input 是目录时注册。

    单独起一台而不是给 `iopaint_server` 加参数：文件管理器会改 /api/v1/server-config
    的 enableFileManager，其余 30 多条用例的 UI 不该被这个开关影响。
    """
    input_dir = tmp_path_factory.mktemp("input")
    proc, base_url = _launch_iopaint(
        tmp_path_factory, _free_port(), ["--input", str(input_dir)]
    )
    yield base_url
    _stop_iopaint(proc)
