"""Shared fixtures for IOPaint tests."""
import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def iopaint_port():
    """Return a free port for the test server."""
    return _free_port()


@pytest.fixture(scope="session")
def iopaint_server(tmp_path_factory, iopaint_port):
    """Start an IOPaint server with the cv2 model for testing.

    Yields the base URL.  The server process is killed after all tests finish.
    """
    output_dir = tmp_path_factory.mktemp("output")
    cmd = [
        sys.executable, "-m", "iopaint",
        "start",
        "--model", "cv2",
        "--device", "cpu",
        "--port", str(iopaint_port),
        "--output-dir", str(output_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until the server is accepting connections (max 30s)
    base_url = f"http://127.0.0.1:{iopaint_port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen(base_url, timeout=2)
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

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
