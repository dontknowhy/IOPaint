"""FcF keeps deterministic cuDNN scoped to its forward (no global leak)."""
from typing import Iterator

import pytest
import torch

from iopaint.model.fcf import _cudnn_deterministic


@pytest.fixture
def restore_cudnn_flags() -> Iterator[None]:
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.deterministic = old_deterministic
    torch.backends.cudnn.benchmark = old_benchmark


def test_deterministic_context_restores_flags(restore_cudnn_flags: None) -> None:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    with _cudnn_deterministic():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_deterministic_context_restores_on_error(restore_cudnn_flags: None) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with pytest.raises(RuntimeError):
        with _cudnn_deterministic():
            assert torch.backends.cudnn.deterministic is True
            raise RuntimeError("boom")

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
