"""Replays the traced big-lama generator with two of its slowest steps replaced.

`big-lama.pt` is a TorchScript module, so its graph cannot be edited. Its
`generator.model` is a `Sequential` of 36 children that `forward` calls in a
fixed order; this module calls those same children in the same order as a plain
`nn.Module`, which makes two steps replaceable:

* the final 7x7 convolution runs as `conv2d(padding=3)` cropped back to size
  instead of `reflection_pad2d(3)` + `conv2d(padding=0)`. cuDNN is ~1.6x faster
  in that shape (16.6 -> 10.2 ms at 512²) and the interior pixels, all the
  pixels the graph's own padding does not reach, are bit-identical;
* the three `BatchNorm2d` layers between the upsampling transposed convolutions
  are folded into those convolutions' weights and biases, removing three kernel
  launches (~1.3 ms).

Every other step stays the original traced module, so it is unchanged bit for
bit. The replacement is only used when it reproduces the traced model on a
random input at init time; otherwise `build_lama_replay` returns None and the
caller keeps the plain module.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple

from .tensors import binary_mask, upload_uint8

_BATCH_NORM_EPS = 1e-5
_CHECK_SIZE = 64
_CHECK_TOLERANCE = 1e-3

_EXPECTED_CHILDREN = (
    ["ReflectionPad2d"]
    + ["FFC_BN_ACT"] * 4
    + ["FFCResnetBlock"] * 18
    + ["ConcatTupleLayer"]
    + ["ConvTranspose2d", "BatchNorm2d", "ReLU"] * 3
    + ["ReflectionPad2d", "Conv2d", "Sigmoid"]
)
_DECONV_INDICES = (24, 27, 30)


def _parameter(module: Any, name: str) -> Optional[torch.Tensor]:
    # ScriptModule parameters live in an OrderedDictWrapper, which has no `.get`,
    # and a bias-free conv raises AttributeError on `module.bias`.
    return dict(module._parameters).get(name)


def _buffer(module: Any, name: str) -> Optional[torch.Tensor]:
    return dict(module._buffers).get(name)


def _required_parameter(module: Any, name: str) -> torch.Tensor:
    value = _parameter(module, name)
    if value is None:
        raise AttributeError(f"missing parameter {name}")
    return value


def _batch_norm_affine(batch_norm: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(scale, shift)` so the batch norm evaluates as `x * scale + shift`."""
    weight = _parameter(batch_norm, "weight")
    bias = _parameter(batch_norm, "bias")
    mean = _buffer(batch_norm, "running_mean")
    var = _buffer(batch_norm, "running_var")
    if weight is None or mean is None or var is None:
        raise AttributeError("batch norm without affine parameters")
    try:
        eps = float(batch_norm.eps)
    except AttributeError:
        eps = _BATCH_NORM_EPS
    scale = weight / torch.sqrt(var + eps)
    if bias is None:
        bias = torch.zeros_like(mean)
    return scale, bias - mean * scale


class LaMaGeneratorReplay(torch.nn.Module):
    output_weight: torch.Tensor
    output_bias: torch.Tensor

    def __init__(self, model: Any):
        super().__init__()
        generator = model.generator.model
        children = [generator._modules[str(index)] for index in range(36)]
        if [child.original_name for child in children] != _EXPECTED_CHILDREN:
            raise ValueError("unexpected big-lama generator layout")

        # Registered as a submodule so `.to(device)` and `.eval()` reach it.
        self.generator = model
        self.pad_start = children[0]
        self.stem = children[1:4]
        self.split = children[4]
        self.blocks = children[5:23]
        self.merge = children[23]
        self.pad_end = children[33]
        self.sigmoid = children[35]

        with torch.no_grad():
            for index in _DECONV_INDICES:
                deconv = children[index]
                scale, shift = _batch_norm_affine(children[index + 1])
                weight = _required_parameter(deconv, "weight")
                bias = _required_parameter(deconv, "bias")
                self.register_buffer(
                    f"deconv{index}_weight",
                    weight * scale.reshape(1, -1, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    f"deconv{index}_bias", bias * scale + shift, persistent=False
                )

            self.register_buffer(
                "output_weight",
                _required_parameter(children[34], "weight").detach(),
                persistent=False,
            )
            self.register_buffer(
                "output_bias",
                _required_parameter(children[34], "bias").detach(),
                persistent=False,
            )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image * (1 - mask), mask], dim=1)
        x = self.pad_start(x)
        for stem in self.stem:
            x = stem(x)
        p, q = self.split(x)
        for block in self.blocks:
            p, q = block(p, q)
        x = self.merge(p, q)

        for index in _DECONV_INDICES:
            x = F.conv_transpose2d(
                x,
                self.get_buffer(f"deconv{index}_weight"),
                self.get_buffer(f"deconv{index}_bias"),
                stride=2,
                padding=1,
                output_padding=1,
            )
            torch.relu_(x)

        x = self.pad_end(x)
        height, width = x.shape[-2:]
        x = F.conv2d(x, self.output_weight, self.output_bias, padding=3)
        x = x[..., 3 : height - 3, 3 : width - 3]
        generated = self.sigmoid(x)
        return mask * generated + (1 - mask) * image


def _reproduces_traced_model(replay: LaMaGeneratorReplay, model: Any) -> bool:
    """Check the replay against the traced module on one random 64² input."""
    device = str(replay.output_weight.device)
    rng = np.random.default_rng(0)
    image = upload_uint8(
        rng.integers(0, 256, (_CHECK_SIZE, _CHECK_SIZE, 3), dtype=np.uint8), device
    )
    mask_np = np.zeros((_CHECK_SIZE, _CHECK_SIZE), dtype=np.uint8)
    mask_np[: _CHECK_SIZE // 2, : _CHECK_SIZE // 2] = 255
    mask = binary_mask(mask_np, device)
    with torch.no_grad():
        expected = model(image, mask)
        actual = replay(image, mask)
    return (
        expected.shape == actual.shape
        and torch.max(torch.abs(expected - actual)).item() <= _CHECK_TOLERANCE
    )


def build_lama_replay(model: Any) -> Optional[LaMaGeneratorReplay]:
    """Return a faster replay of `model`'s forward, or None if it cannot match."""
    try:
        replay = LaMaGeneratorReplay(model)
        if not _reproduces_traced_model(replay, model):
            return None
    except (AttributeError, KeyError, ValueError, TypeError):
        return None
    return replay
