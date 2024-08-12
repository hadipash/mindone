"""
Source: https://github.com/lucidrains/rotary-embedding-torch/
"""
from math import pi
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, dtype, mint, nn, ops

from .operation_selector import get_repeat_interleave_op


def rotate_half(x: Tensor) -> Tensor:
    x = x.reshape(x.shape[:-1] + (-1, 2))  # ... (d r) -> ... d r, r = 2
    x1, x2 = mint.chunk(x, 2, -1)
    x = ops.concat((-x2, x1), axis=-1)
    return x.reshape(x.shape[:-2] + (-1,))  # '... d r -> ... (d r)'


class RotaryEmbedding(nn.Cell):
    """
    Simplified version of Rotary Position Embedding (RoPE) specifically for OpenSora.
    """

    def __init__(
        self,
        dim: int,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            theta *= theta_rescale_factor ** (dim / (dim - 2))
            freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
        elif freqs_for == "pixel":
            freqs = np.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = np.ones(num_freqs)
        else:
            raise ValueError(f"Invalid freqs_for: {freqs_for}")

        self.freqs = Parameter(Tensor(freqs, dtype=dtype.float32), requires_grad=learned_freq)
        self.learned_freq = learned_freq

        self.default_seq_dim = -3 if seq_before_head_dim else -2

        self.repeat_interleave = get_repeat_interleave_op()

    def construct(self, t: Tensor) -> Tensor:
        """
        Args:
            t: tensor of shape (b n h d)
        """
        t = t.swapaxes(1, 2)  # the expected tensor shape is (b h n d), but the input shape is (b n h d)

        seq_pos = ops.arange(t.shape[self.default_seq_dim], dtype=t.dtype)
        freqs = seq_pos[..., None] * self.freqs.astype(t.dtype)
        freqs = self.repeat_interleave(freqs, 2, -1)  # ... n -> ... (n r), r = 2

        if self.default_seq_dim == -3:
            freqs = freqs.unsqueeze(1)  # n d -> n 1 d

        t = t * freqs.cos() + rotate_half(t) * freqs.sin()
        return t.swapaxes(1, 2)  # (b h n d) -> (b n h d)


def rope_1d(x: Tensor, freqs_cis: Tensor) -> Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    x = ops.transpose(x, (0, 2, 1, 3))  # b h n d
    freqs_cis = freqs_cis[:, None, ...]  # b(1) 1 n d
    sin_matrix = ops.sin(freqs_cis)
    cos_matrix = ops.cos(freqs_cis)
    cos_part = ops.mul(x, cos_matrix)
    sin_part = ops.mul(rotate_half(x), sin_matrix)

    x = cos_part + sin_part
    x = ops.transpose(x, (0, 2, 1, 3))  # b n h d
    return x.to(dtype)


def precompute_freqs_cis(seq_len: int, dim: int, theta: float = 10000.0) -> np.ndarray:
    positional_ids = np.arange(seq_len, dtype=np.float32)
    indices = 1.0 / np.power(theta, 2 * np.arange(dim // 2, dtype=np.float32) / dim)
    embeddings = np.outer(positional_ids, indices)
    embeddings = np.repeat(embeddings, 2, axis=-1)
    return embeddings
