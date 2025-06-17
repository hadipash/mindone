from typing import Optional

from mindspore import Tensor, mint, nn
from mindspore import numpy as msnumpy
from mindspore import ops
from mindspore.common import dtype as mstype

BLOCK_SIZE = 128

DTYPES = {"fp32": mstype.float32, "fp16": mstype.float16, "bf16": mstype.bfloat16}

TOLERANCES = {
    "fp32": {"rtol": 1.3e-6, "atol": 1e-5},
    "fp16": {"rtol": 1e-3, "atol": 1e-3},
    "bf16": {"rtol": 1.6e-2, "atol": 1e-2},
}


class Attention(nn.Cell):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self._scale = head_dim**-0.5
        self._bmm = ops.BatchMatMul(transpose_b=True)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sim = self._bmm(q, k) * self._scale

        if mask is not None:
            sim = ops.masked_fill(sim, ~mask, -msnumpy.inf)

        # use fp32 for exponential inside
        attn = mint.softmax(sim.astype(mstype.float32), dim=-1).astype(v.dtype)
        attn = mint.nan_to_num(attn, nan=0)
        out = mint.matmul(attn, v)
        return out
