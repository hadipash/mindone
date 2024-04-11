from typing import Optional

import mindspore as ms
from mindspore import Tensor, nn, ops


class Attention(nn.Cell):
    def __init__(self, dim_head: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.b_matmul = ops.BatchMatMul(transpose_b=True)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sim = self.b_matmul(q, k) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]
            sim = ops.masked_fill(sim, ~mask, -ms.numpy.inf)

        # use fp32 for exponential inside
        attn = ops.softmax(sim.astype(ms.float32), axis=-1).astype(v.dtype)
        attn = self.attn_drop(attn)
        out = ops.matmul(attn, v)
        return out
