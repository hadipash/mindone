import numpy as np
import pytest
from src.sparsification import BlockSparseAttention

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import tensor

from .utils import BLOCK_SIZE, DTYPES, TOLERANCES, Attention


@pytest.fixture(scope="module")
def q_k_v() -> tuple[Tensor, Tensor, Tensor]:
    # B, H, S, D
    q = tensor(np.random.randn(2, 8, 256, 32), dtype=mstype.float32)
    k = tensor(np.random.randn(2, 8, 384, 32), dtype=mstype.float32)
    v = tensor(np.random.randn(2, 8, 384, 32), dtype=mstype.float32)
    return q, k, v


@pytest.fixture(scope="module")
def masks(q_k_v: tuple[Tensor, Tensor, Tensor]) -> dict[str, dict[str, Tensor]]:
    q, k, _ = q_k_v
    assert q.shape[2] % BLOCK_SIZE == 0, f"Q's sequence length should be divisible by the block size ({BLOCK_SIZE})"
    assert (
        k.shape[2] % BLOCK_SIZE == 0
    ), f"K's and V's sequence length should be divisible by the block size ({BLOCK_SIZE})"

    out_masks = {
        "dense": {
            "vanilla": tensor(np.ones((*q.shape[:3], k.shape[2])), dtype=mstype.bool_),
            "block": tensor(
                np.ones((q.shape[0] * q.shape[1], q.shape[2] // BLOCK_SIZE, k.shape[2] // BLOCK_SIZE)),
                dtype=mstype.bool_,
            ),
        }
    }

    block_sparse = np.random.randint(0, 2, (q.shape[0], q.shape[1], q.shape[2] // BLOCK_SIZE, k.shape[2] // BLOCK_SIZE))
    vanilla_sparse = block_sparse.repeat(BLOCK_SIZE, axis=-2).repeat(BLOCK_SIZE, axis=-1)
    out_masks["sparse"] = {
        "vanilla": tensor(vanilla_sparse, dtype=mstype.bool_),
        "block": tensor(block_sparse.reshape(-1, *block_sparse.shape[2:]), dtype=mstype.bool_),
    }

    return out_masks


@pytest.fixture(scope="module")
def vanilla_attention(
    q_k_v: tuple[Tensor, Tensor, Tensor], masks: dict[str, dict[str, Tensor]]
) -> dict[str, dict[str, np.ndarray]]:
    def _attn(mask: Tensor, dtype: mstype) -> np.ndarray:
        q, k, v = q_k_v
        attn = Attention(q.shape[-1]).set_train(False).to_float(dtype)
        return attn(q.to(dtype), k.to(dtype), v.to(dtype), mask).asnumpy().astype(np.float32)

    return {
        attn_type: {k: _attn(masks[attn_type]["vanilla"], v) for k, v in DTYPES.items()}
        for attn_type in ["dense", "sparse"]
    }


@pytest.mark.parametrize("attn_type", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
def test_blocksparse_attention(q_k_v, masks, vanilla_attention, attn_type, dtype):
    q, k, v = q_k_v
    b, h = q.shape[:2]
    q, k, v = q.reshape(-1, *q.shape[2:]), k.reshape(-1, *k.shape[2:]), v.reshape(-1, *v.shape[2:])
    attn = BlockSparseAttention(q.shape[-1], BLOCK_SIZE).set_train(False).to_float(DTYPES[dtype])
    out = attn(q.to(DTYPES[dtype]), k.to(DTYPES[dtype]), v.to(DTYPES[dtype]), masks[attn_type]["block"])
    out = out.asnumpy().reshape(b, h, *out.shape[1:]).astype(np.float32)
    assert np.allclose(
        out, vanilla_attention[attn_type][dtype], **TOLERANCES[dtype]
    ), f"{dtype} failed. Max error: {abs(out - vanilla_attention[attn_type][dtype]).max()}"
