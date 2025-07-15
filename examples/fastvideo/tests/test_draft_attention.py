from time import perf_counter

import numpy as np
import pytest
from src.sparsification import BlockSparseAttention, DraftAttention

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, tensor

from .utils import BLOCK_SIZE, DTYPES, TOLERANCES, Attention


@pytest.fixture(scope="module")
def q_k_v() -> tuple[Tensor, Tensor, Tensor]:
    # B, H, S, D
    q = tensor(np.random.randn(11776, 8, 32), dtype=mstype.float32)
    k = tensor(np.random.randn(11776, 8, 32), dtype=mstype.float32)
    v = tensor(np.random.randn(11776, 8, 32), dtype=mstype.float32)
    return q, k, v


@pytest.fixture(scope="module")
def block_full_attn_speed(q_k_v) -> dict[str, float]:
    q, k, v = q_k_v
    q, k, v = q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1)
    mask = tensor(np.ones((q.shape[0], q.shape[1] // BLOCK_SIZE, k.shape[1] // BLOCK_SIZE)), dtype=mstype.bool_)

    metrics = {}
    for name, dtype in DTYPES.items():
        attn = BlockSparseAttention(q.shape[-1], BLOCK_SIZE).set_train(False).to_float(dtype)

        times = []
        for i in range(6):
            start = perf_counter()
            attn(q.to(dtype), k.to(dtype), v.to(dtype), mask).asnumpy()
            times.append(perf_counter() - start)
        metrics[name] = np.mean(times[1:])  # ignore first run

    return metrics


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("attn_type", ["flash", "block"])
def test_draft_attention_numerically(q_k_v, dtype, attn_type):
    if attn_type == "flash" and dtype == "fp32":
        pytest.skip("Flash Attention doesn't support `FP32` precision.")

    q, k, v = q_k_v

    attn = DraftAttention(
        head_dim=q.shape[-1], block_size=BLOCK_SIZE, visual_len=q.shape[0] - 256, text_len=256, attn_type=attn_type
    )
    attn = attn.set_train(False).to_float(DTYPES[dtype])
    attn_out, mask = attn(
        q.to(DTYPES[dtype]), k.to(DTYPES[dtype]), v.to(DTYPES[dtype]), batch_size=1, return_blockmask=True
    )
    attn_out = attn_out.asnumpy().astype(np.float32)
    if attn_type == "flash":
        mask = ~mask
    if attn_type == "block":
        mask = mint.repeat_interleave(mint.repeat_interleave(mask, BLOCK_SIZE, dim=-1), BLOCK_SIZE, dim=-2)

    vanilla = Attention(head_dim=q.shape[-1]).set_train(False).to_float(DTYPES[dtype])
    # B H N D
    vanilla_out = vanilla(
        q.swapaxes(0, 1)[None].to(DTYPES[dtype]),
        k.swapaxes(0, 1)[None].to(DTYPES[dtype]),
        v.swapaxes(0, 1)[None].to(DTYPES[dtype]),
        mask,
    )
    vanilla_out = vanilla_out.asnumpy().astype(np.float32).swapaxes(1, 2)

    assert np.allclose(
        attn_out, vanilla_out, **TOLERANCES[dtype]
    ), f"{dtype} failed. Max error: {abs(attn_out - vanilla_out).max()}"


@pytest.mark.parametrize("sparsity_ratio", [i / 10 for i in range(5, 10)])
@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
def test_draft_attention_speedup(q_k_v, block_full_attn_speed, sparsity_ratio, dtype):
    q, k, v = q_k_v

    attn = DraftAttention(
        head_dim=q.shape[-1],
        block_size=BLOCK_SIZE,
        visual_len=q.shape[0] - 256,
        text_len=256,
        sparsity_ratio=sparsity_ratio,
        attn_type="block",
    )
    attn = attn.set_train(False).to_float(DTYPES[dtype])

    times = []
    for i in range(6):
        start = perf_counter()
        attn(q.to(DTYPES[dtype]), k.to(DTYPES[dtype]), v.to(DTYPES[dtype]), batch_size=1).asnumpy()
        times.append(perf_counter() - start)
    time = np.mean(times[1:])  # ignore first run

    assert time < block_full_attn_speed[dtype], f"{dtype} failed: {block_full_attn_speed[dtype] / time:.2f}x slower"
    print(f"Speedup over dense attention: {block_full_attn_speed[dtype] / time:.2f}x")
