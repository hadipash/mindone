import numpy as np
import pytest

from mindspore import Tensor, context, float16, float32, grad, nn

from mindone.models.modules import Attention, BlockwiseAttention

context.set_context(
    mode=context.PYNATIVE_MODE, deterministic="ON", ascend_config={"precision_mode": "must_keep_origin_dtype"}
)

fp16_fwd_tolerance = 1e-3
fp32_fwd_tolerance = 1e-6

fp16_bwd_tolerance = 1e-5
fp32_bwd_tolerance = 1e-10


@pytest.fixture(scope="module")
def q_k_v():
    q = Tensor(np.random.randn(16, 256, 32), dtype=float32)
    k = Tensor(np.random.randn(16, 256, 32), dtype=float32)
    v = Tensor(np.random.randn(16, 256, 32), dtype=float32)
    label = Tensor(np.ones((16, 256, 32)), dtype=float32)
    return q, k, v, label


@pytest.fixture(scope="module")
def vanilla_fwd_fp16(q_k_v):
    q, k, v, _ = q_k_v
    attn = Attention(q.shape[-1])
    attn.set_train(False)
    attn.to_float(float16)
    return attn(q, k, v, None).asnumpy().astype(np.float32)


@pytest.fixture(scope="module")
def vanilla_fwd_fp32(q_k_v):
    q, k, v, _ = q_k_v
    attn = Attention(q.shape[-1])
    attn.set_train(False)
    return attn(q, k, v, None).asnumpy()


@pytest.fixture(scope="module")
def loss_fn(q_k_v):
    return nn.MSELoss()


@pytest.fixture(scope="module")
def vanilla_bwd_fp16(q_k_v, loss_fn):
    q, k, v, label = q_k_v
    attn = Attention(q.shape[-1], attn_drop=0.0)
    attn.set_train(True)
    attn.to_float(float16)

    def forward(q_, k_, v_, label_):
        z = attn(q_, k_, v_, None)
        return loss_fn(z, label_)

    grads = grad(forward, grad_position=(0, 1, 2))(q.astype(float16), k.astype(float16), v.astype(float16), label)
    return tuple(g.asnumpy().astype(np.float32) for g in grads)


@pytest.fixture(scope="module")
def vanilla_bwd_fp32(q_k_v, loss_fn):
    q, k, v, label = q_k_v
    attn = Attention(q.shape[-1], attn_drop=0.0)
    attn.set_train(True)

    def forward(q_, k_, v_, label_):
        z = attn(q_, k_, v_, None)
        return loss_fn(z, label_)

    grads = grad(forward, grad_position=(0, 1, 2))(q, k, v, label)
    return tuple(g.asnumpy() for g in grads)


@pytest.mark.parametrize(
    "q_chunks, kv_chunks", [(q_chunks, kv_chunks) for q_chunks in [1, 2, 4, 8] for kv_chunks in [1, 2, 4, 8]]
)
def test_blockwise_fwd_fp16(q_k_v, vanilla_fwd_fp16, q_chunks, kv_chunks):
    q, k, v, _ = q_k_v

    attn = BlockwiseAttention(q.shape[-1], q_chunks, kv_chunks)
    attn.set_train(False)
    attn.to_float(float16)

    out = attn(q, k, v)[0].asnumpy().astype(np.float32)
    print(f"Absolute error: {abs(out - vanilla_fwd_fp16).max()}")
    assert np.allclose(out, vanilla_fwd_fp16, atol=fp16_fwd_tolerance, rtol=0)


@pytest.mark.parametrize(
    "q_chunks, kv_chunks", [(q_chunks, kv_chunks) for q_chunks in [1, 2, 4, 8] for kv_chunks in [1, 2, 4, 8]]
)
def test_blockwise_fwd_fp32(q_k_v, vanilla_fwd_fp32, q_chunks, kv_chunks):
    q, k, v, _ = q_k_v

    attn = BlockwiseAttention(q.shape[-1], q_chunks, kv_chunks)
    attn.set_train(False)

    out = attn(q, k, v)[0].asnumpy()
    print(f"Absolute error: {abs(out - vanilla_fwd_fp32).max()}")
    assert np.allclose(out, vanilla_fwd_fp32, atol=fp32_fwd_tolerance, rtol=0)


@pytest.mark.parametrize(
    "q_chunks, kv_chunks", [(q_chunks, kv_chunks) for q_chunks in [1, 2, 4, 8] for kv_chunks in [1, 2, 4, 8]]
)
def test_blockwise_bwd_fp16(q_k_v, vanilla_bwd_fp16, loss_fn, q_chunks, kv_chunks):
    q, k, v, label = q_k_v

    attn = BlockwiseAttention(q.shape[-1], q_chunks, kv_chunks)
    attn.set_train(True)
    attn.to_float(float16)

    def forward(q_, k_, v_, label_):
        z = attn(q_, k_, v_)
        return loss_fn(z[0], label_)

    grads = grad(forward, grad_position=(0, 1, 2))(q.astype(float16), k.astype(float16), v.astype(float16), label)
    grads = tuple(g.asnumpy().astype(np.float32) for g in grads)
    for g, v_g in zip(grads, vanilla_bwd_fp16):
        print(f"Absolute error: {abs(g - v_g).max()}")
        assert np.allclose(g, v_g, atol=fp16_bwd_tolerance, rtol=0)


@pytest.mark.parametrize(
    "q_chunks, kv_chunks", [(q_chunks, kv_chunks) for q_chunks in [1, 2, 4, 8] for kv_chunks in [1, 2, 4, 8]]
)
def test_blockwise_bwd_fp32(q_k_v, vanilla_bwd_fp32, loss_fn, q_chunks, kv_chunks):
    q, k, v, label = q_k_v

    attn = BlockwiseAttention(q.shape[-1], q_chunks, kv_chunks)
    attn.set_train(True)

    def forward(q_, k_, v_, label_):
        z = attn(q_, k_, v_)
        return loss_fn(z[0], label_)

    grads = grad(forward, grad_position=(0, 1, 2))(q, k, v, label)
    grads = (g.asnumpy() for g in grads)
    for g, v_g in zip(grads, vanilla_bwd_fp32):
        print(f"Absolute error: {abs(g - v_g).max()}")
        assert np.allclose(g, v_g, atol=fp32_bwd_tolerance, rtol=0)
