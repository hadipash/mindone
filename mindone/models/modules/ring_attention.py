from mindspore import dtype, grad, nn, ops


class RingAttention(nn.Cell):
    def __init__(self, head_dim: int, seq_len: int, algorithm: str = "vanilla", rank_id: int = 0, num_devices: int = 1):
        super().__init__()
        self._rank_id = rank_id
        self._num_devices = num_devices

        self.scale = head_dim**-0.5
        self.bmm = ops.BatchMatMul(transpose_b=True)

        # distributed part
        send_rank = (rank_id + 1) % num_devices
        recv_rank = (rank_id - 1) % num_devices
        print(f"rank_id: {rank_id}, send_rank: {send_rank}, recv_rank: {recv_rank}")
        self._exchange = ops.NeighborExchange(
            send_rank_ids=[send_rank],
            recv_rank_ids=[recv_rank],
            recv_shapes=([4096, seq_len // num_devices, head_dim],),  # FIXME
            send_shapes=([4096, seq_len // num_devices, head_dim],),
            recv_type=dtype.float32,
        )
        self._depend = ops.Depend()

    def construct(self, q, k, v):
        prev_max_score, max_score = None, 0
        numerator, denominator = 0, 0
        next_k, next_v = None, None

        # Q and KV blocks are distributed across N hosts.
        # Q blocks stay intact, and KV blocks are exchanged across the hosts.
        # To calculate full attention, KV blocks must be exchanged within all hosts (i.e., N steps).
        for step in range(self._num_devices):
            # initiate KV exchange before calculating attention as calculation takes more time than sending data
            if step < self._num_devices - 1:  # no need to exchange in the penultimate step
                # BUG [MS2.2.10]: can't exchange both tensors at the same time
                next_k = self._exchange((k,))[0].astype(q.dtype)
                next_v = self._exchange((v,))[0].astype(q.dtype)

            # FIXME: causal attention doesn't work in a distributed setting
            attn_weights = self.bmm(q, k) * self.scale

            max_score = attn_weights.max(axis=-1, keepdims=True)
            if prev_max_score is None:
                prev_max_score = max_score
            max_score = ops.maximum(prev_max_score, max_score)
            max_score = ops.stop_gradient(max_score)

            exp_weights = ops.exp(attn_weights - max_score)
            correction = ops.exp(prev_max_score - max_score)

            numerator = numerator * correction + ops.matmul(exp_weights, v)
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

            prev_max_score = max_score

            if step < self._num_devices - 1:
                # TODO: ensure that next_k and next_v are received (necessary?)
                k = self._depend(next_k, next_k)
                v = self._depend(next_v, next_v)

        out = numerator / denominator
        return out, denominator, max_score

    def bprop(self, q, k, v, out, dout):
        out, denominator, max_score = out
        out_grad = dout[0]
        dq, dk, dv = map(lambda x: ops.zeros_like(x), (q, k, v))
        next_k, next_v = None, None

        dl_part = (out_grad * out).sum(axis=-1, keepdims=True)

        for step in range(self._num_devices):
            if step < self._num_devices - 1:  # no need to exchange in the penultimate step
                # BUG [MS2.2.10]: can't exchange both tensors at the same time
                next_k = self._exchange((k,))[0]
                next_v = self._exchange((v,))[0]

            attn_weights = self.bmm(q, k) * self.scale
            exp_weights = ops.exp(attn_weights - max_score) / denominator

            ds = self.bmm(out_grad, v)
            dl = (ds - dl_part) * exp_weights

            dq += ops.matmul(dl, k) * self.scale
            dk += ops.matmul(q.swapaxes(1, 2), dl).swapaxes(1, 2) * self.scale
            dv += ops.matmul(exp_weights.swapaxes(1, 2), out_grad)

            if step < self._num_devices - 1:
                # TODO: ensure that next_k and next_v are received (necessary?)
                k = self._depend(next_k, next_k)
                v = self._depend(next_v, next_v)

            dk = self._exchange((dk,))[0]
            dv = self._exchange((dv,))[0]
            dk = self._depend(dk, dk)
            dv = self._depend(dv, dv)

        return dq, dk, dv


if __name__ == "__main__":
    import os
    import sys

    import numpy as np

    import mindspore as ms
    from mindspore.communication import get_group_size, get_rank, init

    sys.path.append("../../../")
    from mindone.models.dit import Attention

    device_id = int(os.getenv("DEVICE_ID"))
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_id=device_id,
        deterministic="ON",
        ascend_config={"precision_mode": "must_keep_origin_dtype"},
    )
    init()
    device_num = get_group_size()
    rank_id = get_rank()

    ms.set_seed(42)

    seq_len = 256
    chunk_size = seq_len // device_num

    q_ = ms.Tensor(np.random.randn(16, seq_len, 32), dtype=ms.float32)
    k_ = ms.Tensor(np.random.randn(16, seq_len, 32), dtype=ms.float32)
    v_ = ms.Tensor(np.random.randn(16, seq_len, 32), dtype=ms.float32)
    labels = ms.Tensor(np.ones((16, seq_len, 32)), dtype=ms.float32)

    ra = RingAttention(head_dim=q_.shape[2], seq_len=q_.shape[1], rank_id=rank_id, num_devices=device_num)
    vanilla_attn = Attention(q_.shape[-1])

    def forward_test():
        vanilla_attn.set_train(False)
        ra.set_train(False)

        out1_fwd = ra(
            q_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
            k_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
            v_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
        )[0].asnumpy()

        out2 = vanilla_attn(q_, k_, v_, None)[:, rank_id * chunk_size : (rank_id + 1) * chunk_size].asnumpy()

        print("rank_id: ", rank_id, np.allclose(out1_fwd, out2, atol=1e-6, rtol=0))
        print("rank_id: ", rank_id, f"Absolute error: {abs(out1_fwd - out2).max()}")

    def bprop_test():
        loss = nn.MSELoss()

        ra.set_train(True)
        vanilla_attn.set_train(True)

        def forward_ra(q, k, v, label):
            z = ra(q, k, v)
            return loss(z[0], label)

        def forward_vanilla(q, k, v, label):
            z = vanilla_attn(q, k, v, None)
            return loss(z, label)

        grad1 = grad(forward_ra, grad_position=(0, 1, 2))(
            q_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
            k_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
            v_[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
            labels[:, rank_id * chunk_size : (rank_id + 1) * chunk_size],
        )

        grad2 = grad(forward_vanilla, grad_position=(0, 1, 2))(q_, k_, v_, labels)

        for g1, g2 in zip(grad1, grad2):
            g1 = g1.asnumpy() / device_num  # gradients mean
            g2 = g2.asnumpy()[:, rank_id * chunk_size : (rank_id + 1) * chunk_size]
            print("rank_id: ", rank_id, np.allclose(g1, g2, atol=1e-10, rtol=0))
            print("rank_id: ", rank_id, f"Absolute error: {abs(g1 - g2).max()}")

    print("Testing forward pass...")
    forward_test()
    print("Testing backward pass...")
    bprop_test()
