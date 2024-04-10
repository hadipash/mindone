from mindspore import dtype, grad, nn, ops

# from .flash_attention import MSFlashAttention


class RingAttention(nn.Cell):
    def __init__(self, dim_head: int, algorithm: str = "vanilla", rank_id: int = 0, num_devices: int = 1):
        super().__init__()
        self._num_devices = num_devices
        self._rank_id = rank_id

        send_rank = (rank_id + 1) % num_devices
        recv_rank = (rank_id - 1) % num_devices
        print(f"rank_id: {rank_id}, send_rank: {send_rank}, recv_rank: {recv_rank}")
        self._exchange = ops.NeighborExchange(
            send_rank_ids=[send_rank],
            recv_rank_ids=[recv_rank],
            recv_shapes=([10, 128, 16, dim_head],),
            send_shapes=([10, 128, 16, dim_head],),
            recv_type=dtype.float32,
        )
        self._depend = ops.Depend()

        self.scale = dim_head**-0.5
        # self.attn = Attention(128)
        # self.attn = MSFlashAttention(128, 16)

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
                next_k = self._exchange((k,))[0]
                next_v = self._exchange((v,))[0]

            # FIXME: causal attention doesn't work in a distributed setting
            # 'bqhd,bkhd->bhqk'
            attn_weights = ops.matmul(q.swapaxes(1, 2), k.permute((0, 2, 3, 1))) * self.scale

            max_score = attn_weights.max(axis=-1, keepdims=True)
            if prev_max_score is None:
                prev_max_score = max_score
            max_score = ops.maximum(prev_max_score, max_score)
            max_score = ops.stop_gradient(max_score)

            exp_weights = ops.exp(attn_weights - max_score)
            # 'bhqk,bkhd->bqhd'
            exp_values = ops.matmul(exp_weights, v.permute(0, 2, 1, 3)).swapaxes(1, 2)
            correction = ops.exp(prev_max_score - max_score)
            numerator = numerator * correction.swapaxes(1, 2) + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

            if step < self._num_devices - 1:
                # TODO: ensure that next_k and next_v are received (necessary?)
                k = self._depend(next_k, next_k)
                v = self._depend(next_v, next_v)

        denominator = denominator.swapaxes(1, 2)
        out = numerator / denominator
        return out, numerator, denominator, max_score

    def bprop(self, q, k, v, out, dout):
        out, num, denom, max_score = out
        out_grad = dout[0]
        dq, dk, dv = map(lambda x: ops.zeros_like(x), (q, k, v))
        next_k, next_v = None, None

        dl_part = (out_grad * out).sum(axis=-1, keepdims=True).swapaxes(1, 2)  # "bqhd,bqhd->bhq"

        for step in range(self._num_devices):
            if step < self._num_devices - 1:  # no need to exchange in the penultimate step
                # BUG [MS2.2.10]: can't exchange both tensors at the same time
                next_k = self._exchange((k,))[0]
                next_v = self._exchange((v,))[0]

            # 'bqhd,bkhd->bhqk'
            attn_weights = ops.matmul(q.swapaxes(1, 2), k.permute((0, 2, 3, 1))) * self.scale

            exp_weights = ops.exp(attn_weights - max_score) / denom.swapaxes(1, 2)

            ds = ops.matmul(out_grad.swapaxes(1, 2), v.permute((0, 2, 3, 1)))  # "bqhd,bkhd->bhqk"
            dl = (ds - dl_part) * exp_weights

            # 'bhqk,bkhd->bqhd'
            dq += ops.matmul(dl, k.permute(0, 2, 1, 3)).swapaxes(1, 2) * self.scale
            # "bqhd,bhqk->bkhd"
            dk += ops.matmul(q.permute(0, 2, 3, 1), dl).permute(0, 3, 1, 2) * self.scale
            # "bhqk,bqhd->bkhd"
            dv += ops.matmul(exp_weights.swapaxes(2, 3), out_grad.swapaxes(1, 2)).swapaxes(1, 2)

            if step < self._num_devices - 1:
                # TODO: ensure that next_k and next_v are received (necessary?)
                k = self._depend(next_k, next_k)
                v = self._depend(next_v, next_v)

            dk = self._exchange((dk,))[0]
            dv = self._exchange((dv,))[0]

        return dq, dk, dv


if __name__ == "__main__":
    import os
    import sys

    import numpy as np

    import mindspore as ms
    from mindspore.communication import get_group_size, get_rank, init

    sys.path.append("../../../")
    from mindone.models.dit import Attention

    ms.set_seed(42)

    class DistribExec(nn.Cell):
        def __init__(self, model, rank_id, num_devices):
            super().__init__()
            self._model = model
            self._rank_id = rank_id
            self._num_devices = num_devices
            self._broadcast = ops.Broadcast(0)
            self._all_gather = ops.AllGather()

        def construct(self, q, k, v):
            q = self._broadcast((q,))[0]
            k = self._broadcast((k,))[0]
            v = self._broadcast((v,))[0]

            q_size = q.shape[1] // self._num_devices
            kv_size = k.shape[1] // self._num_devices

            q = q[:, self._rank_id * q_size : (self._rank_id + 1) * q_size]
            k = k[:, self._rank_id * kv_size : (self._rank_id + 1) * kv_size]
            v = v[:, self._rank_id * kv_size : (self._rank_id + 1) * kv_size]

            out, numerator, denominator, max_score = self._model(q, k, v)

            out = ops.concat(self._all_gather(out).chunk(self._num_devices), axis=1)
            numerator = ops.concat(self._all_gather(numerator).chunk(self._num_devices), axis=1)
            denominator = ops.concat(self._all_gather(denominator).chunk(self._num_devices), axis=1)
            max_score = ops.concat(self._all_gather(max_score).chunk(self._num_devices), axis=2)

            return out, numerator, denominator, max_score

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

    ms.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.HYBRID_PARALLEL,
        # gradients_mean=True,
        device_num=device_num,
        # search_mode="sharding_propagation"
        # enable_parallel_optimizer=True
    )

    q_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)
    k_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)
    v_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)

    ra = RingAttention(dim_head=q_.shape[-1], rank_id=rank_id, num_devices=device_num)
    ra_dist = DistribExec(ra, rank_id, device_num)

    print("Forward pass...")
    ra_dist.set_train(False)
    out1_fwd = ra_dist(q_, k_, v_)[0].asnumpy()

    print("Backward pass...")
    ra_dist.set_train(True)

    labels = ms.Tensor(np.ones((10, 256, 16, 128)), dtype=ms.float32)
    loss = nn.MSELoss()

    def forward_ra(q, k, v, label):
        z = ra_dist(q, k, v)
        loss_ = loss(z[0], label)
        return loss_

    grad1 = grad(forward_ra)(q_, k_, v_, labels).asnumpy()

    if rank_id == 0:
        vanilla_attn = Attention(q_.shape[-1])

        def _rearange_in(x: ms.Tensor):
            # (b, n, h, d) -> (b*h, n, d)
            x = x.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[3])
            return x

        def _rearange_out(x: ms.Tensor, h: int):
            # (b*h, n, d) -> (b, n, h, d)
            x = x.reshape(-1, h, x.shape[1], x.shape[2]).swapaxes(1, 2)
            return x

        def test_forward():
            vanilla_attn.set_train(False)

            out2 = vanilla_attn(_rearange_in(q_), _rearange_in(k_), _rearange_in(v_), None)
            out2 = _rearange_out(out2, q_.shape[2]).asnumpy()

            print("-" * 100)
            print(np.allclose(out1_fwd, out2, atol=1e-5))
            print(abs(out1_fwd - out2).max())

        def test_bprop():
            vanilla_attn.set_train(True)

            def forward_vanilla(q, k, v, label):
                z = vanilla_attn(q, k, v, None)
                z = _rearange_out(z, q_.shape[2])
                loss_ = loss(z, label)
                return loss_

            grad2 = grad(forward_vanilla)(_rearange_in(q_), _rearange_in(k_), _rearange_in(v_), labels)
            grad2 = _rearange_out(grad2, q_.shape[2]).asnumpy()

            print("-" * 100)
            print(np.allclose(grad1, grad2, atol=1e-5))
            print(abs(grad1 - grad2).max())

        print("Testing forward pass...")
        test_forward()
        print("Testing backward pass...")
        test_bprop()
