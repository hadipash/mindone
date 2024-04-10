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

    # def bprop(self, q, k, v, out, dout):
    #     kv_comm = RingComm(process_group)
    #     d_kv_comm = RingComm(process_group)
    #     dq, dk, dv = None, None, None
    #     next_dk, next_dv = None, None
    #
    #     block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    #     block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    #     block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    #
    #     next_dk, next_dv = None, None
    #     next_k, next_v = None, None
    #
    #     for step in range(kv_comm.world_size):
    #         if step + 1 != kv_comm.world_size:
    #             next_k = kv_comm.send_recv(k)
    #             next_v = kv_comm.send_recv(v)
    #             kv_comm.commit()
    #         if step <= kv_comm.rank or not causal:
    #             bwd_causal = causal and step == 0
    #             dq, dk, dv, _, _, _, _ = grad(self.attn)(q, k, v, out, dout)
    #
    #             if dq is None:
    #                 dq = block_dq_buffer.to(torch.float32)
    #                 dk = block_dk_buffer.to(torch.float32)
    #                 dv = block_dv_buffer.to(torch.float32)
    #             else:
    #                 dq += block_dq_buffer
    #                 d_kv_comm.wait()
    #                 dk = block_dk_buffer + next_dk
    #                 dv = block_dv_buffer + next_dv
    #         elif step != 0:
    #             d_kv_comm.wait()
    #             dk = next_dk
    #             dv = next_dv
    #
    #         if step + 1 != kv_comm.world_size:
    #             kv_comm.wait()
    #             k = next_k
    #             v = next_v
    #
    #         next_dk = d_kv_comm.send_recv(dk)
    #         next_dv = d_kv_comm.send_recv(dv)
    #         d_kv_comm.commit()
    #
    #     d_kv_comm.wait()
    #
    #     return dq, next_dk, next_dv


if __name__ == '__main__':
    import os
    import sys
    import mindspore as ms
    import numpy as np

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

            q = q[:, self._rank_id * q_size:(self._rank_id + 1) * q_size]
            k = k[:, self._rank_id * kv_size:(self._rank_id + 1) * kv_size]
            v = v[:, self._rank_id * kv_size:(self._rank_id + 1) * kv_size]

            out, numerator, denominator, max_score = self._model(q, k, v)

            out = ops.concat(self._all_gather(out).chunk(self._num_devices), axis=1)
            numerator = ops.concat(self._all_gather(numerator).chunk(self._num_devices), axis=1)
            denominator = ops.concat(self._all_gather(denominator).chunk(self._num_devices), axis=1)
            max_score = ops.concat(self._all_gather(max_score).chunk(self._num_devices), axis=2)

            return out, numerator, denominator, max_score


    device_id = int(os.getenv("DEVICE_ID"))
    ms.set_context(mode=ms.GRAPH_MODE, device_id=device_id, deterministic="ON",
                   ascend_config={"precision_mode": "must_keep_origin_dtype"})
    init()
    device_num = get_group_size()
    rank_id = get_rank()
    print(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {device_num}")
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
    ra_dist.set_train(False)

    out_ = ra_dist(q_, k_, v_)
    for item in out_:
        print("-" * 100, "\n", item.shape)

    if rank_id == 0:
        def _rearange_in(x: ms.Tensor):
            # (b, n, h, d) -> (b*h, n, d)
            x = x.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[3])
            return x


        def _rearange_out(x: ms.Tensor, h: int):
            # (b*h, n, d) -> (b, n, h, d)
            x = x.reshape(-1, h, x.shape[1], x.shape[2]).swapaxes(1, 2)
            return x


        def test_forward():
            attn2 = Attention(q_.shape[-1])
            attn2.set_train(False)

            out1 = out_[0].asnumpy()
            out2 = attn2(_rearange_in(q_), _rearange_in(k_), _rearange_in(v_), None)
            out2 = _rearange_out(out2, q_.shape[2]).asnumpy()

            print("-" * 100)
            print(np.allclose(out1, out2, atol=1e-5))
            print(abs(out1 - out2).max())

        test_forward()
