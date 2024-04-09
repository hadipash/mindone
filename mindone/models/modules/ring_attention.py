from mindspore import dtype, grad, nn, ops

# from .flash_attention import MSFlashAttention


class RingAttention(nn.Cell):
    def __init__(self, algorithm: str = "vanilla", rank_id: int = 0, num_devices: int = 1):
        super().__init__()
        self._num_devices = num_devices
        self._rank_id = rank_id

        send_rank = (rank_id + 1) % num_devices
        recv_rank = (rank_id - 1) % num_devices
        print(f"rank_id: {rank_id}, send_rank: {send_rank}, recv_rank: {recv_rank}")
        self._exchange = ops.NeighborExchange(
            send_rank_ids=[send_rank, send_rank],
            recv_rank_ids=[recv_rank, recv_rank],
            recv_shapes=([10, 128, 16, 128], [10, 128, 16, 128]),
            send_shapes=([10, 128, 16, 128], [10, 128, 16, 128]),
            recv_type=dtype.float32,
        )

        self.scale = 128**-0.5
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
                next_k, next_v = self._exchange((k, v))

            if step <= self._rank_id:
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
                # TODO: ensure that next_k and next_v are received
                k = next_k
                v = next_v

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
    import mindspore as ms
    import numpy as np

    from mindspore.communication import get_group_size, get_rank, init


    class DistribExec(nn.Cell):
        def __init__(self, model, rank_id, num_devices):
            super().__init__()
            self._model = model
            self._rank_id = rank_id
            self._num_devices = num_devices

        def construct(self, q, k, v):
            q_size = q.shape[1] // self._num_devices
            kv_size = k.shape[1] // self._num_devices
            return self._model(q[:, self._rank_id * q_size:(self._rank_id + 1) * q_size],
                               k[:, self._rank_id * kv_size:(self._rank_id + 1) * kv_size],
                               v[:, self._rank_id * kv_size:(self._rank_id + 1) * kv_size])


    device_id = int(os.getenv("DEVICE_ID"))
    ms.set_context(mode=ms.GRAPH_MODE, device_id=device_id)
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

    ra = RingAttention(rank_id=rank_id, num_devices=device_num)
    ra_dist = DistribExec(ra, rank_id, device_num)
    ra_dist.set_train(False)

    q_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)
    k_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)
    v_ = ms.Tensor(np.random.randn(10, 256, 16, 128), dtype=ms.float32)

    out = ra_dist(q_, k_, v_)
