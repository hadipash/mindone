from mindspore import nn, ops


class BlockwiseAttention(nn.Cell):
    def __init__(self, dim_head: int, q_chunks: int, kv_chunks: int):
        super().__init__()
        self._q_chunks = q_chunks
        self._kv_chunks = kv_chunks
        self.scale = dim_head ** -0.5

    def construct(self, q, k, v):
        if q.shape[1] % self._q_chunks:
            raise ValueError("Q's sequence length should be divisible by q_chunks"
                             f" ({q.shape[1]} % {self._q_chunks} = {q.shape[1] % self._q_chunks})!")
        if k.shape[1] % self._kv_chunks:
            raise ValueError("K's and V's sequence length should be divisible by kv_chunks"
                             f" ({k.shape[1]} % {self._kv_chunks} = {k.shape[1] % self._kv_chunks})!")

        q_size = q.shape[1] // self._q_chunks
        k_size = k.shape[1] // self._kv_chunks

        nums, denoms, max_scores = [], [], []
        for step in range(0, q.shape[1], q_size):
            prev_max_score, max_score = None, 0
            numerator, denominator = 0, 0
            q_chunk = q[:, step : step + q_size]

            for k_step in range(0, k.shape[1], k_size):
                k_chunk = k[:, k_step : k_step + k_size]
                v_chunk = v[:, k_step : k_step + k_size]

                # 'bqhd,bkhd->bhqk'
                attn_weights = ops.matmul(q_chunk.swapaxes(1, 2), k_chunk.permute((0, 2, 3, 1))) * self.scale

                max_score = attn_weights.max(axis=-1, keepdims=True)
                if prev_max_score is None:
                    prev_max_score = max_score
                max_score = ops.maximum(prev_max_score, max_score)
                max_score = ops.stop_gradient(max_score)

                exp_weights = ops.exp(attn_weights - max_score)
                # 'bhqk,bkhd->bqhd'
                exp_values = ops.matmul(exp_weights, v_chunk.permute(0, 2, 1, 3)).swapaxes(1, 2)
                correction = ops.exp(prev_max_score - max_score)
                numerator = numerator * correction.swapaxes(1, 2) + exp_values
                denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

            nums.append(numerator)
            denoms.append(denominator.swapaxes(1, 2))
            max_scores.append(max_score)

        nums = ops.concat(nums, axis=1)
        denoms = ops.concat(denoms, axis=1)
        max_scores = ops.concat(max_scores, axis=2)  # bhqk
        out = nums / denoms

        return out, nums, denoms, max_scores

    def bprop(self, q, k, v, out, dout):
        out, num, denom, max_score = out
        out_grad = dout[0]

        q_size = q.shape[1] // self._q_chunks
        kv_size = k.shape[1] // self._kv_chunks

        dq, dk, dv = (map(lambda x: ops.zeros_like(x), (q, k, v)))

        for q_step in range(0, q.shape[1], q_size):
            q_chunk = q[:, q_step: q_step + q_size]
            max_score_chunk = max_score[:, :, q_step: q_step + q_size]  # bhqk
            denominator_chunk = denom[:, q_step: q_step + q_size]
            grad_chunk = out_grad[:, q_step: q_step + q_size]
            out_chunk = out[:, q_step: q_step + q_size]
            dl_part = (grad_chunk * out_chunk).sum(axis=-1, keepdims=True).swapaxes(1, 2)   # "bqhd,bqhd->bhq"


            for kv_step in range(0, k.shape[1], kv_size):
                k_chunk = k[:, kv_step: kv_step + kv_size]
                v_chunk = v[:, kv_step: kv_step + kv_size]

                # 'bqhd,bkhd->bhqk'
                attn_weights = ops.matmul(q_chunk.swapaxes(1, 2), k_chunk.permute((0, 2, 3, 1))) * self.scale

                exp_weights = ops.exp(attn_weights - max_score_chunk) / denominator_chunk.swapaxes(1, 2)

                ds = ops.matmul(grad_chunk.swapaxes(1, 2), v_chunk.permute((0, 2, 3, 1)))  # "bqhd,bkhd->bhqk"
                dl = (ds - dl_part) * exp_weights

                # 'bhqk,bkhd->bqhd'
                dq[:, q_step: q_step + q_size] += ops.matmul(dl, k_chunk.permute(0, 2, 1, 3)).swapaxes(1, 2) * self.scale
                # "bqhd,bhqk->bkhd"
                dk[:, kv_step: kv_step + kv_size] += ops.matmul(q_chunk.permute(0, 2, 3, 1), dl).permute(0, 3, 1, 2) * self.scale
                # "bhqk,bqhd->bkhd"
                dv[:, kv_step: kv_step + kv_size] += ops.matmul(exp_weights.swapaxes(2, 3), grad_chunk.swapaxes(1, 2)).swapaxes(1, 2)

        return dq, dk, dv
