from mindspore import Tensor, nn, ops

__all__ = ["BlockSparseAttention"]


@ops.kernel
def block_sparse_attention_hybrid(q, k, v, base_blockmask, scale, block_size):
    batch_heads, n, d = q.shape
    n_blocks = n // block_size
    out = output_tensor((batch_heads, n, d), q.dtype)
    num = allocate((batch_heads, n, d), q.dtype)
    den = allocate((batch_heads, n, 1), q.dtype)
    max_score = allocate((batch_heads, n), q.dtype)

    for bh in range(batch_heads):
        for i in range(n):
            max_score[bh, i] = -1e9
            for di in range(d):
                num[bh, i, di] = 0.0
            den[bh, i] = 0.0

    for bh in range(batch_heads):
        for j in range(n_blocks):
            k_start = j * block_size
            k_end = k_start + block_size
            k_block = k[bh, k_start:k_end, :]

            for i in range(n_blocks):
                if base_blockmask[bh, i, j]:
                    q_start = i * block_size
                    q_end = q_start + block_size
                    q_block = q[bh, q_start:q_end, :]
                    attn_weights = allocate((block_size, block_size), q.dtype)

                    for r in range(block_size):
                        for c in range(block_size):
                            dot = 0.0
                            for di in range(d):
                                dot += q_block[r, di] * k_block[c, di]
                            attn_weights[r, c] = dot * scale

                    for r in range(block_size):
                        global_r = q_start + r
                        row_max = attn_weights[r, 0]
                        for c in range(1, block_size):
                            if attn_weights[r, c] > row_max:
                                row_max = attn_weights[r, c]

                        new_max = max_score[bh, global_r]
                        if row_max > new_max:
                            new_max = row_max

                        exp_sum = 0.0
                        exp_weights = allocate((block_size,), q.dtype)
                        for c in range(block_size):
                            exp_val = exp(attn_weights[r, c] - new_max)
                            exp_weights[c] = exp_val
                            exp_sum += exp_val

                        correction = exp(max_score[bh, global_r] - new_max)
                        for di in range(d):
                            num[bh, global_r, di] = num[bh, global_r, di] * correction
                            weighted_val = 0.0
                            for c in range(block_size):
                                weighted_val += exp_weights[c] * v[bh, k_start + c, di]
                            num[bh, global_r, di] += weighted_val

                        den[bh, global_r] = den[bh, global_r] * correction + exp_sum
                        max_score[bh, global_r] = new_max

    for bh in range(batch_heads):
        for i in range(n):
            if den[bh, i] == 0:
                for di in range(d):
                    out[bh, i, di] = 0.0
            else:
                for di in range(d):
                    out[bh, i, di] = num[bh, i, di] / den[bh, i]

    return out


class BlockSparseAttention(nn.Cell):
    def __init__(self, head_dim: int, block_size: int = 128):
        super().__init__()
        self._bsize = block_size
        self._scale = head_dim**-0.5
        self.block_sparse_attention = ops.Custom(
            block_sparse_attention_hybrid,
            # out_shape=lambda q, k, v, base_blockmask, scale, block_size: q,
            # out_dtype=lambda q, k, v, base_blockmask, scale, block_size: q,
            func_type="aot",
        )

    def construct(self, q: Tensor, k: Tensor, v: Tensor, base_blockmask: Tensor) -> Tensor:
        if q.shape[1] % self._bsize != 0:
            raise ValueError(f"Q's sequence length {q.shape[1]} must be divisible by block size {self._bsize}")
        if k.shape[1] % self._bsize != 0:
            raise ValueError(f"K's sequence length {k.shape[1]} must be divisible by block size {self._bsize}")

        return self.block_sparse_attention(q, k, v, base_blockmask, self._scale, self._bsize)
