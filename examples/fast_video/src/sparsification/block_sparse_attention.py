from mindspore import Tensor, mint, nn, ops

__all__ = ["BlockSparseAttention"]


class BlockSparseAttention(nn.Cell):
    """
    This Block Sparse Attention is purposefully built in a straightforward, simple way without any optimizations
    (e.g., vectorization) to measure how different sparsification techniques affect execution time.
    """

    def __init__(self, head_dim: int, block_size: int = 128):
        super().__init__()
        self._bsize = block_size
        self._scale = head_dim**-0.5
        self._bmm = ops.BatchMatMul(transpose_b=True)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, base_blockmask: Tensor) -> Tensor:
        """

        Args:
            q (b*h, n, d):
            k (b*h, n, d):
            v (b*h, n, d):
            base_blockmask (b*h, n, n):

        Returns:

        """
        if q.shape[1] % self._bsize:
            raise ValueError(
                "Q's sequence length should be divisible by the block size"
                f" ({q.shape[1]} % {self._bsize} = {q.shape[1] % self._bsize})!"
            )
        if k.shape[1] % self._bsize:
            raise ValueError(
                "K's and V's sequence length should be divisible by the block size"
                f" ({k.shape[1]} % {self._bsize} = {k.shape[1] % self._bsize})!"
            )

        # initialize variables as lists and concatenate them at the end, instead of using mint.zeros_like()
        # because slicing and then updating tensors (ScatterNdUpdate) is slow.
        out_nums, out_denoms = [], []
        for bh in range(q.shape[0]):
            nums, denoms = [], []
            for q_step in range(0, q.shape[1], self._bsize):
                prev_max_score, max_score = None, 0
                q_chunk = q[bh, q_step : q_step + self._bsize]
                numerator = mint.zeros_like(q_chunk)
                denominator = mint.zeros((q_chunk.shape[0], 1), dtype=q_chunk.dtype)

                for kv_step in range(0, k.shape[1], self._bsize):
                    if base_blockmask[bh, q_step // self._bsize, kv_step // self._bsize]:
                        k_chunk = k[bh, kv_step : kv_step + self._bsize]
                        v_chunk = v[bh, kv_step : kv_step + self._bsize]

                        attn_weights = self._bmm(q_chunk, k_chunk) * self._scale

                        max_score = attn_weights.max(axis=-1, keepdims=True)
                        if prev_max_score is None:
                            prev_max_score = max_score
                        max_score = mint.maximum(prev_max_score, max_score)

                        exp_weights = mint.exp(attn_weights - max_score)
                        correction = mint.exp(prev_max_score - max_score)

                        numerator = numerator * correction + mint.matmul(exp_weights, v_chunk)
                        denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

                        prev_max_score = max_score

                nums.append(numerator)
                denoms.append(denominator)

            out_nums.append(mint.concat(nums, dim=0))
            out_denoms.append(mint.concat(denoms, dim=0))

        out_nums = mint.stack(out_nums, dim=0)
        out_denoms = mint.stack(out_denoms, dim=0)
        out = out_nums / out_denoms

        return mint.nan_to_num(out, nan=0)
