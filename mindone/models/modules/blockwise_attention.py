from typing import Tuple

from mindspore import Tensor, nn, ops


class BlockwiseAttention(nn.Cell):
    def __init__(self, head_dim: int, q_chunks: int, kv_chunks: int):
        super().__init__()
        self._q_chunks = q_chunks
        self._kv_chunks = kv_chunks
        self.scale = head_dim**-0.5
        self.b_matmul = ops.BatchMatMul(transpose_b=True)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Args:
            q (b*h, n, d):
            k (b*h, n, d):
            v (b*h, n, d):

        Returns:

        """
        if q.shape[1] % self._q_chunks:
            raise ValueError(
                "Q's sequence length should be divisible by q_chunks"
                f" ({q.shape[1]} % {self._q_chunks} = {q.shape[1] % self._q_chunks})!"
            )
        if k.shape[1] % self._kv_chunks:
            raise ValueError(
                "K's and V's sequence length should be divisible by kv_chunks"
                f" ({k.shape[1]} % {self._kv_chunks} = {k.shape[1] % self._kv_chunks})!"
            )

        q_size = q.shape[1] // self._q_chunks
        k_size = k.shape[1] // self._kv_chunks

        # initialize variables as lists and concatenate them at the end, instead of using ops.zeros_like()
        # because slicing and then updating tensors (ScatterNdUpdate) is slow.
        nums, denoms, max_scores = [], [], []
        for q_step in range(0, q.shape[1], q_size):
            prev_max_score, max_score = None, 0
            numerator, denominator = 0, 0
            q_chunk = q[:, q_step : q_step + q_size]

            for kv_step in range(0, k.shape[1], k_size):
                k_chunk = k[:, kv_step : kv_step + k_size]
                v_chunk = v[:, kv_step : kv_step + k_size]

                attn_weights = self.b_matmul(q_chunk, k_chunk) * self.scale

                max_score = attn_weights.max(axis=-1, keepdims=True)
                if prev_max_score is None:
                    prev_max_score = max_score
                max_score = ops.maximum(prev_max_score, max_score)
                max_score = ops.stop_gradient(max_score)

                exp_weights = ops.exp(attn_weights - max_score)
                correction = ops.exp(prev_max_score - max_score)

                numerator = numerator * correction + ops.matmul(exp_weights, v_chunk)
                denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)

                prev_max_score = max_score

            nums.append(numerator)
            denoms.append(denominator)
            max_scores.append(max_score)

        nums = ops.concat(nums, axis=1)
        denoms = ops.concat(denoms, axis=1)
        max_scores = ops.concat(max_scores, axis=1)
        out = nums / denoms

        return out, denoms, max_scores

    def bprop(self, q, k, v, out, dout):
        out, denominator, max_score = out
        out_grad = dout[0]

        q_size = q.shape[1] // self._q_chunks
        kv_size = k.shape[1] // self._kv_chunks

        # initialize variables as lists and concatenate them at the end, instead of using ops.zeros_like()
        # because slicing and then updating tensors (ScatterNdUpdate) is slow.
        dq, dk, dv = [0] * self._q_chunks, [0] * self._kv_chunks, [0] * self._kv_chunks
        for q_step in range(self._q_chunks):
            q_chunk = q[:, q_step * q_size : (q_step + 1) * q_size]
            max_score_chunk = max_score[:, q_step * q_size : (q_step + 1) * q_size]
            denominator_chunk = denominator[:, q_step * q_size : (q_step + 1) * q_size]
            grad_chunk = out_grad[:, q_step * q_size : (q_step + 1) * q_size]
            out_chunk = out[:, q_step * q_size : (q_step + 1) * q_size]
            dl_part = (grad_chunk * out_chunk).sum(axis=-1, keepdims=True)

            for kv_step in range(self._kv_chunks):
                k_chunk = k[:, kv_step * kv_size : (kv_step + 1) * kv_size]
                v_chunk = v[:, kv_step * kv_size : (kv_step + 1) * kv_size]

                attn_weights = self.b_matmul(q_chunk, k_chunk) * self.scale
                exp_weights = ops.exp(attn_weights - max_score_chunk) / denominator_chunk

                ds = self.b_matmul(grad_chunk, v_chunk)
                dl = (ds - dl_part) * exp_weights

                dq[q_step] += ops.matmul(dl, k_chunk) * self.scale
                dk[kv_step] += ops.matmul(q_chunk.swapaxes(1, 2), dl).swapaxes(1, 2) * self.scale
                dv[kv_step] += ops.matmul(exp_weights.swapaxes(1, 2), grad_chunk)

        return ops.concat(dq, axis=1), ops.concat(dk, axis=1), ops.concat(dv, axis=1)
