from mindspore import Tensor, mint, nn, ops
import numpy as np

__all__ = ["BlockSparseAttention"]


class BlockSparseAttention(nn.Cell):
    """
    High-performance Block Sparse Attention implemented using Ascend C.
    This implementation uses block-wise computation for efficiency, leveraging
    the Ascend 910B NPU's parallel processing capabilities.
    """

    def __init__(self, head_dim: int, block_size: int = 128):
        super().__init__()
        self._bsize = block_size
        self._scale = head_dim**-0.5
        
        # Register the custom Ascend C kernel
        self._ascend_kernel = ops.Custom(
            "block_sparse_attention_ascendc",
            out_shape=lambda q, k, v, mask: q,
            out_dtype=lambda q, k, v, mask: q,
            func_type="aot",
            reg_info="block_sparse_attention_kernel"
        )

    def construct(self, q: Tensor, k: Tensor, v: Tensor, base_blockmask: Tensor) -> Tensor:
        """
        Compute block sparse attention using Ascend C kernel.
        
        Args:
            q (b*h, n, d): Query tensor
            k (b*h, n, d): Key tensor  
            v (b*h, n, d): Value tensor
            base_blockmask (b*h, n//block_size, n//block_size): Block mask tensor
            
        Returns:
            Tensor: Output tensor with same shape as q
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

        # Call the Ascend C kernel
        output = self._ascend_kernel(q, k, v, base_blockmask, 
                                   block_size=self._bsize, scale=self._scale)
        
        # Handle NaN values (equivalent to nan_to_num)
        output = mint.where(mint.isnan(output), mint.zeros_like(output), output)
        
        return output
