# BlockSparseAttention Ascend C Implementation

This directory contains the high-performance Ascend C implementation of Block Sparse Attention for MindSpore, targeting Ascend 910B (Atlas A2) NPU.

## Overview

The BlockSparseAttention operator implements efficient sparse attention computation using block-wise masking. This implementation leverages Ascend C to achieve significant performance improvements over the naive Python implementation.

## Key Features

- **Ascend C Kernel**: Optimized kernel targeting Ascend 910B NPU
- **Block-wise Computation**: Processes attention in blocks for efficiency
- **Sparse Masking**: Skips unnecessary computation using block masks
- **Numerical Stability**: Implements stable softmax computation
- **Memory Efficiency**: Optimized memory access patterns

## Files Structure

```
sparsification/
├── block_sparse_attention.py          # Python wrapper class
├── block_sparse_attention_ascendc.cc  # Ascend C kernel implementation
├── block_sparse_attention.json        # Operator registration
├── CMakeLists.txt                    # Build configuration
├── build_ascendc.sh                  # Build script
└── README.md                        # This documentation
```

## Usage

### Python API

```python
from fastvideo.src.sparsification import BlockSparseAttention

# Initialize the attention module
head_dim = 32
block_size = 128
attention = BlockSparseAttention(head_dim, block_size)

# Input tensors
q = Tensor(...)  # Shape: (batch*heads, seq_len_q, head_dim)
k = Tensor(...)  # Shape: (batch*heads, seq_len_kv, head_dim)
v = Tensor(...)  # Shape: (batch*heads, seq_len_kv, head_dim)
mask = Tensor(...)  # Shape: (batch*heads, seq_len_q//block_size, seq_len_kv//block_size)

# Compute attention
output = attention(q, k, v, mask)
```

### Requirements

- Ascend 910B NPU (Atlas A2)
- MindSpore >= 2.6.0
- Ascend Toolkit (latest version)

### Build Instructions

1. **Prerequisites**:
   - Ascend 910B hardware
   - Ascend Toolkit installed
   - MindSpore installed

2. **Build the Ascend C kernel**:
   ```bash
   cd examples/fastvideo/fastvideo/src/sparsification/
   chmod +x build_ascendc.sh
   ./build_ascendc.sh
   ```

3. **Install the kernel**:
   ```bash
   # The build script installs to ./install/
   # Copy the built library to your Python path
   cp install/lib/libblock_sparse_attention.so /path/to/python/site-packages/
   ```

## Performance Characteristics

- **Block Size**: 128 (configurable)
- **Supported Data Types**: float16, float32
- **Memory Access**: Optimized for Ascend memory hierarchy
- **Parallelism**: Leverages Ascend 910B's parallel processing units

## Mathematical Formulation

The block sparse attention computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Where the computation is masked by block mask M:
- M[i,j] = 1: Compute the block
- M[i,j] = 0: Skip the block

## Implementation Details

### Ascend C Kernel

The kernel implements:
1. **Block-wise Loading**: Efficient loading of Q, K, V blocks
2. **Masked Computation**: Skips masked blocks entirely
3. **Stable Softmax**: Numerically stable softmax with max correction
4. **Memory Coalescing**: Optimized memory access patterns

### Memory Layout

- **Global Memory (GM)**: Stores input/output tensors
- **Unified Buffer (UB)**: Temporary storage for intermediate results
- **Scalar Buffer**: Small scalar values and constants

### Tiling Strategy

The kernel uses tiling to process large sequences efficiently:
- **Block Size**: Configurable (default 128)
- **Tile Size**: 16 for optimal Ascend utilization
- **Batch Processing**: Processes multiple heads in parallel

## Testing

Run the test suite to verify correctness:

```bash
cd examples/fastvideo/fastvideo/tests/
pytest test_blockwise_attention.py -v
```

## Troubleshooting

### Common Issues

1. **Build Errors**:
   - Ensure Ascend Toolkit is properly installed
   - Check environment variables (ASCEND_TOOLKIT_PATH, etc.)
   - Verify MindSpore version compatibility

2. **Runtime Errors**:
   - Ensure sequence lengths are divisible by block size
   - Check tensor shapes match expected format
   - Verify Ascend 910B hardware availability

3. **Performance Issues**:
   - Tune block size for your specific use case
   - Ensure proper memory alignment
   - Check for memory bandwidth bottlenecks

### Debug Mode

Enable debug logging:
```python
import os
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
```

## Performance Benchmarks

Expected performance improvements over naive Python implementation:
- **Dense Attention**: ~10-50x speedup
- **Sparse Attention**: ~20-100x speedup (depending on sparsity)

Benchmark results may vary based on:
- Sequence length
- Block size
- Sparsity pattern
- Hardware configuration

## Contributing

To contribute improvements:
1. Test on Ascend 910B hardware
2. Ensure compatibility with existing tests
3. Update documentation for any changes
4. Submit pull request with performance benchmarks

## License

This implementation follows the same license as the MindSpore project.
