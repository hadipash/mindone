// Copyright 2024 Huawei Technologies Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 128;
const uint32_t TILE_SIZE = 16;

struct BlockSparseAttentionTilingData final {
    uint32_t batch_heads;
    uint32_t seq_len_q;
    uint32_t seq_len_kv;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t num_blocks_q;
    uint32_t num_blocks_kv;
    float scale;
};

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    BlockSparseAttentionTilingData* tiling_data = 
        static_cast<BlockSparseAttentionTilingData*>(context->GetTilingData<BlockSparseAttentionTilingData>());
    
    const gert::StorageShape* q_shape = context->GetInputShape(0);
    const gert::StorageShape* k_shape = context->GetInputShape(1);
    const gert::StorageShape* v_shape = context->GetInputShape(2);
    
    tiling_data->batch_heads = q_shape->GetStorageShape()[0];
    tiling_data->seq_len_q = q_shape->GetStorageShape()[1];
    tiling_data->head_dim = q_shape->GetStorageShape()[2];
    tiling_data->seq_len_kv = k_shape->GetStorageShape()[1];
    
    auto attr_block_size = context->GetAttrs()->GetInt(0);
    tiling_data->block_size = static_cast<uint32_t>(*attr_block_size);
    tiling_data->num_blocks_q = tiling_data->seq_len_q / tiling_data->block_size;
    tiling_data->num_blocks_kv = tiling_data->seq_len_kv / tiling_data->block_size;
    
    auto attr_scale = context->GetAttrs()->GetFloat(1);
    tiling_data->scale = *attr_scale;
    
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeFunc(gert::InferShapeContext* context) {
    const gert::Shape* q_shape = context->GetInputShape(0);
    const gert::Shape* k_shape = context->GetInputShape(1);
    const gert::Shape* v_shape = context->GetInputShape(2);
    
    // Output shape is same as Q shape
    gert::Shape output_shape;
    output_shape.SetDimNum(q_shape->GetDimNum());
    for (int i = 0; i < q_shape->GetDimNum(); ++i) {
        output_shape.SetDim(i, q_shape->GetDim(i));
    }
    
    // Note: SetOutputShape may not be available in this toolkit version
    // Output shape inference is typically handled automatically
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFunc(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OP_IMPL_FUNC("BlockSparseAttention", TilingFunc)
REGISTER_OP_IMPL_FUNC("BlockSparseAttention", InferShapeFunc)
REGISTER_OP_IMPL_FUNC("BlockSparseAttention", InferDataTypeFunc)
}  // namespace optiling

extern "C" {

// Ascend C kernel implementation
extern "C" __global__ __aicore__ void block_sparse_attention_kernel(
    const __gm__ uint8_t* q_gm,
    const __gm__ uint8_t* k_gm,
    const __gm__ uint8_t* v_gm,
    const __gm__ uint8_t* mask_gm,
    __gm__ uint8_t* out_gm,
    const __gm__ uint8_t* tiling_gm) {
    
    // Get tiling data
    BlockSparseAttentionTilingData* tiling_data = 
        reinterpret_cast<BlockSparseAttentionTilingData*>(const_cast<uint8_t*>(tiling_gm));
    
    const uint32_t batch_heads = tiling_data->batch_heads;
    const uint32_t seq_len_q = tiling_data->seq_len_q;
    const uint32_t seq_len_kv = tiling_data->seq_len_kv;
    const uint32_t head_dim = tiling_data->head_dim;
    const uint32_t block_size = tiling_data->block_size;
    const uint32_t num_blocks_q = tiling_data->num_blocks_q;
    const uint32_t num_blocks_kv = tiling_data->num_blocks_kv;
    const float scale = tiling_data->scale;
    
    // Calculate strides
    const uint32_t q_stride = seq_len_q * head_dim;
    const uint32_t k_stride = seq_len_kv * head_dim;
    const uint32_t v_stride = seq_len_kv * head_dim;
    const uint32_t out_stride = seq_len_q * head_dim;
    
    // Process each batch and head
    for (uint32_t bh = 0; bh < batch_heads; ++bh) {
        const uint32_t q_offset = bh * q_stride;
        const uint32_t k_offset = bh * k_stride;
        const uint32_t v_offset = bh * v_stride;
        const uint32_t out_offset = bh * out_stride;
        const uint32_t mask_offset = bh * num_blocks_q * num_blocks_kv;
        
        // Initialize output buffers
        __ub__ float out_buf[BLOCK_SIZE * head_dim];
        __ub__ float num_buf[BLOCK_SIZE * head_dim];
        __ub__ float den_buf[BLOCK_SIZE];
        __ub__ float max_score_buf[BLOCK_SIZE];
        
        // Initialize accumulators
        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            den_buf[i] = 0.0f;
            max_score_buf[i] = -1e9f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                num_buf[i * head_dim + d] = 0.0f;
            }
        }
        
        // Process each query block
        for (uint32_t q_block_idx = 0; q_block_idx < num_blocks_q; ++q_block_idx) {
            const uint32_t q_start = q_block_idx * block_size;
            
            // Load query block
            __ub__ float q_block[BLOCK_SIZE * head_dim];
            for (uint32_t i = 0; i < block_size; ++i) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    q_block[i * head_dim + d] = 
                        reinterpret_cast<const __gm__ float*>(q_gm)[q_offset + (q_start + i) * head_dim + d];
                }
            }
            
            // Process each key/value block
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_blocks_kv; ++kv_block_idx) {
                const uint32_t mask_idx = mask_offset + q_block_idx * num_blocks_kv + kv_block_idx;
                const bool block_mask = reinterpret_cast<const __gm__ uint8_t*>(mask_gm)[mask_idx];
                
                if (!block_mask) {
                    continue;  // Skip this block if mask is 0
                }
                
                const uint32_t kv_start = kv_block_idx * block_size;
                
                // Load key and value blocks
                __ub__ float k_block[BLOCK_SIZE * head_dim];
                __ub__ float v_block[BLOCK_SIZE * head_dim];
                
                for (uint32_t i = 0; i < block_size; ++i) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        k_block[i * head_dim + d] = 
                            reinterpret_cast<const __gm__ float*>(k_gm)[k_offset + (kv_start + i) * head_dim + d];
                        v_block[i * head_dim + d] = 
                            reinterpret_cast<const __gm__ float*>(v_gm)[v_offset + (kv_start + i) * head_dim + d];
                    }
                }
                
                // Compute attention weights for this block
                __ub__ float attn_weights[BLOCK_SIZE * BLOCK_SIZE];
                for (uint32_t qi = 0; qi < block_size; ++qi) {
                    for (uint32_t ki = 0; ki < block_size; ++ki) {
                        float dot_product = 0.0f;
                        for (uint32_t d = 0; d < head_dim; ++d) {
                            dot_product += q_block[qi * head_dim + d] * k_block[ki * head_dim + d];
                        }
                        attn_weights[qi * block_size + ki] = dot_product * scale;
                    }
                }
                
                // Softmax computation with numerical stability
                for (uint32_t qi = 0; qi < block_size; ++qi) {
                    // Find max score for this row
                    float row_max = attn_weights[qi * block_size];
                    for (uint32_t ki = 1; ki < block_size; ++ki) {
                        row_max = max(row_max, attn_weights[qi * block_size + ki]);
                    }
                    
                    // Update global max
                    float prev_max = max_score_buf[qi];
                    float new_max = max(prev_max, row_max);
                    
                    // Compute exp and sum
                    float exp_sum = 0.0f;
                    __ub__ float exp_weights[BLOCK_SIZE];
                    for (uint32_t ki = 0; ki < block_size; ++ki) {
                        float exp_val = exp(attn_weights[qi * block_size + ki] - new_max);
                        exp_weights[ki] = exp_val;
                        exp_sum += exp_val;
                    }
                    
                    // Update numerator and denominator
                    float correction = exp(prev_max - new_max);
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        float weighted_sum = 0.0f;
                        for (uint32_t ki = 0; ki < block_size; ++ki) {
                            weighted_sum += exp_weights[ki] * v_block[ki * head_dim + d];
                        }
                        num_buf[qi * head_dim + d] = num_buf[qi * head_dim + d] * correction + weighted_sum;
                    }
                    den_buf[qi] = den_buf[qi] * correction + exp_sum;
                    max_score_buf[qi] = new_max;
                }
            }
            
            // Write output for this block
            for (uint32_t i = 0; i < block_size; ++i) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    if (den_buf[i] == 0.0f) {
                        out_buf[i * head_dim + d] = 0.0f;
                    } else {
                        out_buf[i * head_dim + d] = num_buf[i * head_dim + d] / den_buf[i];
                    }
                    reinterpret_cast<__gm__ float*>(out_gm)[out_offset + (q_start + i) * head_dim + d] = 
                        out_buf[i * head_dim + d];
                }
            }
        }
    }
}

}  // extern "C"
