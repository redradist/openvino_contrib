// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "details/cuda_type_traits.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernels for PagedAttentionExtension (M1 correctness kernels).
 *
 * KV cache layout (genai/OpenVINO default): both key_cache and value_cache are
 *   [num_blocks, num_kv_heads, block_size, head_size]   (row-major)
 *
 * Two kernels:
 *  - reshape_and_cache: scatter the new per-token K/V (ports 1/2) into the paged
 *    cache at positions given by `slot_mapping` (built from the block table).
 *  - paged_attention: one CTA per (tile of up to kPagedAttentionTileRows consecutive query
 *    rows of one subsequence, query head); the K/V of one cache block at a time are staged
 *    in shared memory and shared by the rows of the tile, with an online softmax over each
 *    row's causal KV horizon. Handles prefill and decode in one launch, GQA, distinct k/v
 *    head sizes, ALiBi, sliding window. Falls back to one warp per (row, head) without
 *    shared memory when the device cannot provide the shared memory of a tile.
 *
 * See docs/design-rationales/PagedAttentionExtension.md §3. head_size and
 * v_head_size must be <= 256.
 */
struct PagedAttentionParams {
    int num_heads;       // query heads H
    int num_kv_heads;    // Hk
    int head_size;       // S  (key head size)
    int v_head_size;     // SV (value head size)
    int block_size;      // 32
    float scale;         // QK scale (defaults to 1/sqrt(S) at the op level)
    int sliding_window;  // 0 == disabled
    bool has_alibi;
};

/// Scatter new K/V into the paged cache. key/value/caches are `elem_type`.
void reshape_and_cache(cudaStream_t stream,
                       Type_t elem_type,
                       const void* key,
                       const void* value,
                       void* key_cache,
                       void* value_cache,
                       const std::int32_t* slot_mapping,
                       int num_tokens,
                       int num_kv_heads,
                       int head_size,
                       int v_head_size,
                       int block_size);

/// Maximum number of query rows (of one subsequence) processed by one CTA of paged_attention.
constexpr int kPagedAttentionTileRows = 16;

/**
 * @brief Paged attention. query/caches/out are `elem_type`; metadata is i32; alibi is f32.
 * @param row_to_subseq subsequence of every query row [num_tokens]
 * @param tiles         [num_tiles][2]: first query row and subsequence of each tile; a tile
 *                      covers up to kPagedAttentionTileRows rows of its subsequence
 */
void paged_attention(cudaStream_t stream,
                     Type_t elem_type,
                     const void* query,
                     const void* key_cache,
                     const void* value_cache,
                     const std::int32_t* block_indices,
                     const std::int32_t* block_indices_begins,
                     const std::int32_t* past_lens,
                     const std::int32_t* subsequence_begins,
                     const std::int32_t* row_to_subseq,
                     const std::int32_t* tiles,
                     int num_tiles,
                     const float* alibi_slopes,
                     void* out,
                     const PagedAttentionParams& params,
                     int num_tokens);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
