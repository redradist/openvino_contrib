// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_fp16.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

#include "details/error.hpp"
#include "paged_attention.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {

// Element access helpers (cache/query stored as T, math done in float).
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ void from_float(float& dst, float v) { dst = v; }
__device__ __forceinline__ void from_float(__half& dst, float v) { dst = __float2half(v); }

constexpr int kMaxPerLane = 8;  // supports head_size <= 256 (8 * 32 lanes)

template <typename T>
__global__ void reshape_and_cache_kernel(const T* __restrict__ key,
                                         const T* __restrict__ value,
                                         T* __restrict__ key_cache,
                                         T* __restrict__ value_cache,
                                         const int* __restrict__ slot_mapping,
                                         int num_tokens,
                                         int num_kv_heads,
                                         int head_size,
                                         int v_head_size,
                                         int block_size) {
    const int token = blockIdx.x;
    const int hk = blockIdx.y;
    const int lane = threadIdx.x;
    if (token >= num_tokens || hk >= num_kv_heads) return;
    const int slot = slot_mapping[token];
    if (slot < 0) return;
    const int block = slot / block_size;
    const int offset = slot % block_size;
    if (lane < head_size) {
        const T v = key[token * (num_kv_heads * head_size) + hk * head_size + lane];
        const size_t idx = ((size_t)block * num_kv_heads + hk) * block_size * head_size +
                           (size_t)offset * head_size + lane;
        key_cache[idx] = v;
    }
    if (lane < v_head_size) {
        const T v = value[token * (num_kv_heads * v_head_size) + hk * v_head_size + lane];
        const size_t idx = ((size_t)block * num_kv_heads + hk) * block_size * v_head_size +
                           (size_t)offset * v_head_size + lane;
        value_cache[idx] = v;
    }
}

// Fallback: one warp (32 lanes) per (query token, query head). Online softmax over the
// causal KV horizon read from the paged cache.
template <typename T>
__global__ void paged_attention_rowwise_kernel(const T* __restrict__ query,
                                       const T* __restrict__ key_cache,
                                       const T* __restrict__ value_cache,
                                       const int* __restrict__ block_indices,
                                       const int* __restrict__ block_indices_begins,
                                       const int* __restrict__ past_lens,
                                       const int* __restrict__ subsequence_begins,
                                       const int* __restrict__ row_to_subseq,
                                       const float* __restrict__ alibi_slopes,
                                       T* __restrict__ out,
                                       PagedAttentionParams p,
                                       int num_tokens) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= num_tokens || h >= p.num_heads) return;

    const int i = row_to_subseq[row];
    const int hk = h / (p.num_heads / p.num_kv_heads);
    const int row_in_subseq = row - subsequence_begins[i];
    const int cur_kv_len = past_lens[i] + row_in_subseq + 1;  // causal horizon
    const int S = p.head_size, SV = p.v_head_size, BS = p.block_size;
    const float alibi = p.has_alibi ? alibi_slopes[h] : 0.0f;

    float qreg[kMaxPerLane];
    for (int t = 0; t < kMaxPerLane; ++t) {
        const int d = lane + t * 32;
        qreg[t] = (d < S) ? to_float(query[row * (p.num_heads * S) + h * S + d]) * p.scale : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[kMaxPerLane];
    for (int t = 0; t < kMaxPerLane; ++t) acc[t] = 0.0f;

    const int blk_begin = block_indices_begins[i];
    // With a sliding window only the last `sliding_window` positions are attended to.
    const int first_pos = (p.sliding_window > 0) ? max(0, cur_kv_len - p.sliding_window) : 0;
    for (int pcur = first_pos; pcur < cur_kv_len; ++pcur) {
        const int bn = block_indices[blk_begin + pcur / BS];
        if (bn < 0) continue;
        const int off = pcur % BS;
        const size_t kbase = ((size_t)bn * p.num_kv_heads + hk) * BS * S + (size_t)off * S;
        float partial = 0.0f;
        for (int t = 0; t < kMaxPerLane; ++t) {
            const int d = lane + t * 32;
            if (d < S) partial += qreg[t] * to_float(key_cache[kbase + d]);
        }
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
        float s = partial;  // query already scaled
        if (p.has_alibi) s += alibi * (float)(-(cur_kv_len - 1 - pcur));
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);
        const float e = __expf(s - m_new);
        l = l * corr + e;
        const size_t vbase = ((size_t)bn * p.num_kv_heads + hk) * BS * SV + (size_t)off * SV;
        for (int t = 0; t < kMaxPerLane; ++t) {
            const int d = lane + t * 32;
            if (d < SV) acc[t] = acc[t] * corr + e * to_float(value_cache[vbase + d]);
        }
        m = m_new;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int t = 0; t < kMaxPerLane; ++t) {
        const int d = lane + t * 32;
        if (d < SV) from_float(out[row * (p.num_heads * SV) + h * SV + d], acc[t] * inv);
    }
}

constexpr int kTileWarps = 4;
constexpr int kTileRows = kPagedAttentionTileRows;
constexpr int kRowsPerWarp = kTileRows / kTileWarps;
static_assert(kTileRows % kTileWarps == 0, "tile rows must be distributed evenly over the warps");

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

/// Shared memory of one tiled CTA: the tile's scaled queries and one cache block of K and V (f32).
std::size_t tiled_smem_bytes(int head_size, int v_head_size, int block_size) {
    return sizeof(float) * (static_cast<std::size_t>(kTileRows) * head_size +
                            static_cast<std::size_t>(block_size) * (head_size + 1) +
                            static_cast<std::size_t>(block_size) * v_head_size);
}

// One CTA (kTileWarps warps) per (tile of up to kTileRows consecutive query rows of one
// subsequence, query head). The K/V of one cache block are staged in shared memory and shared
// by all rows of the tile (prefill rows attend to the same context); warp w owns rows
// w, w + kTileWarps, ...; lane j scores key j of the block, so a block costs two warp
// reductions per row instead of a reduction per position. Requires block_size == 32.
template <typename T>
__global__ void paged_attention_tiled_kernel(const T* __restrict__ query,
                                             const T* __restrict__ key_cache,
                                             const T* __restrict__ value_cache,
                                             const int* __restrict__ block_indices,
                                             const int* __restrict__ block_indices_begins,
                                             const int* __restrict__ past_lens,
                                             const int* __restrict__ subsequence_begins,
                                             const int* __restrict__ tiles,
                                             const float* __restrict__ alibi_slopes,
                                             T* __restrict__ out,
                                             PagedAttentionParams p) {
    extern __shared__ float smem[];
    const int S = p.head_size, SV = p.v_head_size, BS = p.block_size;
    float* q_s = smem;                 // [kTileRows][S], pre-scaled
    float* k_s = q_s + kTileRows * S;  // [BS][S + 1] (padded: lane j reads row j)
    float* v_s = k_s + BS * (S + 1);   // [BS][SV]

    const int first_row = tiles[2 * blockIdx.x];
    const int i = tiles[2 * blockIdx.x + 1];
    const int h = blockIdx.y;
    const int hk = h / (p.num_heads / p.num_kv_heads);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int seq_begin = subsequence_begins[i];
    const int nrows = min(kTileRows, subsequence_begins[i + 1] - first_row);
    const int blk_begin = block_indices_begins[i];
    const float alibi = p.has_alibi ? alibi_slopes[h] : 0.0f;

    // Causal horizon (KV length) of the first and the last row of the tile.
    const int first_kv_len = past_lens[i] + (first_row - seq_begin) + 1;
    const int last_kv_len = first_kv_len + nrows - 1;
    // With a sliding window, positions before the first row's window are never visible.
    const int pos_begin = (p.sliding_window > 0) ? max(0, first_kv_len - p.sliding_window) : 0;

    for (int idx = threadIdx.x; idx < nrows * S; idx += blockDim.x) {
        const int r = idx / S, d = idx % S;
        q_s[idx] = to_float(query[(size_t)(first_row + r) * (p.num_heads * S) + h * S + d]) * p.scale;
    }

    float m[kRowsPerWarp], l[kRowsPerWarp], acc[kRowsPerWarp][kMaxPerLane];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int t = 0; t < kMaxPerLane; ++t) acc[r][t] = 0.0f;
    }

    for (int c = pos_begin / BS; c * BS < last_kv_len; ++c) {
        const int bn = block_indices[blk_begin + c];
        __syncthreads();  // the previous block has been consumed (and q_s written, on the first pass)
        if (bn >= 0) {
            const size_t kbase = ((size_t)bn * p.num_kv_heads + hk) * BS * S;
            for (int idx = threadIdx.x; idx < BS * S; idx += blockDim.x) {
                k_s[(idx / S) * (S + 1) + idx % S] = to_float(key_cache[kbase + idx]);
            }
            const size_t vbase = ((size_t)bn * p.num_kv_heads + hk) * BS * SV;
            for (int idx = threadIdx.x; idx < BS * SV; idx += blockDim.x) {
                v_s[idx] = to_float(value_cache[vbase + idx]);
            }
        }
        __syncthreads();
        if (bn < 0) continue;  // padding block: uniform across the CTA

        const int pos = c * BS + lane;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row_local = warp + r * kTileWarps;
            if (row_local < nrows) {
                const int kv_len = first_kv_len + row_local;
                const bool visible =
                    pos < kv_len && (p.sliding_window <= 0 || pos >= kv_len - p.sliding_window);
                float s = -INFINITY;
                if (visible) {
                    const float* q = q_s + row_local * S;
                    const float* k = k_s + lane * (S + 1);
                    float dot = 0.0f;
                    for (int d = 0; d < S; ++d) dot += q[d] * k[d];
                    s = dot + (p.has_alibi ? alibi * (float)(-(kv_len - 1 - pos)) : 0.0f);
                }
                const float block_max = warp_max(s);
                if (block_max != -INFINITY) {  // warp-uniform
                    const float m_new = fmaxf(m[r], block_max);
                    const float corr = __expf(m[r] - m_new);
                    const float e = visible ? __expf(s - m_new) : 0.0f;
                    l[r] = l[r] * corr + warp_sum(e);
#pragma unroll
                    for (int t = 0; t < kMaxPerLane; ++t) acc[r][t] *= corr;
                    for (int j = 0; j < BS; ++j) {
                        const float ej = __shfl_sync(0xffffffff, e, j);
                        // Invisible slots (beyond the row's horizon, outside its window) may
                        // never have been written: skip them rather than add 0 * V (NaN-unsafe).
                        if (ej == 0.0f) continue;  // warp-uniform: ej is broadcast
                        const float* v = v_s + j * SV;
#pragma unroll
                        for (int t = 0; t < kMaxPerLane; ++t) {
                            const int d = lane + t * 32;
                            if (d < SV) acc[r][t] += ej * v[d];
                        }
                    }
                    m[r] = m_new;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row_local = warp + r * kTileWarps;
        if (row_local < nrows) {
            const float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;
#pragma unroll
            for (int t = 0; t < kMaxPerLane; ++t) {
                const int d = lane + t * 32;
                if (d < SV) {
                    from_float(out[(size_t)(first_row + row_local) * (p.num_heads * SV) + h * SV + d],
                               acc[r][t] * inv);
                }
            }
        }
    }
}

template <typename T>
void launch_reshape_and_cache(cudaStream_t stream, const void* key, const void* value, void* key_cache,
                              void* value_cache, const int* slot_mapping, int num_tokens, int num_kv_heads,
                              int head_size, int v_head_size, int block_size) {
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, num_kv_heads);
    const unsigned threads = static_cast<unsigned>(std::max(head_size, v_head_size));
    reshape_and_cache_kernel<T><<<grid, threads, 0, stream>>>(
        static_cast<const T*>(key), static_cast<const T*>(value), static_cast<T*>(key_cache),
        static_cast<T*>(value_cache), slot_mapping, num_tokens, num_kv_heads, head_size, v_head_size, block_size);
}

/**
 * @brief Dynamic shared memory available to paged_attention_tiled_kernel<T> on the current
 *        device. On first use per device, the kernel's limit is raised to the device's opt-in
 *        maximum (the same value for every caller, so concurrent requests do not race).
 */
template <typename T>
std::size_t tiled_kernel_smem_limit() {
    constexpr int kMaxDevices = 64;
    static std::array<std::once_flag, kMaxDevices> once;
    static std::array<std::size_t, kMaxDevices> limit{};
    int device = 0;
    throwIfError(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) return 0;  // falls back to the row-wise kernel
    std::call_once(once[device], [device] {
        int bytes = 0;
        throwIfError(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        throwIfError(cudaFuncSetAttribute(
            paged_attention_tiled_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        limit[device] = static_cast<std::size_t>(bytes);
    });
    return limit[device];
}

template <typename T>
void launch_paged_attention(cudaStream_t stream, const void* query, const void* key_cache, const void* value_cache,
                            const int* block_indices, const int* block_indices_begins, const int* past_lens,
                            const int* subsequence_begins, const int* row_to_subseq, const int* tiles,
                            int num_tiles, const float* alibi_slopes, void* out, const PagedAttentionParams& p,
                            int num_tokens) {
    if (num_tokens == 0) return;
    const std::size_t smem = tiled_smem_bytes(p.head_size, p.v_head_size, p.block_size);
    if (p.block_size == 32 && smem <= tiled_kernel_smem_limit<T>()) {
        dim3 grid(num_tiles, p.num_heads);
        paged_attention_tiled_kernel<T><<<grid, kTileWarps * 32, smem, stream>>>(
            static_cast<const T*>(query), static_cast<const T*>(key_cache), static_cast<const T*>(value_cache),
            block_indices, block_indices_begins, past_lens, subsequence_begins, tiles, alibi_slopes,
            static_cast<T*>(out), p);
        return;
    }
    dim3 grid(num_tokens, p.num_heads);
    paged_attention_rowwise_kernel<T><<<grid, 32, 0, stream>>>(
        static_cast<const T*>(query), static_cast<const T*>(key_cache), static_cast<const T*>(value_cache),
        block_indices, block_indices_begins, past_lens, subsequence_begins, row_to_subseq, alibi_slopes,
        static_cast<T*>(out), p, num_tokens);
}

}  // namespace

void reshape_and_cache(cudaStream_t stream, Type_t elem_type, const void* key, const void* value, void* key_cache,
                       void* value_cache, const std::int32_t* slot_mapping, int num_tokens, int num_kv_heads,
                       int head_size, int v_head_size, int block_size) {
    if (head_size > kMaxPerLane * 32 || v_head_size > kMaxPerLane * 32) {
        throw_ov_exception("PagedAttention: head_size/v_head_size must be <= 256");
    }
    // Clear any stale sticky error left by earlier launches so the post-launch
    // check below reports only THIS kernel's launch status.
    (void)cudaGetLastError();
    switch (elem_type) {
        case Type_t::f32:
            launch_reshape_and_cache<float>(stream, key, value, key_cache, value_cache, slot_mapping, num_tokens,
                                            num_kv_heads, head_size, v_head_size, block_size);
            break;
        case Type_t::f16:
            launch_reshape_and_cache<__half>(stream, key, value, key_cache, value_cache, slot_mapping, num_tokens,
                                             num_kv_heads, head_size, v_head_size, block_size);
            break;
        default:
            throw_ov_exception(fmt::format("PagedAttention reshape_and_cache: unsupported element type {}",
                                           static_cast<int>(elem_type)));
    }
    throwIfError(cudaGetLastError());
}

void paged_attention(cudaStream_t stream, Type_t elem_type, const void* query, const void* key_cache,
                     const void* value_cache, const std::int32_t* block_indices,
                     const std::int32_t* block_indices_begins, const std::int32_t* past_lens,
                     const std::int32_t* subsequence_begins, const std::int32_t* row_to_subseq,
                     const std::int32_t* tiles, int num_tiles, const float* alibi_slopes, void* out,
                     const PagedAttentionParams& params, int num_tokens) {
    if (params.head_size > kMaxPerLane * 32 || params.v_head_size > kMaxPerLane * 32) {
        throw_ov_exception("PagedAttention: head_size/v_head_size must be <= 256");
    }
    if (params.num_kv_heads <= 0 || params.num_heads % params.num_kv_heads != 0) {
        throw_ov_exception("PagedAttention: num_heads must be a multiple of num_kv_heads");
    }
    // Clear any stale sticky error left by earlier launches (see reshape_and_cache).
    (void)cudaGetLastError();
    switch (elem_type) {
        case Type_t::f32:
            launch_paged_attention<float>(stream, query, key_cache, value_cache, block_indices, block_indices_begins,
                                          past_lens, subsequence_begins, row_to_subseq, tiles, num_tiles,
                                          alibi_slopes, out, params, num_tokens);
            break;
        case Type_t::f16:
            launch_paged_attention<__half>(stream, query, key_cache, value_cache, block_indices, block_indices_begins,
                                           past_lens, subsequence_begins, row_to_subseq, tiles, num_tiles,
                                           alibi_slopes, out, params, num_tokens);
            break;
        default:
            throw_ov_exception(
                fmt::format("PagedAttention: unsupported element type {}", static_cast<int>(elem_type)));
    }
    throwIfError(cudaGetLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
