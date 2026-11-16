// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cuda/runtime.hpp"
#include "memory_manager/tensor_types.hpp"
#include "ops/paged_attention_mapping.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Page-locked host memory for asynchronous copies, handed out linearly during one infer.
 * Memory handed out stays valid until reset(), which the caller may only invoke once every
 * copy that uses it has completed.
 */
class PinnedArena {
public:
    void* allocate(std::size_t bytes) {
        bytes = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        if (chunks_.empty() || chunks_.back().size - chunks_.back().used < bytes) {
            const std::size_t last = chunks_.empty() ? 0 : chunks_.back().size;
            add_chunk(std::max({bytes, 2 * last, kMinChunk}));
        }
        Chunk& c = chunks_.back();
        void* p = c.data.get() + c.used;
        c.used += bytes;
        return p;
    }

    /**
     * @brief Makes all memory available again. Several chunks are merged into one big enough for
     *        all of them, unless that exceeds kMaxRetained: then the memory is released and the
     *        next infer allocates what it needs.
     */
    void reset() noexcept {
        std::size_t total = 0;
        for (const auto& c : chunks_) total += c.size;
        if (chunks_.size() > 1 || total > kMaxRetained) {
            chunks_.clear();
            if (total <= kMaxRetained) {
                try {
                    add_chunk(total);
                } catch (...) {
                    chunks_.clear();  // allocate on demand next time
                }
            }
        }
        for (auto& c : chunks_) c.used = 0;
    }

private:
    static constexpr std::size_t kAlignment = 256;
    static constexpr std::size_t kMinChunk = std::size_t{1} << 20;
    static constexpr std::size_t kMaxRetained = std::size_t{64} << 20;
    struct FreeHost {
        void operator()(uint8_t* p) const noexcept { ::logIfError(cudaFreeHost(p)); }
    };
    struct Chunk {
        std::unique_ptr<uint8_t, FreeHost> data;
        std::size_t size = 0;
        std::size_t used = 0;
    };

    void add_chunk(std::size_t size) {
        void* p = nullptr;
        ::throwIfError(cudaMallocHost(&p, size));
        chunks_.push_back(Chunk{std::unique_ptr<uint8_t, FreeHost>{static_cast<uint8_t*>(p)}, size, 0});
    }

    std::vector<Chunk> chunks_;
};

/**
 * @brief Per-request device mirrors of the paged KV caches (key_cache.N / value_cache.N).
 *
 * GenAI owns the caches as host ov::Tensors and modifies them outside the model in two
 * ways: it copies the old contents into a bigger tensor when the cache grows, and it
 * copies blocks for copy-on-write (beam search, parallel sampling, prefix caching).
 * The host tensor is therefore the source of truth, and the device copy is kept coherent
 * with it by the following invariant:
 *
 *   After every infer, every slot written by PagedAttention is identical on host and device.
 *
 * It is maintained by
 *  - bind():             growth keeps the device contents (device-to-device copy); a new or
 *                        desynchronized cache is seeded with a full host-to-device upload;
 *  - refresh_prefixes(): before a step appends into a partially filled block, the filled part
 *                        of that block is re-uploaded from the host (the block may be a
 *                        copy-on-write destination);
 *  - write_back_rows():  the K/V rows a step writes into the cache are downloaded as they are
 *                        (they are PagedAttention's key/value inputs) and finish_infer() scatters
 *                        them into the host tensor at their slots, before infer() returns (GenAI
 *                        reads the host tensor on its next schedule()).
 * Both copy directions go through page-locked staging memory, so they are asynchronous and
 * never stall the stream.
 *
 * Owned by CudaInferRequest (so it survives across infer() calls); passed by reference to
 * InferenceRequestContext. All methods are called on the thread that runs the request.
 */
class PagedAttentionContext {
public:
    /// Starts an infer.
    void begin_infer() {
        steps_.clear();
        pending_.clear();
        staging_.reset();
    }

    /**
     * @brief Binds the device mirror of cache `id` to the host tensor bound for this infer
     *        and returns its device pointer.
     */
    void* bind(BufferID id, ov::Tensor& host, const CUDA::Stream& stream) {
        const auto& shape = host.get_shape();
        OPENVINO_ASSERT(!shape.empty(), "PagedAttention: KV cache tensor must have a rank");
        const std::size_t bytes = host.get_byte_size();
        const std::size_t num_blocks = shape[0];

        auto it = caches_.find(id);
        if (it == caches_.end()) {
            it = caches_.emplace(id, Entry{}).first;
        }
        Entry& e = it->second;
        if (bytes > e.capacity) {
            // GenAI grows the cache a few blocks at a time: reserve some headroom when the
            // device has room for it, otherwise allocate exactly what is needed.
            std::size_t capacity = std::max(bytes, e.capacity + e.capacity / 2);
            CUDA::DefaultAllocation alloc{nullptr};
            try {
                alloc = CUDA::DefaultStream::stream().malloc(capacity);
            } catch (const std::exception&) {
                capacity = bytes;
                alloc = CUDA::DefaultStream::stream().malloc(capacity);
            }
            if (e.synced && e.bytes > 0) {
                // GenAI copied the old host contents into the leading part of the new tensor;
                // by the invariant they equal the old device contents.
                stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                                CUDA::DevicePointer<const void*>{e.alloc.get()},
                                e.bytes);
                // Free the old buffer right away (only after the copy has read it): GenAI grows
                // all layers at once, so keeping the old buffers until the end of the infer
                // would multiply the peak device memory.
                stream.synchronize();
            }
            e.alloc = std::move(alloc);
            e.capacity = capacity;
        }
        OPENVINO_ASSERT(shape.size() == 4, "PagedAttention: expected a [blocks, heads, block_size, head_size] KV cache");
        e.bytes = bytes;
        e.block_bytes = num_blocks ? bytes / num_blocks : 0;
        e.heads = shape[1];
        e.block_size = shape[2];
        e.row_bytes = shape[3] * host.get_element_type().size();
        e.host = static_cast<uint8_t*>(host.data());
        if (!e.synced) {
            if (bytes > 0) {
                stream.upload(CUDA::DevicePointer<void*>{e.alloc.get()}, e.host, bytes);
            }
            e.synced = true;
        }
        return e.alloc.get();
    }

    /**
     * @brief Re-uploads the first filled[i] slots of block blocks[i] of cache `id` from the
     *        host tensor bound by bind().
     */
    void refresh_prefixes(BufferID id,
                          const std::vector<int32_t>& blocks,
                          const std::vector<int32_t>& filled,
                          const CUDA::Stream& stream) {
        OPENVINO_ASSERT(blocks.size() == filled.size(), "PagedAttention: refresh blocks/filled size mismatch");
        const Entry& e = entry(id);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            const std::size_t block = static_cast<std::size_t>(blocks[i]);
            const std::size_t slots = static_cast<std::size_t>(filled[i]);
            if (slots == 0) continue;
            OPENVINO_ASSERT(slots <= e.block_size && (block + 1) * e.block_bytes <= e.bytes,
                            "PagedAttention: refreshed block is out of the KV cache");
            // [heads, block_size, row] block: the filled prefix is `slots` rows of every head.
            const std::size_t width = slots * e.row_bytes;
            const std::size_t pitch = e.block_size * e.row_bytes;
            auto* staging = static_cast<uint8_t*>(staging_.allocate(e.heads * width));
            const uint8_t* src = e.host + block * e.block_bytes;
            for (std::size_t h = 0; h < e.heads; ++h) {
                std::memcpy(staging + h * width, src + h * pitch, width);
            }
            ::throwIfError(cudaMemcpy2DAsync(static_cast<uint8_t*>(e.alloc.get()) + block * e.block_bytes,
                                                 pitch,
                                                 staging,
                                                 width,
                                                 width,
                                                 e.heads,
                                                 cudaMemcpyHostToDevice,
                                                 stream.get()));
        }
    }

    /**
     * @brief Schedules the write-back of the rows a step writes into cache `id`.
     * @param device_rows  the step's new K or V rows, [slot_mapping.size(), heads, row] on the device
     * @param slot_mapping the cache slot of every row (-1 = not written)
     * The rows are downloaded now (stream-ordered) and scattered into the host tensor by
     * finish_infer().
     */
    void write_back_rows(BufferID id,
                         const void* device_rows,
                         const std::vector<int32_t>& slot_mapping,
                         const CUDA::Stream& stream) {
        const Entry& e = entry(id);
        const std::size_t bytes = slot_mapping.size() * e.heads * e.row_bytes;
        if (bytes == 0) return;
        void* staging = staging_.allocate(bytes);
        ::throwIfError(cudaMemcpyAsync(staging, device_rows, bytes, cudaMemcpyDeviceToHost, stream.get()));
        pending_.push_back(PendingRows{e.host, e.bytes, e.block_bytes, e.block_size, e.heads, e.row_bytes,
                                       static_cast<const uint8_t*>(staging), slot_mapping});
    }

    /**
     * @brief Completes an infer whose stream has been synchronized: writes the downloaded rows
     *        into the host tensors and releases the per-infer data.
     */
    void finish_infer() {
        for (const auto& p : pending_) {
            for (std::size_t t = 0; t < p.slots.size(); ++t) {
                if (p.slots[t] < 0) continue;
                const std::size_t block = static_cast<std::size_t>(p.slots[t]) / p.block_size;
                const std::size_t offset = static_cast<std::size_t>(p.slots[t]) % p.block_size;
                OPENVINO_ASSERT((block + 1) * p.block_bytes <= p.host_bytes,
                                "PagedAttention: written slot is out of the KV cache");
                uint8_t* dst = p.host + block * p.block_bytes + offset * p.row_bytes;
                const uint8_t* src = p.rows + t * p.heads * p.row_bytes;
                for (std::size_t h = 0; h < p.heads; ++h) {
                    std::memcpy(dst + h * p.block_size * p.row_bytes, src + h * p.row_bytes, p.row_bytes);
                }
            }
        }
        pending_.clear();
        staging_.reset();
        steps_.clear();
    }

    /**
     * @brief Ends an infer that failed or was cancelled half-way on `stream`.
     *
     * Waits for the work the infer already enqueued (copies into and out of the staging memory,
     * kernels writing the device mirrors): the next infer of the request may run on another
     * stream. The pending write-back is dropped, so every device mirror is marked as not matching
     * its host tensor; the next bind() re-seeds it with a full upload.
     */
    void abort_infer(const CUDA::Stream& stream) noexcept {
        // Errors are only logged: the caller is propagating the exception that aborted the infer.
        ::logIfError(cudaStreamSynchronize(stream.get()));
        for (auto& kv : caches_) {
            kv.second.synced = false;
        }
        pending_.clear();
        steps_.clear();
        staging_.reset();
    }

    bool empty() const { return caches_.empty(); }

    /**
     * @brief Scheduling data of one infer, shared by all PagedAttention layers that read the
     *        same scheduling inputs (normally all of them): the host-side mapping and its
     *        device copies.
     */
    struct StepData {
        PagedAttentionMapping mapping;
        CUDA::Allocation slot_mapping;   ///< device copy of mapping.slot_mapping
        CUDA::Allocation row_to_subseq;  ///< device copy of mapping.row_to_subseq
        CUDA::Allocation tiles;          ///< device copy of mapping.tiles
    };

    /// Identifies the scheduling inputs of a layer within one infer: the host tensors of
    /// past_lens, subsequence_begins, block_indices, block_indices_begins, plus the step's
    /// token count and the number of blocks of the bound cache.
    struct StepKey {
        std::array<const void*, 4> metadata;
        int num_tokens;
        int64_t num_cache_blocks;
        bool operator==(const StepKey& o) const {
            return metadata == o.metadata && num_tokens == o.num_tokens && num_cache_blocks == o.num_cache_blocks;
        }
    };

    /// Uploads `mapping` to the device; the result is valid for the current infer.
    static std::unique_ptr<StepData> make_step_data(PagedAttentionMapping mapping, const CUDA::Stream& stream) {
        const auto upload = [&stream](const std::vector<int32_t>& v) {
            const std::size_t bytes = v.size() * sizeof(int32_t);
            auto alloc = stream.malloc(std::max<std::size_t>(bytes, 1));
            if (bytes) {
                stream.upload(CUDA::DevicePointer<void*>{alloc.get()}, v.data(), bytes);
            }
            return alloc;
        };
        auto slot = upload(mapping.slot_mapping);
        auto r2s = upload(mapping.row_to_subseq);
        auto tiles = upload(mapping.tiles);
        // The host vectors are moved into the result, so they outlive the (pageable, staged) uploads.
        return std::unique_ptr<StepData>(
            new StepData{std::move(mapping), std::move(slot), std::move(r2s), std::move(tiles)});
    }

    /// The step data registered for `key` during the current infer, or nullptr.
    const StepData* find_step(const StepKey& key) const {
        for (const auto& s : steps_) {
            if (s.first == key) return s.second.get();
        }
        return nullptr;
    }

    const StepData& add_step(const StepKey& key, std::unique_ptr<StepData> data) {
        steps_.emplace_back(key, std::move(data));
        return *steps_.back().second;
    }

private:
    struct Entry {
        CUDA::DefaultAllocation alloc{nullptr};
        std::size_t capacity = 0;     ///< allocated device bytes
        std::size_t bytes = 0;        ///< bytes of the currently bound host tensor
        std::size_t block_bytes = 0;  ///< bytes of one cache block
        std::size_t heads = 0;        ///< KV heads (dim 1)
        std::size_t block_size = 0;   ///< slots per block (dim 2)
        std::size_t row_bytes = 0;    ///< bytes of one slot of one head (dim 3)
        uint8_t* host = nullptr;      ///< data of the currently bound host tensor
        bool synced = false;          ///< device holds every slot written on the host
    };
    struct PendingRows {
        uint8_t* host;
        std::size_t host_bytes;
        std::size_t block_bytes;
        std::size_t block_size;
        std::size_t heads;
        std::size_t row_bytes;
        const uint8_t* rows;  ///< staging copy of the rows, [slots.size(), heads, row_bytes]
        std::vector<int32_t> slots;
    };

    const Entry& entry(BufferID id) const {
        const auto it = caches_.find(id);
        OPENVINO_ASSERT(it != caches_.end(), "PagedAttention: KV cache ", id, " is not bound");
        return it->second;
    }

    std::unordered_map<BufferID, Entry> caches_;
    std::vector<PendingRows> pending_;
    std::vector<std::pair<StepKey, std::unique_ptr<StepData>>> steps_;
    PinnedArena staging_;
};

}  // namespace nvidia_gpu
}  // namespace ov
