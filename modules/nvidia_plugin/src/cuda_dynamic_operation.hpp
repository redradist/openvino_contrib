// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cuda_creation_context.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace op {
namespace v0 {
class Parameter;
class Result;
}  // namespace v0
namespace util {
class ReadValueBase;
class AssignBase;
}  // namespace util
}  // namespace op
namespace nvidia_gpu {

class DynamicBufferContext;

/**
 * @brief Key for caching static operations by input shapes (and, for shape-
 *        dependent ops, by input values).
 *
 * Most operations' output shape is determined by input shapes alone. But ops
 * like Broadcast/Reshape derive their output shape from input VALUES (the
 * target-shape tensor), so for those the downloaded small integer values are
 * folded into the key. input_values stays empty for ordinary ops, leaving the
 * key purely shape-based.
 */
struct ShapeKey {
    std::vector<ov::Shape> input_shapes;
    std::vector<std::vector<int64_t>> input_values;

    bool operator==(const ShapeKey& other) const {
        return input_shapes == other.input_shapes && input_values == other.input_values;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& key) const {
        size_t seed = 0;
        auto mix = [&seed](size_t v) { seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
        for (const auto& shape : key.input_shapes) {
            for (auto dim : shape) {
                mix(std::hash<size_t>{}(dim));
            }
            mix(std::hash<size_t>{}(shape.size()));
        }
        for (const auto& vals : key.input_values) {
            for (auto v : vals) {
                mix(std::hash<int64_t>{}(v));
            }
            mix(std::hash<size_t>{}(vals.size()) ^ 0x1234);
        }
        return seed;
    }
};

/**
 * @brief Cached static operation variant for a specific input shape combination.
 */
struct CachedOperation {
    OperationBase::Ptr operation;
    std::vector<ov::Shape> output_shapes;
    std::vector<size_t> output_sizes;
    WorkbufferRequest workbuffer_request;
    std::vector<CUDA::DefaultAllocation> immutable_wb_allocations;
    std::vector<CUDA::DevicePointer<void*>> immutable_wb_ptrs;
};

/**
 * @brief LRU cache of the per-shape instances of the dynamic operations of one compiled model.
 *
 * Shared across all DynamicOperation instances within a CompiledModel, so the plugin holds a
 * single bounded cache for the whole model instead of one cache per operation. Entries (operation
 * identity + input shapes/values) live in a vector; a hash index maps the hash of a key to the
 * slots that hold it, so a hit neither copies the key nor scans the entries. Every lookup stamps
 * its entry from one model-wide clock; eviction removes the entry with the oldest stamp.
 *
 * Bounds:
 *  - capacity (number of entries): every dynamic operation looks up its entry on every
 *    inference, so a capacity below the number of operations would evict each entry right before
 *    it is needed again. By default the compiled model allows kEntriesPerOperation entries per
 *    dynamic operation, i.e. the instances of about that many different input shapes (prompt
 *    lengths of an LLM) for the whole model; ov::nvidia_gpu::dynamic_operation_cache_capacity
 *    overrides it. An instance costs ~1-2 KB of host memory (GPT-2, bloom-560m).
 *  - device memory: instances own device memory (their immutable workbuffers); the total is
 *    bounded by max(model constants / kConstantsPerDeviceBudget, kMinDeviceBudget).
 *  - out of device memory while creating an instance: the least recently used entry is evicted
 *    and the creation retried, until the cache is empty.
 * An evicted instance is re-created when its shape comes back, so a model whose working set does
 * not fit keeps running, only slower. An instance that an inference is executing stays alive
 * through its shared_ptr until the inference is done with it.
 */
class DynamicOperationCache {
public:
    static constexpr size_t kDefaultCapacity = 300;
    static constexpr size_t kEntriesPerOperation = 32;
    static constexpr size_t kConstantsPerDeviceBudget = 4;
    static constexpr size_t kMinDeviceBudget = size_t{64} << 20;

    explicit DynamicOperationCache(size_t capacity = kDefaultCapacity) : capacity_{std::max<size_t>(capacity, 1)} {}

    /// Sizes the cache for a model with `num_operations` dynamic operations whose constant tensors
    /// take `constant_bytes` of device memory. `capacity` == 0 selects kEntriesPerOperation entries
    /// per operation.
    void configure(size_t num_operations, size_t constant_bytes, size_t capacity) {
        std::vector<std::shared_ptr<CachedOperation>> evicted;  // released after unlocking
        std::lock_guard<std::mutex> lock{mutex_};
        capacity_ = std::max<size_t>(capacity != 0 ? capacity : num_operations * kEntriesPerOperation, 1);
        device_budget_ = std::max(constant_bytes / kConstantsPerDeviceBudget, kMinDeviceBudget);
        entries_.reserve(std::min(capacity_, num_operations * kEntriesPerOperation));
        shrink(nullptr, evicted);
    }

    void set_capacity(size_t capacity) {
        std::vector<std::shared_ptr<CachedOperation>> evicted;
        std::lock_guard<std::mutex> lock{mutex_};
        capacity_ = std::max<size_t>(capacity, 1);
        shrink(nullptr, evicted);
    }

    void set_device_budget(size_t bytes) {
        std::vector<std::shared_ptr<CachedOperation>> evicted;
        std::lock_guard<std::mutex> lock{mutex_};
        device_budget_ = bytes;
        shrink(nullptr, evicted);
    }

    size_t capacity() {
        std::lock_guard<std::mutex> lock{mutex_};
        return capacity_;
    }

    size_t device_budget() {
        std::lock_guard<std::mutex> lock{mutex_};
        return device_budget_;
    }

    /// Device memory held by the cached instances.
    size_t device_bytes() {
        std::lock_guard<std::mutex> lock{mutex_};
        return device_bytes_;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock{mutex_};
        return entries_.size();
    }

    /**
     * Returns the instance of operation `op_id` for `key`, creating it with `factory()` on a miss
     * (outside the lock, so other operations proceed meanwhile).
     */
    template <typename Factory>
    std::shared_ptr<CachedOperation> getOrCreate(const void* op_id, const ShapeKey& key, Factory factory) {
        const size_t hash = hashOf(op_id, key);
        {
            std::lock_guard<std::mutex> lock{mutex_};
            const size_t slot = find(op_id, key, hash);
            if (slot != kNone) {
                entries_[slot].last_use = ++clock_;
                return entries_[slot].operation;
            }
        }
        std::shared_ptr<CachedOperation> created = create(factory);
        // Declared before the lock: evicted instances free their device memory after unlocking.
        std::vector<std::shared_ptr<CachedOperation>> evicted;
        std::lock_guard<std::mutex> lock{mutex_};
        const size_t slot = find(op_id, key, hash);
        if (slot != kNone) {  // created meanwhile by another inference
            entries_[slot].last_use = ++clock_;
            return entries_[slot].operation;
        }
        const size_t bytes = deviceBytesOf(*created);
        entries_.push_back(Entry{op_id, key, hash, created, bytes, ++clock_});
        index_.emplace(hash, entries_.size() - 1);
        device_bytes_ += bytes;
        shrink(created.get(), evicted);
        return created;
    }

private:
    static constexpr size_t kNone = static_cast<size_t>(-1);

    struct Entry {
        const void* op_id;
        ShapeKey key;
        size_t hash;
        std::shared_ptr<CachedOperation> operation;
        size_t device_bytes;
        uint64_t last_use;
    };

    static size_t hashOf(const void* op_id, const ShapeKey& key) {
        size_t seed = std::hash<const void*>{}(op_id);
        seed ^= ShapeKeyHash{}(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    static size_t deviceBytesOf(const CachedOperation& operation) {
        size_t bytes = 0;
        for (size_t size : operation.workbuffer_request.immutable_sizes) {
            bytes += std::max(size, size_t{1});
        }
        return bytes;
    }

    size_t find(const void* op_id, const ShapeKey& key, size_t hash) const {
        const auto range = index_.equal_range(hash);
        for (auto it = range.first; it != range.second; ++it) {
            const Entry& entry = entries_[it->second];
            if (entry.op_id == op_id && entry.key == key) {
                return it->second;
            }
        }
        return kNone;
    }

    /// Makes the index entry (hash -> from) point to `to`, or removes it when `to` is kNone.
    void reindex(size_t hash, size_t from, size_t to) {
        const auto range = index_.equal_range(hash);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == from) {
                if (to == kNone) {
                    index_.erase(it);
                } else {
                    it->second = to;
                }
                return;
            }
        }
    }

    template <typename Factory>
    std::shared_ptr<CachedOperation> create(Factory& factory) {
        while (true) {
            try {
                return factory();
            } catch (...) {
                // CUDA records the error of the failed call; an out-of-memory error is not sticky.
                if (cudaGetLastError() != cudaErrorMemoryAllocation) {
                    throw;
                }
                std::shared_ptr<CachedOperation> victim;
                {
                    std::lock_guard<std::mutex> lock{mutex_};
                    victim = evictLeastRecentlyUsed(nullptr);
                }
                if (!victim) {
                    throw;
                }
            }  // `victim` frees its device memory here unless an inference still executes it
        }
    }

    /// Evicts least recently used entries (never `keep`) until the bounds hold.
    void shrink(const CachedOperation* keep, std::vector<std::shared_ptr<CachedOperation>>& evicted) {
        while (entries_.size() > capacity_ || device_bytes_ > device_budget_) {
            auto victim = evictLeastRecentlyUsed(keep);
            if (!victim) {
                break;  // only `keep` is left: it is needed right now
            }
            evicted.push_back(std::move(victim));
        }
    }

    /// Removes the least recently used entry except the one of `keep`; nullptr when there is none.
    std::shared_ptr<CachedOperation> evictLeastRecentlyUsed(const CachedOperation* keep) {
        size_t victim = kNone;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].operation.get() != keep &&
                (victim == kNone || entries_[i].last_use < entries_[victim].last_use)) {
                victim = i;
            }
        }
        if (victim == kNone) {
            return nullptr;
        }
        std::shared_ptr<CachedOperation> operation = std::move(entries_[victim].operation);
        device_bytes_ -= entries_[victim].device_bytes;
        reindex(entries_[victim].hash, victim, kNone);
        const size_t last = entries_.size() - 1;
        if (victim != last) {  // move the last entry into the hole
            reindex(entries_[last].hash, last, victim);
            entries_[victim] = std::move(entries_[last]);
        }
        entries_.pop_back();
        return operation;
    }

    std::mutex mutex_;
    std::vector<Entry> entries_;
    std::unordered_multimap<size_t, size_t> index_;  // hash of the key -> slot in entries_
    uint64_t clock_ = 0;
    size_t capacity_;
    size_t device_bytes_ = 0;
    size_t device_budget_ = kMinDeviceBudget;
};

/**
 * @brief Delegator operation for nodes with dynamic shapes.
 *
 * Sits in exec_sequence_ like a normal operation. On Execute() it reads the
 * actual input shapes from DynamicBufferContext, looks up (or creates) a cached
 * static operation for those shapes in the model-global cache, allocates
 * dynamic GPU memory for the outputs, delegates execution, and registers the
 * output buffers/shapes. Parameter/Result/ReadValue/Assign are handled as
 * boundary conditions. CUDA Graphs are incompatible with dynamic shapes.
 */
class DynamicOperation : public OperationBase {
public:
    DynamicOperation(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override {
        return CudaGraphCompatibility::NONE;
    }

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {};
    }

    void setBuffersToRelease(std::vector<BufferID> ids) {
        release_ids_ = std::move(ids);
    }

private:
    bool computeIsPagedAttentionCache() const;

    void executeParameter(const ov::op::v0::Parameter& paramNode,
                          const InferenceRequestContext& context,
                          const CUDA::Stream& stream,
                          DynamicBufferContext& dynBufCtx) const;

    void executeResult(const ov::op::v0::Result& resultNode,
                       const InferenceRequestContext& context,
                       const CUDA::Stream& stream,
                       DynamicBufferContext& dynBufCtx) const;

    void executeReadValue(const ov::op::util::ReadValueBase& readValueNode,
                          const InferenceRequestContext& context,
                          const CUDA::Stream& stream,
                          Inputs inputTensors,
                          DynamicBufferContext& dynBufCtx) const;

    void executeAssign(const ov::op::util::AssignBase& assignNode,
                       const InferenceRequestContext& context,
                       const CUDA::Stream& stream,
                       Inputs inputTensors,
                       DynamicBufferContext& dynBufCtx) const;

    // PagedAttentionExtension: writes new K/V into a per-request persistent device
    // cache and computes paged attention. Handled here (not via the generic cached
    // path) so it can key the persistent cache by the real cache-input BufferIDs.
    void executePagedAttention(const InferenceRequestContext& context,
                               const CUDA::Stream& stream,
                               Inputs inputTensors,
                               DynamicBufferContext& dynBufCtx) const;

    // The prepared state of the cached path: the cached static operation, the
    // resolved input pointers, and the freshly allocated output buffers. Built
    // by prepareCachedOperationContext(), then consumed by executeCachedOperation()
    // and finalizeOutputs().
    struct CachedOperationContext {
        std::shared_ptr<CachedOperation> cached;
        std::vector<CUDA::DevicePointer<const void*>> input_ptrs;
        std::vector<CUDA::Allocation> output_allocs;
    };

    // Collect input shapes/values, resolve input pointers, look up or create the
    // cached static op, and allocate outputs.
    CachedOperationContext prepareCachedOperationContext(const InferenceRequestContext& context,
                                                         const CUDA::Stream& stream,
                                                         Inputs inputTensors,
                                                         DynamicBufferContext& dynBufCtx) const;

    // Produce this op's output: a zero-copy input -> output copy for
    // Reshape/Squeeze/Unsqueeze (modeled by the registry as a NopOp), otherwise
    // the cached static kernel (skipped for a zero-element output, modeled by a
    // null operation).
    void executeCachedOperation(const InferenceRequestContext& context,
                                const CUDA::Stream& stream,
                                const CachedOperationContext& ctx) const;

    // Register output buffers/shapes and release buffers whose last consumer is
    // this operation.
    void finalizeOutputs(DynamicBufferContext& dynBufCtx, CachedOperationContext& ctx) const;

    std::shared_ptr<CachedOperation> createCachedOperation(
        const ShapeKey& key,
        const std::vector<CUDA::DevicePointer<const void*>>& input_ptrs,
        const CUDA::Stream& stream) const;

    std::shared_ptr<ov::Node> original_node_;
    CreationContext creation_context_;

    // Real output IDs stored separately. OperationBase gets empty output IDs
    // so that MemoryManager::outputTensorPointers() doesn't try to resolve
    // dynamic output buffers that haven't been allocated yet.
    IndexCollection dynamic_output_ids_;

    // BufferIDs to release from DynamicBufferContext after Execute().
    // Set by SubGraph::initExecuteSequence() based on buffer lifespan analysis.
    std::vector<BufferID> release_ids_;

    // True when this op's output shape stays dynamic even with concrete input
    // shapes, i.e. it depends on input VALUES (e.g. Broadcast/Reshape target
    // shape). When set, Execute() folds those values into the cache key. Computed
    // once at construction (single-threaded) via computeHasDynamicOutput() so the
    // first Execute is already value-sensitive -- never mutated afterwards, so the
    // shared instance is safe to read concurrently from multiple request threads.
    bool has_dynamic_output_ = false;
    // A KV cache Parameter consumed only by PagedAttentionExtension (no per-infer upload).
    bool is_paged_attention_cache_ = false;

    // PagedAttentionExtension inputs 9-11 folded to constants at construction (single-threaded:
    // constant folding evaluates bounds on the shared model). Empty = runtime value or absent.
    struct PagedAttentionConstants {
        std::optional<float> scale;
        std::optional<int> sliding_window;
        std::optional<std::vector<float>> alibi_slopes;
    };
    PagedAttentionConstants pa_constants_;
    // Device copy of pa_constants_.alibi_slopes, uploaded once on first use (shared by the
    // infer requests of the compiled model).
    mutable std::once_flag pa_alibi_once_;
    mutable std::optional<CUDA::DefaultAllocation> pa_alibi_device_;

    // Probe, at construction, whether the output shape is value-dependent.
    bool computeHasDynamicOutput() const;
};

}  // namespace nvidia_gpu
}  // namespace ov
