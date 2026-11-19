// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cuda/runtime.hpp>
#include <list>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_dynamic_operation.hpp"

using ov::nvidia_gpu::CachedOperation;
using ov::nvidia_gpu::DynamicOperationCache;
using ov::nvidia_gpu::ShapeKey;

namespace {

ShapeKey shapeKey(size_t length) { return ShapeKey{{ov::Shape{1, length, 8}}, {}}; }

/// Counts its calls and creates instances holding `device_bytes` of device memory.
struct Factory {
    size_t device_bytes = 0;
    int calls = 0;
    std::shared_ptr<CachedOperation> operator()() {
        ++calls;
        auto operation = std::make_shared<CachedOperation>();
        if (device_bytes > 0) {
            operation->workbuffer_request.immutable_sizes.push_back(device_bytes);
        }
        return operation;
    }
};

std::shared_ptr<CachedOperation> lookup(DynamicOperationCache& cache,
                                        const void* op_id,
                                        size_t length,
                                        Factory& factory) {
    return cache.getOrCreate(op_id, shapeKey(length), [&factory] { return factory(); });
}

/// Fails like an instance whose workbuffers do not fit into device memory (cudaErrorMemoryAllocation).
void allocateTooMuchDeviceMemory() { CUDA::DefaultStream::stream().malloc(size_t{1} << 52); }

}  // namespace

TEST(DynamicOperationCache, CyclicAccessOverManyOperationsCreatesEachOnce) {
    // A model's dynamic operations are looked up in the same order on every inference.
    DynamicOperationCache cache;
    std::vector<char> ops(1000);
    cache.configure(ops.size(), 0, 0);
    Factory factory;
    for (int round = 0; round < 3; ++round) {
        for (const char& op : ops) {
            lookup(cache, &op, 1, factory);
        }
    }
    EXPECT_EQ(factory.calls, 1000);
    EXPECT_EQ(cache.size(), 1000u);
}

TEST(DynamicOperationCache, CapacityBelowTheOperationsKeepsWorking) {
    // The working set of one inference does not fit: every lookup re-creates its instance (slow, but works).
    DynamicOperationCache cache{500};
    std::vector<char> ops(1000);
    Factory factory;
    for (int round = 0; round < 3; ++round) {
        for (const char& op : ops) {
            ASSERT_NE(lookup(cache, &op, 1, factory), nullptr);
        }
    }
    EXPECT_EQ(factory.calls, 3000);
    EXPECT_EQ(cache.size(), 500u);
}

TEST(DynamicOperationCache, ReturnsTheSameInstanceForTheSameKey) {
    DynamicOperationCache cache;
    char a = 0, b = 0;
    Factory factory;
    const auto first = lookup(cache, &a, 5, factory);
    EXPECT_EQ(lookup(cache, &a, 5, factory), first);
    EXPECT_NE(lookup(cache, &a, 6, factory), first);
    EXPECT_NE(lookup(cache, &b, 5, factory), first);
    EXPECT_EQ(factory.calls, 3);
}

TEST(DynamicOperationCache, EvictsTheLeastRecentlyUsedEntry) {
    DynamicOperationCache cache{2};
    char a = 0, b = 0, c = 0;
    Factory factory;
    lookup(cache, &a, 1, factory);
    lookup(cache, &b, 1, factory);
    lookup(cache, &a, 1, factory);  // a becomes more recent than b
    lookup(cache, &c, 1, factory);  // evicts b
    EXPECT_EQ(factory.calls, 3);
    lookup(cache, &a, 1, factory);
    lookup(cache, &c, 1, factory);
    EXPECT_EQ(factory.calls, 3);
    lookup(cache, &b, 1, factory);
    EXPECT_EQ(factory.calls, 4);
}

TEST(DynamicOperationCache, MatchesAReferenceLruOverALongSequence) {
    // Evictions move entries inside the cache; every hit and miss must match a plain LRU list.
    constexpr size_t kCapacity = 37;
    DynamicOperationCache cache{kCapacity};
    std::vector<char> ops(7);
    std::list<std::pair<size_t, size_t>> reference;  // (operation, length), most recent first
    Factory factory;
    for (size_t step = 0; step < 5000; ++step) {
        const size_t op = (step * 5 + step / 11) % ops.size();
        const size_t length = (step * step + 3 * step) % 13;
        const auto key = std::make_pair(op, length);
        const auto it = std::find(reference.begin(), reference.end(), key);
        const bool hit = it != reference.end();
        if (hit) {
            reference.erase(it);
        } else if (reference.size() == kCapacity) {
            reference.pop_back();
        }
        reference.push_front(key);

        const int calls = factory.calls;
        lookup(cache, &ops[op], length, factory);
        ASSERT_EQ(factory.calls == calls, hit) << "step " << step;
    }
    EXPECT_EQ(cache.size(), kCapacity);
}

TEST(DynamicOperationCache, DeviceBudgetEvictsTheLeastRecentlyUsedEntries) {
    DynamicOperationCache cache;
    cache.set_device_budget(250);
    char a = 0, b = 0, c = 0;
    Factory factory{100};
    lookup(cache, &a, 1, factory);
    lookup(cache, &b, 1, factory);
    lookup(cache, &a, 1, factory);
    lookup(cache, &c, 1, factory);  // 300 > 250: evicts b
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_EQ(cache.device_bytes(), 200u);
    lookup(cache, &a, 1, factory);
    EXPECT_EQ(factory.calls, 3);
    lookup(cache, &b, 1, factory);  // re-created
    EXPECT_EQ(factory.calls, 4);
    EXPECT_LE(cache.device_bytes(), 250u);
}

TEST(DynamicOperationCache, KeepsANewInstanceLargerThanTheDeviceBudget) {
    DynamicOperationCache cache;
    cache.set_device_budget(50);
    char a = 0, b = 0;
    Factory factory{100};
    lookup(cache, &a, 1, factory);
    const auto instance = lookup(cache, &b, 1, factory);
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(cache.size(), 1u);  // a is evicted; b stays, it is needed right now
    EXPECT_EQ(lookup(cache, &b, 1, factory), instance);
}

TEST(DynamicOperationCache, RetriesAfterEvictingWhenOutOfDeviceMemory) {
    DynamicOperationCache cache;
    char a = 0, b = 0, c = 0;
    Factory factory;
    lookup(cache, &a, 1, factory);
    lookup(cache, &b, 1, factory);
    int attempts = 0;
    const auto instance = cache.getOrCreate(&c, shapeKey(1), [&attempts, &factory] {
        if (++attempts == 1) {
            allocateTooMuchDeviceMemory();
        }
        return factory();
    });
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(cache.size(), 2u);  // a, the least recently used entry, was evicted to retry
    const int calls = factory.calls;
    lookup(cache, &b, 1, factory);
    EXPECT_EQ(factory.calls, calls);
    lookup(cache, &a, 1, factory);
    EXPECT_EQ(factory.calls, calls + 1);
}

TEST(DynamicOperationCache, OutOfDeviceMemoryWithNothingToEvictIsThrown) {
    DynamicOperationCache cache;
    char a = 0;
    EXPECT_ANY_THROW(cache.getOrCreate(&a, shapeKey(1), [] {
        allocateTooMuchDeviceMemory();
        return std::make_shared<CachedOperation>();
    }));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cache.size(), 0u);
}

TEST(DynamicOperationCache, OtherErrorsAreThrownWithoutEvicting) {
    DynamicOperationCache cache;
    char a = 0, b = 0;
    Factory factory;
    lookup(cache, &a, 1, factory);
    EXPECT_THROW(cache.getOrCreate(&b,
                                   shapeKey(1),
                                   []() -> std::shared_ptr<CachedOperation> {
                                       throw std::runtime_error{"not a CUDA error"};
                                   }),
                 std::runtime_error);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(DynamicOperationCache, ConfigureDerivesTheBoundsFromTheModel) {
    DynamicOperationCache cache;
    cache.configure(10, size_t{2} << 30, 0);
    EXPECT_EQ(cache.capacity(), 10 * DynamicOperationCache::kEntriesPerOperation);
    EXPECT_EQ(cache.device_budget(), (size_t{2} << 30) / DynamicOperationCache::kConstantsPerDeviceBudget);
    cache.configure(10, 1000, 77);  // an explicit capacity; small constants get the minimum budget
    EXPECT_EQ(cache.capacity(), 77u);
    EXPECT_EQ(cache.device_budget(), DynamicOperationCache::kMinDeviceBudget);
}

TEST(DynamicOperationCache, ShrinkingTheCapacityEvictsTheOldestEntries) {
    DynamicOperationCache cache{8};
    std::vector<char> ops(8);
    Factory factory;
    for (const char& op : ops) {
        lookup(cache, &op, 1, factory);
    }
    cache.set_capacity(3);
    EXPECT_EQ(cache.size(), 3u);
    const int calls = factory.calls;
    for (size_t i = 5; i < 8; ++i) {
        lookup(cache, &ops[i], 1, factory);  // the three most recent survive
    }
    EXPECT_EQ(factory.calls, calls);
}
