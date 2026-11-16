// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "utils/lru_cache.hpp"

using ov::nvidia_gpu::LruCache;

TEST(LruCache, EvictsLeastRecentlyUsed) {
    LruCache<int, std::string> cache{2};
    cache.insert(1, "one");
    cache.insert(2, "two");
    ASSERT_NE(cache.find(1), nullptr);  // 1 becomes the most recently used
    cache.insert(3, "three");           // evicts 2
    EXPECT_EQ(cache.find(2), nullptr);
    ASSERT_NE(cache.find(1), nullptr);
    EXPECT_EQ(*cache.find(1), "one");
    EXPECT_EQ(*cache.find(3), "three");
    EXPECT_EQ(cache.size(), 2u);
}

TEST(LruCache, InsertExistingKeyUpdatesValue) {
    LruCache<int, std::string> cache{2};
    cache.insert(1, "one");
    cache.insert(2, "two");
    cache.insert(1, "uno");  // update + promote; nothing evicted
    EXPECT_EQ(cache.size(), 2u);
    cache.insert(3, "three");  // evicts 2, the least recently used
    EXPECT_EQ(cache.find(2), nullptr);
    EXPECT_EQ(*cache.find(1), "uno");
}

TEST(LruCache, CyclicAccessHitsWhenCapacityCoversAllKeys) {
    // A model's dynamic operations are looked up in the same order on every inference.
    LruCache<int, int> cache{4};
    for (int round = 0; round < 3; ++round) {
        for (int k = 0; k < 4; ++k) {
            if (!cache.find(k)) {
                ASSERT_EQ(round, 0) << "miss for key " << k << " in round " << round;
                cache.insert(k, k);
            }
        }
    }
}

TEST(LruCache, SetCapacityGrowsAndShrinks) {
    LruCache<int, int> cache{2};
    cache.set_capacity(4);
    for (int k = 0; k < 4; ++k) cache.insert(k, k);
    EXPECT_EQ(cache.size(), 4u);
    cache.find(0);
    cache.set_capacity(2);  // keeps the two most recently used: 0 and 3
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_NE(cache.find(0), nullptr);
    EXPECT_NE(cache.find(3), nullptr);
    EXPECT_EQ(cache.find(1), nullptr);
}
