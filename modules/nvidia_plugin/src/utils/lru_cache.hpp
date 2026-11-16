// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Generic LRU (Least Recently Used) cache with O(1) lookup, insertion and eviction.
 *
 * Entries live in a list ordered from most to least recently used; a hash map indexes them.
 * When the capacity is exceeded, the least recently used entry is evicted.
 *
 * @tparam Key   Cache key type (must support operator==)
 * @tparam Value Cached value type
 * @tparam Hash  Hash functor for Key
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache {
public:
    explicit LruCache(size_t capacity) : capacity_{std::max<size_t>(capacity, 1)} {}

    /**
     * Find a cached value by key.
     * If found, promotes the entry to most-recently-used.
     * @return Pointer to cached value, or nullptr on miss.
     */
    Value* find(const Key& key) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return nullptr;
        }
        entries_.splice(entries_.begin(), entries_, it->second);
        return &entries_.front().second;
    }

    /**
     * Insert or update a cache entry.
     * If at capacity, evicts the least recently used entry.
     * @return Reference to the stored value.
     */
    Value& insert(const Key& key, Value value) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->second = std::move(value);
            entries_.splice(entries_.begin(), entries_, it->second);
            return entries_.front().second;
        }
        while (entries_.size() >= capacity_) {
            evict_lru();
        }
        entries_.emplace_front(key, std::move(value));
        index_.emplace(key, entries_.begin());
        return entries_.front().second;
    }

    /// Changes the capacity; shrinking evicts the least recently used entries.
    void set_capacity(size_t capacity) {
        capacity_ = std::max<size_t>(capacity, 1);
        while (entries_.size() > capacity_) {
            evict_lru();
        }
    }

    size_t capacity() const { return capacity_; }

    size_t size() const { return entries_.size(); }

    void clear() {
        index_.clear();
        entries_.clear();
    }

private:
    void evict_lru() {
        index_.erase(entries_.back().first);
        entries_.pop_back();
    }

    size_t capacity_;
    std::list<std::pair<Key, Value>> entries_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator, Hash> index_;
};

}  // namespace nvidia_gpu
}  // namespace ov
