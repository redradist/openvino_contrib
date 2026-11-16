// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>

#include "cuda_paged_attention_context.hpp"

using ov::nvidia_gpu::PagedAttentionContext;

namespace {

// Cache layout [num_blocks, heads=2, block_size=2, head_size=3] (f32).
constexpr size_t kHeads = 2;
constexpr size_t kBlockSize = 2;
constexpr size_t kHeadSize = 3;
constexpr size_t kBlockElems = kHeads * kBlockSize * kHeadSize;

ov::Tensor make_cache(size_t num_blocks, float first_value) {
    ov::Tensor t(ov::element::f32, ov::Shape{num_blocks, kHeads, kBlockSize, kHeadSize});
    std::iota(t.data<float>(), t.data<float>() + t.get_size(), first_value);
    return t;
}

std::vector<float> device_contents(const void* device, size_t elems, const CUDA::Stream& stream) {
    std::vector<float> v(elems);
    stream.download(v.data(), CUDA::DevicePointer<const void*>{device}, elems * sizeof(float));
    stream.synchronize();
    return v;
}

std::vector<float> host_contents(const ov::Tensor& t) {
    return {t.data<const float>(), t.data<const float>() + t.get_size()};
}

std::vector<float> blocks_of(const std::vector<float>& v, size_t first_block, size_t num_blocks) {
    return {v.begin() + first_block * kBlockElems, v.begin() + (first_block + num_blocks) * kBlockElems};
}

float* block(ov::Tensor& t, size_t b) { return t.data<float>() + b * kBlockElems; }

/// Element (block, head, slot, d) of a cache laid out as [blocks, kHeads, kBlockSize, kHeadSize].
size_t cache_index(size_t b, size_t h, size_t slot, size_t d) {
    return ((b * kHeads + h) * kBlockSize + slot) * kHeadSize + d;
}

/// The new K/V rows of a step: [tokens, kHeads, kHeadSize], on the device.
struct DeviceRows {
    std::vector<float> host;
    CUDA::Allocation device;
};
DeviceRows upload_rows(const std::vector<float>& rows, const CUDA::Stream& stream) {
    auto alloc = stream.malloc(rows.size() * sizeof(float));
    stream.upload(CUDA::DevicePointer<void*>{alloc.get()}, rows.data(), rows.size() * sizeof(float));
    return {rows, std::move(alloc)};
}

/// What reshape_and_cache does on the device: writes row t to slot slots[t] (-1 = skipped).
void write_rows_on_device(void* cache, const std::vector<float>& rows, const std::vector<int32_t>& slots,
                          const CUDA::Stream& stream) {
    for (size_t t = 0; t < slots.size(); ++t) {
        if (slots[t] < 0) continue;
        const size_t b = slots[t] / kBlockSize, slot = slots[t] % kBlockSize;
        for (size_t h = 0; h < kHeads; ++h) {
            stream.upload(CUDA::DevicePointer<void*>{static_cast<float*>(cache) + cache_index(b, h, slot, 0)},
                          rows.data() + (t * kHeads + h) * kHeadSize,
                          kHeadSize * sizeof(float));
        }
    }
}

class PagedAttentionContextTest : public testing::Test {
protected:
    void SetUp() override { ctx.begin_infer(); }

    /// Ends the current infer the way CudaInferRequest::wait_pipeline does, and starts the next.
    void next_infer() {
        stream.synchronize();
        ctx.finish_infer();
        ctx.begin_infer();
    }

    CUDA::Stream stream;
    PagedAttentionContext ctx;
    static constexpr ov::nvidia_gpu::BufferID kId = 42;
};

}  // namespace

TEST_F(PagedAttentionContextTest, FirstBindSeedsDeviceFromHost) {
    auto host = make_cache(3, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, WrittenRowsReachTheHostWhenTheInferFinishes) {
    auto host = make_cache(3, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    // Two new tokens: slot 1 (block 0, offset 1) and slot 4 (block 2, offset 0); one padding row.
    const std::vector<int32_t> slots{1, -1, 4};
    std::vector<float> rows(slots.size() * kHeads * kHeadSize);
    std::iota(rows.begin(), rows.end(), -100.0f);
    const auto device_rows = upload_rows(rows, stream);
    write_rows_on_device(dev, rows, slots, stream);

    const auto before = host_contents(host);
    ctx.write_back_rows(kId, device_rows.device.get(), slots, stream);
    EXPECT_EQ(host_contents(host), before);  // deferred until the infer finishes
    next_infer();

    auto expected = before;
    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t d = 0; d < kHeadSize; ++d) {
            expected[cache_index(0, h, 1, d)] = rows[(0 * kHeads + h) * kHeadSize + d];
            expected[cache_index(2, h, 0, d)] = rows[(2 * kHeads + h) * kHeadSize + d];
        }
    }
    EXPECT_EQ(host_contents(host), expected);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), expected);  // host and device agree
}

TEST_F(PagedAttentionContextTest, GrowthKeepsExistingDeviceContents) {
    auto host = make_cache(2, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    // A step writes slot 3 (block 1, offset 1) on the device and writes it back.
    const std::vector<float> rows(kHeads * kHeadSize, 7.0f);
    const auto device_rows = upload_rows(rows, stream);
    write_rows_on_device(dev, rows, {3}, stream);
    ctx.write_back_rows(kId, device_rows.device.get(), {3}, stream);
    next_infer();

    // GenAI grows the cache: a new, bigger host tensor holding the old contents in front.
    auto grown = make_cache(5, 100.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    const void* grown_dev = ctx.bind(kId, grown, stream);
    EXPECT_EQ(blocks_of(device_contents(grown_dev, grown.get_size(), stream), 0, 2),
              blocks_of(host_contents(grown), 0, 2));
}

TEST_F(PagedAttentionContextTest, GrowthWithinCapacityAndShrinkKeepTheBuffer) {
    auto host = make_cache(4, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    auto grown = make_cache(5, 1.0f);  // 4 -> 5 blocks: reallocates with headroom (6 blocks)
    const void* dev5 = ctx.bind(kId, grown, stream);
    EXPECT_NE(dev5, dev);
    auto grown6 = make_cache(6, 1.0f);  // fits into the headroom: no reallocation
    EXPECT_EQ(ctx.bind(kId, grown6, stream), dev5);
    auto shrunk = make_cache(2, 1.0f);  // GenAI's clear() + smaller cache: buffer is kept
    EXPECT_EQ(ctx.bind(kId, shrunk, stream), dev5);
    EXPECT_EQ(ctx.bind(kId, grown6, stream), dev5);
}

TEST_F(PagedAttentionContextTest, RefreshAndWriteBackAfterGrowthUseTheNewTensors) {
    auto host = make_cache(2, 1.0f);
    ctx.bind(kId, host, stream);
    auto grown = make_cache(8, 1.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    void* dev = ctx.bind(kId, grown, stream);

    // copy-on-write into a block that only exists in the grown cache
    std::memcpy(block(grown, 6), block(grown, 1), kBlockElems * sizeof(float));
    ctx.refresh_prefixes(kId, {6}, {static_cast<int32_t>(kBlockSize)}, stream);
    const auto dev_after = device_contents(dev, grown.get_size(), stream);
    const auto host_after = host_contents(grown);
    EXPECT_EQ(blocks_of(dev_after, 0, 2), blocks_of(host_after, 0, 2));
    EXPECT_EQ(blocks_of(dev_after, 6, 1), blocks_of(host_after, 6, 1));

    // a row written to slot 14 (block 7, offset 0) lands in the grown host tensor only
    const std::vector<float> rows(kHeads * kHeadSize, 9.0f);
    const auto device_rows = upload_rows(rows, stream);
    ctx.write_back_rows(kId, device_rows.device.get(), {14}, stream);
    next_infer();
    for (size_t h = 0; h < kHeads; ++h) {
        EXPECT_EQ(grown.data<float>()[cache_index(7, h, 0, 0)], 9.0f);
    }
    EXPECT_EQ(host_contents(host), host_contents(make_cache(2, 1.0f)));
}

TEST_F(PagedAttentionContextTest, GrowthWhileDesyncedReseedsFromHost) {
    auto host = make_cache(2, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    const std::vector<float> garbage(host.get_size(), -1.0f);  // device diverged from the host
    stream.upload(CUDA::DevicePointer<void*>{dev}, garbage.data(), garbage.size() * sizeof(float));
    ctx.abort_infer(stream);
    ctx.begin_infer();
    auto grown = make_cache(4, 50.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    dev = ctx.bind(kId, grown, stream);
    EXPECT_EQ(device_contents(dev, grown.get_size(), stream), host_contents(grown));
}

TEST_F(PagedAttentionContextTest, RefreshUploadsOnlyTheFilledPrefixOfEveryHead) {
    auto host = make_cache(4, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    // GenAI's CacheManager::copy_blocks: block 0 -> block 3 on the host only.
    std::memcpy(block(host, 3), block(host, 0), kBlockElems * sizeof(float));
    const auto stale = device_contents(dev, host.get_size(), stream);
    EXPECT_NE(stale, host_contents(host));

    ctx.refresh_prefixes(kId, {3}, {1}, stream);  // one slot of block 3 is filled
    auto expected = stale;
    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t d = 0; d < kHeadSize; ++d) {
            expected[cache_index(3, h, 0, d)] = host.data<float>()[cache_index(3, h, 0, d)];
        }
    }
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), expected);

    ctx.refresh_prefixes(kId, {3}, {static_cast<int32_t>(kBlockSize)}, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, AbortedInferDropsTheWriteBackAndReseedsOnNextBind) {
    auto host = make_cache(3, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    const std::vector<float> rows(kHeads * kHeadSize, -5.0f);
    const auto device_rows = upload_rows(rows, stream);
    write_rows_on_device(dev, rows, {2}, stream);
    ctx.write_back_rows(kId, device_rows.device.get(), {2}, stream);
    // The infer fails before it finishes: the copies in flight complete, the pending write-back
    // is dropped...
    ctx.abort_infer(stream);
    ctx.begin_infer();
    EXPECT_EQ(host_contents(host), host_contents(make_cache(3, 1.0f)));
    // ...and the mirror is re-seeded from the host on the next bind.
    std::fill_n(host.data<float>(), host.get_size(), 3.0f);
    dev = ctx.bind(kId, host, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, UnboundCacheThrows) {
    EXPECT_ANY_THROW(ctx.refresh_prefixes(kId, {0}, {1}, stream));
    EXPECT_ANY_THROW(ctx.write_back_rows(kId, nullptr, {0}, stream));
}

TEST(PinnedArena, HandsOutDistinctMemoryAndReusesItAfterReset) {
    ov::nvidia_gpu::PinnedArena arena;
    auto* a = static_cast<uint8_t*>(arena.allocate(100));
    auto* b = static_cast<uint8_t*>(arena.allocate(3 << 20));  // does not fit the first chunk
    EXPECT_TRUE(b + (3 << 20) <= a || a + 100 <= b);
    std::memset(a, 1, 100);
    std::memset(b, 2, 3 << 20);
    arena.reset();  // merges the chunks into one
    auto* c = static_cast<uint8_t*>(arena.allocate((3 << 20) + 100));
    auto* d = static_cast<uint8_t*>(arena.allocate(100));
    EXPECT_TRUE(d >= c + (3 << 20) + 100);  // both fit the merged chunk, one after the other
    std::memset(c, 3, (3 << 20) + 100);
    arena.reset();
    EXPECT_EQ(arena.allocate(16), c);  // memory is reused from the start
}
