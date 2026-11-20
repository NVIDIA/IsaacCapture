// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/cuda_pool.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace core::camera;
using Catch::Matchers::ContainsSubstring;

namespace
{

// Not a multiple of the slot alignment, so slots are padded apart.
constexpr std::uint64_t SLOT_BYTES = 5000;

// What another process ends up with after the pool's descriptor was sent to it.
UniqueFd duplicate_descriptor(const CudaPool& pool)
{
    UniqueFd result(::fcntl(pool.descriptor(), F_DUPFD_CLOEXEC, 0));
    REQUIRE(result.valid());
    return result;
}

std::vector<std::byte> bytes_of(std::uint64_t count, std::uint8_t first)
{
    std::vector<std::byte> bytes(count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        bytes[i] = static_cast<std::byte>(static_cast<std::uint8_t>(first + i));
    }
    return bytes;
}

std::vector<std::byte> read_from(const CudaDeviceUuid& device, CudaAddress address, std::uint64_t count)
{
    std::vector<std::byte> bytes(count);
    copy_from_cuda(device, bytes, address);
    return bytes;
}

} // namespace

TEST_CASE("a device's name is its UUID as nvidia-smi prints it", "[unit][camera][cuda_pool]")
{
    const CudaDeviceUuid device = { 0xb4, 0x37, 0x72, 0x62, 0xe0, 0x0d, 0x16, 0x89,
                                    0xcd, 0xf9, 0xd9, 0xc0, 0x2b, 0x8a, 0xee, 0x17 };
    REQUIRE(cuda_device_name(device) == "GPU-b4377262-e00d-1689-cdf9-d9c02b8aee17");
}

TEST_CASE("a CUDA pool without slots or without bytes is refused", "[unit][camera][cuda_pool]")
{
    // Before the driver is asked for anything, so also where there is none.
    const CudaDeviceUuid any = { 1 };
    REQUIRE_THROWS_AS(CudaPool(any, 0, SLOT_BYTES), std::invalid_argument);
    REQUIRE_THROWS_AS(CudaPool(any, 2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(CudaPoolView(UniqueFd(), any, 0, SLOT_BYTES), std::invalid_argument);
}

TEST_CASE("a view of a CUDA pool sees what the owner writes", "[gpu][camera][cuda_pool]")
{
    if (!cuda_unavailable().empty())
    {
        SKIP(cuda_unavailable());
    }
    const CudaDeviceUuid device = cuda_device_uuid(0);
    CudaPool pool(device, 3, SLOT_BYTES);
    REQUIRE(pool.slot_count() == 3);
    REQUIRE(pool.slot_bytes() == SLOT_BYTES);

    // Slots are separate from each other and start on the alignment both sides use.
    REQUIRE(pool.slot(1) >= pool.slot(0) + SLOT_BYTES);
    REQUIRE(pool.slot(2) - pool.slot(1) == pool.slot(1) - pool.slot(0));
    REQUIRE((pool.slot(1) - pool.slot(0)) % 4096 == 0);
    REQUIRE_THROWS_AS(pool.slot(3), std::out_of_range);

    const std::vector<std::byte> first = bytes_of(SLOT_BYTES, 3);
    const std::vector<std::byte> second = bytes_of(SLOT_BYTES, 101);
    copy_to_cuda(device, pool.slot(0), first);
    copy_to_cuda(device, pool.slot(1), second);

    const CudaPoolView view(duplicate_descriptor(pool), device, 3, SLOT_BYTES);
    REQUIRE(view.slot_count() == 3);
    REQUIRE(view.device_index() == 0);
    REQUIRE(read_from(device, view.slot(0), SLOT_BYTES) == first);
    REQUIRE(read_from(device, view.slot(1), SLOT_BYTES) == second);
    REQUIRE_THROWS_AS(view.slot(3), std::out_of_range);

    // The mapping is live: a later write shows up without mapping again.
    copy_to_cuda(device, pool.slot(0), second);
    REQUIRE(read_from(device, view.slot(0), SLOT_BYTES) == second);
}

TEST_CASE("a view of a CUDA pool cannot be written through", "[gpu][camera][cuda_pool]")
{
    if (!cuda_unavailable().empty())
    {
        SKIP(cuda_unavailable());
    }
    const CudaDeviceUuid device = cuda_device_uuid(0);
    CudaPool pool(device, 2, SLOT_BYTES);
    const std::vector<std::byte> content = bytes_of(SLOT_BYTES, 7);
    copy_to_cuda(device, pool.slot(0), content);
    const CudaPoolView view(duplicate_descriptor(pool), device, 2, SLOT_BYTES);

    // The driver refuses the write, the owner's content stays, and the device still works.
    REQUIRE_THROWS_AS(copy_to_cuda(device, view.slot(0), bytes_of(SLOT_BYTES, 99)), std::runtime_error);
    REQUIRE(read_from(device, pool.slot(0), SLOT_BYTES) == content);
    REQUIRE(read_from(device, view.slot(0), SLOT_BYTES) == content);
}

TEST_CASE("a view keeps a CUDA pool's memory when the pool is gone", "[gpu][camera][cuda_pool]")
{
    if (!cuda_unavailable().empty())
    {
        SKIP(cuda_unavailable());
    }
    const CudaDeviceUuid device = cuda_device_uuid(0);
    const std::vector<std::byte> content = bytes_of(SLOT_BYTES, 42);
    auto pool = std::make_unique<CudaPool>(device, 2, SLOT_BYTES);
    copy_to_cuda(device, pool->slot(1), content);
    const CudaPoolView view(duplicate_descriptor(*pool), device, 2, SLOT_BYTES);

    pool.reset();
    REQUIRE(read_from(device, view.slot(1), SLOT_BYTES) == content);
}

TEST_CASE("a CUDA pool that cannot be made or mapped says why", "[gpu][camera][cuda_pool]")
{
    if (!cuda_unavailable().empty())
    {
        SKIP(cuda_unavailable());
    }
    const CudaDeviceUuid device = cuda_device_uuid(0);
    const CudaDeviceUuid elsewhere = { 0xde, 0xad, 0xbe, 0xef };
    REQUIRE_THROWS_WITH(CudaPool(elsewhere, 2, SLOT_BYTES), ContainsSubstring("GPU-deadbeef-0000-0000-0000-000000000000") &&
                                                                ContainsSubstring("not one of this process's devices"));
    REQUIRE_THROWS_WITH(cuda_device_uuid(cuda_device_count()), ContainsSubstring("there is no CUDA device"));

    CudaPool pool(device, 4, SLOT_BYTES);
    // A reader that is told of more memory than the pool has cannot map it: a view never
    // reaches past the pool.
    REQUIRE_THROWS_WITH(
        CudaPoolView(duplicate_descriptor(pool), device, 4, 64 << 20), ContainsSubstring("mapping a pool"));
    // A descriptor that is not a pool's.
    UniqueFd not_a_pool(::open("/dev/null", O_RDONLY | O_CLOEXEC));
    REQUIRE(not_a_pool.valid());
    REQUIRE_THROWS_WITH(
        CudaPoolView(std::move(not_a_pool), device, 4, SLOT_BYTES), ContainsSubstring("importing a pool"));
}
