// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_ipc/host_pool.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace core::camera;

namespace
{

// Not a whole number of pages, so slots are padded apart.
constexpr std::uint64_t SLOT_BYTES = 5000;

// What another process ends up with after the pool's descriptor was sent to it.
UniqueFd duplicate_descriptor(const HostPool& pool)
{
    UniqueFd result(::fcntl(pool.descriptor(), F_DUPFD_CLOEXEC, 0));
    REQUIRE(result.valid());
    return result;
}

} // namespace

TEST_CASE("a view sees what the owner writes", "[unit][camera_ipc][host_pool]")
{
    HostPool pool(3, SLOT_BYTES, "camera_ipc_test");
    REQUIRE(pool.slot_count() == 3);
    REQUIRE(pool.slot_bytes() == SLOT_BYTES);

    const HostPoolView view(duplicate_descriptor(pool), 3, SLOT_BYTES);
    REQUIRE(view.slot_count() == 3);

    // Slots start zeroed and are separate from each other.
    const auto is_zero = [](std::byte value) { return value == std::byte{ 0 }; };
    REQUIRE(std::all_of(view.slot(0).begin(), view.slot(0).end(), is_zero));

    std::fill(pool.slot(1).begin(), pool.slot(1).end(), std::byte{ 0x5a });
    REQUIRE(view.slot(1).size() == SLOT_BYTES);
    REQUIRE(view.slot(1).front() == std::byte{ 0x5a });
    REQUIRE(view.slot(1).back() == std::byte{ 0x5a });
    REQUIRE(std::all_of(view.slot(0).begin(), view.slot(0).end(), is_zero));
    REQUIRE(std::all_of(view.slot(2).begin(), view.slot(2).end(), is_zero));

    // The mapping is live: a later write shows up without mapping again.
    pool.slot(2)[100] = std::byte{ 0x11 };
    REQUIRE(view.slot(2)[100] == std::byte{ 0x11 });
}

TEST_CASE("slots start on page boundaries", "[unit][camera_ipc][host_pool]")
{
    HostPool pool(2, SLOT_BYTES, "camera_ipc_test");
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    REQUIRE(reinterpret_cast<std::uintptr_t>(pool.slot(0).data()) % page == 0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(pool.slot(1).data()) % page == 0);
    REQUIRE(pool.slot(1).data() >= pool.slot(0).data() + SLOT_BYTES);
}

TEST_CASE("writing and reading a pool for the first time causes no page faults", "[unit][camera_ipc][host_pool]")
{
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    SKIP("a sanitizer's shadow memory faults on first touch whatever the pool does");
#else
    constexpr std::uint32_t SLOTS = 8;
    constexpr std::uint64_t BYTES = 2 * 1024 * 1024;
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const auto faults = []
    {
        rusage usage = {};
        REQUIRE(::getrusage(RUSAGE_SELF, &usage) == 0);
        return usage.ru_minflt + usage.ru_majflt;
    };

    HostPool pool(SLOTS, BYTES, "camera_ipc_test");
    HostPoolView view(duplicate_descriptor(pool), SLOTS, BYTES);

    // Touch every page of every slot: write as the owner, then read as a reader would.
    const long before = faults();
    unsigned sum = 0;
    for (std::uint32_t slot = 0; slot < SLOTS; ++slot)
    {
        for (std::size_t offset = 0; offset < BYTES; offset += page)
        {
            pool.slot(slot)[offset] = std::byte{ 1 };
            sum += std::to_integer<unsigned>(view.slot(slot)[offset]);
        }
    }
    const long caused = faults() - before;
    REQUIRE(sum == SLOTS * (BYTES / page));

    // The pool has thousands of pages; without them mapped up front, each would fault twice.
    REQUIRE(caused < 32);
#endif
}

TEST_CASE("a process that is sent the pool cannot write to it or resize it", "[unit][camera_ipc][host_pool]")
{
    HostPool pool(1, SLOT_BYTES, "camera_ipc_test");
    const UniqueFd received = duplicate_descriptor(pool);

    void* writable = ::mmap(nullptr, SLOT_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, received.get(), 0);
    REQUIRE(writable == MAP_FAILED);
    REQUIRE(errno == EPERM);

    const char byte = 1;
    REQUIRE(::write(received.get(), &byte, 1) == -1);
    REQUIRE(::ftruncate(received.get(), 16) == -1);
    REQUIRE(::ftruncate(received.get(), static_cast<off_t>(16 * SLOT_BYTES)) == -1);

    // The owner still can.
    pool.slot(0)[0] = std::byte{ 0x7f };
    const HostPoolView view(duplicate_descriptor(pool), 1, SLOT_BYTES);
    REQUIRE(view.slot(0)[0] == std::byte{ 0x7f });
}

TEST_CASE("a view outlives the pool and can be moved", "[unit][camera_ipc][host_pool]")
{
    std::vector<HostPoolView> views;
    {
        HostPool pool(2, SLOT_BYTES, "camera_ipc_test");
        pool.slot(1)[7] = std::byte{ 0x42 };
        views.emplace_back(duplicate_descriptor(pool), 2, SLOT_BYTES);
    }
    // The memory lives as long as anything maps it.
    HostPoolView moved = std::move(views.front());
    REQUIRE(views.front().slot_count() == 0);
    REQUIRE(moved.slot_count() == 2);
    REQUIRE(moved.slot(1)[7] == std::byte{ 0x42 });
}

TEST_CASE("misuse of a host pool is rejected", "[unit][camera_ipc][host_pool]")
{
    REQUIRE_THROWS_AS(HostPool(0, SLOT_BYTES, "camera_ipc_test"), std::invalid_argument);
    REQUIRE_THROWS_AS(HostPool(1, 0, "camera_ipc_test"), std::invalid_argument);

    HostPool pool(2, SLOT_BYTES, "camera_ipc_test");
    REQUIRE_THROWS_AS(pool.slot(2), std::out_of_range);

    const HostPoolView view(duplicate_descriptor(pool), 2, SLOT_BYTES);
    REQUIRE_THROWS_AS(view.slot(2), std::out_of_range);

    // A descriptor that holds less than the pool claims would fault on access, so it is refused.
    REQUIRE_THROWS_AS(HostPoolView(duplicate_descriptor(pool), 3, SLOT_BYTES), std::invalid_argument);
    REQUIRE_THROWS_AS(HostPoolView(duplicate_descriptor(pool), 2, 4 * SLOT_BYTES), std::invalid_argument);
    REQUIRE_THROWS_AS(HostPoolView(duplicate_descriptor(pool), 0, SLOT_BYTES), std::invalid_argument);

    REQUIRE_THROWS_AS(HostPoolView(UniqueFd(-1), 2, SLOT_BYTES), std::system_error);
}
