// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "core/buffer_owner.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using plugins::sensing::BufferOwner;
using plugins::sensing::BufferPort;
using plugins::sensing::BufferReturnMailbox;
using plugins::sensing::CaptureFrame;
using plugins::sensing::DmaBufNv12Pixels;

namespace
{

using namespace std::chrono_literals;

constexpr auto kWait = 5ms;
constexpr auto kDeadline = 5s;
constexpr size_t kMaxSlots = 16;

template <typename Pred>
bool wait_until(Pred pred,
                std::chrono::milliseconds limit = std::chrono::duration_cast<std::chrono::milliseconds>(kDeadline))
{
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!pred())
    {
        if (std::chrono::steady_clock::now() >= end)
            return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

/// A bounded pool of slots standing in for the stream. Frames are produced while `budget` is positive and a slot
/// is free; otherwise acquire sleeps the requested timeout and reports Timeout.
class FakePort : public BufferPort
{
public:
    explicit FakePort(size_t pool_size) : m_pool_size(pool_size)
    {
    }

    Acquired acquire(uint64_t timeout_ns) override
    {
        Guard guard(*this);
        Acquired out;
        int slot = -1;
        bool produce = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_acquire_calls;
            if (m_throw_on_acquire)
                throw std::runtime_error("acquire threw");
            for (size_t i = 0; i < m_pool_size; ++i)
            {
                if (!m_busy[i])
                {
                    slot = static_cast<int>(i);
                    break;
                }
            }
            if (slot >= 0 && m_budget > 0)
            {
                --m_budget;
                m_busy[slot] = true;
                ++m_acquired[slot];
                ++m_acquired_total;
                m_events.push_back({ Event::Kind::Acquire, static_cast<uint32_t>(slot) });
                produce = true;
                if (m_fail_next)
                {
                    m_fail_next = false;
                    out.result = Result::Failed;
                    out.holds_buffer = true;
                    out.slot = static_cast<uint32_t>(slot);
                    out.error = "stream exploded";
                    return out;
                }
            }
        }
        if (!produce)
        {
            std::this_thread::sleep_for(std::chrono::nanoseconds(timeout_ns));
            out.result = Result::Timeout;
            return out;
        }
        std::this_thread::sleep_for(200us);
        out.result = Result::Frame;
        out.holds_buffer = true;
        out.slot = static_cast<uint32_t>(slot);
        out.frame.sequence = m_sequence.fetch_add(1) + 1;
        out.frame.pixels = DmaBufNv12Pixels{ slot };
        return out;
    }

    bool release(uint32_t slot) override
    {
        Guard guard(*this);
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_release_calls;
        m_events.push_back({ Event::Kind::Release, slot });
        if (m_throw_on_release)
            throw std::runtime_error("release threw");
        if (slot >= m_pool_size || !m_busy[slot])
        {
            ++m_bad_releases;
            return m_release_ok;
        }
        m_busy[slot] = false;
        ++m_released[slot];
        ++m_released_total;
        return m_release_ok;
    }

    bool stop_capture() override
    {
        Guard guard(*this);
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_stop_calls;
        m_events.push_back({ Event::Kind::StopBegin, 0 });
        return m_stop_ok;
    }

    void set_budget(int64_t n)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_budget = n;
    }
    void set_throw_on_acquire(bool v)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_throw_on_acquire = v;
    }
    void set_fail_next(bool v)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_fail_next = v;
    }
    void set_release_ok(bool v)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_release_ok = v;
    }
    void set_throw_on_release(bool v)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_throw_on_release = v;
    }
    void set_stop_ok(bool v)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop_ok = v;
    }

    uint64_t acquire_calls() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_acquire_calls;
    }
    uint64_t acquired_total() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_acquired_total;
    }
    uint64_t released_total() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_released_total;
    }
    uint64_t release_calls() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_release_calls;
    }
    uint64_t bad_releases() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_bad_releases;
    }
    /// Port calls in the order they happened, across all slots.
    struct Event
    {
        enum class Kind
        {
            Acquire,
            Release,
            StopBegin,
        };
        Kind kind;
        uint32_t slot;
    };
    std::vector<Event> events() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_events;
    }
    int stop_calls() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_stop_calls;
    }
    uint64_t acquired(size_t slot) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_acquired[slot];
    }
    uint64_t released(size_t slot) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_released[slot];
    }
    int max_in_flight() const
    {
        return m_max_in_flight.load();
    }
    std::set<std::thread::id> threads() const
    {
        std::lock_guard<std::mutex> lock(m_threads_mutex);
        return m_threads;
    }

    /// Every acquired buffer was released exactly once, slot by slot.
    bool balanced() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (size_t i = 0; i < m_pool_size; ++i)
        {
            if (m_acquired[i] != m_released[i] || m_busy[i])
                return false;
        }
        return m_bad_releases == 0;
    }

private:
    /// Records the calling thread and how many calls are in flight at once.
    struct Guard
    {
        explicit Guard(FakePort& port) : p(port)
        {
            const int now = ++p.m_in_flight;
            int seen = p.m_max_in_flight.load();
            while (now > seen && !p.m_max_in_flight.compare_exchange_weak(seen, now))
            {
            }
            std::lock_guard<std::mutex> lock(p.m_threads_mutex);
            p.m_threads.insert(std::this_thread::get_id());
        }
        ~Guard()
        {
            --p.m_in_flight;
        }
        FakePort& p;
    };

    const size_t m_pool_size;
    mutable std::mutex m_mutex;
    std::array<bool, kMaxSlots> m_busy{};
    std::array<uint64_t, kMaxSlots> m_acquired{};
    std::array<uint64_t, kMaxSlots> m_released{};
    int64_t m_budget = 0;
    bool m_throw_on_acquire = false;
    bool m_fail_next = false;
    bool m_release_ok = true;
    bool m_stop_ok = true;
    bool m_throw_on_release = false;
    std::vector<Event> m_events;
    uint64_t m_acquire_calls = 0;
    uint64_t m_acquired_total = 0;
    uint64_t m_released_total = 0;
    uint64_t m_release_calls = 0;
    uint64_t m_bad_releases = 0;
    int m_stop_calls = 0;
    std::atomic<uint64_t> m_sequence{ 0 };

    std::atomic<int> m_in_flight{ 0 };
    std::atomic<int> m_max_in_flight{ 0 };
    mutable std::mutex m_threads_mutex;
    std::set<std::thread::id> m_threads;
};

constexpr int64_t kUnlimited = INT64_MAX / 2;

/// Polls the owner until `count` frames have been taken, holding them (and their leases).
std::vector<CaptureFrame> take_frames(BufferOwner& owner, size_t count)
{
    std::vector<CaptureFrame> held;
    wait_until(
        [&]
        {
            if (auto frame = owner.take_latest())
                held.push_back(std::move(*frame));
            return held.size() >= count;
        });
    return held;
}

bool take_throws(BufferOwner& owner, std::string* what = nullptr)
{
    return wait_until(
        [&]
        {
            try
            {
                owner.take_latest();
            }
            catch (const std::runtime_error& e)
            {
                if (what)
                    *what = e.what();
                return true;
            }
            return false;
        });
}

} // namespace

TEST_CASE("every port call runs on one owner thread, never two at once", "[buffer_owner]")
{
    FakePort port(8);
    port.set_budget(kUnlimited);
    BufferOwner owner(port, 8, kWait);
    owner.start();

    std::atomic<bool> stop{ false };
    std::vector<std::thread> droppers;
    for (int t = 0; t < 4; ++t)
    {
        droppers.emplace_back(
            [&]
            {
                while (!stop)
                {
                    // The temporary's lease goes back on this thread, not the owner's.
                    owner.take_latest();
                    std::this_thread::sleep_for(100us);
                }
            });
    }
    REQUIRE(wait_until([&] { return port.released_total() > 100; }));
    owner.request_stop();
    CHECK(owner.wait_stopped(kDeadline));
    stop = true;
    for (auto& t : droppers)
        t.join();
    owner.close();

    const auto threads = port.threads();
    REQUIRE(threads.size() == 1);
    CHECK(*threads.begin() != std::this_thread::get_id());
    CHECK(port.max_in_flight() == 1);
}

TEST_CASE("concurrent lease returns release every slot exactly once", "[buffer_owner]")
{
    FakePort port(8);
    port.set_budget(kUnlimited);
    BufferOwner owner(port, 8, kWait);
    owner.start();

    std::atomic<bool> stop{ false };
    std::vector<std::thread> consumers;
    for (int t = 0; t < 6; ++t)
    {
        consumers.emplace_back(
            [&]
            {
                std::vector<CaptureFrame> held;
                while (!stop)
                {
                    if (auto frame = owner.take_latest())
                    {
                        held.push_back(std::move(*frame));
                        if (held.size() >= 2)
                            held.clear(); // drops leases in bulk
                    }
                    std::this_thread::sleep_for(50us);
                }
            });
    }
    std::this_thread::sleep_for(300ms);
    port.set_budget(0);
    stop = true;
    for (auto& t : consumers)
        t.join();
    owner.close();

    CHECK(port.acquired_total() > 50);
    CHECK(port.acquired_total() == port.released_total());
    CHECK(port.balanced());
    CHECK(owner.rejected_returns() == 0);
    CHECK(port.max_in_flight() == 1);
}

TEST_CASE("a newer frame replaces an older one and the replaced buffer is released once", "[buffer_owner]")
{
    FakePort port(8);
    port.set_budget(5);
    BufferOwner owner(port, 8, kWait);
    owner.start();

    REQUIRE(wait_until([&] { return port.acquired_total() == 5; }));
    REQUIRE(wait_until([&] { return port.released_total() == 4; }));

    auto frame = owner.take_latest();
    REQUIRE(frame.has_value());
    CHECK(frame->sequence == 5);
    CHECK(port.released_total() == 4);
    CHECK_FALSE(owner.take_latest().has_value());

    frame.reset();
    REQUIRE(wait_until([&] { return port.released_total() == 5; }));
    CHECK(port.balanced());
    CHECK(owner.rejected_returns() == 0);
    owner.close();
}

TEST_CASE("an exhausted pool stops acquiring and resumes when a lease returns", "[buffer_owner]")
{
    constexpr size_t kPool = 3;
    FakePort port(kPool);
    port.set_budget(kUnlimited);
    BufferOwner owner(port, kPool, kWait);
    owner.start();

    std::vector<CaptureFrame> held = take_frames(owner, kPool);
    REQUIRE(held.size() == kPool);

    // Let any in-flight acquire finish, then the count must stay put.
    std::this_thread::sleep_for(50ms);
    const uint64_t settled = port.acquire_calls();
    std::this_thread::sleep_for(150ms);
    CHECK(port.acquire_calls() == settled);
    CHECK_FALSE(owner.take_latest().has_value());

    held.pop_back();
    REQUIRE(wait_until([&] { return port.acquire_calls() > settled; }));
    auto next = take_frames(owner, 1);
    CHECK(next.size() == 1);

    held.clear();
    next.clear();
    owner.close();
    CHECK(port.balanced());
    CHECK(owner.rejected_returns() == 0);
}

TEST_CASE("stop clears the latest frame; held leases still return afterwards", "[buffer_owner]")
{
    FakePort port(8);
    port.set_budget(kUnlimited);
    BufferOwner owner(port, 8, kWait);
    owner.start();

    std::vector<CaptureFrame> held = take_frames(owner, 2);
    REQUIRE(held.size() == 2);
    const uint64_t acquired_before_stop = port.acquired_total();
    REQUIRE(acquired_before_stop >= 2);

    owner.request_stop();
    REQUIRE(owner.wait_stopped(kDeadline));
    CHECK(port.stop_calls() == 1);
    CHECK_FALSE(owner.take_latest().has_value());

    // Nothing is acquired once stopped.
    const uint64_t acquired_at_stop = port.acquired_total();
    std::this_thread::sleep_for(50ms);
    CHECK(port.acquired_total() == acquired_at_stop);

    // Every acquired buffer comes back while the owner keeps draining, before close().
    const size_t held_count = held.size();
    held.clear();
    REQUIRE(wait_until([&] { return port.released_total() == port.acquired_total(); }));
    CHECK(held_count == 2);
    CHECK(port.stop_calls() == 1);

    owner.close();
    CHECK(port.balanced());
    CHECK(owner.rejected_returns() == 0);
}

TEST_CASE("close releases every buffer posted before it and is idempotent", "[buffer_owner]")
{
    SECTION("buffers posted just before close")
    {
        FakePort port(8);
        port.set_budget(kUnlimited);
        BufferOwner owner(port, 8, kWait);
        owner.start();
        std::vector<CaptureFrame> held = take_frames(owner, 4);
        REQUIRE(held.size() == 4);
        port.set_budget(0);
        held.clear();
        owner.close();
        CHECK(port.balanced());
        owner.close();
        CHECK(port.balanced());
        CHECK(owner.rejected_returns() == 0);
    }
    SECTION("destructor without start")
    {
        FakePort port(4);
        {
            BufferOwner owner(port, 4, kWait);
        }
        CHECK(port.acquire_calls() == 0);
        CHECK(port.threads().empty());
    }
    SECTION("close without start")
    {
        FakePort port(4);
        BufferOwner owner(port, 4, kWait);
        owner.close();
        owner.close();
        CHECK(port.threads().empty());
    }
    SECTION("destructor without close")
    {
        FakePort port(4);
        port.set_budget(kUnlimited);
        {
            BufferOwner owner(port, 4, kWait);
            owner.start();
            REQUIRE(wait_until([&] { return port.acquired_total() >= 2; }));
        }
        CHECK(port.balanced());
    }
    SECTION("stop without start completes")
    {
        FakePort port(4);
        BufferOwner owner(port, 4, kWait);
        owner.request_stop();
        CHECK(owner.wait_stopped(kDeadline));
        CHECK(port.stop_calls() == 0);
    }
    SECTION("starting twice throws")
    {
        FakePort port(4);
        BufferOwner owner(port, 4, kWait);
        owner.start();
        CHECK_THROWS_AS(owner.start(), std::logic_error);
        owner.close();
    }
}

TEST_CASE("failures surface from take_latest and never lose a buffer", "[buffer_owner]")
{
    SECTION("acquire failed while holding a buffer")
    {
        FakePort port(4);
        port.set_fail_next(true);
        port.set_budget(1);
        BufferOwner owner(port, 4, kWait);
        owner.start();
        std::string what;
        REQUIRE(take_throws(owner, &what));
        CHECK_THAT(what, ContainsSubstring("stream exploded"));
        REQUIRE(wait_until([&] { return port.released_total() == 1; }));
        owner.close();
        CHECK(port.released_total() == 1);
        CHECK(port.balanced());
    }
    SECTION("release refused")
    {
        FakePort port(4);
        port.set_release_ok(false);
        port.set_budget(2); // the second frame replaces, and so releases, the first
        BufferOwner owner(port, 4, kWait);
        owner.start();
        std::string what;
        REQUIRE(take_throws(owner, &what));
        CHECK_THAT(what, ContainsSubstring("refused"));
        owner.close();
    }
    SECTION("stop_capture refused")
    {
        FakePort port(4);
        port.set_stop_ok(false);
        BufferOwner owner(port, 4, kWait);
        owner.start();
        owner.request_stop();
        CHECK_FALSE(owner.wait_stopped(kDeadline));
        CHECK(port.stop_calls() == 1);
        owner.close();
    }
    SECTION("wait_stopped times out when nobody asked to stop")
    {
        FakePort port(4);
        BufferOwner owner(port, 4, kWait);
        owner.start();
        CHECK_FALSE(owner.wait_stopped(20ms));
        owner.close();
    }
    SECTION("a throwing acquire does not terminate the process")
    {
        FakePort port(4);
        port.set_throw_on_acquire(true);
        BufferOwner owner(port, 4, kWait);
        owner.start();
        std::string what;
        REQUIRE(take_throws(owner, &what));
        CHECK_THAT(what, ContainsSubstring("acquire threw"));
        owner.request_stop();
        CHECK(owner.wait_stopped(kDeadline));
        owner.close();
    }
}

TEST_CASE("a lease keeps its keepalive object alive; the owner holds it only weakly", "[buffer_owner]")
{
    struct Probe
    {
        explicit Probe(std::atomic<bool>& flag) : destroyed(flag)
        {
        }
        ~Probe()
        {
            destroyed = true;
        }
        std::atomic<bool>& destroyed;
    };

    SECTION("with no leases the owner does not keep it")
    {
        FakePort port(4);
        BufferOwner owner(port, 4, kWait);
        std::atomic<bool> destroyed{ false };
        std::shared_ptr<void> keep = std::make_shared<Probe>(destroyed);
        owner.set_keepalive(keep);
        owner.start();
        keep.reset();
        CHECK(destroyed);
        owner.close();
    }
    SECTION("a held lease keeps it until dropped")
    {
        FakePort port(4);
        port.set_budget(1);
        BufferOwner owner(port, 4, kWait);
        std::atomic<bool> destroyed{ false };
        std::shared_ptr<void> keep = std::make_shared<Probe>(destroyed);
        owner.set_keepalive(keep);
        owner.start();

        std::vector<CaptureFrame> held = take_frames(owner, 1);
        REQUIRE(held.size() == 1);
        keep.reset();
        CHECK_FALSE(destroyed);

        held.clear();
        CHECK(destroyed);
        REQUIRE(wait_until([&] { return port.released_total() == 1; }));
        owner.close();
        CHECK(port.balanced());
    }
}

TEST_CASE("the mailbox rejects a post beyond its capacity", "[buffer_owner]")
{
    BufferReturnMailbox mailbox(2);
    CHECK(mailbox.post(0));
    CHECK(mailbox.post(1));
    CHECK_FALSE(mailbox.post(1));
    CHECK(mailbox.rejected() == 1);
    std::vector<uint32_t> out;
    mailbox.take(out);
    CHECK(out == std::vector<uint32_t>{ 0, 1 });
}

TEST_CASE("stop returns the held frame's buffer before stop_capture begins", "[buffer_owner]")
{
    FakePort port(8);
    port.set_budget(kUnlimited);
    BufferOwner owner(port, 8, kWait);
    owner.start();

    // The consumer never takes a frame, so the latest one is always held by the owner.
    REQUIRE(wait_until([&] { return port.released_total() >= 3; }));
    owner.request_stop();
    REQUIRE(owner.wait_stopped(kDeadline));

    const auto events = port.events();
    size_t stop_at = events.size();
    for (size_t i = 0; i < events.size(); ++i)
    {
        if (events[i].kind == FakePort::Event::Kind::StopBegin)
        {
            stop_at = i;
            break;
        }
    }
    REQUIRE(stop_at < events.size());

    // The last buffer acquired before the stop is the latest frame.
    bool found_acquire = false;
    uint32_t latest_slot = 0;
    for (size_t i = 0; i < stop_at; ++i)
    {
        if (events[i].kind == FakePort::Event::Kind::Acquire)
        {
            found_acquire = true;
            latest_slot = events[i].slot;
        }
    }
    REQUIRE(found_acquire);
    size_t acquired_before = 0;
    size_t released_before = 0;
    bool latest_released_first = false;
    size_t last_acquire = 0;
    for (size_t i = 0; i < stop_at; ++i)
        if (events[i].kind == FakePort::Event::Kind::Acquire)
            last_acquire = i;
    for (size_t i = 0; i < stop_at; ++i)
    {
        acquired_before += events[i].kind == FakePort::Event::Kind::Acquire;
        released_before += events[i].kind == FakePort::Event::Kind::Release;
        if (i > last_acquire && events[i].kind == FakePort::Event::Kind::Release && events[i].slot == latest_slot)
            latest_released_first = true;
    }
    CHECK(latest_released_first);
    CHECK(released_before == acquired_before);
    for (size_t i = stop_at + 1; i < events.size(); ++i)
        CHECK(events[i].kind != FakePort::Event::Kind::Acquire);

    owner.close();
    CHECK(port.balanced());
    CHECK(owner.rejected_returns() == 0);
}

TEST_CASE("a refused or throwing release keeps the buffer out and ends acquisition", "[buffer_owner]")
{
    constexpr size_t kPool = 4;
    FakePort port(kPool);
    port.set_budget(kUnlimited);

    std::string expected;
    SECTION("refused")
    {
        port.set_release_ok(false);
        expected = "refused";
    }
    SECTION("throws")
    {
        port.set_throw_on_release(true);
        expected = "release threw";
    }

    BufferOwner owner(port, kPool, kWait);
    owner.start();
    std::string what;
    REQUIRE(take_throws(owner, &what));
    CHECK_THAT(what, ContainsSubstring(expected));

    // Let anything in flight finish; after that nothing more is acquired or released.
    std::this_thread::sleep_for(50ms);
    const uint64_t acquires = port.acquire_calls();
    const uint64_t acquired = port.acquired_total();
    const uint64_t releases = port.release_calls();
    std::this_thread::sleep_for(150ms);
    CHECK(port.acquire_calls() == acquires);
    CHECK(port.release_calls() == releases);
    CHECK(acquired <= kPool);
    for (size_t i = 0; i < kPool; ++i)
        CHECK(port.acquired(i) <= 1); // the refused slot is never handed out again
    CHECK(port.release_calls() == 1); // the refused buffer is not retried
    CHECK_THROWS_AS(owner.take_latest(), std::runtime_error);

    owner.request_stop();
    CHECK(owner.wait_stopped(kDeadline));
    owner.close();
    CHECK(port.acquire_calls() == acquires);
}

TEST_CASE("a refused release with a full pool does not over-acquire", "[buffer_owner]")
{
    // The stream has more slots than the owner's pool, so only the owner's own count can stop it.
    constexpr size_t kPool = 2;
    FakePort port(8);
    port.set_budget(kUnlimited);
    port.set_release_ok(false);
    BufferOwner owner(port, kPool, kWait);
    owner.start();
    REQUIRE(take_throws(owner));
    std::this_thread::sleep_for(50ms);
    const uint64_t acquires = port.acquire_calls();
    std::this_thread::sleep_for(100ms);
    CHECK(port.acquire_calls() == acquires);
    CHECK(port.acquired_total() <= kPool);
    owner.close();
}
