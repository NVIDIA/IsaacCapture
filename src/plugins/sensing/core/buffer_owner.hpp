// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// One thread owns every runtime call on a camera's buffer stream. Frame leases are released from any thread, so they
// only post the buffer's slot to a mailbox; the owner makes the actual release. No vendor SDK type appears here, so
// the ownership rules are testable with a fake port.

#pragma once

#include "inc/sensing_core/camera_capture.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace plugins
{
namespace sensing
{

/// Buffers waiting to go back to the stream. post() is for any thread and makes no outside call; the owner drains.
/// Capacity is the pool size, fixed up front: a slot is posted at most once per lease, so it never reallocates.
class BufferReturnMailbox
{
public:
    explicit BufferReturnMailbox(size_t capacity) : m_capacity(capacity)
    {
        m_pending.reserve(capacity);
    }

    /// False, and counted, if the mailbox is full: that means a slot was returned twice, which is a bug.
    bool post(uint32_t slot)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_pending.size() >= m_capacity)
            {
                ++m_rejected;
                return false;
            }
            m_pending.push_back(slot);
        }
        m_cv.notify_one();
        return true;
    }

    /// Wakes the owner's wait without posting anything.
    void wake()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_woken = true;
        }
        m_cv.notify_one();
    }

    /// Owner: moves every pending slot into `out`.
    void take(std::vector<uint32_t>& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        out.insert(out.end(), m_pending.begin(), m_pending.end());
        m_pending.clear();
    }

    /// Owner: blocks until a slot is posted, wake() is called, or the time is up.
    void wait(std::chrono::nanoseconds timeout)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait_for(lock, timeout, [this] { return !m_pending.empty() || m_woken; });
        m_woken = false;
    }

    uint64_t rejected() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_rejected;
    }

private:
    const size_t m_capacity;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<uint32_t> m_pending;
    bool m_woken = false;
    uint64_t m_rejected = 0;
};

/// The thin SDK calls the owner makes. All of them run on the owner's thread, never two at once.
class BufferPort
{
public:
    enum class Result
    {
        Frame,
        Timeout,
        Failed,
    };

    struct Acquired
    {
        Result result = Result::Timeout;
        /// True when the stream handed over a buffer, whether or not the frame could be built from it: that buffer
        /// is then the owner's to return. The frame's lease is left empty; the owner sets it.
        bool holds_buffer = false;
        uint32_t slot = 0;
        CaptureFrame frame;
        std::string error;
    };

    virtual ~BufferPort() = default;
    /// Waits up to `timeout_ns` for the next frame.
    virtual Acquired acquire(uint64_t timeout_ns) = 0;
    /// Gives the buffer in `slot` back to the stream. False: the stream refused it.
    virtual bool release(uint32_t slot) = 0;
    /// Stops new captures from starting (no waiting). False: it could not.
    virtual bool stop_capture() = 0;
    /// Waits for capture that was stopped to go idle. The owner calls it once every buffer is back, so the wait
    /// never depends on a buffer a consumer still holds. False: it did not go idle.
    virtual bool wait_idle() = 0;
};

/**
 * @brief Runs one camera's buffer stream on its own thread and keeps only the newest frame.
 *
 * The thread blocks on the port's finite acquire, stamps nothing itself, and replaces the single latest frame; a
 * replaced frame goes back through the same mailbox as any other. When every buffer is out it waits on the mailbox
 * instead of in an acquire that needs a returned buffer. Stopping has two parts: request_stop() stops new captures and
 * clears the latest frame but keeps returning buffers; close() ends the thread after draining the mailbox, and only
 * then waits for the stream to go idle. Frames handed
 * out stay valid across both; the owner must outlive every lease it issued, which `keepalive` can tie to a larger
 * object.
 */
class BufferOwner
{
public:
    BufferOwner(BufferPort& port, size_t pool_size, std::chrono::nanoseconds wait = std::chrono::milliseconds(100))
        : m_port(port), m_pool_size(pool_size), m_wait(wait), m_mailbox(pool_size)
    {
    }
    BufferOwner(const BufferOwner&) = delete;
    BufferOwner& operator=(const BufferOwner&) = delete;
    ~BufferOwner()
    {
        close();
    }

    /// Objects the issued leases keep alive until they are released; weak, so the owner never owns its owner.
    void set_keepalive(std::weak_ptr<void> keepalive)
    {
        m_keepalive = std::move(keepalive);
    }

    /// @throws std::system_error if the thread cannot be created.
    void start()
    {
        if (m_started)
            throw std::logic_error("BufferOwner: already started");
        m_thread = std::thread([this] { run(); });
        m_started = true;
    }

    /// Asks the owner thread to stop capture; returns at once.
    void request_stop()
    {
        if (!m_started || !m_thread.joinable())
        {
            finish_stop(true); // no thread is running, so there is nothing to stop through it
            return;
        }
        Command expected = Command::Run;
        m_command.compare_exchange_strong(expected, Command::Stop);
        m_mailbox.wake();
    }

    /// True once capture has stopped and gone idle; false if that took longer than `timeout` or the port said no.
    bool wait_stopped(std::chrono::nanoseconds timeout)
    {
        std::unique_lock<std::mutex> lock(m_state_mutex);
        return m_state_cv.wait_for(lock, timeout, [this] { return m_stop_done; }) && m_stop_ok;
    }

    /// Ends the thread after returning every buffer posted so far; it does not stop capture (see request_stop).
    /// Idempotent. Not for the owner thread itself.
    void close()
    {
        m_command.store(Command::Exit);
        m_mailbox.wake();
        if (m_thread.joinable())
            m_thread.join();
    }

    /// The newest frame not yet taken. @throws std::runtime_error with the reason if the owner failed.
    std::optional<CaptureFrame> take_latest()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (!m_error.empty())
            throw std::runtime_error(m_error);
        std::optional<CaptureFrame> frame = std::move(m_latest);
        m_latest.reset();
        return frame;
    }

    /// After close(): every buffer was returned and, if capture was stopped, the stream went idle with nothing out.
    bool closed_cleanly() const
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return m_closed_clean;
    }

    uint64_t rejected_returns() const
    {
        return m_mailbox.rejected();
    }

private:
    enum class Command
    {
        Run,
        Stop,
        Exit,
    };

    void run()
    {
        std::vector<uint32_t> returned;
        returned.reserve(m_pool_size);
        bool stopped = false;
        for (;;)
        {
            drain(returned);
            const Command command = m_command.load();
            if (command == Command::Exit)
                break;
            if (command == Command::Stop && !stopped)
            {
                // Return the held frame's buffer before the blocking idle wait: the owner cannot return anything while
                // it is inside that wait.
                clear_latest();
                drain(returned);
                bool ok = false;
                try
                {
                    ok = m_port.stop_capture();
                }
                catch (const std::exception& e)
                {
                    fail(e.what());
                }
                finish_stop(ok);
                stopped = true;
                continue;
            }
            // Nothing to acquire into, or nothing more to acquire: wait for returns or a command instead.
            if (stopped || !m_acquiring || m_outstanding >= m_pool_size)
            {
                m_mailbox.wait(m_wait);
                continue;
            }
            BufferPort::Acquired got;
            try
            {
                got = m_port.acquire(static_cast<uint64_t>(m_wait.count()));
            }
            catch (const std::exception& e)
            {
                // A port reports failure through its result; a throw is a bug, and it took no buffer we know of.
                fail(e.what());
                m_acquiring = false;
                continue;
            }
            if (got.holds_buffer)
                ++m_outstanding;
            if (got.result == BufferPort::Result::Timeout)
                continue;
            if (got.result == BufferPort::Result::Failed)
            {
                if (got.holds_buffer)
                    give_back(got.slot);
                fail(got.error.empty() ? "the buffer stream failed" : got.error);
                m_acquiring = false;
                continue;
            }
            got.frame.lease = make_lease(got.slot);
            std::optional<CaptureFrame> replaced;
            {
                std::lock_guard<std::mutex> lock(m_state_mutex);
                replaced = std::move(m_latest);
                m_latest = std::move(got.frame);
            }
            // `replaced` goes out of scope here, outside the lock: its lease posts the buffer back.
        }
        // The thread is ending: every posted buffer goes back, then (if capture was stopped) the stream must go idle
        // with nothing out. A buffer still out, or a stream that will not go idle, is reported to close()'s caller.
        drain(returned);
        clear_latest();
        drain(returned);
        bool clean = m_outstanding == 0;
        if (stopped)
        {
            bool idle = false;
            try
            {
                idle = m_port.wait_idle();
            }
            catch (const std::exception& e)
            {
                fail(e.what());
            }
            if (!idle)
                fail("the buffer stream did not go idle");
            clean = clean && idle;
            drain(returned);
            clean = clean && m_outstanding == 0;
        }
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_closed_clean = clean;
    }

    // Returns every posted buffer to the stream. A buffer the stream refuses stays counted as out, and acquiring
    // stops: the stream is failing and the pool no longer adds up.
    void drain(std::vector<uint32_t>& returned)
    {
        returned.clear();
        m_mailbox.take(returned);
        for (uint32_t slot : returned)
            if (!give_back(slot))
                m_acquiring = false;
    }

    bool give_back(uint32_t slot)
    {
        bool ok = false;
        try
        {
            ok = m_port.release(slot);
        }
        catch (const std::exception& e)
        {
            fail(e.what());
            return false;
        }
        if (!ok)
        {
            fail("the buffer stream refused a returned buffer");
            return false;
        }
        if (m_outstanding > 0)
            --m_outstanding;
        return true;
    }

    void clear_latest()
    {
        std::optional<CaptureFrame> dropped;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            dropped = std::move(m_latest);
            m_latest.reset();
        }
    }

    FrameLease make_lease(uint32_t slot)
    {
        // The owner outlives its leases (see the class comment), so `this` is valid when a lease is dropped. A
        // rejected post means a buffer came back twice: that breaks the pool count, so it fails the owner.
        return FrameLease(
            [this, keep = m_keepalive.lock(), slot]()
            {
                if (!m_mailbox.post(slot))
                    fail("a buffer was returned twice");
            });
    }

    void fail(const std::string& why)
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_error.empty())
            m_error = why;
    }

    void finish_stop(bool ok)
    {
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_stop_done = true;
            m_stop_ok = ok;
        }
        m_state_cv.notify_all();
    }

    BufferPort& m_port;
    const size_t m_pool_size;
    const std::chrono::nanoseconds m_wait;
    BufferReturnMailbox m_mailbox;
    std::weak_ptr<void> m_keepalive;
    std::atomic<Command> m_command{ Command::Run };
    bool m_started = false;
    std::thread m_thread;

    size_t m_outstanding = 0; ///< owner thread only: buffers acquired and not yet released
    bool m_acquiring = true; ///< owner thread only: false once the stream has failed

    mutable std::mutex m_state_mutex; ///< guards the members below
    std::condition_variable m_state_cv;
    std::optional<CaptureFrame> m_latest;
    std::string m_error;
    bool m_stop_done = false;
    bool m_stop_ok = false;
    bool m_closed_clean = true; ///< set by the owner thread as it ends; true if no thread ever ran
};

} // namespace sensing
} // namespace plugins
