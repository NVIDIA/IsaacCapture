// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sensing_livekit/video_track_sink.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace plugins
{
namespace sensing
{

struct DirectSenderConfig
{
    std::string host;
    uint16_t port = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 30;
    /// Encoded bytes allowed to wait for the network. A frame that does not fit is dropped, and so is everything after
    /// it up to the next keyframe, because those frames reference it.
    std::size_t max_queued_bytes = 16U << 20;
    /// A unit that has waited this long for the network is stale. On a slow or lossy link (Wi-Fi, a congested switch) a
    /// backlog is worse than a gap: when the oldest queued unit is older than this, the whole queue is dropped and the
    /// stream resumes at the next keyframe, so the PC never receives video that is already late.
    std::chrono::milliseconds max_queue_age{ 250 };
};

/**
 * @brief Sends each encoded access unit to the PC over TCP as an AVS1 record, beside the LiveKit publication.
 *
 * submit() never blocks the encoder thread: it copies the unit into a bounded queue that a sender thread drains. The
 * connection starts at a keyframe. When the link is down or the queue overflowed, units are discarded until the next
 * keyframe and a keyframe is requested; take_keyframe_request() hands that request to the encoder's owner. The PC
 * connects nothing back: this side connects and reconnects, so the worker can be restarted freely.
 */
class DirectSender
{
public:
    explicit DirectSender(const DirectSenderConfig& config);
    ~DirectSender();

    DirectSender(const DirectSender&) = delete;
    DirectSender& operator=(const DirectSender&) = delete;

    /// Offer one unit. `frame.user_timestamp_us` and `frame.frame_id` travel as the record's stamp and frame number.
    void submit(const EncodedVideoFrame& frame);

    /// True once after a (re)connection or a drop that needs a fresh keyframe.
    bool take_keyframe_request();

    /// Send to host:port from now on (empty host stops sending). Whatever is queued is discarded, the current
    /// connection is cut, and the new destination starts at a keyframe. Safe to call from any thread.
    void set_target(const std::string& host, std::uint16_t port);
    std::string host() const;
    /// Why the link is not up (the last connect or send failure), empty while it is up or not yet tried.
    std::string last_error() const;
    std::uint16_t port() const;
    /// Both connections to the current destination are up.
    bool connected() const
    {
        return m_connected.load();
    }

    std::uint64_t sent() const
    {
        return m_sent.load();
    }
    std::uint64_t dropped() const
    {
        return m_dropped.load();
    }

private:
    void run();
    void control_loop(int fd);
    void run_control(int fd);
    /// Connects without blocking past a stop request; \p slot (0 data, 1 control) publishes the fd to the destructor.
    bool connect_once(
        int& fd, int slot, const std::string& host, std::uint16_t port, std::uint64_t generation, std::string& why);
    void set_error(const std::string& error);
    void current_target(std::string& host, std::uint16_t& port, std::uint64_t& generation) const;
    /// Drops every queued unit and waits for a keyframe, asking the encoder for one. Caller holds m_mutex.
    void flush_stale_locked();
    bool control_alive() const
    {
        return !m_control_ended.load();
    }

    DirectSenderConfig m_config;
    mutable std::mutex m_target_mutex;
    std::string m_host; ///< guarded by m_target_mutex
    std::uint16_t m_port = 0; ///< guarded by m_target_mutex
    std::atomic<bool> m_has_target{ false };
    std::atomic<std::uint64_t> m_target_generation{ 0 };
    std::atomic<bool> m_connected{ false };
    mutable std::mutex m_error_mutex;
    std::string m_last_error; ///< guarded by m_error_mutex
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    struct Queued
    {
        std::vector<uint8_t> bytes;
        std::chrono::steady_clock::time_point enqueued;
    };
    std::deque<Queued> m_queue;
    std::size_t m_queued_bytes = 0;
    bool m_need_keyframe = true; ///< guarded by m_mutex
    bool m_stop = false; ///< guarded by m_mutex
    std::uint64_t m_sequence = 0;
    std::atomic<bool> m_stopping{ false };
    std::mutex m_fd_mutex;
    int m_fds[2] = { -1, -1 }; ///< live sockets, guarded by m_fd_mutex
    std::atomic<bool> m_keyframe_requested{ false };
    std::atomic<bool> m_control_ended{ false };
    std::atomic<std::uint64_t> m_sent{ 0 };
    std::atomic<std::uint64_t> m_dropped{ 0 };
};

} // namespace sensing
} // namespace plugins
