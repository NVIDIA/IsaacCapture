// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sensing_livekit/video_track_sink.hpp>

#include <atomic>
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
    bool connect_once(int& fd, std::string& why);
    bool control_alive() const
    {
        return !m_control_ended.load();
    }

    DirectSenderConfig m_config;
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::deque<std::vector<uint8_t>> m_queue;
    std::size_t m_queued_bytes = 0;
    bool m_need_keyframe = true; ///< guarded by m_mutex
    bool m_stop = false; ///< guarded by m_mutex
    std::uint64_t m_sequence = 0;
    std::atomic<bool> m_keyframe_requested{ false };
    std::atomic<bool> m_control_ended{ false };
    std::atomic<std::uint64_t> m_sent{ 0 };
    std::atomic<std::uint64_t> m_dropped{ 0 };
};

} // namespace sensing
} // namespace plugins
