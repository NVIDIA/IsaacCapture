// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "direct_sender.hpp"

#include <arpa/inet.h>
#include <log_bridge/logger.hpp>
#include <netinet/tcp.h>
#include <sensing_livekit/clock_sync.hpp>
#include <sys/socket.h>
#include <video_stage/avs1.hpp>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <netdb.h>
#include <unistd.h>

namespace plugins
{
namespace sensing
{

namespace
{

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.sensing.DirectSender");
}

bool recv_exact(int fd, uint8_t* out, std::size_t size)
{
    std::size_t done = 0;
    while (done < size)
    {
        const ssize_t n = ::recv(fd, out + done, size - done, 0);
        if (n == 0)
            return false;
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

bool send_all(int fd, const uint8_t* data, std::size_t size)
{
    while (size > 0)
    {
        const ssize_t n = ::send(fd, data, size, MSG_NOSIGNAL);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        data += n;
        size -= static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace

DirectSender::DirectSender(const DirectSenderConfig& config) : m_config(config)
{
    m_thread = std::thread(&DirectSender::run, this);
}

DirectSender::~DirectSender()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    if (m_thread.joinable())
        m_thread.join();
}

void DirectSender::submit(const EncodedVideoFrame& frame)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_need_keyframe)
    {
        if (!frame.keyframe)
            return; // a delta frame cannot start a stream; the keyframe request is already pending
        m_need_keyframe = false;
    }
    if (m_queued_bytes + frame.size > m_config.max_queued_bytes)
    {
        // Dropping any frame breaks the chain after it: discard up to the next keyframe, and ask for one.
        ++m_dropped;
        m_need_keyframe = true;
        m_keyframe_requested = true;
        return;
    }
    video_stage::avs1::Sample sample;
    sample.width = m_config.width;
    sample.height = m_config.height;
    sample.fps_numerator = m_config.fps;
    sample.bundle_sequence = ++m_sequence;
    sample.session_id = 1;
    sample.frame_number = frame.frame_id;
    sample.frame_timestamp_ns = frame.user_timestamp_us * 1000;
    std::vector<uint8_t> record;
    video_stage::avs1::encode(sample, frame.data, frame.size, record);
    m_queued_bytes += record.size();
    m_queue.push_back(std::move(record));
    m_wake.notify_one();
}

bool DirectSender::take_keyframe_request()
{
    return m_keyframe_requested.exchange(false);
}

bool DirectSender::connect_once(int& fd)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (::getaddrinfo(m_config.host.c_str(), std::to_string(m_config.port).c_str(), &hints, &found) != 0)
        return false;
    fd = -1;
    for (addrinfo* a = found; a != nullptr; a = a->ai_next)
    {
        const int s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s < 0)
            continue;
        if (::connect(s, a->ai_addr, a->ai_addrlen) == 0)
        {
            fd = s;
            break;
        }
        ::close(s);
    }
    ::freeaddrinfo(found);
    if (fd < 0)
        return false;
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return true;
}

// The control connection carries only small messages from the PC: a keyframe request ('K') and clock-exchange
// requests ('Q', answered with 'R'). This thread blocks in recv so t2 is stamped when a request actually arrives, not
// when the record sender next looks at the socket; it ends when the connection does.
void DirectSender::control_loop(int fd)
{
    m_control_ended = false;
    run_control(fd);
    m_control_ended = true;
}

void DirectSender::run_control(int fd)
{
    while (true)
    {
        uint8_t type = 0;
        if (!recv_exact(fd, &type, 1))
            return;
        const int64_t t2 = monotonic_ns();
        if (type == 'K')
        {
            m_keyframe_requested = true;
        }
        else if (type == 'Q')
        {
            std::vector<uint8_t> request(kClockSyncRequestSize);
            request[0] = type;
            if (!recv_exact(fd, request.data() + 1, request.size() - 1))
                return;
            const auto parsed = parse_clock_sync_request(request);
            if (!parsed)
                continue;
            std::vector<uint8_t> reply = build_clock_sync_reply(*parsed, t2, 0);
            set_clock_sync_reply_t3(reply, monotonic_ns());
            if (!send_all(fd, reply.data(), reply.size()))
                return;
        }
        // Anything else is ignored: the PC sends nothing more.
    }
}

void DirectSender::run()
{
    int data_fd = -1;
    int control_fd = -1;
    std::thread control;
    const auto close_connection = [&]
    {
        if (control_fd >= 0)
            ::shutdown(control_fd, SHUT_RDWR); // unblocks the control thread
        if (control.joinable())
            control.join();
        if (data_fd >= 0)
            ::close(data_fd);
        if (control_fd >= 0)
            ::close(control_fd);
        data_fd = control_fd = -1;
    };
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop)
                break;
        }
        if (data_fd < 0)
        {
            const bool up = connect_once(data_fd) && send_all(data_fd, reinterpret_cast<const uint8_t*>("DATA"), 4) &&
                            connect_once(control_fd) &&
                            send_all(control_fd, reinterpret_cast<const uint8_t*>("CTRL"), 4);
            if (!up)
            {
                close_connection();
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait_for(lock, std::chrono::seconds(1), [this] { return m_stop; });
                continue;
            }
            logger()->info("connected to {}:{}", m_config.host, m_config.port);
            control = std::thread(&DirectSender::control_loop, this, control_fd);
            // The PC needs a keyframe first: drop anything queued before the connection and ask for one.
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.clear();
            m_queued_bytes = 0;
            m_need_keyframe = true;
            m_keyframe_requested = true;
        }
        std::vector<uint8_t> record;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            // Wake periodically so a control connection that ended is noticed even when no frames are arriving.
            m_wake.wait_for(lock, std::chrono::milliseconds(200), [this] { return m_stop || !m_queue.empty(); });
            if (m_stop)
                break;
            if (m_queue.empty())
            {
                if (!control_alive())
                {
                    logger()->warn("{}:{} closed the control connection; reconnecting", m_config.host, m_config.port);
                    lock.unlock();
                    close_connection();
                }
                continue;
            }
            record = std::move(m_queue.front());
            m_queue.pop_front();
            m_queued_bytes -= record.size();
        }
        if (send_all(data_fd, record.data(), record.size()))
        {
            ++m_sent;
            continue;
        }
        logger()->warn("send to {}:{} failed; reconnecting", m_config.host, m_config.port);
        close_connection();
    }
    close_connection();
}

} // namespace sensing
} // namespace plugins
