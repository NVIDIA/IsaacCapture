// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/direct_sender.hpp"

#include <arpa/inet.h>
#include <log_bridge/logger.hpp>
#include <netinet/tcp.h>
#include <sensing_livekit/clock_sync.hpp>
#include <sys/socket.h>
#include <video_stage/avs1.hpp>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
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

DirectSender::DirectSender(const DirectSenderConfig& config)
    : m_config(config), m_host(config.host), m_port(config.port), m_has_target(!config.host.empty())
{
    m_thread = std::thread(&DirectSender::run, this);
}

DirectSender::~DirectSender()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_stopping = true;
    m_wake.notify_all();
    // Unblock a connect, send or recv that the worker thread is sitting in; it then sees m_stopping and exits.
    {
        std::lock_guard<std::mutex> lock(m_fd_mutex);
        for (const int fd : m_fds)
            if (fd >= 0)
                ::shutdown(fd, SHUT_RDWR);
    }
    if (m_thread.joinable())
        m_thread.join();
}

void DirectSender::flush_stale_locked()
{
    m_dropped += m_queue.size();
    m_queue.clear();
    m_queued_bytes = 0;
    m_need_keyframe = true;
    m_keyframe_requested = true;
}

void DirectSender::set_target(const std::string& host, uint16_t port)
{
    {
        std::lock_guard<std::mutex> lock(m_target_mutex);
        m_host = host;
        m_port = host.empty() ? 0 : port;
        m_has_target = !host.empty();
        ++m_target_generation;
    }
    set_error({});
    {
        // Whatever waits was meant for the old destination, and the new one needs a keyframe first.
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.clear();
        m_queued_bytes = 0;
        m_need_keyframe = true;
        m_keyframe_requested = true;
    }
    {
        // A connect, send or receive in progress is for the old destination: cut it so the worker moves on.
        std::lock_guard<std::mutex> lock(m_fd_mutex);
        for (const int fd : m_fds)
            if (fd >= 0)
                ::shutdown(fd, SHUT_RDWR);
    }
    m_wake.notify_all();
}

std::string DirectSender::last_error() const
{
    std::lock_guard<std::mutex> lock(m_error_mutex);
    return m_last_error;
}

void DirectSender::set_error(const std::string& error)
{
    std::lock_guard<std::mutex> lock(m_error_mutex);
    m_last_error = error;
}

std::string DirectSender::host() const
{
    std::lock_guard<std::mutex> lock(m_target_mutex);
    return m_host;
}

uint16_t DirectSender::port() const
{
    std::lock_guard<std::mutex> lock(m_target_mutex);
    return m_port;
}

void DirectSender::current_target(std::string& host, uint16_t& port, uint64_t& generation) const
{
    std::lock_guard<std::mutex> lock(m_target_mutex);
    host = m_host;
    port = m_port;
    generation = m_target_generation;
}

void DirectSender::submit(const EncodedVideoFrame& frame)
{
    // With no destination there is nothing to queue, and nothing to ask the encoder for.
    if (!m_has_target.load())
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_queue.empty() && std::chrono::steady_clock::now() - m_queue.front().enqueued > m_config.max_queue_age)
        flush_stale_locked();
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
    m_queue.push_back({ std::move(record), std::chrono::steady_clock::now() });
    m_wake.notify_one();
}

bool DirectSender::take_keyframe_request()
{
    return m_keyframe_requested.exchange(false);
}

bool DirectSender::connect_once(
    int& fd, int slot, const std::string& host, uint16_t port, uint64_t generation, std::string& why)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0)
    {
        why = "cannot resolve the address";
        return false;
    }
    fd = -1;
    for (addrinfo* a = found; a != nullptr && fd < 0 && !m_stopping; a = a->ai_next)
    {
        const int s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s < 0)
            continue;
        ::fcntl(s, F_SETFL, ::fcntl(s, F_GETFL, 0) | O_NONBLOCK);
        // Set before connect so the handshake itself is bounded, and publish so the destructor can cancel it.
        const unsigned user_timeout_ms = 5000;
        ::setsockopt(s, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_timeout_ms, sizeof(user_timeout_ms));
        {
            std::lock_guard<std::mutex> lock(m_fd_mutex);
            m_fds[slot] = s;
        }
        bool ok = false;
        int rc = ::connect(s, a->ai_addr, a->ai_addrlen);
        if (rc == 0)
        {
            ok = true;
        }
        else if (errno == EINPROGRESS)
        {
            // Poll in short slices so a stop request is noticed within 100 ms.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!m_stopping && m_target_generation.load() == generation)
            {
                pollfd pfd{ s, POLLOUT, 0 };
                const int n = ::poll(&pfd, 1, 100);
                if (n < 0 && errno != EINTR)
                {
                    why = std::strerror(errno);
                    break;
                }
                if (n > 0)
                {
                    int err = 0;
                    socklen_t len = sizeof(err);
                    ::getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &len);
                    if (err == 0)
                        ok = true;
                    else
                        why = std::strerror(err);
                    break;
                }
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    why = "connection timed out";
                    break;
                }
            }
        }
        else
        {
            why = std::strerror(errno);
        }
        if (ok && !m_stopping && m_target_generation.load() == generation)
        {
            ::fcntl(s, F_SETFL, ::fcntl(s, F_GETFL, 0) & ~O_NONBLOCK);
            fd = s;
            break;
        }
        {
            std::lock_guard<std::mutex> lock(m_fd_mutex);
            m_fds[slot] = -1;
            ::close(s);
        }
    }
    ::freeaddrinfo(found);
    if (fd < 0)
    {
        if (m_stopping)
            why = "stopping";
        return false;
    }
    // The link may be wired or Wi-Fi, so make it behave under both: send small packets at once, keep the kernel's send
    // buffer small so a backlog stays in our queue (where its age is bounded) instead of hiding in the kernel, and
    // notice a dead link in seconds, not minutes.
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const int send_buffer = 256 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    const int idle = 5, interval = 2, count = 3;
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
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
    bool logged_failure = false;
    std::chrono::steady_clock::time_point last_failure_log;
    std::thread control;
    uint64_t connected_generation = 0;
    const auto close_connection = [&]
    {
        if (control_fd >= 0)
            ::shutdown(control_fd, SHUT_RDWR); // unblocks the control thread
        if (control.joinable())
            control.join();
        // Close under the lock the destructor shuts down under, so it never touches a closed or reused descriptor.
        std::lock_guard<std::mutex> lock(m_fd_mutex);
        if (data_fd >= 0)
            ::close(data_fd);
        if (control_fd >= 0)
            ::close(control_fd);
        data_fd = control_fd = -1;
        m_fds[0] = m_fds[1] = -1;
        m_connected = false;
    };
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop)
                break;
        }
        std::string host;
        uint16_t port = 0;
        uint64_t generation = 0;
        current_target(host, port, generation);
        if (data_fd >= 0 && generation != connected_generation)
            close_connection(); // retargeted: leave the old destination
        if (host.empty())
        {
            // No destination: idle until one is set.
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait_for(lock, std::chrono::milliseconds(200), [this] { return m_stop; });
            continue;
        }
        if (data_fd < 0)
        {
            std::string why = "the hello could not be sent";
            const bool up = connect_once(data_fd, 0, host, port, generation, why) &&
                            send_all(data_fd, reinterpret_cast<const uint8_t*>("DATA"), 4) &&
                            connect_once(control_fd, 1, host, port, generation, why) &&
                            send_all(control_fd, reinterpret_cast<const uint8_t*>("CTRL"), 4);
            if (!up)
            {
                if (m_stopping)
                {
                    close_connection();
                    break;
                }
                if (m_target_generation.load() != generation)
                {
                    close_connection(); // retargeted while connecting: the failure belongs to the old destination
                    continue;
                }
                // Say so on the first failure and then every 30 s: a producer that cannot reach the PC is otherwise
                // indistinguishable from one that never tried.
                const auto now = std::chrono::steady_clock::now();
                if (!logged_failure || now - last_failure_log >= std::chrono::seconds(30))
                {
                    logger()->warn("cannot connect to {}:{}: {}; retrying every second", host, port, why);
                    logged_failure = true;
                    last_failure_log = now;
                }
                set_error("cannot connect to " + host + ":" + std::to_string(port) + ": " + why);
                close_connection();
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait_for(lock, std::chrono::seconds(1), [this] { return m_stop; });
                continue;
            }
            logged_failure = false;
            connected_generation = generation;
            m_connected = true;
            set_error({});
            logger()->info("connected to {}:{}", host, port);
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
                    // Only if this is still the destination: a retarget cuts the old connection on purpose.
                    if (m_target_generation.load() == generation)
                        set_error(host + ":" + std::to_string(port) + " closed the control connection");
                    logger()->warn("{}:{} closed the control connection; reconnecting", host, port);
                    lock.unlock();
                    close_connection();
                }
                continue;
            }
            if (std::chrono::steady_clock::now() - m_queue.front().enqueued > m_config.max_queue_age)
            {
                // The link could not keep up: what waits is already late, so drop it rather than send it.
                flush_stale_locked();
                continue;
            }
            record = std::move(m_queue.front().bytes);
            m_queue.pop_front();
            m_queued_bytes -= record.size();
        }
        if (send_all(data_fd, record.data(), record.size()))
        {
            ++m_sent;
            continue;
        }
        if (m_target_generation.load() == generation)
        {
            logger()->warn("send to {}:{} failed; reconnecting", host, port);
            set_error("sending to " + host + ":" + std::to_string(port) + " failed");
        }
        close_connection();
    }
    close_connection();
}

} // namespace sensing
} // namespace plugins
