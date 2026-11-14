// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_ipc/message_socket.hpp"

#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace core::camera
{

namespace
{

[[noreturn]] void throw_errno(const std::string& what)
{
    throw std::system_error(errno, std::generic_category(), what);
}

sockaddr_un address_of(const std::string& path)
{
    sockaddr_un address = {};
    address.sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof(address.sun_path))
    {
        throw std::invalid_argument("socket path '" + path + "' is empty or too long");
    }
    std::memcpy(address.sun_path, path.data(), path.size());
    return address;
}

UniqueFd new_socket(int extra_flags = 0)
{
    UniqueFd fd(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | extra_flags, 0));
    if (!fd.valid())
    {
        throw_errno("creating a socket");
    }
    return fd;
}

// Waits until the descriptor is readable. Returns false on timeout.
bool wait_readable(int fd, std::chrono::milliseconds timeout)
{
    using Milliseconds = std::chrono::milliseconds;
    const Milliseconds longest(std::numeric_limits<int>::max()); // what poll() can wait
    const auto deadline = std::chrono::steady_clock::now() + std::clamp(timeout, Milliseconds::zero(), longest);
    pollfd entry = { fd, POLLIN, 0 };
    for (;;)
    {
        // An interrupted wait resumes with the time that is left, not the whole timeout again.
        const auto left = std::chrono::ceil<Milliseconds>(deadline - std::chrono::steady_clock::now());
        const int ready = ::poll(&entry, 1, static_cast<int>(std::max(left, Milliseconds::zero()).count()));
        if (ready > 0)
        {
            return true;
        }
        if (ready == 0)
        {
            return false;
        }
        if (errno != EINTR)
        {
            throw_errno("waiting on a socket");
        }
    }
}

} // namespace

MessageSocket MessageSocket::connect(const std::string& path)
{
    const sockaddr_un address = address_of(path);
    UniqueFd fd = new_socket();
    if (::connect(fd.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0)
    {
        throw_errno("connecting to '" + path + "'");
    }
    return MessageSocket(std::move(fd));
}

MessageSocket::MessageSocket(UniqueFd fd) : m_fd(std::move(fd))
{
}

SendStatus MessageSocket::send(std::span<const std::byte> payload, std::span<const int> descriptors)
{
    if (payload.empty() || descriptors.size() > MAX_DESCRIPTORS_PER_MESSAGE)
    {
        throw std::invalid_argument("a message needs a payload and carries at most " +
                                    std::to_string(MAX_DESCRIPTORS_PER_MESSAGE) + " descriptors");
    }
    iovec data = { const_cast<std::byte*>(payload.data()), payload.size() };
    msghdr message = {};
    message.msg_iov = &data;
    message.msg_iovlen = 1;

    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * MAX_DESCRIPTORS_PER_MESSAGE)];
    if (!descriptors.empty())
    {
        message.msg_control = control;
        message.msg_controllen = CMSG_SPACE(descriptors.size_bytes());
        cmsghdr* header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(descriptors.size_bytes());
        std::memcpy(CMSG_DATA(header), descriptors.data(), descriptors.size_bytes());
    }

    for (;;)
    {
        if (::sendmsg(m_fd.get(), &message, MSG_DONTWAIT | MSG_NOSIGNAL) >= 0)
        {
            return SendStatus::SENT;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return SendStatus::WOULD_BLOCK;
        }
        if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)
        {
            return SendStatus::CLOSED;
        }
        if (errno != EINTR)
        {
            throw_errno("sending a message");
        }
    }
}

ReceivedMessage MessageSocket::receive(std::span<std::byte> buffer, std::chrono::milliseconds timeout)
{
    ReceivedMessage result;
    if (!wait_readable(m_fd.get(), timeout))
    {
        return result;
    }
    iovec data = { buffer.data(), buffer.size() };
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * MAX_DESCRIPTORS_PER_MESSAGE)];
    msghdr message = {};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);

    ssize_t received = 0;
    do
    {
        received = ::recvmsg(m_fd.get(), &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    } while (received < 0 && errno == EINTR);
    if (received < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            return result;
        }
        if (errno == ECONNRESET)
        {
            result.status = ReceiveStatus::CLOSED;
            return result;
        }
        throw_errno("receiving a message");
    }

    // Take ownership of every descriptor first, so none leaks if the message is rejected below.
    for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr; header = CMSG_NXTHDR(&message, header))
    {
        if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS)
        {
            continue;
        }
        const std::size_t count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        for (std::size_t i = 0; i < count; ++i)
        {
            int fd = -1;
            std::memcpy(&fd, CMSG_DATA(header) + i * sizeof(int), sizeof(int));
            result.descriptors.emplace_back(fd);
        }
    }
    // Checked before the empty read below: a cut message can also come back with no bytes.
    if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0)
    {
        throw std::length_error("received a message larger than the buffer or with too many descriptors");
    }
    if (received == 0 && result.descriptors.empty())
    {
        result.status = ReceiveStatus::CLOSED; // an orderly shutdown reads as an empty message
        return result;
    }
    result.status = ReceiveStatus::MESSAGE;
    result.bytes = static_cast<std::size_t>(received);
    return result;
}

MessageListener::MessageListener(const std::string& path) : m_path(path)
{
    const sockaddr_un address = address_of(path);

    // The lock decides who owns the path. The kernel drops it when its holder dies, so a socket
    // file found while holding it was left by a process that is gone.
    const std::string lock_path = path + ".lock";
    m_lock.reset(::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR));
    if (!m_lock.valid())
    {
        throw_errno("opening '" + lock_path + "'");
    }
    if (::flock(m_lock.get(), LOCK_EX | LOCK_NB) != 0)
    {
        if (errno == EWOULDBLOCK)
        {
            throw std::system_error(EADDRINUSE, std::generic_category(), "another process listens on '" + path + "'");
        }
        throw_errno("locking '" + lock_path + "'");
    }
    struct stat entry = {};
    if (::lstat(path.c_str(), &entry) == 0)
    {
        if (!S_ISSOCK(entry.st_mode))
        {
            throw std::system_error(EEXIST, std::generic_category(), "'" + path + "' exists and is not a socket");
        }
        ::unlink(path.c_str());
    }
    else if (errno != ENOENT)
    {
        throw_errno("inspecting '" + path + "'");
    }

    // Not blocking: a connection that goes away between the wait and the accept must not stall accept().
    m_fd = new_socket(SOCK_NONBLOCK);
    // The socket file is created with the socket's own mode: only this user may connect.
    if (::fchmod(m_fd.get(), S_IRUSR | S_IWUSR) != 0)
    {
        throw_errno("restricting '" + path + "'");
    }
    if (::bind(m_fd.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0)
    {
        throw_errno("binding '" + path + "'");
    }
    if (::listen(m_fd.get(), SOMAXCONN) != 0)
    {
        const int error = errno;
        ::unlink(path.c_str());
        throw std::system_error(error, std::generic_category(), "listening on '" + path + "'");
    }
}

MessageListener::~MessageListener()
{
    ::unlink(m_path.c_str());
}

std::optional<MessageSocket> MessageListener::accept(std::chrono::milliseconds timeout)
{
    if (!wait_readable(m_fd.get(), timeout))
    {
        return std::nullopt;
    }
    UniqueFd connection(::accept4(m_fd.get(), nullptr, nullptr, SOCK_CLOEXEC));
    if (!connection.valid())
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED || errno == EINTR)
        {
            return std::nullopt;
        }
        throw_errno("accepting a connection on '" + m_path + "'");
    }
    return MessageSocket(std::move(connection));
}

} // namespace core::camera
