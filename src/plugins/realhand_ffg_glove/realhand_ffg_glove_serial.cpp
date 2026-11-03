// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_serial.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <glob.h>
#include <stdexcept>

#ifndef _WIN32
#    include <sys/file.h>
#    include <sys/ioctl.h>
#    include <sys/select.h>
#    include <sys/stat.h>

#    include <fcntl.h>
#    include <poll.h>
#    include <termios.h>
#    include <unistd.h>
#endif

namespace plugins::realhand_ffg_glove
{

#ifndef _WIN32
namespace
{

speed_t baud_to_termios(int baudrate)
{
    switch (baudrate)
    {
#    ifdef B2000000
    case 2'000'000:
        return B2000000;
#    endif
#    ifdef B1000000
    case 1'000'000:
        return B1000000;
#    endif
#    ifdef B921600
    case 921'600:
        return B921600;
#    endif
#    ifdef B460800
    case 460'800:
        return B460800;
#    endif
    default:
        throw std::runtime_error("RealHand FFG Glove: unsupported serial baud rate " + std::to_string(baudrate));
    }
}

void append_glob(const char* pattern, std::vector<std::string>& paths)
{
    glob_t matches{};
    if (::glob(pattern, GLOB_NOSORT, nullptr, &matches) == 0)
    {
        for (std::size_t i = 0; i < matches.gl_pathc; ++i)
        {
            paths.emplace_back(matches.gl_pathv[i]);
        }
    }
    ::globfree(&matches);
}

bool port_is_open(const std::string& port)
{
    struct stat device
    {
    };
    if (::stat(port.c_str(), &device) != 0 || !S_ISCHR(device.st_mode))
    {
        throw std::runtime_error("not a serial character device: " + port);
    }
#    ifdef __linux__
    // TIOCEXCL blocks future opens, but does not detect existing non-exclusive users.
    std::error_code error;
    for (const auto& process : std::filesystem::directory_iterator("/proc", error))
    {
        const auto name = process.path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char ch) { return std::isdigit(ch); }))
        {
            continue;
        }
        std::error_code fd_error;
        for (const auto& fd : std::filesystem::directory_iterator(process.path() / "fd", fd_error))
        {
            struct stat opened
            {
            };
            if (::stat(fd.path().c_str(), &opened) == 0 && S_ISCHR(opened.st_mode) && opened.st_rdev == device.st_rdev)
            {
                return true;
            }
        }
    }
#    endif
    return false;
}

} // namespace
#endif

SerialGlove::SerialGlove(const std::string& port, int baudrate) : port_(port), baudrate_(baudrate)
{
#ifdef _WIN32
    throw std::runtime_error("RealHand FFG Glove serial backend currently supports POSIX systems only");
#else
    const speed_t speed = baud_to_termios(baudrate);
    if (port_is_open(port))
    {
        throw std::runtime_error("serial port is already in use: " + port);
    }
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
    {
        throw std::runtime_error("cannot open " + port + ": " + std::strerror(errno));
    }
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0 || ::ioctl(fd_, TIOCEXCL) != 0)
    {
        const std::string error = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("cannot exclusively claim " + port + ": " + error);
    }
    termios tty{};
    if (::tcgetattr(fd_, &tty) != 0)
    {
        const std::string error = std::strerror(errno);
        ::ioctl(fd_, TIOCNXCL);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("tcgetattr failed for " + port + ": " + error);
    }
    ::cfmakeraw(&tty);
    ::cfsetispeed(&tty, speed);
    ::cfsetospeed(&tty, speed);
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
#    ifdef CRTSCTS
    tty.c_cflag &= ~CRTSCTS;
#    endif
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    if (::tcsetattr(fd_, TCSANOW, &tty) != 0)
    {
        const std::string error = std::strerror(errno);
        ::ioctl(fd_, TIOCNXCL);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("tcsetattr failed for " + port + ": " + error);
    }
    ::tcflush(fd_, TCIOFLUSH);
#endif
}

SerialGlove::~SerialGlove()
{
#ifndef _WIN32
    if (fd_ >= 0)
    {
        ::ioctl(fd_, TIOCNXCL);
        ::close(fd_);
    }
#endif
}

void SerialGlove::send(Command command, const std::vector<uint8_t>& payload)
{
#ifndef _WIN32
    const auto frame = pack_frame(command, payload);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    std::size_t sent = 0;
    while (sent < frame.size())
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            throw std::runtime_error("write timed out for " + port_);
        }
        const ssize_t count = ::write(fd_, frame.data() + sent, frame.size() - sent);
        if (count < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                pollfd descriptor{ fd_, POLLOUT, 0 };
                const int ready = ::poll(&descriptor, 1, static_cast<int>(std::max<int64_t>(1, remaining.count())));
                if (ready < 0 && errno == EINTR)
                {
                    continue;
                }
                if (ready <= 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)))
                {
                    throw std::runtime_error("write unavailable or timed out for " + port_);
                }
                continue;
            }
            throw std::runtime_error("write failed for " + port_ + ": " + std::strerror(errno));
        }
        if (count == 0)
        {
            throw std::runtime_error("write made no progress for " + port_);
        }
        sent += static_cast<std::size_t>(count);
    }
#else
    (void)command;
    (void)payload;
#endif
}

std::vector<Frame> SerialGlove::read_available(int timeout_ms)
{
    std::vector<Frame> frames;
#ifndef _WIN32
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(fd_, &read_set);
    timeval timeout{ timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    const int ready = ::select(fd_ + 1, &read_set, nullptr, nullptr, &timeout);
    if (ready < 0 && errno != EINTR)
    {
        throw std::runtime_error("select failed for " + port_ + ": " + std::strerror(errno));
    }
    if (ready <= 0)
    {
        return frames;
    }

    std::array<uint8_t, 1024> bytes{};
    while (true)
    {
        const ssize_t count = ::read(fd_, bytes.data(), bytes.size());
        if (count == 0 || (count < 0 && (errno == EAGAIN || errno == EINTR)))
        {
            break;
        }
        if (count < 0)
        {
            throw std::runtime_error("read failed for " + port_ + ": " + std::strerror(errno));
        }
        for (ssize_t i = 0; i < count; ++i)
        {
            if (auto frame = parser_.process(bytes[static_cast<std::size_t>(i)]))
            {
                frames.push_back(std::move(*frame));
            }
        }
        if (count < static_cast<ssize_t>(bytes.size()))
        {
            break;
        }
    }
#else
    (void)timeout_ms;
#endif
    return frames;
}

std::vector<std::string> discover_serial_ports()
{
    std::vector<std::string> paths;
#ifndef _WIN32
    // Generic USB UART IDs cannot distinguish a glove from a robot motor bus.
    append_glob("/dev/serial/by-id/*", paths);
    paths.erase(std::remove_if(paths.begin(), paths.end(),
                               [](const std::string& path)
                               {
                                   auto name = std::filesystem::path(path).filename().string();
                                   std::transform(name.begin(), name.end(), name.begin(),
                                                  [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                                   return name.find("realhand") == std::string::npos &&
                                          name.find("realforce") == std::string::npos &&
                                          name.find("ffg") == std::string::npos;
                               }),
                paths.end());
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
#endif
    return paths;
}

} // namespace plugins::realhand_ffg_glove
