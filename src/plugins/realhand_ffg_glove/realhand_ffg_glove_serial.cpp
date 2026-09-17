// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_serial.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <glob.h>
#include <stdexcept>

#ifndef _WIN32
#    include <sys/select.h>

#    include <fcntl.h>
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

} // namespace
#endif

SerialGlove::SerialGlove(const std::string& port, int baudrate) : port_(port), baudrate_(baudrate)
{
#ifdef _WIN32
    throw std::runtime_error("RealHand FFG Glove serial backend currently supports POSIX systems only");
#else
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
    {
        throw std::runtime_error("cannot open " + port + ": " + std::strerror(errno));
    }
    termios tty{};
    if (::tcgetattr(fd_, &tty) != 0)
    {
        const std::string error = std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("tcgetattr failed for " + port + ": " + error);
    }
    ::cfmakeraw(&tty);
    const speed_t speed = baud_to_termios(baudrate);
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
        ::close(fd_);
    }
#endif
}

void SerialGlove::send(Command command, const std::vector<uint8_t>& payload)
{
#ifndef _WIN32
    const auto frame = pack_frame(command, payload);
    std::size_t sent = 0;
    while (sent < frame.size())
    {
        const ssize_t count = ::write(fd_, frame.data() + sent, frame.size() - sent);
        if (count < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            throw std::runtime_error("write failed for " + port_ + ": " + std::strerror(errno));
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
    append_glob("/dev/ttyUSB*", paths);
    append_glob("/dev/ttyACM*", paths);
    append_glob("/dev/ttyXRUSB*", paths);
    append_glob("/dev/ttyOBC*", paths);
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
#endif
    return paths;
}

std::optional<VersionInfo> probe_glove(SerialGlove& glove, int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline)
    {
        glove.send(Command::Version);
        for (const auto& frame : glove.read_available(40))
        {
            if (auto version = decode_version(frame))
            {
                return version;
            }
        }
    }
    return std::nullopt;
}

} // namespace plugins::realhand_ffg_glove
