// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unistd.h>

namespace core::camera
{

// Owns one file descriptor and closes it.
class UniqueFd
{
public:
    UniqueFd() = default;

    explicit UniqueFd(int fd) : m_fd(fd)
    {
    }

    UniqueFd(UniqueFd&& other) noexcept : m_fd(other.release())
    {
    }

    UniqueFd& operator=(UniqueFd&& other) noexcept
    {
        if (this != &other)
        {
            reset(other.release());
        }
        return *this;
    }

    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;

    ~UniqueFd()
    {
        reset();
    }

    int get() const
    {
        return m_fd;
    }

    bool valid() const
    {
        return m_fd >= 0;
    }

    // Gives the descriptor up without closing it.
    int release()
    {
        const int fd = m_fd;
        m_fd = -1;
        return fd;
    }

    void reset(int fd = -1)
    {
        if (m_fd >= 0)
        {
            ::close(m_fd);
        }
        m_fd = fd;
    }

private:
    int m_fd = -1;
};

} // namespace core::camera
