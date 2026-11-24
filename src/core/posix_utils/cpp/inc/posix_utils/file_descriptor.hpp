// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unistd.h>

namespace isaaccapture::posix_utils
{
class FileDescriptor
{
public:
    explicit FileDescriptor(int fd = -1) : fd_(fd)
    {
    }
    ~FileDescriptor()
    {
        reset();
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int get() const
    {
        return fd_;
    }
    void reset(int fd = -1)
    {
        if (fd_ == fd)
            return;
        // Retrying close after EINTR can close a descriptor reused by another thread.
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = fd;
    }

private:
    int fd_;
};
} // namespace isaaccapture::posix_utils
