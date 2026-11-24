// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "posix_secure_directory.hpp"

#include <sys/stat.h>

#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>
#include <vector>

namespace isaaccapture::detail
{
namespace
{

class FileDescriptor
{
public:
    explicit FileDescriptor(int fd) : fd_(fd)
    {
    }

    ~FileDescriptor()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    void reset(int fd)
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
        fd_ = fd;
    }

    int get() const
    {
        return fd_;
    }

private:
    int fd_;
};

bool has_secure_permissions(const struct ::stat& info, bool target)
{
    if (!S_ISDIR(info.st_mode) || (info.st_uid != 0 && info.st_uid != ::getuid()) ||
        (target && info.st_uid != ::getuid()))
    {
        return false;
    }
    // Sticky shared ancestors protect their entries; the final log directory
    // must instead forbid other writers, even when its sticky bit is set.
    const bool publicly_writable = (info.st_mode & (S_IWGRP | S_IWOTH)) != 0;
    return !publicly_writable || (!target && (info.st_mode & S_ISVTX) != 0);
}

} // namespace

bool ensure_secure_directory(const std::filesystem::path& directory)
{
    std::error_code absolute_ec;
    const auto absolute = std::filesystem::absolute(directory, absolute_ec);
    if (absolute_ec || !absolute.is_absolute())
    {
        return false;
    }

    // Reject traversal before creating anything. Lexical normalization would
    // erase symlink/.. before the no-follow check could see the symlink.
    std::vector<std::filesystem::path> components;
    for (const auto& component : absolute.relative_path())
    {
        if (component == "..")
        {
            return false;
        }
        if (!component.empty() && component != ".")
        {
            components.push_back(component);
        }
    }
    if (components.empty())
    {
        return false;
    }

    // Traversal needs search permission, not read permission on each ancestor.
#ifdef O_PATH
    constexpr int access_mode = O_PATH;
#else
    constexpr int access_mode = O_RDONLY;
#endif
    constexpr int flags = access_mode | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    FileDescriptor current(::open("/", flags));
    if (current.get() < 0)
    {
        return false;
    }

    for (std::size_t index = 0; index < components.size(); ++index)
    {
        const auto& name = components[index];
        if (::mkdirat(current.get(), name.c_str(), 0700) != 0 && errno != EEXIST)
        {
            return false;
        }

        // Resolve only against the opened parent; O_NOFOLLOW also rejects an
        // entry swapped for a symlink between mkdirat and openat.
        const int child = ::openat(current.get(), name.c_str(), flags);
        if (child < 0)
        {
            return false;
        }
        current.reset(child);
        struct ::stat info
        {
        };
        if (::fstat(current.get(), &info) != 0 || !has_secure_permissions(info, index + 1 == components.size()))
        {
            return false;
        }
    }
    return true;
}

} // namespace isaaccapture::detail
