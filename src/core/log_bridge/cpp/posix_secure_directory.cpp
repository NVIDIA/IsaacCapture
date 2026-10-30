// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "posix_secure_directory.hpp"

#include <sys/stat.h>

#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <utility>
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

    FileDescriptor(FileDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1))
    {
    }

    FileDescriptor& operator=(FileDescriptor&& other) noexcept
    {
        if (this != &other)
        {
            if (fd_ >= 0)
            {
                ::close(fd_);
            }
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
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
    if (!S_ISDIR(info.st_mode) || (target && info.st_uid != ::getuid()))
    {
        return false;
    }
    const bool publicly_writable = (info.st_mode & (S_IWGRP | S_IWOTH)) != 0;
    return !publicly_writable || (!target && (info.st_mode & S_ISVTX) != 0);
}

std::vector<std::string> path_components(const std::filesystem::path& path)
{
    std::vector<std::string> components;
    for (const auto& component : path.relative_path())
    {
        if (component != ".")
        {
            components.push_back(component.string());
        }
    }
    return components;
}

bool open_or_create_child(FileDescriptor& parent, const std::string& name, bool target)
{
    struct ::stat entry
    {
    };
    if (::fstatat(parent.get(), name.c_str(), &entry, AT_SYMLINK_NOFOLLOW) != 0)
    {
        if (errno != ENOENT || (::mkdirat(parent.get(), name.c_str(), 0700) != 0 && errno != EEXIST))
        {
            return false;
        }
    }

    // Traverse relative to the vetted parent. O_NOFOLLOW rejects a symlink
    // even if the entry changed after fstatat.
    const int child = ::openat(parent.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (child < 0)
    {
        return false;
    }
    FileDescriptor opened(child);
    if (::fstat(opened.get(), &entry) != 0 || !has_secure_permissions(entry, target))
    {
        return false;
    }
    parent = std::move(opened);
    return true;
}

} // namespace

bool ensure_secure_directory(const std::filesystem::path& directory)
{
    std::error_code absolute_ec;
    const auto absolute = std::filesystem::absolute(directory, absolute_ec).lexically_normal();
    if (absolute_ec || !absolute.is_absolute())
    {
        return false;
    }

    const auto components = path_components(absolute);
    if (components.empty())
    {
        return false;
    }

    FileDescriptor current(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (current.get() < 0)
    {
        return false;
    }

    for (std::size_t index = 0; index < components.size(); ++index)
    {
        const bool target = index + 1 == components.size();
        if (components[index] == ".." || !open_or_create_child(current, components[index], target))
        {
            return false;
        }
    }
    return true;
}

} // namespace isaaccapture::detail
