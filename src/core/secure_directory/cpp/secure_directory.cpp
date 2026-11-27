// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/secure_directory/secure_directory.hpp"

#include <system_error>

#ifndef _WIN32
#    include <posix_utils/file_descriptor.hpp>
#    include <sys/stat.h>

#    include <cerrno>
#    include <fcntl.h>
#    include <unistd.h>
#    include <vector>
#endif

namespace isaaccapture
{
namespace
{
bool refuse(std::string* reason, const std::filesystem::path& path, const std::string& message)
{
    if (reason)
        *reason = path.string() + ": " + message;
    return false;
}

#ifndef _WIN32
const char* ancestor_failure(const struct ::stat& info)
{
    if (info.st_uid != 0 && info.st_uid != ::geteuid())
        return "foreign-owned directory";
    // Sticky shared ancestors protect their entries from other users.
    if ((info.st_mode & (S_IWGRP | S_IWOTH)) && !(info.st_mode & S_ISVTX))
        return "replaceable directory";
    return nullptr;
}

const char* target_failure(const struct ::stat& info)
{
    if (info.st_uid != ::geteuid())
        return "foreign-owned directory";
    if (info.st_mode & (S_IWGRP | S_IWOTH))
        return "replaceable directory";
    return nullptr;
}

using DirectoryPolicy = const char* (*)(const struct ::stat&);

bool validate(int fd, const std::filesystem::path& path, DirectoryPolicy policy, std::string* reason)
{
    struct ::stat info
    {
    };
    if (::fstat(fd, &info) != 0)
        return refuse(
            reason, path, "cannot inspect directory: " + std::error_code(errno, std::generic_category()).message());
    // O_DIRECTORY already guarantees the descriptor's type.
    if (const auto failure = policy(info))
        return refuse(reason, path, failure);
    return true;
}
#endif

bool check_directory(const std::filesystem::path& directory, bool create_missing, std::string* reason)
{
    if (reason)
        reason->clear();
#ifndef _WIN32
    std::error_code error;
    const auto absolute = std::filesystem::absolute(directory, error);
    if (error)
        return refuse(reason, directory, "cannot resolve absolute path: " + error.message());

    // Reject traversal before creating anything; normalization would hide symlink/..
    // before the no-follow check could see the symlink.
    std::vector<std::filesystem::path> components;
    for (const auto& component : absolute.relative_path())
    {
        if (component == "..")
            return refuse(reason, absolute, "parent traversal is not allowed");
        if (!component.empty() && component != ".")
            components.push_back(component);
    }
    if (components.empty())
    {
        return refuse(reason, absolute, "a private directory below the root is required");
    }

    // Search permission suffices for traversal; no ancestor read permission is needed.
#    ifdef O_PATH
    constexpr int access_mode = O_PATH;
#    elif defined(O_SEARCH)
    constexpr int access_mode = O_SEARCH;
#    else
    constexpr int access_mode = O_RDONLY;
#    endif
    constexpr int flags = access_mode | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    posix_utils::FileDescriptor current(::open("/", flags));
    std::filesystem::path component_path = "/";
    if (current.get() < 0)
        return refuse(reason, component_path,
                      "cannot open directory: " + std::error_code(errno, std::generic_category()).message());
    if (!validate(current.get(), component_path, ancestor_failure, reason))
        return false;

    for (std::size_t index = 0; index < components.size(); ++index)
    {
        const auto& name = components[index];
        component_path /= name;
        if (create_missing && ::mkdirat(current.get(), name.c_str(), 0700) != 0 && errno != EEXIST)
            return refuse(reason, component_path,
                          "cannot create directory: " + std::error_code(errno, std::generic_category()).message());
        // Resolve against the opened parent. Refuse an entry swapped for a symlink.
        const int child = ::openat(current.get(), name.c_str(), flags);
        if (child < 0)
            return refuse(reason, component_path,
                          "cannot open directory: " + std::error_code(errno, std::generic_category()).message());
        current.reset(child);
        const auto policy = index + 1 == components.size() ? target_failure : ancestor_failure;
        if (!validate(current.get(), component_path, policy, reason))
            return false;
    }
    return true;
#else
    std::error_code error;
    if (create_missing)
        std::filesystem::create_directories(directory, error);
    if (error)
        return refuse(reason, directory, "cannot create directory: " + error.message());
    const bool exists = std::filesystem::is_directory(directory, error);
    if (!exists || error)
        return refuse(reason, directory, error ? error.message() : "not a directory");
    return true;
#endif
}
} // namespace

bool ensure_secure_directory(const std::filesystem::path& directory, std::string* reason)
{
    return check_directory(directory, true, reason);
}

bool is_secure_directory(const std::filesystem::path& directory, std::string* reason)
{
    return check_directory(directory, false, reason);
}
} // namespace isaaccapture
