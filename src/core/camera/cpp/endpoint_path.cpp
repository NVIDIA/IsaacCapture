// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera/endpoint_path.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace core::camera
{

namespace
{

// An instance name becomes part of a file name, so it is kept to these characters.
bool is_instance_name(std::string_view name)
{
    const auto allowed = [](unsigned char c) { return std::isalnum(c) != 0 || c == '-' || c == '_'; };
    return !name.empty() && std::all_of(name.begin(), name.end(), allowed);
}

// Whether default_endpoint_path() gives this file name: "camera-<instance>.sock".
bool is_endpoint_name(std::string_view name)
{
    constexpr std::string_view prefix = "camera-";
    constexpr std::string_view suffix = ".sock";
    if (name.size() < prefix.size() + suffix.size() || !name.starts_with(prefix) || !name.ends_with(suffix))
    {
        return false;
    }
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return is_instance_name(name);
}

} // namespace

std::string default_endpoint_directory()
{
    const char* runtime_directory = std::getenv("XDG_RUNTIME_DIR");
    return runtime_directory != nullptr && runtime_directory[0] != '\0' ?
               std::string(runtime_directory) + "/isaaccapture" :
               "/tmp/isaaccapture-" + std::to_string(::geteuid());
}

std::string default_endpoint_path(std::string_view instance)
{
    if (!is_instance_name(instance))
    {
        throw std::invalid_argument("a camera plugin's instance name is one or more letters, digits, '-' and '_', not '" +
                                    std::string(instance) + "'");
    }
    return default_endpoint_directory() + "/camera-" + std::string(instance) + ".sock";
}

std::vector<std::string> default_endpoint_paths()
{
    const std::string directory = default_endpoint_directory();
    std::vector<std::string> paths;
    std::error_code error;
    for (std::filesystem::directory_iterator entry(directory, error), end; !error && entry != end; entry.increment(error))
    {
        // An entry that cannot be examined, for instance one that went away since it was read,
        // counts as not a socket.
        std::error_code gone;
        if (is_endpoint_name(entry->path().filename().native()) && std::filesystem::is_socket(entry->symlink_status(gone)))
        {
            paths.push_back(entry->path().native());
        }
    }
    // No directory means no plugin has run for this user, which is not an error.
    if (error && error != std::errc::no_such_file_or_directory)
    {
        throw std::system_error(error, "listing '" + directory + "'");
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

void prepare_endpoint_directory(const std::string& endpoint_path)
{
    const std::size_t slash = endpoint_path.rfind('/');
    if (slash == std::string::npos || slash == 0)
    {
        return; // the current directory or the root: nothing to prepare
    }
    const std::string directory = endpoint_path.substr(0, slash);
    if (::mkdir(directory.c_str(), S_IRWXU) != 0 && errno != EEXIST)
    {
        throw std::system_error(errno, std::generic_category(), "creating '" + directory + "'");
    }
    // A directory somebody else made in a shared place could be used to watch or replace the
    // socket file, so only one that is this user's alone will do.
    struct stat entry = {};
    if (::lstat(directory.c_str(), &entry) != 0)
    {
        throw std::system_error(errno, std::generic_category(), "inspecting '" + directory + "'");
    }
    if (!S_ISDIR(entry.st_mode) || entry.st_uid != ::geteuid() || (entry.st_mode & (S_IRWXG | S_IRWXO)) != 0)
    {
        throw std::system_error(
            EACCES, std::generic_category(), "'" + directory + "' is not a directory that belongs to this user alone");
    }
}

} // namespace core::camera
