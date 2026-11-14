// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_ipc/endpoint_path.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace core::camera
{

std::string default_endpoint_path(std::string_view instance)
{
    const auto allowed = [](unsigned char c) { return std::isalnum(c) != 0 || c == '-' || c == '_'; };
    if (!std::all_of(instance.begin(), instance.end(), allowed))
    {
        throw std::invalid_argument("camera plugin instance '" + std::string(instance) +
                                    "' may only have letters, digits, '-' and '_' in its name");
    }
    const char* runtime_directory = std::getenv("XDG_RUNTIME_DIR");
    std::string path = runtime_directory != nullptr && runtime_directory[0] != '\0' ?
                           std::string(runtime_directory) + "/isaaccapture" :
                           "/tmp/isaaccapture-" + std::to_string(::geteuid());
    path += "/camera";
    if (!instance.empty())
    {
        path += "-" + std::string(instance);
    }
    return path + ".sock";
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
