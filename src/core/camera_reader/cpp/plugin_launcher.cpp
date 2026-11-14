// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "plugin_launcher.hpp"

#include <sys/syscall.h>
#include <sys/wait.h>

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <sstream>
#include <system_error>
#include <unistd.h>

namespace core::camera
{

namespace
{

constexpr const char* PLUGIN_EXECUTABLE = "camera_plugin";

bool can_run(const std::filesystem::path& path)
{
    std::error_code ignored;
    return std::filesystem::is_regular_file(path, ignored) && ::access(path.c_str(), X_OK) == 0;
}

// Closes every descriptor from `first` up except `keep`. Runs between fork and exec, so it
// uses nothing but system calls.
void close_others(int first, int keep, long limit)
{
#ifdef SYS_close_range
    if (::syscall(SYS_close_range, static_cast<unsigned>(first), static_cast<unsigned>(keep - 1), 0U) == 0 &&
        ::syscall(SYS_close_range, static_cast<unsigned>(keep + 1), ~0U, 0U) == 0)
    {
        return;
    }
#endif
    for (int fd = first; fd < limit; ++fd)
    {
        if (fd != keep)
        {
            ::close(fd);
        }
    }
}

} // namespace

std::string find_plugin_executable(const std::string& preferred)
{
    if (!preferred.empty())
    {
        return can_run(preferred) ? preferred : std::string();
    }
    std::error_code ignored;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ignored);
    if (!self.empty() && can_run(self.parent_path() / PLUGIN_EXECUTABLE))
    {
        return (self.parent_path() / PLUGIN_EXECUTABLE).string();
    }
    const char* search = std::getenv("PATH");
    std::istringstream directories(search != nullptr ? search : "");
    for (std::string directory; std::getline(directories, directory, ':');)
    {
        if (!directory.empty() && can_run(std::filesystem::path(directory) / PLUGIN_EXECUTABLE))
        {
            return (std::filesystem::path(directory) / PLUGIN_EXECUTABLE).string();
        }
    }
    return {};
}

UniqueFd launch_plugin(const std::string& executable, const std::vector<std::string>& arguments)
{
    // Everything the new process needs is prepared here: after fork() it may only make system
    // calls, because other threads of this process may hold locks at that moment.
    std::vector<std::string> owned = { executable };
    owned.insert(owned.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (std::string& argument : owned)
    {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);
    const long descriptor_limit = ::sysconf(_SC_OPEN_MAX);

    // The plugin holds the write end for as long as it lives; nothing is ever written.
    int alive[2] = { -1, -1 };
    if (::pipe2(alive, O_CLOEXEC) != 0)
    {
        throw std::system_error(errno, std::generic_category(), "creating a pipe to the camera plugin");
    }
    UniqueFd reading(alive[0]);
    UniqueFd writing(alive[1]);

    const pid_t child = ::fork();
    if (child < 0)
    {
        throw std::system_error(errno, std::generic_category(), "starting the camera plugin");
    }
    if (child == 0)
    {
        // A session of its own, and a parent that exits at once: the plugin is then adopted by
        // the system, which collects it when it ends.
        ::setsid();
        if (::fork() != 0)
        {
            ::_exit(0);
        }
        const int null = ::open("/dev/null", O_RDWR);
        ::dup2(null, STDIN_FILENO);
        ::dup2(null, STDOUT_FILENO);
        ::dup2(null, STDERR_FILENO);
        ::fcntl(writing.get(), F_SETFD, 0); // stays open across exec
        close_others(STDERR_FILENO + 1, writing.get(), descriptor_limit);
        ::execv(argv[0], argv.data());
        ::_exit(127);
    }
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR)
    {
    }
    return reading;
}

} // namespace core::camera
