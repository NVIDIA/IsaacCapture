// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/timesync/command_runner.hpp"

#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <unistd.h>

extern char** environ;

namespace isaaccapture::timesync
{

namespace
{

struct Fd
{
    int fd = -1;
    ~Fd()
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }
};

} // namespace

CommandRunner posix_command_runner()
{
    return [](const std::vector<std::string>& argv, std::chrono::milliseconds timeout)
    {
        CommandResult result;
        if (argv.empty())
        {
            return result;
        }

        int fds[2];
        if (pipe(fds) != 0)
        {
            return result;
        }
        Fd read_end{ fds[0] };
        Fd write_end{ fds[1] };

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, write_end.fd, STDOUT_FILENO);
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        // Descriptors 0-2 were set up above; closing a pipe end that landed on one of them (the parent started with
        // it closed) would undo that. Only higher descriptors need closing in the child.
        if (read_end.fd > STDERR_FILENO)
        {
            posix_spawn_file_actions_addclose(&actions, read_end.fd);
        }
        if (write_end.fd > STDERR_FILENO)
        {
            posix_spawn_file_actions_addclose(&actions, write_end.fd);
        }

        std::vector<char*> args;
        for (const auto& a : argv)
        {
            args.push_back(const_cast<char*>(a.c_str()));
        }
        args.push_back(nullptr);

        pid_t pid = 0;
        const int rc = posix_spawnp(&pid, args[0], &actions, nullptr, args.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        if (rc != 0)
        {
            return result;
        }
        result.started = true;
        close(write_end.fd);
        write_end.fd = -1;

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        char buffer[4096];
        for (;;)
        {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (left.count() <= 0)
            {
                result.timed_out = true;
                break;
            }
            pollfd p{ read_end.fd, POLLIN, 0 };
            const int pr = poll(&p, 1, static_cast<int>(left.count()));
            if (pr < 0 && errno == EINTR)
            {
                continue;
            }
            if (pr == 0)
            {
                result.timed_out = true;
                break;
            }
            if (pr < 0)
            {
                break;
            }
            const ssize_t n = read(read_end.fd, buffer, sizeof(buffer));
            if (n < 0 && errno == EINTR)
            {
                continue;
            }
            if (n <= 0)
            {
                break; // end of output
            }
            const size_t room = kMaxCommandOutputBytes - result.output.size();
            result.output.append(buffer, std::min(static_cast<size_t>(n), room));
            if (static_cast<size_t>(n) > room)
            {
                result.truncated = true;
                break;
            }
        }

        int status = 0;
        bool reaped = false;
        bool wait_failed = false;
        if (!result.timed_out && !result.truncated)
        {
            // The output ended, which does not mean the program did: one that closed stdout and hangs must still be
            // bounded by the same deadline, so reap it without blocking until the deadline passes.
            for (;;)
            {
                const pid_t waited = waitpid(pid, &status, WNOHANG);
                if (waited == pid)
                {
                    reaped = true;
                    break;
                }
                if (waited < 0 && errno == EINTR)
                {
                    continue;
                }
                if (waited < 0)
                {
                    wait_failed = true;
                    break;
                }
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    result.timed_out = true;
                    break;
                }
                poll(nullptr, 0, 5);
            }
        }
        if (!reaped && !wait_failed)
        {
            kill(pid, SIGKILL);
            for (;;)
            {
                const pid_t waited = waitpid(pid, &status, 0);
                if (waited == pid)
                {
                    reaped = true;
                    break;
                }
                if (waited < 0 && errno == EINTR)
                {
                    continue;
                }
                break; // not reapable: the exit status is unknown
            }
        }
        // The exit code is read only from a status that waitpid actually returned: a failed wait leaves it at -1
        // instead of reporting a zero-initialised status as success.
        if (reaped && !result.timed_out && !result.truncated && WIFEXITED(status))
        {
            result.exit_code = WEXITSTATUS(status);
        }
        return result;
    };
}

} // namespace isaaccapture::timesync
