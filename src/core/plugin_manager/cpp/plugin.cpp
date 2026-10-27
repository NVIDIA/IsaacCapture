// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_manager/plugin.hpp"

#include <log_bridge/logger.hpp>

#ifndef _WIN32
#    include <sys/stat.h>
#    include <sys/wait.h>

#    include <fcntl.h>
#    include <signal.h>
#    include <string.h>
#    include <unistd.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace core
{
#ifndef _WIN32
namespace
{

enum class ChildLaunchStage : std::uint8_t
{
    ChangeDirectory,
    Execute,
};

struct ChildLaunchError
{
    ChildLaunchStage stage;
    int error_number;
};

int move_above_std_cloexec(int fd)
{
    if (fd < 0)
    {
        return -1;
    }
    if (fd <= STDERR_FILENO)
    {
#    ifdef F_DUPFD_CLOEXEC
        const int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
#    else
        const int moved = ::fcntl(fd, F_DUPFD, STDERR_FILENO + 1);
        if (moved >= 0)
        {
            const int flags = ::fcntl(moved, F_GETFD);
            if (flags < 0 || ::fcntl(moved, F_SETFD, flags | FD_CLOEXEC) < 0)
            {
                const int error_number = errno;
                ::close(moved);
                ::close(fd);
                errno = error_number;
                return -1;
            }
        }
#    endif
        const int error_number = errno;
        ::close(fd);
        errno = error_number;
        return moved;
    }

    const int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
    {
        const int error_number = errno;
        ::close(fd);
        errno = error_number;
        return -1;
    }
    return fd;
}

// The child may only use async-signal-safe operations before execvp().
void report_child_launch_error(int fd, ChildLaunchStage stage, int error_number)
{
    const ChildLaunchError error{ stage, error_number };
    const auto* data = reinterpret_cast<const char*>(&error);
    std::size_t written = 0;
    while (written < sizeof(error))
    {
        const ssize_t result = ::write(fd, data + written, sizeof(error) - written);
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result <= 0)
        {
            return;
        }
        written += static_cast<std::size_t>(result);
    }
}

std::optional<ChildLaunchError> read_child_launch_error(int fd)
{
    ChildLaunchError error{};
    auto* data = reinterpret_cast<char*>(&error);
    std::size_t received = 0;
    while (received < sizeof(error))
    {
        const ssize_t result = ::read(fd, data + received, sizeof(error) - received);
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result < 0)
        {
            throw std::runtime_error("Failed to read plugin launch status: " + std::string(std::strerror(errno)));
        }
        if (result == 0)
        {
            if (received == 0)
            {
                return std::nullopt; // execvp() closed the write end.
            }
            throw std::runtime_error("Plugin launch status was truncated");
        }
        received += static_cast<std::size_t>(result);
    }
    return error;
}

void reap_child(pid_t pid)
{
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR)
    {
    }
}

std::string child_launch_error_message(const ChildLaunchError& error,
                                       const std::string& command,
                                       const std::string& working_dir)
{
    std::string message = "Plugin process exited immediately: failed to ";
    if (error.stage == ChildLaunchStage::ChangeDirectory)
    {
        message += "change directory to '" + working_dir + "'";
    }
    else
    {
        message += "execute plugin command '" + command + "'";
    }
    return message + ": " + std::strerror(error.error_number);
}

} // namespace
#endif

Plugin::Plugin(const std::string& command,
               const std::string& working_dir,
               const std::string& plugin_root_id,
               const std::vector<std::string>& plugin_args)
{
    start_process(command, working_dir, plugin_root_id, plugin_args);
}

Plugin::~Plugin()
{
    try
    {
        stop_process();
    }
    catch (...)
    {
    }
}

void Plugin::stop()
{
    check_health();
    stop_process();
}

void Plugin::check_health()
{
    const ProcessSnapshot snapshot = get_process_snapshot();
    switch (snapshot.state)
    {
    case ProcessState::EXITED:
        if (snapshot.exit_code.value_or(0) != 0)
        {
            throw PluginCrashException(snapshot.error);
        }
        break;
    case ProcessState::SIGNALED:
    case ProcessState::ERROR:
        throw PluginCrashException(snapshot.error);
    case ProcessState::RUNNING:
    case ProcessState::STOPPED:
        break;
    }
}

ProcessSnapshot Plugin::get_process_snapshot()
{
    std::lock_guard<std::mutex> lock(m_process_mutex);
#ifndef _WIN32
    if (m_pid != -1)
    {
        refresh_process_snapshot_locked(false);
    }
#endif
    return m_process_snapshot;
}

void Plugin::start_process(const std::string& command,
                           const std::string& working_dir,
                           const std::string& plugin_root_id,
                           const std::vector<std::string>& plugin_args)
{
#ifndef _WIN32
    std::vector<std::string> args_str;
    std::stringstream ss(command);
    std::string item;
    while (std::getline(ss, item, ' '))
    {
        if (!item.empty())
        {
            args_str.push_back(item);
        }
    }
    if (args_str.empty())
    {
        throw std::runtime_error("Empty plugin command");
    }
    if (!plugin_root_id.empty())
    {
        args_str.push_back("--plugin-root-id=" + plugin_root_id);
    }
    for (const auto& arg : plugin_args)
    {
        if (!arg.starts_with("--plugin-root-id="))
        {
            args_str.push_back(arg);
        }
        else
        {
            isaaccapture::Logger::get("isaaccapture.core.Plugin")
                ->warn("--plugin-root-id is managed by the plugin launcher, ignoring manual override");
        }
    }

    std::vector<char*> args;
    args.reserve(args_str.size() + 1);
    for (auto& arg : args_str)
    {
        args.push_back(arg.data());
    }
    args.push_back(nullptr);

    char* const executable = args.front();
    char* const* const argv = args.data();
    const char* const working_dir_path = working_dir.empty() ? nullptr : working_dir.c_str();

    // Read before fork(): getenv() is not async-signal-safe, and the child needs
    // the path as a plain pointer it can hand straight to open(). Published by
    // isaaccapture.logging_config (_native_fd.CAPTURE_FILE_ENV); absent only when
    // no file could be opened or this process never imported the Python half.
    // Capture mode "off" still publishes it because it governs the host's
    // descriptors, not those of a process the host launches. With logging off
    // an inherited path is ignored, so the child keeps the parent's descriptors.
    const char* const native_capture_env =
        isaaccapture::logging_enabled() ? std::getenv("ISAACCAPTURE_NATIVE_CAPTURE_FILE") : nullptr;
    const std::string native_capture_path_text = native_capture_env == nullptr ? "" : native_capture_env;
    const char* const native_capture_path = native_capture_path_text.empty() ? nullptr : native_capture_path_text.c_str();

    // CLOEXEC turns pipe EOF into the successful-exec signal.
    int raw_launch_pipe[2];
    if (::pipe(raw_launch_pipe) != 0)
    {
        throw std::runtime_error("Failed to create plugin launch pipe: " + std::string(std::strerror(errno)));
    }
    const int launch_read_fd = move_above_std_cloexec(raw_launch_pipe[0]);
    if (launch_read_fd < 0)
    {
        const int pipe_error = errno;
        ::close(raw_launch_pipe[1]);
        throw std::runtime_error("Failed to prepare plugin launch pipe: " + std::string(std::strerror(pipe_error)));
    }
    const int launch_write_fd = move_above_std_cloexec(raw_launch_pipe[1]);
    if (launch_write_fd < 0)
    {
        const int pipe_error = errno;
        ::close(launch_read_fd);
        throw std::runtime_error("Failed to prepare plugin launch pipe: " + std::string(std::strerror(pipe_error)));
    }

    const pid_t child_pid = fork();
    if (child_pid == -1)
    {
        const int fork_error = errno;
        ::close(launch_read_fd);
        ::close(launch_write_fd);
        throw std::runtime_error("Failed to fork process for plugin: " + std::string(std::strerror(fork_error)));
    }

    if (child_pid == 0)
    {
        ::close(launch_read_fd);

        // Child process, between fork() and execvp(): only async-signal-safe calls
        // are allowed here (POSIX). Never add Logger/spdlog calls in this window --
        // spdlog's registry and sinks are unsafe post-fork-pre-exec. execvp() is
        // the one deliberate exception:
        // POSIX leaves it off the list because its PATH search may allocate, and
        // it is kept because a plugin's command may be a bare name. Removing the
        // exception means resolving the executable before fork() and calling
        // execv().

        // Point the *child's* output at the session's capture file -- the child's
        // descriptors are ours to set, the host's are not, and this is what keeps
        // a plugin's non-logger output off the terminal. open(), fstat(), dup2()
        // and close() are all async-signal-safe. No O_CREAT: the Python leader
        // created this file, so a path that has gone missing is not ours to
        // recreate. O_APPEND, so several plugins and the parent can share it. A
        // failure leaves the inherited descriptors in place and does not stop
        // the plugin from starting.
        //
        // This opens a name it did not create, so O_NOFOLLOW is not enough on
        // its own: it refuses a symlink but not a plain file substituted at the
        // same path, which would collect this plugin's whole output. The leader
        // created the file 0600, so confirm that is what we got.
        if (native_capture_path != nullptr && native_capture_path[0] != '\0')
        {
            const int capture_fd = ::open(native_capture_path, O_WRONLY | O_APPEND | O_NOFOLLOW);
            if (capture_fd >= 0)
            {
                struct ::stat capture_info
                {
                };
                const bool ours = ::fstat(capture_fd, &capture_info) == 0 && S_ISREG(capture_info.st_mode) &&
                                  capture_info.st_uid == ::getuid() && (capture_info.st_mode & (S_IRWXG | S_IRWXO)) == 0;
                if (ours)
                {
                    ::dup2(capture_fd, STDOUT_FILENO);
                    ::dup2(capture_fd, STDERR_FILENO);
                }
                if (!ours || (capture_fd != STDOUT_FILENO && capture_fd != STDERR_FILENO))
                {
                    ::close(capture_fd);
                }
            }
        }

        // Change working directory
        if (working_dir_path != nullptr)
        {
            if (chdir(working_dir_path) != 0)
            {
                const int chdir_error = errno;
                report_child_launch_error(launch_write_fd, ChildLaunchStage::ChangeDirectory, chdir_error);
                _exit(1);
            }
        }

        // Close file descriptors to avoid sharing with parent process
        for (int i = 3; i < 1024; ++i)
        {
            if (i != launch_write_fd)
            {
                close(i);
            }
        }

        execvp(executable, argv);

        const int exec_error = errno;
        report_child_launch_error(launch_write_fd, ChildLaunchStage::Execute, exec_error);
        _exit(1);
    }
    else
    {
        ::close(launch_write_fd);
        std::optional<ChildLaunchError> launch_error;
        try
        {
            launch_error = read_child_launch_error(launch_read_fd);
            ::close(launch_read_fd);
        }
        catch (...)
        {
            ::close(launch_read_fd);
            ::kill(child_pid, SIGKILL);
            reap_child(child_pid);
            throw;
        }
        if (launch_error.has_value())
        {
            reap_child(child_pid);
            const std::string message = child_launch_error_message(*launch_error, command, working_dir);
            isaaccapture::Logger::get("isaaccapture.core.Plugin")->error("{}", message);
            throw std::runtime_error(message);
        }

        {
            std::lock_guard<std::mutex> lock(m_process_mutex);
            m_pid = child_pid;
            m_stop_requested = false;
            m_process_snapshot = ProcessSnapshot{};
            m_process_snapshot.pid = child_pid;
        }

        // Parent process - give the plugin a moment to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check if process died during startup
        ProcessState startup_state;
        std::string startup_error;
        {
            std::lock_guard<std::mutex> lock(m_process_mutex);
            refresh_process_snapshot_locked(false);
            startup_state = m_process_snapshot.state;
            startup_error = m_process_snapshot.error;
        }
        if (startup_state == ProcessState::ERROR)
        {
            throw std::runtime_error("Failed to observe plugin process during startup: " + startup_error);
        }
        if (startup_state == ProcessState::EXITED || startup_state == ProcessState::SIGNALED)
        {
            // Keep the leading phrase whatever else is known. It is the only part
            // that says the exit happened inside the startup window -- startup_error
            // says how the process exited, not when -- and .github/workflows/
            // build-ubuntu.yml matches it verbatim to fail the live CloudXR job
            // fast instead of waiting out its bring-up timeout.
            std::string message = "Plugin process exited immediately";
            if (!startup_error.empty())
            {
                message += ": " + startup_error;
            }
            if (!native_capture_path_text.empty())
            {
                message += "; see native output capture at " + native_capture_path_text;
            }
            throw std::runtime_error(message);
        }
    }
#else
    throw std::runtime_error("Plugin process management not supported on Windows");
#endif
}

void Plugin::stop_process()
{
#ifndef _WIN32
    std::unique_lock<std::mutex> lock(m_process_mutex);
    if (m_pid == -1)
    {
        return;
    }

    refresh_process_snapshot_locked(false);
    if (m_pid == -1)
    {
        return;
    }

    m_stop_requested = true;
    if (kill(m_pid, SIGINT) == -1)
    {
        cache_signal_error_locked(errno, "send SIGINT to plugin process");
        throw PluginCrashException(m_process_snapshot.error);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (m_pid != -1 && std::chrono::steady_clock::now() < deadline)
    {
        refresh_process_snapshot_locked(false);
        if (m_pid != -1)
        {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            lock.lock();
        }
    }

    refresh_process_snapshot_locked(false);
    if (m_pid == -1)
    {
        return;
    }

    if (kill(m_pid, SIGKILL) == -1)
    {
        cache_signal_error_locked(errno, "send SIGKILL to plugin process");
        throw PluginCrashException(m_process_snapshot.error);
    }
    refresh_process_snapshot_locked(true);
#endif
}

void Plugin::refresh_process_snapshot_locked(bool block)
{
#ifndef _WIN32
    if (m_pid == -1)
    {
        return;
    }

    int status = 0;
    pid_t result;
    do
    {
        result = waitpid(m_pid, &status, block ? 0 : WNOHANG);
    } while (result == -1 && errno == EINTR);

    if (result == 0)
    {
        return;
    }

    if (result == -1)
    {
        const int wait_error = errno;
        m_pid = -1;
        m_process_snapshot.state = ProcessState::ERROR;
        m_process_snapshot.reason = ProcessReason::WAIT_ERROR;
        m_process_snapshot.error_code = wait_error;
        m_process_snapshot.error = "Failed to check plugin health: " + std::string(std::strerror(wait_error));
        return;
    }

    // waitpid returned the child, so the PID must no longer be used for signaling.
    m_pid = -1;
    m_process_snapshot.exit_code.reset();
    m_process_snapshot.term_signal.reset();
    m_process_snapshot.error_code.reset();
    m_process_snapshot.error.clear();

    if (WIFEXITED(status))
    {
        m_process_snapshot.exit_code = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status))
    {
        m_process_snapshot.term_signal = WTERMSIG(status);
    }

    if (m_stop_requested)
    {
        m_process_snapshot.state = ProcessState::STOPPED;
        m_process_snapshot.reason = ProcessReason::EXPLICIT_STOP;
        m_process_snapshot.exit_code.reset();
        m_process_snapshot.term_signal.reset();
        return;
    }

    if (m_process_snapshot.exit_code.has_value())
    {
        m_process_snapshot.state = ProcessState::EXITED;
        if (*m_process_snapshot.exit_code == 0)
        {
            m_process_snapshot.reason = ProcessReason::CLEAN_EXIT;
        }
        else
        {
            m_process_snapshot.reason = ProcessReason::NONZERO_EXIT;
            m_process_snapshot.error =
                "Plugin process unexpectedly exited with code " + std::to_string(*m_process_snapshot.exit_code);
        }
        return;
    }

    if (m_process_snapshot.term_signal.has_value())
    {
        const int term_signal = *m_process_snapshot.term_signal;
        const char* signal_name = strsignal(term_signal);
        m_process_snapshot.state = ProcessState::SIGNALED;
        m_process_snapshot.reason = ProcessReason::SIGNAL;
        m_process_snapshot.error = "Plugin process crashed with signal " + std::to_string(term_signal);
        if (signal_name != nullptr)
        {
            m_process_snapshot.error += " (" + std::string(signal_name) + ")";
        }
        return;
    }

    m_process_snapshot.state = ProcessState::ERROR;
    m_process_snapshot.reason = ProcessReason::WAIT_ERROR;
    m_process_snapshot.error = "Plugin process ended with an unrecognized wait status";
#else
    (void)block;
#endif
}

void Plugin::cache_signal_error_locked(int error_code, const std::string& operation)
{
    m_process_snapshot.state = ProcessState::ERROR;
    m_process_snapshot.reason = ProcessReason::SIGNAL_ERROR;
    m_process_snapshot.error_code = error_code;
    m_process_snapshot.error = "Failed to " + operation + ": " + std::string(std::strerror(error_code));
}

} // namespace core
