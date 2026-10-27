// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef _WIN32
#    include <sys/types.h>
#endif

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace core
{

enum class ProcessState
{
    RUNNING,
    EXITED,
    SIGNALED,
    STOPPED,
    ERROR,
};

enum class ProcessReason
{
    NONE,
    CLEAN_EXIT,
    NONZERO_EXIT,
    SIGNAL,
    EXPLICIT_STOP,
    WAIT_ERROR,
    SIGNAL_ERROR,
};

struct ProcessSnapshot
{
    ProcessState state = ProcessState::RUNNING;
    ProcessReason reason = ProcessReason::NONE;
    std::int64_t pid = -1;
    std::optional<int> exit_code;
    std::optional<int> term_signal;
    std::optional<int> error_code;
    std::string error;
};

/**
 * @brief Custom exception thrown when a plugin crashes or exits unexpectedly
 */
class PluginCrashException : public std::runtime_error
{
public:
    explicit PluginCrashException(const std::string& message) : std::runtime_error(message)
    {
    }
};

class Plugin
{
public:
    /**
     * @brief Construct a new Plugin object (starts the plugin process)
     * @param command The command to run the plugin (from metadata)
     * @param working_dir The directory to run the plugin in (where metadata was found)
     * @param plugin_root_id The root ID for the plugin
     * @param plugin_args Optional list of arguments to append to the command
     */
    Plugin(const std::string& command,
           const std::string& working_dir,
           const std::string& plugin_root_id,
           const std::vector<std::string>& plugin_args = {});

    /**
     * @brief Destructor - stops the plugin process
     */
    ~Plugin();

    /**
     * @brief Explicitly stop the plugin (same as destructor)
     * @throws PluginCrashException if the plugin has crashed
     */
    void stop();

    /**
     * @brief Check if plugin crashed (lightweight, non-blocking)
     * @throws PluginCrashException if the plugin has crashed
     */
    void check_health();

    /**
     * @brief Poll and return the current process state without throwing process-health exceptions.
     *
     * Terminal state is cached, so repeated calls return the same exit, signal, or observation error.
     */
    ProcessSnapshot get_process_snapshot();

private:
    void start_process(const std::string& command,
                       const std::string& working_dir,
                       const std::string& plugin_root_id,
                       const std::vector<std::string>& plugin_args);

    void stop_process();
    void refresh_process_snapshot_locked(bool block);
    void cache_signal_error_locked(int error_code, const std::string& operation);

#ifndef _WIN32
    pid_t m_pid = -1;
#else
    int m_pid = -1;
#endif
    bool m_stop_requested = false;
    ProcessSnapshot m_process_snapshot;
    std::mutex m_process_mutex;
};

} // namespace core
