// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace isaaccapture::timesync
{

struct CommandResult
{
    bool started = false; ///< The program was found and launched.
    bool timed_out = false; ///< It was killed for exceeding the timeout.
    bool truncated = false; ///< Output beyond the cap was discarded.
    int exit_code = -1; ///< Valid only when it ran to completion.
    std::string output; ///< Standard output; standard error is discarded.
};

/// Runs a program with an argument vector, never through a shell. Tests substitute fakes.
using CommandRunner =
    std::function<CommandResult(const std::vector<std::string>& argv, std::chrono::milliseconds timeout)>;

/// Output cap for the default runner.
inline constexpr size_t kMaxCommandOutputBytes = 64 * 1024;

/// posix_spawnp the program, read its stdout up to the cap, kill it when the timeout passes.
CommandRunner posix_command_runner();

} // namespace isaaccapture::timesync
