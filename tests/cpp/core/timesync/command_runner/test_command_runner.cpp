// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <timesync/command_runner.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace isaaccapture::timesync;
using std::chrono::milliseconds;

static int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                              \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

CommandResult run(const std::vector<std::string>& argv, milliseconds timeout = milliseconds(5000))
{
    return posix_command_runner()(argv, timeout);
}

void test_echo()
{
    const auto r = run({ "/bin/echo", "hello" });
    CHECK(r.started);
    CHECK(!r.timed_out);
    CHECK(!r.truncated);
    CHECK(r.exit_code == 0);
    CHECK(r.output == "hello\n");
}

void test_no_shell()
{
    const auto r = run({ "/bin/echo", "$HOME; echo x" });
    CHECK(r.started);
    CHECK(r.output == "$HOME; echo x\n");
}

void test_missing_program()
{
    const auto r = run({ "/nonexistent/isaaccapture-no-such-program" });
    CHECK(!r.started);
    CHECK(r.exit_code == -1);
}

void test_exit_code()
{
    const auto r = run({ "/bin/false" });
    CHECK(r.started);
    CHECK(r.exit_code == 1);
    CHECK(!r.timed_out);
}

void test_timeout()
{
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = run({ "/bin/sleep", "5" }, milliseconds(200));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    CHECK(r.started);
    CHECK(r.timed_out);
    CHECK(r.exit_code == -1);
    CHECK(elapsed < std::chrono::seconds(2));
}

void test_output_cap()
{
    // Shell pipeline is a test fixture only; the runner under test does not use a shell.
    const auto r = run({ "/bin/sh", "-c", "yes | head -c 200000" });
    CHECK(r.started);
    CHECK(r.truncated);
    CHECK(r.output.size() == kMaxCommandOutputBytes);
}

void test_empty_argv()
{
    const auto r = run({});
    CHECK(!r.started);
}

void test_stderr_discarded()
{
    const auto r = run({ "/bin/sh", "-c", "echo err 1>&2; echo out" });
    CHECK(r.started);
    CHECK(r.exit_code == 0);
    CHECK(r.output == "out\n");
}

} // namespace

int main()
{
    test_echo();
    test_no_shell();
    test_missing_program();
    test_exit_code();
    test_timeout();
    test_output_cap();
    test_empty_argv();
    test_stderr_discarded();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("timesync_command_runner: all checks passed\n");
    return 0;
}
