// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The plugin binary's LiveKit options, run as an operator would run it. None of these needs a server:
// each one must be refused before any connection is tried.

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sys/wait.h>

#include <array>
#include <cstdio>
#include <fstream>

using namespace sensing_test;

namespace
{

struct Run
{
    int exit_code = -1;
    std::string output; // stdout and stderr together
};

Run run_plugin(const std::string& arguments)
{
    const std::string command = std::string("'") + SENSING_PLUGIN_BINARY + "' " + arguments + " 2>&1";
    Run run;
    if (FILE* pipe = ::popen(command.c_str(), "r"))
    {
        std::array<char, 4096> buffer{};
        while (size_t n = std::fread(buffer.data(), 1, buffer.size(), pipe))
            run.output.append(buffer.data(), n);
        const int status = ::pclose(pipe);
        run.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    return run;
}

void write_text(const std::string& path, const std::string& text)
{
    std::ofstream(path) << text;
}

bool mentions(const Run& run, const std::string& text)
{
    return run.output.find(text) != std::string::npos;
}

// Settings that are fine, so a test can break exactly one thing.
std::string good(const TempFile& token)
{
    return "--livekit-url=ws://127.0.0.1:9 --livekit-token-file=" + token.path() +
           " --add-stream=sensor=0,livekit=left --add-stream=sensor=1,livekit=right";
}

} // namespace

TEST_CASE("the help describes the LiveKit options", "[unit][sensing][livekit]")
{
    const Run run = run_plugin("--help");
    CHECK(run.exit_code == 0);
    for (const char* text :
         { "livekit=<track>", "--livekit-url", "--livekit-token-file", "--livekit-fps", "--livekit-scale-div" })
        CHECK(mentions(run, text));
}

TEST_CASE("livekit streams need a URL and a token file", "[unit][sensing][livekit]")
{
    const Run run = run_plugin("--add-stream=sensor=0,livekit=left");
    CHECK(run.exit_code == 1);
    CHECK(mentions(run, "--livekit-url and --livekit-token-file"));
}

TEST_CASE("LiveKit options without a livekit stream are refused", "[unit][sensing][livekit]")
{
    TempFile token(".token");
    write_text(token.path(), "x");
    const Run run = run_plugin("--livekit-url=ws://127.0.0.1:9 --livekit-token-file=" + token.path() +
                               " --add-stream=sensor=0,ipc=/tmp/nope.sock");
    CHECK(run.exit_code == 1);
    CHECK(mentions(run, "no stream has livekit="));
}

TEST_CASE("track names must be unique", "[unit][sensing][livekit]")
{
    TempFile token(".token");
    write_text(token.path(), "x");
    const Run run = run_plugin("--livekit-url=ws://127.0.0.1:9 --livekit-token-file=" + token.path() +
                               " --add-stream=sensor=0,livekit=same --add-stream=sensor=1,livekit=same");
    CHECK(run.exit_code == 1);
    CHECK(mentions(run, "used twice"));
}

TEST_CASE("a missing or empty token file is refused", "[unit][sensing][livekit]")
{
    const Run missing = run_plugin(
        "--livekit-url=ws://127.0.0.1:9 --livekit-token-file=/nonexistent/token --add-stream=sensor=0,livekit=left");
    CHECK(missing.exit_code == 1);
    CHECK(mentions(missing, "cannot read --livekit-token-file"));

    TempFile empty(".token");
    write_text(empty.path(), "\n");
    const Run blank = run_plugin(good(empty));
    CHECK(blank.exit_code == 1);
    CHECK(mentions(blank, "is empty"));
}

TEST_CASE("the publication rate must be positive", "[unit][sensing][livekit]")
{
    TempFile token(".token");
    write_text(token.path(), "x");
    const Run run = run_plugin(good(token) + " --livekit-fps=0");
    CHECK(run.exit_code == 1);
    CHECK(mentions(run, "--livekit-fps must be at least 1"));
}

TEST_CASE("a divisor that does not fit the sensor is refused before connecting", "[unit][gpu][sensing][livekit]")
{
    TempFile token(".token");
    write_text(token.path(), "x");
    // The vendor config's 2560x1984: 3 does not divide it, and 64 leaves an odd height.
    const Run three = run_plugin(good(token) + " --livekit-scale-div=3");
    CHECK(three.exit_code == 1);
    CHECK(mentions(three, "exactly"));
    const Run odd = run_plugin(good(token) + " --livekit-scale-div=64");
    CHECK(odd.exit_code == 1);
    CHECK(mentions(odd, "even"));
}

TEST_CASE("the join token is never printed", "[unit][gpu][sensing][livekit]")
{
    TempFile token(".token");
    const std::string secret = "eyJ-THIS-IS-A-SECRET-JOIN-TOKEN-0123456789";
    write_text(token.path(), secret + "\n");

    // Nothing listens on the URL, so this fails to connect: the error path must not echo the token.
    const Run run = run_plugin(good(token) + " --livekit-fps=15");
    CHECK(run.exit_code == 1);
    CHECK(mentions(run, "cannot join"));
    CHECK_FALSE(mentions(run, secret));
    CHECK_FALSE(mentions(run, "THIS-IS-A-SECRET"));
}
