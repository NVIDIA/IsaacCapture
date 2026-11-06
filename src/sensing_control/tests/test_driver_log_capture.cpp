// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <sensing_control/driver_log_capture.hpp>

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

using plugins::sensing::DriverLogCapture;

namespace
{

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

std::size_t count_of(const std::string& text, const std::string& part)
{
    std::size_t n = 0;
    for (auto at = text.find(part); at != std::string::npos; at = text.find(part, at + 1))
        ++n;
    return n;
}

/// Points fd 2 at a scratch file for the lifetime of the object, so a test can see what the code under test replayed.
class StderrSink
{
public:
    StderrSink()
    {
        std::fflush(stderr);
        char path[] = "/tmp/driver-log-sink-XXXXXX";
        m_file = ::mkstemp(path);
        ::unlink(path);
        m_saved = ::dup(STDERR_FILENO);
        ::dup2(m_file, STDERR_FILENO);
    }

    ~StderrSink()
    {
        restore();
        ::close(m_saved);
        ::close(m_file);
    }

    std::string restore_and_read()
    {
        restore();
        std::string text;
        ::lseek(m_file, 0, SEEK_SET);
        char buffer[4096];
        ssize_t n = 0;
        while ((n = ::read(m_file, buffer, sizeof(buffer))) > 0)
            text.append(buffer, static_cast<std::size_t>(n));
        return text;
    }

private:
    void restore()
    {
        if (!m_restored)
        {
            std::fflush(stderr);
            ::dup2(m_saved, STDERR_FILENO);
            m_restored = true;
        }
    }

    int m_file = -1;
    int m_saved = -1;
    bool m_restored = false;
};

/// What fd `fd` points at (the /proc link text), so two descriptors can be compared.
std::string target_of(int fd)
{
    char target[256] = {};
    const ssize_t n = ::readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), target, sizeof(target) - 1);
    return n < 0 ? std::string() : std::string(target, static_cast<std::size_t>(n));
}

bool same_file(int fd, int reference)
{
    const auto a = target_of(fd);
    return !a.empty() && a == target_of(reference);
}

void raw_write(int fd, const std::string& text)
{
    REQUIRE(::write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size()));
}

} // namespace

TEST_CASE("output on fd 1 and fd 2 inside the scope is captured and replayed to stderr", "[driver_log_capture]")
{
    StderrSink sink;
    std::string interesting;
    {
        DriverLogCapture capture;
        raw_write(STDOUT_FILENO, "out: CAMERA_DETECT failed\n");
        raw_write(STDERR_FILENO, "err: lock timeout on link 1\n");
        raw_write(STDERR_FILENO, "harmless chatter\n");
        interesting = capture.finish();
    }
    const std::string replayed = sink.restore_and_read();

    CHECK(contains(interesting, "CAMERA_DETECT failed"));
    CHECK(contains(interesting, "lock timeout on link 1"));
    CHECK_FALSE(contains(interesting, "harmless chatter"));
    // Everything, interesting or not, reaches the journal.
    CHECK(contains(replayed, "out: CAMERA_DETECT failed"));
    CHECK(contains(replayed, "err: lock timeout on link 1"));
    CHECK(contains(replayed, "harmless chatter"));
}

TEST_CASE("the descriptors are restored after finish and a second finish is empty", "[driver_log_capture]")
{
    StderrSink sink;
    const int probe_out = ::dup(STDOUT_FILENO);
    const int probe_err = ::dup(STDERR_FILENO);
    {
        DriverLogCapture capture;
        CHECK_FALSE(same_file(STDOUT_FILENO, probe_out));
        capture.finish();
        CHECK(same_file(STDOUT_FILENO, probe_out));
        CHECK(same_file(STDERR_FILENO, probe_err));
        // Text written after finish() goes to the restored stderr, not into a capture.
        raw_write(STDERR_FILENO, "after finish\n");
        CHECK(capture.finish().empty());
    }
    const std::string replayed = sink.restore_and_read();
    CHECK(count_of(replayed, "after finish") == 1);
    ::close(probe_out);
    ::close(probe_err);
}

TEST_CASE("the descriptors are restored when the scope ends without finish", "[driver_log_capture]")
{
    StderrSink sink;
    const int probe_out = ::dup(STDOUT_FILENO);
    const int probe_err = ::dup(STDERR_FILENO);
    {
        DriverLogCapture capture;
        raw_write(STDERR_FILENO, "lost without finish: error\n");
        CHECK_FALSE(same_file(STDERR_FILENO, probe_err));
    }
    CHECK(same_file(STDOUT_FILENO, probe_out));
    CHECK(same_file(STDERR_FILENO, probe_err));
    raw_write(STDERR_FILENO, "after destruction\n");
    const std::string replayed = sink.restore_and_read();
    // The destructor replays what was captured, and later output is not swallowed.
    CHECK(contains(replayed, "lost without finish: error"));
    CHECK(count_of(replayed, "after destruction") == 1);
    ::close(probe_out);
    ::close(probe_err);
}

TEST_CASE("interesting_lines picks failure lines", "[driver_log_capture]")
{
    // Fewer matching lines than the six-line limit per call, so each word is seen.
    const std::vector<std::vector<std::string>> groups = {
        { "an ERROR happened", "operation failed", "Fail to open", "request timeout", "I2C NACK at 0x10" },
        { "OPEN_CIRCUIT", "CIRCUIT_OPEN_DISCONNECTION", "DES_LOCK_ERR", "CAMERA_DETECT" }
    };
    for (const auto& group : groups)
    {
        std::string text = "starting up\n";
        for (const auto& line : group)
            text += line + "\n";
        text += "all good here\n";
        const auto out = DriverLogCapture::interesting_lines(text);
        for (const auto& part : group)
        {
            INFO(part);
            CHECK(contains(out, part));
        }
        CHECK_FALSE(contains(out, "starting up"));
        CHECK_FALSE(contains(out, "all good here"));
    }
    CHECK(DriverLogCapture::interesting_lines("").empty());
}

TEST_CASE("interesting_lines strips the Severity prefix and joins with semicolons", "[driver_log_capture]")
{
    const std::string text =
        "[TID=1234] Module_id 30 Severity 2 : lock timeout on link 1\n"
        "[TID=1234] Module_id 30 Severity 2 : I2C NACK\n";
    CHECK(DriverLogCapture::interesting_lines(text) == "lock timeout on link 1; I2C NACK");

    // A line with " : " but no Severity keeps its text.
    CHECK(DriverLogCapture::interesting_lines("camera : error 5\n") == "camera : error 5");
}

TEST_CASE("interesting_lines collapses consecutive duplicates", "[driver_log_capture]")
{
    CHECK(DriverLogCapture::interesting_lines("error one\nerror one\nerror one\nerror two\n") == "error one; error two");
}

TEST_CASE("interesting_lines keeps the first six lines and at most 600 characters", "[driver_log_capture]")
{
    std::string text;
    for (int i = 1; i <= 9; ++i)
        text += "error line " + std::to_string(i) + "\n";
    const auto six = DriverLogCapture::interesting_lines(text);
    CHECK(contains(six, "error line 1"));
    CHECK(contains(six, "error line 6"));
    CHECK_FALSE(contains(six, "error line 7"));
    CHECK(count_of(six, "; ") == 5);

    const std::string long_line = "error " + std::string(1000, 'x') + "\n";
    CHECK(DriverLogCapture::interesting_lines(long_line).size() == 600);
}

TEST_CASE("interesting_lines does not match block or clock", "[driver_log_capture]")
{
    CHECK(DriverLogCapture::interesting_lines("block 3 ready\nclock running\nblocked on mutex\n").empty());
    // The substring rule is about "lock", not "block": "lock timeout" style words need the failure word.
    CHECK(DriverLogCapture::interesting_lines("pll lock acquired\n").empty());
}
