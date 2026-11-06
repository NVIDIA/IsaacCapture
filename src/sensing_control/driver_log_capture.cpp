// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/driver_log_capture.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace plugins
{
namespace sensing
{

namespace
{

bool looks_like_failure(const std::string& line)
{
    static const char* const kWords[] = { "ERROR", "error",        "fail",         "Fail",         "timeout",
                                          "NACK",  "OPEN_CIRCUIT", "CIRCUIT_OPEN", "DES_LOCK_ERR", "CAMERA_DETECT" };
    return std::any_of(
        std::begin(kWords), std::end(kWords), [&](const char* word) { return line.find(word) != std::string::npos; });
}

/// Drop the driver's own prefix noise ("[TID=1234] Module_id 30 Severity 2 : ") so the useful words lead.
std::string trim_prefix(const std::string& line)
{
    const auto at = line.find(" : ");
    return (line.find("Severity") != std::string::npos && at != std::string::npos) ? line.substr(at + 3) : line;
}

} // namespace

DriverLogCapture::DriverLogCapture()
{
    std::fflush(stdout);
    std::fflush(stderr);
    char path[] = "/tmp/sensing-driver-log-XXXXXX";
    m_file = ::mkstemp(path);
    if (m_file < 0)
        return; // no capture: start-up still works, only the driver's words are lost
    ::unlink(path); // the file lives only as long as the descriptor
    m_saved_out = ::dup(STDOUT_FILENO);
    m_saved_err = ::dup(STDERR_FILENO);
    if (m_saved_out < 0 || m_saved_err < 0 || ::dup2(m_file, STDOUT_FILENO) < 0 || ::dup2(m_file, STDERR_FILENO) < 0)
    {
        if (m_saved_out >= 0)
            ::dup2(m_saved_out, STDOUT_FILENO);
        if (m_saved_err >= 0)
            ::dup2(m_saved_err, STDERR_FILENO);
        return;
    }
    m_active = true;
}

DriverLogCapture::~DriverLogCapture()
{
    finish();
    if (m_saved_out >= 0)
        ::close(m_saved_out);
    if (m_saved_err >= 0)
        ::close(m_saved_err);
    if (m_file >= 0)
        ::close(m_file);
}

std::string DriverLogCapture::finish()
{
    if (!m_active)
        return {};
    m_active = false;
    std::fflush(stdout);
    std::fflush(stderr);
    ::dup2(m_saved_out, STDOUT_FILENO);
    ::dup2(m_saved_err, STDERR_FILENO);

    std::string text;
    ::lseek(m_file, 0, SEEK_SET);
    char buffer[4096];
    ssize_t n = 0;
    while ((n = ::read(m_file, buffer, sizeof(buffer))) > 0)
        text.append(buffer, static_cast<std::size_t>(n));
    // The journal keeps the driver's lines exactly as if nothing had intervened.
    for (std::size_t done = 0; done < text.size();)
    {
        const ssize_t w = ::write(STDERR_FILENO, text.data() + done, text.size() - done);
        if (w <= 0)
            break;
        done += static_cast<std::size_t>(w);
    }
    return interesting_lines(text);
}

std::string DriverLogCapture::interesting_lines(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line))
    {
        if (looks_like_failure(line))
        {
            std::string cleaned = trim_prefix(line);
            if (lines.empty() || lines.back() != cleaned)
                lines.push_back(std::move(cleaned));
        }
    }
    // Keep the first lines: the cause comes first and the rest is the teardown that follows it.
    std::string joined;
    for (std::size_t i = 0; i < lines.size() && i < 6; ++i)
        joined += (joined.empty() ? "" : "; ") + lines[i];
    if (joined.size() > 600)
        joined.resize(600);
    return joined;
}

} // namespace sensing
} // namespace plugins
