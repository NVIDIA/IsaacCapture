// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Scaffolding shared by the translation units of the mcap_tests binary. Every case here
// writes a recording to a temp file and reads it back; only what goes into the file differs.

#include <catch2/catch_test_macros.hpp>
#include <mcap/reader.hpp>
#include <mcap/recorded_schemas.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/head_generated.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace mcap_test
{

namespace fs = std::filesystem;

//! Process id, so two test binaries running at once cannot land on the same temp file.
inline int current_pid()
{
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}

//! A temp .mcap path unique to this process and call. `prefix` names the suite asking.
inline std::string temp_mcap_path(std::string_view prefix)
{
#ifndef _WIN32
    auto pattern = (fs::temp_directory_path() / (std::string(prefix) + "_XXXXXX")).string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (::mkdtemp(buffer.data()) == nullptr)
    {
        throw std::runtime_error("Failed to create private MCAP test directory");
    }
    // Keep the recording below an atomically created owner-only directory;
    // McapWriter may safely create and reopen its predictable child name.
    return (fs::path(buffer.data()) / "recording.mcap").string();
#else
    static std::atomic<int> counter{ 0 };
    const auto name =
        std::string(prefix) + "_" + std::to_string(current_pid()) + "_" + std::to_string(counter++) + ".mcap";
    return (fs::temp_directory_path() / name).string();
#endif
}

//! Removes the file when it goes out of scope, whether or not the case that wrote it passed.
class TempFileCleanup
{
public:
    explicit TempFileCleanup(std::string path) : path_(std::move(path))
    {
    }
    ~TempFileCleanup() noexcept
    {
        std::error_code ec;
#ifndef _WIN32
        fs::remove_all(fs::path(path_).parent_path(), ec);
#else
        fs::remove(path_, ec);
#endif
    }
    TempFileCleanup(const TempFileCleanup&) = delete;
    TempFileCleanup& operator=(const TempFileCleanup&) = delete;

private:
    std::string path_;
};

inline std::unique_ptr<mcap::McapReader> open_reader(const std::string& path)
{
    auto reader = std::make_unique<mcap::McapReader>();
    REQUIRE(reader->open(path).ok());
    return reader;
}

//! What `path` declares, read the way the replay factory reads it: from a reader of its own,
//! so the reader a viewer goes on to use is handed over in the state production hands it over in.
inline core::RecordedSchemas recorded_schemas(const std::string& path)
{
    const std::unique_ptr<mcap::McapReader> reader = open_reader(path);
    return core::RecordedSchemas(*reader);
}

//! The record type these tests write and read back.
using HeadChannels = core::McapTrackerChannels<core::HeadPoseRecord>;
using HeadViewers = core::McapTrackerViewers<core::HeadPoseRecord>;

} // namespace mcap_test
