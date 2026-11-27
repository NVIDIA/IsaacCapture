// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Scaffolding shared by the translation units of the mcap_tests binary. Every case here
// writes a recording to a temp file and reads it back; only what goes into the file differs.

#include <catch2/catch_test_macros.hpp>
#include <mcap/reader.hpp>
#include <mcap/recorded_schemas.hpp>
#include <mcap/tracker_channels.hpp>
#include <mcap/writer.hpp>
#include <schema/head_generated.h>
#include <test_support/temporary_directory.hpp>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace mcap_test
{

namespace fs = std::filesystem;

//! Use this owning fixture for recordings; keep it alive until readers and writers close.
class TemporaryMcapFile
{
public:
    const std::string& path() const
    {
        return path_;
    }

private:
    test_support::TemporaryDirectory directory_;
    std::string path_ = (directory_.path() / "recording.mcap").string();
};

inline std::unique_ptr<mcap::McapWriter> open_writer(const std::string& path)
{
    auto writer = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions options("teleop-test");
    options.compression = mcap::Compression::None;
    REQUIRE(writer->open(path, options).ok());
    return writer;
}

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
