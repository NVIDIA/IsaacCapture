// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PoC: Guman's push+pull tracker pair (GumanMcapRecordingPlan.md) actually records and reads
// back. Exercises McapTrackerChannels<GumanManusGloveFrameRecord> directly -- the same recording
// primitive the generated reader tracker's live impl uses -- rather than a live OpenXR session:
// OpenXRSession::wait_for_system defaults true with no bounded timeout (src/core/AGENTS.md), and
// this environment has no CloudXR runtime to satisfy it, so a true push-tracker-to-reader-tracker
// runtime test needs a real headset/CloudXR environment, not a plain build. This test instead
// proves the half of the mechanism that doesn't need one: the schema + MCAP recording path the
// plan's "push tracker alone has no recording path" finding is about.

// MCAP_IMPLEMENTATION is defined exactly once per link unit -- test_mcap_tracker_channels.cpp
// already does it for this executable; a second definition here is an ODR violation (confirmed:
// "multiple definition of `mcap::McapWriter::write...'" at link time).
#include "mcap_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/writer.hpp>
#include <schema/guman_manus_glove_generated.h>

#include <memory>
#include <string>
#include <vector>

namespace
{

using namespace mcap_test;

using GumanManusGloveChannels = core::McapTrackerChannels<core::GumanManusGloveFrameRecord>;

std::string get_temp_mcap_path()
{
    return mcap_test::temp_mcap_path("test_guman_manus_glove");
}

std::unique_ptr<mcap::McapWriter> open_writer(const std::string& path)
{
    auto writer = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions options("teleop-test");
    options.compression = mcap::Compression::None;
    auto status = writer->open(path, options);
    REQUIRE(status.ok());
    return writer;
}

std::vector<float> flat_pose_25x4x4(float fill)
{
    return std::vector<float>(25 * 4 * 4, fill);
}

} // namespace

TEST_CASE("GumanManusGloveFrame: pushed-shaped record writes to MCAP and reads back exactly",
          "[mcap][guman][tracker_channels]")
{
    auto path = get_temp_mcap_path();
    TempFileCleanup cleanup(path);

    // The payload a push tracker's .push() would encode from Guman's native ManusFrame.
    core::GumanManusGloveFrameT frame_data;
    frame_data.left_hand_pose = flat_pose_25x4x4(1.5f);
    frame_data.right_hand_pose = flat_pose_25x4x4(-2.5f);

    {
        auto writer = open_writer(path);
        // What the generated reader tracker's live impl does on every sample: write to the
        // per-sample channel, then the last one again to the "tracked" channel.
        GumanManusGloveChannels ch(*writer, "guman_manus_glove", { "guman_manus_glove", "guman_manus_glove_tracked" });
        const auto record =
            core::pack_record<core::GumanManusGloveFrameRecord>(&frame_data, core::DeviceDataTimestamp(100, 100, 1));
        ch.write(0, record);
        ch.write(1, record);
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    size_t msg_count = 0;
    for (const auto& view : reader.readMessages())
    {
        CHECK(view.schema->name == core::GumanManusGloveFrameRecord::GetFullyQualifiedName());

        auto record = flatbuffers::GetRoot<core::GumanManusGloveFrameRecord>(view.message.data);
        REQUIRE(record != nullptr);
        REQUIRE(record->data() != nullptr);

        const auto* left = record->data()->left_hand_pose();
        const auto* right = record->data()->right_hand_pose();
        REQUIRE(left != nullptr);
        REQUIRE(right != nullptr);
        CHECK(left->size() == 25 * 4 * 4);
        CHECK(right->size() == 25 * 4 * 4);
        CHECK((*left)[0] == 1.5f);
        CHECK((*right)[0] == -2.5f);

        REQUIRE(record->timestamp() != nullptr);
        CHECK(record->timestamp()->sample_time_raw_device_clock() == 1);

        msg_count++;
    }
    reader.close();

    // One write to each of the two channels (per-sample + tracked), matching ch.write() above.
    CHECK(msg_count == 2);
}
