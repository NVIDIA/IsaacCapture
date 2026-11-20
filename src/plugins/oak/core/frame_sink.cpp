// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define MCAP_IMPLEMENTATION
#include "frame_sink.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <mcap/writer.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/oak_bfbs_generated.h>

#include <filesystem>
#include <memory>
#include <stdexcept>

namespace plugins
{
namespace oak
{

// =============================================================================
// FrameSink
// =============================================================================

FrameSink::FrameSink(const std::vector<StreamConfig>& streams, std::unique_ptr<IMetadataPusher> metadata_pusher)
    : m_metadata_pusher(std::move(metadata_pusher))
{
    for (const auto& config : streams)
    {
        std::filesystem::path p(config.output_path);
        auto parent = p.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        m_writers[config.camera] = std::make_unique<RawDataWriter>(config.output_path);
        m_logger->info("Add stream:  {} -> {}", core::EnumNameStreamType(config.camera), config.output_path);
    }
}

void FrameSink::on_frame(const OakFrame& frame)
{
    auto it = m_writers.find(frame.stream);
    if (it != m_writers.end())
        it->second->write(frame.h264_data);

    if (m_metadata_pusher)
        m_metadata_pusher->on_frame_metadata(
            frame.metadata, frame.sample_time_local_common_clock_ns, frame.sample_time_raw_device_clock_ns);
}

// =============================================================================
// SchemaMetadataPusher — pushes frame metadata over OpenXR
// =============================================================================

class SchemaMetadataPusher : public IMetadataPusher
{
public:
    SchemaMetadataPusher(const std::vector<StreamConfig>& streams, const std::string& collection_prefix)
        : m_oxr_session(
              std::make_shared<core::OpenXRSession>("OakCameraPlugin", core::SchemaPusher::get_required_extensions()))
    {
        for (const auto& config : streams)
        {
            auto collection_id = collection_prefix + "/" + core::EnumNameStreamType(config.camera);
            m_pushers[config.camera] = std::make_unique<core::SchemaPusher>(
                m_oxr_session->get_handles(), core::SchemaPusherConfig{ .collection_id = collection_id,
                                                                        .max_flatbuffer_size = MAX_FLATBUFFER_SIZE,
                                                                        .tensor_identifier = "frame_metadata",
                                                                        .localized_name = "Frame Metadata Pusher",
                                                                        .app_name = "OakCameraPlugin" });
            m_logger->info("  Metadata:  {}", collection_id);
        }
    }

    void on_frame_metadata(const core::FrameMetadataOakT& metadata,
                           int64_t sample_time_local_common_clock_ns,
                           int64_t sample_time_raw_device_clock_ns) override
    {
        auto it = m_pushers.find(metadata.stream);
        if (it == m_pushers.end())
        {
            m_logger->warn("Stream {} not found in SchemaMetadataPusher", core::EnumNameStreamType(metadata.stream));
            return;
        }

        flatbuffers::FlatBufferBuilder builder(MAX_FLATBUFFER_SIZE);
        auto offset = core::FrameMetadataOak::Pack(builder, &metadata);
        builder.Finish(offset);
        it->second->push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_local_common_clock_ns,
                                sample_time_raw_device_clock_ns);
    }

private:
    static constexpr size_t MAX_FLATBUFFER_SIZE = 128;
    std::shared_ptr<core::OpenXRSession> m_oxr_session;
    std::map<core::StreamType, std::unique_ptr<core::SchemaPusher>> m_pushers;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.plugins.oak.SchemaMetadataPusher");
};

// =============================================================================
// McapMetadataPusher — writes frame metadata to an MCAP file
// =============================================================================

class McapMetadataPusher : public IMetadataPusher
{
public:
    McapMetadataPusher(const std::vector<StreamConfig>& streams, const std::string& mcap_filename)
    {
        mcap::McapWriterOptions options("oak_camera");
        options.compression = mcap::Compression::None;

        auto status = m_writer.open(mcap_filename, options);
        if (!status.ok())
            throw std::runtime_error("McapMetadataPusher: Failed to open " + mcap_filename + ": " + status.message);

        mcap::Schema schema("core.FrameMetadataOakRecord", "flatbuffer",
                            std::string(reinterpret_cast<const char*>(core::FrameMetadataOakRecordBinarySchema::data()),
                                        core::FrameMetadataOakRecordBinarySchema::size()));
        m_writer.addSchema(schema);

        for (const auto& config : streams)
        {
            // "<base>/oak_tracked" is the topic McapReplayConfig reads for FrameMetadataTrackerOak, so
            // this file replays with base name "oak_metadata/<Stream>".
            std::string channel_name =
                std::string("oak_metadata/") + core::EnumNameStreamType(config.camera) + "/oak_tracked";
            mcap::Channel channel(channel_name, "flatbuffer", schema.id);
            m_writer.addChannel(channel);
            m_channel_ids[config.camera] = channel.id;
            m_logger->info("  MCAP channel: {}", channel_name);
        }

        m_logger->info("MCAP recording to: {}", mcap_filename);
    }

    ~McapMetadataPusher() override
    {
        m_writer.close();
        m_logger->info("MCAP closed with {} messages", m_message_count);
    }

    void on_frame_metadata(const core::FrameMetadataOakT& metadata,
                           int64_t sample_time_local_common_clock_ns,
                           int64_t sample_time_raw_device_clock_ns) override
    {
        auto it = m_channel_ids.find(metadata.stream);
        if (it == m_channel_ids.end())
        {
            m_logger->warn("Stream {} not found in MCAP", core::EnumNameStreamType(metadata.stream));
            return;
        }

        const int64_t now_ns = core::os_monotonic_now_ns();

        flatbuffers::FlatBufferBuilder builder(MAX_FLATBUFFER_SIZE);
        auto data_offset = core::FrameMetadataOak::Pack(builder, &metadata);
        // available_time: when this record was written (now).
        // sample_time_local_common_clock: when the frame was captured (common clock).
        // sample_time_raw_device_clock: when the frame was captured (device clock).
        core::DeviceDataTimestamp timestamp(now_ns, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns);
        core::FrameMetadataOakRecordBuilder record_builder(builder);
        record_builder.add_data(data_offset);
        record_builder.add_timestamp(&timestamp);
        builder.Finish(record_builder.Finish());

        mcap::Message msg;
        msg.channelId = it->second;
        msg.logTime = static_cast<mcap::Timestamp>(now_ns);
        msg.publishTime = static_cast<mcap::Timestamp>(now_ns);
        msg.sequence = static_cast<uint32_t>(m_message_count);
        msg.data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
        msg.dataSize = builder.GetSize();

        auto status = m_writer.write(msg);
        if (!status.ok())
            // Runs at frame rate: if writes start failing persistently, this will spam the log.
            m_logger->warn("write failed: {}", status.message);

        ++m_message_count;
    }

private:
    static constexpr size_t MAX_FLATBUFFER_SIZE = 128;
    mcap::McapWriter m_writer;
    std::map<core::StreamType, mcap::ChannelId> m_channel_ids;
    uint64_t m_message_count = 0;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.plugins.oak.McapMetadataPusher");
};

// =============================================================================
// Factory
// =============================================================================

std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams,
                                             const std::string& collection_prefix,
                                             const std::string& mcap_filename)
{
    if (!collection_prefix.empty() && !mcap_filename.empty())
        throw std::runtime_error("Cannot specify both --collection-prefix and --mcap-filename");

    std::unique_ptr<IMetadataPusher> pusher;

    if (!collection_prefix.empty())
        pusher = std::make_unique<SchemaMetadataPusher>(streams, collection_prefix);
    else if (!mcap_filename.empty())
        pusher = std::make_unique<McapMetadataPusher>(streams, mcap_filename);

    return std::make_unique<FrameSink>(streams, std::move(pusher));
}

} // namespace oak
} // namespace plugins
