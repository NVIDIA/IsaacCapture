// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define MCAP_IMPLEMENTATION
#include "frame_sink.hpp"

#include <flatbuffers/flatbuffers.h>
#include <mcap/writer.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/sensing_bfbs_generated.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace plugins
{
namespace sensing
{

// =============================================================================
// FrameSink
// =============================================================================

FrameSink::FrameSink(const std::vector<StreamConfig>& streams, std::unique_ptr<IMetadataPusher> metadata_pusher)
    : m_metadata_pusher(std::move(metadata_pusher))
{
    for (const auto& config : streams)
    {
        // An ipc-only stream has no H.264 to write. SensingCamera still calls
        // on_frame for it, so its metadata is recorded.
        if (config.output_path.empty())
            continue;

        std::filesystem::path p(config.output_path);
        auto parent = p.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        m_writers[config.sensor_id] = std::make_unique<RawDataWriter>(config.output_path);
        std::cout << "Add stream:  sensor " << config.sensor_id << " -> " << config.output_path << std::endl;
    }
}

void FrameSink::on_frame(const SensingFrame& frame)
{
    auto it = m_writers.find(frame.sensor_id);
    if (it != m_writers.end())
        it->second->write(frame.h264_data);

    if (m_metadata_pusher)
        m_metadata_pusher->on_frame_metadata(
            frame.metadata, frame.sample_time_local_common_clock_ns, frame.sample_time_raw_device_clock_ns);
}

// =============================================================================
// McapMetadataPusher — writes frame metadata to an MCAP file
// =============================================================================

class McapMetadataPusher : public IMetadataPusher
{
public:
    McapMetadataPusher(const std::vector<StreamConfig>& streams, const std::string& mcap_filename)
    {
        mcap::McapWriterOptions options("sensing_camera");
        options.compression = mcap::Compression::None;

        auto status = m_writer.open(mcap_filename, options);
        if (!status.ok())
            throw std::runtime_error("McapMetadataPusher: Failed to open " + mcap_filename + ": " + status.message);

        mcap::Schema schema(
            "core.FrameMetadataSensingRecord", "flatbuffer",
            std::string(reinterpret_cast<const char*>(core::FrameMetadataSensingRecordBinarySchema::data()),
                        core::FrameMetadataSensingRecordBinarySchema::size()));
        m_writer.addSchema(schema);

        for (const auto& config : streams)
        {
            std::string channel_name = "sensing_metadata/sensor" + std::to_string(config.sensor_id);
            mcap::Channel channel(channel_name, "flatbuffer", schema.id);
            m_writer.addChannel(channel);
            m_channel_ids[config.sensor_id] = channel.id;
            std::cout << "  MCAP channel: " << channel_name << std::endl;
        }

        std::cout << "MCAP recording to: " << mcap_filename << std::endl;
    }

    ~McapMetadataPusher() override
    {
        m_writer.close();
        std::cout << "MCAP closed with " << m_message_count << " messages" << std::endl;
    }

    void on_frame_metadata(const core::FrameMetadataSensingT& metadata,
                           int64_t sample_time_local_common_clock_ns,
                           int64_t sample_time_raw_device_clock_ns) override
    {
        auto it = m_channel_ids.find(metadata.sensor_id);
        if (it == m_channel_ids.end())
        {
            std::cerr << "McapMetadataPusher: Sensor " << metadata.sensor_id << " not found in MCAP" << std::endl;
            return;
        }

        const int64_t now_ns = core::os_monotonic_now_ns();

        flatbuffers::FlatBufferBuilder builder(MAX_FLATBUFFER_SIZE);
        auto data_offset = core::FrameMetadataSensing::Pack(builder, &metadata);
        core::DeviceDataTimestamp timestamp(now_ns, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns);
        core::FrameMetadataSensingRecordBuilder record_builder(builder);
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
            std::cerr << "McapMetadataPusher: write failed: " << status.message << std::endl;

        ++m_message_count;
    }

private:
    static constexpr size_t MAX_FLATBUFFER_SIZE = 128;
    mcap::McapWriter m_writer;
    std::map<uint32_t, mcap::ChannelId> m_channel_ids;
    uint64_t m_message_count = 0;
};

// =============================================================================
// Factory
// =============================================================================

std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams, const std::string& mcap_filename)
{
    std::unique_ptr<IMetadataPusher> pusher;
    if (!mcap_filename.empty())
        pusher = std::make_unique<McapMetadataPusher>(streams, mcap_filename);

    return std::make_unique<FrameSink>(streams, std::move(pusher));
}

} // namespace sensing
} // namespace plugins
