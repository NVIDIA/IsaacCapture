// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/orbbec_ego_camera_bfbs_generated.h>
#include <schema/orbbec_ego_camera_generated.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

std::string extension(core::OrbbecEgoPixelFormat format)
{
    switch (format)
    {
    case core::OrbbecEgoPixelFormat_Mjpg:
        return "mjpg";
    case core::OrbbecEgoPixelFormat_H264:
        return "h264";
    case core::OrbbecEgoPixelFormat_H265:
        return "h265";
    default:
        throw std::runtime_error("unsupported Ego video format in embedded MCAP");
    }
}

std::unique_ptr<mcap::McapReader> open_reader(const std::filesystem::path& path)
{
    auto reader = std::make_unique<mcap::McapReader>();
    const auto status = reader->open(path.string());
    if (!status.ok())
        throw std::runtime_error("unable to open MCAP: " + status.message);
    return reader;
}

void export_media(const std::filesystem::path& input_path, const std::filesystem::path& output_dir)
{
    auto video_reader = open_reader(input_path);
    const auto summary = video_reader->readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
    if (!summary.ok() || !video_reader->footer())
        throw std::runtime_error("embedded MCAP has no valid completed summary/footer");
    const core::RecordedSchemas recorded(*video_reader);
    video_reader.reset();
    using VideoViewer = core::McapTrackerViewers<core::OrbbecEgoEncodedVideoFrameRecord>;
    // A reader per eye prevents one eye's drain from buffering the entire other eye.
    std::array<std::unique_ptr<VideoViewer>, 2> video;
    for (size_t index = 0; index < video.size(); ++index)
        video[index] = std::make_unique<VideoViewer>(open_reader(input_path), "orbbec_ego_media",
                                                     std::vector<std::string>{ core::EnumNameOrbbecEgoCameraStream(
                                                         static_cast<core::OrbbecEgoCameraStream>(index)) },
                                                     recorded);

    std::filesystem::create_directories(output_dir);
    size_t video_count = 0;
    for (size_t index = 0; index < 2; ++index)
    {
        std::ofstream output;
        std::filesystem::path path;
        core::OrbbecEgoPixelFormat format{};
        while (const auto record = video[index]->read(0))
        {
            const auto* data = record->data();
            if (!data || !data->encoded_data())
                continue;
            if (static_cast<size_t>(data->stream()) != index)
                throw std::runtime_error("embedded video stream does not match its MCAP topic");
            if (!output.is_open())
            {
                format = data->pixel_format();
                path = output_dir /
                       (std::string(core::EnumNameOrbbecEgoCameraStream(data->stream())) + "." + extension(format));
                output.open(path, std::ios::binary | std::ios::trunc);
                if (!output)
                    throw std::runtime_error("unable to create video output: " + path.string());
            }
            if (data->pixel_format() != format)
                throw std::runtime_error("embedded video format changed during recording");
            output.write(reinterpret_cast<const char*>(data->encoded_data()->data()),
                         static_cast<std::streamsize>(data->encoded_data()->size()));
            if (!output)
                throw std::runtime_error("failed while exporting embedded video");
        }
        if (output.is_open())
        {
            output.close();
            if (!output)
                throw std::runtime_error("failed while finalizing embedded video output");
            ++video_count;
            std::cout << core::EnumNameOrbbecEgoCameraStream(static_cast<core::OrbbecEgoCameraStream>(index)) << ": "
                      << path << std::endl;
        }
    }

    if (video_count == 0)
        throw std::runtime_error("MCAP contains no embedded Ego media channels");
}

} // namespace

int main(int argc, char** argv)
try
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " INPUT.mcap OUTPUT_DIRECTORY" << std::endl;
        return 1;
    }
    export_media(argv[1], argv[2]);
    return 0;
}
catch (const std::exception& error)
{
    isaaccapture::Logger::get("isaaccapture.plugins.orbbec_ego.export_media")->error("{}: {}", argv[0], error.what());
    return 1;
}
