// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/ego_audio_bfbs_generated.h>
#include <schema/ego_audio_generated.h>
#include <schema/ego_camera_bfbs_generated.h>
#include <schema/ego_camera_generated.h>

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

std::string extension(core::EgoPixelFormat format)
{
    switch (format)
    {
    case core::EgoPixelFormat_Mjpg:
        return "mjpg";
    case core::EgoPixelFormat_H264:
        return "h264";
    case core::EgoPixelFormat_H265:
        return "h265";
    default:
        throw std::runtime_error("unsupported Ego video format in embedded MCAP");
    }
}

void write_le16(std::ofstream& output, uint16_t value)
{
    const uint8_t bytes[] = { static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8) };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_le32(std::ofstream& output, uint32_t value)
{
    const uint8_t bytes[] = { static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                              static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24) };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_wav_header(std::ofstream& output, uint32_t bytes, uint32_t rate, uint16_t channels, uint16_t bits)
{
    output.write("RIFF", 4);
    write_le32(output, 36 + bytes);
    output.write("WAVEfmt ", 8);
    write_le32(output, 16);
    write_le16(output, 1);
    write_le16(output, channels);
    write_le32(output, rate);
    const auto block_align = static_cast<uint16_t>(channels * bits / 8);
    write_le32(output, rate * block_align);
    write_le16(output, block_align);
    write_le16(output, bits);
    output.write("data", 4);
    write_le32(output, bytes);
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
    using VideoViewer = core::McapTrackerViewers<core::EgoEncodedVideoFrameRecord>;
    // A reader per eye prevents one eye's drain from buffering the entire other eye.
    std::array<std::unique_ptr<VideoViewer>, 2> video;
    for (size_t index = 0; index < video.size(); ++index)
        video[index] = std::make_unique<VideoViewer>(
            open_reader(input_path), "ego_media",
            std::vector<std::string>{ core::EnumNameEgoCameraStream(static_cast<core::EgoCameraStream>(index)) },
            recorded);
    core::McapTrackerViewers<core::EgoPcmAudioChunkRecord> audio(
        open_reader(input_path), "ego_media", { "Audio" }, recorded);

    std::filesystem::create_directories(output_dir);
    size_t video_count = 0;
    for (size_t index = 0; index < 2; ++index)
    {
        std::ofstream output;
        std::filesystem::path path;
        core::EgoPixelFormat format{};
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
                path =
                    output_dir / (std::string(core::EnumNameEgoCameraStream(data->stream())) + "." + extension(format));
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
            std::cout << core::EnumNameEgoCameraStream(static_cast<core::EgoCameraStream>(index)) << ": " << path
                      << std::endl;
        }
    }

    std::ofstream wav;
    const auto wav_path = output_dir / "Audio.wav";
    uint64_t pcm_bytes = 0;
    uint32_t audio_rate = 0;
    uint16_t audio_channels = 0;
    uint16_t audio_bits = 0;
    while (const auto record = audio.read(0))
    {
        const auto* data = record->data();
        if (!data || !data->pcm_data())
            continue;
        if (data->sample_rate_hz() == 0 || data->channel_count() == 0 || data->bits_per_sample() != 16 ||
            data->sample_format() != core::EgoAudioSampleFormat_S16LE ||
            static_cast<uint64_t>(data->sample_count()) * data->channel_count() * 2 != data->pcm_data()->size())
            throw std::runtime_error("embedded PCM profile or sample count is invalid");
        if (!wav.is_open())
        {
            audio_rate = data->sample_rate_hz();
            audio_channels = data->channel_count();
            audio_bits = data->bits_per_sample();
            wav.open(wav_path, std::ios::binary | std::ios::trunc);
            if (!wav)
                throw std::runtime_error("unable to create WAV output");
            // The final RIFF sizes are known only after all PCM chunks are copied.
            write_wav_header(wav, 0, audio_rate, audio_channels, audio_bits);
        }
        if (audio_rate != data->sample_rate_hz() || audio_channels != data->channel_count() ||
            audio_bits != data->bits_per_sample())
            throw std::runtime_error("embedded audio profile changed during recording");
        pcm_bytes += data->pcm_data()->size();
        if (pcm_bytes > UINT32_MAX - 36U)
            throw std::runtime_error("embedded PCM exceeds the RIFF/WAV 4 GiB limit");
        wav.write(reinterpret_cast<const char*>(data->pcm_data()->data()),
                  static_cast<std::streamsize>(data->pcm_data()->size()));
        if (!wav)
            throw std::runtime_error("failed while exporting embedded audio");
    }
    if (wav.is_open())
    {
        wav.seekp(0);
        write_wav_header(wav, static_cast<uint32_t>(pcm_bytes), audio_rate, audio_channels, audio_bits);
        wav.close();
        if (!wav)
            throw std::runtime_error("failed while finalizing embedded WAV output");
        std::cout << "Audio: " << wav_path << std::endl;
    }
    if (video_count == 0 && pcm_bytes == 0)
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
    isaaccapture::Logger::get("isaaccapture.plugins.ego.export_media")->error("{}: {}", argv[0], error.what());
    return 1;
}
