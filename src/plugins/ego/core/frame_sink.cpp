// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diagnostics.hpp"
#include "inc/ego_camera/ego_camera.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace plugins::ego
{

void validate_stream_config(const StreamConfig& stream, const CaptureConfig& config)
{
    static_cast<void>(config);
    switch (stream.pixel_format)
    {
    case core::EgoPixelFormat_Mjpg:
    case core::EgoPixelFormat_H264:
    case core::EgoPixelFormat_H265:
        return;
    default:
        throw std::invalid_argument("Unsupported Ego pixel format");
    }
}

void validate_resolved_stream_config(const StreamConfig& stream, uint32_t resolved_fps)
{
    const bool encoded =
        stream.pixel_format == core::EgoPixelFormat_H264 || stream.pixel_format == core::EgoPixelFormat_H265;
    if (encoded && resolved_fps > 30)
    {
        throw std::invalid_argument("resolved encoded profile is " + std::to_string(resolved_fps) +
                                    " FPS, but this build certifies encoded recording only through 30 FPS; "
                                    "no profile fallback was attempted");
    }
}

class FrameSink::Impl
{
public:
    Impl(const std::vector<StreamConfig>& streams, std::unique_ptr<IMetadataSink> metadata_sink, bool write_media_sidecars)
        : metadata_sink_(std::move(metadata_sink))
    {
        for (const auto& stream : streams)
        {
            std::unique_ptr<std::ofstream> file;
            if (write_media_sidecars)
            {
                const std::filesystem::path output(stream.output_path);
                if (!output.parent_path().empty())
                    std::filesystem::create_directories(output.parent_path());
                file = std::make_unique<std::ofstream>(stream.output_path, std::ios::binary | std::ios::trunc);
                if (!*file)
                    throw std::runtime_error("Unable to open encoded output: " + stream.output_path);
                detail::log_info() << "Add stream: " << core::EnumNameEgoCameraStream(stream.camera) << " -> "
                                   << stream.output_path << std::endl;
            }
            writers_.emplace(stream.camera, Writer{ std::move(file), stream.pixel_format });
        }
    }

    void on_frame(const CapturedFrame& frame, const std::function<void(const CapturedFrame&)>& accepted_frame)
    {
        const auto it = writers_.find(frame.metadata.stream);
        if (it == writers_.end())
            return;

        const bool sequence_gap =
            it->second.has_sequence && frame.metadata.sequence_number > it->second.last_sequence_number + 1;
        it->second.has_sequence = true;
        it->second.last_sequence_number = frame.metadata.sequence_number;
        auto encoded_data = it->second.remove_ego_timestamp_sei(frame.encoded_data);
        if (encoded_data.empty() || !it->second.accept(encoded_data, sequence_gap))
            return;

        if (it->second.file)
        {
            it->second.file->write(
                reinterpret_cast<const char*>(encoded_data.data()), static_cast<std::streamsize>(encoded_data.size()));
            if (!*it->second.file)
                throw std::runtime_error("Failed while writing Ego encoded data");
        }

        if (!metadata_sink_ && !accepted_frame)
            return;
        CapturedFrame recorded{ frame.metadata, std::move(encoded_data), frame.sample_time_local_common_clock_ns,
                                frame.sample_time_raw_device_clock_ns };
        recorded.metadata.encoded_bytes = recorded.encoded_data.size();
        if (metadata_sink_)
        {
            metadata_sink_->on_frame_metadata(recorded);
            metadata_sink_->on_encoded_video_frame(recorded);
        }
        if (accepted_frame)
            accepted_frame(recorded);
    }

    void begin_capture_epoch()
    {
        for (auto& [_, writer] : writers_)
            writer.begin_capture_epoch();
    }

    IMetadataSink* metadata_sink()
    {
        return metadata_sink_.get();
    }

    void close_media()
    {
        if (media_failed_)
            throw std::runtime_error("Ego media sidecar finalization previously failed");
        std::string failures;
        for (auto& [stream, writer] : writers_)
        {
            if (!writer.file || !writer.file->is_open())
                continue;
            writer.file->flush();
            bool failed = !*writer.file;
            writer.file->close();
            failed = failed || writer.file->fail();
            if (failed)
            {
                if (!failures.empty())
                    failures += "; ";
                failures += std::string(core::EnumNameEgoCameraStream(stream));
            }
        }
        if (!failures.empty())
        {
            media_failed_ = true;
            throw std::runtime_error("Failed while finalizing Ego media sidecars: " + failures);
        }
        media_closed_ = true;
    }

    void commit_metadata()
    {
        if (!media_closed_ || media_failed_)
            throw std::runtime_error("Cannot commit Ego metadata before media sidecars close successfully");
        if (metadata_sink_)
            metadata_sink_->commit();
    }

private:
    struct Writer
    {
        std::unique_ptr<std::ofstream> file;
        core::EgoPixelFormat format;
        bool parameter_sets_ready = false;
        bool has_vps = false;
        bool has_sps = false;
        bool has_pps = false;
        bool has_sequence = false;
        uint64_t last_sequence_number = 0;

        static size_t start_code_size(const std::vector<uint8_t>& bytes, size_t offset)
        {
            if (offset + 3 <= bytes.size() && bytes[offset] == 0 && bytes[offset + 1] == 0 && bytes[offset + 2] == 1)
                return 3;
            if (offset + 4 <= bytes.size() && bytes[offset] == 0 && bytes[offset + 1] == 0 && bytes[offset + 2] == 0 &&
                bytes[offset + 3] == 1)
                return 4;
            return 0;
        }

        static bool is_ego_timestamp_sei(const std::vector<uint8_t>& bytes,
                                         size_t payload_begin,
                                         size_t payload_end,
                                         core::EgoPixelFormat format)
        {
            const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(payload_begin);
            const auto end = bytes.begin() + static_cast<std::ptrdiff_t>(payload_end);
            static constexpr std::array<uint8_t, 11> kLegacyMarker = { 'O', 'R', 'B', 'B', 'E', 'C',
                                                                       ',', 'E', 'G', 'O', '_' };
            if (std::search(begin, end, kLegacyMarker.begin(), kLegacyMarker.end()) != end)
                return true;

            static constexpr std::string_view kEgo = "EGO";
            static constexpr std::string_view kTimestamp = ",timestamp_us=";
            static constexpr std::string_view kH264Frame = ",frameId=";
            static constexpr std::string_view kH265Frame = ",frame_seq=";
            const auto matches = [&bytes, payload_end](size_t& offset, std::string_view text)
            {
                if (offset + text.size() > payload_end ||
                    !std::equal(text.begin(), text.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset)))
                {
                    return false;
                }
                offset += text.size();
                return true;
            };
            const auto consume_decimal = [&bytes, payload_end](size_t& offset)
            {
                const size_t first = offset;
                while (offset < payload_end && bytes[offset] >= '0' && bytes[offset] <= '9')
                    ++offset;
                return offset != first;
            };

            for (auto marker = std::search(begin, end, kEgo.begin(), kEgo.end()); marker != end;
                 marker = std::search(marker + 1, end, kEgo.begin(), kEgo.end()))
            {
                size_t offset = static_cast<size_t>(std::distance(bytes.begin(), marker));
                if (!matches(offset, kEgo))
                    continue;
                const size_t digits_begin = offset;
                if (!consume_decimal(offset) || offset - digits_begin != 10 || offset + 2 > payload_end ||
                    bytes[offset] != '_' || (bytes[offset + 1] != 'L' && bytes[offset + 1] != 'R'))
                {
                    continue;
                }
                offset += 2;
                if (!matches(offset, kTimestamp) || !consume_decimal(offset) ||
                    !matches(offset, format == core::EgoPixelFormat_H264 ? kH264Frame : kH265Frame) ||
                    !consume_decimal(offset))
                {
                    continue;
                }
                if (offset + 1 == payload_end && bytes[offset] == 0x80)
                    return true;
            }
            return false;
        }

        std::vector<uint8_t> remove_ego_timestamp_sei(const std::vector<uint8_t>& bytes) const
        {
            if (format == core::EgoPixelFormat_Mjpg)
                return bytes;

            std::vector<uint8_t> result;
            size_t current = 0;
            while (current < bytes.size())
            {
                const auto code_size = start_code_size(bytes, current);
                if (code_size == 0)
                    return bytes;
                const size_t nal_begin = current + code_size;
                if (nal_begin >= bytes.size())
                    break;
                size_t next = nal_begin;
                while (next < bytes.size() && start_code_size(bytes, next) == 0)
                    ++next;
                const uint8_t nal_type =
                    format == core::EgoPixelFormat_H264 ? bytes[nal_begin] & 0x1fU : (bytes[nal_begin] >> 1U) & 0x3fU;
                const bool is_sei =
                    format == core::EgoPixelFormat_H264 ? nal_type == 6 : nal_type == 39 || nal_type == 40;
                if (!is_sei || !is_ego_timestamp_sei(bytes, nal_begin + 1, next, format))
                    result.insert(result.end(), bytes.begin() + static_cast<std::ptrdiff_t>(current),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(next));
                current = next;
            }
            return result;
        }

        bool accept(const std::vector<uint8_t>& bytes, bool sequence_gap)
        {
            if (format == core::EgoPixelFormat_Mjpg)
                return true;

            if (sequence_gap)
            {
                // P frames after a missing access unit reference data that is no
                // longer in this elementary stream. Resume from a parameterized IDR.
                parameter_sets_ready = false;
                has_vps = false;
                has_sps = false;
                has_pps = false;
            }

            bool keyframe = false;
            bool picture = false;
            for (size_t offset = 0; offset + 4 < bytes.size();)
            {
                const auto code_size = start_code_size(bytes, offset);
                if (code_size == 0)
                {
                    ++offset;
                    continue;
                }
                const size_t nal_offset = offset + code_size;
                if (nal_offset >= bytes.size())
                    break;
                if (format == core::EgoPixelFormat_H264)
                {
                    const uint8_t nal_type = bytes[nal_offset] & 0x1fU;
                    has_sps = has_sps || nal_type == 7;
                    has_pps = has_pps || nal_type == 8;
                    keyframe = keyframe || nal_type == 5;
                    picture = picture || (nal_type >= 1 && nal_type <= 5);
                }
                else
                {
                    const uint8_t nal_type = (bytes[nal_offset] >> 1U) & 0x3fU;
                    has_vps = has_vps || nal_type == 32;
                    has_sps = has_sps || nal_type == 33;
                    has_pps = has_pps || nal_type == 34;
                    keyframe = keyframe || nal_type == 19 || nal_type == 20 || nal_type == 21;
                    picture = picture || nal_type <= 31;
                }
                offset = nal_offset + 1;
            }
            if (!parameter_sets_ready)
                parameter_sets_ready = keyframe && has_sps && has_pps && (format == core::EgoPixelFormat_H264 || has_vps);
            // Ego emits a separate SEI-only frame that carries a device timestamp.
            // Timestamp metadata is recorded independently; a standalone SEI is not a
            // decodable elementary-video access unit and must not be appended to media.
            return parameter_sets_ready && picture;
        }

        void begin_capture_epoch()
        {
            parameter_sets_ready = false;
            has_vps = false;
            has_sps = false;
            has_pps = false;
            has_sequence = false;
        }
    };

    std::map<core::EgoCameraStream, Writer> writers_;
    std::unique_ptr<IMetadataSink> metadata_sink_;
    bool media_closed_ = false;
    bool media_failed_ = false;
};

FrameSink::FrameSink(const std::vector<StreamConfig>& streams,
                     std::unique_ptr<IMetadataSink> metadata_sink,
                     bool write_media_sidecars)
    : impl_(std::make_unique<Impl>(streams, std::move(metadata_sink), write_media_sidecars))
{
}

FrameSink::~FrameSink() = default;

void FrameSink::on_frame(const CapturedFrame& frame, const std::function<void(const CapturedFrame&)>& accepted_frame)
{
    impl_->on_frame(frame, accepted_frame);
}

void FrameSink::begin_capture_epoch()
{
    impl_->begin_capture_epoch();
}

IMetadataSink* FrameSink::metadata_sink()
{
    return impl_->metadata_sink();
}

void FrameSink::close_media()
{
    impl_->close_media();
}

void FrameSink::close_metadata()
{
    if (auto* sink = impl_->metadata_sink())
        sink->close();
}

void FrameSink::abort_metadata()
{
    if (auto* sink = impl_->metadata_sink())
        sink->abort();
}

std::string FrameSink::metadata_error() const
{
    if (const auto* sink = impl_->metadata_sink())
        return sink->error();
    return {};
}

void FrameSink::commit_metadata()
{
    impl_->commit_metadata();
}

} // namespace plugins::ego
