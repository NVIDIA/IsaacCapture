// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diagnostics.hpp"
#include "metadata_sink.hpp"

#include <ego_mcap/checked_write.hpp>
#include <mcap/writer.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/orbbec_ego_audio_bfbs_generated.h>
#include <schema/orbbec_ego_calibration_bfbs_generated.h>
#include <schema/orbbec_ego_camera_bfbs_generated.h>
#include <schema/orbbec_ego_device_state_bfbs_generated.h>
#include <schema/orbbec_ego_imu_bfbs_generated.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <utility>

namespace plugins::ego
{

namespace
{
class McapMetadataSink final : public IMetadataSink
{
public:
    McapMetadataSink(const std::vector<StreamConfig>& streams,
                     const std::string& filename,
                     bool include_media,
                     bool include_structured,
                     bool record_device_diagnostics)
        : final_filename_(filename), temporary_filename_(filename + ".partial")
    {
        if (std::filesystem::exists(final_filename_) || std::filesystem::is_symlink(final_filename_) ||
            std::filesystem::exists(temporary_filename_) || std::filesystem::is_symlink(temporary_filename_))
            throw std::runtime_error("Ego MCAP output already exists: " + filename);
        try
        {
            mcap::McapWriterOptions options("ego");
            options.compression = mcap::Compression::None;
            const std::filesystem::path output_path(filename);
            if (!output_path.parent_path().empty())
                std::filesystem::create_directories(output_path.parent_path());
            output_.open(temporary_filename_);
            writer_.open(output_, options);
            std::vector<std::string> video_names;
            for (const auto& stream : streams)
                video_names.emplace_back(core::EnumNameOrbbecEgoCameraStream(stream.camera));
            if (include_structured)
            {
                video_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoFrameMetadataRecord>>(
                    writer_, "orbbec_ego_metadata", video_names);
            }
            for (size_t index = 0; index < streams.size(); ++index)
                video_indices_.emplace(streams[index].camera, index);
            if (include_structured)
            {
                imu_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoImuBatchRecord>>(
                    writer_, "orbbec_ego_imu", std::vector<std::string>{ "Accel", "Gyro" });
                audio_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoAudioChunkRecord>>(
                    writer_, "orbbec_ego_audio", std::vector<std::string>{ "Audio" });
                calibration_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoCalibrationRecord>>(
                    writer_, "orbbec_ego_calibration", std::vector<std::string>{ "Calibration" });
                if (record_device_diagnostics)
                    device_state_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoDeviceStateRecord>>(
                        writer_, "orbbec_ego_device", std::vector<std::string>{ "DeviceState" });
            }
            mcap::Metadata policy;
            policy.name = "ego.recording";
            policy.metadata = { { "record_device_diagnostics", record_device_diagnostics ? "on" : "off" },
                                { "diagnostic_version", "2" } };
            const auto policy_status = writer_.write(policy);
            if (!policy_status.ok())
                throw std::runtime_error("Unable to write EGO recording policy: " + policy_status.message);
            if (include_media)
            {
                media_video_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoEncodedVideoFrameRecord>>(
                    writer_, "orbbec_ego_media", video_names);
                media_audio_ = std::make_unique<core::McapTrackerChannels<core::OrbbecEgoPcmAudioChunkRecord>>(
                    writer_, "orbbec_ego_media", std::vector<std::string>{ "Audio" });
            }
        }
        catch (...)
        {
            writer_.terminate();
            throw;
        }
    }

    ~McapMetadataSink() override
    {
        try
        {
            abort();
        }
        catch (const std::exception& error)
        {
            // Destructors run during capture-error unwinding. Preserve the primary
            // failure rather than terminating while attempting to write the footer.
            detail::log_error() << "Ego MCAP shutdown failed: " << error.what() << std::endl;
        }
    }

    void close() override
    {
        if (closed_)
            return;
        try
        {
            writer_.close();
            output_.end();
            closed_ = true;
            closed_successfully_ = true;
        }
        catch (...)
        {
            // McapWriter retries close from its destructor unless it is reset.
            // Terminate after an I/O failure so an error-path footer retry cannot abort.
            writer_.terminate();
            closed_ = true;
            throw;
        }
    }

    void commit() override
    {
        if (committed_)
            return;
        if (!closed_successfully_ || aborted_)
            throw std::runtime_error("Cannot commit unfinished Ego MCAP output");
        if (std::filesystem::exists(final_filename_) || std::filesystem::is_symlink(final_filename_))
            throw std::runtime_error("Ego MCAP final output appeared before commit: " + final_filename_);
        std::filesystem::rename(temporary_filename_, final_filename_);
        committed_ = true;
    }

    void abort() override
    {
        aborted_ = true;
        close();
    }

    void on_frame_metadata(const CapturedFrame& frame) override
    {
        if (video_)
            write_record<core::OrbbecEgoFrameMetadataRecord>(*video_, video_indices_.at(frame.metadata.stream),
                                                             frame.metadata, frame.sample_time_local_common_clock_ns,
                                                             frame.sample_time_raw_device_clock_ns);
    }
    void on_imu_batch(const core::OrbbecEgoImuBatchT& batch, int64_t local_ns, int64_t device_ns) override
    {
        if (imu_)
            write_record<core::OrbbecEgoImuBatchRecord>(
                *imu_, static_cast<size_t>(batch.sensor), batch, local_ns, device_ns);
    }
    void on_audio_chunk(const core::OrbbecEgoAudioChunkT& chunk, int64_t local_ns, int64_t device_ns) override
    {
        if (audio_)
            write_record<core::OrbbecEgoAudioChunkRecord>(*audio_, 0, chunk, local_ns, device_ns);
    }
    void on_calibration(const core::OrbbecEgoCalibrationT& calibration, int64_t local_ns, int64_t device_ns) override
    {
        if (calibration_)
            write_record<core::OrbbecEgoCalibrationRecord>(*calibration_, 0, calibration, local_ns, device_ns);
    }
    void on_device_state(const core::OrbbecEgoDeviceStateT& state, int64_t local_ns, int64_t device_ns) override
    {
        if (device_state_)
            write_record<core::OrbbecEgoDeviceStateRecord>(*device_state_, 0, state, local_ns, device_ns);
    }
    void on_encoded_video_frame(const CapturedFrame& frame) override
    {
        if (!media_video_)
            return;
        core::OrbbecEgoEncodedVideoFrameT data;
        data.stream = frame.metadata.stream;
        data.sequence_number = frame.metadata.sequence_number;
        data.width = frame.metadata.width;
        data.height = frame.metadata.height;
        data.fps = frame.metadata.fps;
        data.pixel_format = frame.metadata.pixel_format;
        data.encoded_data = frame.encoded_data;
        data.capture_epoch = frame.metadata.capture_epoch;
        write_record<core::OrbbecEgoEncodedVideoFrameRecord>(*media_video_, video_indices_.at(data.stream), data,
                                                             frame.sample_time_local_common_clock_ns,
                                                             frame.sample_time_raw_device_clock_ns);
    }
    void on_pcm_audio_chunk(const core::OrbbecEgoPcmAudioChunkT& chunk, int64_t local_ns, int64_t device_ns) override
    {
        if (media_audio_)
            write_record<core::OrbbecEgoPcmAudioChunkRecord>(*media_audio_, 0, chunk, local_ns, device_ns);
    }

private:
    template <typename RecordT>
    void write_record(core::McapTrackerChannels<RecordT>& channels,
                      size_t index,
                      const core::record_payload_t<RecordT>& value,
                      int64_t local_ns,
                      int64_t device_ns)
    {
        try
        {
            if (closed_)
                throw std::runtime_error("Attempted to write a closed Ego MCAP output");
            ego_mcap::write_checked(
                writer_, channels, index, core::pack_record<RecordT>(&value, timestamp(local_ns, device_ns)));
        }
        catch (...)
        {
            aborted_ = true;
            throw;
        }
    }

    class CheckedMcapFileWriter final : public mcap::IWritable
    {
    public:
        ~CheckedMcapFileWriter() override
        {
            try
            {
                end();
            }
            catch (const std::exception&)
            {
            }
        }

        void open(const std::string& filename)
        {
            end();
            size_ = 0;
            // Exclusive creation preserves a prior failed capture, including dangling symlinks.
            file_ = std::fopen(filename.c_str(), "wbx");
            if (!file_)
                throw std::runtime_error("Unable to open Ego MCAP output: " + filename + ": " + std::strerror(errno));
        }

        void end() override
        {
            if (!file_)
                return;
            FILE* const file = std::exchange(file_, nullptr);
            const int flush_status = std::fflush(file);
            const int close_status = std::fclose(file);
            if (flush_status != 0 || close_status != 0)
                throw std::runtime_error("Failed while closing Ego MCAP output: " + std::string(std::strerror(errno)));
        }

        uint64_t size() const override
        {
            return size_;
        }

    protected:
        void handleWrite(const std::byte* data, uint64_t size) override
        {
            if (!file_)
                throw std::runtime_error("Attempted to write a closed Ego MCAP output");
            const size_t written = std::fwrite(data, 1, static_cast<size_t>(size), file_);
            if (written != size)
                throw std::runtime_error("Failed while writing Ego MCAP output: " + std::string(std::strerror(errno)));
            size_ += size;
        }

    private:
        FILE* file_ = nullptr;
        uint64_t size_ = 0;
    };

    static core::DeviceDataTimestamp timestamp(int64_t local_ns, int64_t device_ns)
    {
        return core::DeviceDataTimestamp(core::os_monotonic_now_ns(), local_ns, device_ns);
    }

    CheckedMcapFileWriter output_;
    mcap::McapWriter writer_;
    std::string final_filename_;
    std::string temporary_filename_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoFrameMetadataRecord>> video_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoImuBatchRecord>> imu_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoAudioChunkRecord>> audio_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoCalibrationRecord>> calibration_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoDeviceStateRecord>> device_state_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoEncodedVideoFrameRecord>> media_video_;
    std::unique_ptr<core::McapTrackerChannels<core::OrbbecEgoPcmAudioChunkRecord>> media_audio_;
    std::map<core::OrbbecEgoCameraStream, size_t> video_indices_;
    bool closed_ = false;
    bool closed_successfully_ = false;
    bool aborted_ = false;
    bool committed_ = false;
};

}

std::unique_ptr<IMetadataSink> make_mcap_metadata_sink(const std::vector<StreamConfig>& streams,
                                                       const std::string& filename,
                                                       bool include_media,
                                                       bool include_structured,
                                                       bool record_device_diagnostics)
{
    return std::make_unique<McapMetadataSink>(
        streams, filename, include_media, include_structured, record_device_diagnostics);
}

} // namespace plugins::ego
