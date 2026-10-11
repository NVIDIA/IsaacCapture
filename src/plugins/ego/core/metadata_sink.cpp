// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "metadata_sink.hpp"

#include "diagnostics.hpp"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace plugins::ego
{

namespace
{
class CompositeMetadataSink final : public IMetadataSink
{
public:
    explicit CompositeMetadataSink(std::vector<std::unique_ptr<IMetadataSink>> sinks) : sinks_(std::move(sinks))
    {
    }
    void on_frame_metadata(const CapturedFrame& frame) override
    {
        for_each([&](auto& sink) { sink.on_frame_metadata(frame); });
    }
    void on_imu_batch(const core::EgoImuBatchT& value, int64_t local, int64_t device) override
    {
        for_each([&](auto& sink) { sink.on_imu_batch(value, local, device); });
    }
    void on_audio_chunk(const core::EgoAudioChunkT& value, int64_t local, int64_t device) override
    {
        for_each([&](auto& sink) { sink.on_audio_chunk(value, local, device); });
    }
    void on_calibration(const core::EgoCalibrationT& value, int64_t local, int64_t device) override
    {
        for_each([&](auto& sink) { sink.on_calibration(value, local, device); });
    }
    void on_device_state(const core::EgoDeviceStateT& value, int64_t local, int64_t device) override
    {
        for_each([&](auto& sink) { sink.on_device_state(value, local, device); });
    }
    void on_encoded_video_frame(const CapturedFrame& frame) override
    {
        for_each([&](auto& sink) { sink.on_encoded_video_frame(frame); });
    }
    void on_pcm_audio_chunk(const core::EgoPcmAudioChunkT& value, int64_t local, int64_t device) override
    {
        for_each([&](auto& sink) { sink.on_pcm_audio_chunk(value, local, device); });
    }
    void on_capture_status(const CaptureStatus& status) override
    {
        for_each([&](auto& sink) { sink.on_capture_status(status); });
    }
    void close() override
    {
        closed_successfully_ = false;
        finish([](IMetadataSink& sink) { sink.close(); });
        closed_successfully_ = true;
    }
    void commit() override
    {
        if (!closed_successfully_)
            throw std::runtime_error("Cannot commit Ego metadata before every sink closes successfully");
        for_each([](IMetadataSink& sink) { sink.commit(); });
    }
    void abort() override
    {
        closed_successfully_ = false;
        finish([](IMetadataSink& sink) { sink.abort(); });
    }
    std::string error() const override
    {
        for (const auto& sink : sinks_)
            if (!sink->error().empty())
                return sink->error();
        return {};
    }

private:
    template <typename Finish>
    void finish(Finish&& finish_sink)
    {
        std::string failures;
        // Seal local archives first; promotion waits for every sink to close.
        for (size_t offset = 0; offset < sinks_.size(); ++offset)
        {
            const size_t index = sinks_.size() - offset - 1;
            try
            {
                finish_sink(*sinks_[index]);
            }
            catch (const std::exception& error)
            {
                if (!failures.empty())
                    failures += "; ";
                failures += "sink " + std::to_string(index) + ": " + error.what();
            }
            catch (...)
            {
                if (!failures.empty())
                    failures += "; ";
                failures += "sink " + std::to_string(index) + ": unknown error";
            }
        }
        if (!failures.empty())
            throw std::runtime_error(failures);
    }
    template <typename Function>
    void for_each(Function&& function)
    {
        for (auto& sink : sinks_)
            function(*sink);
    }
    bool closed_successfully_ = false;
    std::vector<std::unique_ptr<IMetadataSink>> sinks_;
};

}

std::unique_ptr<IMetadataSink> compose_metadata_sinks(std::vector<std::unique_ptr<IMetadataSink>> sinks)
{
    return std::make_unique<CompositeMetadataSink>(std::move(sinks));
}

std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams,
                                             const std::string& collection_prefix)
{
    std::unique_ptr<IMetadataSink> metadata_sink;
    if (!collection_prefix.empty())
        metadata_sink = make_schema_metadata_sink(streams, collection_prefix);
    return std::make_unique<FrameSink>(streams, std::move(metadata_sink));
}

std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams, const CaptureConfig& config)
{
    if (!config.collection_prefix.empty() && !config.mcap_filename.empty())
        throw std::invalid_argument("--collection-prefix and --mcap-filename are mutually exclusive");
    if (config.mcap_media_mode == McapMediaMode::Embedded && config.collection_prefix.empty() &&
        config.mcap_filename.empty())
        throw std::invalid_argument("--mcap-media=embedded requires --mcap-filename or --collection-prefix");
    if (config.mcap_media_mode == McapMediaMode::Embedded && !config.collection_prefix.empty() &&
        config.mcap_media_spool.empty())
    {
        throw std::invalid_argument(
            "--mcap-media=embedded with --collection-prefix requires --mcap-media-spool=PATH; "
            "the TeleopSession merger consumes this media fragment after the session closes");
    }
    std::set<std::filesystem::path> outputs;
    const auto require_new_output = [&outputs](const std::string& path)
    {
        if (path.empty())
            return;
        if (std::filesystem::exists(path))
            throw std::invalid_argument("Ego output already exists; refusing to overwrite: " + path);
        if (!outputs.insert(std::filesystem::weakly_canonical(std::filesystem::absolute(path))).second)
            throw std::invalid_argument("Ego outputs refer to the same path: " + path);
    };
    if (config.mcap_media_mode == McapMediaMode::MetadataOnly || config.keep_media_sidecars)
    {
        for (const auto& stream : streams)
            require_new_output(stream.output_path);
        require_new_output(config.audio_output);
    }
    require_new_output(config.calibration_output);
    require_new_output(config.mcap_filename);
    if (config.mcap_media_mode == McapMediaMode::Embedded)
        require_new_output(config.mcap_media_spool);
    std::vector<std::unique_ptr<IMetadataSink>> sinks;
    if (!config.collection_prefix.empty())
        sinks.push_back(make_schema_metadata_sink(streams, config.collection_prefix, config.plugin_root_id));
    else if (!config.plugin_root_id.empty())
    {
        try
        {
            sinks.push_back(make_schema_metadata_sink(streams, {}, config.plugin_root_id));
        }
        catch (const std::exception& error)
        {
            // Local recording must not acquire a mandatory OpenXR dependency for monitoring.
            detail::log_warning() << "Ego device status session unavailable; local capture continues: " << error.what();
        }
    }
    if (!config.mcap_filename.empty())
        sinks.push_back(make_mcap_metadata_sink(streams, config.mcap_filename,
                                                config.mcap_media_mode == McapMediaMode::Embedded, true,
                                                config.record_device_diagnostics));
    if (!config.collection_prefix.empty() && config.mcap_media_mode == McapMediaMode::Embedded)
        sinks.push_back(
            make_mcap_metadata_sink(streams, config.mcap_media_spool, true, true, config.record_device_diagnostics));
    auto metadata_sink = sinks.empty() ? nullptr : compose_metadata_sinks(std::move(sinks));
    return std::make_unique<FrameSink>(
        streams, std::move(metadata_sink),
        config.mcap_media_mode == McapMediaMode::MetadataOnly || config.keep_media_sidecars);
}

} // namespace plugins::ego
