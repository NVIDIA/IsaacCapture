// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rawdata_writer.hpp"
#include "sensing_types.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/**
 * @brief Interface to push per-sensor frame metadata.
 *
 * McapMetadataPusher is the only implementation; the interface stays because
 * src/plugins/oak/core/frame_sink.hpp has the same shape and the two are
 * candidates to hoist into plugin_utils together.
 */
class IMetadataPusher
{
public:
    virtual ~IMetadataPusher() = default;
    virtual void on_frame_metadata(const core::FrameMetadataSensingT& metadata,
                                   int64_t sample_time_local_common_clock_ns,
                                   int64_t sample_time_raw_device_clock_ns) = 0;
};

/**
 * @brief Multi-sensor output sink for SENSING frames.
 *
 * Writes raw H.264 per sensor that asked for it, and optionally delegates to an
 * IMetadataPusher. A stream with no output path still reaches on_frame, so its
 * metadata is recorded even though it has no bitstream.
 */
class FrameSink
{
public:
    explicit FrameSink(const std::vector<StreamConfig>& streams,
                       std::unique_ptr<IMetadataPusher> metadata_pusher = nullptr);

    FrameSink(const FrameSink&) = delete;
    FrameSink& operator=(const FrameSink&) = delete;

    void on_frame(const SensingFrame& frame);

private:
    std::map<uint32_t, std::unique_ptr<RawDataWriter>> m_writers;
    std::unique_ptr<IMetadataPusher> m_metadata_pusher;
};

/**
 * @brief Create a FrameSink, attaching an MCAP pusher when a filename is given.
 */
std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams, const std::string& mcap_filename);

} // namespace sensing
} // namespace plugins
