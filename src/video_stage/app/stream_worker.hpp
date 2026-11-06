// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "livekit_output.hpp"
#include "pipeline.hpp"

#include <video_stage/cuda_context.hpp>
#include <video_stage/sine_warp.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace video_stage
{

/// One camera's stream: where the Orin connects, and where the processed video goes.
struct StreamSpec
{
    uint16_t port = 0;
    /// LiveKit track name; used when a LiveKit session is given.
    std::string track;
    /// Annex-B output file; used instead of LiveKit when set (for checking a run).
    std::string file;
};

struct StreamSettings
{
    PipelineConfig pipeline;
    SineWarpParams warp;
    LiveKitOutputConfig livekit;
    bool once = false; ///< serve one Orin session, then finish
};

/**
 * @brief One stream's thread: accept an Orin session, run it to its end, accept the next.
 *
 * Owns that stream's receiver, control channel, pipeline and output. Streams share the CUDA context and, when
 * publishing, one LiveKit participant, and otherwise run independently so one camera's trouble stays its own.
 */
class StreamWorker
{
public:
    StreamWorker(CudaContext& context, const StreamSpec& spec, const StreamSettings& settings, SharedLiveKit* livekit);
    ~StreamWorker();

    StreamWorker(const StreamWorker&) = delete;
    StreamWorker& operator=(const StreamWorker&) = delete;

    void start();
    /// Ends the wait for an Orin and shuts down any running session.
    void request_stop();
    /// Waits for the thread; false if any session of this stream failed.
    bool join();

    void print_summary(std::ostream& out) const;
    /// One compact line of the current percentiles; safe to call while the stream runs.
    void print_live(std::ostream& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace video_stage
