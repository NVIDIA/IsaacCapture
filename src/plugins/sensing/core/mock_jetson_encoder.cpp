// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend for JetsonEncoder: no V4L2 and no NVENC. It runs the real
// RGBA-to-NV12 kernel and the same device-to-host copy as the V4L2 path, then
// a worker thread turns each frame into an Annex-B *shaped* unit.
//
// The units are NOT decodable video. Each carries start codes, plausible NAL
// types (SPS/PPS/IDR on the first frame of every GOP, non-IDR slices between)
// and a payload that records the frame, so sinks and tests can follow a frame
// through. What it reproduces from the real encoder is the behaviour callers
// depend on:
//   * a fixed pool of input buffers, so submit() returns false when the encoder
//     falls behind rather than blocking;
//   * warm-up: a unit appears only after more frames have been submitted;
//   * the capture stamp comes back at microsecond resolution (a timeval), not
//     the nanoseconds it went in with;
//   * end_of_stream() returns the remaining tail as one concatenated unit.
//
// Environment (read once, at construction):
//   SENSING_MOCK_ENCODER_LATENCY_MS=N   per-frame encode time (default 0)
//   SENSING_MOCK_ENCODER_DEPTH=N        frames held back before output (default 2)

#include "jetson_encoder.hpp"
#include "rgba_to_nv12.cuh"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace plugins
{
namespace sensing
{

namespace
{

constexpr uint32_t kInputBuffers = 6;

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("mock JetsonEncoder: ") + what + ": " + cudaGetErrorString(err));
}

uint32_t env_uint(const char* name, uint32_t fallback)
{
    const char* value = std::getenv(name);
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
}

void append_start_code(std::vector<uint8_t>& out)
{
    for (uint8_t byte : { 0, 0, 0, 1 })
        out.push_back(byte);
}

/// One byte becomes two printable bytes, so a payload never contains the zero
/// bytes that would read as a start code.
void append_hex(std::vector<uint8_t>& out, uint32_t value, int nibbles)
{
    for (int shift = (nibbles - 1) * 4; shift >= 0; shift -= 4)
        out.push_back(static_cast<uint8_t>('A' + ((value >> shift) & 0xF)));
}

} // namespace

struct JetsonEncoder::Impl
{
    struct Job
    {
        uint32_t buffer = 0;
        uint32_t frame = 0;
        int64_t timestamp_ns = 0;
    };

    explicit Impl(const EncoderConfig& cfg) : config(cfg)
    {
        gop = config.gop ? config.gop : (config.fps ? config.fps * 5 : 150);
        latency = std::chrono::milliseconds(env_uint("SENSING_MOCK_ENCODER_LATENCY_MS", 0));
        depth = env_uint("SENSING_MOCK_ENCODER_DEPTH", 2);

        check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_y), &d_y_pitch, config.width, config.height),
                   "cudaMallocPitch(Y)");
        check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_uv), &d_uv_pitch, config.width, config.height / 2),
                   "cudaMallocPitch(UV)");

        // Host staging, standing in for the V4L2 output plane's mapped buffers.
        const size_t nv12_bytes = static_cast<size_t>(config.width) * config.height * 3 / 2;
        for (uint32_t i = 0; i < kInputBuffers; ++i)
        {
            host_buffers.emplace_back(nv12_bytes);
            free_buffers.push_back(i);
        }
        worker = std::thread([this] { run(); });
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closing = true;
        }
        wake.notify_all();
        if (worker.joinable())
            worker.join();
        cudaFree(d_y);
        cudaFree(d_uv);
    }

    std::vector<uint8_t> build_unit(const Job& job, uint32_t checksum) const
    {
        std::vector<uint8_t> unit;
        if (job.frame % gop == 0)
        {
            append_start_code(unit);
            unit.push_back(0x67); // SPS
            append_hex(unit, config.width, 4);
            append_hex(unit, config.height, 4);
            append_hex(unit, config.fps, 4);
            append_start_code(unit);
            unit.push_back(0x68); // PPS
            append_hex(unit, 0, 2);
        }
        const bool idr = job.frame % gop == 0;
        append_start_code(unit);
        unit.push_back(idr ? 0x65 : 0x41);
        for (char c : { 'M', 'O', 'C', 'K' })
            unit.push_back(static_cast<uint8_t>(c));
        append_hex(unit, job.frame, 8);
        append_hex(unit, checksum, 8);

        // Size follows the bitrate budget (capped), so throughput looks plausible.
        const size_t budget = config.bitrate_bps / 8 / std::max<uint32_t>(config.fps, 1);
        const size_t target = std::min<size_t>(std::max<size_t>(budget, unit.size()), 64 * 1024);
        unit.resize(target, 'U');
        return unit;
    }

    void run()
    {
        for (;;)
        {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return closing || !jobs.empty(); });
                if (jobs.empty())
                    return;
                job = jobs.front();
                jobs.pop_front();
                busy = true;
            }

            if (latency.count() > 0)
                std::this_thread::sleep_for(latency);

            // Cheap content fingerprint: a sparse sum over the NV12 bytes.
            uint32_t checksum = 0;
            const auto& nv12 = host_buffers[job.buffer];
            for (size_t i = 0; i < nv12.size(); i += 97)
                checksum = checksum * 31u + nv12[i];

            EncodedUnit unit;
            unit.data = build_unit(job, checksum);
            // A V4L2 timeval holds microseconds; the stamp does not survive finer.
            unit.timestamp_ns = (job.timestamp_ns / 1000) * 1000;

            {
                std::lock_guard<std::mutex> lock(mutex);
                free_buffers.push_back(job.buffer);
                held.push_back(std::move(unit));
                while (held.size() > depth)
                {
                    ready.push_back(std::move(held.front()));
                    held.pop_front();
                }
                busy = false;
            }
            idle.notify_all();
        }
    }

    EncoderConfig config;
    uint32_t gop = 150;
    uint32_t depth = 2;
    std::chrono::milliseconds latency{ 0 };

    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    size_t d_y_pitch = 0;
    size_t d_uv_pitch = 0;

    std::vector<std::vector<uint8_t>> host_buffers;
    uint32_t submitted = 0;

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<uint32_t> free_buffers;
    std::deque<Job> jobs;
    std::deque<EncodedUnit> held; // encoded, still inside the warm-up window
    std::deque<EncodedUnit> ready; // what poll() hands out
    bool busy = false;
    bool closing = false;
    std::thread worker;
};

JetsonEncoder::JetsonEncoder(const EncoderConfig& config)
{
    if (config.width == 0 || config.height == 0)
        throw std::runtime_error("JetsonEncoder: width/height must be non-zero");
    if ((config.width % 2) != 0 || (config.height % 2) != 0)
        throw std::runtime_error("JetsonEncoder: width/height must be even for NV12");
    m_impl = std::make_unique<Impl>(config);
}

JetsonEncoder::~JetsonEncoder() = default;

bool JetsonEncoder::submit(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes, int64_t timestamp_ns)
{
    auto& impl = *m_impl;
    const auto& config = impl.config;

    uint32_t buffer = 0;
    {
        std::lock_guard<std::mutex> lock(impl.mutex);
        if (impl.free_buffers.empty())
            return false; // encoder still busy; drop this frame rather than block
        buffer = impl.free_buffers.front();
        impl.free_buffers.pop_front();
    }

    launch_rgba_to_nv12(reinterpret_cast<const uint8_t*>(rgba_device_ptr), static_cast<int>(row_pitch_bytes), impl.d_y,
                        static_cast<int>(impl.d_y_pitch), impl.d_uv, static_cast<int>(impl.d_uv_pitch),
                        static_cast<int>(config.width), static_cast<int>(config.height), config.full_range, nullptr);
    check_cuda(cudaGetLastError(), "rgba_to_nv12 launch");
    check_cuda(cudaStreamSynchronize(nullptr), "rgba_to_nv12 sync");

    uint8_t* host = impl.host_buffers[buffer].data();
    check_cuda(
        cudaMemcpy2D(host, config.width, impl.d_y, impl.d_y_pitch, config.width, config.height, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D(Y)");
    check_cuda(cudaMemcpy2D(host + static_cast<size_t>(config.width) * config.height, config.width, impl.d_uv,
                            impl.d_uv_pitch, config.width, config.height / 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy2D(UV)");

    {
        std::lock_guard<std::mutex> lock(impl.mutex);
        impl.jobs.push_back({ buffer, impl.submitted++, timestamp_ns });
    }
    impl.wake.notify_one();
    return true;
}

EncodedUnit JetsonEncoder::poll()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->ready.empty())
        return {};
    auto out = std::move(m_impl->ready.front());
    m_impl->ready.pop_front();
    return out;
}

EncodedUnit JetsonEncoder::end_of_stream()
{
    auto& impl = *m_impl;
    std::unique_lock<std::mutex> lock(impl.mutex);
    impl.idle.wait(lock, [&] { return impl.jobs.empty() && !impl.busy; });

    // The warm-up window empties at EOS, then the tail is concatenated, so it
    // carries the stamp of the last unit in it.
    EncodedUnit out;
    for (auto* queue : { &impl.ready, &impl.held })
    {
        for (auto& chunk : *queue)
        {
            out.data.insert(out.data.end(), chunk.data.begin(), chunk.data.end());
            out.timestamp_ns = chunk.timestamp_ns;
        }
        queue->clear();
    }
    return out;
}

} // namespace sensing
} // namespace plugins
