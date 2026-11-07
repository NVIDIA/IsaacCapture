// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Animated RGBA test pattern on the CUDA IPC socket, with no camera attached.
//
// Lets the consumer side be developed and tested with no camera attached. Also
// the quickest way to tell a broken consumer from a broken camera: if the
// pattern animates here and the camera does not, the fault is upstream of the
// IPC.
//
//   ./sensing_ipc_testsrc --socket=/tmp/sensing0.sock --width=1920 --height=1080
//   ./camera_viz.sh run configs/cuda_ipc.yaml

#include <sensing_cuda_ipc/cuda_ipc_publisher.hpp>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <thread>

namespace
{

std::sig_atomic_t volatile g_stop = 0;

void on_signal(int)
{
    g_stop = 1;
}

/// Scrolling colour bars plus a box tracking the frame counter, so a stale or
/// torn frame is obvious on screen rather than merely plausible.
__global__ void test_pattern(uint8_t* out, int width, int height, size_t pitch, float t, unsigned frame)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const float u = static_cast<float>(x) / width;
    const float v = static_cast<float>(y) / height;

    uint8_t r = static_cast<uint8_t>(255.0f * fminf(1.0f, fmaxf(0.0f, 0.5f + 0.5f * __sinf(6.2831f * (u + t)))));
    uint8_t g = static_cast<uint8_t>(255.0f * v);
    uint8_t b = static_cast<uint8_t>(255.0f * fminf(1.0f, fmaxf(0.0f, 0.5f + 0.5f * __cosf(6.2831f * (v - t)))));

    // A white square orbiting the centre: any duplicated frame freezes it.
    const float cx = 0.5f + 0.3f * __cosf(6.2831f * t);
    const float cy = 0.5f + 0.3f * __sinf(6.2831f * t);
    if (fabsf(u - cx) < 0.04f && fabsf(v - cy) < 0.04f * width / height)
    {
        r = g = b = 255;
    }

    // Top-left binary readout of the frame counter, 16 bits, 24px cells.
    if (y < 24 && x < 16 * 24)
    {
        const unsigned bit = static_cast<unsigned>(x / 24);
        const bool on = (frame >> (15u - bit)) & 1u;
        r = g = b = on ? 255 : 0;
    }

    uint8_t* px = out + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * 4;
    px[0] = r;
    px[1] = g;
    px[2] = b;
    px[3] = 255;
}

uint64_t monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

bool match(const std::string& arg, const char* key, std::string& value)
{
    const std::string prefix = std::string("--") + key + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;
    value = arg.substr(prefix.size());
    return true;
}

} // namespace

int main(int argc, char** argv)
try
{
    plugins::sensing::CudaIpcConfig config;
    config.socket_path = "/tmp/sensing_cuda0.sock";
    double fps = 30.0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        std::string value;
        if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [--socket=PATH] [--width=N] [--height=N]\n"
                      << "       [--fps=N] [--sensor=N] [--gpu-id=N] [--slots=N]\n";
            return 0;
        }
        else if (match(arg, "socket", value))
            config.socket_path = value;
        else if (match(arg, "width", value))
            config.width = static_cast<uint32_t>(std::stoul(value));
        else if (match(arg, "height", value))
            config.height = static_cast<uint32_t>(std::stoul(value));
        else if (match(arg, "sensor", value))
            config.sensor_id = static_cast<uint32_t>(std::stoul(value));
        else if (match(arg, "gpu-id", value))
            config.gpu_id = std::stoi(value);
        else if (match(arg, "slots", value))
            config.slot_count = static_cast<uint32_t>(std::stoul(value));
        else if (match(arg, "fps", value))
            fps = std::stod(value);
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }
    if (fps <= 0.0)
        fps = 30.0;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    plugins::sensing::CudaIpcPublisher publisher(config);

    // Scratch frame standing in for the camera's converted RGBA output, so the
    // publisher's copy path is exercised exactly as it is in the plugin.
    uint8_t* scratch = nullptr;
    size_t scratch_pitch = 0;
    if (cudaMallocPitch(&scratch, &scratch_pitch, static_cast<size_t>(config.width) * 4, config.height) != cudaSuccess)
        throw std::runtime_error("cudaMallocPitch failed");

    const dim3 block(16, 16);
    const dim3 grid((config.width + block.x - 1) / block.x, (config.height + block.y - 1) / block.y);
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / fps));

    std::cout << "Test source running at " << fps << " fps. Ctrl+C to stop." << std::endl;

    auto next = std::chrono::steady_clock::now();
    auto last_report = std::chrono::steady_clock::now();
    unsigned frame = 0;

    while (!g_stop)
    {
        publisher.poll();

        if (publisher.has_consumer())
        {
            const float t = static_cast<float>(frame % 120) / 120.0f;
            test_pattern<<<grid, block>>>(scratch, config.width, config.height, scratch_pitch, t, frame);
            if (cudaGetLastError() != cudaSuccess)
                throw std::runtime_error("test_pattern launch failed");
            cudaStreamSynchronize(0);
            publisher.publish(reinterpret_cast<uintptr_t>(scratch), scratch_pitch, monotonic_ns());
            ++frame;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_report >= std::chrono::seconds(5))
        {
            std::cout << "  published=" << publisher.published_count() << " dropped=" << publisher.dropped_count()
                      << (publisher.has_consumer() ? " (consumer attached)" : " (waiting for consumer)") << std::endl;
            last_report = now;
        }

        next += period;
        if (next > now)
            std::this_thread::sleep_until(next);
        else
            next = now; // Fell behind; do not spiral trying to catch up.
    }

    std::cout << "\nStopped. published=" << publisher.published_count() << " dropped=" << publisher.dropped_count()
              << std::endl;
    cudaFree(scratch);
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
