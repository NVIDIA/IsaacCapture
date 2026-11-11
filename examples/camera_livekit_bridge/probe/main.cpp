// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Connects to camera_receiver's CUDA IPC socket and prints per-frame metadata with a small checksum of the frame's
// first pixels, to show that frames reach this process without leaving the GPU.

#include <ipc_client/cuda_ipc_client.hpp>

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

namespace
{

std::uint64_t first_row_checksum(const camera_bridge::IpcSlot& slot)
{
    std::vector<unsigned char> host(std::min<std::size_t>(4096, slot.nbytes));
    if (host.empty() || cudaMemcpy(host.data(), slot.device_ptr, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;
    std::uint64_t sum = 0;
    for (const unsigned char byte : host)
        sum += byte;
    return sum;
}

} // namespace

int main(int argc, char** argv)
{
    std::string socket_path = "/tmp/camera_receiver_avs1_cuda_ipc.sock";
    std::vector<std::string> streams;
    int frames = 60;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--socket" && i + 1 < argc)
            socket_path = argv[++i];
        else if (arg == "--stream" && i + 1 < argc)
            streams.emplace_back(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc)
            frames = std::atoi(argv[++i]);
        else
        {
            std::cerr << "usage: camera_ipc_probe [--socket PATH] [--stream NAME]... [--frames N]\n";
            return 2;
        }
    }

    try
    {
        camera_bridge::CudaIpcClient client(socket_path, streams);
        client.connect();
        for (const auto& stream : client.streams())
        {
            std::cout << "stream " << stream.name << " " << stream.width << "x" << stream.height << " "
                      << stream.pixel_format << " slots=" << stream.slots.size() << " layout=" << stream.layout.dump()
                      << "\n";
        }
        camera_bridge::FrameNotice notice;
        for (int count = 0; count < frames;)
        {
            if (!client.next_frame(notice, std::chrono::seconds(5)))
            {
                std::cerr << "no frame for 5 s\n";
                return 1;
            }
            const auto& slot = client.slot(notice);
            const auto& meta = notice.metadata;
            std::cout << notice.stream << " slot=" << notice.slot << " gen=" << notice.slot_generation
                      << " stream_sequence=" << meta.value("stream_sequence", -1)
                      << " session=" << meta.value("session_id", std::uint64_t{ 0 })
                      << " sensor_sof_ns=" << (meta.contains("sensor_sof_ns") ? meta["sensor_sof_ns"].dump() : "null")
                      << " checksum=" << first_row_checksum(slot) << "\n";
            ++count;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
