// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The other process of the camera integration tests.
//
//   camera_test_peer serve ENDPOINT          serves group "head": two gray 64x48 streams, 8 slots.
//                                            Every byte of frame set n is n modulo 256.
//   camera_test_peer hold ENDPOINT MARKER    reads "head" in order with a window of three, keeps
//                                            the three frame sets, then creates the file MARKER.
//
// Either runs until it is killed.

#include <camera_endpoint/endpoint.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_reader/camera_reader.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace core::camera;

namespace
{

constexpr std::uint32_t WIDTH = 64;
constexpr std::uint32_t HEIGHT = 48;
constexpr std::uint32_t SLOTS = 8;

[[noreturn]] void serve(const std::string& endpoint_path)
{
    GroupDescription group;
    group.id = "head";
    for (const char* eye : { "left", "right" })
    {
        StreamDescription stream;
        stream.id = std::string("head/") + eye;
        stream.pixel_format = PixelFormat::GRAY8;
        stream.width = WIDTH;
        stream.height = HEIGHT;
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = WIDTH;
        stream.slot_bytes = WIDTH * HEIGHT;
        stream.slot_count = SLOTS;
        stream.frame_rate = 200.0;
        group.streams.push_back(stream);
    }

    Publisher publisher;
    const GroupId head = publisher.add_group(group);
    HostPool left(SLOTS, WIDTH * HEIGHT, "peer_left");
    HostPool right(SLOTS, WIDTH * HEIGHT, "peer_right");
    Endpoint endpoint(publisher, endpoint_path);
    endpoint.set_group_memory(head, { left.descriptor(), right.descriptor() });
    endpoint.start();

    HostPool* pools[] = { &left, &right };
    for (std::uint64_t sequence = 1;; ++sequence)
    {
        std::vector<ImageInfo> images(2);
        bool complete = true;
        for (std::size_t stream = 0; stream < 2 && complete; ++stream)
        {
            const std::optional<std::uint32_t> slot = publisher.acquire_slot(head, stream);
            if (!slot)
            {
                for (std::size_t taken = 0; taken < stream; ++taken)
                {
                    publisher.abandon_slot(head, taken, images[taken].slot);
                }
                complete = false;
                break;
            }
            const std::span<std::byte> pixels = pools[stream]->slot(*slot);
            std::fill(pixels.begin(), pixels.end(), static_cast<std::byte>(sequence & 0xff));
            images[stream].slot = *slot;
        }
        if (complete)
        {
            // A dropped frame set takes no sequence number, so this stays the publisher's count.
            if (publisher.publish(head, images, {}) != sequence)
            {
                std::fprintf(stderr, "camera_test_peer: the publisher's sequence is not the peer's\n");
                ::_exit(4);
            }
        }
        else
        {
            --sequence;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

[[noreturn]] void hold(const std::string& endpoint_path, const std::string& marker)
{
    CameraReader reader =
        CameraReader::open("head", { .window = 3, .order = DeliveryOrder::IN_ORDER }, { .endpoint = endpoint_path });
    std::vector<FrameSet> held;
    while (held.size() < 3)
    {
        if (std::optional<FrameSet> frames = reader.next(std::chrono::seconds(5)))
        {
            held.push_back(std::move(*frames));
        }
    }
    std::ofstream(marker) << "holding\n";
    for (;;)
    {
        ::pause();
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "serve" && argc == 3)
        {
            serve(argv[2]);
        }
        if (mode == "hold" && argc == 4)
        {
            hold(argv[2], argv[3]);
        }
        std::fprintf(stderr, "usage: camera_test_peer serve ENDPOINT | hold ENDPOINT MARKER\n");
        return 2;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "camera_test_peer: %s\n", error.what());
        return 1;
    }
}
