// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The endpoint, the wire messages and the reader together, with the two ends in different
// processes: what a camera plugin and its readers are.

#include "endpoint_directory.hpp"

#include <camera/camera_reader.hpp>
#include <camera/endpoint_path.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sys/prctl.h>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace core::camera;
using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_peer_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    std::string file(const std::string& name) const
    {
        return (m_path / name).string();
    }

private:
    std::filesystem::path m_path;
};

// camera_test_peer as a process of its own; killed when the test is done with it.
class Peer
{
public:
    explicit Peer(const std::vector<std::string>& arguments)
    {
        std::vector<std::string> owned = { CAMERA_TEST_PEER_EXECUTABLE };
        owned.insert(owned.end(), arguments.begin(), arguments.end());
        std::vector<char*> argv;
        for (std::string& argument : owned)
        {
            argv.push_back(argument.data());
        }
        argv.push_back(nullptr);
        m_pid = ::fork();
        REQUIRE(m_pid >= 0);
        if (m_pid == 0)
        {
            // A test process that dies without running ~Peer takes its peers with it.
            ::prctl(PR_SET_PDEATHSIG, SIGKILL);
            ::execv(argv[0], argv.data());
            ::_exit(127);
        }
    }

    ~Peer()
    {
        kill();
    }

    // Ends the process the way a crash does: no chance to clean up.
    void kill()
    {
        if (m_pid > 0)
        {
            ::kill(m_pid, SIGKILL);
            ::waitpid(m_pid, nullptr, 0);
            m_pid = -1;
        }
    }

private:
    pid_t m_pid = -1;
};

bool wait_until(const std::function<bool()>& condition, std::chrono::seconds timeout = 10s)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

// A peer that was just started needs a moment before it serves.
std::optional<CameraReader> open_when_ready(const std::string& group,
                                            const ReaderOptions& options = {},
                                            const PluginLocation& plugin = {})
{
    std::optional<CameraReader> reader;
    wait_until(
        [&]
        {
            try
            {
                reader = CameraReader::open(group, options, plugin);
            }
            catch (const CameraReaderError&)
            {
            }
            return reader.has_value();
        });
    return reader;
}

// Every byte of both images of frame set n is n modulo 256.
void require_whole(const FrameSet& frames)
{
    const auto expected = static_cast<std::byte>(frames.sequence() & 0xff);
    for (std::size_t stream = 0; stream < frames.image_count(); ++stream)
    {
        const auto pixels = frames.image(stream).data;
        REQUIRE(std::all_of(pixels.begin(), pixels.end(), [expected](std::byte value) { return value == expected; }));
    }
}

} // namespace

TEST_CASE("a reader gets whole frame sets from a plugin in another process", "[integration][camera]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.file("cameras.sock");
    Peer plugin({ "serve", endpoint });

    std::optional<CameraReader> reader = open_when_ready("head", {}, { .endpoint = endpoint });
    REQUIRE(reader.has_value());
    REQUIRE(reader->group().streams.size() == 2);

    std::uint64_t previous = 0;
    for (int i = 0; i < 50; ++i)
    {
        const std::optional<FrameSet> frames = reader->next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->image_count() == 2);
        // The image is the frame set the event named, and all of it: the slot was not rewritten
        // between the plugin sending the event and this process reading the pixels.
        require_whole(*frames);
        if (previous != 0)
        {
            REQUIRE(frames->sequence() == previous + 1 + frames->skipped());
        }
        previous = frames->sequence();
    }

    // A frame set that is held stays whole while the plugin goes through its pool many times.
    const std::optional<FrameSet> held = reader->next(2s);
    REQUIRE(held.has_value());
    std::this_thread::sleep_for(200ms);
    require_whole(*held);
}

TEST_CASE("a reader that was away is handed a fresh frame set by a plugin in another process",
          "[integration][camera][newest]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.file("cameras.sock");
    Peer plugin({ "serve", endpoint });
    std::optional<CameraReader> reader = open_when_ready("head", {}, { .endpoint = endpoint });
    REQUIRE(reader.has_value());
    REQUIRE(reader->next(2s).has_value()); // taken and given back at once

    const auto now_ns = []
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    for (int round = 0; round < 5; ++round)
    {
        // Away for sixty frame periods of the peer, holding nothing.
        const std::int64_t left_ns = now_ns();
        std::this_thread::sleep_for(300ms);
        const std::optional<FrameSet> frames = reader->next(2s);
        REQUIRE(frames.has_value());

        // A frame set held for the reader when it left would have been published before it left.
        // This one was published well into the time it was away. Neither bound depends on how
        // soon this process runs again after next() returns.
        const std::int64_t published_ns = frames->timeline().published_ns;
        REQUIRE(published_ns > left_ns + std::chrono::nanoseconds(150ms).count());
        REQUIRE(published_ns <= now_ns());
        REQUIRE(frames->skipped() > 0);
        require_whole(*frames);
    }
}

TEST_CASE("a reader process that is killed gives back what it held", "[integration][camera]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.file("cameras.sock");
    const std::string marker = directory.file("holding");
    Peer plugin({ "serve", endpoint });
    REQUIRE(open_when_ready("head", {}, { .endpoint = endpoint }).has_value());

    // Another process reads in order with a window of three and keeps all three frame sets.
    Peer holder({ "hold", endpoint, marker });
    REQUIRE(wait_until([&] { return std::filesystem::exists(marker); }));

    // While it lives, its share of the pool is taken: a reader that needs that room is refused.
    const ReaderOptions wide{ .window = 2, .order = DeliveryOrder::NEWEST };
    REQUIRE_THROWS_WITH(
        CameraReader::open("head", wide, { .endpoint = endpoint }), ContainsSubstring("does not fit the pool"));

    // Killed, it returns nothing itself. The plugin notices and takes everything back.
    holder.kill();
    std::optional<CameraReader> reader = open_when_ready("head", wide, { .endpoint = endpoint });
    REQUIRE(reader.has_value());
    for (int i = 0; i < 40; ++i)
    {
        const std::optional<FrameSet> frames = reader->next(2s);
        REQUIRE(frames.has_value());
        require_whole(*frames);
    }
}

TEST_CASE("a reader is told when the plugin's process is killed", "[integration][camera]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.file("cameras.sock");
    Peer plugin({ "serve", endpoint });
    // A window of two, so that the reader can hold one frame set and still ask for another.
    std::optional<CameraReader> reader =
        open_when_ready("head", { .window = 2, .order = DeliveryOrder::NEWEST }, { .endpoint = endpoint });
    REQUIRE(reader.has_value());
    std::optional<FrameSet> held = reader->next(2s);
    REQUIRE(held.has_value());
    const std::uint64_t sequence = held->sequence();

    plugin.kill();
    REQUIRE_THROWS_AS(
        [&]
        {
            for (int i = 0; i < 1000; ++i)
            {
                (void)reader->next(2s);
            }
        }(),
        CameraDisconnected);

    // What the reader holds is still its to read: the memory outlives the process that made it.
    REQUIRE(held->sequence() == sequence);
    require_whole(*held);
    held.reset(); // giving it back to nobody is harmless

    // The killed plugin left its socket file behind; nothing answers there.
    REQUIRE_THROWS_WITH(
        CameraReader::open("head", {}, { .endpoint = endpoint }), ContainsSubstring("no camera plugin answers on"));
}

TEST_CASE("a reader finds each camera on the plugin process that publishes it", "[integration][camera]")
{
    const EndpointDirectory endpoints;
    Peer head_plugin({ "serve", default_endpoint_path("first"), "head" });
    Peer wrist_plugin({ "serve", default_endpoint_path("second"), "wrist" });

    // The reader is told the name of each camera group and nothing about where it is served.
    std::optional<CameraReader> head = open_when_ready("head");
    std::optional<CameraReader> wrist = open_when_ready("wrist");
    REQUIRE(head.has_value());
    REQUIRE(wrist.has_value());
    REQUIRE(head->group().streams[0].id == "head/left");
    REQUIRE(wrist->group().streams[0].id == "wrist/left");
    for (CameraReader* reader : { &*head, &*wrist })
    {
        const std::optional<FrameSet> frames = reader->next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->image_count() == 2);
        require_whole(*frames);
    }

    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "head");
    REQUIRE(groups[0].endpoint == default_endpoint_path("first"));
    REQUIRE(groups[1].group.id == "wrist");
    REQUIRE(groups[1].endpoint == default_endpoint_path("second"));

    // One plugin is killed and leaves its socket file behind. Its reader is told, and the other
    // camera is found and read as before.
    head_plugin.kill();
    REQUIRE(std::filesystem::exists(default_endpoint_path("first")));
    REQUIRE_THROWS_AS(
        [&]
        {
            for (int i = 0; i < 1000; ++i)
            {
                (void)head->next(2s);
            }
        }(),
        CameraDisconnected);
    REQUIRE_THROWS_WITH(
        CameraReader::open("head"), ContainsSubstring("publishes group 'head'") &&
                                        ContainsSubstring("'wrist' by '" + default_endpoint_path("second") + "'"));
    CameraReader again = CameraReader::open("wrist");
    const std::optional<FrameSet> frames = again.next(2s);
    REQUIRE(frames.has_value());
    require_whole(*frames);
}
