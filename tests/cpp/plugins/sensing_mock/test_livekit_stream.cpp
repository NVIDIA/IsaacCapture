// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The SENSING plugin publishing to a real LiveKit server (a local livekit-server process) and a
// second participant decoding what arrives. Set LIVEKIT_SERVER to the server binary to run these.

#include "livekit_test_support.hpp"
#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <core/frame_sink.hpp>
#include <core/sensing_camera.hpp>

#include <memory>
#include <optional>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace livekit_test;
using namespace std::chrono_literals;

#define REQUIRE_SERVER()                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!LocalServer::binary())                                                                                    \
            SKIP("set LIVEKIT_SERVER to a livekit-server binary to run this test");                                    \
    } while (0)

namespace
{

constexpr const char* kRoom = "sensing-room";
constexpr uint64_t kMockTscBaseNs = 1'000'000'000'000'000ull;

SensingConfig stream_config(const std::string& url,
                            uint32_t scale_div,
                            uint32_t fps = 30,
                            const std::string& secret = kApiSecret)
{
    SensingConfig config;
    config.platform_config_json = SENSING_PLATFORM_CONFIG;
    config.bitrate_bps = 8'000'000;
    config.livekit.url = url;
    config.livekit.token = make_token("sensing-publisher", kRoom, true, secret);
    config.livekit.fps = fps;
    config.livekit.scale_div = scale_div;
    return config;
}

std::vector<StreamConfig> both_sensors(const std::string& left = "sensing-left",
                                       const std::string& right = "sensing-right")
{
    std::vector<StreamConfig> streams(2);
    streams[0].sensor_id = 0;
    streams[0].livekit_track = left;
    streams[1].sensor_id = 1;
    streams[1].livekit_track = right;
    return streams;
}

std::unique_ptr<SensingCamera> make_camera(const SensingConfig& config, const std::vector<StreamConfig>& streams)
{
    return std::make_unique<SensingCamera>(config, streams, std::make_unique<FrameSink>(streams));
}

// One of black, red, blue or white, allowing for what H.264 does to a flat colour.
bool is_cycle_colour_lossy(const Rgb& c)
{
    return near(c, 0, 0, 0, 50) || near(c, 255, 0, 0, 50) || near(c, 0, 0, 255, 50) || near(c, 255, 255, 255, 50);
}

} // namespace

TEST_CASE("two sensors reach a real server as named tracks and decode correctly", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    LocalServer server;
    FastMock fast; // 320x248 sensors playing a black, red, blue, white cycle
    auto camera = make_camera(stream_config(server.url(), 2), both_sensors());
    UpdateLoop loop(*camera);

    Subscriber subscriber(server.url(), kRoom);
    REQUIRE(subscriber.wait_for_frames({ "sensing-left", "sensing-right" }, 25, 30s));

    CHECK(subscriber.track_names() == std::set<std::string>{ "sensing-left", "sensing-right" });
    for (const char* name : { "sensing-left", "sensing-right" })
    {
        INFO(name);
        const TrackSeen seen = subscriber.seen(name);
        // 320x248 reduced by 2.
        CHECK(seen.width == 160);
        CHECK(seen.height == 124);
        REQUIRE(seen.last_rgba.size() == 160u * 124u * 4u);
        Shot shot;
        shot.width = 160;
        shot.height = 124;
        shot.rgba = seen.last_rgba;
        CHECK(is_cycle_colour_lossy(shot.at(80, 62)));

        // The camera's own frame ids and capture clock survive to the viewer.
        REQUIRE(seen.frame_ids.size() >= 20);
        for (size_t i = 1; i < seen.frame_ids.size(); ++i)
            CHECK(seen.frame_ids[i] > seen.frame_ids[i - 1]);
        CHECK(seen.with_user_timestamp >= seen.frames - 2);
        REQUIRE_FALSE(seen.tsc_by_frame.empty());
        for (const auto& [id, tsc] : seen.tsc_by_frame)
            CHECK(tsc == kMockTscBaseNs + static_cast<uint64_t>((id - 1) * (1e9 / 60.0)));
    }
}

TEST_CASE("a track is published at the sensor's native geometry by default", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    LocalServer server;
    ScopedEnv fps("SENSING_MOCK_FPS", "15"); // the bundled GIF at the real 2560x1984, at a rate any machine can encode
    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 0;
    streams[0].livekit_track = "native";
    auto camera = make_camera(stream_config(server.url(), 1, 15), streams);
    UpdateLoop loop(*camera);

    Subscriber subscriber(server.url(), kRoom);
    REQUIRE(subscriber.wait_for_frames({ "native" }, 5, 60s));
    const TrackSeen seen = subscriber.seen("native");
    CHECK(seen.width == 2560);
    CHECK(seen.height == 1984);
}

TEST_CASE("a viewer can join late, leave and join again", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    LocalServer server;
    FastMock fast;
    auto camera = make_camera(stream_config(server.url(), 2), both_sensors());
    UpdateLoop loop(*camera);

    {
        Subscriber first(server.url(), kRoom);
        REQUIRE(first.wait_for_frames({ "sensing-left", "sensing-right" }, 10, 30s));
    }
    Subscriber second(server.url(), kRoom);
    REQUIRE(second.wait_for_frames({ "sensing-left", "sensing-right" }, 10, 30s));
}

TEST_CASE("losing the server ends capture with a clear error", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    auto server = std::make_unique<LocalServer>();
    FastMock fast;
    auto camera = make_camera(stream_config(server->url(), 2), both_sensors());
    {
        Subscriber subscriber(server->url(), kRoom);
        // Poll as the plugin's main loop does, on this thread so a failure can be caught.
        const auto warm = std::chrono::steady_clock::now() + 20s;
        bool seen = false;
        while (!seen && std::chrono::steady_clock::now() < warm)
        {
            camera->update();
            seen = subscriber.wait_for_frames({ "sensing-left" }, 5, 5ms);
        }
        REQUIRE(seen);
    }

    server->stop();
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    while (error.empty() && std::chrono::steady_clock::now() - start < 90s)
    {
        try
        {
            camera->update();
        }
        catch (const std::runtime_error& e)
        {
            error = e.what();
        }
        std::this_thread::sleep_for(5ms);
    }
    INFO("took " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count()
                 << " s");
    REQUIRE_FALSE(error.empty());
    CHECK(error.find("LiveKit") != std::string::npos);
}

TEST_CASE("a token signed with the wrong secret is refused", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    LocalServer server;
    FastMock fast;
    CHECK_THROWS_WITH(make_camera(stream_config(server.url(), 2, 30, "not-the-secret"), both_sensors()),
                      Catch::Matchers::ContainsSubstring("cannot join"));
}

TEST_CASE("a server that is not there is an error, not a hang", "[unit][gpu][sensing][livekit]")
{
    FastMock fast;
    const auto start = std::chrono::steady_clock::now();
    CHECK_THROWS_WITH(make_camera(stream_config("ws://127.0.0.1:9", 2), both_sensors()),
                      Catch::Matchers::ContainsSubstring("cannot join"));
    CHECK(std::chrono::steady_clock::now() - start < 60s);
}

TEST_CASE("stopping the plugin unpublishes and leaves the room", "[unit][gpu][sensing][livekit][server]")
{
    REQUIRE_SERVER();
    LocalServer server;
    FastMock fast;
    Subscriber subscriber(server.url(), kRoom);
    {
        auto camera = make_camera(stream_config(server.url(), 2), both_sensors());
        UpdateLoop loop(*camera);
        REQUIRE(subscriber.wait_for_frames({ "sensing-left", "sensing-right" }, 10, 30s));

        // Destroy in the plugin's order: the loop stops, then the camera and its tracks and room.
    }
    const auto start = std::chrono::steady_clock::now();
    while (subscriber.participants_left() == 0 && std::chrono::steady_clock::now() - start < 10s)
        std::this_thread::sleep_for(50ms);
    CHECK(subscriber.participants_left() >= 1);
}

TEST_CASE("bad LiveKit settings fail before any connection is made", "[unit][gpu][sensing][livekit]")
{
    FastMock fast;
    // The URL points nowhere: these must be refused without trying it, so they return at once. The
    // sensors here are 320x248: 3 does not divide them, and 8 leaves 40x31, which NV12 cannot hold.
    const auto start = std::chrono::steady_clock::now();
    CHECK_THROWS_WITH(make_camera(stream_config("ws://127.0.0.1:9", 2), both_sensors("same", "same")),
                      Catch::Matchers::ContainsSubstring("used twice"));
    CHECK_THROWS_WITH(make_camera(stream_config("ws://127.0.0.1:9", 3), both_sensors()),
                      Catch::Matchers::ContainsSubstring("exactly"));
    CHECK_THROWS_WITH(
        make_camera(stream_config("ws://127.0.0.1:9", 8), both_sensors()), Catch::Matchers::ContainsSubstring("even"));
    CHECK(std::chrono::steady_clock::now() - start < 5s);
}
