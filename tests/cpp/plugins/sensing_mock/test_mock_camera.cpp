// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <core/sipl_camera.hpp>

#include <map>
#include <stdexcept>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

TEST_CASE("mock query renumbers sensors and honours the link masks", "[unit][gpu][sensing]")
{
    const std::string json = SENSING_PLATFORM_CONFIG;

    const auto both = SiplCamera::query(json, "SHW5G_2", { 0x0000, 0x1100 });
    REQUIRE(both.size() == 2);
    // The JSON names these sensors 2 and 3; the pipeline indices are 0 and 1.
    CHECK(both[0].id == 0);
    CHECK(both[1].id == 1);
    CHECK(both[0].name == "SHW5G_MIX_L2");
    CHECK(both[1].name == "SHW5G_MIX_L3");
    CHECK(both[0].width == 2560);
    CHECK(both[0].height == 1984);
    CHECK(both[0].fps == 60.0);

    // One link enabled leaves one sensor, which is renumbered to 0.
    const auto link2 = SiplCamera::query(json, "SHW5G_2", { 0x0000, 0x0100 });
    REQUIRE(link2.size() == 1);
    CHECK(link2[0].id == 0);
    CHECK(link2[0].name == "SHW5G_MIX_L2");

    const auto link3 = SiplCamera::query(json, "SHW5G_2", { 0x0000, 0x1000 });
    REQUIRE(link3.size() == 1);
    CHECK(link3[0].id == 0);
    CHECK(link3[0].name == "SHW5G_MIX_L3");

    // The mask for the wrong deserializer enables nothing.
    CHECK(SiplCamera::query(json, "SHW5G_2", { 0x1100, 0x0000 }).empty());
    CHECK(SiplCamera::query(json, "SHW5G_2", { 0x0000, 0x0000 }).empty());

    // No mask list keeps every module.
    CHECK(SiplCamera::query(json, "SHW5G_2", {}).size() == 2);
}

TEST_CASE("mock query rejects a missing file or config", "[unit][gpu][sensing]")
{
    CHECK_THROWS_AS(SiplCamera::query("/nonexistent/platform.json", "SHW5G_2", {}), std::runtime_error);
    CHECK_THROWS_AS(SiplCamera::query(SENSING_PLATFORM_CONFIG, "NO_SUCH_CONFIG", {}), std::runtime_error);
}

TEST_CASE("SENSING_MOCK_SCALE_DIV shrinks the reported geometry and keeps it even", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    const auto sensors = SiplCamera::query(SENSING_PLATFORM_CONFIG, "SHW5G_2", { 0x0000, 0x1100 });
    REQUIRE(sensors.size() == 2);
    CHECK(sensors[0].width == 320);
    CHECK(sensors[0].height == 248);
    CHECK(sensors[0].width % 2 == 0);
    CHECK(sensors[0].height % 2 == 0);
}

TEST_CASE("camera construction validates the masks", "[unit][gpu][sensing]")
{
    auto config = make_sipl_config();

    config.link_masks = {};
    CHECK_THROWS_AS(SiplCamera(config), std::runtime_error);

    config.link_masks = { 0x0000, 0x0000 };
    CHECK_THROWS_WITH(SiplCamera(config), Catch::Matchers::ContainsSubstring("no modules left"));
}

TEST_CASE("frames arrive whole, with rising sequence and stamps, and the picture moves", "[unit][gpu][sensing]")
{
    FastMock fast;
    SiplCamera camera(make_sipl_config());
    REQUIRE(camera.sensors().size() == 2);
    camera.start();

    for (const auto& sensor : camera.sensors())
    {
        INFO("sensor " << sensor.id);
        uint64_t last_sequence = 0;
        uint64_t last_stamp = 0;
        std::vector<Shot> shots;
        REQUIRE(wait_for(
            [&]
            {
                auto view = camera.latest(sensor.id);
                if (!view)
                    return false;
                CHECK(view->width == sensor.width);
                CHECK(view->height == sensor.height);
                CHECK(view->ptr != 0);
                CHECK(view->sequence > last_sequence);
                CHECK(view->timestamp_ns > last_stamp);
                last_sequence = view->sequence;
                last_stamp = view->timestamp_ns;

                Shot shot;
                shot.width = view->width;
                shot.height = view->height;
                shot.sequence = view->sequence;
                shot.rgba = download_rgba(view->ptr, view->pitch, view->width, view->height);
                CHECK(is_whole(shot));
                shots.push_back(std::move(shot));
                return shots.size() >= 8;
            }));

        // Eight captures span more than one 40 ms GIF frame, so the colour must have changed.
        bool changed = false;
        for (const auto& shot : shots)
        {
            const Rgb first = shots[0].at(shots[0].width / 2, shots[0].height / 2);
            const Rgb here = shot.at(shot.width / 2, shot.height / 2);
            changed = changed || !near(here, first.r, first.g, first.b, 30);
        }
        CHECK(changed);
    }
    camera.stop();
}

TEST_CASE("both sensors carry the same capture TSC for the same frame", "[unit][gpu][sensing]")
{
    FastMock fast;
    SiplCamera camera(make_sipl_config());
    camera.start();

    std::map<uint64_t, uint64_t> tsc[2];
    REQUIRE(wait_for(
        [&]
        {
            for (uint32_t id = 0; id < 2; ++id)
            {
                if (auto view = camera.latest(id))
                {
                    // Not on the CLOCK_MONOTONIC timebase, so nothing can confuse the two.
                    CHECK(view->capture_tsc_ns != view->timestamp_ns);
                    tsc[id][view->sequence] = view->capture_tsc_ns;
                }
            }
            int common = 0;
            for (const auto& [sequence, value] : tsc[0])
            {
                auto other = tsc[1].find(sequence);
                if (other != tsc[1].end())
                {
                    CHECK(other->second == value);
                    ++common;
                }
            }
            return common >= 3;
        }));
    camera.stop();
}

TEST_CASE("a leased slot is not overwritten while the producer keeps running", "[unit][gpu][sensing]")
{
    FastMock fast;
    SiplCamera camera(make_sipl_config());
    camera.start();

    std::optional<FrameView> held;
    REQUIRE(wait_for([&] { return (held = camera.latest(0)).has_value(); }));
    const auto before = download_rgba(held->ptr, held->pitch, held->width, held->height);

    // The GIF changes colour every 40 ms and the other two slots are rewritten on every capture. If
    // the lease failed, the held slot would soon show a different colour.
    const auto end = std::chrono::steady_clock::now() + 250ms;
    while (std::chrono::steady_clock::now() < end)
    {
        CHECK(download_rgba(held->ptr, held->pitch, held->width, held->height) == before);
        std::this_thread::sleep_for(5ms);
    }

    // The producer did keep going while the slot was held.
    uint64_t latest_sequence = held->sequence;
    REQUIRE(wait_for(
        [&]
        {
            if (auto view = camera.latest(1))
                latest_sequence = std::max<uint64_t>(latest_sequence, view->sequence);
            return latest_sequence >= held->sequence + 5;
        }));
    camera.stop();
}

TEST_CASE("an injected link failure surfaces from latest()", "[unit][gpu][sensing]")
{
    FastMock fast;
    ScopedEnv fail("SENSING_MOCK_FAIL_AFTER", "3");
    SiplCamera camera(make_sipl_config());
    camera.start();

    std::string message;
    REQUIRE(wait_for(
        [&]
        {
            try
            {
                camera.latest(0);
            }
            catch (const std::runtime_error& e)
            {
                message = e.what();
                return true;
            }
            return false;
        }));
    CHECK(message.find("capture error") != std::string::npos);
    camera.stop();
}

TEST_CASE("an unknown sensor index is an error, and stop is idempotent", "[unit][gpu][sensing]")
{
    FastMock fast;
    SiplCamera camera(make_sipl_config());
    CHECK_THROWS_AS(camera.latest(7), std::runtime_error);
    camera.start();
    camera.start();
    camera.stop();
    camera.stop();
}
