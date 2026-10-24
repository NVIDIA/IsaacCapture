// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gif_fixture.hpp"
#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <core/mock_gif_source.hpp>
#include <core/sipl_camera.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <optional>
#include <stdexcept>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

bool is_white(const Rgb& c)
{
    return near(c, 255, 255, 255, 14);
}

// Number of consecutive white pixels along a row or column starting at (x, y).
uint32_t white_run_x(const Shot& shot, uint32_t y)
{
    uint32_t count = 0;
    for (uint32_t x = 0; x < shot.width; ++x)
        count += is_white(shot.at(x, y)) ? 1 : 0;
    return count;
}

uint32_t white_run_y(const Shot& shot, uint32_t x)
{
    uint32_t count = 0;
    for (uint32_t y = 0; y < shot.height; ++y)
        count += is_white(shot.at(x, y)) ? 1 : 0;
    return count;
}

} // namespace

TEST_CASE("GIF frame ends clamp tiny delays to 20 ms and accumulate", "[unit][sensing]")
{
    const auto ends = gif_frame_ends({ 0, 10, 100, 20 });
    REQUIRE(ends.size() == 4);
    CHECK(ends[0] == 0.02);
    CHECK(ends[1] == 0.04);
    CHECK(ends[2] == 0.14);
    CHECK(ends[3] == 0.16);
}

TEST_CASE("GIF frame index follows elapsed time and loops", "[unit][sensing]")
{
    const auto ends = gif_frame_ends({ 100, 100, 200 }); // 0.1, 0.2, 0.4
    CHECK(gif_frame_index(ends, 0.0) == 0);
    CHECK(gif_frame_index(ends, 0.099) == 0);
    CHECK(gif_frame_index(ends, 0.1) == 1);
    CHECK(gif_frame_index(ends, 0.25) == 2);
    CHECK(gif_frame_index(ends, 0.399) == 2);
    CHECK(gif_frame_index(ends, 0.41) == 0); // loops
    CHECK(gif_frame_index(ends, 0.65) == 2);
    CHECK(gif_frame_index({}, 1.0) == 0);
}

TEST_CASE("a GIF that is missing or is not a GIF fails construction", "[unit][gpu][sensing]")
{
    CHECK_THROWS_WITH(MockGifSource("/nonexistent/none.gif"), Catch::Matchers::ContainsSubstring("cannot open"));

    TempFile text(".gif");
    std::ofstream(text.path()) << "this is not a GIF";
    CHECK_THROWS_WITH(MockGifSource(text.path()), Catch::Matchers::ContainsSubstring("cannot decode"));

    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv gif("SENSING_MOCK_GIF", "/nonexistent/none.gif");
    CHECK_THROWS_AS(SiplCamera(make_sipl_config()), std::runtime_error);
}

TEST_CASE("the camera shows the GIF frame for its capture time while sequence and stamps keep advancing",
          "[unit][gpu][sensing]")
{
    TempFile gif(".gif");
    // Delays are in centiseconds: 4 is 40 ms.
    write_gif(gif.path(), 16, 16, { solid_frame(16, 16, kGifRed, 4), solid_frame(16, 16, kGifBlue, 4) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());
    ScopedEnv fps("SENSING_MOCK_FPS", "50"); // 20 ms per capture: two captures per GIF frame

    SiplCamera camera(make_sipl_config());
    camera.start();

    std::vector<Shot> shots;
    uint64_t last_sequence = 0;
    REQUIRE(wait_for(
        [&]
        {
            if (auto view = camera.latest(0))
            {
                Shot shot;
                shot.width = view->width;
                shot.height = view->height;
                shot.sequence = view->sequence;
                shot.rgba = download_rgba(view->ptr, view->pitch, view->width, view->height);
                CHECK(shot.sequence > last_sequence);
                last_sequence = shot.sequence;
                shots.push_back(std::move(shot));
            }
            return shots.size() >= 14;
        }));
    camera.stop();

    // Capture n shows GIF time n / 50 s: red until 0.04 s, blue until 0.08 s, then red again.
    int repeats = 0;
    int red = 0;
    int blue = 0;
    for (size_t i = 0; i < shots.size(); ++i)
    {
        const double t = (shots[i].sequence - 1) / 50.0;
        const double phase = std::fmod(t, 0.08);
        // A capture exactly on a frame boundary could land on either side by rounding.
        if (std::abs(phase - 0.04) < 1e-6 || phase < 1e-6 || phase > 0.08 - 1e-6)
            continue;
        const bool expect_red = phase < 0.04;
        const Rgb centre = shots[i].at(shots[i].width / 2, shots[i].height / 2);
        INFO("capture " << shots[i].sequence << " t=" << t);
        CHECK((expect_red ? near(centre, 255, 0, 0) : near(centre, 0, 0, 255)));
        (expect_red ? red : blue)++;
        if (i > 0 && near(centre, 255, 0, 0) == near(shots[i - 1].at(0, 0), 255, 0, 0))
            ++repeats;
    }
    CHECK(red > 0);
    CHECK(blue > 0);
    // The same image on consecutive captures is normal; it must not stall the sequence.
    CHECK(repeats > 0);
}

TEST_CASE("an enlarged GIF keeps its proportions on the sensor's geometry", "[unit][gpu][sensing]")
{
    // 32x32 with a centred 16x16 square, shown on 320x248: scale 10 fills the width, the height is
    // cropped from 320 to 248, so the square stays square.
    TempFile gif(".gif");
    write_gif(gif.path(), 32, 32, { square_frame(32, 32, 16, 100) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());

    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot shot = grab(camera, 0);
    camera.stop();

    REQUIRE(shot.width == 320);
    REQUIRE(shot.height == 248);
    const uint32_t cx = shot.width / 2;
    const uint32_t cy = shot.height / 2;
    CHECK(is_white(shot.at(cx, cy)));
    CHECK(near(shot.at(30, cy), 0, 0, 0));
    CHECK(near(shot.at(cx, 10), 0, 0, 0));
    // 16 px at scale 10 is about 160 px both ways, a little less where bilinear softens the edge.
    // Equal extents in x and y mean no independent stretch.
    const int run_x = static_cast<int>(white_run_x(shot, cy));
    const int run_y = static_cast<int>(white_run_y(shot, cx));
    CHECK(std::abs(run_x - 160) <= 14);
    CHECK(std::abs(run_x - run_y) <= 3);
}

TEST_CASE("a wide GIF is reduced and cropped to the sensor's aspect", "[unit][gpu][sensing]")
{
    // 640x320 (2:1), left red and right blue, on 320x248 (about 1.29:1): scale 0.775 makes the image
    // 496x248, the sides are cropped, and the red/blue boundary stays in the middle.
    TempFile gif(".gif");
    write_gif(gif.path(), 640, 320, { split_frame(640, 320, kGifRed, kGifBlue, 100) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());

    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot shot = grab(camera, 0);
    camera.stop();

    REQUIRE(shot.width == 320);
    REQUIRE(shot.height == 248);
    CHECK(near(shot.at(60, 124), 255, 0, 0));
    CHECK(near(shot.at(150, 124), 255, 0, 0));
    CHECK(near(shot.at(172, 124), 0, 0, 255));
    CHECK(near(shot.at(260, 124), 0, 0, 255));
}

TEST_CASE("a tall GIF is cropped top and bottom", "[unit][gpu][sensing]")
{
    // 100x400 (1:4) on 320x248: scale 3.2 gives 320x1280, so only the middle 248 rows show.
    // The GIF is white in the middle band and red elsewhere, so the visible window is all white.
    TempFile gif(".gif");
    GifFrame frame{ std::vector<uint8_t>(100 * 400, kGifRed), 100 };
    for (int y = 150; y < 250; ++y)
        std::fill_n(frame.pixels.begin() + y * 100, 100, kGifWhite);
    write_gif(gif.path(), 100, 400, { frame });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());

    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot shot = grab(camera, 0);
    camera.stop();

    REQUIRE(shot.height == 248);
    CHECK(is_white(shot.at(160, 124)));
    CHECK(is_white(shot.at(160, 20)));
    CHECK(is_white(shot.at(160, 228)));
}

TEST_CASE("a large reduction averages the source instead of skipping pixels", "[unit][gpu][sensing]")
{
    // 200x200 reduced to 80x62 (scale 0.4): left red, right blue.
    TempFile gif(".gif");
    write_gif(gif.path(), 200, 200, { split_frame(200, 200, kGifRed, kGifBlue, 100) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "32");
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());

    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot shot = grab(camera, 0);
    camera.stop();

    REQUIRE(shot.width == 80);
    REQUIRE(shot.height == 62);
    CHECK(near(shot.at(10, 31), 255, 0, 0));
    CHECK(near(shot.at(70, 31), 0, 0, 255));
}

TEST_CASE("without a scale override the GIF is shown at the sensor's configured size", "[unit][gpu][sensing]")
{
    TempFile gif(".gif");
    write_gif(gif.path(), 32, 32, { square_frame(32, 32, 16, 100) });
    ScopedEnv source("SENSING_MOCK_GIF", gif.path().c_str());
    ScopedEnv fps("SENSING_MOCK_FPS", "5");

    SiplCamera camera(make_sipl_config());
    const auto& sensors = camera.sensors();
    REQUIRE(sensors.size() == 2);
    camera.start();
    const Shot shot = grab(camera, 0);
    camera.stop();

    // The vendor config's 2560x1984, not the GIF's 32x32.
    CHECK(shot.width == sensors[0].width);
    CHECK(shot.height == sensors[0].height);
    CHECK(shot.width == 2560);
    CHECK(shot.height == 1984);
    // Scale 80: the square is about 1280 px wide and tall, centred, with soft edges.
    CHECK(is_white(shot.at(shot.width / 2, shot.height / 2)));
    const int run_x = static_cast<int>(white_run_x(shot, shot.height / 2));
    const int run_y = static_cast<int>(white_run_y(shot, shot.width / 2));
    CHECK(std::abs(run_x - 1280) <= 110);
    CHECK(std::abs(run_x - run_y) <= 4);
}

TEST_CASE("a sensor's GIF can be chosen by its module name", "[unit][gpu][sensing]")
{
    TempFile red(".gif");
    TempFile blue(".gif");
    write_gif(red.path(), 16, 16, { solid_frame(16, 16, kGifRed, 100) });
    write_gif(blue.path(), 16, 16, { solid_frame(16, 16, kGifBlue, 100) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv everyone("SENSING_MOCK_GIF", red.path().c_str());
    ScopedEnv one("SENSING_MOCK_GIF_SHW5G_MIX_L3", blue.path().c_str());

    SiplCamera camera(make_sipl_config());
    REQUIRE(camera.sensors().size() == 2);
    REQUIRE(camera.sensors()[0].name == "SHW5G_MIX_L2");
    REQUIRE(camera.sensors()[1].name == "SHW5G_MIX_L3");
    camera.start();
    const Shot first = grab(camera, 0);
    const Shot second = grab(camera, 1);
    camera.stop();

    CHECK(near(first.at(first.width / 2, first.height / 2), 255, 0, 0));
    CHECK(near(second.at(second.width / 2, second.height / 2), 0, 0, 255));
}

namespace
{

std::string bundled(const char* name)
{
    return std::string(SENSING_BUNDLED_GIF_DIR) + "/" + name;
}

} // namespace

TEST_CASE("the bundled GIFs decode, animate and have their transparency composited", "[unit][gpu][sensing]")
{
    for (const auto& [name, frames] : { std::pair<const char*, size_t>{ "noto-robot.gif", 72 },
                                        std::pair<const char*, size_t>{ "noto-waving-hand.gif", 46 } })
    {
        INFO(name);
        MockGifSource gif(bundled(name));
        CHECK(gif.width() == 512);
        CHECK(gif.height() == 512);
        REQUIRE(gif.frame_count() == frames);

        auto pixel = [&](size_t frame, uint32_t x, uint32_t y)
        {
            std::array<uint8_t, 4> px{};
            REQUIRE(cudaMemcpy(px.data(), gif.frame(frame) + (static_cast<size_t>(y) * gif.width() + x) * 4, 4,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            return px;
        };

        // The corner is transparent in the file, so it shows the background, fully opaque.
        const auto corner = pixel(0, 0, 0);
        CHECK(corner[0] == MockGifSource::kBackground);
        CHECK(corner[1] == MockGifSource::kBackground);
        CHECK(corner[2] == MockGifSource::kBackground);
        CHECK(corner[3] == 255);

        // The animation changes the picture: some frame differs from the first at the centre.
        const auto first = pixel(0, 256, 256);
        bool changed = false;
        for (size_t f = 1; f < gif.frame_count() && !changed; ++f)
            changed = pixel(f, 256, 256) != first || pixel(f, 200, 300) != pixel(0, 200, 300);
        CHECK(changed);
    }
}

TEST_CASE("by default every sensor shows the animated bundled robot", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv fps("SENSING_MOCK_FPS", "20");
    SiplCamera camera(make_sipl_config());
    camera.start();

    // Captures a second apart show different parts of the 2.5 s animation.
    std::vector<Shot> shots;
    uint64_t last = 0;
    REQUIRE(wait_for(
        [&]
        {
            if (auto view = camera.latest(0))
            {
                if (view->sequence >= last + 10 || shots.empty())
                {
                    Shot shot;
                    shot.width = view->width;
                    shot.height = view->height;
                    shot.sequence = view->sequence;
                    shot.rgba = download_rgba(view->ptr, view->pitch, view->width, view->height);
                    shots.push_back(std::move(shot));
                    last = view->sequence;
                }
            }
            return shots.size() >= 4;
        },
        10s));
    camera.stop();

    // Transparent corners of the emoji read as the background gray, and the picture moves.
    for (const auto& shot : shots)
        CHECK(near(shot.at(2, 2), MockGifSource::kBackground, MockGifSource::kBackground, MockGifSource::kBackground, 8));
    size_t differing = 0;
    for (size_t i = 0; i < shots[0].rgba.size(); i += 4)
        differing += shots[0].rgba[i] != shots[2].rgba[i] ? 1 : 0;
    CHECK(differing > 500);
}

TEST_CASE("the waving hand can be given to one sensor by module name", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv one("SENSING_MOCK_GIF_SHW5G_MIX_L3", bundled("noto-waving-hand.gif").c_str());
    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot robot = grab(camera, 0);
    const Shot hand = grab(camera, 1);
    camera.stop();

    // Different pictures: the robot on one sensor, the hand on the other.
    size_t differing = 0;
    for (size_t i = 0; i < robot.rgba.size(); i += 4)
        differing += std::abs(robot.rgba[i] - hand.rgba[i]) > 20 ? 1 : 0;
    CHECK(differing > 2000);
}
