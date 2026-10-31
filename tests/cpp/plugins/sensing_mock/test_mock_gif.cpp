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
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

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
    ScopedPicture gif("/nonexistent/none.gif");
    CHECK_THROWS_AS(SiplCamera(make_sipl_config()), std::runtime_error);
}

TEST_CASE("the camera shows the GIF frame for its capture time while sequence and stamps keep advancing",
          "[unit][gpu][sensing]")
{
    TempFile gif(".gif");
    // Delays are in centiseconds: 4 is 40 ms.
    write_gif(gif.path(), 16, 16, { solid_frame(16, 16, kGifRed, 4), solid_frame(16, 16, kGifBlue, 4) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedPicture source(gif.path());
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
    ScopedPicture source(gif.path());

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
    ScopedPicture source(gif.path());

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
    ScopedPicture source(gif.path());

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
    ScopedPicture source(gif.path());

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
    ScopedPicture source(gif.path());
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

namespace
{

// Every capture of every sensor, keyed by sequence, until each sensor has `count`. Capture n shows
// the same moment of the animation on every sensor and in every camera, so a sequence number names
// one picture and pictures can be compared across sensors and across cameras.
using Frames = std::map<uint64_t, Shot>;

std::vector<Frames> record(SiplCamera& camera, size_t count)
{
    std::vector<Frames> frames(camera.sensors().size());
    camera.start();
    REQUIRE(wait_for(
        [&]
        {
            bool done = true;
            for (size_t s = 0; s < frames.size(); ++s)
            {
                if (auto view = camera.latest(static_cast<uint32_t>(s)))
                {
                    Shot shot;
                    shot.width = view->width;
                    shot.height = view->height;
                    shot.sequence = view->sequence;
                    shot.rgba = download_rgba(view->ptr, view->pitch, view->width, view->height);
                    frames[s][view->sequence] = std::move(shot);
                }
                done = done && frames[s].size() >= count;
            }
            return done;
        },
        15s));
    camera.stop();
    return frames;
}

// Mean absolute difference per colour channel between two pictures, optionally reading `b` flipped.
double mean_abs_diff(const Shot& a, const Shot& b, bool mirror_b = false)
{
    double sum = 0;
    for (uint32_t y = 0; y < a.height; ++y)
    {
        for (uint32_t x = 0; x < a.width; ++x)
        {
            const Rgb pa = a.at(x, y);
            const Rgb pb = b.at(mirror_b ? a.width - 1 - x : x, y);
            sum += std::abs(pa.r - pb.r) + std::abs(pa.g - pb.g) + std::abs(pa.b - pb.b);
        }
    }
    return sum / (static_cast<double>(a.width) * a.height * 3);
}

// The pictures two recordings have in common, as (sequence, first, second).
std::vector<std::tuple<uint64_t, const Shot*, const Shot*>> shared(const Frames& a, const Frames& b)
{
    std::vector<std::tuple<uint64_t, const Shot*, const Shot*>> out;
    for (const auto& [sequence, shot] : a)
    {
        auto it = b.find(sequence);
        if (it != b.end())
            out.emplace_back(sequence, &shot, &it->second);
    }
    return out;
}

constexpr size_t kFrames = 12;

} // namespace

TEST_CASE("a test picture is shown as it is, on every sensor", "[unit][gpu][sensing]")
{
    TempFile gif(".gif");
    write_gif(gif.path(), 16, 16, { split_frame(16, 16, kGifRed, kGifBlue, 100) });
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedPicture source(gif.path());

    SiplCamera camera(make_sipl_config());
    camera.start();
    const Shot first = grab(camera, 0);
    const Shot second = grab(camera, 1);
    camera.stop();

    // Red on the left and blue on the right for both: the mirror belongs to the fixed pictures.
    for (const Shot* shot : { &first, &second })
    {
        CHECK(near(shot->at(shot->width / 4, shot->height / 2), 255, 0, 0));
        CHECK(near(shot->at(shot->width * 3 / 4, shot->height / 2), 0, 0, 255));
    }
}

namespace
{

// Head stereo pair on link 0, left and right wrist on links 1 and 2. Pipelines: 0 and 1 are the eyes,
// 2 the left wrist, 3 the right wrist.
SiplConfig make_rig_config(std::vector<uint32_t> masks = { 0x0000, 0x0111 })
{
    SiplConfig config;
    config.platform_config_json = SENSING_MOCK_RIG_CONFIG;
    config.platform_config_name = "MOCK_S56C_2_SHF3L";
    config.link_masks = std::move(masks);
    return config;
}

} // namespace

TEST_CASE("the mock rig has the sizes and aspects of the real cameras", "[unit][sensing]")
{
    const auto sensors = SiplCamera::query(SENSING_MOCK_RIG_CONFIG, "MOCK_S56C_2_SHF3L", { 0x0000, 0x0111 });
    REQUIRE(sensors.size() == 4);
    for (size_t eye = 0; eye < 2; ++eye)
    {
        CHECK(sensors[eye].name == "S56C_HEAD");
        CHECK(sensors[eye].width == 2560);
        CHECK(sensors[eye].height == 1984);
    }
    for (size_t wrist = 2; wrist < 4; ++wrist)
    {
        CHECK(sensors[wrist].width == 1920);
        CHECK(sensors[wrist].height == 1536); // 5:4, not the head's 2560:1984
    }
    CHECK(sensors[2].name == "SHF3L_LEFT");
    CHECK(sensors[3].name == "SHF3L_RIGHT");
}

TEST_CASE("each rig camera delivers frames of its configured aspect, and wrists mirror while eyes offset",
          "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv fps("SENSING_MOCK_FPS", "20");
    SiplCamera camera(make_rig_config());
    REQUIRE(camera.sensors().size() == 4);
    const auto frames = record(camera, kFrames);

    for (size_t s = 0; s < 4; ++s)
    {
        const auto& info = camera.sensors()[s];
        REQUIRE_FALSE(frames[s].empty());
        const Shot& shot = frames[s].begin()->second;
        CHECK(shot.width == info.width);
        CHECK(shot.height == info.height);
        CHECK(shot.width == (s < 2 ? 320u : 240u));
        CHECK(shot.height == (s < 2 ? 248u : 192u));
    }

    // Left wrist plain, right wrist its mirror image.
    const auto wrists = shared(frames[2], frames[3]);
    REQUIRE(wrists.size() >= 8);
    for (const auto& [sequence, left, right] : wrists)
    {
        CHECK(mean_abs_diff(*right, *left, true) < 3.0);
        CHECK(mean_abs_diff(*right, *left, false) > 6.0);
    }

    // Head eyes show the same robot offset by the disparity: left(x) is right(x - d), and the eyes differ
    // where they are not offset.
    const auto eyes = shared(frames[0], frames[1]);
    REQUIRE(eyes.size() >= 8);
    const uint32_t d = 2 * static_cast<uint32_t>(320 * 0.02 / 2);
    REQUIRE(d > 0);
    for (const auto& [sequence, left, right] : eyes)
    {
        double sum = 0;
        double unshifted = 0;
        for (uint32_t y = 0; y < left->height; ++y)
        {
            for (uint32_t x = d + 4; x < left->width - 4; ++x)
            {
                const Rgb a = left->at(x, y);
                const Rgb b = right->at(x - d, y);
                const Rgb c = right->at(x, y);
                sum += std::abs(a.r - b.r) + std::abs(a.g - b.g) + std::abs(a.b - b.b);
                unshifted += std::abs(a.r - c.r) + std::abs(a.g - c.g) + std::abs(a.b - c.b);
            }
        }
        const double n = static_cast<double>(left->height) * (left->width - d - 8) * 3;
        CHECK(sum / n < 3.0);
        CHECK(unshifted / n > sum / n + 1.0);
    }
}

TEST_CASE("a rig camera keeps its picture when another camera is masked off", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    ScopedEnv fps("SENSING_MOCK_FPS", "20");

    SiplCamera full(make_rig_config());
    const auto all = record(full, kFrames);

    // Only link 2, the right wrist, is enabled: it becomes pipeline 0 but is still the mirrored hand.
    SiplCamera alone(make_rig_config({ 0x0000, 0x0100 }));
    REQUIRE(alone.sensors().size() == 1);
    REQUIRE(alone.sensors()[0].name == "SHF3L_RIGHT");
    const auto only = record(alone, kFrames);

    const auto pairs = shared(all[3], only[0]);
    REQUIRE(pairs.size() >= 8);
    for (const auto& [sequence, from_rig, from_alone] : pairs)
        CHECK(mean_abs_diff(*from_rig, *from_alone) < 3.0);
}

TEST_CASE("a module with no known picture is an error", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    TempFile config(".json");
    std::ofstream(config.path()) << R"({"cameraConfigs":[
      {"name":"P","cameraModules":[{"name":"UNKNOWN_CAM","linkIndex":0,"transportIndex":0}]},
      {"name":"UNKNOWN_CAM","sensorInfo":[{"virtualChannels":[{"width":64,"height":48,"fps":30.0}]}]}]})";
    SiplConfig sipl;
    sipl.platform_config_json = config.path();
    sipl.platform_config_name = "P";
    sipl.link_masks = { 0x1 };
    CHECK_THROWS_WITH(SiplCamera(sipl), Catch::Matchers::ContainsSubstring("no mock picture for module 'UNKNOWN_CAM'"));
}
