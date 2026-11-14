// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_plugin/camera_host.hpp>
#include <camera_plugin/options.hpp>
#include <camera_reader/camera_reader.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace plugins::camera;
using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;
using core::camera::CameraReader;
using core::camera::FrameSet;
using core::camera::GroupDescription;
using core::camera::ImageInfo;
using core::camera::PixelFormat;
using core::camera::StreamDescription;

namespace
{

constexpr std::uint32_t WIDTH = 64;
constexpr std::uint32_t HEIGHT = 48;

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_host_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    std::string socket_path() const
    {
        return (m_path / "cameras.sock").string();
    }

private:
    std::filesystem::path m_path;
};

// A camera that publishes when the test says so. `control` outlives the source, which the
// host owns.
struct ManualControl
{
    FrameOutput* output = nullptr; // set while the camera runs
    std::size_t streams = 0;

    // Publishes one frame set whose every pixel is `value`. False when it had to be dropped.
    bool publish(std::uint8_t value)
    {
        std::vector<ImageInfo> images(streams);
        for (std::size_t stream = 0; stream < streams; ++stream)
        {
            const std::optional<WritableImage> image = output->acquire(stream);
            if (!image)
            {
                for (std::size_t taken = 0; taken < stream; ++taken)
                {
                    output->abandon(taken, images[taken].slot);
                }
                return false;
            }
            std::fill(image->data.begin(), image->data.end(), std::byte{ value });
            images[stream].slot = image->slot;
        }
        output->publish(images, {});
        return true;
    }
};

class ManualSource final : public CameraSource
{
public:
    ManualSource(std::string id, std::size_t streams, ManualControl& control, std::uint32_t slot_count = 0)
        : m_id(std::move(id)), m_slot_count(slot_count), m_control(control)
    {
        m_control.streams = streams;
    }

    GroupDescription describe() const override
    {
        GroupDescription group;
        group.id = m_id;
        for (std::size_t i = 0; i < m_control.streams; ++i)
        {
            StreamDescription stream;
            stream.id = m_id + "/" + std::to_string(i);
            stream.pixel_format = PixelFormat::GRAY8;
            stream.width = WIDTH;
            stream.height = HEIGHT;
            stream.plane_count = 1;
            stream.planes[0].stride_bytes = WIDTH;
            stream.slot_bytes = WIDTH * HEIGHT;
            stream.slot_count = m_slot_count;
            stream.frame_rate = 30.0;
            group.streams.push_back(stream);
        }
        return group;
    }

    void start(FrameOutput& output) override
    {
        m_control.output = &output;
    }

    void stop() override
    {
        m_control.output = nullptr;
    }

private:
    std::string m_id;
    std::uint32_t m_slot_count;
    ManualControl& m_control;
};

} // namespace

TEST_CASE("a camera argument names an id, a backend and the backend's options", "[unit][camera_plugin]")
{
    const CameraConfig bare = parse_camera_argument("head:synthetic");
    REQUIRE(bare.id == "head");
    REQUIRE(bare.backend == "synthetic");
    REQUIRE(bare.options.empty());

    const CameraConfig full = parse_camera_argument("wrist:synthetic:width=640,height=480,format=gray8");
    REQUIRE(full.id == "wrist");
    REQUIRE(full.options.size() == 3);
    REQUIRE(full.options.at("width") == "640");
    REQUIRE(full.options.at("format") == "gray8");

    for (const char* malformed : { "head", ":synthetic", "head:", "head:synthetic:width", "head:synthetic:=3",
                                   "head:synthetic:width=1,width=2" })
    {
        REQUIRE_THROWS_WITH(parse_camera_argument(malformed), ContainsSubstring("ID:BACKEND"));
    }
}

TEST_CASE("options a backend does not read are refused", "[unit][camera_plugin]")
{
    const CameraConfig config = parse_camera_argument("head:zed:width=640,serial=1234,widht=7");
    OptionReader options(config);
    REQUIRE(options.number("width", 1280, 16, 8192) == 640);
    REQUIRE(options.number("height", 720, 16, 8192) == 720); // not set: the fallback
    REQUIRE(options.text("serial", "") == "1234");
    REQUIRE(options.text("mode", "hd720") == "hd720");
    REQUIRE_THROWS_WITH(options.finish(), ContainsSubstring("camera 'head' (zed backend): there is no option 'widht'"));

    REQUIRE_THROWS_WITH(options.number("serial", 0, 0, 100), ContainsSubstring("whole number from 0 to 100"));
    const CameraConfig words = parse_camera_argument("head:zed:width=wide,height=-3,fps=60.5");
    OptionReader strict(words);
    REQUIRE_THROWS_AS(strict.number("width", 1, 0, 100), std::invalid_argument);
    REQUIRE_THROWS_AS(strict.number("height", 1, 0, 100), std::invalid_argument);
    REQUIRE_THROWS_AS(strict.number("fps", 1, 0, 100), std::invalid_argument);
}

TEST_CASE("a backend that is not in the build is named with the ones that are", "[unit][camera_plugin]")
{
    const std::vector<Backend> backends = { { "synthetic", "", {} }, { "zed", "", {} } };
    REQUIRE(find_backend(backends, "zed").name == "zed");
    REQUIRE_THROWS_WITH(find_backend(backends, "oak"), ContainsSubstring("built without a 'oak' backend") &&
                                                           ContainsSubstring("synthetic, zed") &&
                                                           ContainsSubstring("chosen when the plugin is built"));
}

TEST_CASE("a host gives each camera its pools and serves them all on one endpoint", "[unit][camera_plugin]")
{
    TemporaryDirectory directory;
    ManualControl head;
    ManualControl wrist;
    CameraHost host(directory.socket_path());
    host.add_camera(std::make_unique<ManualSource>("head", 2, head));
    host.add_camera(std::make_unique<ManualSource>("wrist", 1, wrist, 4), 16); // the camera's own count wins
    REQUIRE(host.camera_count() == 2);

    ManualControl unused;
    REQUIRE_THROWS_AS(host.add_camera(std::make_unique<ManualSource>("head", 1, unused)), std::invalid_argument);
    REQUIRE_THROWS_AS(host.add_camera(std::make_unique<ManualSource>("empty", 0, unused)), std::invalid_argument);

    host.start();
    REQUIRE(head.output != nullptr);
    REQUIRE_THROWS_AS(host.add_camera(std::make_unique<ManualSource>("late", 1, unused)), std::logic_error);

    const std::vector<GroupDescription> groups = CameraReader::list({ .endpoint = directory.socket_path() });
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].id == "head");
    REQUIRE(groups[0].streams[0].slot_count == DEFAULT_SLOT_COUNT);
    REQUIRE(groups[1].id == "wrist");
    REQUIRE(groups[1].streams[0].slot_count == 4);

    CameraReader head_reader = CameraReader::open("head", {}, { .endpoint = directory.socket_path() });
    CameraReader wrist_reader = CameraReader::open("wrist", {}, { .endpoint = directory.socket_path() });
    REQUIRE(host.reader_count() == 2);

    REQUIRE(head.publish(0x21));
    REQUIRE(wrist.publish(0x42));
    const std::optional<FrameSet> pair = head_reader.next(2s);
    REQUIRE(pair.has_value());
    REQUIRE(pair->image_count() == 2);
    REQUIRE(pair->image(0).data.front() == std::byte{ 0x21 });
    REQUIRE(pair->image(1).data.back() == std::byte{ 0x21 });
    const std::optional<FrameSet> single = wrist_reader.next(2s);
    REQUIRE(single.has_value());
    REQUIRE(single->image(0).data.front() == std::byte{ 0x42 });
    REQUIRE(host.publisher().stats(0).published == 1);

    host.stop();
    REQUIRE(head.output == nullptr);
    REQUIRE(host.reader_count() == 0);
}

TEST_CASE("a camera drops a frame set when its pool is full, and never overwrites a held one", "[unit][camera_plugin]")
{
    TemporaryDirectory directory;
    ManualControl camera;
    CameraHost host(directory.socket_path());
    host.add_camera(std::make_unique<ManualSource>("head", 2, camera), 3);
    host.start();

    // A window of one in a pool of three: the newest, the reader's, and one to write ahead.
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = directory.socket_path() });
    REQUIRE(camera.publish(1));
    const std::optional<FrameSet> held = reader.next(2s);
    REQUIRE(held.has_value());
    for (std::uint8_t value = 2; value < 20; ++value)
    {
        REQUIRE(camera.publish(value)); // the ring lets go of the frame set before, so there is room
    }
    REQUIRE(held->image(0).data.front() == std::byte{ 1 });
    REQUIRE(host.publisher().stats(0).no_free_slot == 0);

    // A backend that takes a slot and neither publishes nor abandons it is what fills a pool.
    const std::optional<WritableImage> forgotten = camera.output->acquire(0);
    REQUIRE(forgotten.has_value());
    REQUIRE_FALSE(camera.publish(99)); // stream 0 has no free slot left; the frame set is dropped
    REQUIRE(host.publisher().stats(0).no_free_slot == 1);
    REQUIRE(held->image(0).data.front() == std::byte{ 1 });

    camera.output->abandon(0, forgotten->slot);
    REQUIRE(camera.publish(100));
}
