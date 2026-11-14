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
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
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

// A camera whose SDK is given its buffers up front, fills the one it chooses and stops capturing
// when it runs out. `control` plays the SDK and outlives the source, which the host owns.
struct SdkControl
{
    static constexpr std::uint32_t BUFFERS = 6;
    static constexpr std::uint32_t RESERVE = 3; // buffers the SDK needs in hand to keep capturing

    FrameOutput* output = nullptr; // set while the camera runs
    std::mutex mutex; // a buffer comes back on whichever thread let go of it
    std::deque<std::uint32_t> with_sdk; // the buffers the SDK has to fill
    std::vector<std::span<std::byte>> buffers; // the memory it was given, by slot

    std::size_t in_hand()
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return with_sdk.size();
    }

    // The SDK fills the next of its buffers with `value` and the camera publishes it.
    void capture(std::uint8_t value)
    {
        std::uint32_t slot = 0;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            slot = with_sdk.front();
            with_sdk.pop_front();
        }
        std::fill(buffers[slot].begin(), buffers[slot].end(), std::byte{ value });
        output->claim(0, slot);
        ImageInfo image;
        image.slot = slot;
        output->publish({ &image, 1 }, {});
    }
};

class SdkSource final : public CameraSource
{
public:
    explicit SdkSource(SdkControl& control) : m_control(control)
    {
    }

    GroupDescription describe() const override
    {
        StreamDescription stream;
        stream.id = "sdk/0";
        stream.pixel_format = PixelFormat::GRAY8;
        stream.width = WIDTH;
        stream.height = HEIGHT;
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = WIDTH;
        stream.slot_bytes = WIDTH * HEIGHT;
        stream.slot_count = SdkControl::BUFFERS;
        stream.frame_rate = 30.0;
        return { .id = "sdk", .streams = { stream } };
    }

    std::uint32_t backend_reserve() const override
    {
        return SdkControl::RESERVE;
    }

    void start(FrameOutput& output) override
    {
        m_control.output = &output;
        output.on_slot_free(
            [this](std::size_t, std::uint32_t slot)
            {
                const std::lock_guard<std::mutex> lock(m_control.mutex);
                m_control.with_sdk.push_back(slot);
            });
        // The SDK is given every slot of the pool as one of its buffers.
        for (std::uint32_t slot = 0; slot < output.slot_count(0); ++slot)
        {
            m_control.buffers.push_back(output.slot_memory(0, slot));
            m_control.with_sdk.push_back(slot);
        }
    }

    void stop() override
    {
        m_control.output = nullptr;
    }

private:
    SdkControl& m_control;
};

bool wait_until(const std::function<bool()>& condition)
{
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!condition())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

bool all_pixels_are(const FrameSet& frames, std::uint8_t value)
{
    const auto pixels = frames.image(0).data;
    return std::all_of(pixels.begin(), pixels.end(), [value](std::byte pixel) { return pixel == std::byte{ value }; });
}

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

    const std::vector<core::camera::GroupListing> groups = CameraReader::list({ .endpoint = directory.socket_path() });
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "head");
    REQUIRE(groups[0].group.streams[0].slot_count == DEFAULT_SLOT_COUNT);
    REQUIRE(groups[1].group.id == "wrist");
    REQUIRE(groups[1].group.streams[0].slot_count == 4);

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

TEST_CASE("a backend is opened once, with all of its cameras", "[unit][camera_plugin]")
{
    // "rig" stands for an SDK that serves all its cameras from one object, which it opens once.
    int rig_opens = 0;
    std::vector<std::string> rig_cameras;
    std::deque<ManualControl> controls;
    const auto one_source_each = [&controls](std::span<const CameraConfig> cameras)
    {
        std::vector<std::unique_ptr<CameraSource>> sources;
        for (const CameraConfig& camera : cameras)
        {
            sources.push_back(std::make_unique<ManualSource>(camera.id, 1, controls.emplace_back()));
        }
        return sources;
    };
    const std::vector<Backend> backends = {
        { "rig", "",
          [&](std::span<const CameraConfig> cameras)
          {
              ++rig_opens;
              for (const CameraConfig& camera : cameras)
              {
                  rig_cameras.push_back(camera.id);
              }
              return one_source_each(cameras);
          } },
        { "single", "", one_source_each },
        { "broken", "", [](std::span<const CameraConfig>) { return std::vector<std::unique_ptr<CameraSource>>(); } },
    };

    const std::vector<CameraConfig> cameras = { { "left", "rig", {} }, { "top", "single", {} }, { "right", "rig", {} } };
    const std::vector<std::unique_ptr<CameraSource>> sources = open_cameras(backends, cameras);
    REQUIRE(rig_opens == 1);
    REQUIRE(rig_cameras == std::vector<std::string>{ "left", "right" });
    REQUIRE(sources.size() == 3);
    REQUIRE(sources[0]->describe().id == "left");
    REQUIRE(sources[1]->describe().id == "right");
    REQUIRE(sources[2]->describe().id == "top");

    const std::vector<CameraConfig> unknown = { { "head", "oak", {} } };
    REQUIRE_THROWS_WITH(open_cameras(backends, unknown), ContainsSubstring("built without a 'oak' backend"));
    const std::vector<CameraConfig> lost = { { "head", "broken", {} } };
    REQUIRE_THROWS_WITH(open_cameras(backends, lost), ContainsSubstring("was given 1 camera(s) and opened 0"));
}

TEST_CASE("a camera whose SDK owns the buffers is never left short of them", "[unit][camera_plugin]")
{
    TemporaryDirectory directory;
    SdkControl sdk;
    CameraHost host(directory.socket_path());
    host.add_camera(std::make_unique<SdkSource>(sdk));
    host.start();
    REQUIRE(sdk.in_hand() == SdkControl::BUFFERS);
    REQUIRE_THROWS_AS(sdk.output->acquire(0), std::logic_error); // this camera names its slots itself

    // Of six buffers the SDK keeps three. That leaves the newest frame set and two for readers.
    using core::camera::DeliveryOrder;
    REQUIRE_THROWS_WITH(CameraReader::open("sdk", { .window = 3, .order = DeliveryOrder::NEWEST },
                                           { .endpoint = directory.socket_path() }),
                        ContainsSubstring("3 for the backend"));
    CameraReader reader = CameraReader::open(
        "sdk", { .window = 2, .order = DeliveryOrder::NEWEST }, { .endpoint = directory.socket_path() });

    sdk.capture(1);
    std::optional<FrameSet> first = reader.next(2s);
    REQUIRE(first.has_value());
    sdk.capture(2);
    std::optional<FrameSet> second = reader.next(2s);
    REQUIRE(second.has_value());

    // The camera runs on while the reader holds both. The SDK finds its reserve in hand before
    // every capture, and is never handed a buffer a reader still looks at.
    for (std::uint8_t value = 3; value < 40; ++value)
    {
        REQUIRE(sdk.in_hand() >= SdkControl::RESERVE);
        sdk.capture(value);
    }
    REQUIRE(all_pixels_are(*first, 1));
    REQUIRE(all_pixels_are(*second, 2));
    REQUIRE_THROWS_AS(sdk.output->claim(0, first->image(0).info.slot), std::invalid_argument);

    // A buffer goes back to the SDK when its last holder lets go of it.
    const std::size_t before = sdk.in_hand();
    first.reset();
    REQUIRE(wait_until([&] { return sdk.in_hand() == before + 1; }));

    // And when the readers are gone, every buffer but the newest frame set's is the SDK's again.
    host.stop();
    REQUIRE(sdk.in_hand() == SdkControl::BUFFERS - 1);
    REQUIRE(all_pixels_are(*second, 2)); // still mapped, and as it was
}

TEST_CASE("a camera that restarts says so, and what a reader holds stays as it was", "[unit][camera_plugin]")
{
    TemporaryDirectory directory;
    ManualControl camera;
    CameraHost host(directory.socket_path());
    host.add_camera(std::make_unique<ManualSource>("head", 1, camera), 4);
    host.start();
    CameraReader reader = CameraReader::open(
        "head", { .window = 2, .order = core::camera::DeliveryOrder::NEWEST }, { .endpoint = directory.socket_path() });

    REQUIRE(camera.publish(1));
    const std::optional<FrameSet> held = reader.next(2s);
    REQUIRE(held.has_value());
    REQUIRE(held->generation() == 0);

    camera.output->restart();
    for (std::uint8_t value = 2; value <= 12; ++value)
    {
        REQUIRE(camera.publish(value));
    }
    REQUIRE(all_pixels_are(*held, 1));

    const std::optional<FrameSet> after = reader.next(2s);
    REQUIRE(after.has_value());
    REQUIRE(after->generation() == 1);
    REQUIRE(all_pixels_are(*after, 12));
}
