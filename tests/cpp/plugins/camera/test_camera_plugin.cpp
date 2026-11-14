// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_backend_synthetic/synthetic_backend.hpp>
#include <camera_plugin/camera_host.hpp>
#include <camera_plugin/options.hpp>
#include <camera_plugin_backends/backends.hpp>
#include <camera_reader/camera_reader.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
using core::camera::CameraReaderError;
using core::camera::DeliveryOrder;
using core::camera::FrameSet;
using core::camera::ImageView;
using core::camera::PixelFormat;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_plugin_XXXXXX").string();
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

std::unique_ptr<CameraSource> open_synthetic(const std::string& argument)
{
    const CameraConfig config = parse_camera_argument(argument);
    const std::vector<Backend> backends = built_backends();
    return find_backend(backends, config.backend).open(config);
}

// Every byte of a synthetic image is known from its frame number and its stream.
void require_synthetic(const ImageView& image, std::size_t stream)
{
    const std::uint64_t frame = synthetic_frame_number(image.data);
    REQUIRE(image.info.device_frame_id == frame);
    const std::byte fill = synthetic_fill(frame, stream);
    const auto rest = image.data.subspan(sizeof(frame));
    REQUIRE(std::all_of(rest.begin(), rest.end(), [fill](std::byte value) { return value == fill; }));
}

// The plugin as a process of its own.
class PluginProcess
{
public:
    explicit PluginProcess(const std::vector<std::string>& arguments)
    {
        std::vector<std::string> owned = { CAMERA_PLUGIN_EXECUTABLE };
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
            ::execv(argv[0], argv.data());
            ::_exit(127);
        }
    }

    ~PluginProcess()
    {
        if (m_pid > 0)
        {
            ::kill(m_pid, SIGKILL);
            ::waitpid(m_pid, nullptr, 0);
        }
    }

    // Waits for the process to end and returns its exit code, or -1 if it did not exit by itself.
    int wait()
    {
        int status = 0;
        REQUIRE(::waitpid(m_pid, &status, 0) == m_pid);
        m_pid = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    int stop(int signal)
    {
        REQUIRE(::kill(m_pid, signal) == 0);
        return wait();
    }

private:
    pid_t m_pid = -1;
};

// A plugin that was just started needs a moment before it listens.
std::optional<CameraReader> open_when_ready(const std::string& endpoint,
                                            const std::string& group,
                                            const core::camera::ReaderOptions& options = {})
{
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        try
        {
            return CameraReader::open(endpoint, group, options);
        }
        catch (const CameraReaderError&)
        {
            std::this_thread::sleep_for(10ms);
        }
    }
    return std::nullopt;
}

// Reads whatever is still on its way; ends by throwing when the plugin's side has closed.
void read_until_closed(CameraReader& reader)
{
    for (int i = 0; i < 1000; ++i)
    {
        (void)reader.next(2s);
    }
}

} // namespace

TEST_CASE("the synthetic backend is in every build", "[unit][camera_plugin]")
{
    const std::vector<Backend> backends = built_backends();
    REQUIRE(find_backend(backends, "synthetic").name == "synthetic");
    REQUIRE_THROWS_WITH(
        find_backend(backends, "no_such_sdk"), ContainsSubstring("built without a 'no_such_sdk' backend"));
}

TEST_CASE("the synthetic backend refuses options it does not know", "[unit][camera_plugin]")
{
    REQUIRE_THROWS_WITH(open_synthetic("head:synthetic:widht=640"), ContainsSubstring("no option 'widht'"));
    REQUIRE_THROWS_WITH(open_synthetic("head:synthetic:width=wide"), ContainsSubstring("whole number from 16 to 8192"));
    REQUIRE_THROWS_WITH(open_synthetic("head:synthetic:fps=0"), ContainsSubstring("camera 'head' (synthetic backend)"));
    REQUIRE_THROWS_WITH(open_synthetic("head:synthetic:streams=9"), ContainsSubstring("option 'streams'"));
    REQUIRE_THROWS_WITH(open_synthetic("head:synthetic:format=jpeg"), ContainsSubstring("bgra8, rgba8, gray8 or nv12"));

    const core::camera::GroupDescription pair = open_synthetic("head:synthetic")->describe();
    REQUIRE(pair.id == "head");
    REQUIRE(pair.streams.size() == 2);
    REQUIRE(pair.streams[0].id == "head/left");
    REQUIRE(pair.streams[1].id == "head/right");
    REQUIRE(pair.streams[0].slot_bytes == 1280u * 720u * 4u);

    const core::camera::GroupDescription nv12 =
        open_synthetic("rig:synthetic:streams=3,format=nv12,width=64,height=48")->describe();
    REQUIRE(nv12.streams[2].id == "rig/2");
    REQUIRE(nv12.streams[2].pixel_format == PixelFormat::NV12);
    REQUIRE(nv12.streams[2].plane_count == 2);
    REQUIRE(nv12.streams[2].planes[1].offset_bytes == 64u * 48u);
    REQUIRE(nv12.streams[2].slot_bytes == 64u * 48u * 3u / 2u);
}

TEST_CASE("synthetic cameras reach readers whole and in step", "[unit][camera_plugin]")
{
    TemporaryDirectory directory;
    CameraHost host(directory.socket_path());
    host.add_camera(open_synthetic("head:synthetic:width=160,height=120,fps=200"));
    host.add_camera(open_synthetic("wrist:synthetic:width=64,height=48,fps=200,streams=1,format=gray8"), 6);
    REQUIRE_THROWS_AS(host.add_camera(open_synthetic("head:synthetic")), std::invalid_argument); // id taken
    REQUIRE(host.camera_count() == 2);
    host.start();
    REQUIRE_THROWS_AS(host.add_camera(open_synthetic("late:synthetic")), std::logic_error);

    const auto groups = CameraReader::list(directory.socket_path());
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].streams[0].slot_count == DEFAULT_SLOT_COUNT);
    REQUIRE(groups[1].streams[0].slot_count == 6);

    // A pair arrives as one frame set: the same frame number in both eyes, each with its own fill.
    CameraReader head = CameraReader::open(directory.socket_path(), "head");
    CameraReader wrist = CameraReader::open(directory.socket_path(), "wrist", { .window = 2 });
    std::uint64_t previous = 0;
    for (int i = 0; i < 20; ++i)
    {
        const std::optional<FrameSet> frames = head.next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->image_count() == 2);
        require_synthetic(frames->image(0), 0);
        require_synthetic(frames->image(1), 1);
        REQUIRE(frames->image(0).info.device_frame_id == frames->image(1).info.device_frame_id);
        REQUIRE(frames->sequence() > previous);
        REQUIRE(frames->timeline().capture_ns <= frames->timeline().sdk_ready_ns);
        REQUIRE(frames->timeline().sdk_ready_ns <= frames->timeline().published_ns);
        previous = frames->sequence();
    }

    // Whatever a reader was not given is accounted for: no frame set goes missing unreported,
    // however far the reader fell behind while the loop above ran.
    std::optional<FrameSet> first = wrist.next(2s);
    REQUIRE(first.has_value());
    std::uint64_t last = first->sequence();
    first.reset();
    for (int i = 0; i < 20; ++i)
    {
        const std::optional<FrameSet> frames = wrist.next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->sequence() == last + 1 + frames->skipped());
        require_synthetic(frames->image(0), 0);
        last = frames->sequence();
    }
    REQUIRE(host.reader_count() == 2);

    host.stop();
    // What was already on its way is still delivered; then the reader is told.
    REQUIRE_THROWS_AS(read_until_closed(head), core::camera::CameraDisconnected);
}

TEST_CASE("a reader in another process gets frames from the plugin", "[unit][camera_plugin][process]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.socket_path();
    PluginProcess plugin({ "--endpoint=" + endpoint, "--camera=head:synthetic:width=320,height=240,fps=120",
                           "--plugin-root-id=ignored" });

    std::optional<CameraReader> reader = open_when_ready(endpoint, "head");
    REQUIRE(reader.has_value());
    std::uint64_t previous = 0;
    for (int i = 0; i < 30; ++i)
    {
        const std::optional<FrameSet> frames = reader->next(2s);
        REQUIRE(frames.has_value());
        require_synthetic(frames->image(0), 0);
        require_synthetic(frames->image(1), 1);
        REQUIRE(frames->sequence() > previous);
        previous = frames->sequence();
    }

    // A second plugin cannot take the endpoint of a running one.
    PluginProcess second({ "--endpoint=" + endpoint, "--camera=head:synthetic" });
    REQUIRE(second.wait() == 1);
    REQUIRE(reader->next(2s).has_value());

    // SIGTERM is a clean stop; the reader is told.
    REQUIRE(plugin.stop(SIGTERM) == 0);
    REQUIRE_THROWS_AS(read_until_closed(*reader), core::camera::CameraDisconnected);

    // A plugin that was killed leaves a socket file behind; the next one replaces it.
    {
        PluginProcess killed({ "--endpoint=" + endpoint, "--camera=head:synthetic:width=64,height=48" });
        REQUIRE(open_when_ready(endpoint, "head").has_value());
        REQUIRE(killed.stop(SIGKILL) == -1);
    }
    REQUIRE(std::filesystem::exists(endpoint));
    PluginProcess next({ "--endpoint=" + endpoint, "--camera=head:synthetic:width=64,height=48" });
    std::optional<CameraReader> again = open_when_ready(endpoint, "head");
    REQUIRE(again.has_value());
    REQUIRE(again->next(2s).has_value());
}

TEST_CASE("the plugin refuses arguments it cannot use", "[unit][camera_plugin][process]")
{
    TemporaryDirectory directory;
    const std::string endpoint = "--endpoint=" + directory.socket_path();
    REQUIRE(PluginProcess({ "--list-backends" }).wait() == 0);
    REQUIRE(PluginProcess({ "--help" }).wait() == 0);
    REQUIRE(PluginProcess({}).wait() == 2);
    REQUIRE(PluginProcess({ endpoint }).wait() == 2);
    REQUIRE(PluginProcess({ endpoint, "--camera=head:synthetic", "--frobnicate" }).wait() == 2);
    REQUIRE(PluginProcess({ endpoint, "--camera=head" }).wait() == 2);
    // Arguments that parse but cannot be honoured end the plugin with an error in its log.
    REQUIRE(PluginProcess({ endpoint, "--camera=head:zed" }).wait() == 1);
    REQUIRE(PluginProcess({ endpoint, "--camera=head:synthetic:colour=red" }).wait() == 1);
    REQUIRE(PluginProcess({ endpoint, "--camera=head:synthetic", "--slots=1" }).wait() == 1);
}
