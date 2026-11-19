// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/camera_host.hpp>
#include <camera/camera_reader.hpp>
#include <camera/options.hpp>
#include <camera_synthetic/synthetic_backend.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <plugin_manager/plugin_manager.hpp>
#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace core::camera;
using namespace plugins::camera_synthetic;
using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_synth_XXXXXX").string();
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

    const std::filesystem::path& path() const
    {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

// Sets an environment variable for one test, and for the processes it starts, and puts back
// what was there.
class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const std::string& value) : m_name(name)
    {
        if (const char* previous = std::getenv(name))
        {
            m_previous = previous;
        }
        ::setenv(name, value.c_str(), 1);
    }

    ~ScopedEnvironment()
    {
        if (m_previous)
        {
            ::setenv(m_name, m_previous->c_str(), 1);
        }
        else
        {
            ::unsetenv(m_name);
        }
    }

private:
    const char* m_name;
    std::optional<std::string> m_previous;
};

// A user's runtime and configuration directories for one test and the plugins it starts. A
// plugin that is not told where to listen or which cameras to serve looks there, and so does a
// reader that looks for the plugin; without them a test would meet the developer's own plugins
// and cameras.yaml.
class UserDirectories
{
public:
    UserDirectories()
        : m_runtime("XDG_RUNTIME_DIR", (m_directory.path() / "run").string()),
          m_config_home("XDG_CONFIG_HOME", (m_directory.path() / "config").string())
    {
        std::filesystem::create_directories(m_directory.path() / "run");
        std::filesystem::create_directories(m_directory.path() / "config" / "isaaccapture");
    }

    // The socket path of an endpoint in the user's endpoint directory.
    std::string endpoint(const std::string& socket_name) const
    {
        return (m_directory.path() / "run" / "isaaccapture" / socket_name).string();
    }

    std::string file(const std::string& name) const
    {
        return (m_directory.path() / name).string();
    }

    // Writes a file and returns its path.
    std::string write(const std::string& name, const std::string& text) const
    {
        std::ofstream(file(name)) << text;
        return file(name);
    }

    // Writes the user's own configuration file, the one a plugin reads when it is not given one.
    void write_config(const std::string& text) const
    {
        write("config/isaaccapture/cameras.yaml", text);
    }

private:
    TemporaryDirectory m_directory;
    ScopedEnvironment m_runtime;
    ScopedEnvironment m_config_home;
};

// One synthetic camera, opened from what follows `--camera=` on the plugin's command line.
std::unique_ptr<CameraSource> open_synthetic(const std::string& argument)
{
    const Backend backend = synthetic_backend();
    const CameraConfig config = parse_camera_argument(argument, backend.name);
    return std::move(backend.open({ &config, 1 }).front());
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
        std::vector<std::string> owned = { CAMERA_SYNTHETIC_EXECUTABLE };
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

    // Waits for the process to end and returns its exit code: -1 if a signal ended it, -2 if it
    // was still running after `timeout`.
    int wait(std::chrono::seconds timeout = 30s)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        int status = 0;
        while (::waitpid(m_pid, &status, WNOHANG) == 0)
        {
            if (std::chrono::steady_clock::now() > deadline)
            {
                return -2;
            }
            std::this_thread::sleep_for(10ms);
        }
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
std::optional<CameraReader> open_when_ready(const PluginLocation& plugin,
                                            const std::string& group,
                                            const ReaderOptions& options = {})
{
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        try
        {
            return CameraReader::open(group, options, plugin);
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

TEST_CASE("the synthetic backend refuses options it does not know", "[unit][camera_synthetic]")
{
    // The name is what a camera's configuration says in `backend:`, and the plugin's instance name.
    REQUIRE(synthetic_backend().name == "synthetic");

    REQUIRE_THROWS_WITH(open_synthetic("head:widht=640"), ContainsSubstring("no option 'widht'"));
    REQUIRE_THROWS_WITH(open_synthetic("head:width=wide"), ContainsSubstring("whole number from 16 to 8192"));
    REQUIRE_THROWS_WITH(open_synthetic("head:fps=0"), ContainsSubstring("camera 'head' (synthetic backend)"));
    REQUIRE_THROWS_WITH(open_synthetic("head:streams=9"), ContainsSubstring("option 'streams'"));
    REQUIRE_THROWS_WITH(open_synthetic("head:format=jpeg"), ContainsSubstring("bgra8, rgba8, gray8 or nv12"));

    const GroupDescription pair = open_synthetic("head")->describe();
    REQUIRE(pair.id == "head");
    REQUIRE(pair.streams.size() == 2);
    REQUIRE(pair.streams[0].id == "head/left");
    REQUIRE(pair.streams[1].id == "head/right");
    REQUIRE(pair.streams[0].slot_bytes == 1280u * 720u * 4u);

    const GroupDescription nv12 = open_synthetic("rig:streams=3,format=nv12,width=64,height=48")->describe();
    REQUIRE(nv12.streams[2].id == "rig/2");
    REQUIRE(nv12.streams[2].pixel_format == PixelFormat::NV12);
    REQUIRE(nv12.streams[2].plane_count == 2);
    REQUIRE(nv12.streams[2].planes[1].offset_bytes == 64u * 48u);
    REQUIRE(nv12.streams[2].slot_bytes == 64u * 48u * 3u / 2u);
}

TEST_CASE("synthetic cameras reach readers whole and in step", "[unit][camera_synthetic]")
{
    TemporaryDirectory directory;
    CameraHost host(directory.socket_path());
    host.add_camera(open_synthetic("head:width=160,height=120,fps=200"));
    host.add_camera(open_synthetic("wrist:width=64,height=48,fps=200,streams=1,format=gray8"), 6);
    REQUIRE_THROWS_AS(host.add_camera(open_synthetic("head")), std::invalid_argument); // id taken
    REQUIRE(host.camera_count() == 2);
    host.start();
    REQUIRE_THROWS_AS(host.add_camera(open_synthetic("late")), std::logic_error);

    const auto groups = CameraReader::list({ .endpoint = directory.socket_path() });
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.streams[0].slot_count == DEFAULT_SLOT_COUNT);
    REQUIRE(groups[1].group.streams[0].slot_count == 6);

    // A pair arrives as one frame set: the same frame number in both eyes, each with its own fill.
    CameraReader head = CameraReader::open("head", {}, { .endpoint = directory.socket_path() });
    CameraReader wrist = CameraReader::open("wrist", { .window = 2 }, { .endpoint = directory.socket_path() });
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
    REQUIRE_THROWS_AS(read_until_closed(head), CameraDisconnected);
}

TEST_CASE("a reader in another process gets frames from the plugin", "[integration][camera_synthetic]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.socket_path();
    PluginProcess plugin(
        { "--endpoint=" + endpoint, "--camera=head:width=320,height=240,fps=120", "--plugin-root-id=ignored" });

    std::optional<CameraReader> reader = open_when_ready({ .endpoint = endpoint }, "head");
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
    PluginProcess second({ "--endpoint=" + endpoint, "--camera=head" });
    REQUIRE(second.wait() == 1);
    REQUIRE(reader->next(2s).has_value());

    // SIGTERM is a clean stop; the reader is told.
    REQUIRE(plugin.stop(SIGTERM) == 0);
    REQUIRE_THROWS_AS(read_until_closed(*reader), CameraDisconnected);

    // A plugin that was killed leaves a socket file behind; the next one replaces it.
    {
        PluginProcess killed({ "--endpoint=" + endpoint, "--camera=head:width=64,height=48" });
        REQUIRE(open_when_ready({ .endpoint = endpoint }, "head").has_value());
        REQUIRE(killed.stop(SIGKILL) == -1);
    }
    REQUIRE(std::filesystem::exists(endpoint));
    PluginProcess next({ "--endpoint=" + endpoint, "--camera=head:width=64,height=48" });
    std::optional<CameraReader> again = open_when_ready({ .endpoint = endpoint }, "head");
    REQUIRE(again.has_value());
    REQUIRE(again->next(2s).has_value());
}

TEST_CASE("a reader finds the plugin that serves its camera, and a second instance beside it",
          "[integration][camera_synthetic]")
{
    // Neither side is told where the other is: the plugin has no --endpoint and the reader no
    // location. Both look in this user's directories, which are this test's own.
    const UserDirectories user;
    user.write_config(
        "cameras:\n"
        "  head:\n"
        "    backend: synthetic\n"
        "    width: 160\n"
        "    height: 120\n"
        "    fps: 120\n"
        "  depth:\n" // another plugin's camera, in the file all of them share
        "    backend: other\n");

    PluginProcess plugin({});
    std::optional<CameraReader> reader = open_when_ready({}, "head");
    REQUIRE(reader.has_value());
    const std::optional<FrameSet> frames = reader->next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->image(0).stream->width == 160);
    require_synthetic(frames->image(0), 0);
    // The plugin's instance name is its backend's, and it serves that backend's cameras only.
    REQUIRE(std::filesystem::is_socket(user.endpoint("camera-synthetic.sock")));
    REQUIRE(CameraReader::list().size() == 1);

    // A second synthetic plugin runs beside it under another instance name, with cameras of its
    // own. A reader finds each camera by its group, whichever plugin has it.
    PluginProcess second({ "--instance=bench", "--camera=wrist:width=64,height=48,streams=1" });
    std::optional<CameraReader> wrist = open_when_ready({}, "wrist");
    REQUIRE(wrist.has_value());
    REQUIRE(wrist->next(2s).has_value());
    REQUIRE(CameraReader::open("head").next(2s).has_value());
    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "wrist"); // in the order of the socket paths
    REQUIRE(groups[0].endpoint == user.endpoint("camera-bench.sock"));
    REQUIRE(groups[1].group.id == "head");
    REQUIRE(groups[1].endpoint == user.endpoint("camera-synthetic.sock"));

    // A reader that names an instance reads from that plugin and no other.
    REQUIRE(CameraReader::open("wrist", {}, { .instance = "bench" }).next(2s).has_value());
    REQUIRE_THROWS_WITH(
        CameraReader::open("wrist", {}, { .instance = "synthetic" }), ContainsSubstring("no group 'wrist'"));
    REQUIRE_THROWS_WITH(
        CameraReader::open("head", {}, { .instance = "nobody" }), ContainsSubstring("camera-nobody.sock"));
}

TEST_CASE("the plugin takes its cameras from a file that is named, or says why it cannot",
          "[integration][camera_synthetic]")
{
    const UserDirectories user;

    // Another file than the user's can be named.
    const std::string other =
        user.write("other.yaml", "cameras:\n  chest:\n    backend: synthetic\n    width: 64\n    height: 48\n");
    PluginProcess plugin({ "--config=" + other });
    REQUIRE(open_when_ready({}, "chest").has_value());

    // A file whose cameras are all another backend's has nothing for this plugin: the camera
    // is misspelt, or it is another plugin's and this one was started by mistake.
    const std::string theirs = user.write("theirs.yaml", "cameras:\n  head:\n    backend: other\n");
    REQUIRE(PluginProcess({ "--instance=second", "--config=" + theirs }).wait() == 1);
    REQUIRE(PluginProcess({ "--instance=third", "--config=" + user.file("missing.yaml") }).wait() == 1);

    // Without a file the plugin looks for cameras, and the synthetic backend has none to find.
    REQUIRE(PluginProcess({ "--instance=fourth" }).wait() == 1);
    // --auto sets the user's file aside, so it is the same with a camera in that file.
    user.write_config("cameras:\n  head:\n    backend: synthetic\n");
    REQUIRE(PluginProcess({ "--instance=fifth", "--auto" }).wait() == 1);
}

TEST_CASE("the plugin manager can start the camera_synthetic plugin", "[integration][camera_synthetic]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.socket_path();

    // A plugin directory is a manifest next to its executable; the build tree has one.
    core::PluginManager manager({ std::filesystem::path(CAMERA_SYNTHETIC_DIRECTORY).parent_path().string() });
    const std::vector<std::string> names = manager.get_plugin_names();
    REQUIRE(std::find(names.begin(), names.end(), "camera_synthetic") != names.end());
    {
        const std::unique_ptr<core::Plugin> plugin = manager.start(
            "camera_synthetic", "camera_test", { "--endpoint=" + endpoint, "--camera=head:width=64,height=48" });
        std::optional<CameraReader> reader = open_when_ready({ .endpoint = endpoint }, "head");
        REQUIRE(reader.has_value());
        REQUIRE(reader->next(2s).has_value());
        REQUIRE_NOTHROW(plugin->check_health());
    }
    // The handle going away stops the plugin, which removes its socket file on the way out.
    REQUIRE_FALSE(std::filesystem::exists(endpoint));
}

TEST_CASE("the plugin refuses arguments it cannot use", "[integration][camera_synthetic]")
{
    // The --help plugin is given no endpoint. Should it start after all, it does so in this
    // test's directories and not in the developer's.
    const UserDirectories user;
    const std::string endpoint = "--endpoint=" + user.file("cameras.sock");
    REQUIRE(PluginProcess({ "--help" }).wait() == 0);
    REQUIRE(PluginProcess({ endpoint, "--instance=other" }).wait() == 2);
    REQUIRE(PluginProcess({ endpoint, "--auto", "--camera=head" }).wait() == 2);
    REQUIRE(PluginProcess({ endpoint, "--camera=head", "--frobnicate" }).wait() == 2);
    // What follows a camera's id is options, each KEY=VALUE.
    REQUIRE(PluginProcess({ endpoint, "--camera=head:synthetic" }).wait() == 2);
    REQUIRE(PluginProcess({ endpoint, "--camera=" }).wait() == 2);
    // Arguments that parse but cannot be honoured end the plugin with an error in its log.
    REQUIRE(PluginProcess({ endpoint, "--camera=head:colour=red" }).wait() == 1);
    REQUIRE(PluginProcess({ endpoint, "--camera=head", "--slots=1" }).wait() == 1);
}
