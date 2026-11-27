// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/camera_plugin.hpp>
#include <camera/camera_reader.hpp>
#include <camera/camera_source.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace core::camera;
using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;

namespace
{

// Sets an environment variable for one test and puts back what was there.
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

// A user's runtime and configuration directories for one test. A plugin that is not told where
// to listen or which cameras to serve looks there, and so does a reader that looks for the
// plugin; without them a test would meet the developer's own plugins and cameras.yaml.
class UserDirectories
{
public:
    UserDirectories()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_run_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
        std::filesystem::create_directories(m_path / "run");
        std::filesystem::create_directories(m_path / "config" / "isaaccapture");
        m_runtime.emplace("XDG_RUNTIME_DIR", (m_path / "run").string());
        m_config_home.emplace("XDG_CONFIG_HOME", (m_path / "config").string());
    }

    ~UserDirectories()
    {
        m_config_home.reset();
        m_runtime.reset();
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    UserDirectories(const UserDirectories&) = delete;
    UserDirectories& operator=(const UserDirectories&) = delete;

    // The socket path of an endpoint in the user's endpoint directory.
    std::string endpoint(const std::string& socket_name) const
    {
        return (m_path / "run" / "isaaccapture" / socket_name).string();
    }

    // The path of a file of this test. "config/isaaccapture/cameras.yaml" is the user's own.
    std::string file(const std::string& name) const
    {
        return (m_path / name).string();
    }

    // Writes a file and returns its path.
    std::string write(const std::string& name, const std::string& text) const
    {
        std::ofstream(file(name)) << text;
        return file(name);
    }

private:
    std::filesystem::path m_path;
    std::optional<ScopedEnvironment> m_runtime;
    std::optional<ScopedEnvironment> m_config_home;
};

// A camera that describes itself and publishes nothing. These cases are about which cameras a
// plugin serves and where; test_camera_host.cpp has the frames.
class IdleSource final : public CameraSource
{
public:
    explicit IdleSource(std::string id) : m_id(std::move(id))
    {
    }

    GroupDescription describe() const override
    {
        StreamDescription stream;
        stream.id = m_id + "/0";
        stream.pixel_format = PixelFormat::GRAY8;
        stream.width = 64;
        stream.height = 48;
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = 64;
        stream.slot_bytes = 64 * 48;
        stream.frame_rate = 30.0;
        return { .id = m_id, .streams = { stream } };
    }

    void start(FrameOutput&) override
    {
    }

    void stop() override
    {
    }

private:
    std::string m_id;
};

// What a stand-in backend was given, one entry per call of its open(). The plugin calls it on
// its thread and the test looks from its own.
class Opened
{
public:
    void add(std::span<const CameraConfig> cameras)
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_calls.emplace_back(cameras.begin(), cameras.end());
    }

    std::vector<std::vector<CameraConfig>> calls()
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_calls;
    }

private:
    std::mutex m_mutex;
    std::vector<std::vector<CameraConfig>> m_calls;
};

// Stands for a vendor's capture code: opens an idle camera for each one it is given.
Backend standin_backend(const std::string& name, Opened& opened)
{
    Backend backend;
    backend.name = name;
    backend.description = "a stand-in for a vendor's capture code";
    backend.open = [&opened](std::span<const CameraConfig> cameras)
    {
        opened.add(cameras);
        std::vector<std::unique_ptr<CameraSource>> sources;
        for (const CameraConfig& camera : cameras)
        {
            sources.push_back(std::make_unique<IdleSource>(camera.id));
        }
        return sources;
    };
    return backend;
}

// run_camera_plugin() as the main() of a plugin, on a thread of this process.
class PluginRun
{
public:
    PluginRun(Backend backend, const std::vector<std::string>& arguments, const std::string& program = "camera_rig")
        : m_backend(std::move(backend))
    {
        m_arguments.push_back(program);
        m_arguments.insert(m_arguments.end(), arguments.begin(), arguments.end());

        // A plugin is stopped with a signal, and here the signal is aimed at the plugin's
        // thread. That thread is born with it blocked, so the plugin's own wait is the only
        // place that can take it, however early it is sent.
        sigset_t stop_signals;
        sigemptyset(&stop_signals);
        sigaddset(&stop_signals, SIGINT);
        sigaddset(&stop_signals, SIGTERM);
        sigset_t previous;
        pthread_sigmask(SIG_BLOCK, &stop_signals, &previous);
        m_thread = std::thread(
            [this]
            {
                std::vector<char*> argv;
                for (std::string& argument : m_arguments)
                {
                    argv.push_back(argument.data());
                }
                argv.push_back(nullptr);
                m_exit_code = run_camera_plugin(static_cast<int>(m_arguments.size()), argv.data(), m_backend);
                m_ended = true;
            });
        pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    }

    ~PluginRun()
    {
        if (m_thread.joinable())
        {
            stop();
        }
    }

    PluginRun(const PluginRun&) = delete;
    PluginRun& operator=(const PluginRun&) = delete;

    // Waits for the plugin to return by itself and gives its exit code: -2 if it still runs
    // after `timeout`.
    int wait(std::chrono::seconds timeout = 30s)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!m_ended)
        {
            if (std::chrono::steady_clock::now() > deadline)
            {
                return -2;
            }
            std::this_thread::sleep_for(5ms);
        }
        m_thread.join();
        return m_exit_code;
    }

    // Stops the plugin as the plugin manager stops a process, and gives its exit code.
    int stop()
    {
        if (!m_ended)
        {
            pthread_kill(m_thread.native_handle(), SIGINT);
        }
        m_thread.join();
        return m_exit_code;
    }

private:
    Backend m_backend;
    std::vector<std::string> m_arguments;
    std::thread m_thread;
    std::atomic<bool> m_ended{ false };
    int m_exit_code = -1; // written by the thread before m_ended, read after it
};

// What a reader that looks for cameras finds, once a plugin serves. A plugin that was just
// started needs a moment; one that never serves gives an empty list.
std::vector<GroupListing> list_when_served(const PluginLocation& plugin = {})
{
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        try
        {
            std::vector<GroupListing> groups = CameraReader::list(plugin);
            if (!groups.empty())
            {
                return groups;
            }
        }
        catch (const CameraReaderError&)
        {
            // The plugin that was named does not listen yet.
        }
        std::this_thread::sleep_for(10ms);
    }
    return {};
}

struct Ended
{
    int exit_code = -1;
    std::string terminal; // what the plugin wrote for the operator: its usage, a refused argument
};

// Runs a plugin that is expected to return by itself, and keeps what it wrote to the terminal.
Ended run_to_end(const Backend& backend,
                 const std::vector<std::string>& arguments,
                 const std::string& program = "camera_rig")
{
    std::ostringstream terminal;
    std::streambuf* const output = std::cout.rdbuf(terminal.rdbuf());
    std::streambuf* const errors = std::cerr.rdbuf(terminal.rdbuf());
    Ended ended;
    {
        PluginRun plugin(backend, arguments, program);
        ended.exit_code = plugin.wait();
    }
    std::cout.rdbuf(output);
    std::cerr.rdbuf(errors);
    ended.terminal = terminal.str();
    return ended;
}

} // namespace

TEST_CASE("a plugin's instance name is its backend's, unless it is given another", "[unit][camera]")
{
    const UserDirectories user;
    Opened opened;
    {
        // Neither the plugin nor the reader is told where the other is.
        PluginRun plugin(standin_backend("rig", opened), { "--camera=head", "--plugin-root-id=ignored" });
        const std::vector<GroupListing> groups = list_when_served();
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].group.id == "head");
        REQUIRE(groups[0].endpoint == user.endpoint("camera-rig.sock"));
        REQUIRE(CameraReader::list({ .instance = "rig" }).size() == 1);
        REQUIRE(plugin.stop() == 0);
    }
    REQUIRE_FALSE(std::filesystem::exists(user.endpoint("camera-rig.sock")));
    {
        // A second plugin of the same backend takes another name.
        PluginRun plugin(standin_backend("rig", opened), { "--camera=head", "--instance=bench" });
        const std::vector<GroupListing> groups = list_when_served();
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].endpoint == user.endpoint("camera-bench.sock"));
        REQUIRE(plugin.stop() == 0);
    }
    {
        // A socket file that is named outright is the endpoint, wherever it is.
        const std::string elsewhere = user.file("elsewhere.sock");
        PluginRun plugin(standin_backend("rig", opened), { "--camera=head", "--endpoint=" + elsewhere });
        const std::vector<GroupListing> groups = list_when_served({ .endpoint = elsewhere });
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].endpoint == elsewhere);
        REQUIRE(CameraReader::list().empty());
        REQUIRE(plugin.stop() == 0);
    }

    // A camera of the command line is the plugin's own: it carries the backend's name.
    const std::vector<std::vector<CameraConfig>> calls = opened.calls();
    REQUIRE(calls.size() == 3);
    REQUIRE(calls[0].size() == 1);
    REQUIRE(calls[0][0].id == "head");
    REQUIRE(calls[0][0].backend == "rig");
}

TEST_CASE("a plugin opens its backend once, with the cameras a shared file names for it", "[unit][camera]")
{
    // "rig" stands for an SDK that serves all its cameras from one object, which it opens once.
    // The file is also the one of the plugin for "single", whose camera is not this plugin's.
    const UserDirectories user;
    const std::string file = user.write("rig.yaml",
                                        "cameras:\n"
                                        "  left:\n"
                                        "    backend: rig\n"
                                        "    serial: 17\n"
                                        "  top:\n"
                                        "    backend: single\n"
                                        "  right:\n"
                                        "    backend: rig\n"
                                        "slots: 5\n");
    Opened opened;
    PluginRun plugin(standin_backend("rig", opened), { "--config=" + file });
    const std::vector<GroupListing> groups = list_when_served();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "left");
    REQUIRE(groups[1].group.id == "right");
    REQUIRE(groups[0].group.streams[0].slot_count == 5);

    const std::vector<std::vector<CameraConfig>> calls = opened.calls();
    REQUIRE(calls.size() == 1);
    REQUIRE(calls[0].size() == 2);
    REQUIRE(calls[0][0].id == "left");
    REQUIRE(calls[0][0].options.at("serial") == "17");
    REQUIRE(calls[0][1].id == "right");
    REQUIRE(plugin.stop() == 0);
}

TEST_CASE("a plugin that is not told which cameras to serve reads the user's file, or looks for cameras", "[unit][camera]")
{
    const UserDirectories user;
    Opened opened;
    Backend backend = standin_backend("rig", opened);
    backend.discover = [] {
        return std::vector<CameraConfig>{ { "rig-17", "rig", { { "serial", "17" } } }, { "rig-18", "rig", {} } };
    };

    {
        // No file: the cameras the backend finds plugged in, as it described them.
        PluginRun plugin(backend, {});
        const std::vector<GroupListing> groups = list_when_served();
        REQUIRE(groups.size() == 2);
        REQUIRE(groups[0].group.id == "rig-17");
        REQUIRE(groups[1].group.id == "rig-18");
        REQUIRE(opened.calls().back().at(0).options.at("serial") == "17");
        REQUIRE(plugin.stop() == 0);
    }

    user.write("config/isaaccapture/cameras.yaml",
               "cameras:\n"
               "  head:\n"
               "    backend: rig\n"
               "  depth:\n"
               "    backend: other\n");
    {
        // The user's file says which cameras, and this plugin's are the ones of its backend.
        PluginRun plugin(backend, {});
        const std::vector<GroupListing> groups = list_when_served();
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].group.id == "head");
        REQUIRE(plugin.stop() == 0);
    }
    {
        // --auto sets the file aside.
        PluginRun plugin(backend, { "--auto" });
        REQUIRE(list_when_served().size() == 2);
        REQUIRE(plugin.stop() == 0);
    }

    // A backend with nothing to look for has no camera to serve then.
    REQUIRE(run_to_end(standin_backend("rig", opened), { "--auto" }).exit_code == 1);
}

TEST_CASE("a plugin that cannot run ends with an error and serves nothing", "[unit][camera]")
{
    const UserDirectories user;
    Opened opened;
    const Backend backend = standin_backend("rig", opened);

    // A file whose cameras are all other plugins', and a file that is not there.
    const std::string others = user.write("others.yaml", "cameras:\n  head:\n    backend: rgi\n");
    REQUIRE(run_to_end(backend, { "--config=" + others }).exit_code == 1);
    REQUIRE(run_to_end(backend, { "--config=" + user.file("missing.yaml") }).exit_code == 1);
    REQUIRE(opened.calls().empty());

    // A backend that does not return a camera for each one it was given.
    Backend losing = backend;
    losing.open = [](std::span<const CameraConfig>) { return std::vector<std::unique_ptr<CameraSource>>(); };
    REQUIRE(run_to_end(losing, { "--camera=head" }).exit_code == 1);

    // A backend that refuses a camera's options, and a pool too small for any reader.
    Backend refusing = backend;
    refusing.open = [](std::span<const CameraConfig> cameras) -> std::vector<std::unique_ptr<CameraSource>>
    { throw std::invalid_argument("camera '" + cameras.front().id + "' (rig backend): there is no option 'colour'"); };
    REQUIRE(run_to_end(refusing, { "--camera=head:colour=red" }).exit_code == 1);
    REQUIRE(run_to_end(backend, { "--camera=head", "--slots=1" }).exit_code == 1);

    // An endpoint another plugin has taken: the second plugin finds out before it opens a camera.
    PluginRun first(backend, { "--camera=head" });
    REQUIRE(list_when_served().size() == 1);
    const std::size_t opens = opened.calls().size();
    REQUIRE(run_to_end(backend, { "--camera=wrist" }).exit_code == 1);
    REQUIRE(opened.calls().size() == opens);
    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].group.id == "head");
    REQUIRE(first.stop() == 0);
}

TEST_CASE("a plugin answers arguments it cannot use with its usage, under its own name", "[unit][camera]")
{
    const UserDirectories user;
    Opened opened;
    const Backend backend = standin_backend("rig", opened);
    const auto refused = [&](const std::vector<std::string>& arguments)
    {
        const Ended ended = run_to_end(backend, arguments, "/opt/plugins/camera_rig");
        REQUIRE(ended.exit_code == 2);
        REQUIRE_THAT(ended.terminal, ContainsSubstring("Usage: camera_rig ["));
        return ended.terminal;
    };

    REQUIRE_THAT(
        refused({ "--camera=head", "--frobnicate" }), ContainsSubstring("camera_rig: unknown argument '--frobnicate'"));
    // What follows a camera's id is options, each KEY=VALUE.
    REQUIRE_THAT(refused({ "--camera=head:rig" }), ContainsSubstring("'rig' is not KEY=VALUE"));
    REQUIRE_THAT(refused({ "--camera=head", "--endpoint=" + user.endpoint("x.sock"), "--instance=other" }),
                 ContainsSubstring("both say where to listen"));
    REQUIRE_THAT(refused({ "--auto", "--camera=head" }), ContainsSubstring("give one kind"));
    REQUIRE_THAT(refused({ "--camera=head", "--slots=many" }), ContainsSubstring("--slots takes a whole number"));
    REQUIRE(opened.calls().empty());

    const Ended help = run_to_end(backend, { "--help" }, "/opt/plugins/camera_rig");
    REQUIRE(help.exit_code == 0);
    REQUIRE_THAT(help.terminal, ContainsSubstring("Usage: camera_rig [") && ContainsSubstring("the 'rig' backend") &&
                                    ContainsSubstring("a stand-in for a vendor's capture code"));
}
