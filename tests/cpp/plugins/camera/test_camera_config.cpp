// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_plugin/config.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace plugins::camera;
using Catch::Matchers::ContainsSubstring;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_config_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    // Writes a configuration file and returns its path.
    std::string write(const std::string& text) const
    {
        const std::string path = (m_path / "cameras.yaml").string();
        std::ofstream(path) << text;
        return path;
    }

private:
    std::filesystem::path m_path;
};

// Sets an environment variable for one test and puts back what was there.
class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* previous = std::getenv(name))
        {
            m_previous = previous;
        }
        if (value != nullptr)
        {
            ::setenv(name, value, 1);
        }
        else
        {
            ::unsetenv(name);
        }
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

} // namespace

TEST_CASE("a configuration file names cameras, three lines each", "[unit][camera_plugin][config]")
{
    TemporaryDirectory directory;
    const PluginConfig config =
        load_config_file(directory.write("cameras:\n"
                                         "  head:\n"
                                         "    backend: synthetic\n"
                                         "  wrist:\n"
                                         "    backend: synthetic\n"
                                         "    width: 640\n"
                                         "    format: gray8\n"
                                         "slots: 12\n"));
    REQUIRE(config.slot_count == 12);
    REQUIRE(config.cameras.size() == 2);
    REQUIRE(config.cameras[0].id == "head"); // in the order of the file
    REQUIRE(config.cameras[0].backend == "synthetic");
    REQUIRE(config.cameras[0].options.empty());
    REQUIRE(config.cameras[1].id == "wrist");
    REQUIRE(config.cameras[1].options.size() == 2);
    REQUIRE(config.cameras[1].options.at("width") == "640");
    REQUIRE(config.cameras[1].options.at("format") == "gray8");

    // Without `slots:` the pools have the default size; a file may also name no camera yet.
    REQUIRE(load_config_file(directory.write("cameras:\n  head:\n    backend: synthetic\n")).slot_count ==
            DEFAULT_SLOT_COUNT);
    REQUIRE(load_config_file(directory.write("cameras: {}\n")).cameras.empty());
}

TEST_CASE("a configuration file that cannot be used says where and why", "[unit][camera_plugin][config]")
{
    TemporaryDirectory directory;
    const auto error_of = [&](const std::string& text)
    {
        try
        {
            (void)load_config_file(directory.write(text));
        }
        catch (const std::invalid_argument& error)
        {
            return std::string(error.what());
        }
        return std::string("no error");
    };

    REQUIRE_THAT(error_of("cameras:\n  head:\n    width: 640\n"),
                 ContainsSubstring("cameras.yaml:3: camera 'head' needs a backend"));
    REQUIRE_THAT(error_of("cameras:\n  head: synthetic\n"), ContainsSubstring("camera 'head' needs a backend"));
    REQUIRE_THAT(error_of("cameras:\n  head:\n    backend: synthetic\n    modes: [a, b]\n"),
                 ContainsSubstring("cameras.yaml:4: camera 'head' takes settings of the form `name: value`"));
    REQUIRE_THAT(error_of("camera:\n  head:\n    backend: synthetic\n"),
                 ContainsSubstring("cameras.yaml:1: there is no setting 'camera'"));
    REQUIRE_THAT(error_of("cameras:\n  - head\n"), ContainsSubstring("lists cameras by id"));
    REQUIRE_THAT(
        error_of("cameras: {}\nslots: many\n"), ContainsSubstring("cameras.yaml:2: `slots:` takes a whole number"));
    REQUIRE_THAT(error_of("- just\n- a list\n"), ContainsSubstring("starts with `cameras:`"));
    REQUIRE_THAT(error_of("cameras:\n  head: [unclosed\n"), ContainsSubstring("not valid YAML"));

    REQUIRE_THROWS_WITH(load_config_file("/nonexistent/cameras.yaml"), ContainsSubstring("cannot be read"));
}

TEST_CASE("the default configuration file is in the user's configuration directory", "[unit][camera_plugin][config]")
{
    {
        const ScopedEnvironment config_home("XDG_CONFIG_HOME", "/etc/xdg-test");
        REQUIRE(default_config_path() == "/etc/xdg-test/isaaccapture/cameras.yaml");
    }
    const ScopedEnvironment config_home("XDG_CONFIG_HOME", nullptr);
    const ScopedEnvironment home("HOME", "/home/somebody");
    REQUIRE(default_config_path() == "/home/somebody/.config/isaaccapture/cameras.yaml");
}

TEST_CASE("the backends that can look for cameras are asked what is plugged in", "[unit][camera_plugin][config]")
{
    const auto found = [](std::vector<CameraConfig> cameras) { return [cameras] { return cameras; }; };
    const std::vector<Backend> backends = {
        { "synthetic", "", {} }, // cannot look: no hardware behind it
        { "zed", "", {}, found({ { "zed-1234", "zed", { { "serial", "1234" } } } }) },
        { "oak", "", {}, found({}) },
        { "argus", "", {}, found({ { "argus-0", "argus", {} }, { "argus-1", "argus", {} } }) },
    };
    const std::vector<CameraConfig> cameras = discover_cameras(backends);
    REQUIRE(cameras.size() == 3);
    REQUIRE(cameras[0].id == "zed-1234");
    REQUIRE(cameras[0].options.at("serial") == "1234");
    REQUIRE(cameras[2].id == "argus-1");

    REQUIRE(discover_cameras(std::vector<Backend>{ { "synthetic", "", {} } }).empty());
}
