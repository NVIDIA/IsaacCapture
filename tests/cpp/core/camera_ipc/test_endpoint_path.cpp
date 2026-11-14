// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_ipc/endpoint_path.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sys/stat.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace core::camera;

namespace
{

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

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_path_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

} // namespace

TEST_CASE("the default endpoint is in the user's runtime directory", "[unit][camera_ipc][endpoint_path]")
{
    {
        const ScopedEnvironment runtime("XDG_RUNTIME_DIR", "/run/user/1234");
        REQUIRE(default_endpoint_path() == "/run/user/1234/isaaccapture/camera.sock");
        REQUIRE(default_endpoint_path("wrist_2") == "/run/user/1234/isaaccapture/camera-wrist_2.sock");
    }
    {
        const ScopedEnvironment runtime("XDG_RUNTIME_DIR", nullptr);
        const std::string user = std::to_string(::geteuid());
        REQUIRE(default_endpoint_path() == "/tmp/isaaccapture-" + user + "/camera.sock");
        REQUIRE(default_endpoint_path("b") == "/tmp/isaaccapture-" + user + "/camera-b.sock");
    }
    // An instance name becomes part of a file name, so it cannot carry a path.
    REQUIRE_THROWS_AS(default_endpoint_path("../other"), std::invalid_argument);
    REQUIRE_THROWS_AS(default_endpoint_path("a b"), std::invalid_argument);
}

TEST_CASE("an endpoint's directory is made for its owner alone", "[unit][camera_ipc][endpoint_path]")
{
    TemporaryDirectory directory;
    const std::filesystem::path made = directory.path / "isaaccapture";
    const std::string endpoint = (made / "camera.sock").string();

    prepare_endpoint_directory(endpoint);
    REQUIRE(std::filesystem::is_directory(made));
    REQUIRE(std::filesystem::status(made).permissions() == std::filesystem::perms::owner_all);
    prepare_endpoint_directory(endpoint); // already there and fine

    // One that others can enter is refused rather than used.
    REQUIRE(::chmod(made.c_str(), 0755) == 0);
    REQUIRE_THROWS_AS(prepare_endpoint_directory(endpoint), std::system_error);

    // So is something that is not a directory, and a directory that cannot be made.
    std::filesystem::remove(made);
    REQUIRE(::symlink(directory.path.c_str(), made.c_str()) == 0);
    REQUIRE_THROWS_AS(prepare_endpoint_directory(endpoint), std::system_error);
    REQUIRE_THROWS_AS(prepare_endpoint_directory((directory.path / "missing" / "deeper" / "camera.sock").string()),
                      std::system_error);

    // A bare file name is in the current directory, which is left as it is.
    REQUIRE_NOTHROW(prepare_endpoint_directory("camera.sock"));
}
