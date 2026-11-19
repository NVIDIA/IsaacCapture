// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/endpoint_path.hpp>
#include <camera/message_socket.hpp>
#include <camera/unique_fd.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

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

// What a process that was killed leaves behind: a socket file with nothing behind it.
void leave_socket_file(const std::filesystem::path& path)
{
    sockaddr_un address = {};
    address.sun_family = AF_UNIX;
    REQUIRE(path.native().size() < sizeof(address.sun_path));
    std::memcpy(address.sun_path, path.c_str(), path.native().size());
    const UniqueFd dead(::socket(AF_UNIX, SOCK_SEQPACKET, 0));
    REQUIRE(::bind(dead.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
}

} // namespace

TEST_CASE("an endpoint is in the user's endpoint directory", "[unit][camera][endpoint_path]")
{
    {
        const ScopedEnvironment runtime("XDG_RUNTIME_DIR", "/run/user/1234");
        REQUIRE(default_endpoint_directory() == "/run/user/1234/isaaccapture");
        REQUIRE(default_endpoint_path("wrist_2") == "/run/user/1234/isaaccapture/camera-wrist_2.sock");
    }
    {
        const ScopedEnvironment runtime("XDG_RUNTIME_DIR", nullptr);
        const std::string user = std::to_string(::geteuid());
        REQUIRE(default_endpoint_directory() == "/tmp/isaaccapture-" + user);
        REQUIRE(default_endpoint_path("b") == "/tmp/isaaccapture-" + user + "/camera-b.sock");
    }
    // An instance name becomes part of a file name, so it cannot carry a path, and every
    // endpoint has one.
    REQUIRE_THROWS_AS(default_endpoint_path("../other"), std::invalid_argument);
    REQUIRE_THROWS_AS(default_endpoint_path("a b"), std::invalid_argument);
    REQUIRE_THROWS_AS(default_endpoint_path(""), std::invalid_argument);
}

TEST_CASE("an endpoint's directory is made for its owner alone", "[unit][camera][endpoint_path]")
{
    TemporaryDirectory directory;
    const std::filesystem::path made = directory.path / "isaaccapture";
    const std::string endpoint = (made / "camera-a.sock").string();

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
    REQUIRE_THROWS_AS(prepare_endpoint_directory((directory.path / "missing" / "deeper" / "camera-a.sock").string()),
                      std::system_error);

    // A bare file name is in the current directory, which is left as it is.
    REQUIRE_NOTHROW(prepare_endpoint_directory("camera-a.sock"));
}

TEST_CASE("the camera endpoints of this user are listed", "[unit][camera][endpoint_path]")
{
    TemporaryDirectory directory;
    const ScopedEnvironment runtime("XDG_RUNTIME_DIR", directory.path.c_str());
    const std::filesystem::path endpoints = directory.path / "isaaccapture";

    // No plugin has run for this user: nothing is listed, and looking does not make the directory.
    REQUIRE(default_endpoint_paths().empty());
    REQUIRE_FALSE(std::filesystem::exists(endpoints));

    prepare_endpoint_directory(default_endpoint_path("wrist_2"));
    REQUIRE(default_endpoint_paths().empty());

    // Two plugins that are running, one that has its endpoint and does not listen yet, and one
    // that was killed. Whether anything answers behind a socket file is for the caller to find out.
    MessageListener wrist(default_endpoint_path("wrist_2"));
    wrist.listen();
    MessageListener chest(default_endpoint_path("chest"));
    chest.listen();
    const MessageListener starting(default_endpoint_path("synthetic"));
    leave_socket_file(default_endpoint_path("gone"));

    // What else is in the directory: the listeners' lock files, a file that is not a socket, a
    // link to a socket under an endpoint's name, and sockets whose names no camera endpoint has.
    REQUIRE(std::filesystem::is_regular_file(default_endpoint_path("wrist_2") + ".lock"));
    std::ofstream(endpoints / "camera-file.sock") << "not a socket";
    std::filesystem::create_symlink(default_endpoint_path("gone"), endpoints / "camera-link.sock");
    for (const char* name :
         { "viewer-1.sock", "camera-1.lock", "camera_2.sock", "camera.sock", "camera-.sock", "camera-a.b.sock" })
    {
        leave_socket_file(endpoints / name);
    }

    const std::vector<std::string> sorted = { default_endpoint_path("chest"), default_endpoint_path("gone"),
                                              default_endpoint_path("synthetic"), default_endpoint_path("wrist_2") };
    REQUIRE(default_endpoint_paths() == sorted);
}

TEST_CASE("an endpoint directory that cannot be read is an error", "[unit][camera][endpoint_path]")
{
    TemporaryDirectory directory;
    const ScopedEnvironment runtime("XDG_RUNTIME_DIR", directory.path.c_str());
    std::ofstream(directory.path / "isaaccapture") << "not a directory";
    REQUIRE_THROWS_AS(default_endpoint_paths(), std::system_error);
}
