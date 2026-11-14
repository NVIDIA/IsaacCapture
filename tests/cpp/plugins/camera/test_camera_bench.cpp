// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_plugin/camera_host.hpp>
#include <camera_plugin/options.hpp>
#include <camera_plugin_backends/backends.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sys/wait.h>

#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
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
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_bench_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    std::string file(const std::string& name) const
    {
        return (m_path / name).string();
    }

private:
    std::filesystem::path m_path;
};

struct BenchRun
{
    int exit_code = -1;
    std::string output; // what the bench printed, results and errors together
};

// Runs camera_bench to its end with its output going to a file.
BenchRun run_bench(const TemporaryDirectory& directory, const std::vector<std::string>& arguments)
{
    const std::string capture = directory.file("bench_output.txt");
    std::vector<std::string> owned = { CAMERA_BENCH_EXECUTABLE };
    owned.insert(owned.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (std::string& argument : owned)
    {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0)
    {
        const int file = ::open(capture.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        ::dup2(file, STDOUT_FILENO);
        ::dup2(file, STDERR_FILENO);
        ::execv(argv[0], argv.data());
        ::_exit(127);
    }
    int status = 0;
    REQUIRE(::waitpid(pid, &status, 0) == pid);

    BenchRun result;
    result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    std::ifstream stream(capture);
    result.output.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return result;
}

} // namespace

TEST_CASE("the bench reads a camera from another process and reports each step", "[unit][camera_plugin][process]")
{
    TemporaryDirectory directory;
    const std::string endpoint = directory.file("cameras.sock");
    const std::vector<Backend> backends = built_backends();
    const CameraConfig config = parse_camera_argument("head:synthetic:width=320,height=240,fps=200");
    CameraHost host(endpoint);
    host.add_camera(find_backend(backends, config.backend).open(config));
    host.start();

    const BenchRun listing = run_bench(directory, { "--endpoint=" + endpoint });
    REQUIRE(listing.exit_code == 0);
    REQUIRE_THAT(listing.output, ContainsSubstring("head: 2 stream(s)") && ContainsSubstring("head/left") &&
                                     ContainsSubstring("320x240 bgra8, 200.00 fps"));

    const BenchRun run =
        run_bench(directory, { "--endpoint=" + endpoint, "--group=head", "--frames=50", "--check-synthetic" });
    REQUIRE(run.exit_code == 0);
    REQUIRE_THAT(run.output, ContainsSubstring("read 50 frame sets"));
    REQUIRE_THAT(run.output, ContainsSubstring("capture -> sdk ready") && ContainsSubstring("sent -> reader") &&
                                 ContainsSubstring("capture -> reader"));
    REQUIRE_THAT(run.output, ContainsSubstring("synthetic pattern: 0 image(s) did not match"));

    const BenchRun in_order = run_bench(
        directory, { "--endpoint=" + endpoint, "--group=head", "--frames=20", "--window=3", "--order=in_order" });
    REQUIRE(in_order.exit_code == 0);
    REQUIRE_THAT(in_order.output, ContainsSubstring("read 20 frame sets"));

    // The plugin's own counters saw the bench's readers come, read and go.
    REQUIRE(host.publisher().stats(0).events_sent >= 70);
    host.log_activity();

    const BenchRun unknown = run_bench(directory, { "--endpoint=" + endpoint, "--group=chest" });
    REQUIRE(unknown.exit_code == 1);
    REQUIRE_THAT(unknown.output, ContainsSubstring("no group 'chest'"));

    REQUIRE(run_bench(directory, { "--endpoint=" + endpoint, "--instance=other" }).exit_code == 2);
    REQUIRE(run_bench(directory, { "--endpoint=" + endpoint, "--order=backwards" }).exit_code == 2);
    REQUIRE(run_bench(directory, { "--help" }).exit_code == 0);

    host.stop();
    const BenchRun nobody = run_bench(directory, { "--endpoint=" + endpoint, "--group=head" });
    REQUIRE(nobody.exit_code == 1);
    REQUIRE_THAT(nobody.output, ContainsSubstring("no camera plugin answers on"));
}
