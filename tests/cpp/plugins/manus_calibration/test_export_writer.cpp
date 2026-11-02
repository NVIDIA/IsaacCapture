// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <manus_calibration/export_writer.hpp>
#include <sys/resource.h>
#include <sys/stat.h>

#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using plugins::manus::calibration::PosixExportWriter;

// A unique directory removed with its contents when the test ends.
class TempDir
{
public:
    TempDir()
    {
        std::string pattern = (fs::temp_directory_path() / "manus_cal_writer_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TempDir()
    {
        // Permissions may have been dropped by a test; restore them so the cleanup can descend.
        std::error_code ignored;
        fs::permissions(m_path, fs::perms::owner_all, fs::perm_options::add, ignored);
        fs::remove_all(m_path, ignored);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::string str() const
    {
        return m_path.string();
    }

    fs::path operator/(const std::string& name) const
    {
        return m_path / name;
    }

private:
    fs::path m_path;
};

std::vector<uint8_t> read_file(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<std::string> names_in(const fs::path& dir)
{
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        names.push_back(entry.path().filename().string());
    }
    return names;
}

} // namespace

TEST_CASE("directory_usable accepts only an existing writable directory", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;

    CHECK(writer.directory_usable(dir.str()));
    CHECK_FALSE(writer.directory_usable((dir / "missing").string()));

    std::ofstream(dir / "plain_file") << "x";
    CHECK_FALSE(writer.directory_usable((dir / "plain_file").string()));
}

TEST_CASE("directory_usable rejects a directory the worker cannot write to", "[unit][export_writer]")
{
    if (::geteuid() == 0)
    {
        SKIP("root ignores directory permissions");
    }
    TempDir dir;
    fs::create_directory(dir / "locked");
    fs::permissions(dir / "locked", fs::perms::owner_read | fs::perms::owner_exec);

    CHECK_FALSE(PosixExportWriter{}.directory_usable((dir / "locked").string()));
}

TEST_CASE("write publishes the exact bytes under the final name and leaves no .part file", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;
    const std::vector<uint8_t> payload = { 0, 1, 2, 0xFF, 0x00, 0x7F };

    REQUIRE_FALSE(writer.write(dir.str(), "att_left.mcal", payload).has_value());

    CHECK(read_file(dir / "att_left.mcal") == payload);
    CHECK(names_in(dir.str()) == std::vector<std::string>{ "att_left.mcal" });

    struct stat info
    {
    };
    REQUIRE(::stat((dir / "att_left.mcal").c_str(), &info) == 0);
    // Calibration data is private to the user that ran the worker.
    CHECK((info.st_mode & 0777) == 0600);
}

TEST_CASE("write handles an empty payload and a large payload", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;

    REQUIRE_FALSE(writer.write(dir.str(), "empty.mcal", {}).has_value());
    CHECK(fs::file_size(dir / "empty.mcal") == 0);

    std::vector<uint8_t> large(3 * 1024 * 1024);
    for (size_t i = 0; i < large.size(); ++i)
    {
        large[i] = static_cast<uint8_t>(i * 31);
    }
    REQUIRE_FALSE(writer.write(dir.str(), "large.mcal", large).has_value());
    CHECK(read_file(dir / "large.mcal") == large);
}

TEST_CASE("write refuses a basename that could leave the staging directory", "[unit][export_writer]")
{
    TempDir dir;
    fs::create_directory(dir / "inner");
    PosixExportWriter writer;

    CHECK(writer.write((dir / "inner").string(), "", { 1 }).has_value());
    CHECK(writer.write((dir / "inner").string(), "../escape.mcal", { 1 }).has_value());
    CHECK(writer.write((dir / "inner").string(), "sub/file.mcal", { 1 }).has_value());
    CHECK(writer.write((dir / "inner").string(), "/abs.mcal", { 1 }).has_value());

    CHECK(names_in(dir.str()) == std::vector<std::string>{ "inner" });
    CHECK(names_in(dir / "inner").empty());
}

TEST_CASE("write fails cleanly when the directory does not exist", "[unit][export_writer]")
{
    TempDir dir;
    const auto error = PosixExportWriter{}.write((dir / "missing").string(), "a.mcal", { 1 });

    REQUIRE(error.has_value());
    CHECK_THAT(*error, ContainsSubstring("a.mcal.part"));
}

TEST_CASE("write does not touch a stale .part file or write through a planted symlink", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;

    std::ofstream(dir / "stale.mcal.part") << "old";
    CHECK(writer.write(dir.str(), "stale.mcal", { 9 }).has_value());
    CHECK(read_file(dir / "stale.mcal.part") == std::vector<uint8_t>{ 'o', 'l', 'd' });
    CHECK_FALSE(fs::exists(dir / "stale.mcal"));

    std::ofstream(dir / "target") << "victim";
    fs::create_symlink(dir / "target", dir / "link.mcal.part");
    CHECK(writer.write(dir.str(), "link.mcal", { 9 }).has_value());
    CHECK(read_file(dir / "target") == std::vector<uint8_t>{ 'v', 'i', 'c', 't', 'i', 'm' });
}

TEST_CASE("write never overwrites an existing final file", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;
    const std::vector<uint8_t> original = { 1, 2, 3 };

    REQUIRE_FALSE(writer.write(dir.str(), "keep.mcal", original).has_value());
    const auto second = writer.write(dir.str(), "keep.mcal", { 9, 9, 9, 9 });

    CHECK(second.has_value());
    CHECK(read_file(dir / "keep.mcal") == original);
    CHECK(names_in(dir.str()) == std::vector<std::string>{ "keep.mcal" });
}

TEST_CASE("write reports a publish failure and removes the .part file", "[unit][export_writer]")
{
    TempDir dir;
    // Anything already on the final name makes the publish step fail.
    fs::create_directory(dir / "taken.mcal");
    std::ofstream(dir / "taken.mcal" / "inside") << "x";

    const auto error = PosixExportWriter{}.write(dir.str(), "taken.mcal", { 1, 2, 3 });

    REQUIRE(error.has_value());
    CHECK_THAT(*error, ContainsSubstring("publish"));
    CHECK_FALSE(fs::exists(dir / "taken.mcal.part"));
    CHECK(fs::is_directory(dir / "taken.mcal"));
}

TEST_CASE("write reports a write failure and removes the .part file", "[unit][export_writer]")
{
    TempDir dir;
    PosixExportWriter writer;

    // A file-size limit turns the write into EFBIG without needing a full disk.
    struct rlimit original
    {
    };
    REQUIRE(::getrlimit(RLIMIT_FSIZE, &original) == 0);
    void (*previous)(int) = std::signal(SIGXFSZ, SIG_IGN);
    struct rlimit limited = original;
    limited.rlim_cur = 4096;
    if (::setrlimit(RLIMIT_FSIZE, &limited) != 0)
    {
        std::signal(SIGXFSZ, previous);
        SKIP("cannot lower RLIMIT_FSIZE here");
    }

    const auto error = writer.write(dir.str(), "big.mcal", std::vector<uint8_t>(1024 * 1024, 0xAB));

    ::setrlimit(RLIMIT_FSIZE, &original);
    std::signal(SIGXFSZ, previous);

    REQUIRE(error.has_value());
    CHECK_THAT(*error, ContainsSubstring("cannot write"));
    CHECK(names_in(dir.str()).empty());
}
