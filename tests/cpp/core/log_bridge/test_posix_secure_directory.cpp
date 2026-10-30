// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "posix_secure_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iterator>
#include <unistd.h>

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::array<char, 32> pattern{};
        constexpr char kPattern[] = "/tmp/isaaccapture-log-test-XXXXXX";
        std::copy(std::begin(kPattern), std::end(kPattern), pattern.begin());
        char* created = ::mkdtemp(pattern.data());
        REQUIRE(created != nullptr);
        path_ = created;
    }

    ~TemporaryDirectory()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

} // namespace

TEST_CASE("secure directory creation uses owner-only permissions", "[unit]")
{
    TemporaryDirectory temporary;
    const auto directory = temporary.path() / "nested" / "logs";

    REQUIRE(isaaccapture::detail::ensure_secure_directory(directory));

    struct ::stat info
    {
    };
    REQUIRE(::lstat(directory.c_str(), &info) == 0);
    REQUIRE((info.st_mode & 0777) == 0700);
    REQUIRE(info.st_uid == ::getuid());
}

TEST_CASE("secure directory creation rejects symlinked parents", "[unit]")
{
    TemporaryDirectory temporary;
    const auto destination = temporary.path() / "destination";
    const auto link = temporary.path() / "runtime";
    REQUIRE(std::filesystem::create_directory(destination));
    REQUIRE(::symlink(destination.c_str(), link.c_str()) == 0);

    REQUIRE_FALSE(isaaccapture::detail::ensure_secure_directory(link / "logs"));
    REQUIRE_FALSE(std::filesystem::exists(destination / "logs"));
}

TEST_CASE("secure directory creation rejects writable parents", "[unit]")
{
    TemporaryDirectory temporary;
    const auto shared = temporary.path() / "shared";
    REQUIRE(std::filesystem::create_directory(shared));
    REQUIRE(::chmod(shared.c_str(), 0777) == 0);

    REQUIRE_FALSE(isaaccapture::detail::ensure_secure_directory(shared / "logs"));
}
