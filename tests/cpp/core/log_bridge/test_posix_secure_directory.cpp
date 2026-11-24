// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "posix_secure_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sys/stat.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = "/tmp/isaaccapture-log-test-XXXXXX";
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

    const auto previous_umask = ::umask(0);
    const bool created = isaaccapture::detail::ensure_secure_directory(directory);
    ::umask(previous_umask);
    REQUIRE(created);

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

TEST_CASE("secure directory creation distinguishes sticky ancestors from targets", "[unit]")
{
    TemporaryDirectory temporary;
    const auto shared = temporary.path() / "shared";
    REQUIRE(std::filesystem::create_directory(shared));
    REQUIRE(::chmod(shared.c_str(), 01777) == 0);

    REQUIRE(isaaccapture::detail::ensure_secure_directory(shared / "logs"));
    REQUIRE_FALSE(isaaccapture::detail::ensure_secure_directory(shared));
}

TEST_CASE("secure directory creation rejects traversal before creating children", "[unit]")
{
    TemporaryDirectory temporary;
    const auto child = temporary.path() / "new";
    REQUIRE_FALSE(isaaccapture::detail::ensure_secure_directory(child / ".." / "logs"));
    REQUIRE_FALSE(std::filesystem::exists(child));
}

TEST_CASE("secure directory creation accepts trailing separators and preserves existing permissions", "[unit]")
{
    TemporaryDirectory temporary;
    const auto directory = temporary.path() / "logs";
    REQUIRE(std::filesystem::create_directory(directory));
    REQUIRE(::chmod(directory.c_str(), 0755) == 0);
    REQUIRE(isaaccapture::detail::ensure_secure_directory(directory / ""));
    struct ::stat info
    {
    };
    REQUIRE(::lstat(directory.c_str(), &info) == 0);
    REQUIRE((info.st_mode & 0777) == 0755);
}

#ifdef __linux__
TEST_CASE("secure directory traversal does not require ancestor read permission", "[unit]")
{
    if (::getuid() == 0)
    {
        SKIP("root bypasses directory permissions");
    }
    TemporaryDirectory temporary;
    const auto ancestor = temporary.path() / "searchable";
    REQUIRE(std::filesystem::create_directory(ancestor));
    REQUIRE(std::filesystem::create_directory(ancestor / "existing"));
    REQUIRE(::chmod(ancestor.c_str(), 0300) == 0);
    const bool existing = isaaccapture::detail::ensure_secure_directory(ancestor / "existing");
    const bool created = isaaccapture::detail::ensure_secure_directory(ancestor / "new");
    REQUIRE(::chmod(ancestor.c_str(), 0700) == 0);
    REQUIRE(existing);
    REQUIRE(created);
}
#endif
