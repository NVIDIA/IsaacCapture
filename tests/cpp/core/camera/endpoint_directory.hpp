// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera/endpoint_path.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

// This user's endpoint directory for one test. XDG_RUNTIME_DIR points at a short temporary
// directory, so a reader that looks through the user's camera plugins finds what the test put
// there and nothing else. What the variable held is put back afterwards.
class EndpointDirectory
{
public:
    EndpointDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_find_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_runtime = pattern;
        if (const char* previous = std::getenv("XDG_RUNTIME_DIR"))
        {
            m_previous = previous;
        }
        ::setenv("XDG_RUNTIME_DIR", m_runtime.c_str(), 1);
        // As a plugin does before it takes its endpoint.
        core::camera::prepare_endpoint_directory(core::camera::default_endpoint_path("synthetic"));
    }

    ~EndpointDirectory()
    {
        if (m_previous)
        {
            ::setenv("XDG_RUNTIME_DIR", m_previous->c_str(), 1);
        }
        else
        {
            ::unsetenv("XDG_RUNTIME_DIR");
        }
        std::error_code ignored;
        std::filesystem::remove_all(m_runtime, ignored);
    }

    EndpointDirectory(const EndpointDirectory&) = delete;
    EndpointDirectory& operator=(const EndpointDirectory&) = delete;

    // The directory a reader looks through.
    std::string path() const
    {
        return (m_runtime / "isaaccapture").string();
    }

private:
    std::filesystem::path m_runtime;
    std::optional<std::string> m_previous;
};
