// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace test_support
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        // Canonicalize platform aliases such as macOS /tmp before no-follow checks.
        const auto root = std::filesystem::canonical(std::filesystem::temp_directory_path());
#ifndef _WIN32
        auto pattern = (root / "isaaccapture-test-XXXXXX").string();
        if (::mkdtemp(pattern.data()) == nullptr)
        {
            throw std::system_error(errno, std::generic_category(), "mkdtemp");
        }
        // mkdtemp creates the directory atomically with owner-only access.
        m_path = pattern;
#else
        const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int suffix = 0; suffix < 100; ++suffix)
        {
            m_path = root / ("isaaccapture-test-" + std::to_string(seed) + "_" + std::to_string(suffix));
            if (std::filesystem::create_directory(m_path))
            {
                return;
            }
        }
        throw std::runtime_error("Failed to create temporary test directory");
#endif
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const
    {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

} // namespace test_support
