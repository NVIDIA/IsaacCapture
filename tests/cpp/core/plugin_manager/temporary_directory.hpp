// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int suffix = 0; suffix < 100; ++suffix)
        {
            m_path = std::filesystem::temp_directory_path() /
                     ("isaacteleop_plugin_manager_" + std::to_string(seed) + "_" + std::to_string(suffix));
            if (std::filesystem::create_directory(m_path))
            {
                return;
            }
        }
        throw std::runtime_error("Failed to create temporary test directory");
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
