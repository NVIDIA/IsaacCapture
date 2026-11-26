// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <realhand_ffg_glove/realhand_ffg_glove_skeleton.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

namespace realhand_ffg_test_support
{
using plugins::realhand_ffg_glove::kNumSensors;
inline YAML::Node calibration(float origin = 0.0f, float span = 1.0f)
{
    YAML::Node data;
    data["format"] = "realhand-ffg-glove-human-calibration-v1";
    for (const auto* side : { "left", "right" })
    {
        for (const auto* pose : { "open", "fist", "thumb_curl", "thumb_opposition", "spread" })
        {
            for (std::size_t sensor = 0; sensor < kNumSensors; ++sensor)
            {
                data["hands"][side][pose].push_back(origin + (std::string(pose) == "open" ? 0.0f : span));
            }
        }
    }
    return data;
}

struct CalibrationFile
{
    std::filesystem::path path;
    CalibrationFile()
    {
        char name[] = "/tmp/realhand-ffg-human-XXXXXX";
        const int fd = mkstemp(name);
        if (fd < 0)
            throw std::runtime_error("create temporary calibration");
        close(fd);
        path = name;
    }
    ~CalibrationFile()
    {
        std::filesystem::remove(path);
    }
    void save(const YAML::Node& data) const
    {
        std::ofstream stream(path);
        stream << data;
        if (!stream.good())
            throw std::runtime_error("write calibration");
    }
};

}
