// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <realhand_ffg_glove/realhand_ffg_glove_skeleton.hpp>
#include <realhand_ffg_test_support/calibration.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numbers>

using namespace plugins::realhand_ffg_glove;
using namespace realhand_ffg_test_support;

void write_skeleton_fixtures(const char* path)
{
    CalibrationFile file;
    file.save(calibration());
    const HumanHandSkeleton model(file.path.string());
    YAML::Node data;
    for (const auto side : { HandSide::Left, HandSide::Right })
    {
        const auto* name = side == HandSide::Left ? "left" : "right";
        for (const float flexion : { 0.0f, 0.5f, 1.0f })
        {
            std::array<float, kNumSensors> sensors{};
            sensors.fill(flexion);
            auto joints = *model.reconstruct(sensors, side);
            // Spread is tested independently; use neutral lateral channels in the curl sweep.
            for (std::size_t sensor = 5; sensor < kNumSensors; sensor += 4)
            {
                sensors[sensor] = 0.0f;
            }
            joints = *model.reconstruct(sensors, side);
            YAML::Node sample;
            sample["flexion"] = flexion;
            for (const auto& joint : joints)
            {
                const auto& p = joint.pose.position;
                const auto& q = joint.pose.orientation;
                YAML::Node location;
                location["position"] = std::vector<float>{ p.x, p.y, p.z };
                location["orientation"] = std::vector<float>{ q.x, q.y, q.z, q.w };
                location["radius"] = joint.radius;
                sample["joints"].push_back(location);
            }
            data[name].push_back(sample);
        }
    }
    std::ofstream stream(path);
    stream << data;
    if (!stream.good())
        throw std::runtime_error("write production skeleton fixtures");
}

void write_calibrated_skeleton_fixtures(const char* calibration_path, const char* output_path)
{
    const auto calibration_data = YAML::LoadFile(calibration_path);
    const HumanHandSkeleton model(calibration_path);
    YAML::Node data;
    for (const auto side : { HandSide::Left, HandSide::Right })
    {
        const auto* name = side == HandSide::Left ? "left" : "right";
        const auto poses = calibration_data["hands"][name];
        if (!poses)
        {
            continue;
        }
        const auto append =
            [&](const std::string& key, const std::array<float, kNumSensors>& sensors, const char* collection)
        {
            const auto joints = *model.reconstruct(sensors, side);
            YAML::Node sample;
            sample["name"] = key;
            for (float value : sensors)
            {
                sample["raw"].push_back(value);
            }
            for (const auto& joint : joints)
            {
                const auto& p = joint.pose.position;
                const auto& q = joint.pose.orientation;
                YAML::Node location;
                location["position"] = std::vector<float>{ p.x, p.y, p.z };
                location["orientation"] = std::vector<float>{ q.x, q.y, q.z, q.w };
                sample["joints"].push_back(location);
            }
            data[collection][name].push_back(sample);
        };
        for (std::size_t finger = 0; finger < 4; ++finger)
        {
            for (float delta : { -0.05f, 0.05f })
            {
                std::array<float, kNumSensors> sensors{};
                for (std::size_t i = 0; i < kNumSensors; ++i)
                {
                    sensors[i] = poses["open"][i].as<float>();
                }
                sensors[5 + 4 * finger] += (side == HandSide::Left ? delta : -delta);
                append(std::to_string(finger), sensors, "lateral");
            }
        }
        for (const auto* key :
             { "open", "fist", "thumb_curl", "thumb_index", "thumb_middle", "thumb_ring", "thumb_little" })
        {
            if (!poses[key])
            {
                continue;
            }
            std::array<float, kNumSensors> sensors{};
            for (std::size_t i = 0; i < kNumSensors; ++i)
            {
                sensors[i] = poses[key][i].as<float>();
            }
            append(key, sensors, "poses");
            for (int step = 0; step <= 100; ++step)
            {
                auto current = sensors;
                for (std::size_t i = 0; i < kNumSensors; ++i)
                {
                    const float origin = poses["open"][i].as<float>();
                    current[i] =
                        origin + (step / 100.0f) * std::remainder(sensors[i] - origin, 2.0f * std::numbers::pi_v<float>);
                }
                append(key + std::string("_") + std::to_string(step), current, "sweeps");
            }
            for (float shift : { -0.8f, 0.8f })
            {
                auto current = sensors;
                for (std::size_t i = 5; i < kNumSensors; ++i)
                {
                    current[i] += shift;
                }
                append(key + std::string("_other_fingers"), current, "independence");
            }
            for (float shift : { -2.0f * std::numbers::pi_v<float>, 2.0f * std::numbers::pi_v<float> })
            {
                auto current = sensors;
                for (float& value : current)
                {
                    value += shift;
                }
                append(key + std::string("_phase_wrap"), current, "phase_wrap");
            }
        }
    }
    std::ofstream stream(output_path);
    stream << data;
    if (!stream.good())
        throw std::runtime_error("write calibrated production skeleton fixtures");
}

int main(int argc, char** argv)
try
{
    if (argc == 3 && std::strcmp(argv[1], "--write-skeleton-fixtures") == 0)
        write_skeleton_fixtures(argv[2]);
    else if (argc == 4 && std::strcmp(argv[1], "--write-calibrated-skeleton-fixtures") == 0)
        write_calibrated_skeleton_fixtures(argv[2], argv[3]);
    else
        throw std::invalid_argument(
            "Expected --write-skeleton-fixtures OUTPUT or --write-calibrated-skeleton-fixtures CALIBRATION OUTPUT");
    return 0;
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "%s\\n", error.what());
    return 1;
}
