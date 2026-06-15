// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "steering_wheel_plugin.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <thread>

using namespace plugins::steering_wheel;

namespace
{

int parse_axis_arg(int argc, char** argv, int index, int default_value)
{
    return (argc > index) ? std::stoi(argv[index]) : default_value;
}

} // namespace

int main(int argc, char** argv)
try
{
    if (argc == 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <device_path> <collection_id> <steering_axis> <throttle_axis> <brake_axis> <clutch_axis>"
                  << std::endl;
        return 1;
    }

    const std::string device_path = (argc > 1) ? argv[1] : "/dev/input/js0";
    const std::string collection_id = (argc > 2) ? argv[2] : "steering_wheel";
    SteeringWheelAxisMapping axis_mapping;
    axis_mapping.steering_axis = parse_axis_arg(argc, argv, 3, axis_mapping.steering_axis);
    axis_mapping.throttle_axis = parse_axis_arg(argc, argv, 4, axis_mapping.throttle_axis);
    axis_mapping.brake_axis = parse_axis_arg(argc, argv, 5, axis_mapping.brake_axis);
    axis_mapping.clutch_axis = parse_axis_arg(argc, argv, 6, axis_mapping.clutch_axis);

    std::cout << "Steering Wheel (device: " << device_path << ", collection: " << collection_id
              << ", steering_axis: " << axis_mapping.steering_axis << ", throttle_axis: " << axis_mapping.throttle_axis
              << ", brake_axis: " << axis_mapping.brake_axis << ", clutch_axis: " << axis_mapping.clutch_axis << ")"
              << std::endl;

    SteeringWheelPlugin plugin(device_path, collection_id, axis_mapping);

    const auto frame_duration = std::chrono::nanoseconds(1000000000 / 90);
    const auto program_start = std::chrono::steady_clock::now();
    std::size_t frame_count = 0;

    while (true)
    {
        plugin.update();
        frame_count++;
        std::this_thread::sleep_until(program_start + frame_duration * frame_count);
    }

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error" << std::endl;
    return 1;
}
