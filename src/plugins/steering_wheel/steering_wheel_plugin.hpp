// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/schema_pusher.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace steering_wheel
{

struct SteeringWheelAxisMapping
{
    int steering_axis = 0;
    int throttle_axis = 1;
    int brake_axis = 2;
    int clutch_axis = -1;
};

/*!
 * @brief Reads a Linux joystick (e.g. /dev/input/js0), maps axes/buttons to
 *        SteeringWheelOutput, and pushes it via OpenXR SchemaPusher.
 */
class SteeringWheelPlugin
{
public:
    SteeringWheelPlugin(const std::string& device_path,
                        const std::string& collection_id,
                        SteeringWheelAxisMapping axis_mapping);
    ~SteeringWheelPlugin();

    void update();

private:
    bool open_device();
    void close_device();
    void push_current_state();
    float axis_value(int axis_index) const;
    void resize_state_from_device();

    std::string device_path_;
    int device_fd_ = -1;
    SteeringWheelAxisMapping axis_mapping_;
    std::vector<float> axes_;
    std::vector<uint8_t> buttons_;
    std::array<int, 2> hat_ = { 0, 0 };

    std::shared_ptr<core::OpenXRSession> session_;
    core::SchemaPusher pusher_;
};

} // namespace steering_wheel
} // namespace plugins
