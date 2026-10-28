// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <libobsensor/ObSensor.hpp>

#include <memory>
#include <string>
#include <string_view>

namespace plugins::ego
{
OBPropertyItem find_property(const std::shared_ptr<ob::Device>& device, const std::string& name);
double read_property(const std::shared_ptr<ob::Device>& device, const OBPropertyItem& item);
void validate_property_value(const std::shared_ptr<ob::Device>& device,
                             const OBPropertyItem& item,
                             double value,
                             bool validate_step = true);
void write_property(const std::shared_ptr<ob::Device>& device,
                    const OBPropertyItem& item,
                    double value,
                    bool validate_step = true);
void verify_property_readback(const std::shared_ptr<ob::Device>& device,
                              const OBPropertyItem& item,
                              double requested,
                              std::string_view operation);
void print_capabilities(const std::shared_ptr<ob::Device>& device);
} // namespace plugins::ego
