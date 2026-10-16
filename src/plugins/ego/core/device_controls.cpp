// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "device_controls.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace plugins::ego
{

OBPropertyItem find_property(const std::shared_ptr<ob::Device>& device, const std::string& name)
{
    for (int index = 0; index < device->getSupportedPropertyCount(); ++index)
    {
        const auto item = device->getSupportedProperty(static_cast<uint32_t>(index));
        if (name == item.name)
            return item;
    }
    throw std::invalid_argument("Unsupported Ego property: " + name);
}

double read_property(const std::shared_ptr<ob::Device>& device, const OBPropertyItem& item)
{
    switch (item.type)
    {
    case OB_BOOL_PROPERTY:
        return device->getBoolProperty(item.id) ? 1.0 : 0.0;
    case OB_INT_PROPERTY:
        return device->getIntProperty(item.id);
    case OB_FLOAT_PROPERTY:
        return device->getFloatProperty(item.id);
    default:
        throw std::invalid_argument(std::string(item.name) + " is not a scalar property");
    }
}

void validate_property_value(const std::shared_ptr<ob::Device>& device,
                             const OBPropertyItem& item,
                             double value,
                             bool validate_step)
{
    if ((item.permission & OB_PERMISSION_WRITE) == 0)
        throw std::invalid_argument(std::string(item.name) + " is read-only");
    if (!std::isfinite(value))
        throw std::out_of_range(std::string(item.name) + " requires a finite value");
    switch (item.type)
    {
    case OB_BOOL_PROPERTY:
        if (value != 0.0 && value != 1.0)
            throw std::out_of_range(std::string(item.name) + " accepts only 0 or 1");
        break;
    case OB_INT_PROPERTY:
    {
        const auto range = device->getIntPropertyRange(item.id);
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
            throw std::out_of_range(std::string(item.name) + " value is outside the int32 range");
        const auto integer = static_cast<int32_t>(value);
        if (validate_step && range.max > range.min && range.step > range.max - range.min)
        {
            throw std::runtime_error(std::string(item.name) +
                                     " reports an invalid SDK range/step; refusing to change a property that "
                                     "cannot be restored safely");
        }
        if (value != integer || integer < range.min || integer > range.max ||
            (validate_step && range.step > 0 && (integer - range.min) % range.step != 0))
        {
            throw std::out_of_range(std::string(item.name) + " requested=" + std::to_string(value) + " range=[" +
                                    std::to_string(range.min) + "," + std::to_string(range.max) +
                                    "] step=" + std::to_string(range.step));
        }
        break;
    }
    case OB_FLOAT_PROPERTY:
    {
        const auto range = device->getFloatPropertyRange(item.id);
        const double tolerance =
            std::max(std::max(1.0, std::abs(value)) * 1e-6, static_cast<double>(std::abs(range.step)) * 1e-6);
        const double steps = range.step > 0 ? (value - range.min) / range.step : 0.0;
        const double nearest = range.min + std::round(steps) * range.step;
        if (value < range.min - tolerance || value > range.max + tolerance ||
            (validate_step && range.step > 0 && std::abs(value - nearest) > tolerance))
        {
            throw std::out_of_range(std::string(item.name) + " requested=" + std::to_string(value) + " range=[" +
                                    std::to_string(range.min) + "," + std::to_string(range.max) +
                                    "] step=" + std::to_string(range.step));
        }
        break;
    }
    default:
        throw std::invalid_argument(std::string(item.name) + " is not a scalar property");
    }
}

void write_property(const std::shared_ptr<ob::Device>& device, const OBPropertyItem& item, double value, bool validate_step)
{
    validate_property_value(device, item, value, validate_step);
    switch (item.type)
    {
    case OB_BOOL_PROPERTY:
        device->setBoolProperty(item.id, value != 0.0);
        break;
    case OB_INT_PROPERTY:
        device->setIntProperty(item.id, static_cast<int32_t>(value));
        break;
    case OB_FLOAT_PROPERTY:
        device->setFloatProperty(item.id, static_cast<float>(value));
        break;
    default:
        throw std::invalid_argument(std::string(item.name) + " is not a scalar property");
    }
}

void verify_property_readback(const std::shared_ptr<ob::Device>& device,
                              const OBPropertyItem& item,
                              double requested,
                              std::string_view operation)
{
    const double actual = read_property(device, item);
    const double tolerance = item.type == OB_FLOAT_PROPERTY ? std::max(1.0, std::abs(requested)) * 1e-6 : 0.0;
    if (std::abs(actual - requested) > tolerance)
    {
        throw std::runtime_error(std::string(item.name) + " " + std::string(operation) + " readback=" +
                                 std::to_string(actual) + " differs from requested=" + std::to_string(requested));
    }
}

void print_capabilities(const std::shared_ptr<ob::Device>& device)
{
    const auto info = device->getDeviceInfo();
    std::cout << "SDK version=" << ob::Version::getMajor() << "." << ob::Version::getMinor() << "."
              << ob::Version::getPatch() << " full=" << ob::Version::getVersion()
              << " stage=" << ob::Version::getStageVersion() << std::endl;
    std::cout << info->getName() << " uid=" << info->getUid() << " serial=" << info->getSerialNumber()
              << " firmware=" << info->getFirmwareVersion() << " vid=0x" << std::hex << info->getVid() << " pid=0x"
              << info->getPid() << std::dec << " usb=" << info->getConnectionType()
              << " global_timestamp_supported=" << std::boolalpha << device->isGlobalTimestampSupported() << std::endl;
    std::cout << "Profiles are per-sensor advertisements; capture validates the exact simultaneous combination."
              << std::endl;
    std::cout << "Certification policy: encoded profiles above 30 FPS are advertised but rejected by this build; "
                 "sustained recording validation must pass before they are enabled."
              << std::endl;
    const auto sensors = device->getSensorList();
    for (uint32_t sensor_index = 0; sensor_index < sensors->getCount(); ++sensor_index)
    {
        const auto sensor_type = sensors->getSensorType(sensor_index);
        std::cout << "Sensor " << ob::TypeHelper::convertOBSensorTypeToString(sensor_type) << std::endl;
        const auto profiles = device->getSensor(sensor_type)->getStreamProfileList();
        for (uint32_t profile_index = 0; profile_index < profiles->getCount(); ++profile_index)
        {
            const auto profile = profiles->getProfile(profile_index);
            std::cout << "  " << ob::TypeHelper::convertOBFormatTypeToString(profile->getFormat());
            if (profile->is<ob::VideoStreamProfile>())
            {
                const auto video = profile->as<ob::VideoStreamProfile>();
                std::cout << " " << video->getWidth() << "x" << video->getHeight() << "@" << video->getFps();
            }
            else if (profile->is<ob::AccelStreamProfile>())
            {
                const auto imu = profile->as<ob::AccelStreamProfile>();
                std::cout << " rate=" << ob::TypeHelper::convertOBIMUSampleRateTypeToValue(imu->getSampleRate())
                          << "Hz full_scale="
                          << ob::TypeHelper::convertOBAccelFullScaleRangeTypeToString(imu->getFullScaleRange());
            }
            else if (profile->is<ob::GyroStreamProfile>())
            {
                const auto imu = profile->as<ob::GyroStreamProfile>();
                std::cout << " rate=" << ob::TypeHelper::convertOBIMUSampleRateTypeToValue(imu->getSampleRate())
                          << "Hz full_scale="
                          << ob::TypeHelper::convertOBGyroFullScaleRangeTypeToString(imu->getFullScaleRange());
            }
            else if (profile->is<ob::AudioStreamProfile>())
            {
                const auto audio = profile->as<ob::AudioStreamProfile>();
                std::cout << " " << audio->getSampleRate() << "Hz " << audio->getChannelCount() << "ch "
                          << audio->getBitsPerSample() << "bit";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "Properties (ranges can depend on the most recently active video profile):" << std::endl;
    for (int index = 0; index < device->getSupportedPropertyCount(); ++index)
    {
        const auto item = device->getSupportedProperty(static_cast<uint32_t>(index));
        std::cout << "  " << item.name << " id=" << item.id << " type=" << item.type << " permission=" << item.permission;
        try
        {
            if (item.type == OB_INT_PROPERTY)
            {
                const auto range = device->getIntPropertyRange(item.id);
                std::cout << " range=[" << range.min << "," << range.max << "] step=" << range.step;
            }
            else if (item.type == OB_FLOAT_PROPERTY)
            {
                const auto range = device->getFloatPropertyRange(item.id);
                std::cout << " range=[" << range.min << "," << range.max << "] step=" << range.step;
            }
            if ((item.permission & OB_PERMISSION_READ) != 0 &&
                (item.type == OB_BOOL_PROPERTY || item.type == OB_INT_PROPERTY || item.type == OB_FLOAT_PROPERTY))
            {
                std::cout << " value=" << read_property(device, item);
            }
        }
        catch (const ob::Error&)
        {
        }
        std::cout << std::endl;
    }
}

} // namespace plugins::ego
