// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A throwaway sysfs + /dev pair for the SpaceMouse discovery and connection tests.

#include <sys/stat.h>
#include <test_support/temporary_directory.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace spacemouse_test
{

class FakeDevices
{
public:
    FakeDevices()
    {
        std::filesystem::create_directories(class_dir());
        std::filesystem::create_directories(dev_dir());
    }
    FakeDevices(const FakeDevices&) = delete;
    FakeDevices& operator=(const FakeDevices&) = delete;

    std::filesystem::path class_dir() const
    {
        return root_.path() / "class" / "hidraw";
    }
    std::filesystem::path dev_dir() const
    {
        return root_.path() / "dev";
    }

    //! Registers `hidraw_name` with `HID_NAME=hid_name` and, when given, a `report_descriptor`;
    //! also creates its device node as a FIFO.
    std::filesystem::path add(const std::string& hidraw_name,
                              const std::string& hid_name,
                              const std::vector<uint8_t>* report_descriptor = nullptr)
    {
        const auto device_dir = class_dir() / hidraw_name / "device";
        std::filesystem::create_directories(device_dir);
        std::ofstream(device_dir / "uevent") << "DRIVER=hid-generic\nHID_NAME=" << hid_name << "\nHID_PHYS=usb\n";
        if (report_descriptor)
            std::ofstream(device_dir / "report_descriptor", std::ios::binary)
                .write(reinterpret_cast<const char*>(report_descriptor->data()),
                       static_cast<std::streamsize>(report_descriptor->size()));
        const auto node = dev_dir() / hidraw_name;
        ::mkfifo(node.c_str(), 0600);
        return node;
    }

private:
    test_support::TemporaryDirectory root_;
};

} // namespace spacemouse_test
