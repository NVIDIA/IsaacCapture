// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A throwaway sysfs + /dev pair for the SpaceMouse discovery and connection tests.

#include <sys/stat.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#ifndef _WIN32
        auto pattern = (std::filesystem::temp_directory_path() / "spacemouse_test_XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        if (::mkdtemp(buffer.data()) == nullptr)
        {
            throw std::runtime_error("Failed to create temporary SpaceMouse test directory");
        }
        // mkdtemp prevents another user from pre-creating the test root.
        root_ = buffer.data();
#else
        root_ = std::filesystem::temp_directory_path() /
                ("spacemouse_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root_);
#endif
        std::filesystem::create_directories(class_dir());
        std::filesystem::create_directories(dev_dir());
    }
    ~FakeDevices()
    {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }
    FakeDevices(const FakeDevices&) = delete;
    FakeDevices& operator=(const FakeDevices&) = delete;

    std::filesystem::path class_dir() const
    {
        return root_ / "class" / "hidraw";
    }
    std::filesystem::path dev_dir() const
    {
        return root_ / "dev";
    }

    //! Registers `hidraw_name` with `HID_NAME=hid_name`; also creates its device node as a FIFO.
    std::filesystem::path add(const std::string& hidraw_name, const std::string& hid_name)
    {
        const auto device_dir = class_dir() / hidraw_name / "device";
        std::filesystem::create_directories(device_dir);
        std::ofstream(device_dir / "uevent") << "DRIVER=hid-generic\nHID_NAME=" << hid_name << "\nHID_PHYS=usb\n";
        const auto node = dev_dir() / hidraw_name;
        ::mkfifo(node.c_str(), 0600);
        return node;
    }

private:
    std::filesystem::path root_;
};

} // namespace spacemouse_test
