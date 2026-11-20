// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Config.h"

#include <iostream>
#include <stdexcept>

int main()
{
    zed_sender::Config c;
    zed_sender::validateConfig(c);
    c.width = 640;
    c.height = 360;
    c.fps = 30;
    zed_sender::validateConfig(c);
    unsigned failed = 0;
    auto reject = [&](zed_sender::Config bad)
    {
        try
        {
            zed_sender::validateConfig(bad);
            ++failed;
        }
        catch (const std::runtime_error&)
        {
        }
    };
    auto bad = c;
    bad.fps = 61;
    reject(bad);
    bad = c;
    bad.width = 641;
    reject(bad);
    bad = c;
    bad.height = 480;
    reject(bad);
    bad = c;
    bad.bitrateMbps = 0;
    reject(bad);
    bad = c;
    bad.port = 65536;
    reject(bad);
    bad = c;
    bad.rotation = 90;
    reject(bad);
    bad = c;
    bad.timeoutMs = 0;
    reject(bad);
    bad = c;
    bad.width = 2560;
    bad.height = 1440;
    reject(bad);
    if (failed)
        std::cerr << failed << " invalid configurations accepted\n";
    return failed ? 1 : 0;
}
