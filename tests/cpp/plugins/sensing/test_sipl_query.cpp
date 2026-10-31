// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The SIPL query API resolves a vendor platform config to pipeline indices, and
// those indices are NOT the numbers the JSON appears to name. For SHW5G_2 the
// GMSL link indices, the CSI virtual channels and the JSON sensorInfo.id are all
// 2 and 3, while the pipeline indices SetPipelineCfg and --add-stream sensor=N
// take are 0 and 1. In S56C_1_SHF3L_2 they coincide, so this is exactly the kind
// of renumbering that hides until someone points a camera at the wrong eye.
//
// No hardware: the query API only parses the UDDF driver database and the JSON,
// so this runs on a rig with the cameras unplugged. The config is the in-tree
// copy, so it does not need the vendor package either.

#include "core/sipl_camera.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifndef SENSING_PLATFORM_CONFIG
#    error "SENSING_PLATFORM_CONFIG must be defined by the build"
#endif

using namespace plugins::sensing;

namespace
{

int g_failures = 0;

#define CHECK(cond, ...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                           \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                         \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

} // namespace

int main()
{
    const std::string config = SENSING_PLATFORM_CONFIG;
    std::printf("platform config: %s\n", config.c_str());

    const std::vector<uint32_t> masks{ 0x0000, 0x1100 };
    const auto sensors = SiplCamera::query(config, "SHW5G_2", masks);

    CHECK(sensors.size() == 2, "expected 2 sensors in SHW5G_2, got %zu", sensors.size());
    if (sensors.size() != 2)
    {
        return 1;
    }

    // The load-bearing assertion. If the query ever starts echoing the JSON's
    // sensorInfo.id instead of renumbering, this catches it here rather than as
    // a swapped or missing eye at runtime.
    CHECK(sensors[0].id == 0, "first pipeline index is %u, expected 0", sensors[0].id);
    CHECK(sensors[1].id == 1, "second pipeline index is %u, expected 1", sensors[1].id);

    for (const auto& s : sensors)
    {
        std::printf("  sensor=%u %s %ux%u @ %.2f fps\n", s.id, s.name.c_str(), s.width, s.height, s.fps);
        CHECK(s.width == 2560, "sensor %u width is %u, expected 2560", s.id, s.width);
        CHECK(s.height == 1984, "sensor %u height is %u, expected 1984", s.id, s.height);
        CHECK(s.fps > 59.0 && s.fps < 61.0, "sensor %u fps is %.2f, expected 60", s.id, s.fps);
        CHECK(!s.name.empty(), "sensor %u has an empty module name", s.id);
    }

    // An empty mask must yield no modules rather than silently falling back to
    // every link on the deserializer.
    const auto none = SiplCamera::query(config, "SHW5G_2", { 0x0000, 0x0000 });
    CHECK(none.empty(), "masking every link left %zu sensor(s)", none.size());

    if (g_failures == 0)
    {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
}
