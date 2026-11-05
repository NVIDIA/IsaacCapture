// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/rh_frame.hpp"

#include <revohuman_plugin/tactile.hpp>

#include <stdexcept>
#include <string>

namespace plugins
{
namespace revohuman
{

static_assert(RH_JOINT_COUNT == kGloveJointCount, "SDK joint count");
static_assert(RH_TACTILE_MAX_POINTS == kMaxTaxels, "SDK taxel limit");
static_assert(shape_size_bound(kHandJointCount, RH_TACTILE_POINTS_TYPE_B) <= kShapeMaxBytes,
              "the largest glove taxel layout must fit one TactileMetadata");

GloveSample from_rh_frame(const RhGloveFrame& frame, const std::array<double, kGloveJointCount>& joints_rad)
{
    if (frame.struct_size < sizeof(RhGloveFrame))
    {
        throw std::invalid_argument("RhGloveFrame.struct_size " + std::to_string(frame.struct_size) +
                                    " is smaller than " + std::to_string(sizeof(RhGloveFrame)));
    }
    GloveSample out;
    if (frame.hand == RH_HAND_LEFT)
    {
        out.hand = Hand::Left;
    }
    else if (frame.hand == RH_HAND_RIGHT)
    {
        out.hand = Hand::Right;
    }
    else
    {
        throw std::invalid_argument("RhGloveFrame.hand " + std::to_string(frame.hand) + " is not left or right");
    }
    out.generation = frame.generation;
    out.device_tick = frame.device_tick;
    out.joints_valid_mask = frame.joints_valid_mask;
    out.has_sample_time = (frame.flags & RH_FRAME_FLAG_HAS_SAMPLE_TIME) != 0;
    out.sample_mono_ns = frame.sample_mono_ns;
    out.joints_rad = joints_rad;
    if (frame.flags & RH_FRAME_FLAG_HAS_TACTILE)
    {
        const RhTactile& src = frame.tactile;
        if (src.point_count > RH_TACTILE_MAX_POINTS)
        {
            throw std::invalid_argument("RhTactile.point_count " + std::to_string(src.point_count) +
                                        " exceeds the protocol limit");
        }
        TactileSample t;
        t.device_tick = src.device_tick;
        t.sample_mono_ns = src.sample_mono_ns;
        t.has_sample_time = src.has_sample_time != 0;
        t.valid = src.valid != 0;
        t.calibrated = src.calibrated != 0;
        t.values.assign(src.values, src.values + src.point_count);
        out.tactile = std::move(t);
    }
    return out;
}

} // namespace revohuman
} // namespace plugins
