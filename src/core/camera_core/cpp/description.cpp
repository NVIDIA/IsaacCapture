// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_core/types.hpp"

#include <cmath>

namespace core::camera
{

namespace
{

// Whether every plane of a pixel stream lies inside its slot. Written with subtractions and a
// division: the numbers may come from another process, and a sum of them could wrap.
bool planes_fit(const StreamDescription& stream)
{
    for (std::uint32_t i = 0; i < stream.plane_count; ++i)
    {
        const PlaneSize size = plane_size_of(stream.pixel_format, stream.width, stream.height, i);
        const std::uint64_t offset = stream.planes[i].offset_bytes;
        const std::uint64_t stride = stream.planes[i].stride_bytes;
        if (size.rows == 0 || size.row_bytes == 0 || stride < size.row_bytes)
        {
            return false;
        }
        // offset + stride * (rows - 1) + row_bytes <= slot_bytes
        if (offset > stream.slot_bytes || size.row_bytes > stream.slot_bytes - offset ||
            size.rows - 1 > (stream.slot_bytes - offset - size.row_bytes) / stride)
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::string problem_with(const GroupDescription& group)
{
    if (group.id.empty())
    {
        return "a camera group needs an id";
    }
    const std::string group_name = "camera group '" + group.id + "'";
    if (group.streams.empty() || group.streams.size() > MAX_STREAMS_PER_GROUP)
    {
        return group_name + " needs between 1 and " + std::to_string(MAX_STREAMS_PER_GROUP) + " streams";
    }
    for (std::size_t i = 0; i < group.streams.size(); ++i)
    {
        const StreamDescription& stream = group.streams[i];
        if (stream.id.empty())
        {
            return group_name + " has a stream without an id";
        }
        const std::string name = "camera stream '" + stream.id + "'";
        for (std::size_t other = 0; other < i; ++other)
        {
            if (group.streams[other].id == stream.id)
            {
                return group_name + " has two streams called '" + stream.id + "'";
            }
        }
        if (stream.slot_count == 0 || stream.slot_count > MAX_SLOTS_PER_STREAM)
        {
            return name + " needs between 1 and " + std::to_string(MAX_SLOTS_PER_STREAM) + " slots";
        }
        if (stream.slot_bytes == 0 || stream.slot_bytes > MAX_SLOT_BYTES)
        {
            return name + " needs a slot size between 1 and " + std::to_string(MAX_SLOT_BYTES) + " bytes";
        }
        if (!std::isfinite(stream.frame_rate) || stream.frame_rate < 0.0)
        {
            return name + " needs a frame rate that is a number, 0 when it is not known";
        }
        if (stream.is_encoded())
        {
            if (stream.plane_count != 0)
            {
                return name + " is encoded video, which has no planes";
            }
            continue;
        }
        if (stream.width == 0 || stream.height == 0)
        {
            return name + " needs a width and a height";
        }
        if (stream.plane_count > MAX_PLANES_PER_IMAGE || stream.plane_count != plane_count_of(stream.pixel_format))
        {
            return name + " does not describe every plane of its pixel format";
        }
        if (!planes_fit(stream))
        {
            return name + " has a plane whose rows are shorter than its width needs or do not lie inside the slot";
        }
    }
    return {};
}

} // namespace core::camera
