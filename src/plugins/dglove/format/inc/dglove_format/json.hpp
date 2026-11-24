// SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/minireflect.h>
#include <schema/dglove_generated.h>

#include <array>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>

namespace dglove
{
inline std::string frame_json(const uint8_t* buffer)
{
    struct Visitor : flatbuffers::ToStringVisitor
    {
        Visitor() : ToStringVisitor("", true, "")
        {
        }
        // Print the exact promoted float value so Python comparisons do not
        // confuse decimal formatting loss with changes to the source data.
        void Float(float value) override
        {
            Double(static_cast<double>(value));
        }
        void Double(double value) override
        {
            std::array<char, 64> digits{};
            const auto result = std::to_chars(digits.data(), digits.data() + digits.size(), value,
                                              std::chars_format::general, std::numeric_limits<double>::max_digits10);
            if (result.ec != std::errc{})
                throw std::runtime_error("Cannot format DGlove number");
            s.append(digits.data(), result.ptr);
        }
    } visitor;
    flatbuffers::IterateFlatBuffer(buffer, core::DGloveFrame::MiniReflectTypeTable(), &visitor);
    return visitor.s;
}
} // namespace dglove
