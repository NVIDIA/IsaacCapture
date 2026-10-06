// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/soma_body_joint_poses_generated.h>
#include <schema/soma_body_joint_rotations_generated.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

enum class BodyRepresentation
{
    JointRotations,
    JointPoses,
};

struct Arguments
{
    bool validate_only = false;
    BodyRepresentation representation = BodyRepresentation::JointRotations;
};

Arguments parse_arguments(int argc, char** argv)
{
    Arguments arguments;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--validate-only")
        {
            arguments.validate_only = true;
        }
        else if (argument == "--body-representation" && index + 1 < argc)
        {
            const std::string value = argv[++index];
            if (value == "joint-rotations")
            {
                arguments.representation = BodyRepresentation::JointRotations;
            }
            else if (value == "joint-poses")
            {
                arguments.representation = BodyRepresentation::JointPoses;
            }
            else
            {
                throw std::invalid_argument("Unknown SOMA body representation: " + value);
            }
        }
        else
        {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
    return arguments;
}

uint64_t little_endian(const uint8_t* bytes, size_t size)
{
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i)
    {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        std::cout << "Usage: soma_body_pusher [--body-representation joint-rotations|joint-poses] "
                     "[--validate-only]\n"
                     "Reads framed SOMA body FlatBuffers from stdin.\n";
        return 0;
    }
    auto logger = isaaccapture::Logger::get("isaaccapture.examples.soma_body_pusher");
    try
    {
        const auto arguments = parse_arguments(argc, argv);
        const bool joint_poses = arguments.representation == BodyRepresentation::JointPoses;
        const size_t max_flatbuffer_size = joint_poses ? 4096 : 2048;
        const char* tensor_identifier = joint_poses ? "soma_body_joint_poses" : "soma_body_joint_rotations";
        std::unique_ptr<core::OpenXRSession> session;
        std::unique_ptr<core::SchemaPusher> pusher;
        size_t frames = 0;
        std::array<uint8_t, 12> header{};
        while (std::cin.read(reinterpret_cast<char*>(header.data()), header.size()))
        {
            // Local pipe framing only: byte count + clip-relative device time, both little-endian.
            const auto size = little_endian(header.data(), 4);
            const auto raw_time = little_endian(header.data() + 4, 8);
            if (size == 0 || size > max_flatbuffer_size || raw_time > INT64_MAX)
            {
                throw std::invalid_argument("Invalid SOMA body demo packet header");
            }
            std::vector<uint8_t> bytes(size);
            if (!std::cin.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
            {
                throw std::invalid_argument("Truncated SOMA body demo payload");
            }
            flatbuffers::Verifier verifier(bytes.data(), bytes.size());
            const bool valid = joint_poses ? verifier.VerifyBuffer<core::SomaBodyJointPoses>(nullptr) :
                                             verifier.VerifyBuffer<core::SomaBodyJointRotations>(nullptr);
            if (!valid)
            {
                throw std::invalid_argument("Invalid SOMA body FlatBuffer");
            }
            if (!arguments.validate_only)
            {
                if (!pusher)
                {
                    session = std::make_unique<core::OpenXRSession>(
                        "SomaBodyDemoPublisher", core::SchemaPusher::get_required_extensions(), false);
                    pusher = std::make_unique<core::SchemaPusher>(
                        session->get_handles(), core::SchemaPusherConfig{ .collection_id = "soma_demo",
                                                                          .max_flatbuffer_size = max_flatbuffer_size,
                                                                          .tensor_identifier = tensor_identifier,
                                                                          .localized_name = "SOMA Body Demo" });
                }
                pusher->push_buffer(
                    bytes.data(), bytes.size(), core::os_monotonic_now_ns(), static_cast<int64_t>(raw_time));
            }
            ++frames;
        }
        if (std::cin.gcount() != 0)
        {
            throw std::invalid_argument("Truncated SOMA body demo packet header");
        }
        logger->info("{} {} SOMA body demo frames", arguments.validate_only ? "Validated" : "Published", frames);
        return 0;
    }
    catch (const std::exception& error)
    {
        logger->error("{}", error.what());
        return 1;
    }
}
