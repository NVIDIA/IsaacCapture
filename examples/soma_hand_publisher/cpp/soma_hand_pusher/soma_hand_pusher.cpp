// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/soma_hand_joint_poses_generated.h>
#include <schema/soma_hand_joint_rotations_generated.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

enum class HandRepresentation
{
    JointRotations,
    JointPoses,
};

struct Arguments
{
    bool validate_only = false;
    HandRepresentation representation = HandRepresentation::JointRotations;
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
        else if (argument == "--hand-representation" && index + 1 < argc)
        {
            const std::string value = argv[++index];
            if (value == "joint-rotations")
            {
                arguments.representation = HandRepresentation::JointRotations;
            }
            else if (value == "joint-poses")
            {
                arguments.representation = HandRepresentation::JointPoses;
            }
            else
            {
                throw std::invalid_argument("Unknown SOMA hand representation: " + value);
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
    for (size_t index = 0; index < size; ++index)
    {
        value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
    }
    return value;
}

template <typename Table>
const Table* verify(const std::vector<uint8_t>& bytes)
{
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!verifier.VerifyBuffer<Table>(nullptr))
    {
        throw std::invalid_argument("Invalid SOMA hand FlatBuffer");
    }
    return flatbuffers::GetRoot<Table>(bytes.data());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        std::cout << "Usage: soma_hand_pusher [--hand-representation joint-rotations|joint-poses] "
                     "[--validate-only]\n"
                     "Reads paired, framed SOMA hand FlatBuffers from stdin.\n";
        return 0;
    }
    auto logger = isaaccapture::Logger::get("isaaccapture.examples.soma_hand_pusher");
    try
    {
        const auto arguments = parse_arguments(argc, argv);
        const bool joint_poses = arguments.representation == HandRepresentation::JointPoses;
        const size_t max_flatbuffer_size = joint_poses ? 2048 : 1024;
        const char* tensor_identifier = joint_poses ? "soma_hand_joint_poses" : "soma_hand_joint_rotations";
        std::unique_ptr<core::OpenXRSession> session;
        std::unique_ptr<core::SchemaPusher> left_pusher;
        std::unique_ptr<core::SchemaPusher> right_pusher;
        size_t frames = 0;
        std::array<uint8_t, 16> header{};
        while (std::cin.read(reinterpret_cast<char*>(header.data()), header.size()))
        {
            const auto left_size = little_endian(header.data(), 4);
            const auto right_size = little_endian(header.data() + 4, 4);
            const auto raw_time = little_endian(header.data() + 8, 8);
            if (left_size == 0 || left_size > max_flatbuffer_size || right_size == 0 ||
                right_size > max_flatbuffer_size || raw_time > INT64_MAX)
            {
                throw std::invalid_argument("Invalid SOMA hand demo packet header");
            }
            std::vector<uint8_t> left_bytes(left_size);
            std::vector<uint8_t> right_bytes(right_size);
            if (!std::cin.read(reinterpret_cast<char*>(left_bytes.data()), left_bytes.size()) ||
                !std::cin.read(reinterpret_cast<char*>(right_bytes.data()), right_bytes.size()))
            {
                throw std::invalid_argument("Truncated SOMA hand demo payload");
            }

            core::SomaHandedness left_handedness;
            core::SomaHandedness right_handedness;
            if (joint_poses)
            {
                left_handedness = verify<core::SomaHandJointPoses>(left_bytes)->handedness();
                right_handedness = verify<core::SomaHandJointPoses>(right_bytes)->handedness();
            }
            else
            {
                left_handedness = verify<core::SomaHandJointRotations>(left_bytes)->handedness();
                right_handedness = verify<core::SomaHandJointRotations>(right_bytes)->handedness();
            }
            if (left_handedness != core::SomaHandedness_LEFT || right_handedness != core::SomaHandedness_RIGHT)
            {
                throw std::invalid_argument("SOMA hand demo payload handedness does not match its collection");
            }

            if (!arguments.validate_only)
            {
                if (!left_pusher)
                {
                    session = std::make_unique<core::OpenXRSession>(
                        "SomaHandDemoPublisher", core::SchemaPusher::get_required_extensions(), false);
                    left_pusher = std::make_unique<core::SchemaPusher>(
                        session->get_handles(), core::SchemaPusherConfig{ .collection_id = "soma_hand_left_demo",
                                                                          .max_flatbuffer_size = max_flatbuffer_size,
                                                                          .tensor_identifier = tensor_identifier,
                                                                          .localized_name = "SOMA Left Hand Demo" });
                    right_pusher = std::make_unique<core::SchemaPusher>(
                        session->get_handles(), core::SchemaPusherConfig{ .collection_id = "soma_hand_right_demo",
                                                                          .max_flatbuffer_size = max_flatbuffer_size,
                                                                          .tensor_identifier = tensor_identifier,
                                                                          .localized_name = "SOMA Right Hand Demo" });
                }
                const auto now = core::os_monotonic_now_ns();
                left_pusher->push_buffer(left_bytes.data(), left_bytes.size(), now, static_cast<int64_t>(raw_time));
                right_pusher->push_buffer(right_bytes.data(), right_bytes.size(), now, static_cast<int64_t>(raw_time));
            }
            ++frames;
        }
        if (std::cin.gcount() != 0)
        {
            throw std::invalid_argument("Truncated SOMA hand demo packet header");
        }
        logger->info("{} {} paired SOMA hand demo frames", arguments.validate_only ? "Validated" : "Published", frames);
        return 0;
    }
    catch (const std::exception& error)
    {
        logger->error("{}", error.what());
        return 1;
    }
}
