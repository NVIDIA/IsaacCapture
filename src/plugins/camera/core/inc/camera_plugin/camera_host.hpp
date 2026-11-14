// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_source.hpp"

#include <camera_core/publisher.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace plugins::camera
{

// Slots per stream unless a camera says otherwise: one for the newest frame set, one for the
// camera to write ahead, and six for readers' windows.
inline constexpr std::uint32_t DEFAULT_SLOT_COUNT = 8;

/**
 * @brief Finds a backend by name.
 * @throws std::invalid_argument naming the backends this build has, and how a build gets more.
 */
const Backend& find_backend(std::span<const Backend> backends, const std::string& name);

/**
 * @brief The cameras of one plugin process: gives each its pools and serves them all to
 * readers on one endpoint.
 */
class CameraHost
{
public:
    /**
     * @throws std::system_error if another process serves the endpoint or it cannot be bound.
     */
    explicit CameraHost(const std::string& endpoint_path);

    // Stops the cameras and the endpoint.
    ~CameraHost();

    CameraHost(const CameraHost&) = delete;
    CameraHost& operator=(const CameraHost&) = delete;

    /**
     * @brief Add a camera and allocate its pools. Call before start().
     * @param slot_count For streams the camera described without one.
     * @throws std::invalid_argument if the camera's description is not usable or its id is taken.
     */
    void add_camera(std::unique_ptr<CameraSource> source, std::uint32_t slot_count = DEFAULT_SLOT_COUNT);

    // Starts serving readers, then the cameras.
    void start();

    // Stops the cameras, then closes every reader.
    void stop();

    std::size_t camera_count() const;
    std::size_t reader_count() const;

    /**
     * @brief Log what each camera did since the previous call: frame sets published, and as a
     * warning, frame sets it had to drop for lack of a free slot and events a reader did not take.
     */
    void log_activity();

    // For counters and descriptions; publishing goes through each camera's FrameOutput.
    const core::camera::Publisher& publisher() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace plugins::camera
