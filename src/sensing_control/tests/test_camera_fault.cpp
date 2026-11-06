// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <sensing_control/camera_fault.hpp>

#include <string>
#include <vector>

using plugins::sensing::diagnose_camera_failure;
using plugins::sensing::diagnose_failure;
using plugins::sensing::ProducerState;

namespace
{

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

bool ends_with(const std::string& text, const std::string& suffix)
{
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct Expectation
{
    std::string text;
    std::string code;
    ProducerState state;
};

constexpr ProducerState kCamera = ProducerState::CameraFault;
constexpr ProducerState kError = ProducerState::Error;

// One representative text per signature in camera_fault.cpp, in table order.
const std::vector<Expectation>& expectations()
{
    static const std::vector<Expectation> table = {
        { "Severity 2 : OPEN_CIRCUIT on GPIO 4", "camera_power_missing", kCamera },
        { "fault: CIRCUIT_OPEN_DISCONNECTION", "camera_power_missing", kCamera },
        { "deserializer lock timeout on link 1", "camera_link_not_locked", kCamera },
        { "DES_LOCK_ERR reported", "camera_link_not_locked", kCamera },
        { "CAMERA_DETECT failed for module 2", "camera_not_detected", kCamera },
        { "Parse failed after 3 retries", "camera_not_responding", kCamera },
        { "Init returned SIPLStatus 8", "camera_partly_started", kCamera },
        { "sensor control write failure on sensor 3", "camera_control_bus_failed", kCamera },
        { "input stream failure on pipeline 0", "camera_stream_failed", kCamera },
        { "image processing failure on pipeline 1", "camera_isp_failed", kCamera },
        { "interrupt localization failure", "camera_hardware_fault", kCamera },
        { "deserializer failure on link 0", "camera_link_failed", kCamera },
        { "serializer failure on link 0", "camera_link_failed", kCamera },
        { "sensor failure on sensor 2", "camera_sensor_failed", kCamera },
        { "link lock lost on deserializer 1", "camera_link_not_locked", kCamera },
        { "Failed to detect camera on link 2", "camera_not_detected", kCamera },
        { "no frame for 5 s from sensor 2 (stream sensing-left)", "camera_no_frames", kCamera },
        { "INvSIPLCamera::Init failed", "camera_start_failed", kCamera },
        { "INvSIPLCamera::SetPlatformCfg failed", "camera_config_rejected", kCamera },
        { "INvSIPLCamera::Start failed", "camera_start_failed", kCamera },
        { "Init failed with SIPLStatus 10", "camera_start_failed", kCamera },
        { "pipeline error on sensor 1", "camera_pipeline_error", kCamera },
        { "capture error on sensor 1", "camera_capture_error", kCamera },
        { "no modules left in the link masks", "camera_link_mask_mismatch", kCamera },
        { "configuration declares no sensors", "camera_config_empty", kCamera },
        { "no SIPL pipeline with index 9", "camera_not_in_config", kCamera },
        { "link masks are required", "camera_config_missing", kCamera },
        { "cannot open NITO file /var/nito/x.nito", "camera_tuning_missing", kCamera },
        { "short read on NITO file", "camera_tuning_invalid", kCamera },
        { "Argus: no camera provider", "camera_service_down", kCamera },
        { "Argus: no camera device 2", "camera_not_detected", kCamera },
        { "Argus: no camera at index 3", "camera_not_detected", kCamera },
        { "Argus: session failed", "camera_error", kCamera },
        { "NvBufSurfaceMapEglImage failed", "gpu_buffer_error", kCamera },
        { "NvmmNvSciBufToNvBufSurface failed", "gpu_buffer_error", kCamera },
        { "unsupported CUDA EGL frame type 7", "gpu_buffer_error", kCamera },
        { "CUDA error 700 in cudaMemcpy", "gpu_error", kError },
        { "createVideoEncoder failed", "encoder_unavailable", kError },
        { "JetsonEncoder: dequeue timed out", "encoder_error", kError },
        { "LiveKit: cannot join the room", "livekit_unreachable", kError },
        { "LiveKit: cannot publish track sensing-left", "livekit_publish_failed", kError },
        { "LiveKit: no local participant", "livekit_publish_failed", kError },
        { "LiveKit: disconnected", "livekit_error", kError },
        { "bitrate must be positive", "config_invalid", kError },
        { "scale must divide the frame size", "config_invalid", kError },
        { "std::bad_alloc", "out_of_memory", kError },
        { "open /dev/video0: Permission denied", "permission_denied", kError },
        { "open /dev/foo: No such file or directory", "file_missing", kError },
        { "open /dev/video1: Device or resource busy", "device_busy", kError },
    };
    return table;
}

} // namespace

TEST_CASE("every signature maps its text to the intended code and state", "[camera_fault]")
{
    for (const auto& e : expectations())
    {
        INFO("text: " << e.text);
        const auto d = diagnose_failure(e.text);
        CHECK(d.fault.code == e.code);
        CHECK(d.state == e.state);
        CHECK_FALSE(d.fault.message.empty());
        CHECK(ends_with(d.fault.message, "(" + e.text + ")"));
        CHECK(diagnose_camera_failure(e.text).code == e.code);
    }
}

TEST_CASE("known capture failures are explained and keep the original text", "[camera_fault]")
{
    const std::string sipl = "INvSIPLCamera::Init failed with SIPLStatus 10";
    const auto a = diagnose_failure(sipl);
    CHECK(a.fault.code == "camera_start_failed");
    CHECK(a.state == ProducerState::CameraFault);
    CHECK(contains(a.fault.message, "power cable"));
    CHECK(contains(a.fault.message, "retries automatically"));
    CHECK(contains(a.fault.message, sipl));
}

TEST_CASE("the first matching signature wins", "[camera_fault]")
{
    // Both the generic SIPL status and the driver's own explanation: the explanation is listed first.
    CHECK(diagnose_failure("Init failed with SIPLStatus 10; OPEN_CIRCUIT").fault.code == "camera_power_missing");
    CHECK(diagnose_failure("SIPLStatus 10 CIRCUIT_OPEN_DISCONNECTION").fault.code == "camera_power_missing");
    CHECK(diagnose_failure("link lock failed, SIPLStatus 10").fault.code == "camera_link_not_locked");
    // A LiveKit text naming the room beats the generic LiveKit entry.
    CHECK(diagnose_failure("LiveKit: cannot join the room").fault.code == "livekit_unreachable");
    // A driver NACK line attached to a LiveKit failure does not change the mapping.
    CHECK(diagnose_failure("LiveKit: cannot join the room; I2C: NACK ... errno=121").fault.code ==
          "livekit_unreachable");
    // "Argus:" alone is the last Argus entry, not the first.
    CHECK(diagnose_failure("Argus: no camera provider").fault.code == "camera_service_down");
    // Camera signatures are listed before the generic CUDA one.
    CHECK(diagnose_failure("unsupported CUDA EGL frame type 3").state == ProducerState::CameraFault);
}

TEST_CASE("an I2C NACK alone is not a signature", "[camera_fault]")
{
    // The driver prints NACK lines on healthy starts too.
    for (const char* text : { "I2C: NACK", "CAMI2C_ERROR_BUS_NACK on 0x1a", "I2C: NACK from address 0x36 errno=121" })
    {
        INFO(text);
        CHECK(diagnose_failure(text).fault.code == "startup_failed");
    }
}

TEST_CASE("an unknown failure is a startup_failed camera fault with the original text", "[camera_fault]")
{
    const std::string text = "something entirely new (errno 5)";
    const auto d = diagnose_failure(text);
    CHECK(d.fault.code == "startup_failed");
    CHECK(d.state == ProducerState::CameraFault);
    CHECK(contains(d.fault.message, "retries automatically"));
    CHECK(ends_with(d.fault.message, "(" + text + ")"));

    const auto empty = diagnose_failure("");
    CHECK(empty.fault.code == "startup_failed");
    CHECK(ends_with(empty.fault.message, "()"));
}
