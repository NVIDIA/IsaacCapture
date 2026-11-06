// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/camera_fault.hpp"

#include <string>

namespace plugins
{
namespace sensing
{

namespace
{

struct Signature
{
    /// A fragment of the text the component was seen to produce for that situation.
    std::string_view needle;
    std::string_view code;
    ProducerState state;
    std::string_view message;
};

constexpr ProducerState kCamera = ProducerState::CameraFault;
constexpr ProducerState kOther = ProducerState::Error;

constexpr std::string_view kPowerAndCables =
    "Check that each camera's power cable is connected and that its GMSL cable is seated at both ends.";

// Most specific first; the first needle found wins. Extend this table, not the logic, when a new failure is seen.
// Messages say what to check; the original text is appended by diagnose_failure().
constexpr Signature kSignatures[] = {
    // --- What the capture driver itself says about an absent camera (its log lines are attached to the failure) --
    { "OPEN_CIRCUIT", "camera_power_missing", kCamera,
      "A camera's power switch reports an open circuit: the camera is not drawing power. Connect the camera's power "
      "cable." },
    { "CIRCUIT_OPEN", "camera_power_missing", kCamera,
      "A camera's power switch reports an open circuit: the camera is not drawing power. Connect the camera's power "
      "cable." },
    { "lock timeout", "camera_link_not_locked", kCamera,
      "A camera's GMSL link did not lock. The camera is unpowered or its GMSL cable is not connected. Check its power "
      "cable and that its GMSL cable is seated at both ends." },
    { "DES_LOCK_ERR", "camera_link_not_locked", kCamera,
      "A camera's GMSL link lost lock. The camera lost power or its GMSL cable came loose. Check its power cable and "
      "that its GMSL cable is seated at both ends." },
    { "CAMERA_DETECT", "camera_not_detected", kCamera,
      "A camera was not detected. It is unpowered or its GMSL cable is not connected. Check its power cable and that "
      "its GMSL cable is seated at both ends." },
    { "Parse failed after", "camera_not_responding", kCamera,
      "A camera stopped answering while it was being set up. It is unpowered or its GMSL cable is not connected. Check "
      "its power cable and that its GMSL cable is seated at both ends." },
    { "SIPLStatus 8", "camera_partly_started", kCamera,
      "Some cameras started and at least one did not. Check the power cable and GMSL cable of the camera that is "
      "missing." },
    // --- A camera is gone or never came up ---------------------------------------------------------------------
    { "sensor control write failure", "camera_control_bus_failed", kCamera,
      "A camera's control bus is not answering (settings cannot be written to the sensor). The camera lost power or "
      "its GMSL cable came loose. Check its power cable and GMSL cable." },
    { "input stream failure", "camera_stream_failed", kCamera,
      "A camera's video stream is corrupt or missing. Check its GMSL cable and connectors for damage or a loose "
      "fit, and its power." },
    { "image processing failure", "camera_isp_failed", kCamera,
      "The camera image pipeline failed. Check the camera's cables; if it persists, restart the producer." },
    { "interrupt localization failure", "camera_hardware_fault", kCamera,
      "The camera hardware reported a fault that could not be traced. Power-cycle the cameras." },
    { "deserializer failure", "camera_link_failed", kCamera,
      "A GMSL deserializer reported a failure. A camera lost power or its GMSL cable came loose. Check each camera's "
      "power cable and GMSL cable." },
    { "serializer failure", "camera_link_failed", kCamera,
      "A camera's serializer stopped responding. The camera lost power or its GMSL cable came loose. Check its power "
      "cable and GMSL cable." },
    { "sensor failure", "camera_sensor_failed", kCamera,
      "A camera's image sensor reported a failure. Check the camera's power and cables; if it persists, power-cycle "
      "the camera." },
    { "link lock", "camera_link_not_locked", kCamera,
      "A camera's GMSL link did not lock. Check each camera's power cable and that its GMSL cable is seated at both "
      "ends." },
    { "Failed to detect", "camera_not_detected", kCamera,
      "A camera was not detected. Check each camera's power cable and that its GMSL cable is seated at both ends." },
    { "no frame for", "camera_no_frames", kCamera,
      "A camera stopped delivering frames. Check its power and GMSL cable; the producer restarts it automatically." },
    { "INvSIPLCamera::Init failed", "camera_start_failed", kCamera,
      "The camera system could not start. The usual cause is a camera with no power or no GMSL link. Check each "
      "camera's power cable and GMSL cable." },
    { "INvSIPLCamera::SetPlatformCfg failed", "camera_config_rejected", kCamera,
      "The camera system rejected the platform configuration. Check --platform-config, --config-name and "
      "--link-masks against the installed camera drivers." },
    { "INvSIPLCamera::Start failed", "camera_start_failed", kCamera,
      "The cameras initialised but would not start streaming. Check each camera's power and GMSL cable and try "
      "again; if it persists, power-cycle the cameras." },
    { "SIPLStatus 10", "camera_start_failed", kCamera,
      "The camera system reported an error. The usual cause is a camera with no power or no GMSL link. Check each "
      "camera's power cable and GMSL cable." },
    { "pipeline error", "camera_pipeline_error", kCamera,
      "A camera pipeline reported an error while streaming. Check each camera's power cable and GMSL cable." },
    { "capture error", "camera_capture_error", kCamera,
      "Capturing from a camera failed. Check each camera's power cable and GMSL cable." },
    { "no modules left in", "camera_link_mask_mismatch", kCamera,
      "No camera matches the configured link masks. Check --link-masks and --config-name against how the cameras "
      "are wired." },
    { "declares no sensors", "camera_config_empty", kCamera,
      "The platform configuration names no cameras. Check --platform-config and --config-name." },
    { "no SIPL pipeline with index", "camera_not_in_config", kCamera,
      "A requested sensor is not in the platform configuration. Run --list-sensors for the valid numbers." },
    { "link masks are required", "camera_config_missing", kCamera,
      "The camera configuration needs --link-masks (one mask per deserializer)." },
    { "cannot open NITO file", "camera_tuning_missing", kCamera,
      "The camera's image-processing tuning file is missing. Install the camera tuning files (see --nito-dir)." },
    { "short read on NITO file", "camera_tuning_invalid", kCamera,
      "The camera's image-processing tuning file is truncated or corrupt. Reinstall the camera tuning files." },
    { "Argus: no camera provider", "camera_service_down", kCamera,
      "The Argus camera service is not running. Start it with: sudo systemctl start nvargus-daemon." },
    { "Argus: no camera device", "camera_not_detected", kCamera,
      "The requested Argus camera was not found. Check the camera's cable and power and the --add-stream sensor "
      "number." },
    { "Argus: no camera", "camera_not_detected", kCamera,
      "The requested Argus camera was not found. Check the camera's cable and power and the --add-stream sensor "
      "number." },
    { "Argus:", "camera_error", kCamera,
      "The Argus camera stack reported an error. Check the camera's cable and power; the Argus service "
      "(nvargus-daemon) may need a restart." },

    // --- Frames could not reach the GPU or the encoder ----------------------------------------------------------
    { "NvBufSurfaceMapEglImage", "gpu_buffer_error", kCamera,
      "Camera frames could not be mapped for the GPU. This is usually the graphics environment: the producer must "
      "not run with DISPLAY set to a remote X server, and the Tegra EGL libraries must be installed." },
    { "NvmmNvSciBufToNvBufSurface", "gpu_buffer_error", kCamera,
      "Camera frames could not be converted for the GPU. Check that the Jetson multimedia libraries match the "
      "installed L4T release." },
    { "unsupported CUDA EGL frame type", "gpu_buffer_error", kCamera,
      "The camera delivered a frame format the GPU conversion does not support. Check the camera mode and L4T "
      "release." },
    { "CUDA", "gpu_error", kOther,
      "A CUDA call failed. Check that no other process is using the GPU and that the driver matches the installed "
      "L4T release." },
    { "createVideoEncoder failed", "encoder_unavailable", kOther,
      "The hardware video encoder is not available (/dev/v4l2-nvenc is missing or the driver is not loaded). Check "
      "the L4T installation." },
    { "JetsonEncoder", "encoder_error", kOther,
      "The hardware video encoder reported an error. Another process may be holding the encoder; if it persists, "
      "reduce --livekit-bitrate or the stream count." },

    // --- Publishing ---------------------------------------------------------------------------------------------
    { "cannot join the room", "livekit_unreachable", kOther,
      "The producer cannot join the LiveKit room. Check that the LiveKit server address (--livekit-url) is reachable "
      "from the Orin, the server is running, and the token in --livekit-token-file is valid and not expired." },
    { "cannot publish track", "livekit_publish_failed", kOther,
      "The LiveKit server refused a video track. Check that the token allows publishing and the track names are "
      "unique." },
    { "no local participant", "livekit_publish_failed", kOther,
      "The LiveKit session has no participant to publish from. Check the token in --livekit-token-file." },
    { "LiveKit", "livekit_error", kOther,
      "The LiveKit connection failed. Check the PC's address, the server and the network between the Orin and the "
      "PC." },
    { "bitrate must be", "config_invalid", kOther, "A setting is out of range. Check --livekit-bitrate." },
    { "must divide", "config_invalid", kOther, "A setting is out of range. Check --livekit-scale-div." },
    { "bad_alloc", "out_of_memory", kOther,
      "The producer ran out of memory. Close other programs on the Orin or reduce the number of streams." },
    { "Permission denied", "permission_denied", kOther,
      "The producer is not allowed to use a device or file. Check that its service user is in the video group." },
    { "No such file or directory", "file_missing", kOther,
      "A file or device the producer needs is missing. Check the paths in its configuration." },
    { "Device or resource busy", "device_busy", kOther,
      "A camera or encoder device is in use by another process. Stop any other camera program (for example a "
      "second copy of this producer) and try again." },
};

} // namespace

Diagnosis diagnose_failure(std::string_view what)
{
    for (const auto& signature : kSignatures)
    {
        if (what.find(signature.needle) != std::string_view::npos)
        {
            return { { std::string(signature.code), std::string(signature.message) +
                                                        " The producer retries automatically. (" + std::string(what) +
                                                        ")" },
                     signature.state };
        }
    }
    // Unknown text: still a code and the original words, and a camera fault because a start-up failure that nothing
    // else explains is most often the cameras.
    return { { "startup_failed",
               "The producer could not start. The producer retries automatically. (" + std::string(what) + ")" },
             ProducerState::CameraFault };
}

Fault diagnose_camera_failure(std::string_view what)
{
    return diagnose_failure(what).fault;
}

} // namespace sensing
} // namespace plugins
