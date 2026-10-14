# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera source adapters feeding NV12 in NVMM to the shared Jetson encoder."""


class V4L2Source:
    def __init__(self, spec):
        if spec.get("fourcc", "YUYV") != "YUYV":
            raise ValueError("gstreamer V4L2 camera requires fourcc: YUYV")
        self.spec = spec

    def pipeline(self, width, height, fps):
        return (
            "v4l2src name=camera do-timestamp=true ! "
            f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps} ! "
            "videoconvert ! video/x-raw,format=I420 ! nvvidconv ! "
            "video/x-raw(memory:NVMM),format=NV12 ! "
        )

    def configure(self, element):
        element.set_property("device", str(self.spec["device"]))


class ArgusSource:
    startup_timeout = 20

    def __init__(self, spec):
        self.sensor_id = spec.get("sensor_id", 0)
        self.sensor_mode = spec.get("sensor_mode", -1)
        self.flip_method = spec.get("flip_method", 0)
        for name, value, lower, upper in (
            ("sensor_id", self.sensor_id, 0, 255),
            ("sensor_mode", self.sensor_mode, -1, 255),
            ("flip_method", self.flip_method, 0, 7),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")

    def pipeline(self, width, height, fps):
        caps = (
            "video/x-raw(memory:NVMM),format=NV12,"
            f"width={width},height={height},framerate={fps}"
        )
        return (
            f"nvarguscamerasrc name=camera ! {caps} ! "
            f"nvvidconv flip-method={self.flip_method} ! {caps} ! "
        )

    def configure(self, element):
        element.set_property("sensor-id", self.sensor_id)
        element.set_property("sensor-mode", self.sensor_mode)
        # S56 mode programming takes about 3s; allow the first frame to arrive.
        element.set_property("event-wait", 10_000_000_000)
        element.set_property("acquire-wait", 15_000_000_000)


# Each adapter supplies a pipeline prefix and configures its named camera element.
SOURCES = {"v4l2": V4L2Source, "argus": ArgusSource}


def camera_source(spec):
    if spec.get("stereo", False):
        raise ValueError("gstreamer encoder requires a mono camera")
    source_type = spec.get("type")
    if source_type not in SOURCES:
        raise ValueError(f"Unsupported GStreamer camera type: {source_type!r}")
    return SOURCES[source_type](spec)
