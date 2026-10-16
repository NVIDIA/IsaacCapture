// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SDK-only stereo capture for camera_viz. One Capture owns the device pipeline
// and copies paired host payloads; sources.ego owns the producer lifecycle
// and GPU publication. Sensor recording and MCAP output remain in the EGO plugin.

#include <libobsensor/ObSensor.hpp>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace
{

OBFormat format_from_name(const std::string& name)
{
    if (name == "mjpg" || name == "mjpeg")
        return OB_FORMAT_MJPG;
    if (name == "h264")
        return OB_FORMAT_H264;
    if (name == "h265" || name == "hevc")
        return OB_FORMAT_H265;
    throw std::invalid_argument("format must be mjpg, h264, or h265");
}

int64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The source producer serializes reads and close; never close during a live read.
class Capture
{
public:
    Capture(std::string uid, uint32_t width, uint32_t height, uint32_t fps, std::string format)
        : width_(width), height_(height), format_(format_from_name(format))
    {
#if defined(__linux__) || defined(__ANDROID__)
        context_->setUvcBackendType(OB_UVC_BACKEND_TYPE_LIBUVC);
#endif
        const auto devices = context_->queryDeviceList();
        for (uint32_t index = 0; index < devices->getCount(); ++index)
        {
            const auto candidate = devices->getDevice(index);
            if (uid.empty() || candidate->getDeviceInfo()->getUid() == uid)
            {
                device_ = candidate;
                selected_uid_ = candidate->getDeviceInfo()->getUid();
                break;
            }
        }
        if (!device_)
            throw std::runtime_error("EGO not found");
        pipeline_ = std::make_unique<ob::Pipeline>(device_);
        auto config = std::make_shared<ob::Config>();
        config->enableStream(device_->getSensor(OB_SENSOR_COLOR_LEFT)
                                 ->getStreamProfileList()
                                 ->getVideoStreamProfile(width, height, format_, fps));
        config->enableStream(device_->getSensor(OB_SENSOR_COLOR_RIGHT)
                                 ->getStreamProfileList()
                                 ->getVideoStreamProfile(width, height, format_, fps));
        pipeline_->start(config);
        if (format_ == OB_FORMAT_MJPG)
        {
            left_converter_ = std::make_unique<ob::FormatConvertFilter>();
            right_converter_ = std::make_unique<ob::FormatConvertFilter>();
            left_converter_->setFormatConvertType(FORMAT_MJPG_TO_RGB);
            right_converter_->setFormatConvertType(FORMAT_MJPG_TO_RGB);
        }
    }

    ~Capture()
    {
        try
        {
            close();
        }
        catch (const ob::Error&)
        {
        }
    }

    void close()
    {
        if (pipeline_)
        {
            // A confirmed removed instance has deactivated SDK components; release it through SDK destruction.
            // Connected-device stop errors still propagate through the existing source lifecycle.
            if (!device_disconnected_)
                pipeline_->stop();
            pipeline_.reset();
        }
        left_converter_.reset();
        right_converter_.reset();
        device_.reset();
        context_.reset();
    }

    py::object next_pair(uint32_t timeout_ms)
    {
        if (!pipeline_)
            throw std::runtime_error("EGO capture is closed");
        if (device_disconnected_)
            throw std::runtime_error("EGO device " + selected_uid_ + " disconnected");
        std::shared_ptr<ob::FrameSet> frame_set;
        {
            py::gil_scoped_release release;
            frame_set = pipeline_->waitForFrameset(timeout_ms);
        }
        if (!frame_set)
            return no_pair();
        const auto left_raw = frame_set->getFrame(OB_FRAME_COLOR_LEFT);
        const auto right_raw = frame_set->getFrame(OB_FRAME_COLOR_RIGHT);
        if (!left_raw || !right_raw)
            return no_pair();
        const auto left = left_raw->as<ob::VideoFrame>();
        const auto right = right_raw->as<ob::VideoFrame>();
        const auto left_ts = static_cast<int64_t>(left->getTimeStampUs()) * 1000;
        const auto right_ts = static_cast<int64_t>(right->getTimeStampUs()) * 1000;
        // Match the eyes on raw device time before conversion or payload copies.
        if (std::llabs(left_ts - right_ts) > 2'000'000)
            return no_pair();
        const auto left_bytes = bytes(left);
        const auto right_bytes = bytes(right);
        if (left_bytes.is_none() || right_bytes.is_none())
            return no_pair();
        py::dict result;
        result["left"] = left_bytes;
        result["right"] = right_bytes;
        // Preview time is host monotonic time after extraction, not exposure time.
        result["timestamp_ns"] = monotonic_ns();
        result["device_timestamp_ns"] = (left_ts + right_ts) / 2;
        result["sequence_left"] = left->getIndex();
        result["sequence_right"] = right->getIndex();
        return std::move(result);
    }

private:
    py::object no_pair()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_presence_check_)
        {
            next_presence_check_ = now + std::chrono::seconds(1);
            bool present = false;
            {
                py::gil_scoped_release release;
                const auto devices = context_->queryDeviceList();
                for (uint32_t index = 0; index < devices->getCount(); ++index)
                    if (selected_uid_ == devices->getUid(index))
                        present = true;
            }
            // None also represents transient timeout or incomplete pairs; only confirmed UID loss means disconnect.
            if (!present)
            {
                device_disconnected_ = true;
                throw std::runtime_error("EGO device " + selected_uid_ + " disconnected");
            }
        }
        return py::none();
    }

    py::object bytes(const std::shared_ptr<ob::VideoFrame>& frame)
    {
        std::shared_ptr<ob::VideoFrame> output = frame;
        if (format_ == OB_FORMAT_MJPG)
        {
            auto& converter = frame->getType() == OB_FRAME_COLOR_LEFT ? left_converter_ : right_converter_;
            std::shared_ptr<ob::Frame> converted;
            {
                py::gil_scoped_release release;
                converted = converter->process(frame);
            }
            // The SDK rejects undecodable JPEGs with a null frame; drop the pair.
            if (!converted)
                return py::none();
            output = converted->as<ob::VideoFrame>();
            const auto rgb_bytes = static_cast<uint64_t>(width_) * height_ * 3;
            if (output->getFormat() != OB_FORMAT_RGB || output->getWidth() != width_ ||
                output->getHeight() != height_ || output->getDataSize() != rgb_bytes || !output->getData())
                throw std::runtime_error("EGO MJPEG conversion did not return the requested RGB frame");
        }
        return py::bytes(reinterpret_cast<const char*>(output->getData()), output->getDataSize());
    }

    std::unique_ptr<ob::Context> context_ = std::make_unique<ob::Context>();
    std::shared_ptr<ob::Device> device_;
    std::unique_ptr<ob::Pipeline> pipeline_;
    std::unique_ptr<ob::FormatConvertFilter> left_converter_;
    std::unique_ptr<ob::FormatConvertFilter> right_converter_;
    std::string selected_uid_;
    std::chrono::steady_clock::time_point next_presence_check_{};
    bool device_disconnected_ = false;
    uint32_t width_;
    uint32_t height_;
    OBFormat format_;
};

} // namespace

PYBIND11_MODULE(_ego_capture, module)
{
    module.doc() = "Native EGO stereo capture for camera_viz";
    py::class_<Capture>(module, "Capture")
        .def(py::init<std::string, uint32_t, uint32_t, uint32_t, std::string>(), py::arg("device_uid") = "",
             py::arg("width") = 1600, py::arg("height") = 1300, py::arg("fps") = 30, py::arg("format") = "h264",
             py::call_guard<py::gil_scoped_release>())
        // SDK shutdown must not prevent the consumer thread from joining.
        .def("close", &Capture::close, py::call_guard<py::gil_scoped_release>())
        .def("next_pair", &Capture::next_pair, py::arg("timeout_ms") = 100);
}
