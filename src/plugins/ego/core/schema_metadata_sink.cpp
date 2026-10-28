// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diagnostics.hpp"
#include "metadata_sink.hpp"

#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/ego_limits.hpp>
#include <schema/serialized.hpp>

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace plugins::ego
{

namespace
{
constexpr size_t kMaxFlatbufferSize = core::EGO_MAX_FLATBUFFER_SIZE;
constexpr size_t kMaxQueuedEvents = 4096;
class SchemaMetadataSink final : public IMetadataSink
{
public:
    SchemaMetadataSink(const std::vector<StreamConfig>& streams,
                       const std::string& collection_prefix,
                       const std::string& plugin_root_id)
        // TeleopSession creates DeviceIO before launching plugins; avoid an unbounded HMD wait.
        : session_(std::make_shared<core::OpenXRSession>(
              "EgoCameraPlugin", core::SchemaPusher::get_required_extensions(), false)),
          plugin_root_id_(plugin_root_id)
    {
        if (!collection_prefix.empty())
        {
            for (const auto& stream : streams)
            {
                const std::string name = core::EnumNameOrbbecEgoCameraStream(stream.camera);
                const std::string collection_id = collection_prefix + "/" + name;
                pushers_.emplace(stream.camera, std::make_unique<core::SchemaPusher>(
                                                    session_->get_handles(),
                                                    core::SchemaPusherConfig{ .collection_id = collection_id,
                                                                              .max_flatbuffer_size = kMaxFlatbufferSize,
                                                                              .tensor_identifier = "frame_metadata",
                                                                              .localized_name = "Ego frame metadata",
                                                                              .app_name = "EgoCameraPlugin" }));
                detail::log_info() << "  Metadata: " << collection_id << std::endl;
            }
            imu_pushers_.emplace(core::OrbbecEgoImuSensor_Accel,
                                 make_pusher(collection_prefix + "/Accel", "imu_batch", "Ego accelerometer"));
            imu_pushers_.emplace(
                core::OrbbecEgoImuSensor_Gyro, make_pusher(collection_prefix + "/Gyro", "imu_batch", "Ego gyroscope"));
            audio_pusher_ = make_pusher(collection_prefix + "/Audio", "audio_chunk", "Ego audio index");
            calibration_pusher_ = make_pusher(collection_prefix + "/Calibration", "calibration", "Ego calibration");
            device_state_pusher_ = make_pusher(collection_prefix + "/DeviceState", "device_state", "Ego device state");
        }
        latest_status_.observed_at_ns = core::os_monotonic_now_ns();
        publisher_ = std::thread([this] { publish_loop(); });
    }

    ~SchemaMetadataSink() override
    {
        try
        {
            close();
        }
        catch (const std::exception& error)
        {
            detail::log_error() << "Ego SchemaPusher shutdown failed: " << error.what() << std::endl;
        }
        catch (...)
        {
            detail::log_error() << "Ego SchemaPusher shutdown failed: unknown error" << std::endl;
        }
    }

    void on_frame_metadata(const CapturedFrame& frame) override
    {
        const auto it = pushers_.find(frame.metadata.stream);
        if (it == pushers_.end())
            return;

        enqueue<core::OrbbecEgoFrameMetadata>(*it->second, frame.metadata, frame.sample_time_local_common_clock_ns,
                                              frame.sample_time_raw_device_clock_ns);
    }

    void on_imu_batch(const core::OrbbecEgoImuBatchT& batch, int64_t local_ns, int64_t device_ns) override
    {
        if (const auto found = imu_pushers_.find(batch.sensor); found != imu_pushers_.end())
            enqueue(*found->second, batch, local_ns, device_ns);
    }

    void on_audio_chunk(const core::OrbbecEgoAudioChunkT& chunk, int64_t local_ns, int64_t device_ns) override
    {
        if (audio_pusher_)
            enqueue(*audio_pusher_, chunk, local_ns, device_ns);
    }

    void on_calibration(const core::OrbbecEgoCalibrationT& calibration, int64_t local_ns, int64_t device_ns) override
    {
        if (calibration_pusher_)
            enqueue(*calibration_pusher_, calibration, local_ns, device_ns);
    }

    void on_device_state(const core::OrbbecEgoDeviceStateT& state, int64_t local_ns, int64_t device_ns) override
    {
        if (device_state_pusher_)
            enqueue(*device_state_pusher_, state, local_ns, device_ns);
    }

    void on_capture_status(const CaptureStatus& status) override
    {
        if (plugin_root_id_.empty())
            return;
        std::lock_guard<std::mutex> lock(publish_mutex_);
        if (stopping_ || status.observed_at_ns < latest_status_.observed_at_ns)
            return;
        status_changed_ = status_changed_ || status.availability != latest_status_.availability ||
                          status.error != latest_status_.error;
        latest_status_ = status;
        if (status_changed_)
            publish_wake_.notify_one();
    }

    void close() override
    {
        {
            std::lock_guard<std::mutex> lock(publish_mutex_);
            stopping_ = true;
        }
        publish_wake_.notify_all();
        if (publisher_.joinable())
            publisher_.join();
        pushers_.clear();
        imu_pushers_.clear();
        audio_pusher_.reset();
        calibration_pusher_.reset();
        device_state_pusher_.reset();
        status_publisher_.reset();
        session_.reset();
        const auto failure = error();
        if (!failure.empty())
            throw std::runtime_error("Ego SchemaPusher failed: " + failure);
    }

    std::string error() const override
    {
        std::lock_guard<std::mutex> lock(publish_mutex_);
        return publish_error_;
    }

private:
    std::unique_ptr<core::SchemaPusher> make_pusher(const std::string& collection,
                                                    const std::string& tensor,
                                                    const std::string& name)
    {
        detail::log_info() << "  Metadata: " << collection << std::endl;
        return std::make_unique<core::SchemaPusher>(
            session_->get_handles(), core::SchemaPusherConfig{ .collection_id = collection,
                                                               .max_flatbuffer_size = kMaxFlatbufferSize,
                                                               .tensor_identifier = tensor,
                                                               .localized_name = name,
                                                               .app_name = "EgoCameraPlugin" });
    }

    template <typename TableT>
    void enqueue(core::SchemaPusher& pusher,
                 const typename TableT::NativeTableType& value,
                 int64_t local_ns,
                 int64_t device_ns)
    {
        const auto payload = core::pack<TableT>(value);
        enqueue_task(
            [&pusher, payload, local_ns, device_ns]()
            {
                const auto bytes = payload.buffer();
                pusher.push_buffer(bytes.data(), bytes.size(), local_ns, device_ns);
            });
    }

    void enqueue(core::SchemaPusher& pusher, const core::OrbbecEgoImuBatchT& value, int64_t local_ns, int64_t device_ns)
    {
        enqueue<core::OrbbecEgoImuBatch>(pusher, value, local_ns, device_ns);
    }
    void enqueue(core::SchemaPusher& pusher, const core::OrbbecEgoAudioChunkT& value, int64_t local_ns, int64_t device_ns)
    {
        enqueue<core::OrbbecEgoAudioChunk>(pusher, value, local_ns, device_ns);
    }
    void enqueue(core::SchemaPusher& pusher, const core::OrbbecEgoCalibrationT& value, int64_t local_ns, int64_t device_ns)
    {
        enqueue<core::OrbbecEgoCalibration>(pusher, value, local_ns, device_ns);
    }
    void enqueue(core::SchemaPusher& pusher, const core::OrbbecEgoDeviceStateT& value, int64_t local_ns, int64_t device_ns)
    {
        enqueue<core::OrbbecEgoDeviceState>(pusher, value, local_ns, device_ns);
    }

    void enqueue_task(std::function<void()> task)
    {
        std::lock_guard<std::mutex> lock(publish_mutex_);
        if (stopping_)
            throw std::runtime_error("Ego SchemaPusher is already closed");
        if (!publish_error_.empty())
            throw std::runtime_error("Ego SchemaPusher failed: " + publish_error_);
        if (tasks_.size() >= kMaxQueuedEvents)
            throw std::runtime_error("Ego SchemaPusher queue is full; capture stopped to avoid silent loss");
        if (tasks_.size() + 1 >= kMaxQueuedEvents * 85 / 100 && !warning_emitted_)
        {
            warning_emitted_ = true;
            detail::log_warning()
                << "Warning: Ego SchemaPusher queue reached 85%; capture will stop if it fills." << std::endl;
        }
        tasks_.push_back(std::move(task));
        publish_wake_.notify_one();
    }

    void publish_status(const CaptureStatus& status)
    {
        if (plugin_root_id_.empty())
            return;
        try
        {
            if (!status_publisher_)
                status_publisher_ = std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(
                    session_->get_handles(), plugin_root_id_);
            plugin_utils::PluginDeviceStatusEntry entry{ .path = "/camera/ego",
                                                         .reason = core::PluginDeviceReason_NO_CURRENT_DATA };
            entry.error = status.error;
            switch (status.availability)
            {
            case CaptureAvailability::Unknown:
                entry.state = core::PluginDeviceState_UNKNOWN;
                break;
            case CaptureAvailability::Absent:
                entry.state = core::PluginDeviceState_DISCONNECTED;
                entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
                break;
            case CaptureAvailability::Starting:
            case CaptureAvailability::Recovering:
                entry.state = core::PluginDeviceState_DEGRADED;
                break;
            case CaptureAvailability::Streaming:
                entry.state = core::PluginDeviceState_CONNECTED;
                entry.reason = core::PluginDeviceReason_NONE;
                break;
            case CaptureAvailability::Failed:
                entry.state = core::PluginDeviceState_FAILED;
                entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
                break;
            }
            // A blocked capture owner must age out; publishing is not a new observation.
            status_publisher_->publish_if_changed({ entry }, status.observed_at_ns);
            status_warning_emitted_ = false;
        }
        catch (const std::exception& error)
        {
            if (!status_warning_emitted_)
                detail::log_warning() << "Ego device status unavailable; capture continues: " << error.what();
            status_warning_emitted_ = true;
        }
    }

    void publish_loop()
    {
        auto next_status = std::chrono::steady_clock::now();
        while (true)
        {
            std::function<void()> task;
            CaptureStatus status;
            bool publish_status_now = false;
            {
                std::unique_lock<std::mutex> lock(publish_mutex_);
                const auto ready = [this] { return stopping_ || status_changed_ || !tasks_.empty(); };
                if (plugin_root_id_.empty())
                    publish_wake_.wait(lock, ready);
                else
                    publish_wake_.wait_until(lock, next_status, ready);
                if (stopping_ && tasks_.empty())
                    return;
                const auto now = std::chrono::steady_clock::now();
                if (!plugin_root_id_.empty() && (status_changed_ || now >= next_status))
                {
                    status = latest_status_;
                    publish_status_now = true;
                    status_changed_ = false;
                    // Schedule against wall progress, even when the fact timestamp has stopped.
                    next_status = now + std::chrono::milliseconds(250);
                }
                if (!tasks_.empty())
                {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
            }
            if (publish_status_now)
                publish_status(status);
            if (!task)
                continue;
            try
            {
                task();
            }
            catch (const std::exception& error)
            {
                std::lock_guard<std::mutex> lock(publish_mutex_);
                publish_error_ = error.what();
                tasks_.clear();
                detail::log_error() << "Ego SchemaPusher failure: " << publish_error_ << std::endl;
            }
        }
    }

    std::shared_ptr<core::OpenXRSession> session_;
    std::map<core::OrbbecEgoCameraStream, std::unique_ptr<core::SchemaPusher>> pushers_;
    std::map<core::OrbbecEgoImuSensor, std::unique_ptr<core::SchemaPusher>> imu_pushers_;
    std::unique_ptr<core::SchemaPusher> audio_pusher_;
    std::unique_ptr<core::SchemaPusher> calibration_pusher_;
    std::unique_ptr<core::SchemaPusher> device_state_pusher_;
    std::string plugin_root_id_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_publisher_;
    CaptureStatus latest_status_;
    bool status_changed_ = true;
    bool status_warning_emitted_ = false;
    mutable std::mutex publish_mutex_;
    std::condition_variable publish_wake_;
    std::deque<std::function<void()>> tasks_;
    std::thread publisher_;
    bool stopping_ = false;
    bool warning_emitted_ = false;
    std::string publish_error_;
};

}

std::unique_ptr<IMetadataSink> make_schema_metadata_sink(const std::vector<StreamConfig>& streams,
                                                         const std::string& prefix,
                                                         const std::string& plugin_root_id)
{
    return std::make_unique<SchemaMetadataSink>(streams, prefix, plugin_root_id);
}

} // namespace plugins::ego
