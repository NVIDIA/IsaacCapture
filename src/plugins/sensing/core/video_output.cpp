// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "video_output.hpp"

#include "jetson_encoder.hpp"
#include "rgba_box_downscale.cuh"
#include "sensor_clock.hpp"

#include <log_bridge/logger.hpp>
#include <video_stage/direct_sender.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <variant>

namespace plugins
{
namespace sensing
{

namespace
{

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("Video output: ") + what + ": " + cudaGetErrorString(err));
}

// Congestion control never drives the encoder below this.
constexpr uint32_t kMinBitrateBps = 100'000;
/// Forced keyframes (the PC asking, or a dropped link recovering) are at least this far apart.
constexpr std::chrono::milliseconds kMinIdrInterval{ 1000 };
/// How often the link is judged, how far the bitrate falls back, and how long it must stay calm before it rises.
constexpr std::chrono::milliseconds kAdaptInterval{ 1000 };
constexpr uint32_t kFloorBitrateBps = 1'500'000;
constexpr uint32_t kCalmBeforeRaise = 5;
// A target is applied only when it differs from the current one by this fraction, and this long after
// the last change.

} // namespace

struct VideoOutput::Impl
{
    struct Slot
    {
        uint8_t* ptr = nullptr;
        size_t pitch = 0;
        uint64_t timestamp_ns = 0;
        uint64_t capture_monotonic_ns = 0;
        uint64_t capture_tsc = 0;
        uint64_t sequence = 0;
    };

    explicit Impl(const VideoOutputConfig& cfg) : config(cfg)
    {
        validate_geometry(config.source_width, config.source_height, config.scale_div);
        if (config.fps == 0)
            throw std::runtime_error("Video output: the publication rate must be at least 1 fps");
        out_width = config.source_width / config.scale_div;
        out_height = config.source_height / config.scale_div;
        if (config.input == PixelKind::DmaBufNv12 && config.scale_div != 1)
            throw std::runtime_error("Video output: DMA-BUF input cannot be reduced (scale_div must be 1)");
        period_ns = 1'000'000'000ull / config.fps;
        // Always present: with no host the sender idles, and a management client can give it one at runtime.
        {
            DirectSenderConfig direct_config;
            direct_config.host = config.direct_host;
            direct_config.port = config.direct_port;
            direct_config.width = out_width;
            direct_config.height = out_height;
            direct_config.fps = config.fps;
            direct = std::make_unique<DirectSender>(direct_config);
        }
        if (config.bitrate_bps < kMinBitrateBps)
        {
            throw std::runtime_error("Video output: the bitrate must be at least " + std::to_string(kMinBitrateBps) +
                                     " bps");
        }
        // Congestion control moves the target between the floor and the ceiling; the configured target
        // is never below the floor, so the range is never inverted.

        // A constructor that throws never runs the destructor, so free what was allocated here.
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice");

            // First, so a machine without a usable hardware encoder fails before anything is allocated.
            EncoderConfig encoder_config;
            encoder_config.width = out_width;
            encoder_config.height = out_height;
            encoder_config.bitrate_bps = config.bitrate_bps;
            current_bitrate = config.bitrate_bps;
            encoder_config.peak_bitrate_bps = config.peak_bitrate_bps;
            encoder_config.fps = config.fps;
            encoder_config.gop = config.gop;
            encoder_config.full_range = config.full_range;
            encoder_config.input = config.input;
            encoder = std::make_unique<JetsonEncoder>(encoder_config);

            for (auto& slot : slots)
            {
                if (config.input != PixelKind::CudaRgba)
                    break; // DMA-BUF frames are not copied
                check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&slot.ptr), &slot.pitch,
                                           static_cast<size_t>(config.source_width) * 4, config.source_height),
                           "cudaMallocPitch(snapshot)");
            }
            if (config.scale_div > 1)
            {
                check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&d_scaled), &scaled_pitch,
                                           static_cast<size_t>(out_width) * 4, out_height),
                           "cudaMallocPitch(reduced)");
            }
            check_cuda(cudaStreamCreateWithFlags(&snapshot_stream, cudaStreamNonBlocking), "cudaStreamCreate");
            check_cuda(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking), "cudaStreamCreate");

            worker = std::thread([this] { run(); });
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        if (worker.joinable())
            worker.join();
        release();
    }

    void release()
    {
        for (auto& slot : slots)
            cudaFree(slot.ptr);
        cudaFree(d_scaled);
        if (snapshot_stream)
            cudaStreamDestroy(snapshot_stream);
        if (work_stream)
            cudaStreamDestroy(work_stream);
    }

    void fail(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failed_flag.load())
        {
            failure_message = message;
            failed_flag.store(true);
        }
    }

    // Polling thread.
    void submit(CaptureFrame& frame)
    {
        const bool dma = config.input == PixelKind::DmaBufNv12;
        const auto* rgba = std::get_if<CudaRgbaPixels>(&frame.pixels);
        if (dma ? !std::holds_alternative<DmaBufNv12Pixels>(frame.pixels) : rgba == nullptr)
        {
            fail("Video output: the frame's pixel type does not match the encoder input");
            return;
        }
        ++offered;
        if (failed_flag.load())
            return;

        // Pace on the clock the frame is stamped with: the sensor's capture time when that is the published basis,
        // otherwise the time the camera backend handed the frame over. Either way it is not the poll time, so poll
        // jitter does not drop a frame that is due.
        uint64_t pace_ns = frame.monotonic_ns;
        if (config.timestamp_basis == TimestampBasis::SensorCaptureMonotonic)
        {
            try
            {
                frame.capture_monotonic_ns = capture_time_ns(frame);
            }
            catch (const std::exception& e)
            {
                fail(e.what());
                return;
            }
            pace_ns = frame.capture_monotonic_ns;
        }
        // A clock that steps back would otherwise hold every frame until it caught up: restart pacing instead.
        if (have_last && pace_ns < last_accepted_ns)
            have_last = false;
        // A tenth of a period of slack lets 60 fps capture feed a 30 fps track every second capture.
        if (have_last && pace_ns < last_accepted_ns + period_ns * 9 / 10)
        {
            ++paced_out;
            return;
        }

        if (dma)
        {
            const uint64_t accepted_ns = pace_ns;
            // A frame still waiting is replaced by this newer one. Its lease is released after the lock is dropped:
            // that is a camera API call, and must not run inside the section the worker waits on.
            std::optional<CaptureFrame> replaced_frame;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (pending_dma)
                    ++replaced;
                replaced_frame = std::move(pending_dma);
                pending_dma = std::move(frame);
            }
            replaced_frame.reset();
            ++snapshots;
            have_last = true;
            last_accepted_ns = accepted_ns;
            wake.notify_one();
            return;
        }

        try
        {
            std::lock_guard<std::mutex> lock(mutex);
            int target = -1;
            for (int i = 0; i < 2; ++i)
            {
                if (i != inflight && i != pending)
                {
                    target = i;
                    break;
                }
            }
            if (target < 0)
            {
                // The worker holds one slot and the other waits: the newer capture takes the waiting one.
                target = pending;
            }
            // A capture still waiting is overwritten by this newer one.
            if (pending >= 0)
                ++replaced;

            // Complete before returning: the camera may rewrite its slot at the second following poll.
            Slot& slot = slots[target];
            check_cuda(cudaMemcpy2DAsync(slot.ptr, slot.pitch, reinterpret_cast<const void*>(rgba->device_ptr),
                                         rgba->pitch_bytes, static_cast<size_t>(config.source_width) * 4,
                                         config.source_height, cudaMemcpyDeviceToDevice, snapshot_stream),
                       "snapshot copy");
            check_cuda(cudaStreamSynchronize(snapshot_stream), "snapshot sync");

            slot.timestamp_ns = frame.monotonic_ns;
            slot.capture_monotonic_ns = frame.capture_monotonic_ns;
            slot.capture_tsc = frame.sensor_time;
            slot.sequence = frame.sequence;
            pending = target;
        }
        catch (const std::exception& e)
        {
            fail(e.what());
            return;
        }
        ++snapshots;
        have_last = true;
        last_accepted_ns = pace_ns;
        wake.notify_one();
    }

    // Worker thread.
    void run()
    {
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice(worker)");
            for (;;)
            {
                int index = -1;
                std::optional<CaptureFrame> dma;
                bool flush_now = false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    // Frames in the encoder come out a little later: poll for them soon, and otherwise
                    // wake now and then to pick up the receiving side's requests.
                    const auto wait = in_encoder.empty() ? std::chrono::milliseconds(100) : std::chrono::milliseconds(2);
                    wake.wait_for(lock, wait, [&] { return stopping || pending >= 0 || pending_dma || flush_requested; });
                    if (stopping)
                        return;
                    index = pending;
                    pending = -1;
                    inflight = index;
                    dma = std::move(pending_dma);
                    pending_dma.reset();
                    flush_now = flush_requested;
                }
                if (dma)
                {
                    encode_dma(*dma);
                    dma.reset();
                }
                if (index >= 0)
                {
                    encode(slots[index]);
                    std::lock_guard<std::mutex> lock(mutex);
                    inflight = -1;
                }
                if (flush_now)
                {
                    for (auto& unit : encoder->drain())
                        publish_unit(unit);
                    std::lock_guard<std::mutex> lock(mutex);
                    flush_requested = false;
                    flushed.notify_all();
                }
                else
                {
                    for (auto unit = encoder->poll(); !unit.empty(); unit = encoder->poll())
                        publish_unit(unit);
                }
            }
        }
        catch (const std::exception& e)
        {
            fail(e.what());
            std::lock_guard<std::mutex> lock(mutex);
            flushed.notify_all();
        }
    }

    // A keyframe is several times the size of an ordinary frame. A link that is struggling makes the PC ask for
    // keyframes over and over, and each one adds to the load that caused the request, so forced keyframes are spaced.
    bool idr_allowed()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_forced_idr < kMinIdrInterval)
            return false;
        last_forced_idr = now;
        return true;
    }

    // Worker thread: follow the link. Frames the direct sender had to drop mean the network could not carry the
    // stream, so lower the bitrate; after a calm stretch raise it again, never above the configured target.
    void adapt_to_link()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_adapt < kAdaptInterval)
            return;
        last_adapt = now;
        const uint64_t dropped = direct->dropped();
        const bool congested = dropped != last_dropped;
        last_dropped = dropped;

        uint32_t target = current_bitrate;
        if (congested)
        {
            calm_intervals = 0;
            target = std::max<uint32_t>(kFloorBitrateBps, static_cast<uint32_t>(current_bitrate * 0.7));
        }
        else if (++calm_intervals >= kCalmBeforeRaise)
        {
            calm_intervals = 0;
            target = std::min<uint32_t>(config.bitrate_bps, static_cast<uint32_t>(current_bitrate * 1.15));
        }
        target = std::min(target, config.bitrate_bps);
        if (target == current_bitrate || !encoder->set_bitrate(target))
            return;
        isaaccapture::Logger::get("isaaccapture.plugins.sensing.VideoOutput")
            ->info("link {}: video bitrate {} -> {} kbit/s", congested ? "congested" : "recovering",
                   current_bitrate / 1000, target / 1000);
        current_bitrate = target;
    }

    // Worker thread.
    void encode(const Slot& slot)
    {
        const uint8_t* rgba = slot.ptr;
        size_t pitch = slot.pitch;
        if (config.scale_div > 1)
        {
            launch_rgba_box_downscale(rgba, static_cast<int>(pitch), d_scaled, static_cast<int>(scaled_pitch),
                                      static_cast<int>(out_width), static_cast<int>(out_height),
                                      static_cast<int>(config.scale_div), work_stream);
            check_cuda(cudaGetLastError(), "reduction kernel");
            // The encoder reads on the default stream, which a non-blocking stream does not order against.
            check_cuda(cudaStreamSynchronize(work_stream), "reduction sync");
            rgba = d_scaled;
            pitch = scaled_pitch;
        }

        if (direct && direct->take_keyframe_request())
        {
            ++keyframes_requested;
            keyframe_wanted = true;
        }
        adapt_to_link();
        if (keyframe_wanted && idr_allowed())
        {
            encoder->request_keyframe();
            keyframe_wanted = false;
        }
        const int64_t stamp_ns = static_cast<int64_t>(slot.timestamp_ns);
        if (!encoder->submit(reinterpret_cast<uintptr_t>(rgba), pitch, stamp_ns))
        {
            // No input buffer: dropped before encoding, so no reference is lost. A keyframe already
            // requested of the encoder applies to the next frame it takes.
            ++encoder_busy;
            return;
        }
        track_in_flight(static_cast<uint32_t>(slot.sequence), slot.capture_tsc,
                        application_stamp_us(slot.timestamp_ns, slot.capture_monotonic_ns));
    }

    // Worker thread: a leased DMA-BUF frame goes to the encoder as it is, and its lease stays with the encoder.
    void encode_dma(CaptureFrame& frame)
    {
        if (direct && direct->take_keyframe_request())
        {
            ++keyframes_requested;
            keyframe_wanted = true;
        }
        adapt_to_link();
        if (keyframe_wanted && idr_allowed())
        {
            encoder->request_keyframe();
            keyframe_wanted = false;
        }
        const int fd = std::get<DmaBufNv12Pixels>(frame.pixels).fd;
        if (!encoder->submit_dmabuf(fd, static_cast<int64_t>(frame.monotonic_ns), std::move(frame.lease)))
        {
            ++encoder_busy;
            return;
        }
        track_in_flight(static_cast<uint32_t>(frame.sequence), frame.sensor_time,
                        application_stamp_us(frame.monotonic_ns, frame.capture_monotonic_ns));
    }

    /// Polling thread: the frame's capture event on CLOCK_MONOTONIC. The sensor clock's anchor is built from the
    /// first frame's declared domain.
    uint64_t capture_time_ns(const CaptureFrame& frame)
    {
        if (frame.sensor_domain == SensorTimeDomain::None || frame.sensor_time == 0)
            throw std::runtime_error("Video output: the camera backend reports no sensor timestamp to publish");
        if (!clock_map)
        {
            SensorClockMap::Config map;
            map.domain = frame.sensor_domain;
            if (frame.sensor_domain == SensorTimeDomain::GenericCounterTicks)
                map.counter_hz = read_counter_hz();
            const bool counter = frame.sensor_domain == SensorTimeDomain::GenericCounterTicks;
            clock_map.emplace(map, counter ? SensorClockMap::PairReader(read_counter_pair) :
                                             SensorClockMap::PairReader(read_raw_pair));
        }
        struct timespec now
        {
        };
        clock_gettime(CLOCK_MONOTONIC, &now);
        const int64_t now_ns = static_cast<int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec;
        return static_cast<uint64_t>(
            clock_map->to_monotonic_ns(frame.sensor_time, static_cast<int64_t>(frame.monotonic_ns), now_ns));
    }

    /// The application timestamp published with a frame, in microseconds.
    uint64_t application_stamp_us(uint64_t ready_monotonic_ns, uint64_t capture_monotonic_ns) const
    {
        switch (config.timestamp_basis)
        {
        case TimestampBasis::FrameReadyMonotonic:
            return ready_monotonic_ns / 1000;
        case TimestampBasis::SensorCaptureMonotonic:
            return capture_monotonic_ns / 1000;
        }
        throw std::logic_error("unknown timestamp basis");
    }

    // Worker thread: remember the capture a submitted frame came from, for the unit that comes back.
    void track_in_flight(uint32_t sequence, uint64_t capture_tsc, uint64_t stamp_us)
    {
        in_encoder.push_back({ sequence, capture_tsc, stamp_us });
        // The encoder holds a bounded number of frames, so a longer queue means it dropped some inside it.
        // Trimming bounds how stale the oldest metadata can be; it does not resynchronise units with captures,
        // so frame ids after an encoder-internal drop can be shifted until a supported per-frame correlation exists.
        constexpr std::size_t kMaxInFlight = 12;
        while (in_encoder.size() > kMaxInFlight)
            in_encoder.pop_front();
    }

    // Worker thread: one encoded unit becomes one published frame, with the sequence and capture time
    // of the capture it came from.
    void publish_unit(const EncodedUnit& unit)
    {
        const int64_t stamp_us = unit.timestamp_ns / 1000;
        // The Jetson encoder does not return the stamp it was given: it numbers its units from its own frame
        // rate. It does return one unit per accepted frame in submission order, so units pair with captures
        // by order; the unit's own stamp is for diagnostics only.
        if (in_encoder.empty())
            throw std::runtime_error("Video output: the encoder returned a unit with no frame in flight (unit stamp_us=" +
                                     std::to_string(stamp_us) + ")");
        const FrameMeta meta = in_encoder.front();
        in_encoder.pop_front();

        EncodedVideoFrame frame;
        frame.data = unit.data.data();
        frame.size = unit.data.size();
        frame.keyframe = unit.keyframe;
        frame.timestamp_us = static_cast<uint64_t>(stamp_us);
        frame.frame_id = meta.sequence;
        frame.capture_tsc = meta.capture_tsc;
        frame.user_timestamp_us = meta.stamp_us;
        // The direct link keeps its own queue and keyframe bookkeeping and never blocks this thread.
        direct->submit(frame);
        ++published;
    }

    std::unique_ptr<DirectSender> direct; ///< optional second destination: the PC video stage
    VideoOutputConfig config;
    uint32_t out_width = 0;
    uint32_t out_height = 0;
    uint64_t period_ns = 0;

    struct FrameMeta
    {
        uint32_t sequence = 0;
        uint64_t capture_tsc = 0;
        /// The frame's application timestamp in microseconds, on the sender's monotonic clock (see TimestampBasis).
        uint64_t stamp_us = 0;
    };

    std::unique_ptr<JetsonEncoder> encoder;

    // Worker thread only.
    std::optional<SensorClockMap> clock_map; ///< Polling thread only.
    std::deque<FrameMeta> in_encoder; ///< Submitted to the encoder, no unit back yet.
    bool keyframe_wanted = false;
    std::chrono::steady_clock::time_point last_forced_idr{};
    std::chrono::steady_clock::time_point last_adapt = std::chrono::steady_clock::now();
    uint64_t last_dropped = 0;
    uint32_t calm_intervals = 0;
    uint32_t current_bitrate = 0;

    Slot slots[2];
    /// DMA-BUF input only: the newest accepted frame, waiting for the worker. Guarded by `mutex`.
    std::optional<CaptureFrame> pending_dma;
    uint8_t* d_scaled = nullptr;
    size_t scaled_pitch = 0;
    cudaStream_t snapshot_stream = nullptr;
    cudaStream_t work_stream = nullptr;

    // Guards pending, inflight, stopping and the snapshot slots' contents.
    std::mutex mutex;
    std::condition_variable wake;
    int pending = -1;
    int inflight = -1;
    bool stopping = false;
    bool flush_requested = false;
    std::condition_variable flushed;
    std::thread worker;

    // Polling thread only.
    bool have_last = false;
    uint64_t last_accepted_ns = 0;

    std::atomic<uint64_t> offered{ 0 };
    std::atomic<uint64_t> paced_out{ 0 };
    std::atomic<uint64_t> replaced{ 0 };
    std::atomic<uint64_t> snapshots{ 0 };
    std::atomic<uint64_t> published{ 0 };
    std::atomic<uint64_t> encoder_busy{ 0 };
    std::atomic<uint64_t> keyframes_requested{ 0 };

    std::mutex failure_mutex;
    std::string failure_message;
    std::atomic<bool> failed_flag{ false };
};

void VideoOutput::validate_geometry(uint32_t width, uint32_t height, uint32_t divisor)
{
    if (divisor == 0)
        throw std::runtime_error("Video output: the scale divisor must be at least 1");
    if (width == 0 || height == 0)
        throw std::runtime_error("Video output: the source size is empty");
    if (width % divisor != 0 || height % divisor != 0)
    {
        throw std::runtime_error("Video output: divisor " + std::to_string(divisor) + " does not divide " +
                                 std::to_string(width) + "x" + std::to_string(height) + " exactly");
    }
    if ((width / divisor) % 2 != 0 || (height / divisor) % 2 != 0)
    {
        throw std::runtime_error("Video output: " + std::to_string(width / divisor) + "x" +
                                 std::to_string(height / divisor) + " is not even; NV12 needs even dimensions");
    }
}

VideoOutput::VideoOutput(const VideoOutputConfig& config) : m_impl(std::make_unique<Impl>(config))
{
}

VideoOutput::~VideoOutput() = default;

void VideoOutput::submit(CaptureFrame& frame)
{
    m_impl->submit(frame);
}

VideoOutputStats VideoOutput::stats() const
{
    VideoOutputStats s;
    s.offered = m_impl->offered.load();
    s.paced_out = m_impl->paced_out.load();
    s.replaced = m_impl->replaced.load();
    s.snapshots = m_impl->snapshots.load();
    s.published = m_impl->published.load();
    s.encoder_busy = m_impl->encoder_busy.load();
    s.keyframes_requested = m_impl->keyframes_requested.load();
    return s;
}

bool VideoOutput::failed() const
{
    return m_impl->failed_flag.load();
}

std::string VideoOutput::failure() const
{
    std::lock_guard<std::mutex> lock(m_impl->failure_mutex);
    return m_impl->failure_message;
}

void VideoOutput::set_direct_target(const std::string& host, uint16_t port)
{
    m_impl->direct->set_target(host, port);
}

DirectLinkStatus VideoOutput::direct_status() const
{
    return { m_impl->direct->host(), m_impl->direct->port(), m_impl->direct->connected(), m_impl->direct->last_error() };
}

std::string VideoOutput::describe()
{
    std::string text = std::string(JetsonEncoder::backend_name()) + ", H.264 to the PC video stage";
    return text;
}

void VideoOutput::flush()
{
    std::unique_lock<std::mutex> lock(m_impl->mutex);
    m_impl->flush_requested = true;
    m_impl->wake.notify_one();
    m_impl->flushed.wait(lock, [&] { return !m_impl->flush_requested || m_impl->failed_flag.load(); });
}

} // namespace sensing
} // namespace plugins
