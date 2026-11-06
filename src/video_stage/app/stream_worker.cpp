// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stream_worker.hpp"

#include "control_channel.hpp"
#include "tcp_receiver.hpp"

#include <video_stage/avs1.hpp>
#include <video_stage/h264_sps.hpp>

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

namespace video_stage
{

namespace
{

// Writes the processed access units as one Annex-B stream, so a run can be checked with ffprobe or a player.
class FileSink final : public PacketSink
{
public:
    explicit FileSink(const std::string& path) : m_out(path, std::ios::binary)
    {
        if (!m_out)
            throw std::runtime_error("cannot open " + path);
    }
    void write(const EncodedPacket& packet, const FrameMeta&, const Geometry&) override
    {
        m_out.write(reinterpret_cast<const char*>(packet.data.data()), static_cast<std::streamsize>(packet.data.size()));
        if (!m_out)
            throw std::runtime_error("writing the output file failed");
    }

private:
    std::ofstream m_out;
};

// Records waiting for the processing thread. Bounded by count: past that, the stage is not keeping up.
class RecordQueue
{
public:
    struct Item
    {
        std::vector<uint8_t> bytes;
        int64_t arrived_ns = 0;
    };

    explicit RecordQueue(size_t capacity) : m_capacity(capacity)
    {
    }

    /// False when full.
    bool push(Item&& item)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_items.size() >= m_capacity)
                return false;
            m_items.push_back(std::move(item));
        }
        m_wake.notify_one();
        return true;
    }

    /// Discards everything waiting; returns how many records.
    size_t clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const size_t n = m_items.size();
        m_items.clear();
        return n;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_wake.notify_all();
    }

    /// Blocks for the next record; false once closed and empty.
    bool pop(Item& out)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_wake.wait(lock, [this] { return m_closed || !m_items.empty(); });
        if (m_items.empty())
            return false;
        out = std::move(m_items.front());
        m_items.pop_front();
        return true;
    }

private:
    size_t m_capacity;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::deque<Item> m_items;
    bool m_closed = false;
};

// What a stream without a GPU pipeline reports for the pipeline's own stages: nothing.
const PipelineStats& no_pipeline_stats()
{
    static const PipelineStats empty;
    return empty;
}

void print_stage(std::ostream& out, const char* name, const DurationStats& s)
{
    char line[160];
    std::snprintf(line, sizeof(line), "  %-22s n=%zu  p50 %.2f ms  p95 %.2f ms  p99 %.2f ms\n", name, s.count(),
                  s.percentile(50) / 1e6, s.percentile(95) / 1e6, s.percentile(99) / 1e6);
    out << line;
}

} // namespace

struct StreamWorker::Impl
{
    CudaContext* context;
    StreamSpec spec;
    StreamSettings settings;
    SharedLiveKit* livekit;

    std::unique_ptr<SineWarp> operation; ///< null for Operation::None
    std::unique_ptr<TcpReceiver> receiver;
    std::atomic<ControlChannel*> active_control{ nullptr };
    std::unique_ptr<PacketSink> sink;
    LiveKitOutput* livekit_output = nullptr;
    std::unique_ptr<Pipeline> pipeline;

    std::thread thread;
    std::atomic<bool> stop{ false };
    std::atomic<bool> ok{ true };
    std::atomic<uint64_t> overflows{ 0 };
    std::atomic<uint64_t> discarded{ 0 };
    std::atomic<uint64_t> sessions{ 0 };
    uint64_t epoch_base = 0; ///< advances for every session so a control channel's epoch is never reused
    std::atomic<uint64_t> clock_exchanges{ 0 };
    std::atomic<uint64_t> clock_rejected{ 0 };
    std::atomic<uint64_t> clock_discontinuities{ 0 };
    DurationStats backlog_bytes;
    // Pass-through state (Operation::None): the picture size from the last sequence header, and a count.
    Geometry passthrough_geometry;
    std::atomic<uint64_t> passthrough_records{ 0 };

    Impl(CudaContext* ctx, const StreamSpec& s, const StreamSettings& st, SharedLiveKit* lk)
        : context(ctx), spec(s), settings(st), livekit(lk)
    {
        if (settings.operation == Operation::SineWarp)
        {
            if (context == nullptr)
                throw std::runtime_error("the sine warp needs a GPU context");
            operation = std::make_unique<SineWarp>(settings.warp);
        }
        if (livekit != nullptr)
        {
            LiveKitOutputConfig out = settings.livekit;
            out.track_name = spec.track;
            auto output = std::make_unique<LiveKitOutput>(
                *livekit, out,
                [this]() -> std::optional<UpstreamMapping>
                {
                    ControlChannel* c = active_control.load();
                    return c ? c->mapping() : std::nullopt;
                },
                [this]
                {
                    // A subscriber needs a keyframe: ask this stage's encoder, or, with nothing in between, the Orin's.
                    if (pipeline)
                        pipeline->request_idr();
                    else
                        receiver->request_keyframe();
                });
            livekit_output = output.get();
            sink = std::move(output);
        }
        else
        {
            sink = std::make_unique<FileSink>(spec.file);
        }
        if (operation)
            pipeline = std::make_unique<Pipeline>(*context, *operation, *sink, settings.pipeline);
        receiver = std::make_unique<TcpReceiver>(spec.port);
    }

    // Operation::None: the access unit goes to the sink untouched. The picture size comes from the stream's own
    // sequence header; nothing is published before the first one.
    void forward(const avs1::Record& record, int64_t arrived_ns)
    {
        const bool keyframe = avs1::contains_idr(record.payload, record.payload_size);
        if (keyframe)
        {
            if (const auto sps = h264::parse_sps(record.payload, record.payload_size))
                passthrough_geometry = Geometry{ sps->width, sps->height };
        }
        if (passthrough_geometry.width == 0)
            return;
        FrameMeta meta;
        meta.frame_number = record.sample.frame_number;
        meta.source_timestamp_ns = record.sample.frame_timestamp_ns;
        meta.received_ns = arrived_ns;
        const uint64_t index = passthrough_records.fetch_add(1);
        meta.key = index;
        EncodedPacket packet;
        packet.data.assign(record.payload, record.payload + record.payload_size);
        packet.keyframe = keyframe;
        packet.key = index;
        sink->write(packet, meta, passthrough_geometry);
    }

    // Runs one Orin session to its end. Intake and processing are separate threads, so reading from the socket never
    // waits for the GPU stages and the time a record spends waiting is measured. Compressed records are not dropped
    // one by one: if the queue overflows, everything waiting is discarded and the stream resumes at the next keyframe,
    // which is asked of the sender. A link or decoder error ends the session (reported), after the queued records are
    // processed.
    bool run_session()
    {
        constexpr size_t kQueueCapacity = 4;
        RecordQueue queue(kQueueCapacity);
        std::atomic<bool> session_ok{ true };
        std::thread intake(
            [&]
            {
                try
                {
                    bool catching_up = false;
                    std::vector<uint8_t> bytes;
                    while (receiver->read_record(bytes))
                    {
                        const int64_t arrived_ns = monotonic_ns();
                        backlog_bytes.add(static_cast<int64_t>(receiver->pending_bytes()));
                        std::string error;
                        const auto record = avs1::parse(bytes.data(), bytes.size(), error);
                        if (!record)
                            throw std::runtime_error("bad AVS1 record: " + error);
                        if (catching_up)
                        {
                            if (!avs1::contains_idr(record->payload, record->payload_size))
                            {
                                ++discarded;
                                continue;
                            }
                            catching_up = false;
                        }
                        if (!queue.push({ std::move(bytes), arrived_ns }))
                        {
                            discarded += queue.clear() + 1;
                            ++overflows;
                            catching_up = true;
                            receiver->request_keyframe();
                        }
                        bytes = {};
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "stream " << spec.port << " intake: " << e.what() << std::endl;
                    session_ok = false;
                }
                queue.close();
            });

        try
        {
            RecordQueue::Item item;
            while (queue.pop(item))
            {
                std::string error;
                const auto record = avs1::parse(item.bytes.data(), item.bytes.size(), error);
                if (!record)
                    throw std::runtime_error("bad AVS1 record: " + error);
                if (pipeline)
                    pipeline->process(*record, item.arrived_ns);
                else
                    forward(*record, item.arrived_ns);
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "stream " << spec.port << " processing: " << e.what() << std::endl;
            session_ok = false;
            queue.close();
            receiver->shutdown_session();
        }
        intake.join();
        return session_ok.load();
    }

    void run()
    {
        try
        {
            do
            {
                std::cout << "stream " << spec.port << ": listening" << std::endl;
                receiver->accept_session();
                if (stop)
                    break;
                std::cout << "stream " << spec.port << ": Orin session connected" << std::endl;
                ++sessions;
                {
                    // A new session starts clean: nothing of the previous sender's publication or numbering carries
                    // over.
                    if (livekit_output)
                        livekit_output->end_session();
                    if (pipeline)
                        pipeline->reset_session();
                    epoch_base += 1'000'000;
                    ControlChannel control(*receiver, epoch_base + 1);
                    active_control = &control;
                    if (!run_session())
                        ok = false;
                    active_control = nullptr;
                    if (livekit_output)
                        livekit_output->end_session();
                    clock_exchanges += control.exchanges();
                    clock_rejected += control.rejected();
                    clock_discontinuities += control.discontinuities();
                }
                std::cout << "stream " << spec.port << ": Orin session ended" << std::endl;
            } while (!settings.once && !stop);
        }
        catch (const std::exception& e)
        {
            if (!stop)
            {
                std::cerr << "stream " << spec.port << ": " << e.what() << std::endl;
                ok = false;
            }
        }
    }
};

StreamWorker::StreamWorker(CudaContext* context,
                           const StreamSpec& spec,
                           const StreamSettings& settings,
                           SharedLiveKit* livekit)
    : m_impl(std::make_unique<Impl>(context, spec, settings, livekit))
{
}

StreamWorker::~StreamWorker()
{
    request_stop();
    join();
}

void StreamWorker::start()
{
    m_impl->thread = std::thread([this] { m_impl->run(); });
}

void StreamWorker::request_stop()
{
    m_impl->stop = true;
    m_impl->receiver->stop_listening();
    m_impl->receiver->shutdown_session();
}

bool StreamWorker::join()
{
    if (m_impl->thread.joinable())
        m_impl->thread.join();
    return m_impl->ok.load();
}

void StreamWorker::print_live(std::ostream& out) const
{
    const Impl& d = *m_impl;
    const PipelineStats& s = d.pipeline ? d.pipeline->stats() : no_pipeline_stats();
    char line[512];
    int n = std::snprintf(
        line, sizeof(line),
        "[%u] queue age p50 %.1f p99 %.1f ms | backlog p99 %lld B | decoder lag p50 %lld p99 %lld frames | "
        "submit->picture p50 %.1f p99 %.1f ms | warp+encode p50 %.1f p99 %.1f ms",
        static_cast<unsigned>(d.spec.port), s.queue_age_ns.percentile(50) / 1e6, s.queue_age_ns.percentile(99) / 1e6,
        static_cast<long long>(d.backlog_bytes.percentile(99)), static_cast<long long>(s.decode_lag.percentile(50)),
        static_cast<long long>(s.decode_lag.percentile(99)), s.decode_ns.percentile(50) / 1e6,
        s.decode_ns.percentile(99) / 1e6, s.process_encode_ns.percentile(50) / 1e6,
        s.process_encode_ns.percentile(99) / 1e6);
    if (d.livekit_output && n > 0 && n < static_cast<int>(sizeof(line)))
    {
        std::snprintf(
            line + n, sizeof(line) - static_cast<size_t>(n),
            " | intake->write p50 %.1f p99 %.1f ms | submit p50 %.1f p99 %.1f ms | Orin->PC p50 %.1f p99 %.1f ms | Orin->publish p50 %.1f p99 %.1f ms | mapping +-%.1f ms | published %llu",
            d.livekit_output->intake_to_write_ns().percentile(50) / 1e6,
            d.livekit_output->intake_to_write_ns().percentile(99) / 1e6,
            d.livekit_output->submit_ns().percentile(50) / 1e6, d.livekit_output->submit_ns().percentile(99) / 1e6,
            d.livekit_output->uplink_ns().percentile(50) / 1e6, d.livekit_output->uplink_ns().percentile(99) / 1e6,
            d.livekit_output->source_to_publish_ns().percentile(50) / 1e6,
            d.livekit_output->source_to_publish_ns().percentile(99) / 1e6,
            d.livekit_output->upstream_uncertainty_ns() / 1e6,
            static_cast<unsigned long long>(d.livekit_output->published()));
    }
    out << line << "\n";
}

void StreamWorker::print_summary(std::ostream& out) const
{
    const Impl& d = *m_impl;
    const PipelineStats& s = d.pipeline ? d.pipeline->stats() : no_pipeline_stats();
    char line[256];
    std::snprintf(
        line, sizeof(line),
        "stream %u: sessions %llu, records %llu, pictures %llu, dropped (no surface) %llu, gaps %llu, metadata lost %llu\n"
        "  overflows %llu, records discarded %llu, clock exchanges %llu (rejected %llu, discontinuities %llu)\n",
        static_cast<unsigned>(d.spec.port), static_cast<unsigned long long>(d.sessions.load()),
        static_cast<unsigned long long>(d.pipeline ? s.records : d.passthrough_records.load()),
        static_cast<unsigned long long>(d.pipeline ? s.pictures : d.passthrough_records.load()),
        static_cast<unsigned long long>(s.dropped_no_surface), static_cast<unsigned long long>(s.gaps),
        static_cast<unsigned long long>(s.metadata_lost), static_cast<unsigned long long>(d.overflows.load()),
        static_cast<unsigned long long>(d.discarded.load()), static_cast<unsigned long long>(d.clock_exchanges.load()),
        static_cast<unsigned long long>(d.clock_rejected.load()),
        static_cast<unsigned long long>(d.clock_discontinuities.load()));
    out << line;
    if (d.livekit_output)
    {
        std::snprintf(line, sizeof(line),
                      "  published %llu, unmapped %llu, over budget %llu, refused %llu, republished %llu\n",
                      static_cast<unsigned long long>(d.livekit_output->published()),
                      static_cast<unsigned long long>(d.livekit_output->unmapped()),
                      static_cast<unsigned long long>(d.livekit_output->over_budget()),
                      static_cast<unsigned long long>(d.livekit_output->refused()),
                      static_cast<unsigned long long>(d.livekit_output->republished()));
        out << line;
    }
    std::snprintf(line, sizeof(line), "  socket backlog         p50 %lld B  p99 %lld B\n",
                  static_cast<long long>(d.backlog_bytes.percentile(50)),
                  static_cast<long long>(d.backlog_bytes.percentile(99)));
    out << line;
    if (d.livekit_output)
    {
        print_stage(out, "Orin -> PC arrival", d.livekit_output->uplink_ns());
        print_stage(out, "Orin -> publish", d.livekit_output->source_to_publish_ns());
    }
    print_stage(out, "queue age", s.queue_age_ns);
    {
        char lag[160];
        std::snprintf(lag, sizeof(lag), "  decoder lag (frames)   p50 %lld  p99 %lld  max %lld\n",
                      static_cast<long long>(s.decode_lag.percentile(50)),
                      static_cast<long long>(s.decode_lag.percentile(99)),
                      static_cast<long long>(s.decode_lag.percentile(100)));
        out << lag;
    }
    print_stage(out, "decode", s.decode_ns);
    print_stage(out, "warp + encode", s.process_encode_ns);
    print_stage(out, "receive -> output", s.receive_to_output_ns);
}

} // namespace video_stage
