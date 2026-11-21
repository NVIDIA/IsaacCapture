// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_session/remote_session_host.hpp"

#include "generated_output_watch_includes.inc"

#include <deviceio_trackers/full_body_tracker.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/sample_stream.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/websocket_transport.hpp>
#include <robot_state_store/robot_state_store.hpp>
#include <schema/full_body_generated.h>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <iterator>
#include <mutex>
#include <span>
#include <stdexcept>

namespace core
{

namespace
{

// Link death notifies nobody, so waits wake this often to check it.
constexpr std::chrono::milliseconds kLivenessPoll{ 20 };
// How long close() waits for the robot to close its end after the last message.
constexpr std::chrono::milliseconds kCloseLinger{ 1000 };

std::span<const uint8_t> samples_records(const std::vector<uint8_t>& payload)
{
    return std::span<const uint8_t>(payload).subspan(kSamplesPrefixSize);
}

/*!
 * @brief A tracker type whose live output can go away without a record saying so.
 *
 * Plugin-fed trackers record only new samples. When the plugin's collection goes away, the live impl empties its
 * output, which a local reader sees as no data, and records nothing, so the robot would keep the last sample.
 */
struct OutputWatch
{
    bool (*matches)(const ITracker& tracker);
    bool (*has_output)(const ITracker& tracker, const ITrackerSession& session);
    //! The absent payload, as the channel replay reads records it.
    std::vector<uint8_t> (*no_data_record)(int64_t time_ns);
    const char* channel;
};

template <typename TrackerT, typename RecordT, auto Get>
OutputWatch output_watch(const char* channel)
{
    return OutputWatch{
        [](const ITracker& tracker) { return dynamic_cast<const TrackerT*>(&tracker) != nullptr; },
        [](const ITracker& tracker, const ITrackerSession& session)
        { return static_cast<bool>((static_cast<const TrackerT&>(tracker).*Get)(session)); },
        [](int64_t time_ns)
        {
            const auto record = pack_record<RecordT>(nullptr, DeviceDataTimestamp(time_ns, time_ns, time_ns));
            return std::vector<uint8_t>(record.buffer().begin(), record.buffer().end());
        },
        channel,
    };
}

// Full body is plugin-fed for the Noitom and Xsens vendors.
const OutputWatch kOutputWatches[] = {
    output_watch<FullBodyTracker, FullBodyPoseRecord, &FullBodyTracker::get_body_pose>(
        FullBodyRecordingTraits::replay_channels[0]),
#include "generated_output_watches.inc"
};

} // namespace

struct RemoteSessionHost::WatchedOutput
{
    const ITracker* tracker;
    const OutputWatch* watch;
    std::string topic;
    bool had_output = false;
};

std::vector<uint8_t> merge_samples_payloads(std::vector<uint8_t> older, std::vector<uint8_t> newer)
{
    decode_samples_prefix(older); // validates the length
    const SamplesPrefix prefix = decode_samples_prefix(newer);
    return encode_samples_payload(prefix, merge_samples_records(samples_records(older), samples_records(newer)));
}

struct RemoteSessionHost::State
{
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::optional<std::string> hello;

    //! Guards the members below and is held while the store takes a payload or a session change, so nothing
    //! reaches a store once it is detached.
    std::mutex robot_mutex;
    std::shared_ptr<RobotStateStore> robot_store;
    bool robot_error_logged = false; //!< for the attached store
    bool robot_closed = false; //!< close() ran
};

RemoteSessionHost::RemoteSessionHost(std::unique_ptr<ITransport> transport, LinkOptions link_options)
    : state_(std::make_unique<State>())
{
    link_options.send_pings = false;
    link_options.name = "RemoteSessionHost";
    link_ = std::make_unique<Link>(std::move(transport), std::move(link_options),
                                   [this](ReceivedLinkMessage&& message) { handle(std::move(message)); });
}

RemoteSessionHost::~RemoteSessionHost()
{
    close();
}

std::unique_ptr<RemoteSessionHost> RemoteSessionHost::adopt(int fd, bool websocket, LinkOptions link_options)
{
    return adopt(adopt_transport(fd, websocket), std::move(link_options));
}

std::unique_ptr<RemoteSessionHost> RemoteSessionHost::adopt(std::unique_ptr<ITransport> transport,
                                                            LinkOptions link_options)
{
    std::unique_ptr<RemoteSessionHost> host(new RemoteSessionHost(std::move(transport), std::move(link_options)));
    host->link_->start();
    return host;
}

void RemoteSessionHost::handle(ReceivedLinkMessage&& message)
{
    switch (message.type)
    {
    case LinkMessageType::RobotStateSetup:
    case LinkMessageType::RobotStateSamples:
    case LinkMessageType::RobotStateRecords:
        handle_robot_state(message);
        return;
    default:
        break;
    }
    if (message.type != LinkMessageType::Hello)
    {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->hello)
        {
            state_->hello = std::string(message.payload.begin(), message.payload.end());
        }
    }
    state_->cv.notify_all();
}

void RemoteSessionHost::handle_robot_state(const ReceivedLinkMessage& message)
{
    std::lock_guard<std::mutex> lock(state_->robot_mutex);
    if (!state_->robot_store)
    {
        return;
    }
    // An exception that leaves the handler ends the link; a payload the store refuses ends nothing.
    try
    {
        if (message.type == LinkMessageType::RobotStateSetup)
        {
            state_->robot_store->add_setup(message.payload);
        }
        else
        {
            state_->robot_store->add_samples(message.payload, message.local_receive_ns);
        }
    }
    catch (const std::exception& error)
    {
        if (!state_->robot_error_logged)
        {
            state_->robot_error_logged = true;
            isaaccapture::Logger::get("isaaccapture.core.RemoteSessionHost")
                ->error(
                    "RemoteSessionHost: dropped a {} link message the robot-state store refused: {} (logged once "
                    "per robot app)",
                    to_string(message.type), error.what());
        }
    }
}

void RemoteSessionHost::attach_robot_state(std::shared_ptr<RobotStateStore> store)
{
    std::lock_guard<std::mutex> lock(state_->robot_mutex);
    if (state_->robot_closed)
    {
        return;
    }
    if (state_->robot_store)
    {
        state_->robot_store->end_session();
    }
    state_->robot_store = std::move(store);
    state_->robot_error_logged = false;
    if (state_->robot_store)
    {
        state_->robot_store->begin_session();
    }
}

std::optional<std::string> RemoteSessionHost::wait_hello(std::chrono::milliseconds timeout)
{
    const bool bounded = timeout.count() >= 0;
    const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
    std::unique_lock<std::mutex> lock(state_->mutex);
    while (true)
    {
        if (!link_->alive())
        {
            throw std::runtime_error("Remote session lost: " + link_->close_reason());
        }
        if (state_->hello)
        {
            return state_->hello;
        }
        auto wait = kLivenessPoll;
        if (bounded)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
            {
                return std::nullopt;
            }
            wait = std::min(wait, std::chrono::ceil<std::chrono::milliseconds>(deadline - now));
        }
        state_->cv.wait_for(lock, wait);
    }
}

void RemoteSessionHost::send_text(LinkMessageType type, const std::string& text)
{
    link_->send_reliable(type, std::vector<uint8_t>(text.begin(), text.end()));
}

void RemoteSessionHost::send_hello_ack(std::string json)
{
    send_text(LinkMessageType::HelloAck, json);
}

void RemoteSessionHost::send_status(std::string json)
{
    send_text(LinkMessageType::Status, json);
}

void RemoteSessionHost::send_ready(std::string json)
{
    send_text(LinkMessageType::Ready, json);
}

void RemoteSessionHost::send_error(std::string text)
{
    send_text(LinkMessageType::Error, text);
}

void RemoteSessionHost::start_session(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                      const std::vector<std::pair<const ITracker*, std::string>>& tracker_names,
                                      const OpenXRSessionHandles& handles,
                                      const VendorConfig& vendor_config,
                                      const std::vector<std::string>& topics)
{
    if (session_)
    {
        throw std::logic_error("RemoteSessionHost: session already started");
    }
    // The filter applies from the first record, so it is set before the session's writer opens.
    auto writer = std::make_unique<TrackerSampleStreamWriter>();
    writer->set_topic_filter(topics);
    McapRecordingConfig recording;
    recording.tracker_names = tracker_names;
    recording.sample_stream = writer.get();
    writer_ = std::move(writer);
    session_ = DeviceIOSession::run(trackers, handles, std::move(recording), vendor_config);
    for (const auto& [tracker, name] : tracker_names)
    {
        const auto watch = std::find_if(std::begin(kOutputWatches), std::end(kOutputWatches),
                                        [tracker](const OutputWatch& w) { return w.matches(*tracker); });
        if (watch == std::end(kOutputWatches))
        {
            continue;
        }
        std::string topic = mcap_topic(name, watch->channel);
        if (topics.empty() || std::find(topics.begin(), topics.end(), topic) != topics.end())
        {
            watched_.push_back(WatchedOutput{ tracker, &*watch, std::move(topic) });
        }
    }
    isaaccapture::Logger::get("isaaccapture.core.RemoteSessionHost")
        ->info("RemoteSessionHost: streaming {} trackers, {} topics", tracker_names.size(), topics.size());

    // Channels are declared with their first record, so this is usually empty.
    std::vector<uint8_t> setup = writer_->take_setup();
    if (!setup.empty())
    {
        link_->send_reliable(LinkMessageType::Setup, std::move(setup));
    }
}

void RemoteSessionHost::tick()
{
    if (!session_)
    {
        throw std::logic_error("RemoteSessionHost: tick() before start_session()");
    }
    const int64_t tick_time_ns = os_monotonic_now_ns();
    session_->update();
    for (WatchedOutput& watched : watched_)
    {
        const bool has_output = watched.watch->has_output(*watched.tracker, *session_);
        if (watched.had_output && !has_output)
        {
            writer_->add_record(
                watched.topic, watched.watch->no_data_record(tick_time_ns), static_cast<uint64_t>(tick_time_ns));
        }
        watched.had_output = has_output;
    }
    // Reliable SETUP is queued first; the link never sends a SAMPLES link message ahead of it.
    std::vector<uint8_t> setup = writer_->take_setup();
    if (!setup.empty())
    {
        link_->send_reliable(LinkMessageType::Setup, std::move(setup));
    }
    const std::vector<uint8_t> records = writer_->take_samples();
    link_->send_latest(LinkMessageType::Samples,
                       encode_samples_payload(SamplesPrefix{ next_tick_seq_++, tick_time_ns }, records),
                       merge_samples_payloads);
}

void RemoteSessionHost::stop_session()
{
    if (!session_)
    {
        // A start_session() that threw leaves its writer.
        writer_.reset();
        return;
    }
    const auto logger = isaaccapture::Logger::get("isaaccapture.core.RemoteSessionHost");
    try
    {
        // Records of a tick that threw belong to no tick.
        writer_->take_samples();
        const int64_t tick_time_ns = os_monotonic_now_ns();
        bool reported = false;
        // The robot keeps a topic's last record, which would read as fresh again once a later session ticks.
        for (const WatchedOutput& watched : watched_)
        {
            if (watched.had_output)
            {
                reported |= writer_->add_record(
                    watched.topic, watched.watch->no_data_record(tick_time_ns), static_cast<uint64_t>(tick_time_ns));
            }
        }
        if (reported)
        {
            std::vector<uint8_t> setup = writer_->take_setup();
            if (!setup.empty())
            {
                link_->send_reliable(LinkMessageType::Setup, std::move(setup));
            }
            const std::vector<uint8_t> records = writer_->take_samples();
            link_->send_latest(LinkMessageType::Samples,
                               encode_samples_payload(SamplesPrefix{ next_tick_seq_++, tick_time_ns }, records),
                               merge_samples_payloads);
        }
    }
    catch (const std::exception& error)
    {
        logger->error("RemoteSessionHost: stop_session() could not report a tracker without data: {}", error.what());
    }
    watched_.clear();
    session_.reset();
    writer_.reset();
    logger->info("RemoteSessionHost: stopped streaming");
}

bool RemoteSessionHost::alive() const
{
    return link_->alive();
}

std::string RemoteSessionHost::close_reason() const
{
    return link_->close_reason();
}

void RemoteSessionHost::close()
{
    {
        // Taken out first, so a later close(), the destructor and the handler find no store.
        std::lock_guard<std::mutex> lock(state_->robot_mutex);
        state_->robot_closed = true;
        if (state_->robot_store)
        {
            state_->robot_store->end_session();
            state_->robot_store.reset();
        }
    }
    watched_.clear();
    // The session writes the end of its recording into the writer, so it goes first.
    session_.reset();
    writer_.reset();
    link_->close_gracefully(kCloseLinger);
}

} // namespace core
