// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <revohuman_plugin/source.hpp>
#include <revohuman_plugin/types.hpp>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <revohuman.h>
#include <string>
#include <thread>
#include <vector>

namespace plugins
{
namespace revohuman
{

struct LiveSourceConfig
{
    double rate_hz = 200.0;
    //! Frames the SDK buffers per glove between two polls (about 1.3 s at 200 Hz).
    std::uint32_t queue_size = 256;
    //! Open and close gloves on a worker thread, so that plugging in one glove never holds up the
    //! other glove's samples. false does it inside poll(), for tests.
    bool background_open = true;
};

/*!
 * @brief Glove samples from the RevoHuman C SDK: at most one left and one right glove, found without
 * configuration. Discovery, two-hand kits, retries and give-up follow the plugin README ("Real devices").
 * Samples come from the SDK's gated subscription, each with its CLOCK_MONOTONIC sampling time. A reopened
 * glove starts over with a new HandMeta and SourceBatch::hand_restarted; the SDK cannot take one glove out
 * of a two-hand kit, so the end of a paired glove's stream restarts both hands.
 */
class LiveSource final : public Source
{
public:
    struct HandStats
    {
        std::uint64_t frames = 0; //!< Frames pulled from the SDK.
        std::uint64_t rejected = 0; //!< Frames that could not be converted.
        std::uint64_t without_kinematics = 0; //!< Samples the provider gave no joint poses for.
        std::uint64_t sdk_dropped = 0; //!< Frames the SDK dropped because its queue was full.
        std::uint64_t opens = 0; //!< Times the glove was opened.
        std::uint64_t ends = 0; //!< Times its subscription ended.
    };

    //! Tries to open each glove once. Throws std::invalid_argument for a bad config, and
    //! std::runtime_error with the SDK error when the SDK cannot start or neither glove can ever be
    //! opened (bad options, both gloves of a product without a provider, ...). Without a factory the
    //! samples carry no kinematics.
    explicit LiveSource(LiveSourceConfig config, KinematicsFactory kinematics = nullptr);
    ~LiveSource() override;

    LiveSource(const LiveSource&) = delete;
    LiveSource& operator=(const LiveSource&) = delete;

    SourceBatch poll(std::int64_t now_ns) override;
    std::optional<GloveNote> glove_note(Hand hand) const override;
    std::uint64_t records_released() const override
    {
        return released_;
    }

    //! Whether the hand's glove is open and streaming.
    bool opened(Hand hand) const;
    //! Whether the hand's open glove shares a two-hand kit with the other glove, sampled on the same
    //! USB frame.
    bool paired(Hand hand) const;
    //! The product code the SDK identified on the open glove ("" while the glove is not open).
    std::string product_code(Hand hand) const;
    //! The model_id() of the open glove's provider ("" while the glove is not open or has none).
    std::string model_id(Hand hand) const;
    HandStats stats(Hand hand) const;

private:
    //! rh_init and the log callback; undone (rh_shutdown, callback removed) once every handle is closed.
    struct SdkRuntime
    {
        SdkRuntime();
        ~SdkRuntime();
        SdkRuntime(const SdkRuntime&) = delete;
        SdkRuntime& operator=(const SdkRuntime&) = delete;
    };

    struct HandStream
    {
        RhKit* kit = nullptr;
        RhGlove* glove = nullptr; //!< Borrowed from the kit.
        RhSubscription* sub = nullptr;
        HandMeta meta;
        std::shared_ptr<const KinematicsProvider> kinematics; //!< For the identified product.
        bool gate_ok = true;
        RhSyncCause gate_cause = RH_CAUSE_NONE;
        std::int64_t last_gate_check_ns = 0;
        std::int64_t last_drop_report_ns = 0;
        std::uint64_t reported_dropped = 0;
        //! The kit holds both gloves: release() leaves it to release_pair(), which closes it once.
        bool paired = false;
    };

    enum class OpenResult
    {
        Opened,
        Retry,
        Fatal
    };

    //! What one rh_kit_open attempt gave, made on whichever thread ran it.
    struct OpenOutcome
    {
        OpenResult result = OpenResult::Fatal;
        RhStatus status = RH_OK;
        RhErrorCode code = RH_E_OK;
        std::string error;
        std::optional<HandStream> stream;
    };

    //! What one attempt to open both gloves gave: both from one two-hand kit, or else each from a
    //! kit of its own, tried in the same attempt.
    struct PairOutcome
    {
        std::array<OpenOutcome, 2> hands; //!< Indexed by Hand.
        std::string error; //!< Why the two-hand kit did not open ("" when it did).
    };

    struct HandSlot
    {
        enum class State
        {
            Closed, //!< No glove; looked for at next_open_ns.
            Opening, //!< An attempt is on the worker.
            Open,
            Failed, //!< Given up.
        };
        State state = State::Closed;
        std::optional<HandStream> stream;
        HandStats stats;
        std::uint64_t attempts = 0;
        std::optional<std::int64_t> next_open_ns; //!< Set on the first poll() without a glove.
        RhStatus open_status = RH_OK;
        std::string error; //!< Why the last open failed, the stream ended, or the hand was given up.
        bool meta_due = false; //!< The open glove's HandMeta has not been delivered yet.
        bool streamed = false; //!< A HandMeta was delivered: the next one restarts the hand.
        std::uint64_t dropped_before = 0; //!< sdk_dropped of the earlier streams.
    };

    //! One rh_kit_open attempt for the hand, and its subscription when the kit opens. Any thread.
    OpenOutcome open_hand(Hand hand) const;
    //! One rh_kit_open attempt for both gloves in one kit; when that fails, one for each on its own.
    //! Any thread.
    PairOutcome open_pair() const;
    void open_stream(Hand hand, HandStream& stream) const;
    //! Whether the SDK finds a glove for each hand on the USB bus, without opening them.
    bool both_plugged_in() const;
    void request_open(Hand hand, HandSlot& slot);
    void request_pair();
    void adopt(Hand hand, HandSlot& slot, OpenOutcome outcome);
    void adopt_pair(PairOutcome outcome);
    //! Takes what the worker opened since the last poll().
    void take_opened();
    //! Looks again for the hands without a glove whose time has come: both together when neither
    //! has one and the SDK finds both plugged in, else each on its own.
    void look_for_gloves(std::int64_t now_ns);
    //! Reopens both gloves in one kit when both stream on their own and are the same product.
    void consider_pairing(std::int64_t now_ns);
    //! Closes the subscription here and the kit on the worker (here once the worker stopped).
    void release(HandStream& stream);
    //! Closes the two-hand kit, once neither glove streams from it any more.
    void release_pair();
    void close_kit(RhKit* kit);
    void start_worker();
    void worker_loop();

    void service(Hand hand, HandSlot& slot, std::int64_t now_ns, SourceBatch& batch);
    //! Moves the queued samples into out; returns why the subscription ended, if it did.
    std::optional<std::string> pull(Hand hand, HandSlot& slot, std::vector<SourceRecord>& out);
    void reject(Hand hand, HandSlot& slot, const std::string& reason);
    void check_gate(Hand hand, HandStream& stream, std::int64_t now_ns);
    void check_dropped(Hand hand, HandSlot& slot, std::int64_t now_ns);
    void end_stream(Hand hand, HandSlot& slot, const std::string& reason, std::int64_t now_ns);
    //! Releases the open glove and looks for it again in two seconds.
    void close_stream(HandSlot& slot, std::string error, std::int64_t now_ns);

    //! First, so that it is destroyed last, after the destructor closed every handle.
    std::optional<SdkRuntime> runtime_;
    LiveSourceConfig config_;
    KinematicsFactory kinematics_; //!< Called from open_stream(), on either thread.
    std::array<HandSlot, 2> slots_; //!< Indexed by Hand; poll() thread only.
    RhKit* pair_kit_ = nullptr; //!< The two-hand kit the paired gloves stream from; poll() thread only.
    //! The open gloves streamed on their own after they could not be opened together, so they are
    //! not reopened together until either stream ends; poll() thread only.
    bool unpairable_ = false;
    std::string pair_error_; //!< Why the last attempt to open both gloves together failed.
    std::uint64_t pair_failures_ = 0;
    bool finished_ = false;
    std::uint64_t released_ = 0;

    // Shared with the worker, guarded by mutex_.
    std::mutex mutex_;
    std::condition_variable wake_;
    std::array<bool, 2> open_wanted_{ false, false };
    bool pair_wanted_ = false;
    std::array<std::optional<OpenOutcome>, 2> opened_;
    std::optional<PairOutcome> pair_opened_;
    std::vector<RhKit*> to_close_;
    bool stop_ = false; //!< Written only by the destructor.
    std::thread worker_; //!< Started by the first open or close that goes to the background.
};

//! "E1101 TransportDisconnected (the glove's USB link dropped)" for the calling thread's last SDK error:
//! the error code, the SDK's exception class name and, for errors a user can act on, an English hint.
//! The SDK's own message after the class name is logged at debug level only.
std::string rh_error_text();

} // namespace revohuman
} // namespace plugins
