// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <oxr_utils/os_time.hpp>
#include <revohuman_plugin/hand_model.hpp>
#include <revohuman_plugin/types.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <revohuman.h>
#include <string>

namespace plugins
{
namespace revohuman
{

//! What device status says about a glove that is not streaming normally.
struct GloveNote
{
    enum class Kind
    {
        Waiting, //!< Not open; looked for again.
        NotReady, //!< Open, but the capture gate is closed.
        Unsynced, //!< Streaming without hardware sync to the other glove.
        Ended, //!< Given up, or open with a hand model the plugin cannot use.
    };
    Kind kind = Kind::Waiting;
    std::string text;
};

/*!
 * @brief Gloves through the RevoHuman C SDK's autokit (rh_autokit_*), at most one per hand, found without
 * configuration. The SDK opens, retries, pairs and reopens them on its own thread (its rules are in revohuman.h);
 * this class turns each newly opened stream into hand metadata, feeds its frames to the pipelines and reports.
 * Tactile is opened split from the frames: a glove whose firmware splits it sends every tactile sample (about 215 Hz,
 * above the 200 Hz frame rate) on its own, and the frames carry only the latest; other gloves send tactile with the
 * frames. Which of the two a glove does is settled when its stream opens.
 */
class GloveSource
{
public:
    //! Throws std::runtime_error when the SDK cannot start. Gloves that never open are reported, not thrown.
    GloveSource();
    ~GloveSource();
    GloveSource(const GloveSource&) = delete;
    GloveSource& operator=(const GloveSource&) = delete;

    //! Feeds what the SDK queued since the last call to the hand pipelines (indexed by Hand). A newly opened stream
    //! restarts its pipeline, on the stream's HandMeta when its hand model can be used.
    template <typename Pipeline>
    void poll(std::int64_t now_ns, const std::array<Pipeline*, 2>& pipelines)
    {
        for (const Hand hand : kHands)
        {
            Pipeline& pipeline = *pipelines[index_of(hand)];
            const Stream stream = refresh(hand, now_ns);
            if (stream == Stream::None)
                continue;
            if (stream == Stream::New)
            {
                const std::optional<HandMeta> meta = adopt(hand);
                pipeline.reset();
                if (meta)
                    pipeline.set_meta(*meta);
            }
            watch_tare(hand, now_ns);
            const std::int64_t received_ns = core::os_monotonic_now_ns();
            // Tactile first: the SDK queues a split sample before the frames that carry it.
            std::uint32_t generation = 0;
            std::optional<TactileSample> tactile;
            for (std::uint32_t n = 0; n < kTactileQueueSize && receive_tactile(hand, generation, tactile); ++n)
                if (tactile)
                    pipeline.on_tactile(generation, *tactile, received_ns);
            std::optional<GloveSample> sample;
            for (std::uint32_t n = 0; n < kQueueSize && receive(hand, sample); ++n)
                if (sample)
                    pipeline.on_glove(*sample, received_ns);
        }
    }
    //! nullopt while the hand's glove streams normally.
    std::optional<GloveNote> note(Hand hand) const;
    //! Both gloves were given up.
    bool finished() const;
    //! Asks each streaming glove with tactile sensors to tare them, a write to the glove: it zeroes the taxels at their
    //! current readings. The SDK sends the request once and never resends it; poll() logs how it ends.
    void request_tare();

private:
    //! Frames the SDK buffers per glove between polls (1.3 s at 200 Hz).
    static constexpr std::uint32_t kQueueSize = 256;
    static constexpr std::uint32_t kTactileQueueSize = 512; //!< The SDK's split tactile queue per glove.
    //! Frames of a new stream with tactile but no split sample before its tactile is taken from the frames (0.5 s).
    static constexpr std::uint32_t kSplitWaitFrames = 100;

    enum class Stream
    {
        None, //!< Not streaming, or a new stream whose identity could not be read yet.
        Adopted,
        New, //!< Streaming, and not adopted yet.
    };
    enum class TactileFrom
    {
        Undecided, //!< No split sample yet; the frames' tactile is not used.
        Frames, //!< The firmware does not split tactile.
        Split, //!< Every sample on its own; the frames' copies are not used.
    };
    //! The stream the pipeline holds metadata for; adopt() starts it afresh.
    struct AdoptedStream
    {
        std::uint32_t open_count = 0; //!< The SDK's open count of the stream, from 1.
        std::optional<HandModel> model;
        std::string product_code;
        std::optional<std::string> unusable; //!< Why its hand model cannot be used; its frames are dropped.
        bool has_tactile = false;
        std::uint64_t rejected = 0;
        std::uint64_t tactile_rejected = 0;
        TactileFrom tactile_from = TactileFrom::Undecided;
        std::uint32_t tactile_frames = 0; //!< Frames with tactile while undecided.
        RhTactileTareStatus tare = RH_STRUCT_INIT(RhTactileTareStatus); //!< As last logged.
        std::optional<std::int64_t> last_tare_poll_ns;
    };
    struct Slot
    {
        RhAutoHandStatus status = RH_STRUCT_INIT(RhAutoHandStatus);
        RhProductIdentity identity = RH_STRUCT_INIT(RhProductIdentity); //!< Of the stream refresh() found new.
        RhGloveInfo info = RH_STRUCT_INIT(RhGloveInfo); //!< Of the stream refresh() found new.
        AdoptedStream adopted;
        std::string logged_error; //!< The last open error logged, so retries with the same error stay quiet.
        std::uint64_t dropped_logged = 0; //!< Of the stream in the status.
        std::uint64_t tactile_dropped_logged = 0;
        std::optional<std::int64_t> last_drop_log_ns;
        //! The stream a tare was requested on whose result is not logged yet; it may end before the result.
        std::optional<std::uint32_t> tare_open;
    };

    Stream refresh(Hand hand, std::int64_t now_ns);
    void report(Hand hand, const RhAutoHandStatus& now, std::int64_t now_ns);
    //! Logs what changed in the adopted stream's tactile tare.
    void watch_tare(Hand hand, std::int64_t now_ns);
    //! The HandMeta of the newly opened stream, or nullopt when its hand model cannot be used.
    std::optional<HandMeta> adopt(Hand hand);
    //! Pops one frame of the adopted stream, false when there is none; @p sample is set when the frame is usable.
    bool receive(Hand hand, std::optional<GloveSample>& sample);
    //! Pops one split tactile sample like receive().
    bool receive_tactile(Hand hand, std::uint32_t& generation, std::optional<TactileSample>& tactile);
    //! The forces view newtons_ until the next call.
    TactileSample to_sample(const RhTactile& tactile);
    //! Whether this frame's tactile is the stream's tactile (see TactileFrom).
    bool tactile_on_frames(Hand hand);

    RhAutoKit* kit_ = nullptr;
    std::array<Slot, 2> slots_;
    std::array<float, RH_TACTILE_MAX_POINTS> newtons_{};
};

} // namespace revohuman
} // namespace plugins
