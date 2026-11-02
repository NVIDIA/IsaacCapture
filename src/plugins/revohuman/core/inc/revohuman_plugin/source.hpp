// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! What a source hands to the per-hand pipelines.
using SourceRecord = std::variant<HandMeta, GloveSample>;

struct SourceBatch
{
    //! Indexed by Hand: the hand's glove was opened again before these records, so everything
    //! derived from its previous records (tick anchors, injected hand) must be dropped first.
    std::array<bool, 2> hand_restarted{ false, false };
    //! No more records will come.
    bool finished = false;
    std::vector<SourceRecord> records;
};

//! Feeds the plugin, polled from its tick loop. Sample times are CLOCK_MONOTONIC ns. A source
//! delivers a hand's HandMeta before that hand's first glove sample, and again after the hand
//! restarted.
class Source
{
public:
    virtual ~Source() = default;

    //! The records due at now_ns.
    virtual SourceBatch poll(std::int64_t now_ns) = 0;

    //! What the source knows about a glove that gives no samples, for device status.
    struct GloveNote
    {
        enum class Kind
        {
            Waiting, //!< Not open; the source keeps looking for it.
            NotReady, //!< Open but not delivering, for example while the capture gate is closed.
            Ended, //!< Gone for good.
        };
        Kind kind = Kind::NotReady;
        std::string text;
    };
    //! nullopt when the source has nothing to say about this hand.
    virtual std::optional<GloveNote> glove_note(Hand hand) const
    {
        static_cast<void>(hand);
        return std::nullopt;
    }

    virtual std::uint64_t records_released() const = 0;

protected:
    Source() = default;
    Source(const Source&) = default;
    Source(Source&&) = default;
    Source& operator=(const Source&) = default;
    Source& operator=(Source&&) = default;
};

/*!
 * @brief Hand geometry provider for live glove angles.
 *
 * The plugin-local implementation maps the glove angles to the 26 OpenXR joints, with its own FK
 * or the SDK's mechanical poses.
 * Without a provider, live glove samples carry no kinematics: joint states and tactile skip
 * reasons are still pushed, the hand is not injected.
 */
class KinematicsProvider
{
public:
    virtual ~KinematicsProvider() = default;

    //! Identifies the hand model and the OpenXR joint layout, for the logs.
    virtual std::string model_id() const = 0;

    //! The 26 OpenXR joints in the WRIST frame for these joint angles (SDK order, rad), or
    //! nullopt if calculation failed. Per-point masks preserve partial validity.
    virtual std::optional<HandKinematics> solve(Hand hand,
                                                const std::array<double, kGloveJointCount>& joints_rad,
                                                std::uint32_t joints_valid_mask) const = 0;
};

//! Makes the provider for a glove of the product the SDK identified on it ("P1", "PTV1", ...), on
//! whichever thread opens the glove. Throws when that product has no usable provider; the glove is
//! then given up.
using KinematicsFactory = std::function<std::shared_ptr<const KinematicsProvider>(const std::string& product_code)>;

} // namespace revohuman
} // namespace plugins
