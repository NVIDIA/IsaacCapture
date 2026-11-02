// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A fake of the RevoHuman C SDK functions the live source calls, with controls to script it. It
// has two gloves, each of which can be plugged in or not and belongs to at most one open kit (one
// kit may hold both, unless the SDK identifies different products on them); it
// needs no device and starts no thread, and every function may be called from any thread. Its
// joint mapping is deliberately not the real one, so tests can tell the SDK's result from anything
// else: rad = deg / 100, plus 1 for product "PTV1". Joint names are "<side>_fake_joint_<index>".

#include <cstdint>
#include <optional>
#include <revohuman.h>
#include <string>
#include <vector>

namespace fake_rh
{

//! What the code under test asked of the SDK. Option strings are copied; NULL reads as "<null>".
struct Calls
{
    int init = 0;
    int kit_open = 0;
    int kit_close = 0;
    int scan = 0;
    int subscribe = 0;
    int sub_close = 0;
    int sub_close_after_kit_close = 0;
    bool log_callback_set = false; //!< A callback is installed now (passing NULL removes it).
    int shutdown = 0;
    int kits_open_at_shutdown = -1; //!< Kits still open at the last rh_shutdown.
    bool log_callback_set_at_shutdown = false;

    RhHands hands = -1; //!< Of the last rh_kit_open.
    std::vector<RhHands> open_hands; //!< Of every rh_kit_open, in order.
    std::string product;
    bool tracker_config_null = false;
    double rate_hz = 0.0;
    RhSyncPolicy sync = -1;

    std::uint32_t queue_size = 0;
    int ungated = -1;
    bool callback_given = false;

    std::string joints_product; //!< product_code of the last rh_joints_to_radians call.
};

//! Forgets everything, including kits and subscriptions the code under test leaked.
void reset();
//! Only while no other thread calls the SDK; snapshot() otherwise.
Calls& calls();
Calls snapshot();
//! Kits open now.
int open_kits();
//! Kits open now that hold both gloves.
int open_pair_kits();

//! The variant code rh_kit_identity reports (default "P1"); nullopt makes it fail. Its sensors follow
//! the SDK's product table: tactile only for "PTV1".
void set_variant(std::optional<std::string> code);
//! The same for this glove. rh_kit_open of both gloves fails with RH_E_VARIANT_MISMATCH when they
//! have different ones.
void set_variant(RhHand hand, std::optional<std::string> code);
//! Firmware SN and USB unit id of a glove (defaults: no SN, "fake-unit-<side>").
void set_glove_ids(RhHand hand, const std::string& sn, const std::string& unit_id);
//! The name rh_joint_name gives this joint of this hand; nullopt makes it fail with
//! RH_ERR_INVALID_ARG.
void set_joint_name(RhHand hand, std::uint32_t index, std::optional<std::string> name);
//! Whether the glove is plugged in (default: both are). rh_kit_open of an absent glove fails with
//! RH_ERR_NOT_FOUND and E1000, as the real SDK does (E1400 is a glove it cannot assign to a hand),
//! and rh_scan leaves it out; a kit already holding it is not affected, see end().
void set_present(RhHand hand, bool present);
//! Makes the next `times` rh_scan calls fail with this status and error.
void fail_next_scan(RhStatus status, RhErrorCode code, const std::string& message, int times = 1);
//! Makes the next `times` rh_kit_open calls fail with this status and error.
void fail_next_open(RhStatus status, RhErrorCode code, const std::string& message, int times = 1);
//! The same for opens of this glove only.
void fail_next_open(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message, int times = 1);
//! The same for opens of both gloves in one kit only, as when they cannot be synced.
void fail_next_pair_open(RhStatus status, RhErrorCode code, const std::string& message, int times = 1);
//! rh_kit_open of this glove sleeps this long before it does anything.
void set_open_delay_ms(RhHand hand, int ms);
//! rh_kit_open calls started so far, counted before their delay.
int opens_started();
//! Makes the next `times` rh_glove_subscribe calls of this glove fail with this status and error.
void fail_next_subscribe(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message, int times = 1);

//! Queues a frame on the glove's subscription, dropping the oldest when it is full.
void push(RhHand hand, const RhGloveFrame& frame);
//! Once its queue is empty, the glove's subscription fails with this status and error. A new
//! subscription of the glove starts without it.
void end(RhHand hand, RhStatus status, RhErrorCode code, const std::string& message);
void set_gate(RhHand hand, bool ok, RhSyncCause cause);

//! A gated frame sampled at sample_mono_ns, every joint valid, joint i at 10 * i degrees.
RhGloveFrame frame(RhHand hand, std::int64_t sample_mono_ns);

} // namespace fake_rh
