// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

namespace plugins
{
namespace jmex
{

//! Where the SDK sends channel advertisements when ReceiverConfig::broadcast_ip is unset.
inline constexpr const char* kSdkDefaultBroadcastIp = "255.255.255.255";

//! How logs and --help name an unset ReceiverConfig::bind_ip.
inline constexpr const char* kEveryInterface = "every interface";

/*!
 * @brief What the SDK needs from the host to bring a receiver up: the fields of the SDK's sample
 *        config (MOXIReceiverSDK/samples/MOXIReceiverCfg.h), all properties of this machine
 *        rather than of the device.
 */
struct ReceiverConfig
{
    //! MOXI channel to open, one skeleton per channel. 0 is a suit connected over USB; 1..10 is
    //! what a live capture assigns; the plugin README names the one Player's file replay uses.
    int channel = 1;

    //! Local address the receiver's TCP server binds and sends channel advertisements from; Player
    //! connects back to it. Unset means every interface, advertising from each broadcast-capable,
    //! non-loopback one.
    std::optional<std::string> bind_ip;

    //! Where channel advertisements are broadcast. Unset means kSdkDefaultBroadcastIp.
    std::optional<std::string> broadcast_ip;

    //! TCP port Player connects back on, advertised in every broadcast. Each receiver on one
    //! machine needs its own.
    int tcp_port = 7000;

    //! Base of the per-channel data ports the SDK opens. Advertisements still go to Player's fixed port.
    int udp_start_port = 10100;
};

} // namespace jmex
} // namespace plugins
