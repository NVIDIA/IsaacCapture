// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace plugins
{
namespace jmex
{

/*!
 * @brief What the SDK needs from the host to bring a receiver up: the fields of the SDK's sample
 *        config (MOXIReceiverSDK/samples/MOXIReceiverCfg.h), all properties of this machine
 *        rather than of the device.
 */
struct ReceiverConfig
{
    //! MOXI channel to open, one skeleton per channel. 1..10 is what a live capture assigns; 255
    //! is what Player uses when it replays a recording.
    int channel = 1;

    /*!
     * @brief Local address the receiver's TCP server binds. Empty means every interface.
     *
     * Advertisements go out from every broadcast-capable interface whatever this is set to, and
     * Player connects back to the address it heard one from: bind an address Player can reach.
     */
    std::string bind_ip;

    //! Where channel advertisements are broadcast. Empty means 255.255.255.255, which every MOXI
    //! Player on the subnet hears -- including ones on other machines, which then contend for the
    //! same channel. Naming this host keeps a pairing local.
    std::string broadcast_ip;

    //! TCP port Player connects back on, advertised in every broadcast. Each receiver on one
    //! machine needs its own.
    int tcp_port = 7000;

    //! Base of the per-channel data ports -- channel N receives on this + 1 + N. Only worth moving
    //! to make those ports predictable, for a firewall rule or to tell two receivers apart; a
    //! collision already relocates itself and announces where it landed. It does not move where
    //! advertisements are sent, which Player fixes at 10100.
    int udp_start_port = 10100;
};

} // namespace jmex
} // namespace plugins
