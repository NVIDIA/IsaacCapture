// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_clock.hpp"
#include "full_body_frame.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace plugins
{
namespace jmex
{

/*!
 * @brief Publishes the operator's body pose as ``FullBodyPose`` on the generic full-body path.
 *
 * Each frame arrives from ReceiverSession already in FullBodyPose's slots. The transport is owned
 * here so that publish() can replace it after a runtime restart.
 */
class FullBodyPublisher
{
public:
    //! Pushes one serialized sample: the buffer, its size, the sample time on the common clock and
    //! the raw device clock. Throws when the push fails.
    using PushBufferFunction = std::function<void(const uint8_t*, size_t, int64_t, int64_t)>;

    //! Brings a transport up and returns the function that pushes on it, which owns it: the
    //! transport goes when the function does. Throws if no transport can be brought up.
    //! @a wait_for_system: true waits untimed for an OpenXR system; false throws at once if there is none.
    using TransportFactory = std::function<PushBufferFunction(bool wait_for_system)>;

    //! Waits @a wait_ms before a recovery attempt. Returns false, without waiting the rest, once
    //! @a stop is set.
    using BackoffWait = std::function<bool(int wait_ms, const std::atomic<bool>& stop)>;

    //! The collection's sample size. Must not exceed
    //! LiveFullBodyTrackerJmexImpl::MAX_FLATBUFFER_SIZE, or the reader throws.
    static constexpr size_t kMaxFlatbufferSize = 16 * 1024;

    //! @param collection_id The collection to publish on, used verbatim: what the operator passed
    //!        to ``--collection-id``.
    //! @param device_clock What each sample carries as its raw device clock.
    //! @throws std::runtime_error if the OpenXR session cannot be created.
    FullBodyPublisher(const std::string& collection_id, DeviceClock device_clock);

    //! Publishes on the transports @a create_transport brings up, in place of OpenXR, and sleeps out
    //! each backoff as the constructor above does.
    //! @param create_transport Called here, and again for each recovery attempt.
    //! @throws whatever @a create_transport throws.
    FullBodyPublisher(TransportFactory create_transport, DeviceClock device_clock);

    //! As above, with @a wait waiting out each backoff.
    FullBodyPublisher(TransportFactory create_transport, DeviceClock device_clock, BackoffWait wait);

    FullBodyPublisher(const FullBodyPublisher&) = delete;
    FullBodyPublisher& operator=(const FullBodyPublisher&) = delete;
    FullBodyPublisher(FullBodyPublisher&&) = delete;
    FullBodyPublisher& operator=(FullBodyPublisher&&) = delete;

    /*!
     * @brief Serialize one frame from ReceiverSession::poll() and push it as one sample.
     *
     * @param stop Checked during a recovery backoff. Once a call returns with it set, do not call again.
     * @return Whether the frame went out. A frame whose push failed is not sent again.
     * @throws std::runtime_error when the session is not re-established within the retry budget.
     */
    [[nodiscard]] bool publish(const FullBodyFrame& frame, const std::atomic<bool>& stop);

    //! Pushes that threw, each of which cost the frame it was carrying.
    uint64_t push_failures() const
    {
        return push_failures_;
    }

    //! Sessions re-established after a push threw.
    uint64_t session_recoveries() const
    {
        return session_recoveries_;
    }

    //! Recovery episodes that spent the whole budget without getting a session back. At most one,
    //! since the next thing that happens is the throw that ends the run.
    uint64_t recoveries_abandoned() const
    {
        return recoveries_abandoned_;
    }

private:
    //! Bring a transport up in place of the one held. Also the recovery path.
    //! @param wait_for_system True at startup, false when recovering, where the backoff paces the
    //!        retries. TransportFactory says what each does.
    void establish_session(bool wait_for_system);

    //! Re-establish the session after a push threw, waiting out a bounded backoff before each
    //! attempt. \return false if \a stop was set or the budget ran out.
    bool recover_session(const std::atomic<bool>& stop);

    TransportFactory create_transport_;
    BackoffWait wait_;
    PushBufferFunction push_buffer_;
    DeviceClock device_clock_;
    uint64_t push_failures_ = 0;
    uint64_t session_recoveries_ = 0;
    uint64_t recoveries_abandoned_ = 0;
};

} // namespace jmex
} // namespace plugins
