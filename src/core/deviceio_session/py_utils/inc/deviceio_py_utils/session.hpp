// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/remote_session.hpp>
#include <deviceio_session/remote_session_client.hpp>
#include <deviceio_session/remote_session_host.hpp>
#include <deviceio_session/replay_session.hpp>
#include <mcap/sample_stream.hpp>
#include <pybind11/pybind11.h>
#include <remote_link/tcp_transport.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

//! Whether a destructor may release the GIL around a blocking teardown: pybind runs it with the GIL held, and a
//! finalizing interpreter must keep it.
inline bool can_release_gil()
{
    return Py_IsInitialized() && PyGILState_Check();
}

//! Longest a blocking remote-session call waits without the GIL before it runs Python's signal handlers.
constexpr std::chrono::milliseconds kSignalCheckPeriod{ 100 };
//! Longest TCP connect attempt of RemoteSession.start(); Python's signal handlers run after each. Well above any
//! round trip, since an attempt shorter than one can never connect.
constexpr std::chrono::milliseconds kConnectAttempt{ 2000 };

/*!
 * @brief Calls `wait(slice)` with the GIL released, in slices of at most kSignalCheckPeriod, until it returns true or
 *        `timeout` passes (negative: no limit); false then.
 *
 * Between slices it takes the GIL back and runs Python's signal handlers, so Ctrl-C raises KeyboardInterrupt
 * (thrown as py::error_already_set). Call with the GIL held.
 */
template <typename Wait>
bool wait_checking_signals(std::chrono::milliseconds timeout, Wait&& wait)
{
    const bool bounded = timeout.count() >= 0;
    const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
    while (true)
    {
        std::chrono::milliseconds slice = kSignalCheckPeriod;
        if (bounded)
        {
            const auto left = std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            slice = std::clamp(left, std::chrono::milliseconds{ 0 }, kSignalCheckPeriod);
        }
        bool done = false;
        {
            py::gil_scoped_release release;
            done = wait(slice);
        }
        if (done)
        {
            return true;
        }
        if (PyErr_CheckSignals() != 0)
        {
            throw py::error_already_set();
        }
        if (bounded && std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
    }
}

/**
 * @brief Python-facing session wrapper: destroys the underlying DeviceIOSession in __exit__.
 *
 * Binding DeviceIOSession directly with a no-op __exit__ leaves destruction to the pybind
 * holder, which can run after the OpenXR session is torn down and produces invalid-handle
 * errors. This type inherits ITrackerSession and forwards get_tracker_impl() so tracker
 * accessors and MCAP record() accept the same Python object.
 */
class PyDeviceIOSession : public ITrackerSession
{
public:
    explicit PyDeviceIOSession(std::unique_ptr<DeviceIOSession> impl) : impl_(std::move(impl))
    {
    }

    void update()
    {
        if (!impl_)
        {
            throw std::runtime_error("Session has been closed/destroyed");
        }
        impl_->update();
    }

    void close()
    {
        impl_.reset();
    }

    PyDeviceIOSession& enter()
    {
        if (!impl_)
        {
            throw std::runtime_error("Session has been closed/destroyed");
        }
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

    DeviceIOSession& native()
    {
        if (!impl_)
        {
            throw std::runtime_error("Session has been closed/destroyed");
        }
        return *impl_;
    }

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override
    {
        if (!impl_)
        {
            throw std::runtime_error("Session has been closed/destroyed");
        }
        return impl_->get_tracker_impl(tracker);
    }

private:
    std::unique_ptr<DeviceIOSession> impl_;
};

/**
 * @brief Python-facing wrapper for ReplaySession with the same context-manager
 *        and lifetime semantics as PyDeviceIOSession.
 */
class PyReplaySession : public ITrackerSession
{
public:
    explicit PyReplaySession(std::unique_ptr<ReplaySession> impl) : impl_(std::move(impl))
    {
    }

    void update()
    {
        if (!impl_)
        {
            throw std::runtime_error("ReplaySession has been closed/destroyed");
        }
        impl_->update();
    }

    void close()
    {
        impl_.reset();
    }

    PyReplaySession& enter()
    {
        if (!impl_)
        {
            throw std::runtime_error("ReplaySession has been closed/destroyed");
        }
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override
    {
        if (!impl_)
        {
            throw std::runtime_error("ReplaySession has been closed/destroyed");
        }
        return impl_->get_tracker_impl(tracker);
    }

private:
    std::unique_ptr<ReplaySession> impl_;
};

/**
 * @brief Python-facing RemoteSession: the sample stream, the RemoteSession reading it and, after start() or
 *        connect(), the RemoteSessionClient feeding it. Same context-manager semantics as PyDeviceIOSession.
 *
 * start(), connect(), wait_hello_ack(), wait_ready(), close() and the destructor release the GIL for their blocking
 * part; the waits run Python's signal handlers every kSignalCheckPeriod, a TCP connect after each kConnectAttempt.
 * Members change only while the GIL is held, and a released call keeps its own reference to the client, so close()
 * on another thread cannot free it. A wait that throws PeerClosedError leaves the session unconnected.
 */
class PyRemoteSession : public ITrackerSession
{
public:
    using TrackerNames = std::vector<std::pair<std::shared_ptr<ITracker>, std::string>>;

    ~PyRemoteSession() override
    {
        if (can_release_gil())
        {
            close();
        }
    }

    static std::unique_ptr<PyRemoteSession> create(const TrackerNames& trackers, int64_t max_tick_age_ns)
    {
        if (max_tick_age_ns <= 0)
        {
            throw std::invalid_argument("RemoteSession: max_tick_age_ns must be positive");
        }
        std::unique_ptr<PyRemoteSession> self(new PyRemoteSession());
        RemoteSessionConfig config;
        config.stream = self->stream_;
        config.max_tick_age_ns = max_tick_age_ns;
        for (const auto& [tracker, name] : trackers)
        {
            config.tracker_names.emplace_back(tracker.get(), name);
            self->trackers_.push_back(tracker);
        }
        self->session_ = RemoteSession::run(std::move(config));
        return self;
    }

    std::vector<std::string> subscribed_topics() const
    {
        check_open();
        return stream_->subscribed_topics();
    }

    void start(const std::string& endpoint, std::string hello_json, int64_t connect_timeout_ms)
    {
        check_open();
        if (client_ || connecting_)
        {
            throw std::runtime_error("RemoteSession is already connected");
        }
        connecting_ = true;
        std::unique_ptr<RemoteSessionClient> client;
        try
        {
            client = start_client(endpoint, hello_json, std::chrono::milliseconds(connect_timeout_ms));
        }
        catch (...)
        {
            connecting_ = false;
            throw;
        }
        connecting_ = false;
        client_ = std::move(client);
    }

    bool wait_hello_ack(int64_t timeout_ms)
    {
        return wait_on_client(timeout_ms, [](RemoteSessionClient& client, std::chrono::milliseconds slice)
                              { return client.wait_hello_ack(slice); });
    }

    //! start(), then wait_hello_ack(); on failure the session is left unconnected.
    void connect(const std::string& endpoint, std::string hello_json, int64_t timeout_ms)
    {
        const auto begin = std::chrono::steady_clock::now();
        start(endpoint, std::move(hello_json), timeout_ms);
        try
        {
            int64_t left_ms = timeout_ms;
            if (timeout_ms >= 0)
            {
                const auto spent =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
                left_ms = std::max<int64_t>(0, timeout_ms - spent.count());
            }
            if (!wait_hello_ack(left_ms))
            {
                throw RemoteConnectTimeoutError("No HELLO_ACK from the Teleop Session Server before the connect timeout");
            }
        }
        catch (...)
        {
            disconnect();
            throw;
        }
    }

    bool wait_ready(int64_t timeout_ms)
    {
        return wait_on_client(timeout_ms, [](RemoteSessionClient& client, std::chrono::milliseconds slice)
                              { return client.wait_ready(slice) == RemoteReadyState::Ready; });
    }

    std::optional<std::string> latest_status() const
    {
        return client_ ? client_->latest_status() : std::nullopt;
    }

    std::string hello_ack() const
    {
        return client_ ? client_->hello_ack() : std::string();
    }

    void update()
    {
        connected_client()->throw_if_failed();
        session_->update();
    }

    bool is_fresh() const
    {
        check_open();
        return session_->is_fresh();
    }

    std::optional<int64_t> tick_age_ns() const
    {
        check_open();
        return session_->tick_age_ns();
    }

    void close()
    {
        // All members go before the GIL is released, so a start() on another thread meanwhile finds the session
        // closed. The stream and trackers are declared before the release: they go last, with the GIL held again.
        std::shared_ptr<TrackerSampleStreamReader> stream = std::exchange(stream_, nullptr);
        std::vector<std::shared_ptr<ITracker>> trackers = std::exchange(trackers_, {});
        std::unique_ptr<RemoteSession> session = std::exchange(session_, nullptr);
        std::shared_ptr<RemoteSessionClient> client = std::exchange(client_, nullptr);
        py::gil_scoped_release release;
        if (client)
        {
            client->close();
            client.reset();
        }
        session.reset();
    }

    PyRemoteSession& enter()
    {
        check_open();
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override
    {
        check_open();
        return session_->get_tracker_impl(tracker);
    }

private:
    PyRemoteSession() = default;

    /*!
     * @brief RemoteSessionClient::start() in TCP connect attempts of at most kConnectAttempt, each without the GIL.
     *
     * A timed-out attempt is retried until `timeout` (negative: no limit). Python's signal handlers run after every
     * attempt, so Ctrl-C raises KeyboardInterrupt instead of the attempt's outcome. Throws once close() ran meanwhile.
     */
    std::unique_ptr<RemoteSessionClient> start_client(const std::string& endpoint,
                                                      const std::string& hello_json,
                                                      std::chrono::milliseconds timeout)
    {
        // A reference of its own: close() on another thread drops the member.
        const std::shared_ptr<TrackerSampleStreamReader> stream = stream_;
        const bool bounded = timeout.count() >= 0;
        const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
        while (true)
        {
            std::chrono::milliseconds attempt = kConnectAttempt;
            if (bounded)
            {
                attempt =
                    std::clamp(std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()),
                               std::chrono::milliseconds{ 0 }, kConnectAttempt);
            }
            std::unique_ptr<RemoteSessionClient> client;
            std::exception_ptr error;
            bool timed_out = false;
            {
                py::gil_scoped_release release;
                try
                {
                    client = RemoteSessionClient::start(endpoint, hello_json, stream, attempt);
                }
                catch (const TcpConnectTimeoutError&)
                {
                    timed_out = true;
                    error = std::current_exception();
                }
                catch (...)
                {
                    error = std::current_exception();
                }
            }
            const bool interrupted = PyErr_CheckSignals() != 0;
            if (interrupted || !session_)
            {
                if (client)
                {
                    py::gil_scoped_release release;
                    client.reset();
                }
                if (interrupted)
                {
                    throw py::error_already_set();
                }
                // close() ran while connecting.
                throw std::runtime_error("RemoteSession has been closed/destroyed");
            }
            if (!error)
            {
                return client;
            }
            if (!timed_out || (bounded && std::chrono::steady_clock::now() >= deadline))
            {
                std::rethrow_exception(error);
            }
        }
    }

    void check_open() const
    {
        if (!session_)
        {
            throw std::runtime_error("RemoteSession has been closed/destroyed");
        }
    }

    std::shared_ptr<RemoteSessionClient> connected_client() const
    {
        check_open();
        if (!client_)
        {
            throw std::runtime_error("RemoteSession is not connected; call connect() first");
        }
        return client_;
    }

    //! wait_checking_signals() on the connected client, through a reference of its own. A PeerClosedError drops
    //! the client, so start() may connect again.
    template <typename Wait>
    bool wait_on_client(int64_t timeout_ms, Wait wait)
    {
        // Dropped without the GIL: after a close() on another thread it is the last reference.
        struct Held
        {
            std::shared_ptr<RemoteSessionClient> client;
            ~Held()
            {
                if (can_release_gil())
                {
                    py::gil_scoped_release release;
                    client.reset();
                }
            }
        } held{ connected_client() };
        try
        {
            return wait_checking_signals(std::chrono::milliseconds(timeout_ms),
                                         [&](std::chrono::milliseconds slice) { return wait(*held.client, slice); });
        }
        catch (const PeerClosedError&)
        {
            if (client_ == held.client)
            {
                disconnect();
            }
            throw;
        }
    }

    //! Closes and drops the client, without the GIL. It also ends a wait on another thread.
    void disconnect()
    {
        std::shared_ptr<RemoteSessionClient> client = std::exchange(client_, nullptr);
        if (!client)
        {
            return;
        }
        py::gil_scoped_release release;
        client->close();
        client.reset();
    }

    // Destroyed in reverse order: the client stops feeding the stream before the session and trackers go.
    std::vector<std::shared_ptr<ITracker>> trackers_;
    std::shared_ptr<TrackerSampleStreamReader> stream_ = std::make_shared<TrackerSampleStreamReader>();
    std::unique_ptr<RemoteSession> session_;
    std::shared_ptr<RemoteSessionClient> client_;
    bool connecting_ = false;
};

/**
 * @brief Python-facing RemoteSessionHost.
 *
 * wait_hello(), start_session(), tick(), close() and the destructor release the GIL. A mutex keeps start_session(),
 * tick() and close() from overlapping, as RemoteSessionHost requires; it is only ever waited for without the GIL or
 * held without waiting for the GIL, so the two cannot deadlock.
 */
class PyRemoteSessionHost
{
public:
    // Closing lingers until the robot closes its end (up to about 1.5 s).
    ~PyRemoteSessionHost()
    {
        if (can_release_gil())
        {
            close();
        }
    }

    static std::unique_ptr<PyRemoteSessionHost> adopt(int fd, bool websocket)
    {
        return std::unique_ptr<PyRemoteSessionHost>(new PyRemoteSessionHost(RemoteSessionHost::adopt(fd, websocket)));
    }

    std::optional<std::string> wait_hello(int64_t timeout_ms)
    {
        py::gil_scoped_release release;
        return host_->wait_hello(std::chrono::milliseconds(timeout_ms));
    }

    void send_hello_ack(std::string json)
    {
        host_->send_hello_ack(std::move(json));
    }

    void send_status(std::string json)
    {
        host_->send_status(std::move(json));
    }

    void send_ready(std::string json)
    {
        host_->send_ready(std::move(json));
    }

    void send_error(std::string text)
    {
        host_->send_error(std::move(text));
    }

    void start_session(const PyRemoteSession::TrackerNames& trackers,
                       const OpenXRSessionHandles& handles,
                       const VendorConfig& vendor_config,
                       const std::vector<std::string>& topics)
    {
        std::vector<std::shared_ptr<ITracker>> tracker_list;
        std::vector<std::pair<const ITracker*, std::string>> tracker_names;
        for (const auto& [tracker, name] : trackers)
        {
            if (!tracker)
            {
                throw std::invalid_argument("RemoteSessionHost.start_session: tracker '" + name + "' is None");
            }
            tracker_list.push_back(tracker);
            tracker_names.emplace_back(tracker.get(), name);
        }
        {
            // A close() on another thread holds the mutex while it lingers.
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            check_open();
            host_->start_session(tracker_list, tracker_names, handles, vendor_config, topics);
        }
        trackers_ = std::move(tracker_list);
    }

    void tick()
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        check_open();
        host_->tick();
    }

    bool alive() const
    {
        return host_->alive();
    }

    std::string close_reason() const
    {
        return host_->close_reason();
    }

    void close()
    {
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            host_->close();
        }
        trackers_.clear();
    }

    PyRemoteSessionHost& enter()
    {
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

private:
    explicit PyRemoteSessionHost(std::unique_ptr<RemoteSessionHost> host) : host_(std::move(host))
    {
    }

    //! Call with mutex_ held.
    void check_open() const
    {
        if (closed_)
        {
            throw std::runtime_error("RemoteSessionHost has been closed");
        }
    }

    // The host, which ends the DeviceIOSession, goes before the trackers.
    std::vector<std::shared_ptr<ITracker>> trackers_;
    std::unique_ptr<RemoteSessionHost> host_;
    std::mutex mutex_;
    bool closed_ = false;
};

} // namespace core
