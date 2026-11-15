// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <remote_link/link.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/transport.hpp>
#include <remote_link/websocket_transport.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#    include <unistd.h>
#endif

namespace py = pybind11;

namespace core
{

// can_release_gil(), kSignalCheckPeriod, kConnectAttempt and wait_checking_signals() are copies of the helpers in
// deviceio_py_utils/session.hpp, which this DeviceIO-free module cannot include. They differ in GilRelease.

//! Whether a destructor may release the GIL around a blocking teardown: pybind runs it with the GIL held, and a
//! finalizing interpreter must keep it.
inline bool can_release_gil()
{
    return Py_IsInitialized() && PyGILState_Check();
}

/*!
 * @brief PyEval_RestoreThread(), in a way an exiting interpreter survives.
 *
 * Up to Python 3.13 a thread that takes the GIL back once finalization began is ended inside that call with
 * pthread_exit(), which unwinds the thread's stack without the GIL: through pybind11's dispatcher, which then drops
 * Python references, and through noexcept destructors, which terminate the process. A daemon thread waiting in
 * receive() would crash the exit. Here the unwinding stops where it starts: the thread stays parked, and the process
 * ends without it. Not inlined, so that no noexcept caller decides whether the cleanup runs.
 */
PYBIND11_NOINLINE void restore_thread_or_park(PyThreadState* state)
{
    struct Park
    {
        bool restored = false;
        ~Park()
        {
            while (!restored)
            {
                std::this_thread::sleep_for(std::chrono::hours{ 1 });
            }
        }
    } park;
    PyEval_RestoreThread(state);
    park.restored = true;
}

//! Releases the GIL for a scope, as py::gil_scoped_release does, and takes it back with restore_thread_or_park().
class GilRelease
{
public:
    GilRelease() : m_state(PyEval_SaveThread())
    {
    }
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

    ~GilRelease()
    {
        restore_thread_or_park(m_state);
    }

private:
    PyThreadState* m_state;
};

//! Longest a blocking call waits without the GIL before it runs Python's signal handlers.
constexpr std::chrono::milliseconds kSignalCheckPeriod{ 100 };
//! Longest TCP connect attempt of Link.connect(); Python's signal handlers run after each. Well above any round
//! trip, since an attempt shorter than one can never connect.
constexpr std::chrono::milliseconds kConnectAttempt{ 2000 };
//! Longest time span taken from Python: longer ones would overflow the nanosecond clock arithmetic.
constexpr int64_t kMaxSpanMs = int64_t{ 1 } << 40;

inline std::chrono::milliseconds clamped_ms(int64_t ms)
{
    return std::chrono::milliseconds(std::min(ms, kMaxSpanMs));
}

//! This process's id, which a forked child does not share with its parent. 0 where there is no fork().
inline long process_id()
{
#ifdef _WIN32
    return 0;
#else
    return static_cast<long>(getpid());
#endif
}

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
            GilRelease release;
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

//! Options of a PyLink: the Link's own, the limits of its inbox, and the types the inbox keeps only the newest of.
struct PyLinkOptions
{
    LinkOptions link;
    size_t max_received_messages = 4096;
    size_t max_received_bytes = 64u << 20;
    std::array<bool, 256> latest_types{};
};

// The bound result classes are types of this module: pybind11 binds one C++ type in one module only.

struct PyReceivedLinkMessage
{
    int type;
    py::bytes payload;
    int64_t remote_send_ns;
    int64_t local_receive_ns;
};

struct PyClockEstimate : ClockEstimate
{
};

struct PyLinkStats : LinkStats
{
    uint64_t received_queued = 0; //!< link messages in the inbox now
    uint64_t received_replaced = 0; //!< queued link messages of a latest type replaced by a newer one
};

/*!
 * @brief Marks a descriptor as owned by a live link that adopt() made, so that adopt() refuses it a second time.
 *
 * Two links on one descriptor would split its stream between them and close it twice.
 */
class OwnedDescriptor
{
public:
    explicit OwnedDescriptor(int fd) : m_fd(fd)
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_owned.insert(fd).second)
        {
            throw py::value_error("Link.adopt: fd " + std::to_string(fd) + " already belongs to a Link");
        }
    }
    OwnedDescriptor(OwnedDescriptor&& other) noexcept : m_fd(std::exchange(other.m_fd, -1))
    {
    }
    OwnedDescriptor(const OwnedDescriptor&) = delete;
    OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;

    ~OwnedDescriptor()
    {
        if (m_fd >= 0)
        {
            std::lock_guard<std::mutex> lock(s_mutex);
            s_owned.erase(m_fd);
        }
    }

private:
    int m_fd;
    static inline std::mutex s_mutex;
    static inline std::set<int> s_owned;
};

/*!
 * @brief Received link messages waiting for Link.receive(), bounded by count and by payload bytes.
 *
 * push() runs on the link's receiver thread and never touches Python. A link message of a latest type replaces the
 * queued one of that type and takes the newest place in the order.
 */
class LinkInbox
{
public:
    explicit LinkInbox(const PyLinkOptions& options)
        : m_max_messages(options.max_received_messages),
          m_max_bytes(options.max_received_bytes),
          m_latest_types(options.latest_types)
    {
    }

    //! Throws once a limit would be exceeded, which ends the link: dropping a reliable link message would break
    //! the order.
    void push(ReceivedLinkMessage&& message)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto queued = queued_latest(message.type);
            if (queued != m_messages.end())
            {
                m_bytes -= queued->payload.size();
                m_messages.erase(queued);
                ++m_replaced;
            }
            if (m_messages.size() >= m_max_messages)
            {
                throw std::runtime_error("more than " + std::to_string(m_max_messages) + " received link messages unread");
            }
            if (m_bytes + message.payload.size() > m_max_bytes)
            {
                throw std::runtime_error("more than " + std::to_string(m_max_bytes) +
                                         " bytes of received link messages unread");
            }
            m_bytes += message.payload.size();
            m_messages.push_back(std::move(message));
        }
        m_cv.notify_one();
    }

    //! The oldest queued link message, after waiting up to `timeout` for one.
    std::optional<ReceivedLinkMessage> pop(std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [this] { return !m_messages.empty(); }))
        {
            return std::nullopt;
        }
        ReceivedLinkMessage message = std::move(m_messages.front());
        m_messages.pop_front();
        m_bytes -= message.payload.size();
        return message;
    }

    //! Takes back what pop() handed out, as the oldest again and whatever the limits. A link message of a latest
    //! type is dropped instead when a newer one of its type came meanwhile.
    void put_back(ReceivedLinkMessage&& message)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (queued_latest(message.type) != m_messages.end())
            {
                ++m_replaced;
                return;
            }
            m_bytes += message.payload.size();
            m_messages.push_front(std::move(message));
        }
        m_cv.notify_one();
    }

    uint64_t queued() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_messages.size();
    }

    uint64_t replaced() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_replaced;
    }

private:
    //! The queued link message of `type` when it is a latest type; end() otherwise. Call with m_mutex held.
    std::deque<ReceivedLinkMessage>::iterator queued_latest(LinkMessageType type)
    {
        if (!m_latest_types[static_cast<uint8_t>(type)])
        {
            return m_messages.end();
        }
        return std::find_if(
            m_messages.begin(), m_messages.end(), [type](const ReceivedLinkMessage& m) { return m.type == type; });
    }

    const size_t m_max_messages;
    const size_t m_max_bytes;
    const std::array<bool, 256> m_latest_types;

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<ReceivedLinkMessage> m_messages;
    size_t m_bytes = 0; //!< payload bytes in m_messages
    uint64_t m_replaced = 0;
};

/*!
 * @brief Python-facing Link: a core::Link whose handler queues into a LinkInbox, which receive() reads.
 *
 * connect(), receive(), close() and the destructor release the GIL for their blocking part; the link's threads never
 * take it. A released call keeps its own reference to the state, and close() leaves the state in place: every method
 * works on a closed link.
 */
class PyLink
{
public:
    ~PyLink()
    {
        if (forked())
        {
            // Leaked: destroying the state would close the link, see close().
            static_cast<void>(new std::shared_ptr<State>(std::move(state_)));
        }
        else if (can_release_gil())
        {
            close(0);
        }
    }

    static std::unique_ptr<PyLink> connect(const std::string& endpoint, int64_t timeout_ms, const PyLinkOptions& options)
    {
        const RemoteAddress remote = parse_remote(endpoint);
        if (!names_port(endpoint))
        {
            // parse_remote() fills in the Teleop Session Server's port, and no link message tells a link that it
            // reached another service than its own.
            throw py::value_error("Link.connect: endpoint '" + endpoint + "' names no port");
        }
        return std::unique_ptr<PyLink>(new PyLink(open_transport(remote, clamped_ms(timeout_ms)), options));
    }

    static std::unique_ptr<PyLink> adopt(int fd, bool websocket, const PyLinkOptions& options)
    {
        require_socket(fd);
        OwnedDescriptor owned(fd);
        // A Python socket with a timeout is in non-blocking mode, in which a send fails once the socket's buffer
        // is full, and a failed send ends the link.
        py::module_::import("os").attr("set_blocking")(fd, true);
        return std::unique_ptr<PyLink>(new PyLink(adopt_transport(fd, websocket), options, std::move(owned)));
    }

    void send_reliable(int64_t type, const py::bytes& payload)
    {
        const LinkMessageType checked = checked_type(type);
        state_->link.send_reliable(checked, checked_payload(payload));
    }

    void send_latest(int64_t type, const py::bytes& payload)
    {
        const LinkMessageType checked = checked_type(type);
        state_->link.send_latest(checked, checked_payload(payload));
    }

    std::optional<PyReceivedLinkMessage> receive(int64_t timeout_ms)
    {
        const std::shared_ptr<State> state = state_;
        std::optional<ReceivedLinkMessage> message = state->inbox.pop(std::chrono::milliseconds{ 0 });
        if (!message && timeout_ms != 0)
        {
            wait_checking_signals(clamped_ms(timeout_ms),
                                  [&](std::chrono::milliseconds slice)
                                  {
                                      message = state->inbox.pop(slice);
                                      if (message || timeout_ms > 0 || state->link.alive())
                                      {
                                          return message.has_value();
                                      }
                                      // No limit: a dead link ends the wait, once what it queued last is taken.
                                      message = state->inbox.pop(std::chrono::milliseconds{ 0 });
                                      return true;
                                  });
        }
        if (!message)
        {
            return std::nullopt;
        }
        // A signal that came with the link message: if its handler raises (Ctrl-C), the caller never gets what
        // this call returns, so the link message stays in the inbox.
        if (PyErr_CheckSignals() != 0)
        {
            state->inbox.put_back(std::move(*message));
            throw py::error_already_set();
        }
        return PyReceivedLinkMessage{ static_cast<int>(message->type),
                                      py::bytes(reinterpret_cast<const char*>(message->payload.data()),
                                                message->payload.size()),
                                      message->remote_send_ns, message->local_receive_ns };
    }

    bool alive() const
    {
        return state_->link.alive();
    }

    std::string close_reason() const
    {
        return state_->link.close_reason();
    }

    bool peer_closed_before_any_message() const
    {
        return state_->link.peer_closed_before_any_message();
    }

    PyClockEstimate clock() const
    {
        return PyClockEstimate{ state_->link.clock() };
    }

    PyLinkStats stats() const
    {
        PyLinkStats stats{ state_->link.stats() };
        stats.received_queued = state_->inbox.queued();
        stats.received_replaced = state_->inbox.replaced();
        return stats;
    }

    void close(int64_t linger_ms)
    {
        if (linger_ms < 0)
        {
            throw py::value_error("Link.close: linger_ms must not be negative");
        }
        if (forked())
        {
            // The link's threads do not exist in a forked child: closing would wait for them for good, and shut
            // down the connection, which is the parent's.
            return;
        }
        const std::shared_ptr<State> state = state_;
        GilRelease release;
        if (linger_ms > 0)
        {
            state->link.close_gracefully(clamped_ms(linger_ms));
        }
        else
        {
            state->link.close();
        }
    }

    PyLink& enter()
    {
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close(0);
    }

private:
    struct State
    {
        State(std::unique_ptr<ITransport> transport, const PyLinkOptions& options, std::optional<OwnedDescriptor> mark)
            : inbox(options),
              link(std::move(transport),
                   options.link,
                   [this](ReceivedLinkMessage&& message) { inbox.push(std::move(message)); }),
              owned(std::move(mark))
        {
        }

        LinkInbox inbox;
        Link link; // After the inbox, so its threads are joined before the inbox they push into goes away.
        // Last, so the mark goes before the link closes the descriptor: its number may then be adopted at once.
        std::optional<OwnedDescriptor> owned;
    };

    PyLink(std::unique_ptr<ITransport> transport,
           const PyLinkOptions& options,
           std::optional<OwnedDescriptor> mark = std::nullopt)
        : state_(std::make_shared<State>(std::move(transport), options, std::move(mark)))
    {
        try
        {
            state_->link.start();
        }
        catch (...)
        {
            // A thread could not be started: the link then waits out close_drain for a sender that never ran.
            GilRelease release;
            state_.reset();
            throw;
        }
    }

    //! Whether this process is a forked child of the one that made the link.
    bool forked() const
    {
        return pid_ != process_id();
    }

    //! Whether an endpoint that parse_remote() accepted names its port.
    static bool names_port(const std::string& endpoint)
    {
        const size_t scheme = endpoint.find("://");
        std::string authority = endpoint.substr(scheme == std::string::npos ? 0 : scheme + 3);
        authority.resize(std::min(authority.size(), authority.find('/')));
        return parse_endpoint(authority, 0).port == parse_endpoint(authority, 1).port;
    }

    //! ValueError unless `fd` is an open socket. Asked of Python's os module: the wrapper has no POSIX code.
    static void require_socket(int fd)
    {
        bool is_socket = false;
        try
        {
            const py::object mode = py::module_::import("os").attr("fstat")(fd).attr("st_mode");
            is_socket = py::module_::import("stat").attr("S_ISSOCK")(mode).cast<bool>();
        }
        catch (py::error_already_set& e)
        {
            if (!e.matches(PyExc_OSError))
            {
                throw;
            }
        }
        if (!is_socket)
        {
            throw py::value_error("Link.adopt: fd " + std::to_string(fd) + " is not an open socket descriptor");
        }
    }

    /*!
     * @brief A connected transport for `remote` within `timeout` (negative: no limit).
     *
     * The TCP connect runs in attempts of at most kConnectAttempt, each without the GIL, and a timed-out attempt is
     * retried. Python's signal handlers run after every attempt, so Ctrl-C raises KeyboardInterrupt instead of the
     * attempt's outcome. Over ws:// the answer to the upgrade is then awaited as in wait_checking_signals().
     */
    static std::unique_ptr<ITransport> open_transport(const RemoteAddress& remote, std::chrono::milliseconds timeout)
    {
        const bool bounded = timeout.count() >= 0;
        const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
        const auto left = [&]
        {
            return std::max(std::chrono::milliseconds{ 0 },
                            std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()));
        };
        TcpSocket socket;
        while (true)
        {
            const std::chrono::milliseconds attempt = bounded ? std::min(left(), kConnectAttempt) : kConnectAttempt;
            std::exception_ptr error;
            bool timed_out = false;
            {
                GilRelease release;
                try
                {
                    socket = TcpSocket::connect(remote.endpoint, attempt);
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
            if (PyErr_CheckSignals() != 0)
            {
                throw py::error_already_set();
            }
            if (!error)
            {
                break;
            }
            if (!timed_out || (bounded && std::chrono::steady_clock::now() >= deadline))
            {
                std::rethrow_exception(error);
            }
        }
        if (!remote.websocket)
        {
            return std::make_unique<TcpTransport>(std::move(socket));
        }
        std::unique_ptr<WebSocketTransport> websocket = WebSocketTransport::start_client(std::move(socket), remote);
        if (!wait_checking_signals(bounded ? left() : timeout,
                                   [&](std::chrono::milliseconds slice) { return websocket->wait_upgraded(slice); }))
        {
            throw TcpConnectTimeoutError("cannot connect to " + endpoint_text(remote.endpoint) +
                                         ": no answer to the WebSocket upgrade");
        }
        return websocket;
    }

    //! HEARTBEAT, PING and PONG are the link's own: the peer's link consumes them, and a malformed PONG ends it.
    static LinkMessageType checked_type(int64_t type)
    {
        if (type < 0 || type > 255)
        {
            throw py::value_error("Link: type must be in [0, 255], got " + std::to_string(type));
        }
        const auto checked = static_cast<LinkMessageType>(type);
        if (checked == LinkMessageType::Heartbeat || checked == LinkMessageType::Ping || checked == LinkMessageType::Pong)
        {
            throw py::value_error("Link: types 11, 13 and 14 (HEARTBEAT, PING, PONG) are the link's own");
        }
        return checked;
    }

    static std::vector<uint8_t> checked_payload(const py::bytes& payload)
    {
        const std::string_view view = payload;
        if (view.size() > kMaxPayloadSize)
        {
            throw py::value_error("Link: payload of " + std::to_string(view.size()) + " bytes exceeds MAX_PAYLOAD_SIZE");
        }
        return std::vector<uint8_t>(view.begin(), view.end());
    }

    std::shared_ptr<State> state_;
    const long pid_ = process_id(); //!< of the process that made the link
};

/*!
 * @brief Python-facing listening socket. It holds a core::Listener, which _deviceio_session binds as a class of its
 *        own.
 *
 * listen(), accept() and close() release the GIL. The mutex keeps close() out of a running accept() slice, as
 * core::Listener requires; it is only ever waited for without the GIL.
 */
class PyLinkListener
{
public:
    static std::unique_ptr<PyLinkListener> listen(const std::string& host, int port)
    {
        const Endpoint endpoint{ host, checked_port(port) };
        GilRelease release; // the host name lookup can take seconds
        return std::unique_ptr<PyLinkListener>(new PyLinkListener(endpoint));
    }

    int port() const
    {
        return port_;
    }

    std::optional<std::pair<int, std::string>> accept(int64_t timeout_ms)
    {
        std::optional<Listener::Accepted> accepted;
        wait_checking_signals(clamped_ms(timeout_ms),
                              [&](std::chrono::milliseconds slice)
                              {
                                  // Before the lock, so a waiting close() gets it once this slice is over.
                                  if (closing_)
                                  {
                                      throw std::runtime_error("Listener is closed");
                                  }
                                  std::lock_guard<std::mutex> lock(mutex_);
                                  accepted = listener_.accept(slice);
                                  return accepted.has_value();
                              });
        if (!accepted)
        {
            return std::nullopt;
        }
        return std::make_pair(accepted->fd, accepted->peer);
    }

    void close()
    {
        closing_ = true;
        GilRelease release;
        std::lock_guard<std::mutex> lock(mutex_);
        listener_.close();
    }

    PyLinkListener& enter()
    {
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

private:
    explicit PyLinkListener(const Endpoint& endpoint) : listener_(endpoint)
    {
        port_ = listener_.port();
    }

    static uint16_t checked_port(int port)
    {
        if (port < 0 || port > 65535)
        {
            throw py::value_error("LinkListener: port must be in [0, 65535]");
        }
        return static_cast<uint16_t>(port);
    }

    Listener listener_;
    int port_ = 0;
    std::mutex mutex_;
    std::atomic<bool> closing_{ false };
};

} // namespace core
