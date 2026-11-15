// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "py_link.hpp"

#include <pybind11/stl.h>
#include <pybind11/typing.h>
#include <remote_link/message_header.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/transport.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>

namespace py = pybind11;

namespace
{

//! `value` when it is at least `least`; ValueError otherwise.
int64_t at_least(int64_t value, int64_t least, const char* name)
{
    if (value < least)
    {
        throw py::value_error(std::string("Link: ") + name + " must be at least " + std::to_string(least) + ", got " +
                              std::to_string(value));
    }
    return value;
}

//! Binds a static factory of Link that takes `Lead...` and then the keyword options connect() and adopt() share.
//! The options are checked before `open` runs, so a ValueError for one leaves adopt()'s descriptor with the caller.
template <typename... Lead, typename Open, typename... LeadArgs>
void def_link_factory(
    py::class_<core::PyLink>& link, const char* name, Open open, const char* doc, const LeadArgs&... lead_args)
{
    const core::PyLinkOptions defaults;
    link.def_static(
        name,
        [open](Lead... lead, std::string link_name, int64_t heartbeat_period_ms, int64_t peer_timeout_ms,
               int64_t close_drain_ms, int64_t max_queued_reliable_bytes, bool send_pings,
               bool peer_timeout_from_first_message, int64_t max_received_messages, int64_t max_received_bytes,
               const py::typing::Iterable<py::int_>& latest_types)
        {
            core::PyLinkOptions options;
            options.link.name = std::move(link_name);
            options.link.heartbeat_period = core::clamped_ms(at_least(heartbeat_period_ms, 1, "heartbeat_period_ms"));
            options.link.peer_timeout = core::clamped_ms(at_least(peer_timeout_ms, 1, "peer_timeout_ms"));
            options.link.close_drain = core::clamped_ms(at_least(close_drain_ms, 0, "close_drain_ms"));
            options.link.max_queued_reliable_bytes =
                static_cast<size_t>(at_least(max_queued_reliable_bytes, 1, "max_queued_reliable_bytes"));
            options.link.send_pings = send_pings;
            options.link.peer_timeout_from_first_message = peer_timeout_from_first_message;
            options.max_received_messages =
                static_cast<size_t>(at_least(max_received_messages, 1, "max_received_messages"));
            options.max_received_bytes = static_cast<size_t>(at_least(max_received_bytes, 1, "max_received_bytes"));
            for (const py::handle item : latest_types)
            {
                if (!PyLong_Check(item.ptr()))
                {
                    throw py::type_error("Link: latest_types holds type numbers (int)");
                }
                int overflow = 0;
                const long long type = PyLong_AsLongLongAndOverflow(item.ptr(), &overflow);
                if (overflow != 0 || type < 0 || type > 255)
                {
                    throw py::value_error("Link: latest_types must be in [0, 255], got " +
                                          py::str(item).cast<std::string>());
                }
                options.latest_types[static_cast<size_t>(type)] = true;
            }
            return open(std::forward<Lead>(lead)..., options);
        },
        lead_args..., py::arg("name") = defaults.link.name,
        py::arg("heartbeat_period_ms") = defaults.link.heartbeat_period.count(),
        py::arg("peer_timeout_ms") = defaults.link.peer_timeout.count(),
        py::arg("close_drain_ms") = defaults.link.close_drain.count(),
        py::arg("max_queued_reliable_bytes") = defaults.link.max_queued_reliable_bytes,
        py::arg("send_pings") = defaults.link.send_pings,
        py::arg("peer_timeout_from_first_message") = defaults.link.peer_timeout_from_first_message,
        py::arg("max_received_messages") = defaults.max_received_messages,
        py::arg("max_received_bytes") = defaults.max_received_bytes, py::arg("latest_types") = py::tuple(), doc);
}

} // namespace

PYBIND11_MODULE(_remote_link, m)
{
    m.doc() = "Isaac Capture remote link - link messages between two processes over TCP or WebSocket";

    m.attr("MAX_PAYLOAD_SIZE") = py::int_(core::kMaxPayloadSize);

    py::register_local_exception_translator(
        [](std::exception_ptr p)
        {
            try
            {
                if (p)
                {
                    std::rethrow_exception(p);
                }
            }
            catch (const core::TcpConnectTimeoutError& e)
            {
                py::set_error(PyExc_TimeoutError, e.what());
            }
        });
    auto& peer_closed = py::register_local_exception<core::PeerClosedError>(m, "LinkPeerClosedError", PyExc_RuntimeError);
    peer_closed.attr("__doc__") =
        "RuntimeError raised by Link.connect() over ws:// when the peer closed or reset the connection before any "
        "byte of its answer to the WebSocket upgrade. Connecting again may succeed.";

    py::class_<core::PyReceivedLinkMessage>(m, "ReceivedLinkMessage", "One received link message. Read-only.")
        .def_readonly("type", &core::PyReceivedLinkMessage::type, "The type number, 0 to 255.")
        .def_readonly("payload", &core::PyReceivedLinkMessage::payload, "The payload, as bytes.")
        .def_readonly("remote_send_ns", &core::PyReceivedLinkMessage::remote_send_ns,
                      "When the peer sent it, on the peer's monotonic clock.")
        .def_readonly("local_receive_ns", &core::PyReceivedLinkMessage::local_receive_ns,
                      "When it arrived, on this machine's CLOCK_MONOTONIC.");

    py::class_<core::PyClockEstimate>(m, "ClockEstimate", "A link's estimate of the peer's clock. Read-only.")
        .def_readonly("offset_ns", &core::PyClockEstimate::offset_ns, "This machine's monotonic clock minus the peer's.")
        .def_readonly(
            "two_way", &core::PyClockEstimate::two_way, "Whether PING/PONG round trips made the estimate (send_pings).")
        .def_readonly("valid", &core::PyClockEstimate::valid, "Whether any received link message gave a sample yet.")
        .def_readonly("rtt_min_ns", &core::PyClockEstimate::rtt_min_ns, "The shortest round trip, -1 until two_way.");

    py::class_<core::PyLinkStats>(m, "LinkStats", "Counters of a link. Read-only.")
        .def_readonly("messages_sent", &core::PyLinkStats::messages_sent, "Including HEARTBEAT, PING and PONG.")
        .def_readonly("messages_received", &core::PyLinkStats::messages_received, "Including HEARTBEAT, PING and PONG.")
        .def_readonly("latest_replaced", &core::PyLinkStats::latest_replaced,
                      "send_latest() payloads replaced by a newer one before they were sent.")
        .def_readonly("unknown_received", &core::PyLinkStats::unknown_received,
                      "Received link messages of a type the teleop session protocol does not name; an "
                      "application's own types count here.")
        .def_readonly("received_queued", &core::PyLinkStats::received_queued,
                      "Received link messages that receive() has not taken yet.")
        .def_readonly("received_replaced", &core::PyLinkStats::received_replaced,
                      "Unread link messages of a latest type replaced by a newer one.");

    // Only the blocking calls release the GIL; the link's threads never take it. The waits run Python's signal
    // handlers every 100 ms, and connect() after each TCP connect attempt, so Ctrl-C interrupts them.
    // The class docstring is long because it is the one place that describes the keyword options connect() and
    // adopt() share.
    py::class_<core::PyLink> link(
        m, "Link",
        "One connection to a peer: reliable and newest-wins link messages, heartbeats, liveness and a clock "
        "estimate. connect() dials, adopt() takes an accepted descriptor; neither exchanges a link message. "
        "Received link messages wait in a bounded inbox for receive(). A dead link stays dead, and every method "
        "still works on it. A forked child process must leave a Link of its parent alone: close() and dropping "
        "it do nothing there.\n\n"
        "Keyword options of connect() and adopt(): name prefixes the link's log lines. A HEARTBEAT goes out after "
        "heartbeat_period_ms without a send, and the link dies after peer_timeout_ms without a receive (with "
        "peer_timeout_from_first_message, counted from the first received link message). close() lets queued "
        "reliable link messages go out for close_drain_ms. The link dies when more than max_queued_reliable_bytes "
        "are unsent, or more than max_received_messages or max_received_bytes are unread; no value of these "
        "means no limit. send_pings adds PING/PONG for a two-way clock estimate. Of each type in latest_types "
        "the inbox keeps only the newest unread link message.");
    def_link_factory<const std::string&, int64_t>(
        link, "connect", &core::PyLink::connect,
        "Connect to endpoint ('host:port', 'tcp://host:port' or 'ws://host:port[/path]'; the port is required) "
        "and start the link. timeout_ms (negative: no limit) bounds the TCP connect and, over ws://, the "
        "WebSocket upgrade, not the host name lookup. Raises ValueError for a bad endpoint or option, "
        "TimeoutError when the time ran out, RuntimeError when the connect cannot be made (its text starts with "
        "'cannot connect to ' when nothing accepted the connection; LinkPeerClosedError over ws://). The TCP "
        "connect runs in attempts of at most 2 s, after each of which Ctrl-C raises KeyboardInterrupt. Releases "
        "the GIL.",
        py::arg("endpoint"), py::arg("timeout_ms"), py::kw_only());
    def_link_factory<int, bool>(
        link, "adopt", &core::PyLink::adopt,
        "Take ownership of a connected socket descriptor, put it in blocking mode and start the link; with "
        "websocket the link first answers the peer's WebSocket upgrade. fd must be the caller's alone to give "
        "(socket.detach()). Raises ValueError for a bad option and for an fd that is not an open socket or "
        "already belongs to a Link. After ValueError or TypeError the caller still owns fd; after any other "
        "exception the link has closed it.",
        py::arg("fd"), py::kw_only(), py::arg("websocket") = false);
    link.def("send_reliable", &core::PyLink::send_reliable, py::arg("type"), py::arg("payload"),
             "Queue a link message that arrives in order and is never dropped while the link lives. Does nothing "
             "on a dead link. Raises ValueError for a type outside [0, 255], for types 11, 13 and 14 (the link's "
             "own HEARTBEAT, PING and PONG) and for a payload above MAX_PAYLOAD_SIZE.")
        .def("send_latest", &core::PyLink::send_latest, py::arg("type"), py::arg("payload"),
             "Queue a newest-wins link message: it replaces an unsent one of its type, and never passes a reliable "
             "link message queued before it. Does nothing on a dead link. Raises ValueError as send_reliable().")
        .def("receive", &core::PyLink::receive, py::arg("timeout_ms"),
             "The oldest unread link message, or None after timeout_ms (0: do not wait; negative: no limit). A "
             "dead link still hands out what arrived before; with nothing left it waits out a positive timeout_ms, "
             "and under no limit returns None within about 100 ms. Ctrl-C raises KeyboardInterrupt within about "
             "100 ms, and a link message that came with it stays in the inbox. Releases the GIL.")
        .def_property_readonly("alive", &core::PyLink::alive, "False once the link died or close() ran.")
        .def_property_readonly("close_reason", &core::PyLink::close_reason, "Why the link went down ('' while alive).")
        .def_property_readonly("peer_closed_before_any_message", &core::PyLink::peer_closed_before_any_message,
                               "True once the link died because the connection broke before any link message "
                               "or part of one arrived.")
        .def("clock", &core::PyLink::clock, "The current ClockEstimate.")
        .def("stats", &core::PyLink::stats, "The current LinkStats.")
        .def("close", &core::PyLink::close, py::arg("linger_ms") = 0,
             "Send what is queued as reliable for up to close_drain_ms, then close the connection; what is still "
             "unsent then is dropped. With linger_ms > 0 and everything sent within close_drain_ms (not certain "
             "with close_drain_ms=0), the link then waits up to linger_ms for the peer to close its end, so the "
             "peer reads everything sent. Ctrl-C does not interrupt close(). Idempotent. The socket descriptor is "
             "released when the Link object goes. Releases the GIL.")
        .def("__enter__", &core::PyLink::enter)
        .def("__exit__", &core::PyLink::exit);

    py::class_<core::PyLinkListener>(
        m, "LinkListener", "Listening TCP socket whose accepted descriptors go to Link.adopt().")
        .def(py::init(&core::PyLinkListener::listen), py::arg("host"), py::arg("port"),
             "Bind and listen; port 0 picks a free port. Releases the GIL.")
        .def_property_readonly("port", &core::PyLinkListener::port, "The bound port, also after close().")
        .def("accept", &core::PyLinkListener::accept, py::arg("timeout_ms"),
             "(fd, peer) of the next connection, or None after timeout_ms (negative: no limit). peer is its "
             "numeric 'host:port', an IPv6 host in brackets; a listener bound to '::' reports an IPv4 peer as "
             "'[::ffff:a.b.c.d]:port'. The caller owns fd. Raises RuntimeError once closed. Ctrl-C raises "
             "KeyboardInterrupt within about 100 ms. Releases the GIL.")
        .def("close", &core::PyLinkListener::close,
             "Stop listening: connections not accepted yet are reset and new ones refused. Idempotent, from any "
             "thread; an accept() on another thread ends within about 100 ms. Releases the GIL.")
        .def("__enter__", &core::PyLinkListener::enter)
        .def("__exit__", &core::PyLinkListener::exit);
}
