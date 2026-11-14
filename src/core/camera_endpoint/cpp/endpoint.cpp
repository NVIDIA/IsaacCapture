// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_endpoint/endpoint.hpp"

#include <camera_ipc/message_socket.hpp>
#include <camera_ipc/unique_fd.hpp>
#include <camera_protocol/messages.hpp>
#include <log_bridge/logger.hpp>
#include <sys/eventfd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <poll.h>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace core::camera
{

namespace
{

// One reader. As an EventSink it is called by the publisher with its group's lock held: on the
// capture thread when a frame set is published, on the endpoint's thread when the reader
// released or asked for one. Everything else runs on the endpoint's thread.
class Connection final : public EventSink
{
public:
    explicit Connection(MessageSocket connected) : socket(std::move(connected))
    {
    }

    bool send_event(SubscriptionId /*subscription*/, const FrameEvent& event) noexcept override
    {
        try
        {
            return socket.send(m_event_encoder.encode(event)) == SendStatus::SENT;
        }
        catch (const std::exception&)
        {
            return false; // the reader gets a later frame set, or its closed socket is noticed
        }
    }

    MessageSocket socket;
    std::string name = "unnamed";
    bool greeted = false;
    SubscriptionId subscription = 0;

private:
    MessageEncoder m_event_encoder; // used inside send_event only, which the group's lock serialises
};

// How many messages one reader is served before the others get their turn. A reader that sends
// without pause then slows nobody but itself, and cannot keep the endpoint from stopping: its
// socket stays readable, so the rest is handled in the next round.
constexpr int MESSAGES_PER_TURN = 32;

const char* name_of(DeliveryOrder order)
{
    switch (order)
    {
    case DeliveryOrder::NEWEST:
        return "newest";
    case DeliveryOrder::IN_ORDER:
        return "in order";
    default:
        return "default order";
    }
}

} // namespace

class Endpoint::Impl
{
public:
    Impl(Publisher& publisher, const std::string& path)
        : m_publisher(publisher),
          m_path(path),
          m_listener(std::in_place, path),
          m_wake(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)),
          m_buffer(MAX_MESSAGE_BYTES),
          m_logger(isaaccapture::Logger::get("isaaccapture.camera_endpoint.Endpoint"))
    {
        if (!m_wake.valid())
        {
            throw std::system_error(errno, std::generic_category(), "creating the camera endpoint's wake-up descriptor");
        }
    }

    ~Impl()
    {
        stop();
    }

    void set_group_memory(GroupId group, std::vector<int> descriptors)
    {
        if (m_thread.joinable())
        {
            throw std::logic_error("a camera group's memory is named before the endpoint starts");
        }
        const GroupDescription description = m_publisher.group_description(group);
        if (descriptors.size() != description.streams.size())
        {
            throw std::invalid_argument("camera group '" + description.id + "' needs one descriptor per stream");
        }
        m_memory[group] = std::move(descriptors);
    }

    void start()
    {
        if (m_thread.joinable())
        {
            return;
        }
        if (!m_listener)
        {
            m_listener.emplace(m_path);
        }
        m_stopping = false;
        m_thread = std::thread([this] { run(); });
        m_logger->info("serving cameras on '{}'", m_path);
    }

    void stop()
    {
        if (!m_thread.joinable())
        {
            return;
        }
        m_stopping = true;
        const std::uint64_t one = 1;
        // Cannot fail in a way that matters: the counter only overflows after 2^64 wake-ups.
        [[maybe_unused]] const ssize_t written = ::write(m_wake.get(), &one, sizeof(one));
        m_thread.join();
        // Without the listener a reader that tries to connect is refused at once, instead of
        // waiting in a queue nobody serves.
        m_listener.reset();
    }

    std::size_t reader_count() const
    {
        return m_reader_count.load();
    }

private:
    void run()
    {
        std::vector<pollfd> waits;
        while (!m_stopping)
        {
            waits.assign({ { m_wake.get(), POLLIN, 0 }, { m_listener->descriptor(), POLLIN, 0 } });
            for (const auto& connection : m_connections)
            {
                waits.push_back({ connection->socket.descriptor(), POLLIN, 0 });
            }
            if (::poll(waits.data(), waits.size(), -1) < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                m_logger->error("camera endpoint stops serving: poll failed with errno {}", errno);
                break;
            }

            // Back to front, so dropping a connection leaves the indices still to visit alone.
            for (std::size_t i = m_connections.size(); i-- > 0;)
            {
                if (waits[i + 2].revents != 0 && !serve(*m_connections[i]))
                {
                    drop(i);
                }
            }
            if ((waits[1].revents & POLLIN) != 0)
            {
                accept_readers();
            }
        }
        while (!m_connections.empty())
        {
            drop(m_connections.size() - 1);
        }
        // Empty the wake-up counter, so that serving can start again.
        std::uint64_t wakes = 0;
        [[maybe_unused]] const ssize_t drained = ::read(m_wake.get(), &wakes, sizeof(wakes));
    }

    void accept_readers()
    {
        try
        {
            while (auto socket = m_listener->accept(std::chrono::milliseconds(0)))
            {
                m_connections.push_back(std::make_unique<Connection>(std::move(*socket)));
                m_reader_count = m_connections.size();
            }
        }
        catch (const std::exception& error)
        {
            m_logger->warn("could not accept a camera reader: {}", error.what());
        }
    }

    // Ends the reader's subscription before its sink goes away. unsubscribe() returns only once
    // no send_event is running or can start for it.
    void drop(std::size_t index)
    {
        Connection& connection = *m_connections[index];
        if (connection.subscription != 0)
        {
            m_publisher.unsubscribe(connection.subscription);
        }
        m_logger->info("camera reader '{}' is gone", connection.name);
        m_connections.erase(m_connections.begin() + static_cast<std::ptrdiff_t>(index));
        m_reader_count = m_connections.size();
    }

    // Handles what the reader has sent, up to one turn's worth. Returns false when the
    // connection is to be dropped.
    bool serve(Connection& connection)
    {
        try
        {
            for (int handled = 0; handled < MESSAGES_PER_TURN; ++handled)
            {
                const ReceivedMessage received = connection.socket.receive(m_buffer, std::chrono::milliseconds(0));
                if (received.status == ReceiveStatus::TIMEOUT)
                {
                    return true;
                }
                if (received.status == ReceiveStatus::CLOSED)
                {
                    return false;
                }
                const ProtocolMessage message = decode_message({ m_buffer.data(), received.bytes });
                if (!handle(connection, message))
                {
                    return false;
                }
            }
            return true;
        }
        catch (const std::exception& error)
        {
            m_logger->warn("dropping camera reader '{}': {}", connection.name, error.what());
            return false;
        }
    }

    bool handle(Connection& connection, const ProtocolMessage& message)
    {
        if (const auto* request = std::get_if<HandshakeRequest>(&message))
        {
            return shake_hands(connection, *request);
        }
        if (const auto* release = std::get_if<ReleaseMessage>(&message))
        {
            if (connection.subscription != 0)
            {
                m_publisher.release(connection.subscription, release->lease_id);
            }
            return true;
        }
        if (const auto* acquire = std::get_if<AcquireMessage>(&message))
        {
            if (connection.subscription == 0 ||
                !m_publisher.request_newest(connection.subscription, acquire->after_sequence))
            {
                m_logger->warn(
                    "dropping camera reader '{}': it asked for the newest frame set without a "
                    "newest-order subscription",
                    connection.name);
                return false;
            }
            return true;
        }
        if (std::holds_alternative<std::monostate>(message))
        {
            return true; // something a newer reader sends for information; not for this build
        }
        m_logger->warn("dropping camera reader '{}': it sent a message only a plugin sends", connection.name);
        return false;
    }

    // Answers a reader's first message: what is published, or the group it asked for with the
    // descriptors of its pools. A reader that is refused is told why and then dropped.
    //
    // A subscription is reserved, then the reader is told, then delivery starts: the answer is
    // the first thing the reader receives, and capture never waits for it to be sent. A
    // subscription whose answer could not be sent ends in drop(), like any other.
    bool shake_hands(Connection& connection, const HandshakeRequest& request)
    {
        if (connection.greeted)
        {
            m_logger->warn("dropping camera reader '{}': it sent a second handshake", connection.name);
            return false;
        }
        connection.greeted = true;
        connection.name = request.reader_name.empty() ? "unnamed" : request.reader_name;

        HandshakeResponse response;
        // The connection uses the newest version both sides speak. A reader newer than this
        // plugin is answered in the plugin's version and decides whether it still speaks it.
        response.protocol_version = std::min(request.protocol_version, PROTOCOL_VERSION);
        std::span<const int> descriptors;
        const std::optional<GroupId> group = m_publisher.find_group(request.group);
        if (request.protocol_version < OLDEST_PROTOCOL_VERSION)
        {
            response.protocol_version = OLDEST_PROTOCOL_VERSION;
            response.error = "this camera plugin speaks protocol versions " + std::to_string(OLDEST_PROTOCOL_VERSION) +
                             " to " + std::to_string(PROTOCOL_VERSION) + " and the reader version " +
                             std::to_string(request.protocol_version);
        }
        else if (!request.unsupported.empty())
        {
            response.error = request.unsupported;
        }
        else if (request.group.empty())
        {
            const std::size_t group_count = m_publisher.group_count();
            for (std::size_t i = 0; i < group_count; ++i)
            {
                response.groups.push_back({ m_publisher.group_description(static_cast<GroupId>(i)), {} });
            }
            m_logger->info("camera reader '{}' asked what is published", connection.name);
        }
        else if (!group)
        {
            response.error = "the camera plugin publishes no group '" + request.group + "'";
        }
        else if (!m_memory.contains(*group))
        {
            response.error = "camera group '" + request.group + "' has no memory a reader can map";
        }
        else
        {
            const SubscribeResult result = m_publisher.subscribe(connection, *group, request.options);
            if (result)
            {
                connection.subscription = result.id;
                response.groups.push_back({ m_publisher.group_description(*group), {} });
                descriptors = m_memory.at(*group);
            }
            else
            {
                response.error = result.error;
            }
        }

        if (!response.error.empty())
        {
            m_logger->warn("refusing camera reader '{}': {}", connection.name, response.error);
        }
        if (!send(connection, m_encoder.encode(response), descriptors) || !response.error.empty())
        {
            return false;
        }
        if (connection.subscription != 0)
        {
            m_publisher.activate(connection.subscription);
            m_logger->info("camera reader '{}' reads '{}' (window {}, {})", connection.name, request.group,
                           request.options.window, name_of(resolved_order(request.options)));
        }
        return true;
    }

    static bool send(Connection& connection, std::span<const std::byte> bytes, std::span<const int> descriptors = {})
    {
        return connection.socket.send(bytes, descriptors) == SendStatus::SENT;
    }

    Publisher& m_publisher;
    std::string m_path;
    std::optional<MessageListener> m_listener; // absent between stop() and the next start()
    UniqueFd m_wake;
    std::map<GroupId, std::vector<int>> m_memory;

    std::thread m_thread;
    std::atomic<bool> m_stopping{ false };
    std::atomic<std::size_t> m_reader_count{ 0 };

    // The endpoint's thread only.
    std::vector<std::unique_ptr<Connection>> m_connections;
    std::vector<std::byte> m_buffer;
    MessageEncoder m_encoder;

    std::shared_ptr<spdlog::logger> m_logger;
};

Endpoint::Endpoint(Publisher& publisher, const std::string& path) : m_impl(std::make_unique<Impl>(publisher, path))
{
}

Endpoint::~Endpoint() = default;

void Endpoint::set_group_memory(GroupId group, std::vector<int> descriptors)
{
    m_impl->set_group_memory(group, std::move(descriptors));
}

void Endpoint::start()
{
    m_impl->start();
}

void Endpoint::stop()
{
    m_impl->stop();
}

std::size_t Endpoint::reader_count() const
{
    return m_impl->reader_count();
}

} // namespace core::camera
