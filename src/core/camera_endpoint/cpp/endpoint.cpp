// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_endpoint/endpoint.hpp"

#include <camera_ipc/message_socket.hpp>
#include <camera_ipc/unique_fd.hpp>
#include <camera_wire/messages.hpp>
#include <log_bridge/logger.hpp>
#include <sys/eventfd.h>

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

// One reader. As an EventSink it is called by the publisher, on whichever thread publishes or
// releases, with the publisher's lock held; everything else runs on the endpoint's thread.
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
    WireEncoder m_event_encoder; // used inside send_event only, which the publisher's lock serialises
};

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
          m_listener(path),
          m_wake(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)),
          m_buffer(MAX_WIRE_MESSAGE_BYTES),
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
        m_stopping = false;
        m_thread = std::thread([this] { run(); });
        m_logger->info("serving cameras on '{}'", m_listener.path());
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
            waits.assign({ { m_wake.get(), POLLIN, 0 }, { m_listener.descriptor(), POLLIN, 0 } });
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
            while (auto socket = m_listener.accept(std::chrono::milliseconds(0)))
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

    // Handles everything the reader has sent. Returns false when the connection is to be dropped.
    bool serve(Connection& connection)
    {
        try
        {
            for (;;)
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
                const WireMessage message = decode_wire_message({ m_buffer.data(), received.bytes });
                if (!handle(connection, message))
                {
                    return false;
                }
            }
        }
        catch (const std::exception& error)
        {
            m_logger->warn("dropping camera reader '{}': {}", connection.name, error.what());
            return false;
        }
    }

    bool handle(Connection& connection, const WireMessage& message)
    {
        if (const auto* hello = std::get_if<HelloMessage>(&message))
        {
            return greet(connection, *hello);
        }
        if (const auto* subscribe = std::get_if<SubscribeMessage>(&message))
        {
            if (!connection.greeted)
            {
                m_logger->warn("dropping a camera reader that subscribed before it said hello");
                return false;
            }
            return answer(connection, *subscribe);
        }
        if (const auto* release = std::get_if<ReleaseMessage>(&message))
        {
            if (connection.subscription != 0)
            {
                m_publisher.release(connection.subscription, release->lease_id);
            }
            return true;
        }
        if (std::holds_alternative<std::monostate>(message))
        {
            return true; // from a newer reader; not for this build
        }
        m_logger->warn("dropping camera reader '{}': it sent a message only a plugin sends", connection.name);
        return false;
    }

    bool greet(Connection& connection, const HelloMessage& hello)
    {
        connection.name = hello.reader_name.empty() ? "unnamed" : hello.reader_name;
        WelcomeMessage welcome;
        if (hello.protocol_version != WIRE_PROTOCOL_VERSION)
        {
            welcome.error = "this camera plugin speaks protocol version " + std::to_string(WIRE_PROTOCOL_VERSION) +
                            " and the reader version " + std::to_string(hello.protocol_version);
            m_logger->warn("refusing camera reader '{}': {}", connection.name, welcome.error);
            (void)send(connection, m_encoder.encode(welcome));
            return false;
        }
        const std::size_t group_count = m_publisher.group_count();
        for (std::size_t i = 0; i < group_count; ++i)
        {
            welcome.groups.push_back(m_publisher.group_description(static_cast<GroupId>(i)));
        }
        connection.greeted = true;
        m_logger->info("camera reader '{}' connected", connection.name);
        return send(connection, m_encoder.encode(welcome));
    }

    bool answer(Connection& connection, const SubscribeMessage& subscribe)
    {
        SubscribedMessage reply;
        std::span<const int> descriptors;
        const std::optional<GroupId> group = m_publisher.find_group(subscribe.group);
        if (connection.subscription != 0)
        {
            reply.error = "a connection carries one subscription; open another connection for a second group";
        }
        else if (!group)
        {
            reply.error = "the camera plugin publishes no group '" + subscribe.group + "'";
        }
        else if (!m_memory.contains(*group))
        {
            reply.error = "camera group '" + subscribe.group + "' has no memory a reader can map";
        }
        else
        {
            // From here on the publisher may send this reader frame events at any moment, so a
            // Frame can reach it before the Subscribed below.
            const SubscribeResult result = m_publisher.subscribe(connection, *group, subscribe.options);
            if (result)
            {
                connection.subscription = result.id;
                reply.group = m_publisher.group_description(*group);
                descriptors = m_memory.at(*group);
            }
            else
            {
                reply.error = result.error;
            }
        }

        if (reply.error.empty())
        {
            m_logger->info("camera reader '{}' subscribed to '{}' (window {}, {})", connection.name, subscribe.group,
                           subscribe.options.window, name_of(subscribe.options.order));
        }
        else
        {
            m_logger->warn("refusing camera reader '{}': {}", connection.name, reply.error);
        }
        return send(connection, m_encoder.encode(reply), descriptors);
    }

    static bool send(Connection& connection, std::span<const std::byte> bytes, std::span<const int> descriptors = {})
    {
        return connection.socket.send(bytes, descriptors) == SendStatus::SENT;
    }

    Publisher& m_publisher;
    MessageListener m_listener;
    UniqueFd m_wake;
    std::map<GroupId, std::vector<int>> m_memory;

    std::thread m_thread;
    std::atomic<bool> m_stopping{ false };
    std::atomic<std::size_t> m_reader_count{ 0 };

    // The endpoint's thread only.
    std::vector<std::unique_ptr<Connection>> m_connections;
    std::vector<std::byte> m_buffer;
    WireEncoder m_encoder;

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
