// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_reader/camera_reader.hpp"

#include "plugin_launcher.hpp"

#include <camera_ipc/endpoint_path.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <camera_wire/messages.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <poll.h>
#include <system_error>
#include <utility>
#include <variant>

namespace core::camera
{

namespace
{

// How long the plugin gets to answer the handshake and the subscription.
constexpr std::chrono::milliseconds ANSWER_TIMEOUT(2000);

struct Received
{
    WireMessage message;
    std::vector<UniqueFd> descriptors;
};

// Waits for one message. Nothing on timeout.
std::optional<Received> receive(MessageSocket& socket, std::span<std::byte> buffer, std::chrono::milliseconds timeout)
{
    ReceivedMessage received = socket.receive(buffer, timeout);
    if (received.status == ReceiveStatus::TIMEOUT)
    {
        return std::nullopt;
    }
    if (received.status == ReceiveStatus::CLOSED)
    {
        throw CameraDisconnected("the camera plugin closed the connection");
    }
    try
    {
        return Received{ decode_wire_message(buffer.first(received.bytes)), std::move(received.descriptors) };
    }
    catch (const WireError& error)
    {
        throw CameraReaderError(std::string("the camera plugin sent something that is not a message: ") + error.what());
    }
}

// How long a plugin that this reader started gets until it answers.
constexpr std::chrono::seconds START_TIMEOUT(20);

// Says hello and returns what the plugin publishes.
std::vector<GroupDescription> greet(MessageSocket& socket, WireEncoder& encoder, std::span<std::byte> buffer)
{
    HelloMessage hello;
    hello.reader_name = program_invocation_short_name;
    if (socket.send(encoder.encode(hello)) != SendStatus::SENT)
    {
        throw CameraDisconnected("the camera plugin closed the connection");
    }
    const auto deadline = std::chrono::steady_clock::now() + ANSWER_TIMEOUT;
    for (;;)
    {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        std::optional<Received> received = receive(socket, buffer, left);
        if (!received)
        {
            throw CameraReaderError("the camera plugin did not answer the reader's hello");
        }
        if (auto* welcome = std::get_if<WelcomeMessage>(&received->message))
        {
            if (!welcome->error.empty())
            {
                throw CameraReaderError("the camera plugin refused this reader: " + welcome->error);
            }
            return std::move(welcome->groups);
        }
    }
}

// A connection that got through the handshake.
struct Greeted
{
    MessageSocket socket;
    std::vector<GroupDescription> groups;
};

// Whether the plugin behind `alive` is gone, waiting up to `wait` to find out.
bool plugin_is_gone(const UniqueFd& alive, std::chrono::milliseconds wait)
{
    pollfd entry = { alive.get(), POLLIN, 0 };
    return ::poll(&entry, 1, static_cast<int>(wait.count())) > 0;
}

// Starts a plugin for `endpoint` and connects to it once it serves.
Greeted start_and_connect(const PluginLocation& plugin,
                          const std::string& endpoint,
                          WireEncoder& encoder,
                          std::span<std::byte> buffer)
{
    const std::string nobody = "no camera plugin answers on '" + endpoint + "'";
    const std::string executable = find_plugin_executable(plugin.executable);
    if (executable.empty())
    {
        throw CameraReaderError(
            nobody + ", and there is no camera_plugin to start" +
            (plugin.executable.empty() ? " next to this program or on PATH" : " at '" + plugin.executable + "'"));
    }
    std::vector<std::string> arguments = { "--exit-when-idle=" + std::to_string(plugin.exit_when_idle_seconds) };
    if (!plugin.instance.empty())
    {
        arguments.push_back("--instance=" + plugin.instance);
    }
    UniqueFd alive;
    try
    {
        alive = launch_plugin(executable, arguments);
    }
    catch (const std::system_error& error)
    {
        throw CameraReaderError(nobody + ", and '" + executable + "' could not be started: " + error.code().message());
    }

    // Another reader may have started a plugin at the same moment. Only one of the two plugins
    // gets the endpoint, so this one being gone does not mean nobody serves: keep trying.
    const auto deadline = std::chrono::steady_clock::now() + START_TIMEOUT;
    std::string last_failure = "it did not start serving";
    bool gone = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        try
        {
            MessageSocket socket = MessageSocket::connect(endpoint);
            std::vector<GroupDescription> groups = greet(socket, encoder, buffer);
            return { std::move(socket), std::move(groups) };
        }
        catch (const CameraReaderError& error)
        {
            last_failure = error.what();
        }
        catch (const std::system_error&)
        {
            if (gone)
            {
                break; // the plugin ended and still nobody listens
            }
        }
        gone = gone || plugin_is_gone(alive, std::chrono::milliseconds(20));
        if (gone)
        {
            last_failure = "it stopped before it served; its log says why";
        }
    }
    throw CameraReaderError(nobody + ". '" + executable + "' was started for it, but " + last_failure);
}

// Connects to the plugin, starting one first when that is wanted and needed, and says hello.
Greeted connect_to(const PluginLocation& plugin, WireEncoder& encoder, std::span<std::byte> buffer)
{
    if (!plugin.endpoint.empty() && !plugin.instance.empty())
    {
        throw CameraReaderError("a camera plugin is named by its endpoint or by its instance, not both");
    }
    std::string endpoint = plugin.endpoint;
    try
    {
        if (endpoint.empty())
        {
            endpoint = default_endpoint_path(plugin.instance);
        }
        MessageSocket socket = MessageSocket::connect(endpoint);
        std::vector<GroupDescription> groups = greet(socket, encoder, buffer);
        return { std::move(socket), std::move(groups) };
    }
    catch (const std::system_error& error)
    {
        if (plugin.endpoint.empty() && plugin.start_if_absent)
        {
            return start_and_connect(plugin, endpoint, encoder, buffer);
        }
        throw CameraReaderError("no camera plugin answers on '" + endpoint + "': " + error.code().message());
    }
    catch (const std::invalid_argument& error)
    {
        throw CameraReaderError(error.what());
    }
}

} // namespace

namespace detail
{

// Shared by a reader and the frame sets it handed out, so a frame set can be given back after
// its reader is gone.
class ReaderState
{
public:
    explicit ReaderState(MessageSocket connected) : socket(std::move(connected)), buffer(MAX_WIRE_MESSAGE_BYTES)
    {
    }

    // Called from a FrameSet, possibly on another thread than the one that reads.
    void release(std::uint64_t lease_id) noexcept
    {
        try
        {
            const std::lock_guard<std::mutex> lock(m_release_mutex);
            (void)socket.send(m_release_encoder.encode(ReleaseMessage{ lease_id }));
        }
        catch (const std::exception&)
        {
            // The plugin is gone or will drop this reader; either way it takes the frame set back.
        }
    }

    // The next frame event, from the ones that arrived early or from the socket.
    std::optional<FrameEvent> take(std::chrono::milliseconds timeout)
    {
        if (!early.empty())
        {
            const FrameEvent event = early.front();
            early.pop_front();
            return event;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;)
        {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            const std::optional<Received> received = receive(socket, buffer, left);
            if (!received)
            {
                return std::nullopt;
            }
            if (const auto* event = std::get_if<FrameEvent>(&received->message))
            {
                return *event;
            }
            // Anything else is not for a subscribed reader of this build; keep waiting.
        }
    }

    // A frame set that names a slot or a size the pools do not have cannot be read safely.
    void check(const FrameEvent& event) const
    {
        bool fits = event.image_count == group.streams.size();
        for (std::size_t i = 0; fits && i < event.image_count; ++i)
        {
            fits = event.images[i].slot < group.streams[i].slot_count &&
                   event.images[i].bytes_used <= group.streams[i].slot_bytes;
        }
        if (!fits)
        {
            throw CameraReaderError("the camera plugin sent a frame set that does not match group '" + group.id + "'");
        }
    }

    MessageSocket socket;
    GroupDescription group;
    ReaderOptions options;
    std::vector<HostPoolView> pools; // one per stream
    std::deque<FrameEvent> early; // frame events that arrived before the answer to the subscription
    std::vector<std::byte> buffer; // the reading thread only

private:
    std::mutex m_release_mutex;
    WireEncoder m_release_encoder;
};

} // namespace detail

std::span<const std::byte> ImageView::plane(std::uint32_t index) const
{
    if (stream == nullptr || index >= stream->plane_count)
    {
        throw std::out_of_range("the image has no plane " + std::to_string(index));
    }
    const PlaneSize size = plane_size_of(stream->pixel_format, stream->width, stream->height, index);
    const std::uint64_t offset = stream->planes[index].offset_bytes;
    const std::uint64_t bytes = stream->planes[index].stride_bytes * (size.rows - 1) + size.row_bytes;
    if (size.rows == 0 || offset + bytes > data.size())
    {
        throw std::out_of_range("plane " + std::to_string(index) + " does not lie inside the image's slot");
    }
    return data.subspan(offset, bytes);
}

FrameSet::FrameSet(std::shared_ptr<detail::ReaderState> state, const FrameEvent& event)
    : m_state(std::move(state)), m_event(event)
{
}

FrameSet::FrameSet(FrameSet&& other) noexcept : m_state(std::move(other.m_state)), m_event(other.m_event)
{
}

FrameSet& FrameSet::operator=(FrameSet&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_state = std::move(other.m_state);
        m_event = other.m_event;
    }
    return *this;
}

FrameSet::~FrameSet()
{
    release();
}

ImageView FrameSet::image(std::size_t stream) const
{
    if (!m_state)
    {
        throw std::logic_error("the frame set was given back");
    }
    if (stream >= m_event.image_count)
    {
        throw std::out_of_range("the frame set has no image " + std::to_string(stream));
    }
    const StreamDescription& description = m_state->group.streams[stream];
    const ImageInfo& info = m_event.images[stream];
    const std::span<const std::byte> slot = m_state->pools[stream].slot(info.slot);
    return { &description, info, description.is_encoded() ? slot.first(info.bytes_used) : slot };
}

void FrameSet::release() noexcept
{
    if (m_state)
    {
        m_state->release(m_event.lease_id);
        m_state.reset();
    }
}

std::vector<GroupDescription> CameraReader::list(const PluginLocation& plugin)
{
    WireEncoder encoder;
    std::vector<std::byte> buffer(MAX_WIRE_MESSAGE_BYTES);
    return connect_to(plugin, encoder, buffer).groups;
}

CameraReader CameraReader::open(const std::string& group, const ReaderOptions& options, const PluginLocation& plugin)
{
    WireEncoder encoder;
    std::vector<std::byte> handshake_buffer(MAX_WIRE_MESSAGE_BYTES);
    auto state = std::make_shared<detail::ReaderState>(connect_to(plugin, encoder, handshake_buffer).socket);

    if (state->socket.send(encoder.encode(SubscribeMessage{ group, options })) != SendStatus::SENT)
    {
        throw CameraDisconnected("the camera plugin closed the connection");
    }
    const auto deadline = std::chrono::steady_clock::now() + ANSWER_TIMEOUT;
    for (;;)
    {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        std::optional<Received> received = receive(state->socket, state->buffer, left);
        if (!received)
        {
            throw CameraReaderError("the camera plugin did not answer the subscription to '" + group + "'");
        }
        // The plugin starts sending frame sets as soon as the subscription exists, which can be
        // before its answer is on the way.
        if (const auto* event = std::get_if<FrameEvent>(&received->message))
        {
            state->early.push_back(*event);
            continue;
        }
        auto* subscribed = std::get_if<SubscribedMessage>(&received->message);
        if (subscribed == nullptr)
        {
            continue;
        }
        if (!subscribed->error.empty())
        {
            throw CameraReaderError(subscribed->error);
        }
        if (received->descriptors.size() != subscribed->group.streams.size())
        {
            throw CameraReaderError("the camera plugin did not send one pool per stream of '" + group + "'");
        }
        state->group = std::move(subscribed->group);
        state->options = { options.window, resolved_order(options) };
        try
        {
            for (std::size_t i = 0; i < state->group.streams.size(); ++i)
            {
                const StreamDescription& stream = state->group.streams[i];
                state->pools.emplace_back(std::move(received->descriptors[i]), stream.slot_count, stream.slot_bytes);
            }
        }
        catch (const std::exception& error)
        {
            throw CameraReaderError("cannot map the pools of '" + group + "': " + error.what());
        }
        return CameraReader(std::move(state));
    }
}

CameraReader::CameraReader(std::shared_ptr<detail::ReaderState> state) : m_state(std::move(state))
{
}

CameraReader::CameraReader(CameraReader&& other) noexcept = default;
CameraReader& CameraReader::operator=(CameraReader&& other) noexcept = default;
CameraReader::~CameraReader() = default;

const GroupDescription& CameraReader::group() const
{
    return m_state->group;
}

std::optional<FrameSet> CameraReader::next(std::chrono::milliseconds timeout)
{
    std::optional<FrameEvent> event = m_state->take(timeout);
    if (!event)
    {
        return std::nullopt;
    }
    // With a window of one nothing can be waiting behind this event, so the extra look at the
    // socket is skipped.
    if (m_state->options.order == DeliveryOrder::NEWEST && m_state->options.window > 1)
    {
        while (const std::optional<FrameEvent> newer = m_state->take(std::chrono::milliseconds(0)))
        {
            m_state->release(event->lease_id);
            const std::uint64_t skipped = std::uint64_t{ event->skipped } + newer->skipped + 1;
            event = newer;
            event->skipped = static_cast<std::uint32_t>(std::min<std::uint64_t>(skipped, UINT32_MAX));
        }
    }
    try
    {
        m_state->check(*event);
    }
    catch (const CameraReaderError&)
    {
        m_state->release(event->lease_id);
        throw;
    }
    return FrameSet(m_state, *event);
}

int CameraReader::descriptor() const
{
    return m_state->socket.descriptor();
}

} // namespace core::camera
