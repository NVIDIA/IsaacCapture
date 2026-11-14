// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_reader/camera_reader.hpp"

#include <camera_ipc/endpoint_path.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <camera_protocol/messages.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <exception>
#include <mutex>
#include <system_error>
#include <utility>
#include <variant>

namespace core::camera
{

namespace
{

// How long the plugin gets to answer the handshake.
constexpr std::chrono::milliseconds ANSWER_TIMEOUT(2000);

// How long the plugin gets to answer when a frame set that waited is exchanged for the newest.
// The answer is a frame set the plugin already has, so it comes in well under a millisecond.
constexpr std::chrono::milliseconds EXCHANGE_TIMEOUT(20);

struct Received
{
    ProtocolMessage message;
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
        return Received{ decode_message(buffer.first(received.bytes)), std::move(received.descriptors) };
    }
    catch (const ProtocolError& error)
    {
        throw CameraReaderError(std::string("the camera plugin sent something this reader cannot act on: ") +
                                error.what());
    }
}

MessageSocket connect_to(const PluginLocation& plugin)
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
        return MessageSocket::connect(endpoint);
    }
    catch (const std::system_error& error)
    {
        throw CameraReaderError("no camera plugin answers on '" + endpoint + "': " + error.code().message() +
                                ". Start it through the plugin manager, or run camera_plugin");
    }
    catch (const std::invalid_argument& error)
    {
        throw CameraReaderError(error.what());
    }
}

// What a handshake brought back: the plugin's answer and the descriptors that came with it.
struct Answer
{
    HandshakeResponse response;
    std::vector<UniqueFd> descriptors;
};

// Sends the reader's first message and waits for the plugin's answer, which is the first
// message a plugin sends.
Answer shake_hands(MessageSocket& socket, std::span<std::byte> buffer, const std::string& group, const ReaderOptions& options)
{
    MessageEncoder encoder;
    HandshakeRequest request;
    request.reader_name = program_invocation_short_name;
    request.group = group;
    request.options = options;
    if (socket.send(encoder.encode(request)) != SendStatus::SENT)
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
            throw CameraReaderError("the camera plugin did not answer the reader's handshake");
        }
        if (std::holds_alternative<std::monostate>(received->message))
        {
            continue; // something a newer plugin sends for information
        }
        auto* response = std::get_if<HandshakeResponse>(&received->message);
        if (response == nullptr)
        {
            throw CameraReaderError("the camera plugin sent something else before it answered the reader's handshake");
        }
        if (!response->error.empty())
        {
            throw CameraReaderError(response->error);
        }
        if (response->protocol_version < OLDEST_PROTOCOL_VERSION || response->protocol_version > PROTOCOL_VERSION)
        {
            throw CameraReaderError("the camera plugin answered in protocol version " +
                                    std::to_string(response->protocol_version) + " and this reader speaks versions " +
                                    std::to_string(OLDEST_PROTOCOL_VERSION) + " to " + std::to_string(PROTOCOL_VERSION));
        }
        return { std::move(*response), std::move(received->descriptors) };
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
    explicit ReaderState(MessageSocket connected) : socket(std::move(connected)), buffer(MAX_MESSAGE_BYTES)
    {
    }

    // Called from a FrameSet, possibly on another thread than the one that reads.
    void release(std::uint64_t lease_id) noexcept
    {
        try
        {
            const std::lock_guard<std::mutex> lock(m_send_mutex);
            (void)socket.send(m_encoder.encode(ReleaseMessage{ lease_id }));
        }
        catch (const std::exception&)
        {
            // The plugin is gone or will drop this reader; either way it takes the frame set back.
        }
    }

    // Asks the plugin to hold the newest frame set after the last one this reader took. A
    // request the plugin has no room for is not open, and the next call asks again.
    void ask()
    {
        const std::lock_guard<std::mutex> lock(m_send_mutex);
        const SendStatus status = socket.send(m_encoder.encode(AcquireMessage{ last_taken }));
        if (status == SendStatus::CLOSED)
        {
            throw CameraDisconnected("the camera plugin closed the connection");
        }
        asked = status == SendStatus::SENT;
    }

    // The next frame event from the socket; nothing on timeout.
    std::optional<FrameEvent> take(std::chrono::milliseconds timeout)
    {
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
            if (!std::holds_alternative<std::monostate>(received->message))
            {
                throw CameraReaderError("the camera plugin sent a message only a reader sends");
            }
            // Something a newer plugin sends for information; keep waiting.
        }
    }

    // The newest frame set, for a newest-order reader.
    std::optional<FrameEvent> take_newest(std::chrono::milliseconds timeout)
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        if (asked)
        {
            // An earlier call asked and stopped waiting. A frame set that arrived for it since
            // was held at that time, and whether it is still the newest only the plugin knows:
            // give it back and ask again. The answer exists already, so it comes at once.
            if (const std::optional<FrameEvent> waited = take(std::chrono::milliseconds(0)))
            {
                release(waited->lease_id);
                asked = false;
                deadline = std::max(deadline, std::chrono::steady_clock::now() + EXCHANGE_TIMEOUT);
            }
        }
        if (!asked)
        {
            ask();
        }
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        const std::optional<FrameEvent> event = take(left);
        if (event)
        {
            asked = false;
        }
        return event; // nothing: the request stays open, and its answer is taken by a later call
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
    std::atomic<std::uint32_t> held{ 0 }; // frame sets handed out and not yet given back

    // The reading thread only.
    std::vector<std::byte> buffer;
    bool asked = false; // the plugin owes this reader a frame set
    std::uint64_t last_taken = 0; // the sequence of the last frame set handed out; 0 for none

private:
    std::mutex m_send_mutex; // guards the encoder and sending on the socket
    MessageEncoder m_encoder;
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
    ++m_state->held;
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
        --m_state->held;
        m_state.reset();
    }
}

std::vector<GroupListing> CameraReader::list(const PluginLocation& plugin)
{
    MessageSocket socket = connect_to(plugin);
    std::vector<std::byte> buffer(MAX_MESSAGE_BYTES);
    std::vector<GroupListing> listing;
    for (GroupEntry& entry : shake_hands(socket, buffer, {}, {}).response.groups)
    {
        listing.push_back({ std::move(entry.group), std::move(entry.unsupported) });
    }
    return listing;
}

CameraReader CameraReader::open(const std::string& group, const ReaderOptions& options, const PluginLocation& plugin)
{
    if (group.empty())
    {
        throw CameraReaderError("a camera reader is opened for a group; list() says which there are");
    }
    auto state = std::make_shared<detail::ReaderState>(connect_to(plugin));
    Answer answer = shake_hands(state->socket, state->buffer, group, options);

    // The answer comes from another process: it has to be the group that was asked for, in a
    // form this build reads, described in a way that keeps every view inside its pool.
    if (answer.response.groups.size() != 1)
    {
        throw CameraReaderError("the camera plugin answered the request for group '" + group + "' with " +
                                std::to_string(answer.response.groups.size()) + " groups");
    }
    GroupEntry& entry = answer.response.groups.front();
    if (entry.group.id != group)
    {
        throw CameraReaderError("the camera plugin answered the request for group '" + group + "' with group '" +
                                entry.group.id + "'");
    }
    if (!entry.unsupported.empty())
    {
        throw CameraReaderError("camera group '" + group + "' cannot be read by this build: " + entry.unsupported);
    }
    if (const std::string problem = problem_with(entry.group); !problem.empty())
    {
        throw CameraReaderError("the camera plugin described group '" + group +
                                "' in a way that cannot be read: " + problem);
    }
    if (answer.descriptors.size() != entry.group.streams.size())
    {
        throw CameraReaderError("the camera plugin did not send group '" + group + "' with one pool per stream");
    }
    state->group = std::move(entry.group);
    state->options = { options.window, resolved_order(options) };
    try
    {
        for (std::size_t i = 0; i < state->group.streams.size(); ++i)
        {
            const StreamDescription& stream = state->group.streams[i];
            state->pools.emplace_back(std::move(answer.descriptors[i]), stream.slot_count, stream.slot_bytes);
        }
    }
    catch (const std::exception& error)
    {
        throw CameraReaderError("cannot map the pools of '" + group + "': " + error.what());
    }
    return CameraReader(std::move(state));
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
    detail::ReaderState& state = *m_state;
    if (state.held >= state.options.window)
    {
        throw CameraWindowExhausted("this camera reader holds " + std::to_string(state.held.load()) +
                                    " frame set(s), which is its whole window: give one back before asking for " +
                                    "the next, or open the reader with a larger window");
    }
    const std::optional<FrameEvent> event =
        state.options.order == DeliveryOrder::NEWEST ? state.take_newest(timeout) : state.take(timeout);
    if (!event)
    {
        return std::nullopt;
    }
    try
    {
        state.check(*event);
    }
    catch (const CameraReaderError&)
    {
        state.release(event->lease_id);
        throw;
    }
    state.last_taken = event->sequence;
    return FrameSet(m_state, *event);
}

int CameraReader::descriptor() const
{
    return m_state->socket.descriptor();
}

} // namespace core::camera
