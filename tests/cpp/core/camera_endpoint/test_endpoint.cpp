// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_endpoint/endpoint.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <camera_protocol/messages.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <schema/camera_protocol_generated.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

using namespace core::camera;
using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;

namespace
{

constexpr std::uint32_t WIDTH = 64;
constexpr std::uint32_t HEIGHT = 48;
constexpr std::uint32_t SLOTS = 8;

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_endpoint_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    std::string socket_path() const
    {
        return (m_path / "cameras.sock").string();
    }

private:
    std::filesystem::path m_path;
};

GroupDescription gray_group(const std::string& id, std::size_t streams)
{
    GroupDescription group;
    group.id = id;
    for (std::size_t i = 0; i < streams; ++i)
    {
        StreamDescription stream;
        stream.id = id + "/" + std::to_string(i);
        stream.pixel_format = PixelFormat::GRAY8;
        stream.width = WIDTH;
        stream.height = HEIGHT;
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = WIDTH;
        stream.slot_bytes = WIDTH * HEIGHT;
        stream.slot_count = SLOTS;
        stream.frame_rate = 60.0;
        group.streams.push_back(stream);
    }
    return group;
}

// A plugin in miniature: a stereo group and a single camera, in host memory, behind an endpoint.
struct Served
{
    TemporaryDirectory directory;
    Publisher publisher;
    GroupId head = publisher.add_group(gray_group("head", 2));
    GroupId wrist = publisher.add_group(gray_group("wrist", 1));
    HostPool left{ SLOTS, WIDTH* HEIGHT, "test_left" };
    HostPool right{ SLOTS, WIDTH* HEIGHT, "test_right" };
    HostPool hand{ SLOTS, WIDTH* HEIGHT, "test_wrist" };
    Endpoint endpoint{ publisher, directory.socket_path() };

    Served()
    {
        endpoint.set_group_memory(head, { left.descriptor(), right.descriptor() });
        endpoint.set_group_memory(wrist, { hand.descriptor() });
        endpoint.start();
    }

    // Publishes one stereo frame set whose every pixel is `value`.
    std::uint64_t publish(std::uint8_t value)
    {
        std::vector<ImageInfo> images(2);
        HostPool* pools[] = { &left, &right };
        for (std::size_t i = 0; i < 2; ++i)
        {
            const auto slot = publisher.acquire_slot(head, i);
            REQUIRE(slot.has_value());
            std::fill(pools[i]->slot(*slot).begin(), pools[i]->slot(*slot).end(), std::byte{ value });
            images[i].slot = *slot;
        }
        return publisher.publish(head, images, {});
    }
};

// The reader's part, played with the socket and the codec directly.
struct Reader
{
    struct Received
    {
        ReceiveStatus status = ReceiveStatus::TIMEOUT;
        ProtocolMessage message;
        std::vector<UniqueFd> descriptors;
    };

    explicit Reader(const Served& served) : socket(MessageSocket::connect(served.directory.socket_path()))
    {
    }

    template <typename Message>
    void send(const Message& message)
    {
        REQUIRE(socket.send(encoder.encode(message)) == SendStatus::SENT);
    }

    Received receive(std::chrono::milliseconds timeout = 2s)
    {
        Received result;
        ReceivedMessage received = socket.receive(buffer, timeout);
        result.status = received.status;
        if (received.status == ReceiveStatus::MESSAGE)
        {
            result.message = decode_message({ buffer.data(), received.bytes });
            result.descriptors = std::move(received.descriptors);
        }
        return result;
    }

    template <typename Message>
    Message expect(std::chrono::milliseconds timeout = 2s)
    {
        Received received = receive(timeout);
        REQUIRE(received.status == ReceiveStatus::MESSAGE);
        REQUIRE(std::holds_alternative<Message>(received.message));
        return std::get<Message>(std::move(received.message));
    }

    // The reader's first message and the plugin's answer, with the descriptors that came with it.
    // Without a group it asks what is published.
    std::pair<HandshakeResponse, std::vector<UniqueFd>> handshake(const std::string& group = {},
                                                                  ReaderOptions options = {})
    {
        send(HandshakeRequest{ .reader_name = "test reader", .group = group, .options = options });
        Received received = receive();
        REQUIRE(received.status == ReceiveStatus::MESSAGE);
        REQUIRE(std::holds_alternative<HandshakeResponse>(received.message));
        return { std::get<HandshakeResponse>(std::move(received.message)), std::move(received.descriptors) };
    }

    // Asks for the newest frame set after `after_sequence`, as a newest-order reader does.
    void ask(std::uint64_t after_sequence = 0)
    {
        send(AcquireMessage{ after_sequence });
    }

    // A message of a type no build has yet, as a newer reader might send.
    void send_unknown(bool skippable)
    {
        flatbuffers::FlatBufferBuilder builder;
        const auto body = core::CreateCameraRelease(builder, 1);
        builder.Finish(core::CreateCameraMessage(
            builder, static_cast<core::CameraMessageBody>(core::CameraMessageBody_MAX + 1), body.Union(), skippable));
        REQUIRE(socket.send({ reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() }) ==
                SendStatus::SENT);
    }

    MessageSocket socket;
    MessageEncoder encoder;
    std::vector<std::byte> buffer = std::vector<std::byte>(MAX_MESSAGE_BYTES);
};

bool wait_until(const std::function<bool()>& condition, std::chrono::milliseconds timeout = 2s)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

} // namespace

TEST_CASE("a reader is told what is published", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);

    auto [response, descriptors] = reader.handshake();
    REQUIRE(response.error.empty());
    REQUIRE(response.protocol_version == PROTOCOL_VERSION);
    REQUIRE(descriptors.empty());
    REQUIRE(response.groups.size() == 2);
    REQUIRE(response.groups[0].group.id == "head");
    REQUIRE(response.groups[0].unsupported.empty());
    REQUIRE(response.groups[0].group.session_id != 0);
    REQUIRE(response.groups[0].group.streams.size() == 2);
    REQUIRE(response.groups[0].group.streams[1].id == "head/1");
    REQUIRE(response.groups[0].group.streams[1].width == WIDTH);
    REQUIRE(problem_with(response.groups[0].group).empty());
    REQUIRE(response.groups[1].group.id == "wrist");
    REQUIRE(response.groups[1].group.session_id != response.groups[0].group.session_id);
    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 1; }));
}

TEST_CASE("a connection uses the newest protocol version both sides speak", "[unit][camera_endpoint]")
{
    Served served;

    // A reader from a later release: the plugin answers in its own version, and it is for the
    // reader to say whether it still speaks that.
    Reader newer(served);
    newer.send(HandshakeRequest{ .protocol_version = PROTOCOL_VERSION + 3, .reader_name = "newer", .group = "head" });
    Reader::Received answer = newer.receive();
    REQUIRE(answer.status == ReceiveStatus::MESSAGE);
    const HandshakeResponse& accepted = std::get<HandshakeResponse>(answer.message);
    REQUIRE(accepted.error.empty());
    REQUIRE(accepted.protocol_version == PROTOCOL_VERSION);
    REQUIRE(accepted.groups.size() == 1);
    REQUIRE(answer.descriptors.size() == 2);

    // A reader from before the oldest version this plugin still speaks is refused.
    Reader older(served);
    older.send(
        HandshakeRequest{ .protocol_version = OLDEST_PROTOCOL_VERSION - 1, .reader_name = "older", .group = "head" });
    const HandshakeResponse refused = older.expect<HandshakeResponse>();
    REQUIRE_THAT(refused.error, ContainsSubstring("protocol versions"));
    REQUIRE(refused.groups.empty());
    REQUIRE(older.receive().status == ReceiveStatus::CLOSED);
}

TEST_CASE("a reader that asks for a group maps its pools and is sent the frame sets it asks for",
          "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);

    auto [response, descriptors] = reader.handshake("head");
    REQUIRE(response.error.empty());
    REQUIRE(response.groups.size() == 1);
    REQUIRE(response.groups[0].group.id == "head");
    REQUIRE(descriptors.size() == 2);
    const HostPoolView left(std::move(descriptors[0]), SLOTS, WIDTH * HEIGHT);
    const HostPoolView right(std::move(descriptors[1]), SLOTS, WIDTH * HEIGHT);

    // Nothing is sent to a newest-order reader that has not asked.
    served.publish(0x11);
    REQUIRE(reader.receive(20ms).status == ReceiveStatus::TIMEOUT);

    reader.ask();
    const FrameEvent first = reader.expect<FrameEvent>();
    REQUIRE(first.sequence == 1);
    REQUIRE(first.image_count == 2);
    REQUIRE(first.skipped == 0);
    REQUIRE(first.timeline.published_ns != TIME_UNKNOWN);
    REQUIRE(first.timeline.sent_ns >= first.timeline.published_ns);
    REQUIRE(left.slot(first.images[0].slot).front() == std::byte{ 0x11 });
    REQUIRE(right.slot(first.images[1].slot).back() == std::byte{ 0x11 });

    // A window of one holds one frame set. A request made while it is held is answered when
    // it is given back, with the newest frame set there is then.
    reader.ask(first.sequence);
    served.publish(0x22);
    served.publish(0x33);
    REQUIRE(reader.receive(20ms).status == ReceiveStatus::TIMEOUT);
    REQUIRE(left.slot(first.images[0].slot).front() == std::byte{ 0x11 }); // still held, so not rewritten

    reader.send(ReleaseMessage{ first.lease_id });
    const FrameEvent newest = reader.expect<FrameEvent>();
    REQUIRE(newest.sequence == 3);
    REQUIRE(newest.skipped == 1);
    REQUIRE(left.slot(newest.images[0].slot).front() == std::byte{ 0x33 });
}

TEST_CASE("a reader that paused is handed what the camera saw last", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    auto [response, descriptors] = reader.handshake("head");
    REQUIRE(response.error.empty());
    const HostPoolView left(std::move(descriptors[0]), SLOTS, WIDTH * HEIGHT);

    reader.ask();
    served.publish(1);
    const FrameEvent before = reader.expect<FrameEvent>();
    reader.send(ReleaseMessage{ before.lease_id });

    // The reader does something else, holding nothing and asking for nothing, while the camera
    // runs through its whole pool several times. No frame set is put aside for the reader, so
    // none is waiting for it and the camera never runs short of slots.
    for (std::uint8_t value = 2; value <= 3 * SLOTS; ++value)
    {
        served.publish(value);
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(reader.receive(20ms).status == ReceiveStatus::TIMEOUT);
    REQUIRE(served.publisher.stats(served.head).no_free_slot == 0);

    // Back again, it gets the last frame set published, held from this moment on.
    const auto asked = std::chrono::steady_clock::now().time_since_epoch();
    reader.ask(before.sequence);
    const FrameEvent after = reader.expect<FrameEvent>();
    REQUIRE(after.sequence == 3 * SLOTS);
    REQUIRE(after.skipped == 3 * SLOTS - 2);
    REQUIRE(left.slot(after.images[0].slot).front() == std::byte{ 3 * SLOTS });
    REQUIRE(after.timeline.sent_ns >= std::chrono::duration_cast<std::chrono::nanoseconds>(asked).count());
}

TEST_CASE("the answer to a handshake comes before any frame set", "[unit][camera_endpoint]")
{
    Served served;

    // A camera that publishes without pause while readers come and go.
    std::atomic<bool> stopping{ false };
    std::thread capture(
        [&]
        {
            std::vector<ImageInfo> images(2);
            while (!stopping)
            {
                const auto left = served.publisher.acquire_slot(served.head, 0);
                const auto right = served.publisher.acquire_slot(served.head, 1);
                if (left && right)
                {
                    images[0].slot = *left;
                    images[1].slot = *right;
                    served.publisher.publish(served.head, images, {});
                }
                else
                {
                    if (left)
                    {
                        served.publisher.abandon_slot(served.head, 0, *left);
                    }
                    if (right)
                    {
                        served.publisher.abandon_slot(served.head, 1, *right);
                    }
                    std::this_thread::yield();
                }
            }
        });

    // An in-order reader is sent frame sets without asking, so one published while its
    // handshake is being answered is the one that could arrive first. handshake() requires the
    // first message to be the answer.
    for (int i = 0; i < 200; ++i)
    {
        Reader reader(served);
        auto [response, descriptors] = reader.handshake("head", { .window = 2, .order = DeliveryOrder::IN_ORDER });
        REQUIRE(response.error.empty());
        REQUIRE(descriptors.size() == 2);
        const FrameEvent first = reader.expect<FrameEvent>();
        REQUIRE(first.image_count == 2);
    }
    stopping = true;
    capture.join();
}

TEST_CASE("a reader that cannot be served is told why and its connection ends", "[unit][camera_endpoint]")
{
    Served served;
    {
        Reader reader(served);
        auto [unknown, descriptors] = reader.handshake("chest");
        REQUIRE_THAT(unknown.error, ContainsSubstring("no group 'chest'"));
        REQUIRE(unknown.groups.empty());
        REQUIRE(descriptors.empty());
        REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
    }
    {
        Reader reader(served);
        auto [too_large, descriptors] = reader.handshake("head", { .window = SLOTS });
        REQUIRE_THAT(too_large.error, ContainsSubstring("does not fit the pool"));
        REQUIRE(descriptors.empty());
        REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
    }
    {
        // A delivery order added after this build: refused with the reason, not left unanswered.
        Reader reader(served);
        auto [newer_order, descriptors] = reader.handshake("head", { .order = static_cast<DeliveryOrder>(7) });
        REQUIRE_THAT(newer_order.error, ContainsSubstring("delivery order"));
        REQUIRE(descriptors.empty());
        REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
    }
    // A refusal leaves nothing behind: the pool has all its room for a reader that fits.
    Reader reader(served);
    auto [accepted, descriptors] = reader.handshake("head", { .window = 3, .order = DeliveryOrder::IN_ORDER });
    REQUIRE(accepted.error.empty());
    REQUIRE(descriptors.size() == 2);
}

TEST_CASE("a subscription whose reader left before the answer leaves nothing behind", "[unit][camera_endpoint]")
{
    Served served;
    for (int i = 0; i < 20; ++i)
    {
        // Asks for a place in the pool and is gone before it could be told it has one.
        Reader reader(served);
        reader.send(HandshakeRequest{ .group = "head", .options = { .window = 2, .order = DeliveryOrder::IN_ORDER } });
    }
    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 0; }));

    // Every one of those reservations is gone: a reader that needs all the room there is fits.
    Reader reader(served);
    auto [accepted, descriptors] = reader.handshake("head", { .window = 3, .order = DeliveryOrder::IN_ORDER });
    REQUIRE(accepted.error.empty());
    served.publish(1);
    REQUIRE(reader.expect<FrameEvent>().sequence == 1);
}

TEST_CASE("a reader that disappears returns what it held", "[unit][camera_endpoint]")
{
    Served served;
    {
        Reader reader(served);
        REQUIRE(reader.handshake("head", { .window = 2, .order = DeliveryOrder::IN_ORDER }).first.error.empty());
        served.publish(1);
        served.publish(2);
        REQUIRE(reader.expect<FrameEvent>().sequence == 1);
        REQUIRE(reader.expect<FrameEvent>().sequence == 2);
        REQUIRE(served.endpoint.reader_count() == 1);
        // Gone without releasing either frame set.
    }
    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 0; }));

    // With the held slots back, capture runs through the whole pool again and again.
    for (std::uint8_t i = 0; i < 3 * SLOTS; ++i)
    {
        served.publish(i);
    }
    REQUIRE(served.publisher.stats(served.head).no_free_slot == 0);

    // And the room in the pool is there for the next reader.
    Reader next(served);
    REQUIRE(next.handshake("head", { .window = 2, .order = DeliveryOrder::IN_ORDER }).first.error.empty());
}

TEST_CASE("a frame set held across a camera restart stays as it was", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    auto [response, descriptors] = reader.handshake("head");
    REQUIRE(response.error.empty());
    const HostPoolView left(std::move(descriptors[0]), SLOTS, WIDTH * HEIGHT);

    reader.ask();
    served.publish(0x11);
    const FrameEvent held = reader.expect<FrameEvent>();
    REQUIRE(held.generation == 0);

    // The camera restarts and runs through its pool several times. The slot the reader holds
    // is not given to the camera again, so what the reader sees does not change.
    served.publisher.restart_group(served.head);
    for (std::uint8_t i = 0; i < 3 * SLOTS; ++i)
    {
        served.publish(0x80 + i);
    }
    REQUIRE(served.publisher.stats(served.head).no_free_slot == 0);
    const std::span<const std::byte> pixels = left.slot(held.images[0].slot);
    REQUIRE(std::all_of(pixels.begin(), pixels.end(), [](std::byte pixel) { return pixel == std::byte{ 0x11 }; }));

    // What it is sent next is from after the restart, and says so.
    reader.send(ReleaseMessage{ held.lease_id });
    reader.ask(held.sequence);
    const FrameEvent next = reader.expect<FrameEvent>();
    REQUIRE(next.generation == 1);
    REQUIRE(next.sequence == 1 + 3 * SLOTS);
}

TEST_CASE("a reader that breaks the protocol is dropped and the others are not", "[unit][camera_endpoint]")
{
    Served served;
    Reader good(served);
    REQUIRE(good.handshake("head").first.error.empty());

    Reader garbage(served);
    const std::vector<std::byte> noise(32, std::byte{ 0x5a });
    REQUIRE(garbage.socket.send(noise) == SendStatus::SENT);
    REQUIRE(garbage.receive().status == ReceiveStatus::CLOSED);

    // A message only a plugin sends.
    Reader impostor(served);
    impostor.send(HandshakeResponse{});
    REQUIRE(impostor.receive().status == ReceiveStatus::CLOSED);

    // A connection is for one thing, said once.
    Reader twice(served);
    REQUIRE(twice.handshake().first.error.empty());
    twice.send(HandshakeRequest{ .group = "head" });
    REQUIRE(twice.receive().status == ReceiveStatus::CLOSED);

    // Asking for the newest frame set is for newest-order readers that have a subscription.
    Reader ordered(served);
    REQUIRE(ordered.handshake("head", { .window = 2, .order = DeliveryOrder::IN_ORDER }).first.error.empty());
    ordered.ask();
    REQUIRE(ordered.receive().status == ReceiveStatus::CLOSED);
    Reader lister(served);
    REQUIRE(lister.handshake().first.error.empty());
    lister.ask();
    REQUIRE(lister.receive().status == ReceiveStatus::CLOSED);

    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 1; }));
    good.ask();
    served.publish(7);
    REQUIRE(good.expect<FrameEvent>().sequence == 1);
}

TEST_CASE("a newer reader's message is ignored only when it may be", "[unit][camera_endpoint]")
{
    Served served;

    // Something that only informs: the plugin carries on as if it had not been sent.
    Reader informing(served);
    REQUIRE(informing.handshake("head").first.error.empty());
    informing.send_unknown(true);
    informing.ask();
    served.publish(1);
    REQUIRE(informing.expect<FrameEvent>().sequence == 1);

    // An operation the reader counts on and this plugin cannot carry out: the connection ends,
    // which the reader notices, instead of the reader waiting for something that never comes.
    Reader depending(served);
    REQUIRE(depending.handshake("head").first.error.empty());
    depending.send_unknown(false);
    REQUIRE(depending.receive().status == ReceiveStatus::CLOSED);
    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 1; }));
}

TEST_CASE("readers do not slow each other", "[unit][camera_endpoint]")
{
    Served served;
    Reader stalled(served);
    REQUIRE(stalled.handshake("head").first.error.empty());
    Reader live(served);
    REQUIRE(live.handshake("head").first.error.empty());
    stalled.ask();
    live.ask();

    // One reader never releases; the other still gets every frame set.
    for (std::uint64_t sequence = 1; sequence <= 5; ++sequence)
    {
        REQUIRE(served.publish(static_cast<std::uint8_t>(sequence)) == sequence);
        const FrameEvent event = live.expect<FrameEvent>();
        REQUIRE(event.sequence == sequence);
        REQUIRE(event.skipped == 0);
        // Whether the release and the request are seen before or after the next publish, that
        // frame set is next.
        live.send(ReleaseMessage{ event.lease_id });
        live.ask(event.sequence);
    }
    // The stalled reader asked once and was sent one frame set: the newest there was when its
    // request was served, which is the first unless the endpoint got to it later.
    const FrameEvent held = stalled.expect<FrameEvent>();
    REQUIRE(held.sequence >= 1);
    REQUIRE(held.sequence <= 5);
    REQUIRE(stalled.receive(20ms).status == ReceiveStatus::TIMEOUT);
}

TEST_CASE("a reader that sends without pause keeps nobody waiting", "[unit][camera_endpoint]")
{
    Served served;

    // One connection, and three threads sending valid requests on it as fast as they can:
    // more than the endpoint handles in the same time, so there is always another one waiting.
    Reader flooder(served);
    REQUIRE(flooder.handshake("head").first.error.empty());
    // Stops and joins the senders however the test ends.
    struct Flood
    {
        ~Flood()
        {
            flooding = false;
            for (std::thread& sender : senders)
            {
                sender.join();
            }
        }

        std::atomic<bool> flooding{ true };
        std::vector<std::thread> senders;
    } flood;
    for (int i = 0; i < 3; ++i)
    {
        flood.senders.emplace_back(
            [&]
            {
                MessageEncoder encoder;
                // A request that is never answered, so nothing piles up for the flooder to read.
                const AcquireMessage never{ std::numeric_limits<std::uint64_t>::max() };
                while (flood.flooding)
                {
                    (void)flooder.socket.send(encoder.encode(never));
                }
            });
    }

    // Other readers are served all the same.
    for (std::uint64_t sequence = 1; sequence <= 5; ++sequence)
    {
        Reader other(served);
        REQUIRE(other.handshake("head").first.error.empty());
        // It asks for what comes after the frame set before, so that it is sent this round's
        // whether its request is served before the publish or after it.
        other.ask(sequence - 1);
        served.publish(static_cast<std::uint8_t>(sequence));
        REQUIRE(other.expect<FrameEvent>().sequence == sequence);
    }

    // And the endpoint stops while the flood goes on.
    const auto before = std::chrono::steady_clock::now();
    served.endpoint.stop();
    REQUIRE(std::chrono::steady_clock::now() - before < 2s);
}

TEST_CASE("stopping the endpoint closes every reader, and it can serve again", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    REQUIRE(reader.handshake("head").first.error.empty());

    served.endpoint.stop();
    REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
    REQUIRE(served.endpoint.reader_count() == 0);
    served.endpoint.stop(); // nothing to stop

    // A stopped endpoint refuses a reader at once; it does not leave it waiting for an answer.
    REQUIRE_THROWS_AS(MessageSocket::connect(served.directory.socket_path()), std::system_error);

    served.endpoint.start();
    Reader next(served);
    REQUIRE(next.handshake().first.groups.size() == 2);
    Reader again(served);
    REQUIRE(again.handshake("head").first.error.empty());
    again.ask();
    served.publish(9);
    REQUIRE(again.expect<FrameEvent>().sequence == 1);
}

TEST_CASE("a group needs its memory named before readers can ask for it", "[unit][camera_endpoint]")
{
    TemporaryDirectory directory;
    Publisher publisher;
    const GroupId head = publisher.add_group(gray_group("head", 2));
    HostPool pool(SLOTS, WIDTH * HEIGHT, "test");
    Endpoint endpoint(publisher, directory.socket_path());

    REQUIRE_THROWS_AS(endpoint.set_group_memory(head, { pool.descriptor() }), std::invalid_argument);
    REQUIRE_THROWS_AS(endpoint.set_group_memory(head + 1, {}), std::out_of_range);
    endpoint.start();
    REQUIRE_THROWS_AS(endpoint.set_group_memory(head, { pool.descriptor(), pool.descriptor() }), std::logic_error);

    MessageSocket socket = MessageSocket::connect(directory.socket_path());
    MessageEncoder encoder;
    std::vector<std::byte> buffer(MAX_MESSAGE_BYTES);
    REQUIRE(socket.send(encoder.encode(HandshakeRequest{ .group = "head" })) == SendStatus::SENT);
    const ReceivedMessage received = socket.receive(buffer, 2s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    const ProtocolMessage answer = decode_message({ buffer.data(), received.bytes });
    REQUIRE_THAT(std::get<HandshakeResponse>(answer).error, ContainsSubstring("no memory a reader can map"));

    // Another endpoint cannot take the same path while this one lives.
    REQUIRE_THROWS_AS(Endpoint(publisher, directory.socket_path()), std::system_error);
}
