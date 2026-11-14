// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_endpoint/endpoint.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <camera_wire/messages.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
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
        WireMessage message;
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
            result.message = decode_wire_message({ buffer.data(), received.bytes });
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

    WelcomeMessage hello(const std::string& name = "test reader")
    {
        send(HelloMessage{ .reader_name = name });
        return expect<WelcomeMessage>();
    }

    // Subscribes and returns the answer with the pool descriptors that came with it.
    std::pair<SubscribedMessage, std::vector<UniqueFd>> subscribe(const std::string& group, ReaderOptions options = {})
    {
        send(SubscribeMessage{ group, options });
        Received received = receive();
        REQUIRE(received.status == ReceiveStatus::MESSAGE);
        REQUIRE(std::holds_alternative<SubscribedMessage>(received.message));
        return { std::get<SubscribedMessage>(std::move(received.message)), std::move(received.descriptors) };
    }

    MessageSocket socket;
    WireEncoder encoder;
    std::vector<std::byte> buffer = std::vector<std::byte>(MAX_WIRE_MESSAGE_BYTES);
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

    const WelcomeMessage welcome = reader.hello();
    REQUIRE(welcome.error.empty());
    REQUIRE(welcome.protocol_version == WIRE_PROTOCOL_VERSION);
    REQUIRE(welcome.groups.size() == 2);
    REQUIRE(welcome.groups[0].id == "head");
    REQUIRE(welcome.groups[0].streams.size() == 2);
    REQUIRE(welcome.groups[0].streams[1].id == "head/1");
    REQUIRE(welcome.groups[0].streams[1].width == WIDTH);
    REQUIRE(welcome.groups[1].id == "wrist");
    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 1; }));
}

TEST_CASE("a reader that speaks another protocol version is refused", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);

    reader.send(HelloMessage{ .protocol_version = WIRE_PROTOCOL_VERSION + 1, .reader_name = "from the future" });
    const WelcomeMessage welcome = reader.expect<WelcomeMessage>();
    REQUIRE_THAT(welcome.error, ContainsSubstring("protocol version"));
    REQUIRE(welcome.groups.empty());
    REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
}

TEST_CASE("a subscriber maps the pools and is sent frame sets", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    reader.hello();

    auto [subscribed, descriptors] = reader.subscribe("head");
    REQUIRE(subscribed.error.empty());
    REQUIRE(subscribed.group.id == "head");
    REQUIRE(descriptors.size() == 2);
    const HostPoolView left(std::move(descriptors[0]), SLOTS, WIDTH * HEIGHT);
    const HostPoolView right(std::move(descriptors[1]), SLOTS, WIDTH * HEIGHT);

    served.publish(0x11);
    const FrameEvent first = reader.expect<FrameEvent>();
    REQUIRE(first.sequence == 1);
    REQUIRE(first.image_count == 2);
    REQUIRE(first.skipped == 0);
    REQUIRE(first.timeline.published_ns != TIME_UNKNOWN);
    REQUIRE(first.timeline.sent_ns >= first.timeline.published_ns);
    REQUIRE(left.slot(first.images[0].slot).front() == std::byte{ 0x11 });
    REQUIRE(right.slot(first.images[1].slot).back() == std::byte{ 0x11 });

    // A window of one holds one frame set; the next comes with the release, and is the newest.
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

TEST_CASE("a subscription that cannot be served is refused and the connection stays usable", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    reader.hello();

    auto [unknown, no_descriptors] = reader.subscribe("chest");
    REQUIRE_THAT(unknown.error, ContainsSubstring("no group 'chest'"));
    REQUIRE(no_descriptors.empty());

    auto [too_large, none] = reader.subscribe("head", { .window = SLOTS });
    REQUIRE_THAT(too_large.error, ContainsSubstring("does not fit the pool"));
    REQUIRE(none.empty());

    auto [accepted, descriptors] = reader.subscribe("wrist");
    REQUIRE(accepted.error.empty());
    REQUIRE(descriptors.size() == 1);

    auto [second, extra] = reader.subscribe("head");
    REQUIRE_THAT(second.error, ContainsSubstring("one subscription"));
    REQUIRE(extra.empty());
}

TEST_CASE("a reader that disappears returns what it held", "[unit][camera_endpoint]")
{
    Served served;
    {
        Reader reader(served);
        reader.hello();
        auto [subscribed, descriptors] = reader.subscribe("head", { .window = 2, .order = DeliveryOrder::IN_ORDER });
        REQUIRE(subscribed.error.empty());
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
    next.hello();
    auto [again, descriptors] = next.subscribe("head", { .window = 2, .order = DeliveryOrder::IN_ORDER });
    REQUIRE(again.error.empty());
}

TEST_CASE("a reader that breaks the protocol is dropped and the others are not", "[unit][camera_endpoint]")
{
    Served served;
    Reader good(served);
    good.hello();
    auto [subscribed, descriptors] = good.subscribe("head");
    REQUIRE(subscribed.error.empty());

    Reader garbage(served);
    const std::vector<std::byte> noise(32, std::byte{ 0x5a });
    REQUIRE(garbage.socket.send(noise) == SendStatus::SENT);
    REQUIRE(garbage.receive().status == ReceiveStatus::CLOSED);

    Reader impostor(served);
    impostor.hello();
    impostor.send(WelcomeMessage{});
    REQUIRE(impostor.receive().status == ReceiveStatus::CLOSED);

    Reader impatient(served);
    impatient.send(SubscribeMessage{ "head", {} });
    REQUIRE(impatient.receive().status == ReceiveStatus::CLOSED);

    REQUIRE(wait_until([&] { return served.endpoint.reader_count() == 1; }));
    served.publish(7);
    REQUIRE(good.expect<FrameEvent>().sequence == 1);
}

TEST_CASE("readers do not slow each other", "[unit][camera_endpoint]")
{
    Served served;
    Reader stalled(served);
    stalled.hello();
    REQUIRE(stalled.subscribe("head").first.error.empty());
    Reader live(served);
    live.hello();
    REQUIRE(live.subscribe("head").first.error.empty());

    // One reader never releases; the other still gets every frame set.
    for (std::uint64_t sequence = 1; sequence <= 5; ++sequence)
    {
        REQUIRE(served.publish(static_cast<std::uint8_t>(sequence)) == sequence);
        const FrameEvent event = live.expect<FrameEvent>();
        REQUIRE(event.sequence == sequence);
        REQUIRE(event.skipped == 0);
        // Whether the release is seen before or after the next publish, that frame set is next.
        live.send(ReleaseMessage{ event.lease_id });
    }
    REQUIRE(stalled.expect<FrameEvent>().sequence == 1);
    REQUIRE(stalled.receive(20ms).status == ReceiveStatus::TIMEOUT);
}

TEST_CASE("stopping the endpoint closes every reader, and it can serve again", "[unit][camera_endpoint]")
{
    Served served;
    Reader reader(served);
    reader.hello();
    REQUIRE(reader.subscribe("head").first.error.empty());

    served.endpoint.stop();
    REQUIRE(reader.receive().status == ReceiveStatus::CLOSED);
    REQUIRE(served.endpoint.reader_count() == 0);
    served.endpoint.stop(); // nothing to stop

    // A stopped endpoint refuses a reader at once; it does not leave it waiting for an answer.
    REQUIRE_THROWS_AS(MessageSocket::connect(served.directory.socket_path()), std::system_error);

    served.endpoint.start();
    Reader next(served);
    REQUIRE(next.hello().groups.size() == 2);
    REQUIRE(next.subscribe("head").first.error.empty());
    served.publish(9);
    REQUIRE(next.expect<FrameEvent>().sequence == 1);
}

TEST_CASE("a group needs its memory named before readers can subscribe", "[unit][camera_endpoint]")
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
    WireEncoder encoder;
    std::vector<std::byte> buffer(MAX_WIRE_MESSAGE_BYTES);
    REQUIRE(socket.send(encoder.encode(HelloMessage{})) == SendStatus::SENT);
    REQUIRE(socket.receive(buffer, 2s).status == ReceiveStatus::MESSAGE);
    REQUIRE(socket.send(encoder.encode(SubscribeMessage{ "head", {} })) == SendStatus::SENT);
    const ReceivedMessage received = socket.receive(buffer, 2s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    const WireMessage answer = decode_wire_message({ buffer.data(), received.bytes });
    REQUIRE_THAT(std::get<SubscribedMessage>(answer).error, ContainsSubstring("no memory a reader can map"));

    // Another endpoint cannot take the same path while this one lives.
    REQUIRE_THROWS_AS(Endpoint(publisher, directory.socket_path()), std::system_error);
}
