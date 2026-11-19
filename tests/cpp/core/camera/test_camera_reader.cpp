// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "endpoint_directory.hpp"

#include <camera/camera_reader.hpp>
#include <camera/endpoint.hpp>
#include <camera/endpoint_path.hpp>
#include <camera/host_pool.hpp>
#include <camera/message_socket.hpp>
#include <camera/messages.hpp>
#include <camera/unique_fd.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <schema/camera_protocol_generated.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
constexpr std::uint32_t STRIDE = 80; // rows are padded
constexpr std::uint32_t SLOTS = 8;
constexpr std::uint64_t SLOT_BYTES = STRIDE * HEIGHT;

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_reader_XXXXXX").string();
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

GroupDescription stereo_group(const std::string& id = "head")
{
    GroupDescription group;
    group.id = id;
    for (const char* eye : { "left", "right" })
    {
        StreamDescription stream;
        stream.id = id + "/" + eye;
        stream.pixel_format = PixelFormat::GRAY8;
        stream.width = WIDTH;
        stream.height = HEIGHT;
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = STRIDE;
        stream.slot_bytes = SLOT_BYTES;
        stream.slot_count = SLOTS;
        stream.frame_rate = 60.0;
        group.streams.push_back(stream);
    }
    return group;
}

// A plugin in miniature: a stereo group in host memory behind an endpoint. The endpoint is a
// socket in a directory of its own, or the path it is given: one in the user's endpoint
// directory, where a reader finds it without being told.
struct Served
{
    std::optional<TemporaryDirectory> directory;
    std::string endpoint_path;
    Publisher publisher;
    GroupId head;
    HostPool left{ SLOTS, SLOT_BYTES, "test_left" };
    HostPool right{ SLOTS, SLOT_BYTES, "test_right" };
    Endpoint endpoint;

    explicit Served(const std::string& group = "head", const std::string& path = {})
        : endpoint_path(path.empty() ? directory.emplace().socket_path() : path),
          head(publisher.add_group(stereo_group(group))),
          endpoint(publisher, endpoint_path)
    {
        endpoint.set_group_memory(head, { left.descriptor(), right.descriptor() });
        endpoint.start();
    }

    // Publishes one frame set: every left pixel is `value`, every right pixel `value + 1`.
    std::uint64_t publish(std::uint8_t value, const FrameTimeline& timeline = {})
    {
        std::vector<ImageInfo> images(2);
        HostPool* pools[] = { &left, &right };
        for (std::size_t i = 0; i < 2; ++i)
        {
            const auto slot = publisher.acquire_slot(head, i);
            REQUIRE(slot.has_value());
            const std::byte fill{ static_cast<std::uint8_t>(value + i) };
            std::fill(pools[i]->slot(*slot).begin(), pools[i]->slot(*slot).end(), fill);
            images[i].slot = *slot;
            images[i].device_frame_id = 1000 + value;
        }
        return publisher.publish(head, images, timeline);
    }

    // Gives the endpoint's thread time to serve a request the reader has made: it is served
    // before the publish that answers it or after, and the frame set is sent either way.
    void wait_until_sent(std::uint64_t count)
    {
        for (int i = 0; i < 2000 && publisher.stats(head).events_sent < count; ++i)
        {
            std::this_thread::sleep_for(1ms);
        }
    }
};

std::byte first_pixel(const FrameSet& frames, std::size_t stream)
{
    return frames.image(stream).data.front();
}

// What a plugin that was killed leaves behind: a socket file with nothing behind it.
void leave_socket_file(const std::string& path)
{
    sockaddr_un address = {};
    address.sun_family = AF_UNIX;
    REQUIRE(path.size() < sizeof(address.sun_path));
    std::memcpy(address.sun_path, path.data(), path.size());
    const UniqueFd dead(::socket(AF_UNIX, SOCK_SEQPACKET, 0));
    REQUIRE(::bind(dead.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
}

// A stand-in plugin, to answer a reader with what a real plugin does not send on demand. It
// takes one reader and its handshake, then does what the test says. Like Served, it is on a
// socket in a directory of its own or on the path it is given.
class StandIn
{
public:
    using Conversation = std::function<void(MessageSocket& reader, const HandshakeRequest& request)>;

    explicit StandIn(Conversation conversation, const std::string& path = {})
        : m_path(path.empty() ? m_directory.emplace().socket_path() : path), m_listener(m_path)
    {
        // Before the constructor returns, so that the reader the test opens next is not refused.
        m_listener.listen();
        m_thread = std::thread(
            [this, conversation = std::move(conversation)]
            {
                try
                {
                    std::optional<MessageSocket> reader = m_listener.accept(5s);
                    if (!reader)
                    {
                        return;
                    }
                    std::vector<std::byte> buffer(MAX_MESSAGE_BYTES);
                    const ReceivedMessage received = reader->receive(buffer, 5s);
                    conversation(*reader, std::get<HandshakeRequest>(decode_message({ buffer.data(), received.bytes })));
                }
                catch (const std::exception&)
                {
                    // The reader's side of the test fails.
                }
            });
    }

    ~StandIn()
    {
        m_thread.join();
    }

    PluginLocation location() const
    {
        return { .endpoint = m_path };
    }

    // The answer a real plugin gives to a request for the stereo group.
    static void answer(MessageSocket& reader, const HostPool& left, const HostPool& right)
    {
        MessageEncoder encoder;
        HandshakeResponse response;
        response.groups.push_back({ stereo_group(), {} });
        const std::array<int, 2> descriptors = { left.descriptor(), right.descriptor() };
        reader.send(encoder.encode(response), descriptors);
    }

private:
    std::optional<TemporaryDirectory> m_directory;
    std::string m_path;
    MessageListener m_listener;
    std::thread m_thread;
};

// What the reader does with an answer to a request for "head", built by `respond`.
void open_against(const std::function<void(MessageSocket&)>& respond)
{
    StandIn plugin([&](MessageSocket& reader, const HandshakeRequest&) { respond(reader); });
    CameraReader::open("head", {}, plugin.location());
}

std::span<const std::byte> bytes_of(const flatbuffers::FlatBufferBuilder& builder)
{
    return { reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() };
}

} // namespace

TEST_CASE("a reader lists what a plugin publishes", "[unit][camera]")
{
    Served served;
    const std::vector<GroupListing> groups = CameraReader::list({ .endpoint = served.endpoint_path });
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].group.id == "head");
    REQUIRE(groups[0].unsupported.empty());
    REQUIRE(groups[0].group.session_id != 0);
    REQUIRE(groups[0].group.streams.size() == 2);
    REQUIRE(groups[0].group.streams[0].id == "head/left");
    REQUIRE(groups[0].endpoint == served.endpoint_path);
}

TEST_CASE("a reader gets each frame set with its pixels and what is known about it", "[unit][camera]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
    REQUIRE(reader.group().id == "head");
    REQUIRE(reader.group().streams.size() == 2);
    REQUIRE(reader.group().session_id == served.publisher.group_description(served.head).session_id);
    REQUIRE(reader.descriptor() >= 0);

    REQUIRE_FALSE(reader.next(10ms).has_value()); // nothing published yet

    served.publish(0x10, { .capture_ns = 111, .sdk_ready_ns = 222 });
    std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 1);
    REQUIRE(frames->skipped() == 0);
    REQUIRE(frames->image_count() == 2);
    REQUIRE(frames->timeline().capture_ns == 111);
    REQUIRE(frames->timeline().sdk_ready_ns == 222);
    REQUIRE(frames->timeline().sent_ns >= frames->timeline().published_ns);

    const ImageView left = frames->image(0);
    const ImageView right = frames->image(1);
    REQUIRE(left.stream->id == "head/left");
    REQUIRE(left.info.device_frame_id == 1000 + 0x10);
    REQUIRE(left.data.size() == SLOT_BYTES);
    REQUIRE(left.data.front() == std::byte{ 0x10 });
    REQUIRE(right.data.back() == std::byte{ 0x11 });

    // A plane runs from its first row to the end of its last, without the last row's padding.
    REQUIRE(left.plane(0).size() == STRIDE * (HEIGHT - 1) + WIDTH);
    REQUIRE(left.plane(0).data() == left.data.data());
    REQUIRE_THROWS_AS(left.plane(1), std::out_of_range);
    REQUIRE_THROWS_AS(frames->image(2), std::out_of_range);
}

TEST_CASE("a held frame set is not rewritten and the next one is the newest", "[unit][camera]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });

    served.publish(1);
    std::optional<FrameSet> held = reader.next(2s);
    REQUIRE(held.has_value());

    // Capture runs through the pool several times while the reader holds on.
    for (std::uint8_t value = 2; value <= 3 * SLOTS; ++value)
    {
        served.publish(value);
    }
    REQUIRE(first_pixel(*held, 0) == std::byte{ 1 });
    REQUIRE(first_pixel(*held, 1) == std::byte{ 2 });
    REQUIRE(served.publisher.stats(served.head).no_free_slot == 0);

    held.reset(); // gives it back
    std::optional<FrameSet> newest = reader.next(2s);
    REQUIRE(newest.has_value());
    REQUIRE(newest->sequence() == 3 * SLOTS);
    REQUIRE(newest->skipped() == 3 * SLOTS - 2);
    REQUIRE(first_pixel(*newest, 0) == std::byte{ 3 * SLOTS });

    // release() gives a frame set back before it goes out of scope.
    newest->release();
    REQUIRE_THROWS_AS(newest->image(0), std::logic_error);
    served.publish(99);
    REQUIRE(reader.next(2s)->sequence() == 3 * SLOTS + 1);
}

TEST_CASE("a reader that was busy gets what the camera saw last", "[unit][camera][newest]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
    served.publish(1);
    REQUIRE(reader.next(2s)->sequence() == 1); // taken and given back at once

    // The reader does something else for a while, holding nothing and not asking. The camera
    // goes on; nothing is put aside for the reader.
    std::int64_t last_published_ns = 0;
    for (std::uint8_t value = 2; value <= 12; ++value)
    {
        std::this_thread::sleep_for(5ms);
        last_published_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        served.publish(value);
    }
    REQUIRE(served.publisher.stats(served.head).events_sent == 1);

    // It is handed the last frame set published, not the one that was newest when it stopped.
    const std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 12);
    REQUIRE(frames->skipped() == 10);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 12 });
    REQUIRE(frames->timeline().published_ns >= last_published_ns);
    REQUIRE(served.publisher.stats(served.head).events_sent == 2);
}

TEST_CASE("a frame set that arrived after its reader stopped waiting is exchanged for the newest",
          "[unit][camera][newest]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });

    served.publish(1);
    REQUIRE(reader.next(2s)->sequence() == 1);

    // The reader asks for the next one and stops waiting before it is published.
    REQUIRE_FALSE(reader.next(10ms).has_value());

    // The camera then publishes: the first frame set answers the reader's request and waits in
    // its connection. Two more follow while the reader is away.
    served.publish(2);
    served.wait_until_sent(2);
    served.publish(3);
    served.publish(4);
    REQUIRE(served.publisher.stats(served.head).events_sent == 2);

    // Back again, the reader does not hand out the one that waited: it gives it back and is
    // sent the newest.
    const std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 4);
    REQUIRE(frames->skipped() == 2);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 4 });
    REQUIRE(served.publisher.stats(served.head).events_sent == 3);
}

TEST_CASE("a frame set that waited is handed out when nothing newer exists", "[unit][camera][newest]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
    REQUIRE_FALSE(reader.next(10ms).has_value());
    served.publish(7);
    served.wait_until_sent(1);
    REQUIRE(served.publisher.stats(served.head).events_sent == 1);

    // The exchange finds the same frame set still the newest, and the reader gets it: asking
    // again costs a round trip, never a frame.
    const std::optional<FrameSet> frames = reader.next(0ms);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 1);
    REQUIRE(frames->skipped() == 0);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 7 });
    REQUIRE(served.publisher.stats(served.head).events_sent == 2);
}

TEST_CASE("an in-order reader gets every frame set", "[unit][camera]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", { .window = 3 }, { .endpoint = served.endpoint_path });

    served.publish(1);
    served.publish(2);
    served.publish(3);
    std::vector<FrameSet> held;
    for (std::uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        std::optional<FrameSet> frames = reader.next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->sequence() == sequence);
        REQUIRE(frames->skipped() == 0);
        held.push_back(std::move(*frames));
    }
    // The window is full: three are held together, all three are still readable, and asking
    // for a fourth says what is wrong instead of waiting for something that cannot come.
    served.publish(4);
    REQUIRE_THROWS_WITH(reader.next(2s), ContainsSubstring("holds 3 frame set(s), which is its whole window"));
    REQUIRE_THROWS_AS(reader.next(2s), CameraWindowExhausted);
    REQUIRE(first_pixel(held[0], 0) == std::byte{ 1 });
    REQUIRE(first_pixel(held[2], 0) == std::byte{ 3 });

    held.erase(held.begin());
    REQUIRE(reader.next(2s)->sequence() == 4);
}

TEST_CASE("a newest-order reader with a larger window holds several frame sets at once", "[unit][camera]")
{
    Served served;
    CameraReader reader = CameraReader::open(
        "head", { .window = 3, .order = DeliveryOrder::NEWEST }, { .endpoint = served.endpoint_path });

    served.publish(1);
    served.publish(2);
    served.publish(3);
    // It gets the newest of the three, and nothing of the two before it.
    std::optional<FrameSet> first = reader.next(2s);
    REQUIRE(first.has_value());
    REQUIRE(first->sequence() == 3);
    REQUIRE(first_pixel(*first, 0) == std::byte{ 3 });

    // While it holds that one it can take two more.
    REQUIRE_FALSE(reader.next(10ms).has_value()); // nothing newer yet
    served.publish(4);
    std::optional<FrameSet> second = reader.next(2s);
    REQUIRE(second.has_value());
    REQUIRE(second->sequence() == 4);
    served.publish(5);
    served.publish(6);
    std::optional<FrameSet> third = reader.next(2s);
    REQUIRE(third.has_value());
    REQUIRE(third->sequence() == 6);
    REQUIRE(third->skipped() == 1);
    REQUIRE(first_pixel(*first, 0) == std::byte{ 3 });
    REQUIRE(first_pixel(*second, 0) == std::byte{ 4 });

    // Three held is the window.
    REQUIRE_THROWS_AS(reader.next(10ms), CameraWindowExhausted);
    second.reset();
    served.publish(7);
    REQUIRE(reader.next(2s)->sequence() == 7);
}

TEST_CASE("a frame set can be given back after its reader is gone", "[unit][camera]")
{
    Served served;
    std::optional<FrameSet> frames;
    {
        CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
        served.publish(5);
        frames = reader.next(2s);
        REQUIRE(frames.has_value());
        CameraReader moved = std::move(reader);
        REQUIRE(moved.group().id == "head");
    }
    // The frame set is still this reader's: its slots are not reused while the camera runs
    // through the pool, and the pixels stay as they were.
    for (std::uint8_t value = 0; value < 3 * SLOTS; ++value)
    {
        served.publish(0x80 + value);
    }
    REQUIRE(served.publisher.stats(served.head).no_free_slot == 0);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 5 });
    REQUIRE(served.endpoint.reader_count() == 1);

    // Giving it back is what ends the connection.
    frames.reset();
    for (int i = 0; i < 2000 && served.endpoint.reader_count() != 0; ++i)
    {
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(served.endpoint.reader_count() == 0);
}

TEST_CASE("a reader says why it could not be opened", "[unit][camera]")
{
    TemporaryDirectory directory;
    REQUIRE_THROWS_WITH(CameraReader::open("head", {}, { .endpoint = directory.socket_path() }),
                        ContainsSubstring("no camera plugin answers on '" + directory.socket_path() + "'") &&
                            ContainsSubstring("Start the camera plugin for your camera through the plugin manager"));
    REQUIRE_THROWS_AS(CameraReader::list({ .endpoint = directory.socket_path() }), CameraReaderError);

    Served served;
    REQUIRE_THROWS_WITH(
        CameraReader::open("chest", {}, { .endpoint = served.endpoint_path }), ContainsSubstring("no group 'chest'"));
    REQUIRE_THROWS_WITH(CameraReader::open("head", { .window = SLOTS }, { .endpoint = served.endpoint_path }),
                        ContainsSubstring("does not fit the pool"));

    // A refusal leaves nothing behind: the pool still has room for a reader that fits.
    REQUIRE_NOTHROW(CameraReader::open("head", { .window = 2 }, { .endpoint = served.endpoint_path }));
}

TEST_CASE("a reader finds each group on the plugin that publishes it", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    Served head_plugin("head", default_endpoint_path("synthetic"));
    Served wrist_plugin("wrist", default_endpoint_path("second"));

    // No location: the reader asks the plugins of this user and goes to the one with the group.
    CameraReader head = CameraReader::open("head");
    CameraReader wrist = CameraReader::open("wrist");
    REQUIRE(head.group().id == "head");
    REQUIRE(wrist.group().id == "wrist");

    // Each reads what its own plugin publishes.
    head_plugin.publish(0x10);
    wrist_plugin.publish(0x20);
    const std::optional<FrameSet> from_head = head.next(2s);
    const std::optional<FrameSet> from_wrist = wrist.next(2s);
    REQUIRE(from_head.has_value());
    REQUIRE(from_wrist.has_value());
    REQUIRE(first_pixel(*from_head, 0) == std::byte{ 0x10 });
    REQUIRE(first_pixel(*from_wrist, 0) == std::byte{ 0x20 });

    // Listing gives the groups of every plugin, in the order of the endpoints, each with the
    // endpoint that publishes it.
    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "wrist");
    REQUIRE(groups[0].endpoint == default_endpoint_path("second"));
    REQUIRE(groups[0].group.streams.size() == 2);
    REQUIRE(groups[1].group.id == "head");
    REQUIRE(groups[1].endpoint == default_endpoint_path("synthetic"));
}

TEST_CASE("a group that two plugins publish is not opened by guessing", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    Served first("head", default_endpoint_path("first"));
    Served second("head", default_endpoint_path("second"));

    REQUIRE_THROWS_WITH(CameraReader::open("head"),
                        ContainsSubstring("camera group 'head' is published by more than one camera plugin") &&
                            ContainsSubstring("'" + first.endpoint_path + "'") &&
                            ContainsSubstring("'" + second.endpoint_path + "'"));
    REQUIRE_THROWS_AS(CameraReader::open("head"), CameraReaderError);
    REQUIRE(CameraReader::list().size() == 2);

    // Told which plugin, the reader opens the group there.
    CameraReader reader = CameraReader::open("head", {}, { .instance = "second" });
    first.publish(0x10);
    second.publish(0x20);
    const std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 0x20 });
}

TEST_CASE("a reader that finds no plugin for its group says where it looked and what is there", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    const std::string how_to_start =
        "Start the camera plugin for your camera through the plugin manager, for example camera_synthetic";

    // No plugin runs. That is nothing to list, and not an error until a group is asked for.
    REQUIRE(CameraReader::list().empty());
    REQUIRE_THROWS_WITH(
        CameraReader::open("head"),
        ContainsSubstring("no camera plugin with an endpoint in '" + endpoints.path() + "' publishes group 'head'") &&
            ContainsSubstring("No camera plugin is running there") && ContainsSubstring(how_to_start));
    REQUIRE_THROWS_AS(CameraReader::open("head"), CameraReaderError);

    // Plugins run, with other groups: the message says which, and where each is.
    Served wrist_plugin("wrist", default_endpoint_path("synthetic"));
    Served chest_plugin("chest", default_endpoint_path("second"));
    REQUIRE_THROWS_WITH(
        CameraReader::open("head"), ContainsSubstring("in '" + endpoints.path() + "' publishes group 'head'") &&
                                        ContainsSubstring("'wrist' by '" + wrist_plugin.endpoint_path + "'") &&
                                        ContainsSubstring("'chest' by '" + chest_plugin.endpoint_path + "'") &&
                                        !ContainsSubstring("No camera plugin is running") &&
                                        !ContainsSubstring("Passed over") && ContainsSubstring(how_to_start));
}

TEST_CASE("a reader that cannot look through the user's endpoints says so", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    // Where the endpoints would be, there is something that is not a directory.
    std::filesystem::remove(endpoints.path());
    std::ofstream(endpoints.path()) << "not a directory";

    REQUIRE_THROWS_WITH(CameraReader::list(), ContainsSubstring("cannot look for camera plugins"));
    REQUIRE_THROWS_AS(CameraReader::list(), CameraReaderError);
    REQUIRE_THROWS_AS(CameraReader::open("head"), CameraReaderError);
}

TEST_CASE("a plugin that is still starting is passed over at once", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    Served served("head", default_endpoint_path("synthetic"));

    // A plugin that has taken its endpoint and does not serve it yet, as while it opens its
    // cameras: a connection to it is refused.
    Publisher publisher;
    publisher.add_group(stereo_group("wrist"));
    Endpoint starting(publisher, default_endpoint_path("starting"));
    REQUIRE(default_endpoint_paths().size() == 2);

    const auto before = std::chrono::steady_clock::now();
    CameraReader reader = CameraReader::open("head");
    // Well under the two seconds a plugin gets that a reader was pointed at. That this one was
    // refused, and not asked and waited for, is what the checks below show.
    REQUIRE(std::chrono::steady_clock::now() - before < 1s);
    REQUIRE(reader.group().id == "head");

    // For a reader it is not there yet: not listed, and not named as a plugin that failed to answer.
    REQUIRE(CameraReader::list().size() == 1);
    REQUIRE_THROWS_WITH(CameraReader::open("wrist"), ContainsSubstring("'head' by '" + served.endpoint_path + "'") &&
                                                         !ContainsSubstring(default_endpoint_path("starting")));

    // A reader that is pointed at it is told that nothing answers, instead of waiting for it.
    REQUIRE_THROWS_WITH(CameraReader::open("wrist", {}, { .instance = "starting" }),
                        ContainsSubstring("no camera plugin answers on '" + default_endpoint_path("starting") + "'"));

    // Once it serves, readers find what it publishes.
    starting.start();
    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].group.id == "wrist");
    REQUIRE(groups[0].endpoint == default_endpoint_path("starting"));
}

TEST_CASE("a socket file left by a plugin that was killed is passed over", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    leave_socket_file(default_endpoint_path("gone"));
    REQUIRE(default_endpoint_paths() == std::vector<std::string>{ default_endpoint_path("gone") });

    // Alone, it is no plugin at all.
    REQUIRE(CameraReader::list().empty());
    REQUIRE_THROWS_WITH(CameraReader::open("head"), ContainsSubstring("No camera plugin is running there"));

    // Next to a plugin that runs, it does not keep the reader from that one.
    Served served("head", default_endpoint_path("synthetic"));
    CameraReader reader = CameraReader::open("head");
    served.publish(0x10);
    REQUIRE(reader.next(2s).has_value());
    REQUIRE(CameraReader::list().size() == 1);
    REQUIRE_THROWS_WITH(CameraReader::open("wrist"), ContainsSubstring("'head' by '" + served.endpoint_path + "'") &&
                                                         !ContainsSubstring(default_endpoint_path("gone")));
}

TEST_CASE("a plugin that does not answer keeps a reader from the others only briefly", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    // Connections to it are taken into its queue, and nobody answers them.
    MessageListener silent(default_endpoint_path("silent"));
    silent.listen();
    Served served("head", default_endpoint_path("synthetic"));

    // The reader gives it a short time to say what it publishes, then goes on to the next.
    auto before = std::chrono::steady_clock::now();
    CameraReader reader = CameraReader::open("head");
    auto took = std::chrono::steady_clock::now() - before;
    REQUIRE(took >= 400ms);
    REQUIRE(took < 2s); // less than a plugin gets that a reader was pointed at
    served.publish(0x10);
    REQUIRE(reader.next(2s).has_value());

    // Listing passes it over too, and gives what the others publish.
    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].endpoint == served.endpoint_path);

    // A reader that does not find its group is told which plugin gave no answer.
    REQUIRE_THROWS_WITH(CameraReader::open("wrist"),
                        ContainsSubstring("'head' by '" + served.endpoint_path + "'") &&
                            ContainsSubstring("Passed over: '" + default_endpoint_path("silent") +
                                              "' (the camera plugin did not answer the reader's handshake)"));

    // Pointed at that plugin, a reader waits for it as long as for any plugin it is told to use.
    before = std::chrono::steady_clock::now();
    REQUIRE_THROWS_WITH(
        CameraReader::list({ .instance = "silent" }), ContainsSubstring("did not answer the reader's handshake"));
    took = std::chrono::steady_clock::now() - before;
    REQUIRE(took >= 1500ms);
}

TEST_CASE("a plugin that has stopped accepting keeps no reader waiting", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    // A plugin that listens and never accepts: every connection stays in its queue, also after
    // its reader has gone, and once the queue is full a connect to it is refused.
    const std::string stuck = default_endpoint_path("stuck");
    MessageListener listener(stuck);
    listener.listen();
    bool full = false;
    for (int attempt = 0; attempt < 200000 && !full; ++attempt)
    {
        try
        {
            MessageSocket::connect(stuck);
        }
        catch (const std::system_error&)
        {
            full = true;
        }
    }
    REQUIRE(full);
    Served served("head", default_endpoint_path("synthetic"));

    // Asked first (its file name sorts first), it turns the reader away at once, so the reader
    // does not even wait the short time it gives a plugin to say what it publishes.
    const auto before = std::chrono::steady_clock::now();
    CameraReader reader = CameraReader::open("head");
    REQUIRE(std::chrono::steady_clock::now() - before < 400ms);
    served.publish(0x10);
    REQUIRE(reader.next(2s).has_value());

    const std::vector<GroupListing> groups = CameraReader::list();
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].endpoint == served.endpoint_path);

    // A reader that does not find its group is told that plugin was passed over.
    REQUIRE_THROWS_WITH(CameraReader::open("wrist"), ContainsSubstring("Passed over: '" + stuck + "'"));
}

TEST_CASE("a plugin that answers with anything but what it publishes does not keep a reader from the others",
          "[unit][camera]")
{
    const EndpointDirectory endpoints;
    Served served("head", default_endpoint_path("synthetic"));
    // Asked before the plugin that serves: its file name sorts first.
    const std::string other = default_endpoint_path("other");
    MessageEncoder encoder;
    const StandIn::Conversation refuse = [&](MessageSocket& reader, const HandshakeRequest&)
    { reader.send(encoder.encode(HandshakeResponse{ .error = "not for this reader" })); };
    const StandIn::Conversation leave = [](MessageSocket&, const HandshakeRequest&) {};

    // A stand-in takes one reader, so every time a reader looks there is a new one.
    {
        // A plugin that refuses the reader, as one of another protocol version does: the camera
        // of the other plugin is found and read all the same.
        StandIn plugin(refuse, other);
        CameraReader reader = CameraReader::open("head");
        served.publish(0x10);
        REQUIRE(reader.next(2s).has_value());
    }
    {
        // A reader that does not find its group is told of that plugin, in the plugin's words.
        StandIn plugin(refuse, other);
        REQUIRE_THROWS_WITH(
            CameraReader::open("wrist"), ContainsSubstring("'head' by '" + served.endpoint_path + "'") &&
                                             ContainsSubstring("Passed over: '" + other + "' (not for this reader)"));
    }
    {
        // A plugin that goes away while it is asked.
        StandIn plugin(leave, other);
        REQUIRE_THROWS_WITH(CameraReader::open("wrist"),
                            ContainsSubstring("Passed over: '" + other + "' (the camera plugin closed the connection)"));
    }
    {
        // Listing leaves such a plugin out and gives what the others publish.
        StandIn plugin(leave, other);
        const std::vector<GroupListing> groups = CameraReader::list();
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].endpoint == served.endpoint_path);
    }
}

TEST_CASE("a reader that is pointed at a plugin asks that plugin only", "[unit][camera]")
{
    const EndpointDirectory endpoints;
    Served head_plugin("head", default_endpoint_path("synthetic"));
    Served wrist_plugin("wrist", default_endpoint_path("second"));

    // By its instance and by its socket path.
    for (const PluginLocation& plugin :
         { PluginLocation{ .instance = "synthetic" }, PluginLocation{ .endpoint = head_plugin.endpoint_path } })
    {
        const std::vector<GroupListing> groups = CameraReader::list(plugin);
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].group.id == "head");
        REQUIRE(groups[0].endpoint == head_plugin.endpoint_path);
        REQUIRE(CameraReader::open("head", {}, plugin).group().id == "head");

        // A group of the other plugin is not looked for there: the answer is this plugin's own.
        REQUIRE_THROWS_WITH(CameraReader::open("wrist", {}, plugin),
                            ContainsSubstring("this camera plugin publishes no group 'wrist'"));
    }

    // A plugin that is not there is an error, also for listing; the others are not tried.
    const PluginLocation absent{ .instance = "absent" };
    const std::string nobody = "no camera plugin answers on '" + default_endpoint_path("absent") + "'";
    REQUIRE_THROWS_WITH(CameraReader::open("head", {}, absent), ContainsSubstring(nobody));
    REQUIRE_THROWS_WITH(CameraReader::list(absent), ContainsSubstring(nobody));

    REQUIRE_THROWS_WITH(CameraReader::list({ .endpoint = head_plugin.endpoint_path, .instance = "synthetic" }),
                        ContainsSubstring("by its endpoint or by its instance, not both"));
}

TEST_CASE("a reader checks that the answer is what it asked for", "[unit][camera][validation]")
{
    HostPool left(SLOTS, SLOT_BYTES, "test_left");
    HostPool right(SLOTS, SLOT_BYTES, "test_right");
    const std::array<int, 2> pools = { left.descriptor(), right.descriptor() };
    MessageEncoder encoder;
    const auto answer_with = [&](HandshakeResponse response, std::span<const int> descriptors)
    {
        return [&encoder, response = std::move(response), descriptors](MessageSocket& reader)
        { reader.send(encoder.encode(response), descriptors); };
    };
    const auto describing = [](const GroupDescription& group)
    {
        HandshakeResponse response;
        response.groups.push_back({ group, {} });
        return response;
    };

    // The right answer, to show the stand-in is one a reader accepts.
    REQUIRE_NOTHROW(open_against(answer_with(describing(stereo_group()), pools)));

    // Another group than the one asked for, or more than one.
    REQUIRE_THROWS_WITH(
        open_against(answer_with(describing(stereo_group("wrist")), pools)), ContainsSubstring("with group 'wrist'"));
    HandshakeResponse two = describing(stereo_group());
    two.groups.push_back({ stereo_group("wrist"), {} });
    REQUIRE_THROWS_WITH(open_against(answer_with(two, pools)), ContainsSubstring("with 2 groups"));
    REQUIRE_THROWS_WITH(open_against(answer_with(HandshakeResponse{}, {})), ContainsSubstring("with 0 groups"));

    // A protocol version this reader does not speak.
    HandshakeResponse newer = describing(stereo_group());
    newer.protocol_version = PROTOCOL_VERSION + 1;
    REQUIRE_THROWS_WITH(open_against(answer_with(newer, pools)), ContainsSubstring("answered in protocol version"));
    HandshakeResponse older = describing(stereo_group());
    older.protocol_version = OLDEST_PROTOCOL_VERSION - 1;
    REQUIRE_THROWS_WITH(open_against(answer_with(older, pools)), ContainsSubstring("answered in protocol version"));

    // A description a view could run off the end of a pool with: rows longer than the slot
    // holds, a plane missing, sizes whose product wraps, a slot no pool can have.
    GroupDescription short_rows = stereo_group();
    short_rows.streams[1].planes[0].stride_bytes = WIDTH - 1;
    REQUIRE_THROWS_WITH(
        open_against(answer_with(describing(short_rows), pools)), ContainsSubstring("in a way that cannot be read"));
    GroupDescription tall = stereo_group();
    tall.streams[0].height = HEIGHT + 1;
    REQUIRE_THROWS_WITH(
        open_against(answer_with(describing(tall), pools)), ContainsSubstring("do not lie inside the slot"));
    GroupDescription no_plane = stereo_group();
    no_plane.streams[0].plane_count = 0;
    REQUIRE_THROWS_WITH(open_against(answer_with(describing(no_plane), pools)), ContainsSubstring("every plane"));
    GroupDescription wraps = stereo_group();
    wraps.streams[0].width = 0xFFFFFFFF;
    wraps.streams[0].height = 0xFFFFFFFF;
    wraps.streams[0].planes[0] = { .offset_bytes = 0xFFFFFFFF, .stride_bytes = 0xFFFFFFFF };
    REQUIRE_THROWS_WITH(
        open_against(answer_with(describing(wraps), pools)), ContainsSubstring("in a way that cannot be read"));
    GroupDescription no_slots = stereo_group();
    no_slots.streams[0].slot_count = 0;
    REQUIRE_THROWS_WITH(open_against(answer_with(describing(no_slots), pools)), ContainsSubstring("slots"));

    // Fewer pools than streams, and a pool smaller than its description says.
    REQUIRE_THROWS_WITH(open_against(answer_with(describing(stereo_group()), std::span<const int>(pools).first(1))),
                        ContainsSubstring("one pool per stream"));
    HostPool small(SLOTS / 2, SLOT_BYTES, "test_small");
    const std::array<int, 2> mismatched = { left.descriptor(), small.descriptor() };
    REQUIRE_THROWS_WITH(
        open_against(answer_with(describing(stereo_group()), mismatched)), ContainsSubstring("cannot map the pools"));

    // A frame set before the answer.
    REQUIRE_THROWS_WITH(open_against([&](MessageSocket& reader) { reader.send(encoder.encode(FrameEvent{})); }),
                        ContainsSubstring("before it answered"));

    // A refusal is passed on in the plugin's words.
    REQUIRE_THROWS_WITH(open_against(answer_with(HandshakeResponse{ .error = "the pool is full" }, {})),
                        ContainsSubstring("the pool is full"));
}

TEST_CASE("a group a newer plugin publishes in a form this build cannot read is listed and not opened",
          "[unit][camera][validation]")
{
    // What a newer plugin sends: one camera this build reads and one in a kind of memory added
    // after it. The codec of this build never writes such a group, so it is built by hand.
    const auto answer = [](MessageSocket& reader, const std::vector<std::string>& ids, std::uint32_t memory_of_last)
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<core::CameraGroup>> groups;
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            const core::CameraPlaneLayout plane(0, STRIDE);
            const auto stream = core::CreateCameraStream(
                builder, builder.CreateString(ids[i] + "/0"), core::CameraPixelFormat_Gray8,
                static_cast<core::CameraMemoryKind>(i + 1 == ids.size() ? memory_of_last : core::CameraMemoryKind_Host),
                WIDTH, HEIGHT, builder.CreateVectorOfStructs(&plane, 1), SLOT_BYTES, SLOTS, 60.0);
            groups.push_back(
                core::CreateCameraGroup(builder, builder.CreateString(ids[i]), builder.CreateVector(&stream, 1), 5));
        }
        const auto response =
            core::CreateCameraHandshakeResponse(builder, PROTOCOL_VERSION, 0, builder.CreateVector(groups));
        builder.Finish(
            core::CreateCameraMessage(builder, core::CameraMessageBody_CameraHandshakeResponse, response.Union()));
        HostPool pool(SLOTS, SLOT_BYTES, "test_pool");
        const std::array<int, 1> descriptors = { pool.descriptor() };
        reader.send(bytes_of(builder), ids.size() == 1 ? std::span<const int>(descriptors) : std::span<const int>());
    };
    const std::uint32_t newer_memory = core::CameraMemoryKind_MAX + 1;

    {
        StandIn plugin(
            [&](MessageSocket& reader, const HandshakeRequest& request)
            {
                if (request.group.empty())
                {
                    answer(reader, { "head", "wrist" }, newer_memory);
                }
            });
        const std::vector<GroupListing> groups = CameraReader::list(plugin.location());
        REQUIRE(groups.size() == 2);
        REQUIRE(groups[0].group.id == "head");
        REQUIRE(groups[0].unsupported.empty());
        REQUIRE(groups[0].group.streams.size() == 1);
        REQUIRE(groups[1].group.id == "wrist");
        REQUIRE(groups[1].group.streams.empty());
        REQUIRE_THAT(groups[1].unsupported, ContainsSubstring("kind of memory this build does not know"));
    }
    {
        StandIn plugin([&](MessageSocket& reader, const HandshakeRequest&)
                       { answer(reader, { "wrist" }, newer_memory); });
        REQUIRE_THROWS_WITH(
            CameraReader::open("wrist", {}, plugin.location()), ContainsSubstring("cannot be read by this build"));
    }
    {
        // The group it does read opens as usual.
        StandIn plugin([&](MessageSocket& reader, const HandshakeRequest&)
                       { answer(reader, { "head" }, core::CameraMemoryKind_Host); });
        REQUIRE(CameraReader::open("head", {}, plugin.location()).group().streams.size() == 1);
    }
}

TEST_CASE("a reader is told when the plugin is gone", "[unit][camera]")
{
    Served served;
    CameraReader reader = CameraReader::open(
        "head", { .window = 2, .order = DeliveryOrder::NEWEST }, { .endpoint = served.endpoint_path });
    served.publish(1);
    std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());

    served.endpoint.stop();
    REQUIRE_THROWS_AS(reader.next(2s), CameraDisconnected);
    // What the reader holds is still mapped, and giving it back to nobody is harmless.
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 1 });
    frames.reset();
}

TEST_CASE("a reader copes with what a plugin may send", "[unit][camera]")
{
    HostPool left(SLOTS, SLOT_BYTES, "test_left");
    HostPool right(SLOTS, SLOT_BYTES, "test_right");
    left.slot(2)[0] = std::byte{ 0x42 };

    FrameEvent good;
    good.lease_id = 7;
    good.sequence = 1;
    good.image_count = 2;
    good.images[0].slot = 2;
    FrameEvent bad_slot = good;
    bad_slot.lease_id = 8;
    bad_slot.sequence = 2;
    bad_slot.images[1].slot = SLOTS;

    std::vector<std::uint64_t> asked_after;
    std::vector<std::uint64_t> released;
    {
        StandIn plugin(
            [&](MessageSocket& reader, const HandshakeRequest&)
            {
                MessageEncoder encoder;
                std::vector<std::byte> buffer(MAX_MESSAGE_BYTES);
                const auto receive = [&] {
                    return decode_message({ buffer.data(), reader.receive(buffer, 5s).bytes });
                };
                StandIn::answer(reader, left, right);

                // The reader asks; a newer plugin says something for information first, which is
                // passed over, then sends the frame set.
                asked_after.push_back(std::get<AcquireMessage>(receive()).after_sequence);
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(core::CreateCameraMessage(
                    builder, static_cast<core::CameraMessageBody>(core::CameraMessageBody_MAX + 1),
                    core::CreateCameraRelease(builder, 1).Union(), true));
                reader.send(bytes_of(builder));
                reader.send(encoder.encode(good));
                released.push_back(std::get<ReleaseMessage>(receive()).lease_id);

                // A frame set that names a slot the pool does not have.
                asked_after.push_back(std::get<AcquireMessage>(receive()).after_sequence);
                reader.send(encoder.encode(bad_slot));
                released.push_back(std::get<ReleaseMessage>(receive()).lease_id);

                // An operation this reader's build does not know.
                asked_after.push_back(std::get<AcquireMessage>(receive()).after_sequence);
                builder.Clear();
                builder.Finish(core::CreateCameraMessage(
                    builder, static_cast<core::CameraMessageBody>(core::CameraMessageBody_MAX + 1),
                    core::CreateCameraRelease(builder, 1).Union(), false));
                reader.send(bytes_of(builder));
                (void)reader.receive(buffer, 5s); // until the reader goes
            });

        CameraReader reader = CameraReader::open("head", {}, plugin.location());
        std::optional<FrameSet> frames = reader.next(2s);
        REQUIRE(frames.has_value());
        REQUIRE(frames->sequence() == 1);
        REQUIRE(first_pixel(*frames, 0) == std::byte{ 0x42 });
        frames.reset();

        // Refused and given back; the reader goes on.
        REQUIRE_THROWS_WITH(reader.next(2s), ContainsSubstring("does not match group 'head'"));
        // Not something to pass over: the reader says so instead of waiting on.
        REQUIRE_THROWS_WITH(reader.next(2s), ContainsSubstring("cannot act on"));
    }
    REQUIRE(asked_after == std::vector<std::uint64_t>{ 0, 1, 1 });
    REQUIRE(released == std::vector<std::uint64_t>{ 7, 8 });
}
