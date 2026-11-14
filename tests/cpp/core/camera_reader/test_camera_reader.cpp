// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_endpoint/endpoint.hpp>
#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <camera_protocol/messages.hpp>
#include <camera_reader/camera_reader.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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

GroupDescription stereo_group()
{
    GroupDescription group;
    group.id = "head";
    for (const char* eye : { "left", "right" })
    {
        StreamDescription stream;
        stream.id = std::string("head/") + eye;
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

// A plugin in miniature: a stereo group in host memory behind an endpoint.
struct Served
{
    TemporaryDirectory directory;
    std::string endpoint_path = directory.socket_path();
    Publisher publisher;
    GroupId head = publisher.add_group(stereo_group());
    HostPool left{ SLOTS, SLOT_BYTES, "test_left" };
    HostPool right{ SLOTS, SLOT_BYTES, "test_right" };
    Endpoint endpoint{ publisher, endpoint_path };

    Served()
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
};

std::byte first_pixel(const FrameSet& frames, std::size_t stream)
{
    return frames.image(stream).data.front();
}

} // namespace

TEST_CASE("a reader lists what a plugin publishes", "[unit][camera_reader]")
{
    Served served;
    const std::vector<GroupDescription> groups = CameraReader::list({ .endpoint = served.endpoint_path });
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].id == "head");
    REQUIRE(groups[0].streams.size() == 2);
    REQUIRE(groups[0].streams[0].id == "head/left");
}

TEST_CASE("a reader gets each frame set with its pixels and what is known about it", "[unit][camera_reader]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
    REQUIRE(reader.group().id == "head");
    REQUIRE(reader.group().streams.size() == 2);
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

TEST_CASE("a held frame set is not rewritten and the next one is the newest", "[unit][camera_reader]")
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

TEST_CASE("an in-order reader gets every frame set", "[unit][camera_reader]")
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
    // The window is full: three are held together, and all three are still readable.
    served.publish(4);
    REQUIRE_FALSE(reader.next(20ms).has_value());
    REQUIRE(first_pixel(held[0], 0) == std::byte{ 1 });
    REQUIRE(first_pixel(held[2], 0) == std::byte{ 3 });

    held.erase(held.begin());
    REQUIRE(reader.next(2s)->sequence() == 4);
}

TEST_CASE("a newest-order reader with a larger window skips to the newest waiting frame set", "[unit][camera_reader]")
{
    Served served;
    CameraReader reader = CameraReader::open(
        "head", { .window = 3, .order = DeliveryOrder::NEWEST }, { .endpoint = served.endpoint_path });

    served.publish(1);
    served.publish(2);
    served.publish(3);
    // Three events are waiting; the reader takes the last and gives the others back.
    std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 3);
    REQUIRE(frames->skipped() == 2);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 3 });

    // Giving those two back made room in the window again, with the third still held: once the
    // plugin has seen the releases, a newly published frame set reaches the reader.
    std::optional<FrameSet> more;
    for (std::uint8_t value = 4; value < 200 && !more; ++value)
    {
        served.publish(value);
        more = reader.next(20ms);
    }
    REQUIRE(more.has_value());
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 3 });
}

TEST_CASE("a frame set can be given back after its reader is gone", "[unit][camera_reader]")
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
    // The pixels stay readable for as long as the frame set is held.
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 5 });
    frames.reset();
}

TEST_CASE("a reader says why it could not be opened", "[unit][camera_reader]")
{
    TemporaryDirectory directory;
    REQUIRE_THROWS_WITH(CameraReader::open("head", {}, { .endpoint = directory.socket_path() }),
                        ContainsSubstring("no camera plugin answers on"));
    REQUIRE_THROWS_AS(CameraReader::list({ .endpoint = directory.socket_path() }), CameraReaderError);

    Served served;
    REQUIRE_THROWS_WITH(
        CameraReader::open("chest", {}, { .endpoint = served.endpoint_path }), ContainsSubstring("no group 'chest'"));
    REQUIRE_THROWS_WITH(CameraReader::open("head", { .window = SLOTS }, { .endpoint = served.endpoint_path }),
                        ContainsSubstring("does not fit the pool"));

    // A refusal leaves nothing behind: the pool still has room for a reader that fits.
    REQUIRE_NOTHROW(CameraReader::open("head", { .window = 2 }, { .endpoint = served.endpoint_path }));
}

TEST_CASE("a reader is told when the plugin is gone", "[unit][camera_reader]")
{
    Served served;
    CameraReader reader = CameraReader::open("head", {}, { .endpoint = served.endpoint_path });
    served.publish(1);
    std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());

    served.endpoint.stop();
    REQUIRE_THROWS_AS(reader.next(2s), CameraDisconnected);
    // What the reader holds is still mapped, and giving it back to nobody is harmless.
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 1 });
    frames.reset();
}

TEST_CASE("a reader copes with what a plugin may send", "[unit][camera_reader]")
{
    // A stand-in plugin, to send what a real one does not send on demand.
    TemporaryDirectory directory;
    MessageListener listener(directory.socket_path());
    HostPool left(SLOTS, SLOT_BYTES, "test_left");
    HostPool right(SLOTS, SLOT_BYTES, "test_right");
    left.slot(2)[0] = std::byte{ 0x42 };

    FrameEvent early;
    early.lease_id = 7;
    early.sequence = 1;
    early.image_count = 2;
    early.images[0].slot = 2;
    FrameEvent bad_slot = early;
    bad_slot.lease_id = 8;
    bad_slot.sequence = 2;
    bad_slot.images[1].slot = SLOTS;

    std::vector<std::uint64_t> released;
    std::thread plugin(
        [&]
        {
            try
            {
                MessageEncoder encoder;
                std::vector<std::byte> buffer(MAX_MESSAGE_BYTES);
                std::optional<MessageSocket> reader = listener.accept(2s);
                if (!reader)
                {
                    return;
                }
                const auto receive = [&] {
                    return decode_message({ buffer.data(), reader->receive(buffer, 2s).bytes });
                };
                (void)receive(); // the reader's handshake

                // A frame set overtakes the answer to the handshake.
                reader->send(encoder.encode(early));
                const std::array<int, 2> descriptors = { left.descriptor(), right.descriptor() };
                reader->send(encoder.encode(HandshakeResponse{ .groups = { stereo_group() } }), descriptors);
                reader->send(encoder.encode(bad_slot));
                for (int i = 0; i < 2; ++i)
                {
                    released.push_back(std::get<ReleaseMessage>(receive()).lease_id);
                }
            }
            catch (const std::exception&)
            {
                // The checks on `released` below fail.
            }
        });

    CameraReader reader = CameraReader::open("head", {}, { .endpoint = directory.socket_path() });
    std::optional<FrameSet> frames = reader.next(2s);
    REQUIRE(frames.has_value());
    REQUIRE(frames->sequence() == 1);
    REQUIRE(first_pixel(*frames, 0) == std::byte{ 0x42 });
    frames.reset();

    // A frame set that names a slot the pool does not have is refused and given back.
    REQUIRE_THROWS_WITH(reader.next(2s), ContainsSubstring("does not match group 'head'"));
    plugin.join();
    REQUIRE(released == std::vector<std::uint64_t>{ 7, 8 });
}
