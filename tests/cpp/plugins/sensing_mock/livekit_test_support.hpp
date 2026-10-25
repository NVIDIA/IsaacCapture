// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A local LiveKit server, join tokens, and a subscriber that records what it receives, so the
// SENSING publisher can be checked against a real LiveKit service.

#pragma once

#include <livekit/livekit.h>
#include <netinet/in.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <sensing_livekit/livekit_session.hpp>
#include <sys/socket.h>
#include <sys/wait.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace livekit_test
{

constexpr const char* kApiKey = "devkey";
constexpr const char* kApiSecret = "secret";

inline std::string base64url(const unsigned char* data, size_t size)
{
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    for (size_t i = 0; i < size; i += 3)
    {
        const uint32_t n = (data[i] << 16) | ((i + 1 < size ? data[i + 1] : 0) << 8) | (i + 2 < size ? data[i + 2] : 0);
        out.push_back(alphabet[(n >> 18) & 63]);
        out.push_back(alphabet[(n >> 12) & 63]);
        if (i + 1 < size)
            out.push_back(alphabet[(n >> 6) & 63]);
        if (i + 2 < size)
            out.push_back(alphabet[n & 63]);
    }
    return out;
}

inline std::string base64url(const std::string& text)
{
    return base64url(reinterpret_cast<const unsigned char*>(text.data()), text.size());
}

/// An HS256 LiveKit access token for the local server.
inline std::string make_token(const std::string& identity,
                              const std::string& room,
                              bool can_publish,
                              const std::string& secret = kApiSecret)
{
    const auto now =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string header = R"({"alg":"HS256","typ":"JWT"})";
    const std::string claims = std::string(R"({"iss":")") + kApiKey + R"(","sub":")" + identity + R"(","name":")" +
                               identity + R"(","nbf":)" + std::to_string(now) + R"(,"exp":)" +
                               std::to_string(now + 3600) + R"(,"video":{"roomJoin":true,"room":")" + room +
                               R"(","canPublish":)" + (can_publish ? "true" : "false") + R"(,"canSubscribe":)" +
                               (can_publish ? "false" : "true") + "}}";
    const std::string signing = base64url(header) + "." + base64url(claims);
    unsigned char mac[EVP_MAX_MD_SIZE];
    unsigned int mac_size = 0;
    HMAC(EVP_sha256(), secret.data(), static_cast<int>(secret.size()),
         reinterpret_cast<const unsigned char*>(signing.data()), signing.size(), mac, &mac_size);
    return signing + "." + base64url(mac, mac_size);
}

/// A livekit-server child process on private ports. Tests skip when LIVEKIT_SERVER is not set.
class LocalServer
{
public:
    static const char* binary()
    {
        return std::getenv("LIVEKIT_SERVER");
    }

    LocalServer()
    {
        static std::atomic<int> counter{ 0 };
        m_port = 21000 + static_cast<int>(::getpid() % 2000) * 4 + counter++ * 4000 % 7000;
        const std::string config = "port: " + std::to_string(m_port) +
                                   "\nrtc:\n  tcp_port: " + std::to_string(m_port + 1) +
                                   "\n  udp_port: " + std::to_string(m_port + 2) + "\n  use_external_ip: false\n";
        const std::string keys = std::string(kApiKey) + ": " + kApiSecret;
        m_pid = ::fork();
        if (m_pid == 0)
        {
            // Child: silence the server's logs, then become it.
            if (FILE* devnull = std::fopen("/dev/null", "w"))
            {
                ::dup2(::fileno(devnull), 1);
                ::dup2(::fileno(devnull), 2);
            }
            ::execl(binary(), binary(), "--bind", "127.0.0.1", "--node-ip", "127.0.0.1", "--keys", keys.c_str(),
                    "--config-body", config.c_str(), static_cast<char*>(nullptr));
            ::_exit(127);
        }
        wait_until_listening();
    }

    ~LocalServer()
    {
        stop();
    }

    LocalServer(const LocalServer&) = delete;
    LocalServer& operator=(const LocalServer&) = delete;

    std::string url() const
    {
        return "ws://127.0.0.1:" + std::to_string(m_port);
    }

    /// Kills the server, as a crash or a network loss would.
    void stop()
    {
        if (m_pid > 0)
        {
            ::kill(m_pid, SIGKILL);
            ::waitpid(m_pid, nullptr, 0);
            m_pid = -1;
        }
    }

private:
    void wait_until_listening()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (std::chrono::steady_clock::now() < deadline)
        {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(static_cast<uint16_t>(m_port));
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool up = ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
            ::close(fd);
            if (up)
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        throw std::runtime_error("the local LiveKit server did not start");
    }

    pid_t m_pid = -1;
    int m_port = 0;
};

/// What a subscriber has received on one track.
struct TrackSeen
{
    uint64_t frames = 0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> last_rgba;
    std::vector<uint32_t> frame_ids;
    std::map<uint32_t, uint64_t> tsc_by_frame; ///< capture TSC carried as user data
    uint64_t with_user_timestamp = 0;
};

/// A second participant that subscribes to every video track and records it.
class Subscriber : public livekit::RoomDelegate
{
public:
    Subscriber(const std::string& url, const std::string& room)
    {
        m_room = std::make_unique<livekit::Room>();
        m_room->setDelegate(this);
        livekit::RoomOptions options;
        if (!m_room->connect(url, make_token("test-subscriber", room, false), options))
            throw std::runtime_error("the subscriber could not join the room");
    }

    ~Subscriber() override
    {
        // The SDK callback takes the mutex, so claim the streams and readers under it, then close and join
        // with it released.
        std::vector<std::shared_ptr<livekit::VideoStream>> streams;
        std::vector<std::thread> readers;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closing = true;
            streams.swap(m_streams);
            readers.swap(m_readers);
        }
        for (auto& stream : streams)
            stream->close();
        for (auto& reader : readers)
            reader.join();
        m_room->disconnect();
        m_room.reset();
    }

    void onTrackSubscribed(livekit::Room&, const livekit::TrackSubscribedEvent& event) override
    {
        if (!event.track || event.track->kind() != livekit::TrackKind::KIND_VIDEO)
            return;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_closing)
            return;
        livekit::VideoStream::Options options;
        options.capacity = 16;
        auto stream = livekit::VideoStream::fromTrack(event.track, options);
        const std::string name = event.track->name();
        m_streams.push_back(stream);
        m_readers.emplace_back([this, stream, name] { read(stream, name); });
    }

    void onParticipantDisconnected(livekit::Room&, const livekit::ParticipantDisconnectedEvent&) override
    {
        ++m_participants_left;
    }

    /// True once every named track has delivered at least `frames` frames.
    bool wait_for_frames(const std::vector<std::string>& names, uint64_t frames, std::chrono::milliseconds limit)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_changed.wait_for(lock, limit,
                                  [&]
                                  {
                                      for (const auto& name : names)
                                      {
                                          auto it = m_seen.find(name);
                                          if (it == m_seen.end() || it->second.frames < frames)
                                              return false;
                                      }
                                      return true;
                                  });
    }

    TrackSeen seen(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_seen.find(name);
        return it == m_seen.end() ? TrackSeen{} : it->second;
    }

    std::set<std::string> track_names()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::set<std::string> names;
        for (const auto& entry : m_seen)
            names.insert(entry.first);
        return names;
    }

    int participants_left() const
    {
        return m_participants_left.load();
    }

private:
    void read(const std::shared_ptr<livekit::VideoStream>& stream, const std::string& name)
    {
        livekit::VideoFrameEvent event;
        while (stream->read(event))
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            TrackSeen& seen = m_seen[name];
            ++seen.frames;
            seen.width = event.frame.width();
            seen.height = event.frame.height();
            seen.last_rgba.assign(event.frame.data(), event.frame.data() + event.frame.dataSize());
            if (event.metadata)
            {
                if (event.metadata->frame_id)
                    seen.frame_ids.push_back(*event.metadata->frame_id);
                if (event.metadata->user_timestamp_us)
                    ++seen.with_user_timestamp;
                if (event.metadata->frame_id && event.metadata->user_data && event.metadata->user_data->size() == 8)
                {
                    uint64_t tsc = 0;
                    for (int i = 7; i >= 0; --i)
                        tsc = (tsc << 8) | (*event.metadata->user_data)[i];
                    seen.tsc_by_frame[*event.metadata->frame_id] = tsc;
                }
            }
            m_changed.notify_all();
        }
    }

    // Keeps the SDK up for as long as this subscriber's room exists, alongside the publisher's.
    plugins::sensing::LiveKitSdkReference m_sdk;
    std::unique_ptr<livekit::Room> m_room;
    std::mutex m_mutex;
    std::condition_variable m_changed;
    std::map<std::string, TrackSeen> m_seen;
    std::vector<std::shared_ptr<livekit::VideoStream>> m_streams;
    std::vector<std::thread> m_readers;
    std::atomic<int> m_participants_left{ 0 };
    bool m_closing = false;
};

} // namespace livekit_test
