// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/control_server.hpp"

#include "inc/sensing_control/websocket.hpp"

#include <arpa/inet.h>
#include <log_bridge/logger.hpp>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>

namespace plugins
{
namespace sensing
{

namespace
{

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.sensing.ControlServer");
}

constexpr std::size_t kMaxHandshakeBytes = 8 * 1024;
/// Room beyond one maximum message for the frame header, the mask and a following control frame.
constexpr std::size_t kFrameHeadroom = 4096;
constexpr std::size_t kMaxOutputBytes = 1 << 20;
/// Frames handled per client per poll round, so one chatty peer cannot hold the server thread.
constexpr int kMaxFramesPerRound = 16;
constexpr auto kHandshakeTimeout = std::chrono::seconds(5);

void set_non_blocking(int fd)
{
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
}

} // namespace

std::string read_secret_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot read " + path);
    std::string secret;
    std::getline(file, secret);
    const auto last = secret.find_last_not_of(" \t\r\n");
    secret.erase(last == std::string::npos ? 0 : last + 1);
    const auto first = secret.find_first_not_of(" \t");
    secret.erase(0, first == std::string::npos ? secret.size() : first);
    if (secret.empty())
        throw std::runtime_error(path + " is empty");
    return secret;
}

struct ControlServer::Client
{
    int fd = -1;
    bool open = false; ///< the handshake completed
    bool closing = false; ///< flush the output, then drop the connection
    std::string input;
    std::string output;
    std::string message; ///< a fragmented text message being assembled
    bool fragmenting = false; ///< a text message was started without its final fragment
    std::string peer;
    std::chrono::steady_clock::time_point accepted = std::chrono::steady_clock::now();
};

ControlServer::ControlServer(const ControlServerConfig& config, MessageHandler on_message)
    : m_config(config), m_on_message(std::move(on_message))
{
    if (m_config.token.empty())
        throw std::runtime_error("the management channel needs a token: it can redirect camera video");

    m_listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_listen_fd < 0)
        throw std::runtime_error(std::string("management channel: cannot create a socket: ") + std::strerror(errno));
    const int one = 1;
    ::setsockopt(m_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(m_config.port);
    if (::inet_pton(AF_INET, m_config.bind_address.c_str(), &address.sin_addr) != 1)
    {
        ::close(m_listen_fd);
        throw std::runtime_error("management channel: \"" + m_config.bind_address + "\" is not an IPv4 address");
    }
    if (::bind(m_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(m_listen_fd, 16) != 0)
    {
        const std::string why = std::strerror(errno);
        ::close(m_listen_fd);
        throw std::runtime_error("management channel: cannot listen on " + m_config.bind_address + ":" +
                                 std::to_string(m_config.port) + ": " + why);
    }
    socklen_t length = sizeof(address);
    ::getsockname(m_listen_fd, reinterpret_cast<sockaddr*>(&address), &length);
    m_port = ntohs(address.sin_port);
    set_non_blocking(m_listen_fd);
    m_wake_fd = ::eventfd(0, EFD_NONBLOCK);

    logger()->info("management channel listening on {}:{}", m_config.bind_address, m_port);
    m_thread = std::thread(&ControlServer::run, this);
}

ControlServer::~ControlServer()
{
    m_stop = true;
    wake();
    if (m_thread.joinable())
        m_thread.join();
    ::close(m_listen_fd);
    ::close(m_wake_fd);
}

void ControlServer::wake()
{
    const std::uint64_t one = 1;
    [[maybe_unused]] const ssize_t written = ::write(m_wake_fd, &one, sizeof(one));
}

void ControlServer::broadcast(const std::string& text)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Notifications are advisory: if nobody drains them, keep only the newest.
        if (m_broadcasts.size() >= 64)
            m_broadcasts.erase(m_broadcasts.begin());
        m_broadcasts.push_back(text);
    }
    wake();
}

void ControlServer::run()
{
    std::vector<std::unique_ptr<Client>> clients;

    const auto drop = [&](std::size_t index, const char* why)
    {
        logger()->info("management client {} disconnected: {}", clients[index]->peer, why);
        ::close(clients[index]->fd);
        clients.erase(clients.begin() + static_cast<std::ptrdiff_t>(index));
    };
    const auto refuse = [&](Client& client, const websocket::Handshake& handshake)
    {
        client.output += websocket::handshake_response(handshake);
        client.closing = true;
    };

    while (!m_stop)
    {
        std::vector<pollfd> fds;
        fds.push_back({ m_listen_fd, POLLIN, 0 });
        fds.push_back({ m_wake_fd, POLLIN, 0 });
        for (const auto& client : clients)
            fds.push_back({ client->fd, static_cast<short>(POLLIN | (client->output.empty() ? 0 : POLLOUT)), 0 });

        const int ready = ::poll(fds.data(), fds.size(), 500);
        if (ready < 0 && errno != EINTR)
        {
            logger()->error("management channel poll failed: {}", std::strerror(errno));
            break;
        }

        if (fds[1].revents & POLLIN)
        {
            std::uint64_t counter = 0;
            [[maybe_unused]] const ssize_t got = ::read(m_wake_fd, &counter, sizeof(counter));
            std::vector<std::string> pending;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                pending.swap(m_broadcasts);
            }
            for (const auto& text : pending)
                for (auto& client : clients)
                    if (client->open && !client->closing)
                        client->output += websocket::encode_frame(websocket::kText, text);
        }

        if (fds[0].revents & POLLIN)
        {
            while (true)
            {
                sockaddr_in peer{};
                socklen_t length = sizeof(peer);
                const int fd = ::accept(m_listen_fd, reinterpret_cast<sockaddr*>(&peer), &length);
                if (fd < 0)
                    break;
                if (clients.size() >= m_config.max_clients)
                {
                    ::close(fd);
                    continue;
                }
                set_non_blocking(fd);
                const int one = 1;
                ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                auto client = std::make_unique<Client>();
                client->fd = fd;
                char text[INET_ADDRSTRLEN] = {};
                ::inet_ntop(AF_INET, &peer.sin_addr, text, sizeof(text));
                client->peer = std::string(text) + ":" + std::to_string(ntohs(peer.sin_port));
                clients.push_back(std::move(client));
            }
        }

        // The clients that were polled are the first ones in the list; those accepted just now wait for the next turn.
        const std::size_t polled = fds.size() - 2;
        for (std::size_t i = polled; i-- > 0;)
        {
            Client& client = *clients[i];
            const short events = fds[i + 2].revents;
            bool dead = (events & (POLLERR | POLLNVAL)) != 0;

            if (!dead && (events & (POLLIN | POLLHUP)))
            {
                // Read only what the protocol could still accept: the rest waits in the kernel (TCP backpressure) until
                // the frames already buffered are handled. A peer that streams without end therefore cannot grow this
                // buffer or keep the thread here.
                const std::size_t input_cap =
                    client.open ? m_config.max_message_bytes + kFrameHeadroom : kMaxHandshakeBytes + 1;
                char buffer[4096];
                while (client.input.size() < input_cap)
                {
                    const ssize_t n = ::recv(client.fd, buffer, sizeof(buffer), 0);
                    if (n > 0)
                    {
                        client.input.append(buffer, static_cast<std::size_t>(n));
                        continue;
                    }
                    if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
                        dead = true;
                    break;
                }
            }

            // Handshake: one HTTP request, then WebSocket frames.
            if (!dead && !client.open && !client.closing)
            {
                const std::size_t end = client.input.find("\r\n\r\n");
                if (end != std::string::npos)
                {
                    websocket::Handshake handshake = websocket::parse_handshake(client.input.substr(0, end + 4));
                    client.input.erase(0, end + 4);
                    if (handshake.ok && !websocket::constant_time_equal(handshake.token, m_config.token))
                    {
                        handshake.ok = false;
                        handshake.status = 401;
                        handshake.reason = "a valid token is required";
                        logger()->warn("management client {} refused: missing or wrong token", client.peer);
                    }
                    if (handshake.ok)
                    {
                        client.open = true;
                        m_clients.fetch_add(1);
                        client.output += websocket::handshake_response(handshake);
                        logger()->info("management client {} connected", client.peer);
                    }
                    else
                    {
                        refuse(client, handshake);
                    }
                }
                else if (client.input.size() > kMaxHandshakeBytes)
                {
                    websocket::Handshake bad;
                    bad.reason = "request too large";
                    refuse(client, bad);
                }
                else if (std::chrono::steady_clock::now() - client.accepted > kHandshakeTimeout)
                {
                    dead = true;
                }
            }

            for (int handled = 0; !dead && client.open && !client.closing && handled < kMaxFramesPerRound; ++handled)
            {
                if (client.output.size() > kMaxOutputBytes)
                {
                    dead = true; // a client that does not read its replies is cut off rather than buffered for
                    break;
                }
                const auto parsed = websocket::parse_frame(client.input, m_config.max_message_bytes);
                if (parsed.status == websocket::ParseStatus::NeedMore)
                    break;
                if (parsed.status == websocket::ParseStatus::Error)
                {
                    client.output += websocket::encode_close(parsed.close_code, parsed.error);
                    client.closing = true;
                    break;
                }
                client.input.erase(0, parsed.consumed);
                const auto& frame = parsed.frame;
                if (frame.opcode == websocket::kPing)
                {
                    client.output += websocket::encode_frame(websocket::kPong, frame.payload);
                }
                else if (frame.opcode == websocket::kClose)
                {
                    client.output += websocket::encode_close(1000, "");
                    client.closing = true;
                }
                else if (frame.opcode == websocket::kText || frame.opcode == websocket::kContinuation)
                {
                    const bool continuation = frame.opcode == websocket::kContinuation;
                    // A message is one text frame, or a text frame without FIN followed by continuations ending
                    // with FIN. Anything else (an orphan continuation, a new message inside an unfinished one) is a
                    // protocol error.
                    if (continuation != client.fragmenting)
                    {
                        client.output += websocket::encode_close(1002, "bad fragmentation");
                        client.closing = true;
                        continue;
                    }
                    client.message += frame.payload;
                    client.fragmenting = !frame.fin;
                    if (client.message.size() > m_config.max_message_bytes)
                    {
                        client.output += websocket::encode_close(1009, "message too big");
                        client.closing = true;
                    }
                    else if (frame.fin)
                    {
                        const std::string request = std::move(client.message);
                        client.message.clear();
                        client.output += websocket::encode_frame(websocket::kText, m_on_message(request));
                    }
                }
                else if (frame.opcode == websocket::kBinary)
                {
                    client.output += websocket::encode_close(1003, "text messages only");
                    client.closing = true;
                }
            }

            if (!dead && !client.output.empty())
            {
                const ssize_t n = ::send(client.fd, client.output.data(), client.output.size(), MSG_NOSIGNAL);
                if (n > 0)
                    client.output.erase(0, static_cast<std::size_t>(n));
                else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                    dead = true;
                if (client.output.size() > kMaxOutputBytes)
                    dead = true; // a client that does not read is cut off rather than buffered for
            }

            if (dead || (client.closing && client.output.empty()))
            {
                if (client.open)
                    m_clients.fetch_sub(1);
                drop(i, dead ? "connection ended" : "closed");
            }
        }
    }

    for (const auto& client : clients)
    {
        if (client->open)
            m_clients.fetch_sub(1);
        ::close(client->fd);
    }
}

} // namespace sensing
} // namespace plugins
