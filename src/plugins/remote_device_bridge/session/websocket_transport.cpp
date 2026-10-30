// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/websocket_transport.hpp"

#include "inc/remote_plugin_session/protocol_frame.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <stdexcept>
#include <utility>

namespace remote_plugin_session
{

namespace protocol = remote_plugin_protocol;

namespace
{

constexpr std::chrono::milliseconds kStall{ 3000 }; // matches TcpTransport's mid-message stall
constexpr size_t kMaxHttpHead = 8192;
constexpr uint64_t kMaxMessage = protocol::kHeaderSize + protocol::kMaxPayloadSize;
constexpr char kAcceptGuid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

constexpr uint8_t kContinuation = 0x0;
constexpr uint8_t kBinary = 0x2;
constexpr uint8_t kClose = 0x8;
constexpr uint8_t kPing = 0x9;
constexpr uint8_t kPong = 0xA;

std::array<uint8_t, 20> sha1(const std::string& text)
{
    std::array<uint32_t, 5> h{ 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    std::vector<uint8_t> msg(text.begin(), text.end());
    const uint64_t bits = static_cast<uint64_t>(msg.size()) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56)
    {
        msg.push_back(0);
    }
    for (int i = 7; i >= 0; --i)
    {
        msg.push_back(static_cast<uint8_t>(bits >> (i * 8)));
    }
    const auto rotl = [](uint32_t v, int n) { return (v << n) | (v >> (32 - n)); };
    for (size_t chunk = 0; chunk < msg.size(); chunk += 64)
    {
        std::array<uint32_t, 80> w{};
        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t* p = &msg[chunk + 4 * i];
            w[i] = (uint32_t{ p[0] } << 24) | (uint32_t{ p[1] } << 16) | (uint32_t{ p[2] } << 8) | p[3];
        }
        for (size_t i = 16; i < 80; ++i)
        {
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (size_t i = 0; i < 80; ++i)
        {
            const uint32_t f = i < 20              ? (b & c) | (~b & d) :
                               (i < 40 || i >= 60) ? b ^ c ^ d :
                                                     (b & c) | (b & d) | (c & d);
            const uint32_t k = i < 20 ? 0x5A827999 : i < 40 ? 0x6ED9EBA1 : i < 60 ? 0x8F1BBCDC : 0xCA62C1D6;
            const uint32_t t = rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    std::array<uint8_t, 20> digest{};
    for (size_t i = 0; i < digest.size(); ++i)
    {
        digest[i] = static_cast<uint8_t>(h[i / 4] >> (24 - 8 * (i % 4)));
    }
    return digest;
}

std::string base64(const uint8_t* data, size_t size)
{
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < size; i += 3)
    {
        uint32_t n = uint32_t{ data[i] } << 16;
        n |= i + 1 < size ? uint32_t{ data[i + 1] } << 8 : 0;
        n |= i + 2 < size ? data[i + 2] : 0;
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += i + 1 < size ? kAlphabet[(n >> 6) & 63] : '=';
        out += i + 2 < size ? kAlphabet[n & 63] : '=';
    }
    return out;
}

std::string accept_key(const std::string& key)
{
    const std::array<uint8_t, 20> digest = sha1(key + kAcceptGuid);
    return base64(digest.data(), digest.size());
}

std::string lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

struct HttpHead
{
    std::string start_line;
    std::map<std::string, std::string> headers; //!< lowercase names, trimmed values
};

//! Reads up to the blank line one byte at a time so no WebSocket bytes that follow are consumed.
ReceiveResult read_http_head(TcpSocket& socket, std::chrono::milliseconds timeout, HttpHead& head)
{
    std::string text;
    while (text.size() < 4 || text.compare(text.size() - 4, 4, "\r\n\r\n") != 0)
    {
        if (text.size() >= kMaxHttpHead)
        {
            return ReceiveResult::Malformed;
        }
        uint8_t byte = 0;
        const TcpSocket::RecvResult result = socket.recv_all(&byte, 1, text.empty() ? timeout : kStall);
        if (result != TcpSocket::RecvResult::Ok)
        {
            return result == TcpSocket::RecvResult::Timeout && text.empty() ? ReceiveResult::Timeout :
                                                                              ReceiveResult::Closed;
        }
        text.push_back(static_cast<char>(byte));
    }
    size_t begin = text.find("\r\n");
    head.start_line = text.substr(0, begin);
    for (begin += 2; begin < text.size() - 2;)
    {
        const size_t end = text.find("\r\n", begin);
        const std::string line = text.substr(begin, end - begin);
        begin = end + 2;
        const size_t colon = line.find(':');
        if (colon == std::string::npos)
        {
            return ReceiveResult::Malformed;
        }
        const size_t value = line.find_first_not_of(" \t", colon + 1);
        head.headers[lower(line.substr(0, colon))] =
            value == std::string::npos ? "" : line.substr(value, line.find_last_not_of(" \t") + 1 - value);
    }
    return ReceiveResult::Ok;
}

std::string header(const HttpHead& head, const std::string& name)
{
    const auto it = head.headers.find(name);
    return it == head.headers.end() ? "" : it->second;
}

} // namespace

RemoteAddress parse_remote(const std::string& text, uint16_t default_port)
{
    const std::string scheme = "ws://";
    if (text.rfind("wss://", 0) == 0)
    {
        throw std::invalid_argument("wss:// is not supported ('" + text + "'); terminate TLS in a proxy, use ws://");
    }
    RemoteAddress remote;
    if (text.rfind(scheme, 0) != 0)
    {
        remote.endpoint = parse_endpoint(text, default_port);
        return remote;
    }
    remote.websocket = true;
    std::string authority = text.substr(scheme.size());
    const size_t slash = authority.find('/');
    if (slash != std::string::npos)
    {
        remote.path = authority.substr(slash);
        authority.resize(slash);
    }
    remote.endpoint = parse_endpoint(authority, default_port);
    return remote;
}

std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote, std::chrono::milliseconds timeout)
{
    TcpSocket socket = TcpSocket::connect(remote.endpoint, timeout);
    if (!remote.websocket)
    {
        return std::make_unique<TcpTransport>(std::move(socket));
    }
    return std::make_unique<WebSocketTransport>(std::move(socket), WebSocketTransport::Role::Client,
                                                remote.endpoint.host + ":" + std::to_string(remote.endpoint.port),
                                                remote.path, timeout);
}

WebSocketTransport::WebSocketTransport(
    TcpSocket socket, Role role, const std::string& host, const std::string& path, std::chrono::milliseconds timeout)
    : m_socket(std::move(socket)), m_role(role), m_mask_rng(std::random_device{}())
{
    if (role == Role::Server)
    {
        return;
    }
    std::array<uint8_t, 16> nonce{};
    std::generate(nonce.begin(), nonce.end(), [this] { return static_cast<uint8_t>(m_mask_rng()); });
    const std::string key = base64(nonce.data(), nonce.size());
    const std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + host +
                                "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " + key +
                                "\r\nSec-WebSocket-Version: 13\r\n\r\n";
    HttpHead response;
    if (!m_socket.send_all(reinterpret_cast<const uint8_t*>(request.data()), request.size()) ||
        read_http_head(m_socket, timeout, response) != ReceiveResult::Ok)
    {
        throw std::runtime_error("WebSocket upgrade to " + host + path + ": no HTTP response");
    }
    if (response.start_line.rfind("HTTP/1.1 101", 0) != 0)
    {
        throw std::runtime_error("WebSocket upgrade to " + host + path + " refused: " + response.start_line);
    }
    if (header(response, "sec-websocket-accept") != accept_key(key))
    {
        throw std::runtime_error("WebSocket upgrade to " + host + path + ": wrong Sec-WebSocket-Accept");
    }
    m_open = true;
}

std::unique_ptr<ITransport> WebSocketTransport::accept(TcpSocket& listener,
                                                       std::chrono::milliseconds poll,
                                                       std::string* peer)
{
    TcpSocket client = listener.accept(poll);
    if (!client.valid())
    {
        return nullptr;
    }
    if (peer != nullptr)
    {
        *peer = client.peer_name();
    }
    return std::make_unique<WebSocketTransport>(std::move(client), Role::Server);
}

ReceiveResult WebSocketTransport::accept_upgrade(std::chrono::milliseconds timeout)
{
    HttpHead request;
    const ReceiveResult result = read_http_head(m_socket, timeout, request);
    if (result == ReceiveResult::Timeout || result == ReceiveResult::Closed)
    {
        return result;
    }
    const std::string key = header(request, "sec-websocket-key");
    if (result != ReceiveResult::Ok || request.start_line.rfind("GET ", 0) != 0 ||
        lower(header(request, "upgrade")).find("websocket") == std::string::npos ||
        lower(header(request, "connection")).find("upgrade") == std::string::npos ||
        header(request, "sec-websocket-version") != "13" || key.empty())
    {
        static const std::string kBadRequest = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        m_socket.send_all(reinterpret_cast<const uint8_t*>(kBadRequest.data()), kBadRequest.size());
        m_socket.shutdown();
        return ReceiveResult::Malformed;
    }
    const std::string response =
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
        accept_key(key) + "\r\n\r\n";
    if (!m_socket.send_all(reinterpret_cast<const uint8_t*>(response.data()), response.size()))
    {
        return ReceiveResult::Closed;
    }
    m_open = true;
    return ReceiveResult::Ok;
}

bool WebSocketTransport::send(const uint8_t* frame, size_t size)
{
    return m_open.load() && send_message(kBinary, frame, size);
}

bool WebSocketTransport::send_message(uint8_t opcode, const uint8_t* data, size_t size)
{
    std::lock_guard<std::mutex> lock(m_send_mutex);
    std::vector<uint8_t>& out = m_send_buffer;
    out.assign(1, static_cast<uint8_t>(0x80 | opcode)); // FIN: never fragmented on send
    const uint8_t mask_bit = m_role == Role::Client ? 0x80 : 0x00;
    if (size < 126)
    {
        out.push_back(static_cast<uint8_t>(mask_bit | size));
    }
    else
    {
        const int length_bytes = size <= 0xFFFF ? 2 : 8;
        out.push_back(static_cast<uint8_t>(mask_bit | (length_bytes == 2 ? 126 : 127)));
        for (int i = length_bytes - 1; i >= 0; --i)
        {
            out.push_back(static_cast<uint8_t>(static_cast<uint64_t>(size) >> (8 * i)));
        }
    }
    const size_t payload_at = out.size() + (mask_bit ? 4 : 0);
    if (mask_bit)
    {
        const uint32_t key = m_mask_rng();
        out.insert(out.end(), { static_cast<uint8_t>(key >> 24), static_cast<uint8_t>(key >> 16),
                                static_cast<uint8_t>(key >> 8), static_cast<uint8_t>(key) });
    }
    out.insert(out.end(), data, data + size);
    if (mask_bit)
    {
        for (size_t i = 0; i < size; ++i)
        {
            out[payload_at + i] ^= out[payload_at - 4 + i % 4];
        }
    }
    return m_socket.send_all(out.data(), out.size());
}

ReceiveResult WebSocketTransport::receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    if (!m_open.load())
    {
        const ReceiveResult upgraded = accept_upgrade(timeout);
        if (upgraded != ReceiveResult::Ok)
        {
            return upgraded;
        }
    }
    frame.clear();
    bool in_message = false;
    while (true)
    {
        // The caller's timeout covers the wait for a message to start (pings included); inside one, a stall is a drop.
        const auto remaining = timeout.count() < 0 ? timeout :
                                                     std::max(std::chrono::milliseconds{ 0 },
                                                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                  deadline - std::chrono::steady_clock::now()));
        uint8_t head[2];
        const TcpSocket::RecvResult first = m_socket.recv_all(head, sizeof(head), in_message ? kStall : remaining);
        if (first != TcpSocket::RecvResult::Ok)
        {
            return first == TcpSocket::RecvResult::Timeout && !in_message ? ReceiveResult::Timeout :
                                                                            ReceiveResult::Closed;
        }
        const bool fin = (head[0] & 0x80) != 0;
        const uint8_t opcode = head[0] & 0x0F;
        const bool masked = (head[1] & 0x80) != 0;
        const bool control = (opcode & 0x08) != 0;
        uint64_t length = head[1] & 0x7F;
        // RSV bits need an extension we never negotiate; clients must mask and servers must not.
        if ((head[0] & 0x70) != 0 || masked != (m_role == Role::Server))
        {
            return ReceiveResult::Closed;
        }
        if (length >= 126)
        {
            std::array<uint8_t, 8> ext{};
            const size_t ext_size = length == 126 ? 2 : 8;
            if (m_socket.recv_all(ext.data(), ext_size, kStall) != TcpSocket::RecvResult::Ok)
            {
                return ReceiveResult::Closed;
            }
            length = 0;
            for (size_t i = 0; i < ext_size; ++i)
            {
                length = (length << 8) | ext[i];
            }
        }
        std::array<uint8_t, 4> mask{};
        if (masked && m_socket.recv_all(mask.data(), mask.size(), kStall) != TcpSocket::RecvResult::Ok)
        {
            return ReceiveResult::Closed;
        }

        std::vector<uint8_t> control_payload;
        uint8_t* dest = nullptr;
        if (control)
        {
            if (!fin || length > 125)
            {
                return ReceiveResult::Closed;
            }
            control_payload.resize(length);
            dest = control_payload.data();
        }
        else
        {
            if (opcode != (in_message ? kContinuation : kBinary))
            {
                return ReceiveResult::Closed; // text frames, unknown opcodes, broken fragmentation
            }
            if (length > kMaxMessage - frame.size())
            {
                return ReceiveResult::Malformed;
            }
            const size_t offset = frame.size();
            frame.resize(offset + length);
            dest = frame.data() + offset;
        }
        if (length > 0 && m_socket.recv_all(dest, length, kStall) != TcpSocket::RecvResult::Ok)
        {
            return ReceiveResult::Closed;
        }
        for (size_t i = 0; masked && i < length; ++i)
        {
            dest[i] ^= mask[i % 4];
        }

        if (opcode == kPing)
        {
            send_message(kPong, control_payload.data(), control_payload.size());
            continue;
        }
        if (opcode == kPong)
        {
            continue;
        }
        if (control)
        {
            // Close (and the reserved control opcodes): echo the status code, then report the peer gone.
            send_message(kClose, control_payload.data(), std::min<size_t>(control_payload.size(), 2));
            return ReceiveResult::Closed;
        }
        in_message = !fin;
        if (in_message)
        {
            continue;
        }
        try
        {
            return frame.size() >= protocol::kHeaderSize &&
                           frame.size() == protocol::kHeaderSize + protocol::decode_header(frame).payload_size ?
                       ReceiveResult::Ok :
                       ReceiveResult::Malformed;
        }
        catch (const protocol::DecodeError&)
        {
            return ReceiveResult::Malformed;
        }
    }
}

void WebSocketTransport::shutdown()
{
    m_socket.shutdown();
}

ReceiveResult WebSocketTransport::wait_ready(std::chrono::milliseconds timeout)
{
    return m_open.load() ? ReceiveResult::Ok : accept_upgrade(timeout);
}

} // namespace remote_plugin_session
