// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/manus_calibration/sha256.hpp"

#include <array>
#include <cstddef>

namespace plugins::manus::calibration
{

namespace
{

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

uint32_t rotr(uint32_t value, unsigned bits)
{
    return (value >> bits) | (value << (32 - bits));
}

void compress(std::array<uint32_t, 8>& state, const uint8_t* block)
{
    std::array<uint32_t, 64> w{};
    for (size_t i = 0; i < 16; ++i)
    {
        w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) | (uint32_t(block[i * 4 + 2]) << 8) |
               uint32_t(block[i * 4 + 3]);
    }
    for (size_t i = 16; i < 64; ++i)
    {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    auto v = state;
    for (size_t i = 0; i < 64; ++i)
    {
        const uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
        const uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
        const uint32_t t1 = v[7] + s1 + ch + kRoundConstants[i] + w[i];
        const uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
        const uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        const uint32_t t2 = s0 + maj;
        v[7] = v[6];
        v[6] = v[5];
        v[5] = v[4];
        v[4] = v[3] + t1;
        v[3] = v[2];
        v[2] = v[1];
        v[1] = v[0];
        v[0] = t1 + t2;
    }
    for (size_t i = 0; i < 8; ++i)
    {
        state[i] += v[i];
    }
}

} // namespace

std::string sha256_hex(const std::vector<uint8_t>& data)
{
    std::array<uint32_t, 8> state = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

    // Pad to a multiple of 64 bytes: 0x80, zeros, then the bit length as a big-endian 64-bit value.
    std::vector<uint8_t> padded(data);
    padded.push_back(0x80);
    while (padded.size() % 64 != 56)
    {
        padded.push_back(0);
    }
    const uint64_t bit_length = uint64_t(data.size()) * 8;
    for (int shift = 56; shift >= 0; shift -= 8)
    {
        padded.push_back(uint8_t(bit_length >> shift));
    }

    for (size_t offset = 0; offset < padded.size(); offset += 64)
    {
        compress(state, padded.data() + offset);
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (const uint32_t word : state)
    {
        for (int shift = 28; shift >= 0; shift -= 4)
        {
            hex.push_back(kHex[(word >> shift) & 0xf]);
        }
    }
    return hex;
}

} // namespace plugins::manus::calibration
