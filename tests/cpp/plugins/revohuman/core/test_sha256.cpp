// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/sha256.hpp>

#include <cstdint>
#include <string>
#include <vector>

using plugins::revohuman::sha256_hex;

namespace
{

std::string hash(const std::string& s)
{
    return sha256_hex(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
}

} // namespace

TEST_CASE("sha256 FIPS 180-4 vectors", "[revohuman][sha256]")
{
    REQUIRE(hash("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("sha256 of one million 'a'", "[revohuman][sha256]")
{
    const std::vector<std::uint8_t> data(1'000'000, 'a');
    REQUIRE(sha256_hex(data.data(), data.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("sha256 padding edges", "[revohuman][sha256]")
{
    // 55 and 56 bytes straddle the one-block / two-block padding boundary.
    REQUIRE(hash(std::string(55, 'a')) == "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    REQUIRE(hash(std::string(56, 'a')) == "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    REQUIRE(hash(std::string(64, 'a')) == "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
}
