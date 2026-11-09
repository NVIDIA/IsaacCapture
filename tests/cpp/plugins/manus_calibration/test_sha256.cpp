// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <manus_calibration/sha256.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace
{

using plugins::manus::calibration::sha256_hex;

std::vector<uint8_t> bytes_of(const std::string& text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

std::vector<uint8_t> repeated(uint8_t value, size_t count)
{
    return std::vector<uint8_t>(count, value);
}

} // namespace

TEST_CASE("sha256 matches the standard test vectors", "[unit][sha256]")
{
    CHECK(sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex(bytes_of("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex(bytes_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sha256_hex(bytes_of("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrl"
                              "mnopqrsmnopqrstnopqrstu")) ==
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
}

TEST_CASE("sha256 is correct on either side of every padding boundary", "[unit][sha256]")
{
    // Lengths around the 55/56 byte point where padding spills into a second block, and around full blocks.
    CHECK(sha256_hex(repeated('a', 54)) == "a3f01b6939256127582ac8ae9fb47a382a244680806a3f613a118851c1ca1d47");
    CHECK(sha256_hex(repeated('a', 55)) == "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    CHECK(sha256_hex(repeated('a', 56)) == "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    CHECK(sha256_hex(repeated('a', 57)) == "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6");
    CHECK(sha256_hex(repeated('a', 63)) == "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
    CHECK(sha256_hex(repeated('a', 64)) == "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    CHECK(sha256_hex(repeated('a', 65)) == "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0");
    CHECK(sha256_hex(repeated('a', 119)) == "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
    CHECK(sha256_hex(repeated('a', 120)) == "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
    CHECK(sha256_hex(repeated('a', 127)) == "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a");
    CHECK(sha256_hex(repeated('a', 128)) == "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e");
    CHECK(sha256_hex(repeated('a', 129)) == "c12cb024a2e5551cca0e08fce8f1c5e314555cc3fef6329ee994a3db752166ae");
}

TEST_CASE("sha256 handles binary data and large inputs", "[unit][sha256]")
{
    std::vector<uint8_t> pattern(1000);
    for (size_t i = 0; i < pattern.size(); ++i)
    {
        pattern[i] = static_cast<uint8_t>(i % 251);
    }
    CHECK(sha256_hex(pattern) == "4e4c294b331f7a2099a379bec34b9f9fc03dc46ab465d998f4d683da53487e6d");
    CHECK(sha256_hex(repeated('a', 1000000)) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("sha256 output is 64 lowercase hex characters", "[unit][sha256]")
{
    const std::string digest = sha256_hex(bytes_of("any input"));
    REQUIRE(digest.size() == 64);
    for (const char c : digest)
    {
        CHECK(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
    }
}
