// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fake_sdk/fake_revohuman.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <revohuman_plugin/live_source.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace plugins::revohuman;
using Catch::Matchers::ContainsSubstring;

namespace
{

constexpr std::int64_t kMs = 1'000'000;
constexpr std::int64_t kStart = 5'000 * kMs;

struct FakeSdk
{
    FakeSdk()
    {
        fake_rh::reset();
    }
    ~FakeSdk()
    {
        fake_rh::reset();
    }
};

//! Opens and closes inside poll(), so every step of a test is deterministic.
LiveSourceConfig inline_config()
{
    LiveSourceConfig config;
    config.background_open = false;
    return config;
}

//! Only the left glove is plugged in.
LiveSourceConfig left_only()
{
    fake_rh::set_present(RH_HAND_RIGHT, false);
    return inline_config();
}

using WallClock = std::chrono::steady_clock;

//! Polls, with now_ns advancing like the wall clock from start_ns, until done() or 5 s passed.
template <typename Done>
bool poll_until(LiveSource& source, std::int64_t start_ns, std::vector<SourceBatch>& batches, Done done)
{
    const auto t0 = WallClock::now();
    while (WallClock::now() - t0 < std::chrono::seconds(5))
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(WallClock::now() - t0).count();
        batches.push_back(source.poll(start_ns + elapsed));
        if (done())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

template <typename T>
std::vector<T> records_of(const SourceBatch& batch)
{
    std::vector<T> out;
    for (const auto& record : batch.records)
    {
        if (const auto* r = std::get_if<T>(&record))
        {
            out.push_back(*r);
        }
    }
    return out;
}

//! Joint poses that encode the input: joint 2 sits at x = joints_rad[0]. Gives none when joint 0 is
//! invalid.
class EchoKinematics : public KinematicsProvider
{
public:
    explicit EchoKinematics(std::string id) : id_(std::move(id))
    {
    }
    std::string model_id() const override
    {
        return id_;
    }
    std::optional<HandKinematics> solve(Hand,
                                        const std::array<double, kGloveJointCount>& joints_rad,
                                        std::uint32_t joints_valid_mask) const override
    {
        if ((joints_valid_mask & 1u) == 0)
        {
            return std::nullopt;
        }
        HandKinematics k;
        k.joints[2].p = { joints_rad[0], 0.0, 0.0 };
        return k;
    }

private:
    std::string id_;
};

//! Makes an EchoKinematics named "echo:<product>" and records the products it was asked for, from
//! whichever thread opens the glove.
struct EchoFactory
{
    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::vector<std::string>> products = std::make_shared<std::vector<std::string>>();

    KinematicsFactory make() const
    {
        return [mutex = mutex, products = products](const std::string& product_code)
        {
            std::lock_guard<std::mutex> lock(*mutex);
            products->push_back(product_code);
            return std::make_shared<EchoKinematics>("echo:" + product_code);
        };
    }
    std::vector<std::string> asked() const
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return *products;
    }
};

} // namespace

TEST_CASE("live source opens both plugged-in gloves together in one kit", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSourceConfig config = inline_config();
    config.queue_size = 32;
    LiveSource source(config);

    const fake_rh::Calls& calls = fake_rh::calls();
    CHECK(calls.init == 1);
    CHECK(calls.log_callback_set);
    CHECK(calls.scan == 1);
    CHECK(calls.open_hands == std::vector<RhHands>{ RH_HANDS_BOTH });
    // Sampled on the same USB frame.
    CHECK(calls.sync == RH_SYNC_ON);
    // Nothing declares the product: the SDK identifies it on the gloves.
    CHECK(calls.product == "auto");
    CHECK(calls.tracker_config_null);
    CHECK(calls.rate_hz == 200.0);
    CHECK(calls.subscribe == 2);
    CHECK(calls.queue_size == 32);
    CHECK(calls.ungated == 0);
    CHECK_FALSE(calls.callback_given);
    CHECK(fake_rh::open_kits() == 1);
    CHECK(fake_rh::open_pair_kits() == 1);
    CHECK(source.opened(Hand::Left));
    CHECK(source.opened(Hand::Right));
    CHECK(source.paired(Hand::Left));
    CHECK(source.paired(Hand::Right));

    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart));
    fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart));
    const SourceBatch batch = source.poll(kStart);
    CHECK(records_of<HandMeta>(batch).size() == 2);
    const auto samples = records_of<GloveSample>(batch);
    REQUIRE(samples.size() == 2);
    CHECK(samples[0].hand == Hand::Left);
    CHECK(samples[1].hand == Hand::Right);
    CHECK_FALSE(batch.hand_restarted[0]);
    CHECK_FALSE(batch.hand_restarted[1]);
    // Already together: nothing is reopened.
    source.poll(kStart + 10'000 * kMs);
    CHECK(calls.kit_open == 1);
    CHECK(calls.kit_close == 0);
}

TEST_CASE("live source streams the one glove that is plugged in", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());
    const fake_rh::Calls& calls = fake_rh::calls();
    CHECK(calls.product == "auto");
    CHECK(calls.open_hands == std::vector<RhHands>{ RH_HANDS_LEFT, RH_HANDS_RIGHT });
    CHECK(calls.subscribe == 1);
    CHECK(source.opened(Hand::Left));
    CHECK_FALSE(source.opened(Hand::Right));
    CHECK_FALSE(source.paired(Hand::Left));

    const SourceBatch batch = source.poll(kStart);
    const auto metas = records_of<HandMeta>(batch);
    REQUIRE(metas.size() == 1);
    CHECK(metas[0].hand == Hand::Left);
    CHECK_FALSE(batch.finished);

    CHECK_FALSE(source.glove_note(Hand::Left));
    const auto note = source.glove_note(Hand::Right);
    REQUIRE(note);
    CHECK(note->kind == Source::GloveNote::Kind::Waiting);
    CHECK_THAT(note->text, ContainsSubstring("RH_ERR_NOT_FOUND: E1000 RevoHumanError"));
}

TEST_CASE("live source rejects a config without a queue", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSourceConfig no_queue = inline_config();
    no_queue.queue_size = 0;
    CHECK_THROWS_AS(LiveSource(no_queue), std::invalid_argument);
    CHECK(fake_rh::calls().scan == 0);
    CHECK(fake_rh::calls().kit_open == 0);
}

TEST_CASE("live source delivers each hand's metadata once, before its samples", "[revohuman][live]")
{
    FakeSdk sdk;
    fake_rh::set_glove_ids(RH_HAND_LEFT, "FAKE-SN-LEFT", "1-1.2");
    fake_rh::set_glove_ids(RH_HAND_RIGHT, "", "");
    LiveSource source(inline_config());
    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart));

    const SourceBatch first = source.poll(kStart);
    REQUIRE(first.records.size() == 3);
    const auto* left = std::get_if<HandMeta>(&first.records[0]);
    REQUIRE(left != nullptr);
    CHECK(std::holds_alternative<GloveSample>(first.records[1]));
    const auto* right = std::get_if<HandMeta>(&first.records[2]);
    REQUIRE(right != nullptr);
    CHECK_FALSE(first.hand_restarted[0]);
    CHECK_FALSE(first.hand_restarted[1]);

    CHECK(left->hand == Hand::Left);
    CHECK(left->product_code == "P1");
    // P1 has no tactile sensors; the SDK's product table says so.
    CHECK_FALSE(left->has_tactile);
    CHECK(left->device_id == "FAKE-SN-LEFT");
    CHECK(left->joint_names[0] == "left_fake_joint_0");
    CHECK(left->joint_names[20] == "left_fake_joint_20");
    CHECK(right->hand == Hand::Right);
    CHECK(right->device_id == "revohuman_right");
    CHECK(right->joint_names[3] == "right_fake_joint_3");

    fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart + 5 * kMs));
    const SourceBatch second = source.poll(kStart + 5 * kMs);
    CHECK(records_of<HandMeta>(second).empty());
    CHECK(records_of<GloveSample>(second).size() == 1);
    CHECK(source.records_released() == 4);
    CHECK_FALSE(second.finished);
}

TEST_CASE("live source names a glove without an SN by its hand, not by its USB port", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());
    const auto metas = records_of<HandMeta>(source.poll(kStart));
    REQUIRE(metas.size() == 1);
    CHECK(metas[0].device_id == "revohuman_left");
}

TEST_CASE("live source takes joint angles from rh_joints_to_radians for the identified product", "[revohuman][live]")
{
    FakeSdk sdk;
    SECTION("identified product")
    {
        fake_rh::set_variant("PTV1");
        LiveSource source(left_only());
        CHECK(source.product_code(Hand::Left) == "PTV1");
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart));
        const auto samples = records_of<GloveSample>(source.poll(kStart));
        REQUIRE(samples.size() == 1);
        CHECK(fake_rh::calls().joints_product == "PTV1");
        CHECK(samples[0].joints_rad[0] == Catch::Approx(1.0));
        CHECK(samples[0].joints_rad[7] == Catch::Approx(1.7));
        CHECK(samples[0].sample_mono_ns == kStart);
        CHECK(samples[0].has_sample_time);
        CHECK(samples[0].joints_valid_mask == (1u << 21) - 1u);
        CHECK_FALSE(samples[0].kinematics);
    }
    SECTION("a glove whose product the SDK cannot tell is not opened yet, with no default table")
    {
        fake_rh::set_variant(std::nullopt);
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(fake_rh::open_kits() == 0);
        CHECK(source.product_code(Hand::Left).empty());
        auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("rh_kit_identity failed (RH_ERR_STATE): E1803 ProductUnidentified (the "
                                                 "SDK cannot identify the glove product)"));

        fake_rh::set_variant("P1");
        source.poll(kStart);
        source.poll(kStart + 2'000 * kMs);
        REQUIRE(source.opened(Hand::Left));
        CHECK(source.product_code(Hand::Left) == "P1");
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 2'000 * kMs));
        REQUIRE(records_of<GloveSample>(source.poll(kStart + 2'010 * kMs)).size() == 1);
        CHECK(fake_rh::calls().joints_product == "P1");
    }
    SECTION("a glove the SDK opened without a product code is given up")
    {
        fake_rh::set_variant("");
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(fake_rh::open_kits() == 0);
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK_THAT(note->text, ContainsSubstring("without identifying its product"));
        CHECK(fake_rh::calls().joints_product.empty());
    }
    SECTION("each glove with the product identified on it")
    {
        // The SDK cannot open two products in one kit, so each glove streams on its own.
        fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
        LiveSource source(inline_config());
        CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_BOTH, RH_HANDS_LEFT, RH_HANDS_RIGHT });
        CHECK(fake_rh::open_kits() == 2);
        CHECK_FALSE(source.paired(Hand::Left));
        CHECK_FALSE(source.paired(Hand::Right));
        CHECK(source.product_code(Hand::Left) == "P1");
        CHECK(source.product_code(Hand::Right) == "PTV1");
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart));
        fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart));
        const SourceBatch batch = source.poll(kStart);
        const auto metas = records_of<HandMeta>(batch);
        REQUIRE(metas.size() == 2);
        CHECK(metas[0].product_code == "P1");
        CHECK_FALSE(metas[0].has_tactile);
        CHECK(metas[1].product_code == "PTV1");
        CHECK(metas[1].has_tactile);
        const auto samples = records_of<GloveSample>(batch);
        REQUIRE(samples.size() == 2);
        CHECK(samples[0].hand == Hand::Left);
        CHECK(samples[0].joints_rad[7] == Catch::Approx(0.7));
        CHECK(samples[1].hand == Hand::Right);
        CHECK(samples[1].joints_rad[7] == Catch::Approx(1.7));
        // They are not tried together again.
        source.poll(kStart + 10'000 * kMs);
        CHECK(fake_rh::calls().kit_open == 3);
        CHECK(fake_rh::calls().kit_close == 0);
    }
}

TEST_CASE("live source drops frames it cannot use and counts them", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());

    RhGloveFrame wrong_hand = fake_rh::frame(RH_HAND_RIGHT, kStart);
    RhGloveFrame no_time = fake_rh::frame(RH_HAND_LEFT, kStart + 5 * kMs);
    no_time.flags &= ~RH_FRAME_FLAG_HAS_SAMPLE_TIME;
    RhGloveFrame bad_tactile = fake_rh::frame(RH_HAND_LEFT, kStart + 10 * kMs);
    bad_tactile.flags |= RH_FRAME_FLAG_HAS_TACTILE;
    bad_tactile.tactile.point_count = RH_TACTILE_MAX_POINTS + 1;
    const RhGloveFrame good = fake_rh::frame(RH_HAND_LEFT, kStart + 15 * kMs);
    for (const RhGloveFrame& f : { wrong_hand, no_time, bad_tactile, good })
    {
        fake_rh::push(RH_HAND_LEFT, f);
    }

    const auto samples = records_of<GloveSample>(source.poll(kStart + 15 * kMs));
    REQUIRE(samples.size() == 1);
    CHECK(samples[0].sample_mono_ns == kStart + 15 * kMs);
    const LiveSource::HandStats stats = source.stats(Hand::Left);
    CHECK(stats.frames == 4);
    CHECK(stats.rejected == 3);
    CHECK(stats.opens == 1);
    CHECK(stats.ends == 0);
}

TEST_CASE("live source attaches joint poses from the kinematics provider", "[revohuman][live]")
{
    FakeSdk sdk;
    fake_rh::set_variant("PTV1");
    const EchoFactory factory;
    LiveSource source(left_only(), factory.make());
    CHECK(factory.asked() == std::vector<std::string>{ "PTV1" });
    CHECK(source.model_id(Hand::Left) == "echo:PTV1");

    RhGloveFrame missing_joint = fake_rh::frame(RH_HAND_LEFT, kStart + 5 * kMs);
    missing_joint.joints_valid_mask &= ~1u;
    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart));
    fake_rh::push(RH_HAND_LEFT, missing_joint);

    const SourceBatch batch = source.poll(kStart + 5 * kMs);
    const auto metas = records_of<HandMeta>(batch);
    REQUIRE(metas.size() == 1);
    const auto samples = records_of<GloveSample>(batch);
    REQUIRE(samples.size() == 2);
    REQUIRE(samples[0].kinematics);
    // The provider sees the SDK's radians, not the device degrees.
    CHECK(samples[0].kinematics->joints[2].p[0] == Catch::Approx(1.0));
    CHECK_FALSE(samples[1].kinematics);
    CHECK(source.stats(Hand::Left).without_kinematics == 1);
}

TEST_CASE("live source makes each glove's kinematics for the product identified on it", "[revohuman][live]")
{
    FakeSdk sdk;
    fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
    const EchoFactory factory;
    LiveSource source(inline_config(), factory.make());
    CHECK(factory.asked() == std::vector<std::string>{ "P1", "PTV1" });
    CHECK(source.model_id(Hand::Left) == "echo:P1");
    CHECK(source.model_id(Hand::Right) == "echo:PTV1");

    // A glove opened again is asked for again, with the product identified on it then.
    fake_rh::end(RH_HAND_RIGHT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: unplugged");
    source.poll(kStart);
    CHECK_FALSE(source.opened(Hand::Right));
    CHECK(source.model_id(Hand::Right).empty());
    // Now the same product as the left glove: once opened on its own, both are reopened in one kit.
    fake_rh::set_variant(RH_HAND_RIGHT, "P1");
    const SourceBatch batch = source.poll(kStart + 2'000 * kMs);
    REQUIRE(source.opened(Hand::Right));
    CHECK(source.paired(Hand::Right));
    CHECK(source.paired(Hand::Left));
    CHECK(factory.asked() == std::vector<std::string>{ "P1", "PTV1", "P1", "P1", "P1" });
    CHECK(source.model_id(Hand::Left) == "echo:P1");
    CHECK(source.model_id(Hand::Right) == "echo:P1");
    const auto metas = records_of<HandMeta>(batch);
    REQUIRE(metas.size() == 2);
    CHECK(batch.hand_restarted[0]);
    CHECK(batch.hand_restarted[1]);
}

TEST_CASE("live source gives up on a glove whose product has no kinematics", "[revohuman][live]")
{
    FakeSdk sdk;
    fake_rh::set_variant(RH_HAND_LEFT, "P1");
    fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
    std::string message;
    bool logic_error = false;
    SECTION("the plugin does not support its model")
    {
        logic_error = true;
        message = "SDK hand model 'p' differs from the reviewed models; use a matching C SDK";
    }
    SECTION("the SDK has no model for it")
    {
        message = "the C SDK has no hand model for product 'P1'";
    }
    const LiveSourceConfig config = inline_config();
    const KinematicsFactory factory = [&](const std::string& product_code) -> std::shared_ptr<const KinematicsProvider>
    {
        if (product_code == "P1")
        {
            if (logic_error)
            {
                throw std::invalid_argument(message);
            }
            throw std::runtime_error(message);
        }
        return std::make_shared<EchoKinematics>("echo:" + product_code);
    };
    SECTION("the other glove streams")
    {
        LiveSource source(config, factory);
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(source.opened(Hand::Right));
        CHECK(fake_rh::open_kits() == 1);
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK_THAT(note->text, ContainsSubstring("no hand kinematics for product 'P1': " + message));

        fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart));
        const auto samples = records_of<GloveSample>(source.poll(kStart));
        REQUIRE(samples.size() == 1);
        CHECK(samples[0].kinematics);
        // A given-up glove is not tried again.
        CHECK_FALSE(source.poll(kStart + 10'000 * kMs).finished);
        // The kit of both (refused: different products), then each glove's own.
        CHECK(fake_rh::calls().kit_open == 3);
        CHECK(fake_rh::calls().kit_close == 1);
    }
    SECTION("both gloves of such a product: nothing can ever stream")
    {
        fake_rh::set_variant(RH_HAND_RIGHT, "P1");
        CHECK_THROWS_WITH(LiveSource(config, factory), ContainsSubstring("cannot open either glove"));
        CHECK(fake_rh::open_kits() == 0);
    }
}

TEST_CASE("live source notes a closed capture gate, checked every 100 ms", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());
    CHECK_FALSE(source.glove_note(Hand::Left));

    fake_rh::set_gate(RH_HAND_LEFT, false, RH_CAUSE_CLOCK_DEGRADED);
    source.poll(kStart);
    auto note = source.glove_note(Hand::Left);
    REQUIRE(note);
    CHECK(note->kind == Source::GloveNote::Kind::NotReady);
    CHECK_THAT(note->text, ContainsSubstring("clock_degraded"));

    fake_rh::set_gate(RH_HAND_LEFT, true, RH_CAUSE_NONE);
    source.poll(kStart + 50 * kMs);
    CHECK(source.glove_note(Hand::Left));
    source.poll(kStart + 100 * kMs);
    CHECK_FALSE(source.glove_note(Hand::Left));
}

TEST_CASE("live source counts frames the SDK dropped", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSourceConfig config = left_only();
    config.queue_size = 4;
    LiveSource source(config);
    for (int i = 0; i < 6; ++i)
    {
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + i * 5 * kMs));
    }
    const auto samples = records_of<GloveSample>(source.poll(kStart + 30 * kMs));
    REQUIRE(samples.size() == 4);
    CHECK(samples[0].sample_mono_ns == kStart + 10 * kMs);
    CHECK(source.stats(Hand::Left).sdk_dropped == 2);
}

TEST_CASE("live source looks again for a glove streaming on its own when its subscription ends", "[revohuman][live]")
{
    FakeSdk sdk;
    // Different products, so each glove has a kit of its own.
    fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
    {
        LiveSource source(inline_config());
        REQUIRE_FALSE(source.paired(Hand::Left));
        source.poll(kStart);
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 5 * kMs));
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 10 * kMs));
        fake_rh::end(RH_HAND_LEFT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: unplugged");
        fake_rh::set_present(RH_HAND_LEFT, false);

        // Frames queued before the end are still delivered; then that glove's kit closes.
        SourceBatch batch = source.poll(kStart + 10 * kMs);
        CHECK(records_of<GloveSample>(batch).size() == 2);
        CHECK_FALSE(batch.finished);
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(source.stats(Hand::Left).ends == 1);
        CHECK(fake_rh::calls().sub_close == 1);
        CHECK(fake_rh::calls().kit_close == 1);
        CHECK(fake_rh::open_kits() == 1);
        auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("E1101 TransportDisconnected (the glove's USB link dropped)"));

        // The other glove goes on streaming.
        CHECK_FALSE(source.glove_note(Hand::Right));
        fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart + 15 * kMs));
        batch = source.poll(kStart + 15 * kMs);
        const auto right = records_of<GloveSample>(batch);
        REQUIRE(right.size() == 1);
        CHECK(right[0].hand == Hand::Right);

        // Looked for every 2 s after the end.
        source.poll(kStart + 2'009 * kMs);
        CHECK(fake_rh::calls().kit_open == 3);
        source.poll(kStart + 2'010 * kMs);
        CHECK(fake_rh::calls().kit_open == 4);
        note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("RH_ERR_NOT_FOUND: E1000 RevoHumanError"));

        // Plugged in again: it starts over with new metadata.
        fake_rh::set_glove_ids(RH_HAND_LEFT, "FAKE-SN-LEFT-AGAIN", "fake-unit-left-again");
        fake_rh::set_present(RH_HAND_LEFT, true);
        batch = source.poll(kStart + 4'010 * kMs);
        CHECK(fake_rh::calls().kit_open == 5);
        CHECK(source.opened(Hand::Left));
        CHECK_FALSE(source.glove_note(Hand::Left));
        const auto metas = records_of<HandMeta>(batch);
        REQUIRE(metas.size() == 1);
        CHECK(metas[0].hand == Hand::Left);
        CHECK(metas[0].device_id == "FAKE-SN-LEFT-AGAIN");
        CHECK(batch.hand_restarted[0]);
        CHECK_FALSE(batch.hand_restarted[1]);
        CHECK(source.stats(Hand::Left).opens == 2);

        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 4'015 * kMs));
        batch = source.poll(kStart + 4'015 * kMs);
        CHECK(records_of<GloveSample>(batch).size() == 1);
        CHECK_FALSE(batch.hand_restarted[0]);
        CHECK(source.stats(Hand::Left).frames == 3);
    }
    // Ended subscriptions are not closed twice.
    CHECK(fake_rh::calls().sub_close == 3);
    CHECK(fake_rh::calls().sub_close_after_kit_close == 0);
    CHECK(fake_rh::calls().kit_close == 3);
    CHECK(fake_rh::open_kits() == 0);
}

TEST_CASE("live source closes both paired gloves when one's subscription ends", "[revohuman][live]")
{
    FakeSdk sdk;
    {
        LiveSource source(inline_config());
        REQUIRE(source.paired(Hand::Left));
        source.poll(kStart);
        fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 5 * kMs));
        fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart + 5 * kMs));
        fake_rh::end(RH_HAND_LEFT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: link reset");

        // The left glove's queued frames are delivered; the SDK cannot take it out of the kit alone,
        // so the right glove closes with it, and then the kit.
        SourceBatch batch = source.poll(kStart + 10 * kMs);
        const auto samples = records_of<GloveSample>(batch);
        REQUIRE(samples.size() == 1);
        CHECK(samples[0].hand == Hand::Left);
        CHECK_FALSE(batch.finished);
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK_FALSE(source.opened(Hand::Right));
        CHECK(source.stats(Hand::Left).ends == 1);
        CHECK(source.stats(Hand::Right).ends == 1);
        CHECK(fake_rh::calls().sub_close == 2);
        CHECK(fake_rh::calls().sub_close_after_kit_close == 0);
        CHECK(fake_rh::calls().kit_close == 1);
        CHECK(fake_rh::open_kits() == 0);
        auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("E1101 TransportDisconnected (the glove's USB link dropped)"));
        note = source.glove_note(Hand::Right);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("the paired left glove's stream ended: E1101"));

        // Both are looked for again together 2 s later, and both start over.
        source.poll(kStart + 2'009 * kMs);
        CHECK(fake_rh::calls().kit_open == 1);
        batch = source.poll(kStart + 2'010 * kMs);
        CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_BOTH, RH_HANDS_BOTH });
        CHECK(source.paired(Hand::Left));
        CHECK(source.paired(Hand::Right));
        CHECK(records_of<HandMeta>(batch).size() == 2);
        CHECK(batch.hand_restarted[0]);
        CHECK(batch.hand_restarted[1]);
        CHECK(source.stats(Hand::Left).opens == 2);
        CHECK(source.stats(Hand::Right).opens == 2);
    }
    CHECK(fake_rh::calls().sub_close == 4);
    CHECK(fake_rh::calls().sub_close_after_kit_close == 0);
    CHECK(fake_rh::calls().kit_close == 2);
    CHECK(fake_rh::open_kits() == 0);
}

TEST_CASE("live source reopens a glove plugged in while the other streams in one kit with it", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());
    source.poll(kStart);
    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 5 * kMs));
    REQUIRE(records_of<GloveSample>(source.poll(kStart + 5 * kMs)).size() == 1);

    // The right glove opens on its own first, so its product is known before the left glove
    // restarts; then both open in one kit, and only the kit's HandMeta of the right glove is
    // delivered.
    fake_rh::set_present(RH_HAND_RIGHT, true);
    SourceBatch batch = source.poll(kStart + 2'000 * kMs);
    CHECK(fake_rh::calls().open_hands ==
          std::vector<RhHands>{ RH_HANDS_LEFT, RH_HANDS_RIGHT, RH_HANDS_RIGHT, RH_HANDS_BOTH });
    CHECK(fake_rh::calls().kit_close == 2);
    CHECK(fake_rh::open_kits() == 1);
    CHECK(fake_rh::open_pair_kits() == 1);
    CHECK(source.paired(Hand::Left));
    CHECK(source.paired(Hand::Right));
    const auto metas = records_of<HandMeta>(batch);
    REQUIRE(metas.size() == 2);
    CHECK(batch.hand_restarted[0]);
    CHECK_FALSE(batch.hand_restarted[1]);

    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 2'005 * kMs));
    fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart + 2'005 * kMs));
    batch = source.poll(kStart + 2'005 * kMs);
    CHECK(records_of<GloveSample>(batch).size() == 2);
    CHECK(records_of<HandMeta>(batch).empty());
    CHECK(source.poll(kStart + 10'000 * kMs).records.empty());
    CHECK(fake_rh::calls().kit_open == 4);
}

TEST_CASE("live source does not reopen two gloves in one kit again until either stream ends", "[revohuman][live]")
{
    FakeSdk sdk;
    LiveSource source(left_only());
    source.poll(kStart);

    fake_rh::set_present(RH_HAND_RIGHT, true);
    fake_rh::fail_next_pair_open(
        RH_ERR_PROTOCOL, RH_E_TIME_SYNC_UNSUPPORTED, "TimeSyncUnsupported: the left glove cannot join a sync session");
    SourceBatch batch = source.poll(kStart + 2'000 * kMs);
    // Each glove streams on its own again, without the right glove restarting.
    CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_LEFT, RH_HANDS_RIGHT, RH_HANDS_RIGHT,
                                                               RH_HANDS_BOTH, RH_HANDS_LEFT, RH_HANDS_RIGHT });
    CHECK(source.opened(Hand::Left));
    CHECK(source.opened(Hand::Right));
    CHECK_FALSE(source.paired(Hand::Left));
    CHECK_FALSE(source.paired(Hand::Right));
    CHECK_FALSE(source.glove_note(Hand::Left));
    CHECK_FALSE(source.glove_note(Hand::Right));
    CHECK(records_of<HandMeta>(batch).size() == 2);
    CHECK(batch.hand_restarted[0]);
    CHECK_FALSE(batch.hand_restarted[1]);

    // No loop of reopening them together.
    source.poll(kStart + 4'000 * kMs);
    source.poll(kStart + 10'000 * kMs);
    CHECK(fake_rh::calls().kit_open == 6);
    CHECK(fake_rh::open_kits() == 2);

    // Once a glove is plugged in again, they are tried together once more.
    fake_rh::end(RH_HAND_RIGHT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: unplugged");
    source.poll(kStart + 10'005 * kMs);
    CHECK_FALSE(source.opened(Hand::Right));
    CHECK(source.opened(Hand::Left));
    source.poll(kStart + 12'005 * kMs);
    CHECK(fake_rh::calls().kit_open == 8);
    CHECK(source.paired(Hand::Left));
    CHECK(source.paired(Hand::Right));
}

TEST_CASE("live source opens the gloves on their own when it cannot scan for them", "[revohuman][live]")
{
    FakeSdk sdk;
    fake_rh::fail_next_scan(RH_ERR_IO, RH_E_REVOHUMAN_ERROR, "RevoHumanError: cannot read /sys/bus/usb");
    LiveSource source(inline_config());
    CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_LEFT, RH_HANDS_RIGHT });
    CHECK(fake_rh::open_kits() == 2);

    // Both stream on their own, so the first poll reopens them in one kit before either delivers
    // anything.
    const SourceBatch batch = source.poll(kStart);
    CHECK(fake_rh::open_pair_kits() == 1);
    CHECK(fake_rh::open_kits() == 1);
    CHECK(source.paired(Hand::Left));
    CHECK(records_of<HandMeta>(batch).size() == 2);
    CHECK_FALSE(batch.hand_restarted[0]);
    CHECK_FALSE(batch.hand_restarted[1]);
}

TEST_CASE("live source closes its subscriptions before the kits", "[revohuman][live]")
{
    FakeSdk sdk;
    int kits = 0;
    SECTION("one kit for both gloves")
    {
        kits = 1;
    }
    SECTION("a kit for each glove")
    {
        fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
        kits = 2;
    }
    {
        LiveSource source(inline_config());
        source.poll(kStart);
        REQUIRE(fake_rh::open_kits() == kits);
    }
    CHECK(fake_rh::calls().sub_close == 2);
    CHECK(fake_rh::calls().sub_close_after_kit_close == 0);
    CHECK(fake_rh::calls().kit_close == kits);
}

TEST_CASE("live source starts without gloves and opens them once they can be", "[revohuman][open]")
{
    FakeSdk sdk;
    fake_rh::fail_next_open(
        RH_HAND_LEFT, RH_ERR_BUSY, RH_E_PORT_BUSY, "PortBusyError: /dev/ttyACM0 is held by pid 42", 2);
    LiveSource source(left_only());
    CHECK_FALSE(source.opened(Hand::Left));
    CHECK(fake_rh::calls().kit_open == 2);
    auto note = source.glove_note(Hand::Left);
    REQUIRE(note);
    CHECK(note->kind == Source::GloveNote::Kind::Waiting);
    CHECK_THAT(note->text, ContainsSubstring("RH_ERR_BUSY: E1102 PortBusyError (another process holds the glove's serial "
                                             "port)"));

    // The first poll starts the retry clock; attempts follow every 2 s, for each hand.
    SourceBatch batch = source.poll(kStart);
    CHECK(batch.records.empty());
    CHECK_FALSE(batch.finished);
    CHECK(fake_rh::calls().kit_open == 2);
    CHECK(source.poll(kStart + 1'999 * kMs).records.empty());
    CHECK(fake_rh::calls().kit_open == 2);
    CHECK(source.poll(kStart + 2'000 * kMs).records.empty());
    CHECK(fake_rh::calls().kit_open == 4);
    CHECK_FALSE(source.opened(Hand::Left));

    batch = source.poll(kStart + 4'000 * kMs);
    CHECK(fake_rh::calls().kit_open == 6);
    CHECK(source.opened(Hand::Left));
    CHECK_FALSE(source.opened(Hand::Right));
    CHECK_FALSE(source.glove_note(Hand::Left));
    CHECK(records_of<HandMeta>(batch).size() == 1);
    CHECK(records_of<GloveSample>(batch).empty());
    CHECK_FALSE(batch.hand_restarted[0]);

    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 4'005 * kMs));
    batch = source.poll(kStart + 4'010 * kMs);
    CHECK(records_of<HandMeta>(batch).empty());
    CHECK(records_of<GloveSample>(batch).size() == 1);
    CHECK(fake_rh::calls().kit_open == 6);
}

TEST_CASE("live source fails fast when retrying cannot help either glove", "[revohuman][open]")
{
    FakeSdk sdk;
    SECTION("bad options")
    {
        fake_rh::fail_next_open(RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: rate_hz must be positive", 3);
        CHECK_THROWS_WITH(
            LiveSource(inline_config()), ContainsSubstring("cannot open either glove: left: RH_ERR_INVALID_ARG: E1000 "
                                                           "RevoHumanError; right: RH_ERR_INVALID_ARG: E1000"));
    }
    SECTION("a product conflict")
    {
        fake_rh::fail_next_open(RH_ERR_CONFLICT, RH_E_PRODUCT_MISMATCH, "ProductMismatch: declared P1, found PTV1", 3);
        CHECK_THROWS_WITH(LiveSource(inline_config()), ContainsSubstring("E1804 ProductMismatch"));
    }
    SECTION("incompatible firmware")
    {
        fake_rh::fail_next_open(RH_ERR_PROTOCOL, RH_E_FIRMWARE_INCOMPATIBLE, "FirmwareIncompatible: too old", 3);
        CHECK_THROWS_WITH(LiveSource(inline_config()), ContainsSubstring("E1205 FirmwareIncompatible"));
    }
    // Both together, then each on its own.
    CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_BOTH, RH_HANDS_LEFT, RH_HANDS_RIGHT });
    CHECK(fake_rh::calls().kit_close == 0);
    // The SDK is shut down although the constructor threw.
    CHECK(fake_rh::calls().shutdown == 1);
    CHECK_FALSE(fake_rh::calls().log_callback_set);
}

TEST_CASE("live source keeps retrying a glove it cannot assign to a hand, and says why", "[revohuman][open]")
{
    FakeSdk sdk;
    // The SDK reports it with the status of an absent glove; its own text stays out of the note.
    const std::string sdk_text = "\xe6\x97\xa0\xe6\xb3\x95\xe7\xa1\xae\xe5\xae\x9a\xe6\x89\x8b\xe5\x88\xab";
    fake_rh::fail_next_open(
        RH_HAND_LEFT, RH_ERR_NOT_FOUND, RH_E_UNREGISTERED_GLOVE, "UnregisteredGloveError: " + sdk_text, 2);
    LiveSource source(left_only());
    CHECK_FALSE(source.opened(Hand::Left));
    auto note = source.glove_note(Hand::Left);
    REQUIRE(note);
    CHECK(note->kind == Source::GloveNote::Kind::Waiting);
    CHECK(note->text ==
          "waiting for the glove: RH_ERR_NOT_FOUND: E1400 UnregisteredGloveError (a glove is plugged in, "
          "but its MCU serial number does not say which hand it is)");

    source.poll(kStart);
    source.poll(kStart + 2'000 * kMs);
    CHECK_FALSE(source.opened(Hand::Left));
    source.poll(kStart + 4'000 * kMs);
    CHECK(source.opened(Hand::Left));
    CHECK_FALSE(source.glove_note(Hand::Left));
}

TEST_CASE("live source retries a glove only when opening it may succeed later", "[revohuman][open]")
{
    struct Case
    {
        RhStatus status;
        RhErrorCode code;
        bool retried;
    };
    const Case cases[] = {
        { RH_ERR_NOT_FOUND, RH_E_REVOHUMAN_ERROR, true },
        { RH_ERR_TIMEOUT, RH_E_TRANSPORT_ERROR, true },
        { RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, true },
        { RH_ERR_BUSY, RH_E_PORT_BUSY, true },
        { RH_ERR_STATE, RH_E_REVOHUMAN_ERROR, true },
        { RH_ERR_IO, RH_E_TRANSPORT_ERROR, true },
        { RH_ERR_PROTOCOL, RH_E_PROTOCOL_ERROR, true },
        { RH_ERR_PROTOCOL, RH_E_FIRMWARE_INCOMPATIBLE, false },
        { RH_ERR_PROTOCOL, RH_E_TIME_SYNC_INCOMPATIBLE, false },
        { RH_ERR_PROTOCOL, RH_E_TIME_SYNC_UNSUPPORTED, false },
        { RH_ERR_PROTOCOL, RH_E_UNSUPPORTED_FEATURE, false },
        { RH_ERR_CONFLICT, RH_E_REVOHUMAN_ERROR, true },
        { RH_ERR_CONFLICT, RH_E_PRODUCT_MISMATCH, false },
        { RH_ERR_CONFLICT, RH_E_VARIANT_MISMATCH, false },
        { RH_ERR_CONFLICT, RH_E_STREAM_RATE_CONFLICT, false },
        { RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, false },
        { RH_ERR_UNSUPPORTED, RH_E_UNSUPPORTED_FEATURE, false },
        { RH_ERR_INVALID_DATA, RH_E_PROTOCOL_ERROR, false },
        { RH_ERR_INTERNAL, RH_E_REVOHUMAN_ERROR, false },
    };
    for (const Case& c : cases)
    {
        CAPTURE(c.status, c.code);
        FakeSdk sdk;
        fake_rh::fail_next_open(RH_HAND_LEFT, c.status, c.code, "RevoHumanError: scripted");
        // The right glove is absent, so the source starts even when the left one is given up.
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == (c.retried ? Source::GloveNote::Kind::Waiting : Source::GloveNote::Kind::Ended));
        // Only the error class reaches the note; the SDK's own text goes to the debug log.
        CHECK_THAT(note->text, ContainsSubstring("E" + std::to_string(c.code) + " RevoHumanError"));

        source.poll(kStart);
        source.poll(kStart + 2'000 * kMs);
        CHECK(source.opened(Hand::Left) == c.retried);
        CHECK(fake_rh::open_kits() == (c.retried ? 1 : 0));
    }
}

TEST_CASE("live source gives up on a glove whose joint names it cannot read", "[revohuman][open]")
{
    FakeSdk sdk;
    SECTION("rh_joint_name fails")
    {
        fake_rh::set_joint_name(RH_HAND_LEFT, 7, std::nullopt);
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK_THAT(note->text, ContainsSubstring("rh_joint_name 7 failed: E1000 RevoHumanError"));
    }
    SECTION("a name longer than the buffer")
    {
        fake_rh::set_joint_name(RH_HAND_LEFT, 3, std::string(200, 'j'));
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK_THAT(note->text, ContainsSubstring("rh_joint_name 3 is longer than"));
    }
    // The kit that was opened for it is closed again, and it is not tried again.
    CHECK(fake_rh::open_kits() == 0);
    CHECK(fake_rh::calls().kit_close == 1);
}

TEST_CASE("live source shuts the SDK down after closing every handle", "[revohuman][live]")
{
    FakeSdk sdk;
    {
        LiveSource source(inline_config());
        source.poll(kStart);
        CHECK(fake_rh::calls().shutdown == 0);
    }
    const fake_rh::Calls& calls = fake_rh::calls();
    CHECK(calls.shutdown == 1);
    CHECK(calls.kits_open_at_shutdown == 0);
    // Removed after rh_shutdown, which may still log about handles it closes.
    CHECK(calls.log_callback_set_at_shutdown);
    CHECK_FALSE(calls.log_callback_set);
}

TEST_CASE("live source leaves the SDK running while another live source uses it", "[revohuman][live]")
{
    FakeSdk sdk;
    {
        LiveSource first(left_only());
        {
            // The glove is in the first source's kit, so this one keeps waiting for it.
            LiveSource second(inline_config());
            CHECK_FALSE(second.opened(Hand::Left));
            CHECK(fake_rh::calls().init == 1);
        }
        CHECK(fake_rh::calls().shutdown == 0);
        CHECK(fake_rh::calls().log_callback_set);
        CHECK(first.opened(Hand::Left));
        CHECK(fake_rh::open_kits() == 1);
    }
    CHECK(fake_rh::calls().shutdown == 1);
    CHECK(fake_rh::calls().kits_open_at_shutdown == 0);
}

TEST_CASE("live source gives up on one glove and streams the other", "[revohuman][open]")
{
    FakeSdk sdk;
    // The kit of both fails on the left glove, and so does the left glove's own.
    fake_rh::fail_next_open(
        RH_HAND_LEFT, RH_ERR_PROTOCOL, RH_E_FIRMWARE_INCOMPATIBLE, "FirmwareIncompatible: too old", 2);
    LiveSource source(inline_config());
    CHECK_FALSE(source.opened(Hand::Left));
    CHECK(source.opened(Hand::Right));
    const auto note = source.glove_note(Hand::Left);
    REQUIRE(note);
    CHECK(note->kind == Source::GloveNote::Kind::Ended);
    CHECK_THAT(note->text, ContainsSubstring("E1205 FirmwareIncompatible"));

    fake_rh::push(RH_HAND_RIGHT, fake_rh::frame(RH_HAND_RIGHT, kStart));
    SourceBatch batch = source.poll(kStart);
    CHECK(records_of<GloveSample>(batch).size() == 1);
    batch = source.poll(kStart + 10'000 * kMs);
    CHECK_FALSE(batch.finished);
    // A given-up glove is not tried again.
    CHECK(fake_rh::calls().kit_open == 3);
}

TEST_CASE("live source retries a re-enumerated port but not a product conflict", "[revohuman][open]")
{
    FakeSdk sdk;
    fake_rh::fail_next_open(RH_ERR_CONFLICT, RH_E_REVOHUMAN_ERROR, "RevoHumanError: port re-enumerated, retry", 3);
    LiveSource source(inline_config());
    CHECK_FALSE(source.opened(Hand::Left));
    CHECK_FALSE(source.opened(Hand::Right));

    // Both are looked for together again.
    source.poll(kStart);
    fake_rh::fail_next_open(RH_ERR_CONFLICT, RH_E_VARIANT_MISMATCH, "VariantMismatch: declared PTV1, found P1", 3);
    SourceBatch batch = source.poll(kStart + 2'000 * kMs);
    CHECK(fake_rh::calls().open_hands == std::vector<RhHands>{ RH_HANDS_BOTH, RH_HANDS_LEFT, RH_HANDS_RIGHT,
                                                               RH_HANDS_BOTH, RH_HANDS_LEFT, RH_HANDS_RIGHT });
    CHECK(batch.finished);
    for (const Hand hand : { Hand::Left, Hand::Right })
    {
        const auto note = source.glove_note(hand);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK_THAT(note->text, ContainsSubstring("E1802 VariantMismatch"));
    }
    // Giving up is final.
    CHECK(source.poll(kStart + 10'000 * kMs).finished);
    CHECK(fake_rh::calls().kit_open == 6);
}

TEST_CASE("live source looks again for a glove that went away while it was subscribing", "[revohuman][open]")
{
    FakeSdk sdk;
    SECTION("unplugged after the kit opened")
    {
        fake_rh::fail_next_subscribe(
            RH_HAND_LEFT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: unplugged");
        LiveSource source(left_only());
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(fake_rh::open_kits() == 0);
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Waiting);
        CHECK_THAT(note->text, ContainsSubstring("rh_glove_subscribe failed"));

        source.poll(kStart);
        const SourceBatch batch = source.poll(kStart + 2'000 * kMs);
        CHECK(source.opened(Hand::Left));
        CHECK(records_of<HandMeta>(batch).size() == 1);
        CHECK(fake_rh::calls().subscribe == 2);
    }
    SECTION("but not when the SDK refuses the subscription")
    {
        // In the kit of both, then in its own.
        fake_rh::fail_next_subscribe(
            RH_HAND_LEFT, RH_ERR_INVALID_ARG, RH_E_REVOHUMAN_ERROR, "RevoHumanError: queue_size", 2);
        LiveSource source(inline_config());
        CHECK_FALSE(source.opened(Hand::Left));
        CHECK(source.opened(Hand::Right));
        const auto note = source.glove_note(Hand::Left);
        REQUIRE(note);
        CHECK(note->kind == Source::GloveNote::Kind::Ended);
        CHECK(fake_rh::open_kits() == 1);
    }
}

TEST_CASE("live source opens a glove in the background without holding up the other", "[revohuman][open]")
{
    FakeSdk sdk;
    fake_rh::set_present(RH_HAND_RIGHT, false);
    fake_rh::set_variant(RH_HAND_RIGHT, "PTV1");
    const EchoFactory factory;
    LiveSource source(LiveSourceConfig{}, factory.make());
    REQUIRE(source.opened(Hand::Left));
    source.poll(kStart);

    // The right glove is plugged in and takes 300 ms to open.
    fake_rh::set_open_delay_ms(RH_HAND_RIGHT, 300);
    fake_rh::set_present(RH_HAND_RIGHT, true);
    auto t0 = WallClock::now();
    source.poll(kStart + 2'000 * kMs);
    CHECK(WallClock::now() - t0 < std::chrono::milliseconds(100));
    CHECK_FALSE(source.opened(Hand::Right));

    // The left glove's samples go on meanwhile.
    fake_rh::push(RH_HAND_LEFT, fake_rh::frame(RH_HAND_LEFT, kStart + 2'005 * kMs));
    t0 = WallClock::now();
    const auto left = records_of<GloveSample>(source.poll(kStart + 2'005 * kMs));
    CHECK(WallClock::now() - t0 < std::chrono::milliseconds(100));
    CHECK(left.size() == 1);

    std::vector<SourceBatch> batches;
    REQUIRE(poll_until(source, kStart + 2'010 * kMs, batches, [&] { return source.opened(Hand::Right); }));
    const auto metas = records_of<HandMeta>(batches.back());
    REQUIRE(metas.size() == 1);
    CHECK(metas[0].hand == Hand::Right);
    CHECK_FALSE(batches.back().hand_restarted[1]);
    CHECK(fake_rh::snapshot().subscribe == 2);
    // Its kinematics were made on the worker, for the product identified there.
    CHECK(factory.asked() == std::vector<std::string>{ "P1", "PTV1" });
    CHECK(source.model_id(Hand::Right) == "echo:PTV1");

    // Its kit is closed in the background too once its stream ends.
    fake_rh::end(RH_HAND_RIGHT, RH_ERR_DISCONNECTED, RH_E_DISCONNECTED, "TransportDisconnected: unplugged");
    fake_rh::set_present(RH_HAND_RIGHT, false);
    batches.clear();
    REQUIRE(poll_until(source, kStart + 3'000 * kMs, batches, [] { return fake_rh::open_kits() == 1; }));
    CHECK_FALSE(source.opened(Hand::Right));
    CHECK(source.opened(Hand::Left));
}

TEST_CASE("live source closes a glove that finished opening after it was destroyed", "[revohuman][open]")
{
    FakeSdk sdk;
    fake_rh::set_present(RH_HAND_RIGHT, false);
    {
        LiveSource source(LiveSourceConfig{});
        source.poll(kStart);
        fake_rh::set_open_delay_ms(RH_HAND_RIGHT, 200);
        fake_rh::set_present(RH_HAND_RIGHT, true);
        const int started = fake_rh::opens_started();
        source.poll(kStart + 2'000 * kMs);
        // Wait until the worker has started the open, which then outlives the source.
        const auto deadline = WallClock::now() + std::chrono::seconds(5);
        while (fake_rh::opens_started() == started && WallClock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(fake_rh::opens_started() > started);
    }
    const fake_rh::Calls calls = fake_rh::snapshot();
    CHECK(fake_rh::open_kits() == 0);
    CHECK(calls.kit_open == 3);
    CHECK(calls.kit_close == 2);
    CHECK(calls.sub_close == 2);
    CHECK(calls.sub_close_after_kit_close == 0);
}

TEST_CASE("rh_error_text falls back to the error code without a message", "[revohuman][live]")
{
    FakeSdk sdk;
    CHECK(rh_error_text() == "E0");
}
