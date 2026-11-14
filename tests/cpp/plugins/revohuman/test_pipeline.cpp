// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/pipeline.hpp>
#include <schema/joint_state_generated.h>
#include <schema/tactile_array_generated.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace plugins::revohuman;
using Catch::Approx;

namespace
{

constexpr std::int64_t kMs = 1'000'000;

struct Call
{
    std::string kind;
    std::vector<std::uint8_t> buffer;
    std::int64_t t_ns = 0;
    std::int64_t raw_ns = 0;
    HandJointPoses joints;
};

class RecordingSink
{
public:
    //! Removes the hand when destroyed, like the plugin's handle.
    class MockHand
    {
    public:
        explicit MockHand(RecordingSink& sink) : sink_(&sink)
        {
            sink_->calls.push_back({ "create" });
        }
        MockHand(MockHand&& other) noexcept : sink_(std::exchange(other.sink_, nullptr))
        {
        }
        MockHand& operator=(MockHand&& other) noexcept
        {
            if (this != &other)
            {
                release();
                sink_ = std::exchange(other.sink_, nullptr);
            }
            return *this;
        }
        ~MockHand()
        {
            release();
        }
        bool push(const HandJointPoses& joints, std::int64_t t)
        {
            sink_->calls.push_back({ "hand", {}, t, 0, joints });
            return sink_->wrist_tracked;
        }

    private:
        void release()
        {
            if (sink_)
                std::exchange(sink_, nullptr)->calls.push_back({ "destroy" });
        }
        RecordingSink* sink_;
    };

    void push_joint_state(std::span<const std::uint8_t> b, std::int64_t t)
    {
        calls.push_back({ "joint_state", { b.begin(), b.end() }, t, t, {} });
    }
    void push_tactile(std::span<const std::uint8_t> b, std::int64_t t, std::int64_t raw)
    {
        calls.push_back({ "tactile", { b.begin(), b.end() }, t, raw, {} });
    }
    void clear_tactile()
    {
        calls.push_back({ "clear_tactile" });
    }
    void push_shape(std::span<const std::uint8_t> b, std::int64_t t)
    {
        calls.push_back({ "shape", { b.begin(), b.end() }, t, t, {} });
    }
    void clear_shape()
    {
        calls.push_back({ "clear_shape" });
    }
    MockHand create_hand()
    {
        return MockHand(*this);
    }

    std::vector<std::string> kinds() const
    {
        std::vector<std::string> out;
        for (const auto& c : calls)
            out.push_back(c.kind);
        return out;
    }
    std::vector<Call> of(const std::string& kind) const
    {
        std::vector<Call> out;
        for (const auto& c : calls)
            if (c.kind == kind)
                out.push_back(c);
        return out;
    }

    std::vector<Call> calls;
    bool wrist_tracked = false; //!< What MockHand::push() reports about the host's wrist pose.
};

HandMeta meta(bool tactile = true)
{
    HandMeta m;
    m.product_code = tactile ? "PTV1" : "P1";
    m.device_id = "SN1";
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
        m.joint_names[i] = "j" + std::to_string(i);
    m.has_tactile = tactile;
    if (tactile)
    {
        m.tactile_metadata = std::vector<std::uint8_t>{ 1, 2, 3 };
        m.taxel_count = 2;
    }
    return m;
}

GloveSample glove(std::int64_t t_ns)
{
    GloveSample s;
    s.sample_mono_ns = t_ns;
    s.joints_valid_mask = 1u;
    s.joints_rad[0] = 0.5;
    s.joint_poses.joints[2].mutable_position() = core::Point(0.1f, 0, 0);
    s.joint_poses.position_valid_mask = s.joint_poses.orientation_valid_mask = (1u << kHandJointCount) - 1u;
    return s;
}

// Newtons per taxel; TactileSample only views them, so they outlive every sample.
constexpr std::array<float, 2> kTwoTaxels{ 1.5f, 0.0f };
constexpr std::array<float, 1> kOneTaxel{ 0.001f };

GloveSample with_tactile(GloveSample s, std::uint32_t tick, std::int64_t t_ns, std::span<const float> newtons = kTwoTaxels)
{
    s.tactile = TactileSample{ tick, t_ns, true, true, newtons };
    return s;
}

struct Fixture
{
    RecordingSink sink;
    HandPipeline<RecordingSink> pipeline{ Hand::Left, sink };
};

} // namespace

TEST_CASE("pipeline: joint states go out at the sample time, never ahead of now", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(glove(10 * kMs), 20 * kMs);
    f.pipeline.on_glove(glove(30 * kMs), 25 * kMs);
    const auto states = f.sink.of("joint_state");
    REQUIRE(states.size() == 2);
    CHECK(states[0].t_ns == 10 * kMs);
    CHECK(states[1].t_ns == 25 * kMs);
    const auto* out = flatbuffers::GetRoot<core::JointStateOutput>(states[0].buffer.data());
    CHECK(out->device_id()->str() == "SN1");
    REQUIRE(out->joints()->size() == kGloveJointCount);
    CHECK(out->joints()->Get(0)->position() == Approx(0.5));
    CHECK(out->joints()->Get(0)->valid());
    CHECK_FALSE(out->joints()->Get(1)->valid());
}

TEST_CASE("pipeline: the hand goes out in the WRIST frame and is removed after a stall", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta(false));
    f.sink.wrist_tracked = true;
    f.pipeline.on_glove(glove(10 * kMs), 10 * kMs);
    f.pipeline.on_glove(glove(15 * kMs), 15 * kMs);
    CHECK(f.sink.of("create").size() == 1);
    const auto hands = f.sink.of("hand");
    REQUIRE(hands.size() == 2);
    CHECK(hands[0].t_ns == 10 * kMs);
    CHECK(hands[0].joints.joints[2].position().x() == Approx(0.1)); // The sink places it at the host's wrist.
    CHECK(f.pipeline.health(16 * kMs).wrist_tracked);
    f.sink.wrist_tracked = false;
    f.pipeline.on_glove(glove(20 * kMs), 20 * kMs);
    CHECK_FALSE(f.pipeline.health(21 * kMs).wrist_tracked);

    f.pipeline.on_tick(20 * kMs + kStallNs + 1);
    CHECK(f.sink.kinds().back() == "destroy");
    CHECK_FALSE(f.pipeline.health(80 * kMs).hand_active);
}

TEST_CASE("pipeline: before the first glove sample the hand is held with no valid joints", "[unit]")
{
    Fixture f;
    f.pipeline.on_tick(0);
    f.pipeline.on_tick(1 * kMs);
    f.pipeline.on_tick(5 * kMs);
    CHECK(f.sink.kinds() == std::vector<std::string>{ "create", "hand", "hand" });
    for (const auto& held : f.sink.of("hand"))
    {
        CHECK(held.joints.position_valid_mask == 0u);
        CHECK(held.joints.orientation_valid_mask == 0u);
    }
    CHECK_FALSE(f.pipeline.health(5 * kMs).hand_active);

    // The held hand becomes the glove's, and holding ends with the first sample.
    f.pipeline.set_meta(meta(false));
    f.pipeline.on_glove(glove(6 * kMs), 6 * kMs);
    f.pipeline.on_tick(20 * kMs);
    const auto kinds = f.sink.kinds();
    CHECK(std::count(kinds.begin(), kinds.end(), "create") == 1);
    CHECK(std::count(kinds.begin(), kinds.end(), "destroy") == 0);
    CHECK(f.sink.of("hand").back().joints.position_valid_mask != 0u);
    CHECK(f.pipeline.health(7 * kMs).hand_active);
}

TEST_CASE("pipeline: a held hand is released when no glove streams in time", "[unit]")
{
    Fixture f;
    const std::int64_t hold = kStartupHoldNs;
    f.pipeline.on_tick(10 * kMs);
    f.pipeline.on_tick(10 * kMs + hold);
    f.pipeline.on_tick(20 * kMs + hold);
    CHECK(f.sink.kinds() == std::vector<std::string>{ "create", "hand", "destroy" });

    // A glove streaming later is injected as usual.
    f.pipeline.set_meta(meta(false));
    f.pipeline.on_glove(glove(30 * kMs + hold), 30 * kMs + hold);
    CHECK(f.sink.kinds().back() == "hand");
    CHECK(f.sink.of("create").size() == 2);
}

TEST_CASE("pipeline: forces go out in newtons after their TactileMetadata, once per tick", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(with_tactile(glove(10 * kMs), 7, 9 * kMs), 10 * kMs);
    f.pipeline.on_glove(with_tactile(glove(15 * kMs), 7, 9 * kMs), 15 * kMs); // Same snapshot again.
    const auto kinds = f.sink.kinds();
    CHECK(std::find(kinds.begin(), kinds.end(), "shape") < std::find(kinds.begin(), kinds.end(), "tactile"));
    const auto tactile = f.sink.of("tactile");
    REQUIRE(tactile.size() == 1);
    CHECK(tactile[0].t_ns == 9 * kMs);
    CHECK(tactile[0].raw_ns == 7 * kMs);
    const auto* forces = flatbuffers::GetRoot<core::TactileDataArray>(tactile[0].buffer.data())->taxel_forces();
    REQUIRE(forces->size() == 2);
    CHECK(forces->Get(0)->z() == Approx(1.5f));
    CHECK(forces->Get(0)->x() == 0.0f);
    CHECK(f.pipeline.health(12 * kMs).tactile_live);

    // Forces expire after their maximum age; the metadata stays.
    f.pipeline.on_tick(9 * kMs + kTactileMaxAgeNs + 1);
    CHECK(f.sink.kinds().back() == "clear_tactile");
    const auto h = f.pipeline.health(200 * kMs);
    CHECK_FALSE(h.tactile_live);
    CHECK(h.tactile_skip == TactileSkip::Stale);
}

TEST_CASE("pipeline: tactile stamped ahead of now goes out at now, like the joint states", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(with_tactile(glove(10 * kMs), 1, 12 * kMs), 10 * kMs);
    const auto tactile = f.sink.of("tactile");
    REQUIRE(tactile.size() == 1);
    CHECK(tactile[0].t_ns == 10 * kMs);
}

TEST_CASE("pipeline: unusable tactile samples are skipped with a reason", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    auto uncalibrated = with_tactile(glove(10 * kMs), 1, 10 * kMs);
    uncalibrated.tactile->calibrated = false;
    f.pipeline.on_glove(uncalibrated, 10 * kMs);
    CHECK(f.pipeline.health(10 * kMs).tactile_skip == TactileSkip::Uncalibrated);
    f.pipeline.on_glove(with_tactile(glove(20 * kMs), 2, 20 * kMs, kOneTaxel), 20 * kMs);
    CHECK(f.pipeline.health(20 * kMs).tactile_skip == TactileSkip::CountMismatch);
    f.pipeline.on_glove(with_tactile(glove(400 * kMs), 3, 200 * kMs), 400 * kMs);
    CHECK(f.pipeline.health(400 * kMs).tactile_skip == TactileSkip::Stale);
    CHECK(f.sink.of("tactile").empty());

    // A failed board read is re-sent marked invalid; fresh forces stay live through it.
    f.pipeline.on_glove(with_tactile(glove(500 * kMs), 4, 500 * kMs), 500 * kMs);
    auto invalid = with_tactile(glove(505 * kMs), 4, 500 * kMs);
    invalid.tactile->valid = false;
    f.pipeline.on_glove(invalid, 505 * kMs);
    CHECK(f.pipeline.health(505 * kMs).tactile_live);
    CHECK(f.pipeline.health(700 * kMs).tactile_skip == TactileSkip::Invalid);
}

TEST_CASE("pipeline: a glove without tactile sensors removes the earlier glove's tactile", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(with_tactile(glove(10 * kMs), 1, 10 * kMs), 10 * kMs);
    f.pipeline.reset();
    f.pipeline.set_meta(meta(false));
    const auto kinds = f.sink.kinds();
    CHECK(std::count(kinds.begin(), kinds.end(), "destroy") == 1);
    CHECK(std::count(kinds.begin(), kinds.end(), "clear_tactile") == 1);
    CHECK(kinds.back() == "clear_shape");
    CHECK(f.pipeline.health(20 * kMs).tactile_skip == TactileSkip::NoTactile);

    HandMeta no_layout = meta();
    no_layout.tactile_metadata.reset();
    f.pipeline.set_meta(no_layout);
    CHECK(f.pipeline.health(20 * kMs).tactile_skip == TactileSkip::NoShape);
}

TEST_CASE("pipeline: a reopened glove keeps nothing of its earlier meta", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(with_tactile(glove(10 * kMs), 1, 10 * kMs), 10 * kMs);
    f.pipeline.reset();
    CHECK(f.sink.kinds() == std::vector<std::string>{ "joint_state", "shape", "tactile", "create", "hand", "destroy",
                                                      "clear_tactile", "clear_shape" });
    // The source sets no meta for a reopened stream it cannot use: the earlier TactileMetadata is not resent and the
    // earlier glove's tactile reasons are gone.
    f.pipeline.on_tick(10 * kMs + kShapeResendNs + 1);
    CHECK(f.sink.of("shape").size() == 1);
    const auto h = f.pipeline.health(20 * kMs);
    CHECK_FALSE(h.hand_active);
    CHECK(h.tactile_skip == TactileSkip::None);
}

TEST_CASE("pipeline: a new stream generation may reuse tactile ticks", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(with_tactile(glove(10 * kMs), 50, 10 * kMs), 10 * kMs);
    auto restarted = with_tactile(glove(20 * kMs), 5, 20 * kMs);
    restarted.generation = 1;
    f.pipeline.on_glove(restarted, 20 * kMs);
    CHECK(f.sink.of("tactile").size() == 2);
}

TEST_CASE("pipeline: split tactile samples go out at their own rate, between glove samples", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_glove(glove(10 * kMs), 10 * kMs);
    f.pipeline.on_tactile(0, TactileSample{ 8, 8 * kMs, true, true, kTwoTaxels }, 11 * kMs);
    f.pipeline.on_tactile(0, TactileSample{ 12, 12 * kMs, true, true, kTwoTaxels }, 13 * kMs);
    f.pipeline.on_glove(glove(15 * kMs), 15 * kMs);
    f.pipeline.on_tactile(0, TactileSample{ 13, 13 * kMs, true, true, kTwoTaxels }, 16 * kMs);
    const auto kinds = f.sink.kinds();
    CHECK(std::find(kinds.begin(), kinds.end(), "shape") < std::find(kinds.begin(), kinds.end(), "tactile"));
    const auto tactile = f.sink.of("tactile");
    REQUIRE(tactile.size() == 3);
    CHECK(tactile[0].raw_ns == 8 * kMs);
    CHECK(tactile[1].raw_ns == 12 * kMs);
    CHECK(tactile[2].t_ns == 13 * kMs);
    CHECK(f.pipeline.health(16 * kMs).tactile_live);

    // A health-only sample while the tactile board is unreadable: no forces, and they age out.
    f.pipeline.on_tactile(0, TactileSample{ 14, std::nullopt, false, true, {} }, 17 * kMs);
    CHECK(f.sink.of("tactile").size() == 3);
    CHECK(f.pipeline.health(200 * kMs).tactile_skip == TactileSkip::Invalid);
}

TEST_CASE("pipeline: split tactile of a new stream generation may reuse ticks", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta());
    f.pipeline.on_tactile(0, TactileSample{ 50, 10 * kMs, true, true, kTwoTaxels }, 10 * kMs);
    f.pipeline.on_tactile(1, TactileSample{ 5, 20 * kMs, true, true, kTwoTaxels }, 20 * kMs);
    const auto kinds = f.sink.kinds();
    CHECK(std::count(kinds.begin(), kinds.end(), "clear_tactile") == 1);
    CHECK(f.sink.of("tactile").size() == 2);
    // The glove samples of that generation do not restart it, and their copy of the latest sample is not pushed again.
    auto same_generation = with_tactile(glove(25 * kMs), 5, 20 * kMs);
    same_generation.generation = 1;
    f.pipeline.on_glove(same_generation, 25 * kMs);
    CHECK(f.sink.of("tactile").size() == 2);
    CHECK(f.sink.of("clear_tactile").size() == 1);
}

TEST_CASE("pipeline: split tactile of a glove without tactile sensors is ignored", "[unit]")
{
    Fixture f;
    f.pipeline.set_meta(meta(false));
    f.pipeline.on_tactile(0, TactileSample{ 1, 10 * kMs, true, true, kTwoTaxels }, 10 * kMs);
    CHECK(f.sink.of("tactile").empty());
    CHECK(f.sink.of("shape").empty());
}
