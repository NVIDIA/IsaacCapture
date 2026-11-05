// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_shapes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <revohuman_plugin/event_log.hpp>
#include <revohuman_plugin/pipeline.hpp>
#include <revohuman_plugin/pose_math.hpp>
#include <schema/tactile_array_generated.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
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
    HandFrame frame;
};

class RecordingSink : public PushSink
{
public:
    void push_joint_state(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) override
    {
        calls.push_back({ "joint_state", buffer, sample_ns, raw_ns, {} });
    }
    void push_tactile(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) override
    {
        calls.push_back({ "tactile", buffer, sample_ns, raw_ns, {} });
    }
    void clear_tactile() override
    {
        calls.push_back({ "clear_tactile", {}, 0, 0, {} });
    }
    void push_shape(const std::vector<std::uint8_t>& buffer, std::int64_t now_ns) override
    {
        calls.push_back({ "shape", buffer, now_ns, now_ns, {} });
    }
    void clear_shape() override
    {
        calls.push_back({ "clear_shape", {}, 0, 0, {} });
    }
    void hand_create() override
    {
        calls.push_back({ "create", {}, 0, 0, {} });
    }
    void hand_destroy() override
    {
        calls.push_back({ "destroy", {}, 0, 0, {} });
    }
    void hand_push(const HandFrame& frame) override
    {
        calls.push_back({ "hand", {}, frame.sample_mono_ns, 0, frame });
    }

    std::vector<Call> of(const std::string& kind) const
    {
        std::vector<Call> out;
        for (const auto& c : calls)
        {
            if (c.kind == kind)
            {
                out.push_back(c);
            }
        }
        return out;
    }
    //! Kinds of the hand injector calls, in order.
    std::vector<std::string> hand_calls() const
    {
        std::vector<std::string> out;
        for (const auto& c : calls)
        {
            if (c.kind == "create" || c.kind == "destroy" || c.kind == "hand")
            {
                out.push_back(c.kind);
            }
        }
        return out;
    }

    std::vector<Call> calls;
};

template <typename T>
const T* root_of(const std::vector<std::uint8_t>& bytes)
{
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(verifier.VerifyBuffer<T>(nullptr));
    return flatbuffers::GetRoot<T>(bytes.data());
}

HandMeta meta()
{
    HandMeta m;
    m.hand = Hand::Left;
    m.product_code = "PTV1";
    m.device_id = "revohuman_left";
    for (std::size_t i = 0; i < kGloveJointCount; ++i)
    {
        m.joint_names[i] = "l" + std::to_string(i);
    }
    return m;
}

//! A P1 glove: joints only, no tactile sensors.
HandMeta meta_without_tactile()
{
    HandMeta m = meta();
    m.product_code = "P1";
    m.has_tactile = false;
    return m;
}

//! Index of the first call of @p kind, or the number of calls when there is none.
std::size_t first_call(const RecordingSink& sink, const std::string& kind)
{
    std::size_t i = 0;
    while (i < sink.calls.size() && sink.calls[i].kind != kind)
    {
        ++i;
    }
    return i;
}

//! Joint poses equal to the bind poses of the synthetic shape, so the skinned normals are the
//! bind normals.
HandKinematics bind_kinematics()
{
    HandKinematics k;
    for (std::uint32_t j = 0; j < kHandJointCount; ++j)
    {
        k.joints[j] = revohuman_test::bind_pose(j);
        k.radius[j] = 0.01f;
    }
    k.joints[kWristJoint] = Pose{};
    k.position_valid_mask = k.orientation_valid_mask = (1u << kHandJointCount) - 1;
    return k;
}

//! The bind poses turned by angle_rad about Z.
HandKinematics turned_kinematics(double angle_rad)
{
    HandKinematics k = bind_kinematics();
    const auto turn = revohuman_test::axis_angle({ 0.0, 0.0, 1.0 }, angle_rad);
    for (auto& joint : k.joints)
    {
        joint.q = quat_mul(turn, joint.q);
    }
    return k;
}

//! Taxel-local force is invariant under hand motion.
void require_local_pressure(const Call& call)
{
    const auto* data = root_of<core::TactileDataArray>(call.buffer);
    const auto* f = data->taxel_forces()->Get(0);
    REQUIRE(f->x() == 0);
    REQUIRE(f->y() == 0);
    REQUIRE(f->z() == 100.0f);
}

GloveSample glove_at(std::int64_t t_ns, std::uint32_t tick = 1000, bool with_kinematics = true)
{
    GloveSample g;
    g.hand = Hand::Left;
    g.generation = 1;
    g.device_tick = tick;
    g.joints_valid_mask = (1u << kGloveJointCount) - 1;
    g.has_sample_time = true;
    g.sample_mono_ns = t_ns;
    if (with_kinematics)
    {
        g.kinematics = bind_kinematics();
    }
    return g;
}

TactileSample tactile(std::size_t count, std::int64_t t_ns)
{
    TactileSample t;
    t.device_tick = 500;
    t.has_sample_time = true;
    t.sample_mono_ns = t_ns;
    t.valid = true;
    t.calibrated = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        t.values.push_back(static_cast<std::uint16_t>(100 + 7 * i));
    }
    return t;
}

//! Wrist source stub: reports the pose it is given and records the times it was asked for.
struct FakeWrist : WristProvider
{
    WristPose pose;
    std::vector<std::int64_t> asked_ns;

    WristPose wrist(Hand hand, std::int64_t sample_mono_ns) override
    {
        REQUIRE(hand == Hand::Left);
        asked_ns.push_back(sample_mono_ns);
        return pose;
    }
};

WristPose wrist_at(std::array<double, 3> p, bool tracked = true)
{
    WristPose w;
    w.pose_stage.p = p;
    w.valid = true;
    w.tracked = tracked;
    return w;
}

std::shared_ptr<const LoadedShape> shape(std::size_t taxels)
{
    return std::make_shared<const LoadedShape>(build_shape_buffer(revohuman_test::rigid_shape(taxels)));
}

//! A fresh event log file, removed again when the test ends.
struct TempLog
{
    explicit TempLog(const std::string& name)
        : path(std::filesystem::temp_directory_path() / ("revohuman_" + name + ".jsonl"))
    {
        std::filesystem::remove(path);
    }
    ~TempLog()
    {
        std::filesystem::remove(path);
    }

    //! The lines of one event, in order.
    std::vector<std::string> events(const std::string& event) const
    {
        std::ifstream in(path);
        std::vector<std::string> out;
        std::string line;
        while (std::getline(in, line))
        {
            if (line.find("\"event\":\"" + event + "\"") != std::string::npos)
            {
                out.push_back(line);
            }
        }
        return out;
    }

    std::filesystem::path path;
};

//! The raw JSON value of a field in an event line.
std::string field(const std::string& line, const std::string& key)
{
    const std::string tag = "\"" + key + "\":";
    const auto at = line.find(tag);
    REQUIRE(at != std::string::npos);
    const auto begin = at + tag.size();
    return line.substr(begin, line.find_first_of(",}", begin) - begin);
}

double number(const std::string& line, const std::string& key)
{
    return std::strtod(field(line, key).c_str(), nullptr);
}

//! Every received tactile sample is accounted for exactly once in a tactile_rate window.
void require_rate_identity(const std::string& line)
{
    REQUIRE(number(line, "samples") == number(line, "repeats") + number(line, "skipped") + number(line, "pushed"));
    REQUIRE(number(line, "skipped") ==
            number(line, "skip_stale_future") + number(line, "skip_stale_old") + number(line, "skip_other"));
}

} // namespace

TEST_CASE("glove samples before the metadata are dropped", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.on_glove(glove_at(100 * kMs), 100 * kMs);
    REQUIRE(sink.calls.empty());
    REQUIRE(p.stats().glove_before_meta == 1);

    HandMeta other = meta();
    other.hand = Hand::Right;
    p.set_meta(other);
    p.on_glove(glove_at(101 * kMs), 101 * kMs);
    REQUIRE(sink.calls.empty());
}

TEST_CASE("joint state goes out at the encoder sampling time, which also stands in for the device clock",
          "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.on_glove(glove_at(1000 * kMs, 5000), 1003 * kMs);
    p.on_glove(glove_at(1005 * kMs, 5005), 1008 * kMs);
    GloveSample untimed = glove_at(0, 5010);
    untimed.has_sample_time = false;
    p.on_glove(untimed, 1013 * kMs);

    const auto js = sink.of("joint_state");
    REQUIRE(js.size() == 3);
    // The device tick is when the frame was sent, not when the encoders were sampled.
    REQUIRE(js[0].t_ns == 1000 * kMs);
    REQUIRE(js[0].raw_ns == 1000 * kMs);
    REQUIRE(js[1].t_ns == 1005 * kMs);
    REQUIRE(js[1].raw_ns == 1005 * kMs);
    REQUIRE(js[2].t_ns == 1013 * kMs);
    REQUIRE(js[2].raw_ns == 1013 * kMs);
    REQUIRE(js[0].buffer.size() <= kJointStateMaxBytes);
}

TEST_CASE("a glove sample stamped after it arrived goes out at the time it arrived", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.on_glove(glove_at(1004 * kMs, 5000), 1003 * kMs);
    p.on_glove(glove_at(1005 * kMs, 5005), 1008 * kMs);

    const auto js = sink.of("joint_state");
    REQUIRE(js.size() == 2);
    REQUIRE(js[0].t_ns == 1003 * kMs);
    REQUIRE(js[0].raw_ns == 1003 * kMs);
    REQUIRE(js[1].t_ns == 1005 * kMs);
    const auto hands = sink.of("hand");
    REQUIRE(hands.size() == 2);
    REQUIRE(hands[0].t_ns == 1003 * kMs);
    REQUIRE(hands[1].t_ns == 1005 * kMs);
    REQUIRE(p.stats().glove_future_clamped == 1);
}

TEST_CASE("a restarted device stream re-anchors its tactile tick clock", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    GloveSample g = glove_at(0, 0xFFFFFFF0u);
    g.tactile = tactile(241, 0);
    g.tactile->device_tick = 0xFFFFFFF0u;
    p.on_glove(g, 0);
    GloveSample restarted = glove_at(5 * kMs, 10);
    restarted.generation = 2;
    restarted.tactile = tactile(241, 5 * kMs);
    restarted.tactile->device_tick = 10;
    p.on_glove(restarted, 5 * kMs);
    const auto pushed = sink.of("tactile");
    REQUIRE(pushed.size() == 2);
    REQUIRE(pushed[1].raw_ns == 10 * kMs);
    REQUIRE(p.stats().generation_changes == 1);
}

TEST_CASE("the hand is injected at the glove sample time with the wrist from the provider", "[revohuman][pipeline]")
{
    RecordingSink sink;
    FakeWrist wrist;
    wrist.pose = wrist_at({ 0.5, 1.0, -1.0 });
    HandPipeline p(Hand::Left, {}, sink);
    p.set_wrist_provider(&wrist);
    p.set_meta(meta());

    p.on_glove(glove_at(100 * kMs), 103 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand" });
    // The wrist is asked for at the encoder sampling time, not at the arrival time.
    REQUIRE(wrist.asked_ns == std::vector<std::int64_t>{ 100 * kMs });
    auto frame = sink.of("hand")[0].frame;
    REQUIRE(frame.sample_mono_ns == 100 * kMs);
    REQUIRE(frame.tracked);
    // Stage coordinates as the provider reports them, and the fingers follow the wrist.
    const auto& w = frame.joints[kWristJoint].p;
    REQUIRE(w[0] == Approx(0.5));
    REQUIRE(w[1] == Approx(1.0));
    REQUIRE(w[2] == Approx(-1.0));
    const auto bind = revohuman_test::bind_pose(3).p;
    REQUIRE(frame.joints[3].p[0] == Approx(0.5 + bind[0]));
    REQUIRE(frame.joints[3].p[1] == Approx(1.0 + bind[1]));
    REQUIRE(frame.joints[3].p[2] == Approx(-1.0 + bind[2]));
    REQUIRE(frame.radius[3] == 0.01f);
    REQUIRE(p.health(103 * kMs).wrist_tracked);

    // Every frame takes the wrist anew; the injector stays.
    wrist.pose = wrist_at({ 2.0, 0.0, 0.0 });
    p.on_glove(glove_at(105 * kMs, 1005), 108 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand", "hand" });
    frame = sink.of("hand")[1].frame;
    REQUIRE(frame.joints[kWristJoint].p[0] == Approx(2.0));
    REQUIRE(p.stats().hands_pushed == 2);
    REQUIRE(p.stats().hands_tracked == 2);
    REQUIRE(p.stats().injector_creates == 1);
}

TEST_CASE("a wrist pose the runtime only holds is pushed untracked", "[revohuman][pipeline]")
{
    RecordingSink sink;
    FakeWrist wrist;
    wrist.pose = wrist_at({ 1.0, 0.0, 0.0 }, false);
    HandPipeline p(Hand::Left, {}, sink);
    p.set_wrist_provider(&wrist);
    p.set_meta(meta());
    p.on_glove(glove_at(100 * kMs), 101 * kMs);

    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand" });
    const auto frame = sink.of("hand")[0].frame;
    REQUIRE_FALSE(frame.tracked);
    REQUIRE(frame.joints[kWristJoint].p[0] == Approx(1.0));
    REQUIRE(p.stats().hands_tracked == 0);
    const auto h = p.health(101 * kMs);
    REQUIRE(h.hand_active);
    REQUIRE_FALSE(h.wrist_tracked);
}

TEST_CASE("without a wrist pose the hand is placed at the stage origin, untracked", "[revohuman][pipeline]")
{
    RecordingSink sink;
    FakeWrist wrist;
    HandPipeline p(Hand::Left, {}, sink);
    SECTION("no provider")
    {
    }
    SECTION("the provider has no pose")
    {
        // An invalid pose is never used, whatever it holds.
        wrist.pose.pose_stage.p = { 5.0, 5.0, 5.0 };
        wrist.pose.tracked = true;
        p.set_wrist_provider(&wrist);
    }
    p.set_meta(meta());
    p.on_glove(glove_at(100 * kMs), 101 * kMs);

    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand" });
    const auto frame = sink.of("hand")[0].frame;
    REQUIRE_FALSE(frame.tracked);
    for (const double v : frame.joints[kWristJoint].p)
    {
        REQUIRE(v == Approx(0.0).margin(1e-12));
    }
    REQUIRE(p.stats().hands_tracked == 0);
    REQUIRE_FALSE(p.health(101 * kMs).wrist_tracked);
}

TEST_CASE("glove samples without joint poses never reach the injector", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.on_glove(glove_at(100 * kMs, 1000, false), 101 * kMs);
    p.on_tick(200 * kMs);
    REQUIRE(sink.hand_calls().empty());
    REQUIRE(p.stats().glove_without_kinematics == 1);
    REQUIRE(sink.of("joint_state").size() == 1);
}

TEST_CASE("a hand that is no longer pushed is removed and comes back with the next frame", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.on_glove(glove_at(100 * kMs), 101 * kMs);
    p.on_tick(151 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand" });
    p.on_tick(152 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand", "destroy" });
    REQUIRE_FALSE(p.health(152 * kMs).hand_active);

    p.on_glove(glove_at(160 * kMs, 1060), 161 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand", "destroy", "create", "hand" });
    REQUIRE(p.health(161 * kMs).hand_active);
    REQUIRE(p.stats().injector_creates == 2);
    REQUIRE(p.stats().injector_destroys == 1);
}

TEST_CASE("reset drops the hand and the clock anchors", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    GloveSample g = glove_at(100 * kMs, 0xFFFFFFF0u);
    g.tactile = tactile(241, 100 * kMs);
    g.tactile->device_tick = 0xFFFFFFF0u;
    p.on_glove(g, 101 * kMs);
    p.reset("loop");
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand", "destroy" });
    REQUIRE_FALSE(p.health(101 * kMs).hand_active);
    REQUIRE_FALSE(p.health(101 * kMs).glove_live);

    g = glove_at(105 * kMs, 10);
    g.tactile = tactile(241, 105 * kMs);
    g.tactile->device_tick = 10;
    p.on_glove(g, 106 * kMs);
    REQUIRE(sink.hand_calls() == std::vector<std::string>{ "create", "hand", "destroy", "create", "hand" });
    REQUIRE(sink.of("tactile").back().raw_ns == 10 * kMs);
}

TEST_CASE("a reset keeps the glove and its tactile seen", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 100 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE_FALSE(p.health(101 * kMs).glove_seen);

    p.set_meta(meta());
    p.set_shape(shape(241));
    p.on_glove(g, 102 * kMs);
    p.reset("loop");
    const HandHealth h = p.health(102 * kMs);
    REQUIRE_FALSE(h.glove_live);
    REQUIRE(h.glove_seen);
    REQUIRE(h.tactile_seen);
}

TEST_CASE("tactile needs a matching shape and calibration", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());

    GloveSample g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE(sink.of("tactile").empty());
    REQUIRE(p.health(101 * kMs).tactile_skip == TactileSkip::NoShape);

    p.set_shape(shape(241));
    auto check_skip = [&](GloveSample sample, TactileSkip expected)
    {
        p.on_glove(sample, 102 * kMs);
        REQUIRE(sink.of("tactile").empty());
        REQUIRE(p.health(102 * kMs).tactile_skip == expected);
    };
    GloveSample bad = g;
    bad.tactile->calibrated = false;
    check_skip(bad, TactileSkip::Uncalibrated);
    bad = g;
    bad.tactile->valid = false;
    check_skip(bad, TactileSkip::Invalid);
    bad = g;
    bad.tactile = tactile(240, 98 * kMs);
    check_skip(bad, TactileSkip::CountMismatch);

    p.on_glove(g, 103 * kMs);
    const auto pushed = sink.of("tactile");
    REQUIRE(pushed.size() == 1);
    REQUIRE(pushed[0].t_ns == 98 * kMs);
    REQUIRE(pushed[0].buffer.size() <= kTactileMaxBytes);
    const auto* data = root_of<core::TactileDataArray>(pushed[0].buffer);
    REQUIRE(data->taxel_forces()->size() == 241);
    for (std::size_t i = 0; i < 241; ++i)
    {
        const auto* f = data->taxel_forces()->Get(static_cast<flatbuffers::uoffset_t>(i));
        REQUIRE(force_to_counts({ f->x(), f->y(), f->z() }) == static_cast<long>(100 + 7 * i));
        // Force is expressed in each taxel frame, independent of hand orientation.
        REQUIRE(f->x() == 0);
        REQUIRE(f->y() == 0);
        REQUIRE(f->z() == static_cast<float>(100 + 7 * i));
    }
    REQUIRE(p.health(103 * kMs).tactile_live);
    REQUIRE(p.health(103 * kMs).tactile_skip == TactileSkip::None);
}

TEST_CASE("each tactile tick is pushed once", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));

    // The SDK may repeat the latest tactile frame on glove samples that come faster than it.
    GloveSample g = glove_at(100 * kMs, 1000);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    g = glove_at(105 * kMs, 1005);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 106 * kMs);
    REQUIRE(sink.of("tactile").size() == 1);
    REQUIRE(p.stats().tactile_repeats == 1);

    g = glove_at(110 * kMs, 1010);
    g.tactile = tactile(241, 108 * kMs);
    g.tactile->device_tick = 510;
    p.on_glove(g, 111 * kMs);
    const auto pushed = sink.of("tactile");
    REQUIRE(pushed.size() == 2);
    REQUIRE(pushed[1].t_ns == 108 * kMs);
    REQUIRE(pushed[1].raw_ns - pushed[0].raw_ns == 10 * kMs);

    // A restarted stream may reuse the tick.
    p.reset("test");
    p.set_meta(meta());
    g = glove_at(200 * kMs, 1000);
    g.tactile = tactile(241, 198 * kMs);
    p.on_glove(g, 201 * kMs);
    REQUIRE(sink.of("tactile").size() == 3);
}

TEST_CASE("the shape is pushed at start and refreshed every period", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    const auto s = shape(241);
    p.set_shape(s);
    p.on_tick(0);
    p.on_tick(500 * kMs);
    p.on_tick(999 * kMs);
    p.on_tick(1000 * kMs);
    const auto shapes = sink.of("shape");
    REQUIRE(shapes.size() == 2);
    REQUIRE(shapes[0].buffer == s->bytes());
    REQUIRE(shapes[0].buffer.size() <= kShapeMaxBytes);
    REQUIRE(shapes[1].t_ns == 1000 * kMs);
}

TEST_CASE("the shape waits for a glove with tactile sensors", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_shape(shape(241));
    p.on_tick(0);
    REQUIRE(sink.of("shape").empty());

    p.set_meta(meta());
    // The first force sample brings the shape with it, ahead of the next tick.
    GloveSample g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE(sink.of("shape").size() == 1);
    REQUIRE(sink.of("tactile").size() == 1);
    REQUIRE(first_call(sink, "shape") < first_call(sink, "tactile"));
    p.on_tick(102 * kMs);
    REQUIRE(sink.of("shape").size() == 1);
}

TEST_CASE("a glove without tactile sensors streams joints and the hand without tactile", "[revohuman][pipeline]")
{
    RecordingSink sink;
    FakeWrist wrist;
    wrist.pose = wrist_at({ 0.0, 0.0, 0.0 });
    HandPipeline p(Hand::Left, {}, sink);
    p.set_wrist_provider(&wrist);
    p.set_shape(shape(241));
    p.set_meta(meta_without_tactile());
    p.on_tick(0);

    GloveSample g = glove_at(100 * kMs);
    // Whatever the source reports, a glove without sensors pushes no forces.
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);

    REQUIRE(sink.of("joint_state").size() == 1);
    REQUIRE(sink.of("hand").size() == 1);
    REQUIRE(sink.of("tactile").empty());
    const auto h = p.health(101 * kMs);
    REQUIRE(h.glove_live);
    REQUIRE(h.hand_active);
    REQUIRE_FALSE(h.tactile_live);
    REQUIRE(h.tactile_skip == TactileSkip::NoTactile);

    // Nor does the shape come with a later tick.
    p.on_tick(2000 * kMs);
    REQUIRE(sink.of("shape").empty());
    REQUIRE(p.stats().shapes_pushed == 0);
}

TEST_CASE("a glove without tactile sensors removes the forces and shape of the one before", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_shape(shape(241));
    p.set_meta(meta());
    GloveSample g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE(sink.of("shape").size() == 1);
    REQUIRE(sink.of("tactile").size() == 1);

    p.set_meta(meta_without_tactile());
    REQUIRE(sink.of("clear_tactile").size() == 1);
    REQUIRE(sink.of("clear_shape").size() == 1);
    p.on_tick(2000 * kMs);
    REQUIRE(sink.of("shape").size() == 1);

    // A glove with sensors again gets the shape at once.
    p.reset("test");
    p.set_meta(meta());
    p.on_tick(2001 * kMs);
    const auto shapes = sink.of("shape");
    REQUIRE(shapes.size() == 2);
    REQUIRE(shapes[1].t_ns == 2001 * kMs);
    // Once removed, there is nothing more to remove.
    p.set_meta(meta_without_tactile());
    p.set_meta(meta_without_tactile());
    REQUIRE(sink.of("clear_shape").size() == 2);
}

TEST_CASE("health tells which part of the glove is live", "[revohuman][pipeline]")
{
    RecordingSink sink;
    FakeWrist wrist;
    wrist.pose = wrist_at({ 0.0, 0.0, 0.0 });
    HandPipeline p(Hand::Left, {}, sink);
    p.set_wrist_provider(&wrist);
    p.set_meta(meta());
    auto h = p.health(0);
    REQUIRE_FALSE(h.glove_live);
    REQUIRE_FALSE(h.wrist_tracked);
    REQUIRE_FALSE(h.hand_active);
    REQUIRE(h.tactile_skip == TactileSkip::NoShape);

    p.on_glove(glove_at(10 * kMs, 1000, false), 10 * kMs);
    h = p.health(10 * kMs);
    REQUIRE(h.glove_live);
    REQUIRE(h.kinematics_missing);
    REQUIRE_FALSE(p.health(511 * kMs).glove_live);

    p.on_glove(glove_at(20 * kMs, 1010), 20 * kMs);
    h = p.health(20 * kMs);
    REQUIRE(h.wrist_tracked);
    REQUIRE(h.hand_active);
    REQUIRE_FALSE(h.kinematics_missing);

    p.on_tick(71 * kMs);
    h = p.health(71 * kMs);
    REQUIRE_FALSE(h.hand_active);
    REQUIRE_FALSE(h.wrist_tracked);
}

TEST_CASE("local pressure is independent of missing or invalid encoder poses", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    SECTION("no kinematics")
    {
        g.kinematics.reset();
    }
    SECTION("missing encoder sampling time")
    {
        g.has_sample_time = false;
    }
    SECTION("thumb encoders unavailable")
    {
        g.joints_valid_mask = 0xffff;
        g.kinematics->orientation_valid_mask = 0;
        g.kinematics->position_valid_mask = 0;
    }
    SECTION("nonunit encoder orientation")
    {
        g.kinematics->joints[9].q = { 0, 0, 0, 2 };
    }
    p.on_glove(g, 101 * kMs);
    const auto pushed = sink.of("tactile");
    REQUIRE(pushed.size() == 1);
    REQUIRE(pushed[0].t_ns == 98 * kMs);
    require_local_pressure(pushed[0]);
    REQUIRE(p.health(101 * kMs).tactile_live);
    if (!g.kinematics)
    {
        REQUIRE(sink.of("hand").empty());
        REQUIRE(p.health(101 * kMs).kinematics_missing);
    }
    else
    {
        const auto hands = sink.of("hand");
        REQUIRE(hands.size() == 1);
        REQUIRE(hands[0].frame.orientation_valid_mask == (g.kinematics->orientation_valid_mask & kMappedHandMask));
        REQUIRE(hands[0].frame.position_valid_mask == (g.kinematics->position_valid_mask & kMappedHandMask));
        REQUIRE_FALSE(hands[0].frame.tracked);
    }
}

TEST_CASE("tactile requires its own fresh ordered sampling time", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    TactileSkip expected = TactileSkip::None;
    SECTION("missing sampling time")
    {
        g.tactile->has_sample_time = false;
        expected = TactileSkip::NoSampleTime;
    }
    SECTION("old sample")
    {
        g.tactile->sample_mono_ns = 0;
        expected = TactileSkip::StaleSample;
    }
    SECTION("future sample")
    {
        g.tactile->sample_mono_ns = 102 * kMs;
        expected = TactileSkip::StaleSample;
    }
    SECTION("encoder time does not gate local pressure")
    {
        g.tactile->sample_mono_ns = 94 * kMs;
    }
    SECTION("unused position flags do not block force directions")
    {
        g.kinematics->position_valid_mask = 0;
    }
    SECTION("nonfinite or nonunit orientation despite valid mask")
    {
        g.kinematics->joints[9].q = { 0, 0, 0, 2 };
    }
    SECTION("sampling time regresses despite increasing device tick")
    {
        p.on_glove(g, 100 * kMs);
        sink.calls.clear();
        g.tactile->device_tick += 1;
        g.tactile->sample_mono_ns -= kMs;
        expected = TactileSkip::NonMonotonicSample;
    }
    SECTION("device tick regresses despite increasing sampling time")
    {
        p.on_glove(g, 100 * kMs);
        sink.calls.clear();
        g.tactile->device_tick -= 1;
        g.tactile->sample_mono_ns += kMs;
        expected = TactileSkip::NonMonotonicSample;
    }
    p.on_glove(g, 101 * kMs);
    REQUIRE(sink.of("tactile").size() == (expected == TactileSkip::None ? 1u : 0u));
    REQUIRE(p.health(101 * kMs).tactile_skip == expected);
}

TEST_CASE("local pressure is published immediately without waiting for an encoder pose", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        auto g = glove_at((100 + 5 * i) * kMs, 1000 + 5 * i);
        g.kinematics = turned_kinematics(0.2 * i);
        g.tactile = tactile(241, 111 * kMs);
        p.on_glove(g, (112 + 5 * i) * kMs);
        REQUIRE(sink.of("tactile").size() == 1);
    }
    REQUIRE(p.stats().tactile_repeats == 3);
    const auto pushed = sink.of("tactile");
    REQUIRE(pushed[0].t_ns == 111 * kMs);
    require_local_pressure(pushed[0]);
}

TEST_CASE("tactile generation changes and reset clear freshness", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE(p.health(101 * kMs).tactile_live);
    REQUIRE_FALSE(p.health(199 * kMs).tactile_live);
    SECTION("reset")
    {
        p.reset("reconnect");
        REQUIRE_FALSE(p.health(102 * kMs).tactile_live);
    }
    SECTION("generation")
    {
        g.generation += 1;
        g.tactile->valid = false;
        p.on_glove(g, 102 * kMs);
        REQUIRE_FALSE(p.health(102 * kMs).tactile_live);
        g.tactile->valid = true;
    }
    p.on_glove(g, 103 * kMs);
    REQUIRE(sink.of("tactile").size() == 2);
}

TEST_CASE("an invalid re-send of the pushed tactile sample holds it until the maximum age", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 98 * kMs);
    p.on_glove(g, 101 * kMs);
    REQUIRE(p.health(101 * kMs).tactile_live);

    // One failed read of the tactile board: the firmware re-sends the last snapshot as invalid.
    g.tactile->valid = false;
    p.on_glove(g, 106 * kMs);
    REQUIRE(p.health(106 * kMs).tactile_live);
    REQUIRE(p.stats().tactile_repeats == 1);
    REQUIRE(p.stats().tactile_invalid_repeats == 1);
    REQUIRE(p.stats().tactile_skipped == 0);

    SECTION("the next read recovers")
    {
        auto next = glove_at(110 * kMs, 1010);
        next.tactile = tactile(241, 107 * kMs);
        next.tactile->device_tick = 512;
        p.on_glove(next, 111 * kMs);
        REQUIRE(p.health(111 * kMs).tactile_live);
        REQUIRE(sink.of("tactile").size() == 2);
        REQUIRE(p.stats().tactile_skipped == 0);
    }
    SECTION("a re-send after the maximum age is invalid")
    {
        p.on_glove(g, 198 * kMs);
        REQUIRE(p.health(198 * kMs).tactile_live);
        p.on_glove(g, 199 * kMs);
        REQUIRE_FALSE(p.health(199 * kMs).tactile_live);
        REQUIRE(p.health(199 * kMs).tactile_skip == TactileSkip::Invalid);
        REQUIRE(p.stats().tactile_invalid_repeats == 2);
        REQUIRE(p.stats().tactile_skipped == 1);
    }
    SECTION("an expiry while held reports invalid")
    {
        p.on_tick(199 * kMs);
        REQUIRE(sink.of("clear_tactile").size() == 1);
        REQUIRE(p.health(199 * kMs).tactile_skip == TactileSkip::Invalid);
    }
    SECTION("an expiry after a valid repeat reports a stale sample")
    {
        g.tactile->valid = true;
        p.on_glove(g, 110 * kMs);
        p.on_tick(199 * kMs);
        REQUIRE(p.health(199 * kMs).tactile_skip == TactileSkip::StaleSample);
    }
    SECTION("a new tick marked invalid is invalid at once")
    {
        g.tactile->device_tick = 512;
        g.tactile->sample_mono_ns = 107 * kMs;
        p.on_glove(g, 111 * kMs);
        REQUIRE_FALSE(p.health(111 * kMs).tactile_live);
        REQUIRE(p.health(111 * kMs).tactile_skip == TactileSkip::Invalid);
        REQUIRE(p.stats().tactile_skipped == 1);
    }
    SECTION("a re-send after a reset is invalid")
    {
        p.reset("reconnect");
        p.on_glove(g, 107 * kMs);
        REQUIRE(p.health(107 * kMs).tactile_skip == TactileSkip::Invalid);
    }
}

TEST_CASE("tactile_rate counts immediate local force publication", "[revohuman][pipeline]")
{
    TempLog file("tactile_rate_steady");
    {
        EventLog log(file.path.string());
        RecordingSink sink;
        HandPipeline p(Hand::Left, {}, sink, &log);
        p.set_meta(meta());
        p.set_shape(shape(241));
        // Encoder/tactile sampling skew must not delay taxel-local forces.
        const std::int64_t offsets_ms[] = { 3, 6, 5, 4 };
        for (std::uint32_t i = 0; i <= 600; ++i)
        {
            const std::int64_t glove_ns = 100 * kMs + 5 * kMs * i;
            auto g = glove_at(glove_ns, 1000 + 5 * i);
            g.tactile = tactile(241, glove_ns + offsets_ms[i % 4] * kMs);
            g.tactile->device_tick = 500 + 5 * i;
            const std::int64_t now_ns = glove_ns + 10 * kMs;
            p.on_glove(g, now_ns);
            p.on_tick(now_ns);
        }
        REQUIRE(p.stats().tactile_pushed == 601);
        REQUIRE(p.stats().tactile_skipped == 0);
    }
    const auto rates = file.events("tactile_rate");
    REQUIRE(rates.size() == 3);
    for (std::size_t w = 0; w < rates.size(); ++w)
    {
        const auto& line = rates[w];
        REQUIRE(field(line, "hand") == "\"left\"");
        REQUIRE(number(line, "t_ns") == Approx(static_cast<double>(110 * kMs + 1000 * kMs * (w + 1))));
        REQUIRE(number(line, "window_s") == Approx(1.0));
        REQUIRE(number(line, "samples") == 200);
        REQUIRE(number(line, "pushed") == 200);
        REQUIRE(number(line, "skipped") == 0);
        REQUIRE(number(line, "lag_max_ms") == Approx(7.0));
        REQUIRE(number(line, "samples_total") == 1 + 200 * (w + 1));
        REQUIRE(number(line, "pushed_total") == 1 + 200 * (w + 1));
        require_rate_identity(line);
    }
}

TEST_CASE("tactile_rate accounts for rejection and recovery without pose queues", "[revohuman][pipeline]")
{
    TempLog file("tactile_rate_skips");
    {
        EventLog log(file.path.string());
        RecordingSink sink;
        HandPipeline p(Hand::Left, {}, sink, &log);
        p.set_meta(meta());
        p.set_shape(shape(241));
        p.on_tick(0);
        auto send = [&](std::uint32_t tick, std::int64_t sample_ms, std::int64_t now_ms, bool valid = true)
        {
            auto g = glove_at(now_ms * kMs, tick, false);
            g.tactile = tactile(241, sample_ms * kMs);
            g.tactile->device_tick = tick;
            g.tactile->valid = valid;
            p.on_glove(g, now_ms * kMs);
        };
        send(500, 1002, 1001); // future
        send(501, 800, 1006); // stale
        send(502, 1009, 1011); // publish
        send(502, 1009, 1016); // repeat
        send(503, 1023, 1024); // publish without an encoder pose
        send(503, 1023, 1026, false); // invalid re-send of the pushed sample, held
        p.on_tick(1300 * kMs);
        send(504, 1359, 1361, false); // invalid, never a valid zero
        p.reset("loop");
        send(500, 1403, 1404); // reused device tick after reset
        p.on_tick(2400 * kMs);
        REQUIRE(p.stats().tactile_samples == 8);
        REQUIRE(p.stats().tactile_pushed == 3);
        REQUIRE(p.stats().tactile_skipped == 3);
        REQUIRE(p.stats().tactile_repeats == 2);
        REQUIRE(p.stats().tactile_invalid_repeats == 1);
    }
    const auto rates = file.events("tactile_rate");
    REQUIRE(rates.size() == 2);
    REQUIRE(number(rates[0], "window_s") == Approx(1.3));
    REQUIRE(number(rates[0], "samples") == 6);
    REQUIRE(number(rates[0], "pushed") == 2);
    REQUIRE(number(rates[0], "repeats") == 2);
    REQUIRE(number(rates[0], "invalid_repeats") == 1);
    REQUIRE(number(rates[0], "skip_stale_future") == 1);
    REQUIRE(number(rates[0], "skip_stale_old") == 1);
    require_rate_identity(rates[0]);
    REQUIRE(number(rates[1], "window_s") == Approx(1.1));
    REQUIRE(number(rates[1], "samples") == 2);
    REQUIRE(number(rates[1], "pushed") == 1);
    REQUIRE(number(rates[1], "skip_other") == 1);
    REQUIRE(number(rates[1], "invalid_repeats") == 0);
    REQUIRE(number(rates[1], "lag_max_ms") == Approx(1.0));
    require_rate_identity(rates[1]);
    REQUIRE(number(rates[1], "samples_total") == 8);
    REQUIRE(number(rates[1], "skipped_total") == 3);
    REQUIRE(number(rates[1], "invalid_repeats_total") == 1);
}

TEST_CASE("tactile_rate is off with a zero period", "[revohuman][pipeline]")
{
    TempLog file("tactile_rate_off");
    {
        EventLog log(file.path.string());
        RecordingSink sink;
        PipelineConfig config;
        config.tactile_rate_period_ns = 0;
        HandPipeline p(Hand::Left, config, sink, &log);
        p.set_meta(meta());
        for (std::int64_t t = 0; t <= 3000; t += 100)
        {
            p.on_tick(t * kMs);
        }
    }
    REQUIRE(file.events("tactile_rate").empty());
}

TEST_CASE("skipped tactile samples keep the force output until it expires", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    std::int64_t t = 100;
    std::uint32_t tick = 500;
    auto next = [&]
    {
        t += 5;
        tick += 5;
        g.tactile = tactile(241, t * kMs);
        g.tactile->device_tick = tick;
    };
    g.tactile = tactile(241, t * kMs);
    p.on_glove(g, (t + 1) * kMs);
    REQUIRE(p.health((t + 1) * kMs).tactile_live);

    // Valid samples alternating with each kind of skip leave the collection in place.
    for (int kind = 0; kind < 4; ++kind)
    {
        next();
        switch (kind)
        {
        case 0:
            g.tactile->valid = false;
            break;
        case 1:
            g.tactile->calibrated = false;
            break;
        case 2:
            g.tactile->values.pop_back();
            break;
        default:
            g.tactile->sample_mono_ns = (t + 2) * kMs;
            break;
        }
        p.on_glove(g, (t + 1) * kMs);
        REQUIRE_FALSE(p.health((t + 1) * kMs).tactile_live);
        next();
        p.on_glove(g, (t + 1) * kMs);
        REQUIRE(p.health((t + 1) * kMs).tactile_live);
    }
    REQUIRE(sink.of("tactile").size() == 5);
    REQUIRE(sink.of("clear_tactile").empty());

    // Skips that outlast the maximum age end in one removal, timed from the last pushed sample.
    const std::int64_t pushed = t;
    next();
    g.tactile->valid = false;
    p.on_glove(g, (t + 1) * kMs);
    p.on_tick((pushed + 100) * kMs);
    REQUIRE(sink.of("clear_tactile").empty());
    p.on_tick((pushed + 101) * kMs);
    REQUIRE(sink.of("clear_tactile").size() == 1);
    REQUIRE_FALSE(p.health((pushed + 101) * kMs).tactile_live);
    p.on_tick((pushed + 102) * kMs);
    REQUIRE(sink.of("clear_tactile").size() == 1);

    t = pushed + 105;
    next();
    p.on_glove(g, (t + 1) * kMs);
    REQUIRE(p.health((t + 1) * kMs).tactile_live);
    REQUIRE(sink.of("tactile").size() == 6);
}

TEST_CASE("reset and stream generations clear old force output", "[revohuman][pipeline]")
{
    RecordingSink sink;
    HandPipeline p(Hand::Left, {}, sink);
    p.set_meta(meta());
    p.set_shape(shape(241));
    auto g = glove_at(100 * kMs);
    g.tactile = tactile(241, 100 * kMs);
    p.on_glove(g, 101 * kMs);
    p.reset("loop");
    REQUIRE(sink.of("clear_tactile").size() == 1);
    REQUIRE_FALSE(p.health(101 * kMs).glove_live);
    REQUIRE_FALSE(p.health(101 * kMs).tactile_live);
    p.on_glove(g, 102 * kMs);
    REQUIRE(p.health(102 * kMs).tactile_live);
    g.generation = 2;
    g.tactile.reset();
    p.on_glove(g, 103 * kMs);
    REQUIRE(sink.of("clear_tactile").size() == 2);
    REQUIRE_FALSE(p.health(103 * kMs).tactile_live);
}
