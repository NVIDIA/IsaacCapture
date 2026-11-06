# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads every fixture back and asserts it is what the index claims.

Envelope first (schema, encoding, embedded .bfbs, topics, record count, sequences), then
that each defect's fault is present in the bytes.
"""

from __future__ import annotations

import json
import math
import os
import sys

import hand_model as hm
import script
import toolchain

toolchain.ensure()

from mcap.reader import NonSeekingReader  # noqa: E402

from core.HandPoseRecord import HandPoseRecord  # noqa: E402
from core.Point import Point  # noqa: E402
from core.Pose import Pose  # noqa: E402
from core.Quaternion import Quaternion  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(toolchain.BFBS_PATH, "rb") as _fh:
    GOLDEN_BFBS = _fh.read()

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def read(path: str):
    """-> (envelope, {side: [(sample_ns, joints or None)]})"""
    envelope: dict = {"topics": set(), "sequences": {}}
    hands: dict[str, list] = {"left": [], "right": []}
    with open(path, "rb") as fh:
        reader = NonSeekingReader(fh)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            envelope["schema"] = (schema.name, schema.encoding, schema.data)
            envelope["encoding"] = channel.message_encoding
            envelope["topics"].add(channel.topic)
            seqs = envelope["sequences"].setdefault(channel.topic, [])
            seqs.append(message.sequence)
            record = HandPoseRecord.GetRootAs(bytearray(message.data), 0)
            ts = record.Timestamp()
            expect(
                message.log_time == ts.AvailableTimeLocalCommonClock(),
                f"{path}: logTime is not available_time",
            )
            data = record.Data()
            joints = None
            if data is not None:
                hj = data.Joints()
                joints = []
                for i in range(hm.NUM_JOINTS):
                    j = hj.Poses(i)
                    pose = j.Pose(Pose())
                    p = pose.Position(Point())
                    q = pose.Orientation(Quaternion())
                    joints.append(
                        (
                            (p.X(), p.Y(), p.Z()),
                            (q.X(), q.Y(), q.Z(), q.W()),
                            j.IsValid(),
                        )
                    )
            side = channel.topic.rsplit("/", 1)[-1].split("_")[0]
            hands[side].append((ts.SampleTimeLocalCommonClock(), joints))
    return envelope, hands


def local(joints, index: int) -> hm.Vec:
    wrist_p, wrist_q, _ = joints[hm.WRIST]
    inverse = (-wrist_q[0], -wrist_q[1], -wrist_q[2], wrist_q[3])
    return hm.quat_rotate(inverse, hm.sub(joints[index][0], wrist_p))


def straightness(joints, finger: str) -> float:
    chain = hm.FINGER_JOINTS[finger][1:]
    bones = sum(
        hm.length(hm.sub(joints[b][0], joints[a][0])) for a, b in zip(chain, chain[1:])
    )
    return hm.length(hm.sub(joints[chain[-1]][0], joints[chain[0]][0])) / bones


def in_window(frames, window, start: float = 0.4, end: float = 1.0):
    span = window["end_ns"] - window["start_ns"]
    lo = window["start_ns"] + start * span
    hi = window["start_ns"] + end * span
    return [j for t, j in frames if lo <= t < hi and j is not None]


def verify(entry: dict, labels: dict) -> None:
    name = os.path.basename(entry["filename"])[: -len(".mcap")]
    envelope, hands = read(os.path.join(HERE, entry["filename"]))
    schema_name, schema_encoding, schema_data = envelope["schema"]
    expect(schema_name == "core.HandPoseRecord", f"{name}: schema {schema_name}")
    expect(schema_encoding == "flatbuffer", f"{name}: schema encoding")
    expect(schema_data == GOLDEN_BFBS, f"{name}: embedded schema is not hand.bfbs")
    expect(envelope["encoding"] == "flatbuffer", f"{name}: message encoding")
    expect(
        envelope["topics"] == {"hands/left_hand", "hands/right_hand"},
        f"{name}: topics {envelope['topics']}",
    )
    for topic, seqs in envelope["sequences"].items():
        expect(seqs == list(range(len(seqs))), f"{name}: {topic} sequence gaps")
    expect(
        sum(len(v) for v in hands.values()) == entry["records"],
        f"{name}: record count",
    )

    windows = {w["label"]: w for w in labels["steps"]}
    right = hands["right"]
    first = right[0][1]
    fist_right = in_window(right, windows["fist"])

    if name == "defect_finger_swap_index_middle":
        expect(local(first, 7)[0] > local(first, 12)[0], f"{name}: index not swapped")
    elif name == "defect_flex_reversed":
        rise = local(fist_right[0], 10)[1] - local(fist_right[0], 7)[1]
        expect(rise > 0.01, f"{name}: fist tips are not dorsal ({rise:.3f})")
    elif name == "defect_right_channel_mirrored":
        across = hm.sub(local(first, 22), local(first, 7))
        expect(across[0] < 0, f"{name}: right channel is not left-handed")
    elif name == "defect_wrist_frozen":
        late = [j for t, j in right if (t - right[0][0]) / 1e9 > 8.5]
        repeats = sum(
            a[hm.WRIST][:2] == b[hm.WRIST][:2] for a, b in zip(late, late[1:])
        )
        expect(repeats / len(late) > 0.9, f"{name}: wrist not held ({repeats})")
    elif name == "defect_hand_at_origin":
        expect(hm.length(first[hm.WRIST][0]) < 1e-6, f"{name}: wrist not at origin")
    elif name == "defect_centimetres":
        expect(hm.length(first[hm.WRIST][0]) > 50, f"{name}: not centimetres")
    elif name == "defect_quaternion_wxyz":
        expect(abs(first[hm.WRIST][1][0]) > 0.9, f"{name}: w not in the x slot")
    elif name == "defect_source_switch":
        early = sum(v for *_, v in right[60][1])
        late = sum(v for *_, v in right[-60][1])
        expect((early, late) == (21, 26), f"{name}: valid counts {early}, {late}")
    elif name == "defect_required_joint_invalid":
        rate = sum(not j[9][2] for _, j in right) / len(right)
        expect(0.25 < rate < 0.4, f"{name}: INDEX_DISTAL invalid rate {rate:.2f}")
    elif name == "defect_wrist_orientation_unrolled":
        mid = in_window(right, windows["right_tip_roll"], 0.45, 0.55)[0]
        q = hm.quat_mul(hm.quat_conj(script.TIPS_Q), mid[hm.WRIST][1])
        wrist_turn = math.degrees(2 * math.acos(min(1.0, abs(q[3]))))
        across = hm.unit(hm.sub(mid[22][0], mid[7][0]))
        expect(wrist_turn < 15, f"{name}: wrist quaternion turned {wrist_turn:.0f} deg")
        expect(across[1] > 0.5, f"{name}: positions did not turn palm-out")
    elif name in (
        "defect_wrist_offset_in_hand_frame",
        "defect_right_hand_raised",
        "defect_world_tilted",
        "defect_ring_finger_bent",
    ):
        flat_r = in_window(right, windows["flat_on_table_open"])[0]
        flat_l = in_window(hands["left"], windows["flat_on_table_open"])[0]
        rise = flat_r[hm.WRIST][0][1] - flat_l[hm.WRIST][0][1]
        if name == "defect_wrist_offset_in_hand_frame":
            expect(0.012 < rise < 0.024, f"{name}: dorsal offset reads {rise:.3f} m")
        elif name == "defect_right_hand_raised":
            expect(0.03 < rise < 0.04, f"{name}: right hand raised {rise:.3f} m")
        elif name == "defect_world_tilted":
            dorsal = hm.quat_rotate(flat_r[hm.WRIST][1], (0.0, 1.0, 0.0))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, dorsal[1]))))
            expect(15 < tilt < 25, f"{name}: back of the hand tilted {tilt:.0f} deg")
        else:
            drop = flat_r[15][0][1] - flat_r[20][0][1]
            expect(drop > 0.03, f"{name}: ring tip only {drop:.3f} m below middle")
    elif name == "defect_right_hand_oversized":
        ratio = hm.length(hm.sub(first[15][0], first[12][0])) / hm.length(
            hm.sub(hands["left"][0][1][15][0], hands["left"][0][1][12][0])
        )
        expect(1.25 < ratio < 1.35, f"{name}: right middle finger {ratio:.2f}x left")
    elif name == "retake_shallow_fist":
        s = sum(straightness(fist_right[0], f) for f in hm.FINGERS) / 4
        expect(0.6 < s < 0.75, f"{name}: fist straightness {s:.2f}")
    elif name == "benign_palm_on_little_tip":
        expect(first[hm.PALM][0] == first[25][0], f"{name}: PALM is not on LITTLE_TIP")
    elif name == "golden_glove_21":
        expect(all(sum(v for *_, v in j) == 21 for _, j in right), f"{name}: not 21")
    elif name == "benign_late_arrival":
        early = in_window(right, windows["fist"], 0.0, 0.1)
        s_early = sum(straightness(early[0], f) for f in hm.FINGERS) / 4
        s_late = sum(straightness(fist_right[0], f) for f in hm.FINGERS) / 4
        expect(s_early > 0.9 and s_late < 0.5, f"{name}: {s_early:.2f} -> {s_late:.2f}")
    else:
        s = sum(straightness(fist_right[0], f) for f in hm.FINGERS) / 4
        expect(s < 0.5, f"{name}: fist straightness {s:.2f}")


def main() -> int:
    with open(os.path.join(HERE, "fixtures_index.json")) as fh:
        index = json.load(fh)
    for entry in index["fixtures"]:
        with open(os.path.join(HERE, entry["labels"])) as fh:
            verify(entry, json.load(fh))
    for message in failures:
        print(f"FAIL {message}")
    print(f"{len(index['fixtures'])} fixtures read back, {len(failures)} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
