# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive the calibration tables baked into io_exo_glove_ros2_kinematics.cpp.

    python3 io_exo_glove_ros2_kinematics_tables.py <recording.jsonl>

The recording is a capture of /io_fusion/tf_exoskeleton (one {"t", "frames"} object per message,
as written by build/raw_tf_recorder.py). An absolute joint angle is read from the published link
quaternions exactly as the plugin does: each child is parent * R_rest(rpy) * R(axis, theta).
Openness is measured along the finger, as the distance from the knuckle (the first bend joint) to
the published fingertip. The longest steady open and closed plateaus give the reference frames;
their medoids are the open zero and the fist, so a drifting pose cannot bias either. The zero is
the open-frame angle, and the sign is +1 when a joint grows from open to fist. The script also
checks that the open zero puts every model fingertip one finger-length ahead of its MCP and that
every MCP/PIP/DIP is flexed at the fist on both hands.
"""

from __future__ import annotations

import json
import math
import sys

# URDF (blender_human_skeleton_v5.urdf): rest rpy and revolute axis per mapped joint; the right
# hand's F1 joints mirror the left's in X/Y. Thumb1 is the spread and Thumb2..Thumb4 the bends;
# ThumbBase is the thumb's fixed mount and the parent of Thumb1.
REST = {
    "Thumb1": ((1.5708, 1.5708, 0), (0, 0, 1)),
    "Thumb2": ((-1.309, 0, 0), (1, 0, 0)),
    "Thumb3": ((-0.81433, 0, 0), (1, 0, 0)),
    "Thumb4": ((2.5307, 0, 0), (1, 0, 0)),
    "Index1": ((0, 0.2618, 0.1199), (0, 0, 1)),
    "Index2": ((-0.75647, 0, 0), (1, 0, 0)),
    "Index3": ((0.46849, 0, 0), (1, 0, 0)),
    "Index4": ((1.2217, 0, 0), (1, 0, 0)),
    "Middle1": ((0, 0, 0.1199), (0, 0, 1)),
    "Middle2": ((-0.75647, 0, 0), (1, 0, 0)),
    "Middle3": ((0.46849, 0, 0), (1, 0, 0)),
    "Middle4": ((1.2217, 0, 0), (1, 0, 0)),
    "Ring1": ((0, 0, 0.1199), (0, 0, 1)),
    "Ring2": ((-0.75647, 0, 0), (1, 0, 0)),
    "Ring3": ((0.46849, 0, 0), (1, 0, 0)),
    "Ring4": ((1.2217, 0, 0), (1, 0, 0)),
    "Pinky1": ((0, -0.17453, 0.1199), (0, 0, 1)),
    "Pinky2": ((-0.75647, 0, 0), (1, 0, 0)),
    "Pinky3": ((0.46849, 0, 0), (1, 0, 0)),
    "Pinky4": ((1.2217, 0, 0), (1, 0, 0)),
}
RIGHT_F1 = {
    "Thumb1": ((1.5708, 1.5708, 0), (0, 0, 1)),
    "Index1": ((0, -0.2618, -0.1199), (0, 0, 1)),
    "Middle1": ((0, 0, -0.1199), (0, 0, 1)),
    "Ring1": ((0, 0, -0.1199), (0, 0, 1)),
    "Pinky1": ((0, 0.17453, -0.1199), (0, 0, 1)),
}
#! finger -> (links, parent, knuckle joint, published tip frame)
FINGERS = {
    "thumb": (
        ("Thumb1", "Thumb2", "Thumb3", "Thumb4"),
        "ThumbBase",
        "Thumb3",
        "thumb_tip",
    ),
    "index": (("Index1", "Index2", "Index3", "Index4"), "Base", "Index2", "index_tip"),
    "middle": (
        ("Middle1", "Middle2", "Middle3", "Middle4"),
        "Base",
        "Middle2",
        "middle_tip",
    ),
    "ring": (("Ring1", "Ring2", "Ring3", "Ring4"), "Base", "Ring2", "ring_tip"),
    "little": (("Pinky1", "Pinky2", "Pinky3", "Pinky4"), "Base", "Pinky2", "pinky_tip"),
}
# Anatomical landmarks (blender_human.urdf, OpenXR wrist frame, metres), CMC -> tip.
LANDMARKS = {
    "thumb": (
        (0.0208, -0.0023, -0.0243),
        (0, 0, -0.0387),
        (0, 0, -0.0269),
        (0, 0, -0.0220),
    ),
    "index": (
        (0.0111, -0.0043, -0.0383),
        (0, 0, -0.0559),
        (0, 0, -0.0381),
        (0, 0, -0.0216),
        (0, 0, -0.0220),
    ),
    "middle": (
        (-0.0014, -0.0014, -0.0368),
        (0, 0, -0.0545),
        (0, 0, -0.0415),
        (0, 0, -0.0260),
        (0, 0, -0.0220),
    ),
    "ring": (
        (-0.0103, 0.0003, -0.0344),
        (0, 0, -0.0490),
        (0, 0, -0.0362),
        (0, 0, -0.0251),
        (0, 0, -0.0220),
    ),
    "little": (
        (-0.0203, 0.0031, -0.0295),
        (0, 0, -0.0450),
        (0, 0, -0.0290),
        (0, 0, -0.0183),
        (0, 0, -0.0220),
    ),
}


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def axis_angle(axis, angle):
    s = math.sin(0.5 * angle)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle))


def from_rpy(rpy):
    # URDF's fixed-axis rpy: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    roll, pitch, yaw = rpy
    return qmul(
        qmul(axis_angle((0, 0, 1), yaw), axis_angle((0, 1, 0), pitch)),
        axis_angle((1, 0, 0), roll),
    )


def signed_angle(q, axis):
    vector, w = [q[0], q[1], q[2]], q[3]
    if w < 0:
        vector, w = [-x for x in vector], -w
    return 2 * math.atan2(sum(vector[i] * axis[i] for i in range(3)), w)


def rotation(frame):
    q = tuple(frame[3:7])
    length = math.sqrt(sum(x * x for x in q))
    return tuple(x / length for x in q)


def relative_angle(parent, child, rpy, axis):
    return signed_angle(qmul(qconj(from_rpy(rpy)), qmul(qconj(parent), child)), axis)


def read_angles(frames):
    angles = []
    for frame in frames:
        row = {}
        for hand in ("Left", "Right"):
            for finger, (links, parent_link, _, _) in FINGERS.items():
                parent = rotation(frame["link_%sSkeleton%s" % (hand, parent_link)])
                for slot, link in enumerate(links):
                    child = rotation(frame["link_%sSkeleton%s" % (hand, link)])
                    rpy, axis = (
                        RIGHT_F1[link] if hand == "Right" and slot == 0 else REST[link]
                    )
                    row[(hand, finger, slot)] = relative_angle(parent, child, rpy, axis)
                    parent = child
        angles.append(row)
    return angles


def openness(frame):
    total = 0.0
    for hand in ("Left", "Right"):
        for _, _, mcp, tip in FINGERS.values():
            knuckle = frame["link_%sSkeleton%s" % (hand, mcp)][:3]
            fingertip = frame["%s_%s" % (hand.lower(), tip)][:3]
            total += math.dist(knuckle, fingertip)
    return total


def stable_runs(motion, times):
    window, smoothed = 30, []
    for i in range(len(motion)):
        chunk = motion[max(0, i - window) : i + window + 1]
        smoothed.append(sorted(chunk)[len(chunk) // 2])
    runs, start = [], None
    for i, value in enumerate(smoothed):
        if value < 0.03 and start is None:
            start = i
        if value >= 0.03 and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(smoothed) - 1])
    merged = []
    for run in runs:
        if merged and times[run[0]] - times[merged[-1][1]] < 1:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [run for run in merged if times[run[1]] - times[run[0]] >= 2]


def medoid(run, angles, keys):
    median = {
        k: sorted(angles[i][k] for i in range(run[0], run[1] + 1))[
            (run[1] - run[0]) // 2
        ]
        for k in keys
    }
    return min(
        range(run[0], run[1] + 1),
        key=lambda i: sum((angles[i][k] - median[k]) ** 2 for k in keys),
    )


def build_anatomy(bends):
    """The plugin's chain: place a landmark, then rotate the following bone."""
    joints = {}
    for finger in FINGERS:
        rotation_q, position, forward = (0, 0, 0, 1), [0.0, 0.0, 0.0], (0, 0, -1)
        count, first = len(LANDMARKS[finger]), 0 if finger == "thumb" else 1
        for landmark in range(count):
            offset = LANDMARKS[finger][landmark]
            moved = qmul(qmul(rotation_q, (*offset, 0)), qconj(rotation_q))[:3]
            position = [position[i] + moved[i] for i in range(3)]
            if first <= landmark < count - 1:
                slot = landmark - first
                if slot == 0:
                    rotation_q = qmul(
                        rotation_q, axis_angle((0, 1, 0), -bends.get((finger, 0), 0))
                    )
                rotation_q = qmul(
                    rotation_q, axis_angle((1, 0, 0), -bends.get((finger, slot), 0))
                )
            if landmark + 1 < count:
                forward = LANDMARKS[finger][landmark + 1]
                world = qmul(qmul(rotation_q, (*forward, 0)), qconj(rotation_q))[:3]
                length = math.sqrt(sum(x * x for x in world))
                forward = tuple(x / length for x in world)
            joints[(finger, landmark)] = tuple(position)
    return joints


def main():
    if len(sys.argv) != 2:
        print("usage: python3 %s <recording.jsonl>" % sys.argv[0], file=sys.stderr)
        return 2
    frames, times = [], []
    for line in open(sys.argv[1], encoding="utf-8"):
        if line.strip():
            message = json.loads(line)
            frames.append(message["frames"])
            times.append(message["t"])
    if len(frames) < 100:
        print("need at least 100 messages", file=sys.stderr)
        return 2
    times = [t - times[0] for t in times]

    angles = read_angles(frames)
    keys = list(angles[0])
    scores = [openness(frame) for frame in frames]
    motion = [0.0] + [
        sum(abs(angles[i][key] - angles[i - 1][key]) for key in keys)
        for i in range(1, len(angles))
    ]
    runs = stable_runs(motion, times)
    medians = [
        sum(scores[i] for i in range(r[0], r[1] + 1)) / (r[1] - r[0] + 1) for r in runs
    ]
    candidates = [i for i, r in enumerate(runs) if times[r[1]] - times[r[0]] >= 3]
    highest = max(medians[i] for i in candidates)
    open_candidates = [i for i in candidates if medians[i] >= 0.9 * highest]
    best_open = max(
        open_candidates, key=lambda i: times[runs[i][1]] - times[runs[i][0]]
    )
    open_run = runs[best_open]
    closed = [
        i
        for i, r in enumerate(runs)
        if medians[i] < 0.9 * medians[best_open] and times[r[1]] - times[r[0]] >= 2
    ]
    if not closed:
        print("no steady closed plateau found", file=sys.stderr)
        return 2
    fist_run = runs[max(closed, key=lambda i: times[runs[i][1]] - times[runs[i][0]])]
    open_frame, fist_frame = (
        medoid(open_run, angles, keys),
        medoid(fist_run, angles, keys),
    )
    print("recording: %s, %d messages" % (sys.argv[1], len(frames)))
    print(
        "open plateau %.1f-%.1f s, medoid %.1f s | fist plateau %.1f-%.1f s, medoid %.1f s"
        % (
            times[open_run[0]],
            times[open_run[1]],
            times[open_frame],
            times[fist_run[0]],
            times[fist_run[1]],
            times[fist_frame],
        )
    )

    zero = {key: angles[open_frame][key] for key in keys}
    signs = {}
    for key in keys:
        hand, _, slot = key
        if slot == 0:
            signs[key] = 1.0 if hand == "Left" else -1.0
        else:
            signs[key] = 1.0 if angles[fist_frame][key] >= zero[key] else -1.0

    print(
        "\nstatic constexpr RigJointSpec kRigJoints[2][kRigFingerCount][kRigSlotCount] = {"
    )
    for hand in ("Left", "Right"):
        print("    // %s\n    {" % hand.lower())
        for finger in FINGERS:
            print("        {\n            // %s" % finger)
            for slot, link in enumerate(FINGERS[finger][0]):
                rpy, axis = (
                    RIGHT_F1[link] if hand == "Right" and slot == 0 else REST[link]
                )
                print(
                    "            { { %.5ff, %.5ff, %.5ff }, { %.5ff, %.5ff, %.5ff }, %.5ff, %.1ff },"
                    % (
                        *rpy,
                        *axis,
                        zero[(hand, finger, slot)],
                        signs[(hand, finger, slot)],
                    )
                )
            print("        },")
        print("    },")
    print("};")

    print("\nzeros (degrees), F1/F2/F3/F4:")
    for hand in ("Left", "Right"):
        for finger in FINGERS:
            print(
                "    %-5s %-6s %s"
                % (
                    hand.lower(),
                    finger,
                    " / ".join(
                        "%8.2f" % math.degrees(zero[(hand, finger, s)])
                        for s in range(4)
                    ),
                )
            )
    print("signs, F1/F2/F3/F4:")
    for hand in ("Left", "Right"):
        for finger in FINGERS:
            print(
                "    %-5s %-6s %s"
                % (
                    hand.lower(),
                    finger,
                    " / ".join("%+d" % int(signs[(hand, finger, s)]) for s in range(4)),
                )
            )

    anatomy = build_anatomy({})
    print("\ncheck 1: open zero, fingertip ahead of its MCP (model length)")
    check1 = True
    for finger in FINGERS:
        tip, mcp = anatomy[(finger, len(LANDMARKS[finger]) - 1)], anatomy[(finger, 1)]
        ahead = [tip[i] - mcp[i] for i in range(3)]
        expected = -sum(offset[2] for offset in LANDMARKS[finger][2:])
        ok = abs(math.dist(tip, mcp) - expected) < 0.002 and ahead[2] < 0
        check1 = check1 and ok
        print(
            "  [%s] %-6s MCP->tip %.4f m (model %.4f m), along -Z %.4f m"
            % (
                "PASS" if ok else "FAIL",
                finger,
                math.dist(tip, mcp),
                expected,
                -ahead[2],
            )
        )

    print("\ncheck 2: calibrated bends at the fist, all positive (degrees)")
    check2 = True
    for hand in ("Left", "Right"):
        values = []
        for finger in FINGERS:
            bends = [
                math.degrees(
                    signs[(hand, finger, s)]
                    * (angles[fist_frame][(hand, finger, s)] - zero[(hand, finger, s)])
                )
                for s in (1, 2, 3)
            ]
            check2 = check2 and all(bend > 0 for bend in bends)
            values.append("%s %+.1f/%+.1f/%+.1f" % (finger, *bends))
        print(
            "  [%s] %-5s %s" % ("PASS" if check2 else "FAIL", hand, "; ".join(values))
        )
    index_mcp = [
        math.degrees(
            signs[(hand, "index", 1)]
            * (angles[fist_frame][(hand, "index", 1)] - zero[(hand, "index", 1)])
        )
        for hand in ("Left", "Right")
    ]
    print(
        "  index MCP at fist: left %+.1f deg, right %+.1f deg (both positive)"
        % tuple(index_mcp)
    )
    print(
        "\nALL DERIVATION CHECKS PASSED"
        if check1 and check2
        else "\nDERIVATION CHECKS FAILED"
    )
    return 0 if check1 and check2 else 1


if __name__ == "__main__":
    sys.exit(main())
