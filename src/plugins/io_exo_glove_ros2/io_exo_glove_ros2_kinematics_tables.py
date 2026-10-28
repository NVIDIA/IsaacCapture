# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive and check the anatomy and calibration tables baked into the plugin.

    python3 io_exo_glove_ros2_kinematics_tables.py <recording.jsonl> [--thumb-recording PATH]
                                                    [--urdf-dir DIR]

``--urdf-dir`` (or ``$IO_EXO_URDF_DIR``) points at the vendor SDK's
``src/exo_tf_pkg/description``; it defaults to the path the constants were derived from.

.. note::
   Flexion is derived from the fingertip chord since round 12. The plugin now consumes only the
   knuckle-spread/lift and ThumbBase rows derived below (their open zero and sign); the per-joint
   bend zeros, directions, fist references and gains are kept for the historical derivation and for
   the A-pose checks, and are not read at run time.

The recording is a capture of ``/io_fusion/tf_exoskeleton`` (one ``{"t", "frames"}`` object per
message, as written by ``build/raw_tf_recorder.py``).  The thumb block is derived from the clean
thumb-only recording (``--thumb-recording``, default: the positional recording); the fingers are
still derived from the positional all-gesture recording.  This script is the source of truth for
three things:

- the anatomical landmark chain, built from the vendor's human model
  (``blender_human.urdf``): every joint is placed at its parent-relative ``<origin xyz>`` and
  oriented with its ``<origin rpy>``, so the CMC positions and the bone lengths are the model's;
- the A-pose the model's hand joints are exported without: the metacarpals are fanned to a
  physical knuckle span and levelled, and the thumb gets its own frame, so the injected hand is
  not a flat fifth finger (see ``MCP_SPAN_M`` and ``THUMB_*`` below);
- the per-joint open-hand zero, direction and gain, read from the recording by the same quaternion
  algebra the plugin uses: the open plateau gives the zero, the closed plateau gives the sign and
  the linkage travel that maps onto the joint's human range, and the knuckle-spread sign is measured
  from the mechanism.

Every joint angle is unwrapped sequentially first (the quaternion formula returns the shortest
signed angle, which jumps by 2*pi when the left ThumbBase crosses the boundary).  The thumb's
plateaus come from the vendor ``<side>_thumb_tip`` frame's displacement along (index MCP - thumb
CMC), which grows as the thumb curls; check 4 fails if a thumb joint is flipped or turns back
against that ground truth, and the script fails if any joint's unwrapped travel exceeds 360 deg or
its per-frame jump exceeds 45 deg.

Every check the plugin's harness repeats is derived here from the URDF and the recording, and the
script exits non-zero when one fails.  ``--flip-spread`` / ``--flip-thumb`` invert the derived sign
and ``--identity-gain`` disables the gain calibration for the self-test only: the script must fail
with any of them, which proves the data-driven checks actually constrain the mapping instead of
restating it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Vendor URDFs.  These are the two files the plugin's constants are derived from.
# --------------------------------------------------------------------------------------------------
DEFAULT_URDF_DIR = Path(
    os.environ.get(
        "IO_EXO_URDF_DIR",
        "/home/io/IsaacTeleop/build/vendor/io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4"
        "/src/exo_tf_pkg/description",
    )
)
HUMAN_URDF = "blender_human.urdf"
MECHANISM_URDF = "blender_human_skeleton_v5.urdf"

# --------------------------------------------------------------------------------------------------
# Anatomy: numbers that are not a joint origin/rpy in the human URDF.
# --------------------------------------------------------------------------------------------------
#! Index-to-little MCP span of the A-pose, metres.  The exported human model leaves the
#! metacarpals straight down -Z (all hand-joint rpy are zero in this export), which puts the
#! knuckles 31 mm apart; an adult hand is about 55-60 mm.  The fan below keeps the URDF's CMC
#! positions and metacarpal lengths and only chooses the A-pose direction of each metacarpal.
MCP_SPAN_M = 0.056
#! Knuckle x positions of the A-pose, metres, in the left-hand OpenXR wrist frame.
MCP_X = {"index": 0.0285, "middle": 0.0095, "ring": -0.0095, "little": -0.0275}
#! Distal phalanx beyond the human model's last (DIP/IP) joint, metres.
DISTAL_PHALANX_M = 0.022
#! The thumb's rest frame: the ThumbBase rest yaw about the palm normal, then a lift about the
#! mechanism's Thumb1 axis so the thumb leaves the palm instead of hanging parallel to the
#! fingers.  Both values come from blender_human_skeleton_v5.urdf (ThumbBase rpy 0 0 0.99442,
#! Thumb1 rpy 1.5708 1.5708 0 and axis 0 0 1); the lift magnitude keeps the open thumb clear of
#! the index while staying inside the anatomical range.
THUMB_BASE_YAW = 0.99442
THUMB_REST_LIFT = 0.55
#! Mechanism thumb axes, mapped from the v5 base frame (x, y, z) to the OpenXR wrist frame
#! (x, -z, -y).  ThumbBase/Thumb1 lie in the palm plane (lift), Thumb2..Thumb4 about the palm
#! normal (flexion in the palm plane).  The flex-axis sign is data-derived: +Y moved the model tip
#! away from the index on both recordings, -Y moved it with the vendor thumb tip.
THUMB_BASE_AXIS = (0.54, 0.0, -0.84)
THUMB_LIFT_AXIS = (0.84, 0.0, 0.54)
THUMB_FLEX_AXIS = (0.0, -1.0, 0.0)
#! Video-rate recordings count; the unwrapped travel and per-frame jump limits the script asserts.
MAX_TRAVEL = 2.0 * math.pi
MAX_FRAME_JUMP = math.radians(45.0)

#! Human anatomical maximum per mechanism slot (F1..F4), radians.  Slot 0 is the knuckle spread,
#! which keeps the plugin's symmetric +/-25 deg clamp; slots 1..3 are MCP/PIP/DIP for the fingers and
#! CMC/MCP/IP for the thumb.  References: finger MCP 85, PIP 95, DIP 70 deg; thumb CMC 35, MCP 55,
#! IP 75 deg.  The recorded open-to-fist linkage travel is mapped onto these.
HUMAN_MAX = {
    "thumb": (
        math.radians(25.0),
        math.radians(35.0),
        math.radians(55.0),
        math.radians(75.0),
    ),
    "index": (
        math.radians(25.0),
        math.radians(85.0),
        math.radians(95.0),
        math.radians(70.0),
    ),
    "middle": (
        math.radians(25.0),
        math.radians(85.0),
        math.radians(95.0),
        math.radians(70.0),
    ),
    "ring": (
        math.radians(25.0),
        math.radians(85.0),
        math.radians(95.0),
        math.radians(70.0),
    ),
    "little": (
        math.radians(25.0),
        math.radians(85.0),
        math.radians(95.0),
        math.radians(70.0),
    ),
}
#! ThumbBase is the thumb's revolute mount, not a flexion joint; the plugin keeps its +/-90 deg clamp.
THUMB_BASE_MAX = math.radians(90.0)
#! Dead band at the open end, radians: the plugin allows this much extension below the zero.
DEAD_BAND = math.radians(5.0)
#! Smallest calibrated fist travel the gain can divide by, radians; below this the plugin uses 1:1.
MIN_FIST_TRAVEL = 0.05

FINGERS = ("thumb", "index", "middle", "ring", "little")
#! finger -> human URDF joint names (CMC, MCP, PIP, DIP); the thumb has CMC, MCP, IP.
HUMAN_CHAIN = {
    "thumb": ("HandThumb1", "HandThumb2", "HandThumb3"),
    "index": ("InHandIndex", "HandIndex1", "HandIndex2", "HandIndex3"),
    "middle": ("InHandMiddle", "HandMiddle1", "HandMiddle2", "HandMiddle3"),
    "ring": ("InHandRing", "HandRing1", "HandRing2", "HandRing3"),
    "little": ("InHandPinky", "HandPinky1", "HandPinky2", "HandPinky3"),
}
#! Mechanism chain per finger: (F1..F4, parent link suffix), exactly as the plugin reads it.
MECH_CHAIN = {
    "thumb": (("Thumb1", "Thumb2", "Thumb3", "Thumb4"), "ThumbBase"),
    "index": (("Index1", "Index2", "Index3", "Index4"), "Base"),
    "middle": (("Middle1", "Middle2", "Middle3", "Middle4"), "Base"),
    "ring": (("Ring1", "Ring2", "Ring3", "Ring4"), "Base"),
    "little": (("Pinky1", "Pinky2", "Pinky3", "Pinky4"), "Base"),
}
#! Vendor fingertip frames, used by the data-driven checks.
TIP_FRAME = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "little": "pinky_tip",
}
#! Frame whose quaternion is the mechanism root; the v5 driver publishes it as a link and a joint.
ROOT_FRAME = "Base"


# --------------------------------------------------------------------------------------------------
# Quaternion / rotation helpers.  Same conventions as io_exo_glove_ros2_hand.cpp and the URDF.
# --------------------------------------------------------------------------------------------------
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
    length = math.sqrt(sum(c * c for c in axis))
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(0.5 * angle)
    return (
        axis[0] / length * s,
        axis[1] / length * s,
        axis[2] / length * s,
        math.cos(0.5 * angle),
    )


def from_rpy(rpy):
    # URDF fixed-axis rpy: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    roll, pitch, yaw = rpy
    return qmul(
        qmul(axis_angle((0.0, 0.0, 1.0), yaw), axis_angle((0.0, 1.0, 0.0), pitch)),
        axis_angle((1.0, 0.0, 0.0), roll),
    )


def to_rpy(q):
    x, y, z, w = q
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def qnormalize(q):
    length = math.sqrt(sum(c * c for c in q))
    return tuple(c / length for c in q)


def rotate(q, v):
    r = qmul(qmul(q, (v[0], v[1], v[2], 0.0)), qconj(q))
    return (r[0], r[1], r[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def norm(v):
    length = math.sqrt(dot(v, v))
    return (v[0] / length, v[1] / length, v[2] / length)


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def signed_angle(q, axis):
    vector, w = [q[0], q[1], q[2]], q[3]
    if w < 0.0:
        vector, w = [-c for c in vector], -w
    return 2.0 * math.atan2(sum(vector[i] * axis[i] for i in range(3)), w)


def mirror_quaternion(q):
    # M * R * M for the x mirror M = diag(-1, 1, 1): axis (-x, y, z), angle negated.
    return (q[0], -q[1], -q[2], q[3])


#! The human model is authored for the left hand with its bones along -X and its palm normal
#! along +Y (see its hand joints); rotating -90 deg about Z puts it in the OpenXR wrist frame
#! (-Z along the bones, +Y dorsal, +X little-to-index).
MAP_HUMAN = axis_angle((0.0, 0.0, 1.0), -math.pi / 2.0)


# --------------------------------------------------------------------------------------------------
# URDF parsing.
# --------------------------------------------------------------------------------------------------
class UrdfJoint:
    __slots__ = ("axis", "child", "kind", "name", "parent", "rpy", "xyz")

    def __init__(self, name, parent, child, xyz, rpy, axis, kind):
        self.name = name
        self.parent = parent
        self.child = child
        self.xyz = xyz
        self.rpy = rpy
        self.axis = axis
        self.kind = kind


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            if origin.get("xyz"):
                xyz = tuple(float(c) for c in origin.get("xyz").split())
            if origin.get("rpy"):
                rpy = tuple(float(c) for c in origin.get("rpy").split())
        axis = (0.0, 0.0, 0.0)
        axis_element = element.find("axis")
        if axis_element is not None and axis_element.get("xyz"):
            axis = tuple(float(c) for c in axis_element.get("xyz").split())
        joints[element.get("name")] = UrdfJoint(
            element.get("name"),
            element.find("parent").get("link"),
            element.find("child").get("link"),
            xyz,
            rpy,
            axis,
            element.get("type"),
        )
    return joints


def parent_map(joints):
    by_child = {joint.child: joint for joint in joints.values()}
    return by_child


def chain_to(joints, start_link, target_link):
    """Joints from @p start_link down to @p target_link, root first."""
    by_child = parent_map(joints)
    chain = []
    link = target_link
    while link != start_link:
        joint = by_child.get(link)
        if joint is None:
            raise KeyError("no joint path from %s to %s" % (start_link, target_link))
        chain.append(joint)
        link = joint.parent
    return list(reversed(chain))


def rest_pose(joints, start_link, target_link):
    """(position, orientation) of @p target_link in @p start_link's frame at zero angles."""
    position = (0.0, 0.0, 0.0)
    rotation = (0.0, 0.0, 0.0, 1.0)
    for joint in chain_to(joints, start_link, target_link):
        position = add(position, rotate(rotation, joint.xyz))
        rotation = qmul(rotation, from_rpy(joint.rpy))
    return position, rotation


def human_hand_geometry(path):
    """The human model's left-hand CMC positions and bone lengths, in the OpenXR wrist frame."""
    joints = parse_urdf(path)
    start = "link_LeftHand_Y"
    geometry = {}
    for finger, names in HUMAN_CHAIN.items():
        targets = ["link_Left%s_R" % name for name in names]
        # The joint is the transform into the _R link; the chain to each _R link gives the
        # joint origin itself (the R/P/Y joints after it have zero origins).
        positions = []
        for target in targets:
            position, _ = rest_pose(joints, start, target)
            positions.append(position)
        # The distal tip is one phalanx past the human model's last joint.
        last = positions[-1]
        previous = positions[-2]
        direction = norm(sub(last, previous))
        positions.append(add(last, tuple(DISTAL_PHALANX_M * c for c in direction)))
        geometry[finger] = [rotate(MAP_HUMAN, position) for position in positions]
    return joints, geometry


def mechanism_rest(path):
    """The v5 mechanism's per-joint (rpy, axis) and the first-joint positions, both hands."""
    joints = parse_urdf(path)
    rest = {}
    for hand in ("Left", "Right"):
        rest[hand] = {}
        for finger, (links, _) in MECH_CHAIN.items():
            rest[hand][finger] = []
            for link in links:
                joint = joints["joint_%sSkeleton%s" % (hand, link)]
                rest[hand][finger].append((joint.rpy, joint.axis))
        base = joints["joint_%sSkeletonThumbBase" % hand]
        rest[hand]["base"] = (base.rpy, base.axis)
    return joints, rest


# --------------------------------------------------------------------------------------------------
# The anatomical model, in the left-hand OpenXR wrist frame.  The C++ mirrors it for the right.
# --------------------------------------------------------------------------------------------------
def metacarpal_fan(geometry):
    """Rest rotation per finger that puts the MCP at the A-pose knuckle line (y = 0, fanned x)."""
    fan = {}
    for finger in ("index", "middle", "ring", "little"):
        cmc = geometry[finger][0]
        length = distance(geometry[finger][0], geometry[finger][1])
        target = (MCP_X[finger], 0.0, 0.0)
        dx, dy = target[0] - cmc[0], target[1] - cmc[1]
        dz = -math.sqrt(max(0.0, length * length - dx * dx - dy * dy))
        direction = norm((dx, dy, dz))
        # Minimal rotation from the straight -Z bone to the fanned metacarpal.
        axis = (direction[1], -direction[0], 0.0)
        angle = math.acos(max(-1.0, min(1.0, -direction[2])))
        fan[finger] = (
            axis_angle(axis, angle) if abs(angle) > 1e-12 else (0.0, 0.0, 0.0, 1.0)
        )
    fan["thumb"] = qmul(
        axis_angle(THUMB_LIFT_AXIS, THUMB_REST_LIFT),
        axis_angle((0.0, 1.0, 0.0), -THUMB_BASE_YAW),
    )
    return fan


def human_anatomy(geometry):
    """Landmark offsets and rest rotations for the C++ tables."""
    fan = metacarpal_fan(geometry)
    anatomy = {}
    for finger in FINGERS:
        points = geometry[finger]
        offsets = [(0.0, 0.0, 0.0)]
        for index in range(1, len(points)):
            offsets.append(sub(points[index], points[index - 1]))
        anatomy[finger] = {"cmc": points[0], "offsets": offsets[1:], "fan": fan[finger]}
    return anatomy


def build_skeleton(anatomy, spread, bend, thumb_base):
    """Positions of every landmark, for the left hand, from anatomical angles (radians)."""
    joints = {}
    for finger in ("index", "middle", "ring", "little"):
        offsets = anatomy[finger]["offsets"]
        position = anatomy[finger]["cmc"]
        joints[(finger, 0)] = position
        position = add(
            position,
            rotate(
                anatomy[finger]["fan"],
                (0.0, 0.0, -distance((0.0, 0.0, 0.0), offsets[0])),
            ),
        )
        joints[(finger, 1)] = position
        rotation = axis_angle((0.0, 1.0, 0.0), -spread[finger])
        rotation = qmul(rotation, axis_angle((1.0, 0.0, 0.0), -bend[finger][0]))
        for k in range(1, len(offsets)):
            position = add(position, rotate(rotation, offsets[k]))
            joints[(finger, k + 1)] = position
            if k < len(bend[finger]):
                rotation = qmul(rotation, axis_angle((1.0, 0.0, 0.0), -bend[finger][k]))
    # Thumb: its own frame and axes; ThumbBase and Thumb1 lift the metacarpal, Thumb2..4 flex
    # it in the palm plane.
    offsets = anatomy["thumb"]["offsets"]
    rotation = qmul(
        qmul(
            axis_angle(THUMB_BASE_AXIS, thumb_base),
            axis_angle(THUMB_LIFT_AXIS, spread["thumb"]),
        ),
        anatomy["thumb"]["fan"],
    )
    position = anatomy["thumb"]["cmc"]
    joints[("thumb", 0)] = position
    for k in range(1, len(offsets) + 1):
        rotation = qmul(axis_angle(THUMB_FLEX_AXIS, -bend["thumb"][k - 1]), rotation)
        position = add(position, rotate(rotation, offsets[k - 1]))
        joints[("thumb", k)] = position
    return joints


def plane_normal(points):
    centroid = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
    centred = [sub(p, centroid) for p in points]
    cov = [
        [sum(c[i] * c[j] for c in centred) / len(points) for j in range(3)]
        for i in range(3)
    ]
    trace = sum(cov[i][i] for i in range(3))
    matrix = [
        [(trace if i == j else 0.0) - cov[i][j] for j in range(3)] for i in range(3)
    ]
    normal = norm((0.0, 1.0, 0.0))
    for _ in range(200):
        step = tuple(sum(matrix[i][j] * normal[j] for j in range(3)) for i in range(3))
        if math.sqrt(dot(step, step)) < 1e-12:
            break
        normal = norm(step)
    return normal if normal[1] >= 0.0 else tuple(-c for c in normal)


def neutral_numbers(anatomy):
    zero = dict.fromkeys(FINGERS, 0.0)
    bends = {finger: [0.0, 0.0, 0.0] for finger in FINGERS}
    joints = build_skeleton(anatomy, zero, bends, 0.0)
    index, little = joints[("index", 1)], joints[("little", 1)]
    flat = [
        joints[(finger, landmark)]
        for finger in ("index", "middle", "ring", "little")
        for landmark in range(1, 1 + len(anatomy[finger]["offsets"]))
    ]
    normal = plane_normal(flat)
    across = sub(little, index)
    thumb = joints[("thumb", 0)]
    t = dot(sub(thumb, index), across) / dot(across, across)
    return {
        "span_x_m": abs(little[0] - index[0]),
        "span_3d_m": distance(index, little),
        "level_deg": math.degrees(math.acos(max(-1.0, min(1.0, normal[1])))),
        "thumb_t": t,
        "joints": joints,
    }


# --------------------------------------------------------------------------------------------------
# Recording: raw mechanism angles, plateaus, calibration.
# --------------------------------------------------------------------------------------------------
class Recording:
    def __init__(self, path):
        self.frames, self.times = [], []
        for line in open(path, encoding="utf-8"):
            if line.strip():
                message = json.loads(line)
                self.frames.append(message["frames"])
                self.times.append(message["t"])
        if len(self.frames) < 100:
            raise SystemExit("need at least 100 messages")
        self.times = [t - self.times[0] for t in self.times]


def raw_angles(recording, mech):
    """Absolute mechanism joint angles per frame, relative to the URDF zero pose."""
    rows = []
    for frames in recording.frames:
        row = {}
        for hand in ("Left", "Right"):
            prefix = "link_%sSkeleton" % hand
            root = qnormalize(tuple(frames[prefix + ROOT_FRAME][3:7]))
            base_rpy, base_axis = mech[hand]["base"]
            base_frame = qnormalize(tuple(frames[prefix + "ThumbBase"][3:7]))
            residual = qmul(qconj(from_rpy(base_rpy)), qmul(qconj(root), base_frame))
            row[(hand, "thumb", "base")] = signed_angle(residual, base_axis)
            for finger, (links, _parent) in MECH_CHAIN.items():
                if finger == "thumb":
                    parent_rotation = base_frame
                else:
                    parent_rotation = root
                for slot, link in enumerate(links):
                    child = qnormalize(tuple(frames[prefix + link][3:7]))
                    rpy, axis = mech[hand][finger][slot]
                    residual = qmul(
                        qconj(from_rpy(rpy)), qmul(qconj(parent_rotation), child)
                    )
                    row[(hand, finger, slot)] = signed_angle(residual, axis)
                    parent_rotation = child
        rows.append(row)
    return rows


def unwrap_series(values):
    """Unwrap a joint's raw signed angles so the value stays continuous across +-pi.

        The first sample initialises; every later sample has +-2*pi added until it is within pi of the
        previous one.  This is the same rule the old joint-angle path applied per message; the tip-driven
    runtime no longer needs it, but the derivation below still unwraps before fitting.
    """
    values = list(values)
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        previous = out[-1]
        while value - previous > math.pi:
            value -= 2.0 * math.pi
        while value - previous < -math.pi:
            value += 2.0 * math.pi
        out.append(value)
    return out


def pearson(a, b):
    """Pearson correlation of two equal-length sequences, or 0 when either is constant."""
    count = len(a)
    mean_a = sum(a) / count
    mean_b = sum(b) / count
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 1e-18 or var_b <= 1e-18:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def median_smooth(values, half_window=15):
    """Running median, to compare a joint with the vendor tip without frame-to-frame noise."""
    out = []
    for index in range(len(values)):
        chunk = values[max(0, index - half_window) : index + half_window + 1]
        out.append(sorted(chunk)[len(chunk) // 2])
    return out


def steady_plateaus(recording, values, window_s=0.35, tol=0.008, min_s=0.5):
    """Steady runs of @p values: a smoothed window whose range stays under @p tol for @p min_s."""
    count = len(values)
    dt = recording.times[1] - recording.times[0]
    half = max(1, int(window_s / dt / 2))
    smoothed = []
    for index in range(count):
        chunk = values[max(0, index - half) : index + half + 1]
        smoothed.append(sorted(chunk)[len(chunk) // 2])
    stable = []
    for index in range(count):
        chunk = smoothed[max(0, index - half) : index + half + 1]
        stable.append(max(chunk) - min(chunk) < tol)
    runs = []
    start = None
    for index, is_stable in enumerate(stable):
        if is_stable and start is None:
            start = index
        if not is_stable and start is not None:
            if recording.times[index - 1] - recording.times[start] >= min_s:
                runs.append([start, index - 1])
            start = None
    if start is not None and recording.times[-1] - recording.times[start] >= min_s:
        runs.append([start, count - 1])
    return runs


def thumb_progress(recording):
    """Vendor ground truth per frame: the thumb tip's displacement along (index MCP - thumb CMC).

    Summed over both hands.  It grows monotonically as the thumb curls, which is what the thumb's
    open/fist plateaus and check 4 are anchored to.  Positions are the published vendor frames in
    the mechanism root frame, so no model or calibration enters.
    """
    progress = []
    for frames in recording.frames:
        total = 0.0
        for hand in ("Left", "Right"):
            prefix = "link_%sSkeleton" % hand
            base = frames[prefix + "ThumbBase"][:3]
            tip = frames["%s_thumb_tip" % hand.lower()][:3]
            index = frames[prefix + "Index1"][:3]
            total += dot(sub(tip, base), norm(sub(index, base)))
        progress.append(total)
    return progress


def thumb_plateau_frames(recording, rows, keys):
    """Open (least curled) and full-flexion (most curled) medoid frames from the vendor tip."""
    progress = thumb_progress(recording)
    runs = steady_plateaus(recording, progress)
    if not runs:
        raise SystemExit("no steady thumb plateau found")
    means = [
        (run, sum(progress[run[0] : run[1] + 1]) / (run[1] - run[0] + 1))
        for run in runs
    ]
    open_run = min(means, key=lambda item: item[1])[0]
    fist_run = max(means, key=lambda item: item[1])[0]
    open_frame = medoid(open_run, rows, keys)
    fist_frame = medoid(fist_run, rows, keys)
    return open_frame, fist_frame, open_run, fist_run


def finger_plateau_frames(recording, rows, keys):
    """Open and closed medoid frames from the finger knuckle-to-tip openness.

    Returns ``(open_frame, fist_frame, exercised)``.  A recording with no finger excursion (the
    thumb-only one) returns the same frame twice and ``exercised=False``; the caller then keeps the
    finger rows out of its checks instead of inventing a fist.
    """
    motion = [0.0] + [
        sum(abs(rows[index][key] - rows[index - 1][key]) for key in keys)
        for index in range(1, len(rows))
    ]
    runs = stable_runs(motion, recording.times)
    if not runs:
        raise SystemExit("no steady hand plateau found")
    medians = [
        sum(
            openness(recording, recording.frames[index])
            for index in range(run[0], run[1] + 1)
        )
        / (run[1] - run[0] + 1)
        for run in runs
    ]
    candidates = [
        index
        for index, run in enumerate(runs)
        if recording.times[run[1]] - recording.times[run[0]] >= 3
    ]
    if not candidates:
        candidates = list(range(len(runs)))
    highest = max(medians[index] for index in candidates)
    open_candidates = [index for index in candidates if medians[index] >= 0.9 * highest]
    best_open = max(
        open_candidates,
        key=lambda index: (
            recording.times[runs[index][1]] - recording.times[runs[index][0]]
        ),
    )
    open_run = runs[best_open]
    closed = [
        index
        for index, run in enumerate(runs)
        if medians[index] < 0.9 * medians[best_open]
        and recording.times[run[1]] - recording.times[run[0]] >= 2
    ]
    open_frame = medoid(open_run, rows, keys)
    if not closed:
        return open_frame, open_frame, False
    best_closed = max(
        closed,
        key=lambda index: (
            recording.times[runs[index][1]] - recording.times[runs[index][0]]
        ),
    )
    fist_frame = medoid(runs[best_closed], rows, keys)
    spread_travel = max(
        abs(rows[fist_frame][(hand, finger, 0)] - rows[open_frame][(hand, finger, 0)])
        for hand in ("Left", "Right")
        for finger in FINGERS
        if finger != "thumb"
    )
    return open_frame, fist_frame, spread_travel >= 0.15


def travel_limits(rows, keys):
    """Largest unwrapped travel and per-frame jump over every joint, with where it occurred."""
    worst_travel = (0.0, None)
    worst_jump = (0.0, None)
    for key in keys:
        series = [row[key] for row in rows]
        travel = max(series) - min(series)
        jump = max(
            abs(series[index] - series[index - 1]) for index in range(1, len(series))
        )
        if travel > worst_travel[0]:
            worst_travel = (travel, key)
        if jump > worst_jump[0]:
            worst_jump = (jump, key)
    return worst_travel, worst_jump


def openness(recording, frames):
    total = 0.0
    for hand in ("Left", "Right"):
        for finger, tip in TIP_FRAME.items():
            knuckle = frames["link_%sSkeleton%s" % (hand, MECH_CHAIN[finger][0][0])][:3]
            fingertip = frames["%s_%s" % (hand.lower(), tip)][:3]
            total += distance(knuckle, fingertip)
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


def medoid(run, rows, keys):
    median = {
        key: sorted(rows[i][key] for i in range(run[0], run[1] + 1))[
            (run[1] - run[0]) // 2
        ]
        for key in keys
    }
    return min(
        range(run[0], run[1] + 1),
        key=lambda i: sum((rows[i][key] - median[key]) ** 2 for key in keys),
    )


def solve(matrix, rhs):
    size = len(matrix)
    work = [matrix[row][:] + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        work[column], work[pivot] = work[pivot], work[column]
        if abs(work[column][column]) < 1e-12:
            return None
        for row in range(size):
            if row != column:
                factor = work[row][column] / work[column][column]
                for k in range(column, size + 1):
                    work[row][k] -= factor * work[column][k]
    return [work[i][size] / work[i][i] for i in range(size)]


def regression(points, design):
    """Least-squares coefficients of each output in @points on the columns of @design."""
    size = len(design[0])
    matrix = [
        [sum(row[a] * row[b] for row in design) for b in range(size)]
        for a in range(size)
    ]
    coefficients = []
    for axis in range(3):
        rhs = [
            sum(design[row][a] * points[row][axis] for row in range(len(points)))
            for a in range(size)
        ]
        coefficients.append(solve(matrix, rhs))
    return coefficients


def base_position(frames, hand, name):
    root = qnormalize(tuple(frames["link_%sSkeleton%s" % (hand, ROOT_FRAME)][3:7]))
    return rotate(
        qconj(root),
        sub(
            tuple(frames[name][:3]),
            tuple(frames["link_%sSkeleton%s" % (hand, ROOT_FRAME)][:3]),
        ),
    )


def base_rotation(frames, hand, name):
    root = qnormalize(tuple(frames["link_%sSkeleton%s" % (hand, ROOT_FRAME)][3:7]))
    return qmul(qconj(root), qnormalize(tuple(frames[name][3:7])))


#! v5 base frame -> OpenXR wrist frame: (x, y, z) -> (x, -z, -y).
def mechanism_to_anatomical(v):
    return (v[0], -v[2], -v[1])


#! Fixed pose of the vendor's fingertip frame in the mechanism's last link (metres), measured
#! from the recording and equal for every finger and both hands.
TIP_OFFSET = (-0.0026, 0.0050, -0.0270)


def fk_tip(mech_urdf, hand, angles, finger):
    """The vendor ``<side>_<finger>_tip`` frame for arbitrary absolute joint angles.

    @p angles maps ``(hand, finger, slot)`` (and ``"base"`` for ThumbBase) to the absolute URDF
    angle, i.e. the values ``raw_angles`` produces.  The tip frame is a fixed offset in the last
    link, so this is the same kinematic model the driver uses; the derivation validates it against
    the recorded tip frames before using it to hold the bends steady.
    """
    last = {
        "thumb": "Thumb4",
        "index": "Index4",
        "middle": "Middle4",
        "ring": "Ring4",
        "little": "Pinky4",
    }[finger]
    position = (0.0, 0.0, 0.0)
    rotation = (0.0, 0.0, 0.0, 1.0)
    for joint in chain_to(
        mech_urdf, "link_%sSkeletonBase" % hand, "link_%sSkeleton%s" % (hand, last)
    ):
        suffix = joint.name[len("joint_%sSkeleton" % hand) :]
        if suffix == "ThumbBase":
            theta = angles[(hand, "thumb", "base")]
        else:
            # The path to one finger's last link contains only that finger's joints (and the
            # thumb's mount, handled above).
            theta = angles[(hand, finger, int(suffix[-1]) - 1)]
        position = add(position, rotate(rotation, joint.xyz))
        rotation = qmul(
            qmul(rotation, from_rpy(joint.rpy)), axis_angle(joint.axis, theta)
        )
    return add(position, rotate(rotation, TIP_OFFSET))


# --------------------------------------------------------------------------------------------------
# Derivation.
# --------------------------------------------------------------------------------------------------
def mechanism_spread_sign(mech_urdf):
    """Sign that turns a positive v5 first-joint angle into "towards the index".

    The v5 first joint turns about its local +Z; the finger extends +Y, so R_z(+theta) sends the
    fingertip towards -X.  A positive first-joint angle therefore moves the finger towards the
    index exactly when the index knuckle lies on the -X side of the little knuckle.
    """
    signs = {}
    for hand in ("Left", "Right"):
        index = mech_urdf["joint_%sSkeletonIndex1" % hand].xyz
        little = mech_urdf["joint_%sSkeletonPinky1" % hand].xyz
        towards_index = sub(index, little)
        signs[hand] = 1.0 if -towards_index[0] > 0.0 else -1.0
    return signs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="JSONL capture of /io_fusion/tf_exoskeleton")
    parser.add_argument(
        "--thumb-recording",
        type=Path,
        default=None,
        help="thumb-only JSONL capture the thumb block is derived from (default: the positional recording)",
    )
    parser.add_argument("--urdf-dir", type=Path, default=DEFAULT_URDF_DIR)
    parser.add_argument(
        "--flip-spread",
        action="store_true",
        help="self-test: invert the measured spread sign",
    )
    parser.add_argument(
        "--flip-thumb",
        action="store_true",
        help="self-test: invert the thumb flexion sign",
    )
    parser.add_argument(
        "--identity-gain",
        action="store_true",
        help="self-test: disable the per-joint gain (the old 1:1 linkage-angle mapping)",
    )
    args = parser.parse_args()

    human = args.urdf_dir / HUMAN_URDF
    mechanism = args.urdf_dir / MECHANISM_URDF
    if not human.exists() or not mechanism.exists():
        raise SystemExit("URDFs not found under %s" % args.urdf_dir)

    _human_joints, geometry = human_hand_geometry(human)
    mech_urdf, mech = mechanism_rest(mechanism)
    anatomy = human_anatomy(geometry)

    recording = Recording(args.recording)
    thumb_recording = (
        recording if args.thumb_recording is None else Recording(args.thumb_recording)
    )
    rows = raw_angles(recording, mech)
    keys = list(rows[0])
    for key in keys:
        for index, value in enumerate(unwrap_series([row[key] for row in rows])):
            rows[index][key] = value
    if thumb_recording is recording:
        thumb_rows = rows
    else:
        thumb_rows = raw_angles(thumb_recording, mech)
        for key in keys:
            for index, value in enumerate(
                unwrap_series([row[key] for row in thumb_rows])
            ):
                thumb_rows[index][key] = value

    print("recording: %s, %d messages" % (args.recording, len(rows)))
    if thumb_recording is not recording:
        print(
            "thumb recording: %s, %d messages" % (args.thumb_recording, len(thumb_rows))
        )

    # ---- unwrapped motion: a joint must stay continuous and inside a hand's range ----------------
    worst_travel, worst_jump = travel_limits(rows, keys)
    travel_ok = (
        worst_travel[0] <= MAX_TRAVEL + 1e-9 and worst_jump[0] <= MAX_FRAME_JUMP + 1e-9
    )
    print(
        "\ncheck 0: unwrapped motion on %s (limit %.0f deg travel, %.0f deg per frame)"
        % (args.recording, math.degrees(MAX_TRAVEL), math.degrees(MAX_FRAME_JUMP))
    )
    print(
        "  [%s] largest travel %7.2f deg %s | largest per-frame jump %6.2f deg %s"
        % (
            "PASS" if travel_ok else "FAIL",
            math.degrees(worst_travel[0]),
            worst_travel[1],
            math.degrees(worst_jump[0]),
            worst_jump[1],
        )
    )
    if thumb_recording is not recording:
        thumb_travel, thumb_jump = travel_limits(thumb_rows, keys)
        travel_ok = (
            travel_ok
            and thumb_travel[0] <= MAX_TRAVEL + 1e-9
            and thumb_jump[0] <= MAX_FRAME_JUMP + 1e-9
        )
        print(
            "  [%s] thumb recording: largest travel %7.2f deg %s | largest jump %6.2f deg %s"
            % (
                "PASS" if travel_ok else "FAIL",
                math.degrees(thumb_travel[0]),
                thumb_travel[1],
                math.degrees(thumb_jump[0]),
                thumb_jump[1],
            )
        )

    # ---- the fingers: open and closed plateaus from knuckle-to-tip openness ---------------------
    finger_open, finger_fist, finger_exercised = finger_plateau_frames(
        recording, rows, keys
    )
    print(
        "\nfinger plateaus on %s: open %.1f s, %s %.1f s"
        % (
            args.recording,
            recording.times[finger_open],
            "fist" if finger_exercised else "not exercised",
            recording.times[finger_fist],
        )
    )
    finger_zero = {key: rows[finger_open][key] for key in keys}
    finger_fist_values = {key: rows[finger_fist][key] for key in keys}
    spread_sign = mechanism_spread_sign(mech_urdf)
    finger_signs = {}
    for key in keys:
        hand, finger, slot = key
        if slot == 0 and finger != "thumb":
            # The knuckle's measured sign is a property of the mechanism, not of the plateau.
            finger_signs[key] = spread_sign[hand]
        else:
            finger_signs[key] = (
                1.0 if finger_fist_values[key] >= finger_zero[key] else -1.0
            )
    if args.flip_spread:
        for key in keys:
            if key[2] == 0 and key[1] != "thumb":
                finger_signs[key] = -finger_signs[key]

    # ---- the thumb: plateaus from the vendor thumb tip, and the left mount's rest rotation -------
    thumb_open_frame, thumb_fist_frame, thumb_open_run, _thumb_fist_run = (
        thumb_plateau_frames(thumb_recording, thumb_rows, keys)
    )
    # The left ThumbBase rest is a rig constant, not a per-session zero: the v5 left mount has no pi
    # roll (the right one does), so the physical rest is a half turn from the URDF rest and the open
    # reading sits near +pi.  Solve for the roll that zeroes the open plateau: the URDF rest plus the
    # pi roll and the open offset (mod 2*pi), netted as base_delta, so the mount reads 0 at open.
    base_delta = sorted(
        thumb_rows[index][("Left", "thumb", "base")]
        for index in range(thumb_open_run[0], thumb_open_run[1] + 1)
    )[(thumb_open_run[1] - thumb_open_run[0]) // 2]
    left_base_rpy = (
        mech["Left"]["base"][0][0] + base_delta,
        mech["Left"]["base"][0][1],
        mech["Left"]["base"][0][2],
    )
    print(
        "\nthumb plateaus on %s: open %.1f s, fist %.1f s"
        % (
            args.thumb_recording or args.recording,
            thumb_recording.times[thumb_open_frame],
            thumb_recording.times[thumb_fist_frame],
        )
    )
    print(
        "  left ThumbBase rest %.5f, %.5f, %.5f rad (URDF roll %+.5f rad = %+.2f deg open offset)"
        % (*left_base_rpy, base_delta, math.degrees(base_delta))
    )
    thumb_zero = {}
    thumb_fist = {}
    thumb_signs = {}
    for hand in ("Left", "Right"):
        offset = -base_delta if hand == "Left" else 0.0
        for slot in ("base", 0, 1, 2, 3):
            key = (hand, "thumb", slot)
            source = offset if slot == "base" else 0.0
            thumb_zero[key] = thumb_rows[thumb_open_frame][key] + source
            thumb_fist[key] = thumb_rows[thumb_fist_frame][key] + source
            thumb_signs[key] = 1.0 if thumb_fist[key] >= thumb_zero[key] else -1.0
    if args.flip_thumb:
        for hand in ("Left", "Right"):
            for slot in (1, 2, 3):
                key = (hand, "thumb", slot)
                thumb_signs[key] = -thumb_signs[key]

    zero = dict(finger_zero)
    fist = dict(finger_fist_values)
    signs = dict(finger_signs)
    zero.update(thumb_zero)
    fist.update(thumb_fist)
    signs.update(thumb_signs)

    # ---- calibration helpers (the same arithmetic the plugin performs) -------------------------
    def anatomical(hand, row):
        # The right hand's anatomical angles are the same as the left's: the build mirrors the
        # geometry, not the articulation.
        spread = {}
        bend = {}
        for finger in FINGERS:
            key = (hand, finger, 0)
            spread[finger] = signs[key] * (row[key] - zero[key])
            bend[finger] = [
                signs[(hand, finger, slot)]
                * (row[(hand, finger, slot)] - zero[(hand, finger, slot)])
                for slot in (1, 2, 3)
            ]
        base = signs[(hand, "thumb", "base")] * (
            row[(hand, "thumb", "base")] - zero[(hand, "thumb", "base")]
        )
        return spread, bend, base

    def calibrated_bend(row, hand, finger, slot, identity_gain=False):
        """Calibrated flexion of one bend slot, radians: gain, dead band and human maximum."""
        key = (hand, finger, slot)
        d = signs[key] * (row[key] - zero[key])
        if identity_gain:
            return min(max(d, 0.0), HUMAN_MAX[finger][slot])
        d_fist = signs[key] * (fist[key] - zero[key])
        value = (
            d if abs(d_fist) < MIN_FIST_TRAVEL else d / d_fist * HUMAN_MAX[finger][slot]
        )
        return min(max(value, -DEAD_BAND), HUMAN_MAX[finger][slot])

    def calibrated_spread(row, hand, finger):
        key = (hand, finger, 0)
        return signs[key] * (row[key] - zero[key])

    def calibrated_base(row, hand):
        key = (hand, "thumb", "base")
        offset = -base_delta if hand == "Left" else 0.0
        return signs[key] * (row[key] + offset - zero[key])

    def model_tip(anatomy, spread, bend, base, finger):
        joints = build_skeleton(anatomy, spread, bend, base)
        return joints[(finger, len(anatomy[finger]["offsets"]))]

    # ---- emit the C++ tables -------------------------------------------------------------------
    print(
        "\nstatic constexpr RigJointSpec kRigJoints[2][kRigFingerCount][kRigSlotCount] = {"
    )
    for hand in ("Left", "Right"):
        print("    // %s\n    {" % hand.lower())
        for finger in FINGERS:
            print("        {\n            // %s" % finger)
            for slot in range(4):
                rpy, axis = mech[hand][finger][slot]
                key = (hand, finger, slot)
                print(
                    "            { { %.5ff, %.5ff, %.5ff }, { %.1ff, %.1ff, %.1ff }, %.5ff, %+.1ff, %.5ff, %.5ff },"
                    % (
                        *rpy,
                        *axis,
                        zero[key],
                        signs[key],
                        fist[key],
                        HUMAN_MAX[finger][slot],
                    )
                )
            print("        },")
        print("    },")
    print("};")
    print("\nstatic constexpr RigJointSpec kThumbBase[2] = {")
    for hand in ("Left", "Right"):
        rpy, axis = mech[hand]["base"]
        if hand == "Left":
            rpy = left_base_rpy
        key = (hand, "thumb", "base")
        print(
            "    { { %.5ff, %.5ff, %.5ff }, { %.1ff, %.1ff, %.1ff }, %.5ff, %+.1ff, %.5ff, %.5ff }, // %s"
            % (
                *rpy,
                *axis,
                zero[key],
                signs[key],
                fist[key],
                THUMB_BASE_MAX,
                hand.lower(),
            )
        )
    print("};")

    fan = anatomy["thumb"]["fan"]
    print(
        "\n// Metacarpal A-pose rest rotations (rpy), left hand; the right mirrors in X."
    )
    print("static constexpr float kFingerFanRpy[kRigFingerCount - 1][3] = {")
    for finger in ("index", "middle", "ring", "little"):
        print(
            "    { %.5ff, %.5ff, %.5ff }, // %s"
            % (*to_rpy(anatomy[finger]["fan"]), finger)
        )
    print("};")
    print(
        "\nstatic constexpr float kThumbRestRpy[3] = { %.5ff, %.5ff, %.5ff };"
        % to_rpy(fan)
    )
    print(
        "static constexpr float kThumbBaseAxis[3] = { %.2ff, %.2ff, %.2ff };"
        % THUMB_BASE_AXIS
    )
    print(
        "static constexpr float kThumbLiftAxis[3] = { %.2ff, %.2ff, %.2ff };"
        % THUMB_LIFT_AXIS
    )
    print(
        "static constexpr float kThumbFlexAxis[3] = { %.2ff, %.2ff, %.2ff };"
        % THUMB_FLEX_AXIS
    )

    print(
        "\nthumb block derived from %s (degrees; gain = human_max / |fist - zero|):"
        % (args.thumb_recording or args.recording)
    )
    for hand in ("Left", "Right"):
        for slot, label in (
            ("base", "base"),
            (0, "lift"),
            (1, "CMC"),
            (2, "MCP"),
            (3, "IP"),
        ):
            key = (hand, "thumb", slot)
            gain = (
                math.degrees(HUMAN_MAX["thumb"][slot])
                / abs(math.degrees(fist[key] - zero[key]))
                if slot not in ("base", 0) and abs(fist[key] - zero[key]) > 1e-9
                else float("nan")
            )
            print(
                "  %-5s %-4s zero %8.2f  sign %+d  fist %8.2f  travel %7.2f  gain %.3f"
                % (
                    hand.lower(),
                    label,
                    math.degrees(zero[key]),
                    int(signs[key]),
                    math.degrees(fist[key]),
                    math.degrees(abs(fist[key] - zero[key])),
                    gain,
                )
            )
    print("  left ThumbBase rest rpy (rad): %.5f, %.5f, %.5f" % left_base_rpy)

    # ---- check 1: neutral-frame anatomy --------------------------------------------------------
    neutral = neutral_numbers(anatomy)
    print("\ncheck 1: neutral-frame anatomy")
    span_ok = abs(neutral["span_x_m"] - MCP_SPAN_M) < 0.002
    level_ok = neutral["level_deg"] < 3.0
    thumb_ok = neutral["thumb_t"] < 0.5
    print(
        "  [%s] index->little MCP lateral span %.1f mm (target %.0f mm)"
        % (
            "PASS" if span_ok else "FAIL",
            neutral["span_x_m"] * 1000.0,
            MCP_SPAN_M * 1000.0,
        )
    )
    print(
        "  [%s] table.level with a level wrist %.2f deg (allowed < 3 deg)"
        % ("PASS" if level_ok else "FAIL", neutral["level_deg"])
    )
    print(
        "  [%s] thumb metacarpal t along index->little %.3f (limit < 0.5)"
        % ("PASS" if thumb_ok else "FAIL", neutral["thumb_t"])
    )

    # ---- check 2: the URDF FK reproduces the recorded tip frames -------------------------------
    fk_ok = True
    last_frame = len(recording.frames) - 1
    for hand in ("Left", "Right"):
        for frame_index in (
            finger_open,
            finger_fist,
            min(1234, last_frame),
            min(4321, last_frame),
        ):
            for finger in FINGERS:
                recorded = base_position(
                    recording.frames[frame_index],
                    hand,
                    "%s_%s" % (hand.lower(), TIP_FRAME[finger]),
                )
                predicted = fk_tip(mech_urdf, hand, rows[frame_index], finger)
                if distance(recorded, predicted) > 1e-6:
                    fk_ok = False
    print("\ncheck 2: the URDF FK reproduces the recorded <side>_<finger>_tip frames")
    print(
        "  [%s] %s"
        % (
            "PASS" if fk_ok else "FAIL",
            "four frames x five fingers x both hands agree to 1e-6 m"
            if fk_ok
            else "mismatch: the mechanism URDF and the recording disagree",
        )
    )

    # ---- check 3: spread plateau, model tip vs vendor tip --------------------------------------
    print(
        "\ncheck 3: spread plateau, model tip vs vendor <side>_<finger>_tip (bends held)"
    )
    spread_ok = True
    if not finger_exercised:
        print("  [SKIP] no finger excursion in this recording (thumb-only capture)")
    else:
        for hand in ("Left", "Right"):
            for finger in ("index", "middle", "ring", "little"):
                values = [row[(hand, finger, 0)] for row in rows]
                low = min(values)
                high = max(values)
                open_value = rows[finger_open][(hand, finger, 0)]
                second_value = (
                    low if abs(low - open_value) > abs(high - open_value) else high
                )
                moved_row = dict(rows[finger_open])
                moved_row[(hand, finger, 0)] = second_value
                vendor_before = base_position(
                    recording.frames[finger_open],
                    hand,
                    "%s_%s" % (hand.lower(), TIP_FRAME[finger]),
                )
                vendor_after = fk_tip(mech_urdf, hand, moved_row, finger)
                spread, bend, base = anatomical(hand, rows[finger_open])
                model_base = build_skeleton(anatomy, spread, bend, base)
                tip_index = len(anatomy[finger]["offsets"])
                model_before = model_base[(finger, tip_index)]
                anatomical_spread = dict(spread)
                anatomical_spread[finger] = signs[(hand, finger, 0)] * (
                    second_value - zero[(hand, finger, 0)]
                )
                moved_spread = dict(spread)
                moved_spread[finger] = anatomical_spread[finger]
                model_after = model_tip(anatomy, moved_spread, bend, base, finger)

                vendor_index = base_position(
                    recording.frames[finger_open], hand, "link_%sSkeletonIndex1" % hand
                )
                vendor_little = base_position(
                    recording.frames[finger_open], hand, "link_%sSkeletonPinky1" % hand
                )
                vendor_towards = norm(sub(vendor_index, vendor_little))
                model_index = model_base[("index", 1)]
                model_little = model_base[("little", 1)]
                model_towards = norm(sub(model_index, model_little))
                vendor = dot(sub(vendor_after, vendor_before), vendor_towards)
                model_delta = dot(sub(model_after, model_before), model_towards)
                delta_angle = moved_spread[finger] - spread[finger]
                ratio = abs(model_delta / vendor) if abs(vendor) > 1e-9 else math.inf
                ok = (vendor * model_delta > 0.0) and 0.2 < ratio < 5.0
                spread_ok = spread_ok and ok
                print(
                    "  [%s] %-5s %-6s dspread %+.3f rad: vendor %+.4f m, model %+.4f m (ratio %.2f)"
                    % (
                        "PASS" if ok else "FAIL",
                        hand,
                        finger,
                        delta_angle,
                        vendor,
                        model_delta,
                        ratio,
                    )
                )

    # ---- check 4: thumb joints against the vendor thumb tip ------------------------------------
    def thumb_tip_progress(rec, hand, frame_index):
        base = base_position(
            rec.frames[frame_index], hand, "link_%sSkeletonThumbBase" % hand
        )
        tip = base_position(
            rec.frames[frame_index], hand, "%s_thumb_tip" % hand.lower()
        )
        index = base_position(
            rec.frames[frame_index], hand, "link_%sSkeletonIndex1" % hand
        )
        return dot(sub(tip, base), norm(sub(index, base)))

    print(
        "\ncheck 4: thumb joints vs the vendor <side>_thumb_tip (direction and monotonicity)"
    )
    thumb_ok_check = True
    checks = []
    if thumb_recording is recording:
        checks.append((args.recording, recording, rows, True))
    else:
        checks.append((str(args.thumb_recording), thumb_recording, thumb_rows, True))
        checks.append((args.recording, recording, rows, False))
    for label, rec, rec_rows, is_thumb_recording in checks:
        if rec is thumb_recording and rec is recording:
            label = args.thumb_recording or args.recording
        open_frame, fist_frame, _open_run, _fist_run = thumb_plateau_frames(
            rec, rec_rows, keys
        )
        indices = (
            list(range(open_frame, fist_frame + 1))
            if fist_frame >= open_frame
            else list(range(fist_frame, open_frame + 1))
        )
        base_offset = -base_delta if is_thumb_recording else 0.0
        print(
            "  %s (open %.1f s -> fist %.1f s)"
            % (label, rec.times[open_frame], rec.times[fist_frame])
        )
        for hand in ("Left", "Right"):
            progress = [thumb_tip_progress(rec, hand, index) for index in indices]
            progress_smooth = median_smooth(progress, 15)
            vendor_dir = progress[-1] - progress[0]
            vendor_ok = vendor_dir > 1e-6
            print(
                "    %-5s vendor palmward %+.4f m, %s"
                % (
                    hand,
                    vendor_dir,
                    "rising" if vendor_ok else "NOT RISING",
                )
            )
            thumb_ok_check = thumb_ok_check and vendor_ok
            slots = [("base", "base"), (0, "lift"), (1, "CMC"), (2, "MCP"), (3, "IP")]
            for slot, slot_label in slots:
                if slot in (0, "base") and not is_thumb_recording:
                    continue
                key = (hand, "thumb", slot)
                offset = base_offset if slot == "base" else 0.0
                series = [rec_rows[index][key] + offset for index in indices]
                series_smooth = median_smooth(series, 15)
                corr = pearson(series_smooth, progress_smooth)
                expected = signs[key]
                corr_ok = abs(corr) >= 0.9 and (corr > 0.0) == (expected > 0.0)
                direction_ok = expected * (series[-1] - series[0]) > 0.0
                inversions = 0
                for step in range(1, len(indices)):
                    if progress_smooth[step] - progress_smooth[step - 1] <= 1e-6:
                        continue
                    if expected * (
                        series_smooth[step] - series_smooth[step - 1]
                    ) < -math.radians(2.0):
                        inversions += 1
                mono_ok = inversions == 0
                # The three bends must also track the vendor tip tightly; the mount and lift are
                # not flexion joints, so for them direction and monotonicity are the requirement.
                ok = (
                    direction_ok
                    and mono_ok
                    and (corr_ok if slot not in (0, "base") else True)
                )
                thumb_ok_check = thumb_ok_check and ok
                print(
                    "      [%s] %-5s %-4s corr %+.3f (sign %+d), travel %+7.2f deg, inversions %d"
                    % (
                        "PASS" if ok else "FAIL",
                        hand,
                        slot_label,
                        corr,
                        int(expected),
                        math.degrees(series[-1] - series[0]),
                        inversions,
                    )
                )

    # ---- check 5: gain calibration at each recording's open and fist references ----------------
    print("\ncheck 5: per-joint gain at the open and fist references (degrees)")
    if args.identity_gain:
        print("  (identity gain: the old 1:1 mapping, expected to fail the fist check)")
    open_worst = 0.0
    fist_worst = -1.0
    fist_worst_where = ""
    for hand in ("Left", "Right"):
        for finger in FINGERS:
            labels = (
                ("CMC", "MCP", "IP") if finger == "thumb" else ("MCP", "PIP", "DIP")
            )
            if finger == "thumb":
                open_frame, fist_frame, source = (
                    thumb_open_frame,
                    thumb_fist_frame,
                    thumb_rows,
                )
            else:
                if not finger_exercised:
                    print(
                        "  %-5s %-6s not exercised in this recording"
                        % (hand.lower(), finger)
                    )
                    continue
                open_frame, fist_frame, source = finger_open, finger_fist, rows
            for slot, slot_label in zip((1, 2, 3), labels):
                open_deg = math.degrees(
                    calibrated_bend(
                        source[open_frame], hand, finger, slot, args.identity_gain
                    )
                )
                fist_deg = math.degrees(
                    calibrated_bend(
                        source[fist_frame], hand, finger, slot, args.identity_gain
                    )
                )
                target = math.degrees(HUMAN_MAX[finger][slot])
                open_worst = max(open_worst, abs(open_deg))
                if abs(fist_deg - target) > fist_worst:
                    fist_worst = abs(fist_deg - target)
                    fist_worst_where = "%s %s %s" % (hand.lower(), finger, slot_label)
                print(
                    "  %-5s %-6s %-3s open %+6.1f  fist %+6.1f  human_max %5.1f"
                    % (hand.lower(), finger, slot_label, open_deg, fist_deg, target)
                )
        # Spread and the thumb mount are not gained, but they still read zero at the open reference.
        for finger in FINGERS:
            if finger != "thumb" and not finger_exercised:
                continue
            open_worst = max(
                open_worst,
                abs(
                    math.degrees(
                        calibrated_spread(
                            thumb_rows[thumb_open_frame]
                            if finger == "thumb"
                            else rows[finger_open],
                            hand,
                            finger,
                        )
                    )
                ),
            )
        open_worst = max(
            open_worst,
            abs(math.degrees(calibrated_base(thumb_rows[thumb_open_frame], hand))),
        )
    open_ok = open_worst < 1.0
    fist_ok = 0.0 <= fist_worst < 10.0
    print(
        "  [%s] open: every calibrated joint within %.1f deg of 0 (limit 1 deg)"
        % ("PASS" if open_ok else "FAIL", open_worst)
    )
    print(
        "  [%s] fist: every bend within %.1f deg of its human_max (worst %s, limit 10 deg)"
        % (
            "PASS" if fist_ok else "FAIL",
            fist_worst,
            fist_worst_where if fist_worst_where else "n/a",
        )
    )

    # ---- result --------------------------------------------------------------------------------
    all_ok = (
        span_ok
        and level_ok
        and thumb_ok
        and fk_ok
        and travel_ok
        and spread_ok
        and thumb_ok_check
        and open_ok
        and fist_ok
    )
    print(
        "\n%s"
        % ("ALL DERIVATION CHECKS PASSED" if all_ok else "DERIVATION CHECKS FAILED")
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
