# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the render-only pages capture.mjs draws the CTS step GIFs from.

    python3 docs/tools/cts_media/build_pages.py --out "$TMP/cts_media"

The motion is the oracles' synthetic reference take, with sway, tremor and noise
switched off so the GIFs stay small; the steps, holds and order come from each
suite's capture/session.py. Hand steps other than the rolls are drawn here from the
oracle's poses and placements (HAND_START, HAND_ORBIT); the oracle itself is untouched.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CTS = HERE.parents[2] / "src/python/isaaccapture/cts"
sys.path[:0] = [str(CTS / "hand/oracle"), str(CTS / "full_body/oracle")]

import hand_model as hm  # noqa: E402
import posture_script  # noqa: E402
import script as hand_oracle  # noqa: E402
import skeleton as sk  # noqa: E402

HAND_FPS = 10
BODY_FPS = 12.5
HAND_LEAD_S = 1.2  # rolls: the oracle's move into the pose, before the beep
HAND_MOVE_S = 1.0  # other steps: the drawn move into the pose, before the beep
BODY_LEAD_S = 0.64  # still steps: the oracle's 0.55 s arm blend, before the press

HAND_VIEW = {
    "flat_on_table_open": "above",
    "fist": "thumb",
    "pinch_index": "palm",
    "pinch_middle": "palm",
    "pinch_ring": "palm",
    "pinch_little": "palm",
    "palms_together": "facing",
    "tips_together": "facing",
    "right_tip_roll": "facing",
    "left_tip_roll": "facing",
    "flat_on_table_close": "above",
}

# Non-roll hand steps: where the move into the pose starts. "open" closes from a relaxed
# open hand; a number starts each wrist that much further out along x; None has no move,
# so the step is one still frame unless the camera orbits.
HAND_START = {
    "flat_on_table_open": None,
    "fist": None,
    "pinch_index": "open",
    "pinch_middle": "open",
    "pinch_ring": "open",
    "pinch_little": "open",
    "palms_together": None,
    "tips_together": 0.06,
    "flat_on_table_close": None,
}
# Camera yaw added during the hold, over its fraction u; both end where they start, so
# the GIF loops without a jump.
HAND_ORBIT = {
    "fist": lambda u: 2 * math.pi * hand_oracle._smooth(u),
    "palms_together": lambda u: math.radians(70) * math.sin(2 * math.pi * u),
}

# label -> (view, limbs drawn thick)
BODY_VIEW = {
    "a_pose_still": ("front", []),
    "t_pose_hold_open": ("front", ["LEFT_ARM", "RIGHT_ARM"]),
    "neutral_stance": ("front", []),
    "left_arm_raise": ("front", ["LEFT_ARM"]),
    "right_arm_raise": ("front", ["RIGHT_ARM"]),
    "left_leg_raise": ("three_quarter", ["LEFT_LEG"]),
    "right_leg_raise": ("three_quarter", ["RIGHT_LEG"]),
    "squat_x2": ("three_quarter", ["LEFT_LEG", "RIGHT_LEG"]),
    "march_in_place": ("three_quarter", ["LEFT_LEG", "RIGHT_LEG"]),
    "t_pose_hold_close": ("front", ["LEFT_ARM", "RIGHT_ARM"]),
}
LIMBS = {
    "LEFT_ARM": [
        "LEFT_COLLAR",
        "LEFT_SHOULDER",
        "LEFT_ELBOW",
        "LEFT_WRIST",
        "LEFT_HAND",
    ],
    "RIGHT_ARM": [
        "RIGHT_COLLAR",
        "RIGHT_SHOULDER",
        "RIGHT_ELBOW",
        "RIGHT_WRIST",
        "RIGHT_HAND",
    ],
    "LEFT_LEG": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT"],
    "RIGHT_LEG": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT"],
}

# Palms together: each thumb lies straight in its own hand's finger plane, 10 degrees
# off the index (right-hand wrist frame, metres). The oracle's thumb base sits 12 mm
# palmar, past the seam, so both thumbs would cross the other hand.
PALMS_THUMB_DEG, PALMS_THUMB_Y = 10.0, 0.001
# Flat on the table, fingers together: tips converge ~15 mm apart (hm.SPLAY fans them),
# thumb straight along the index, 15 degrees off it, tip just above the fingers' plane.
TOGETHER_SPLAY = {"index": -0.04, "middle": 0.0, "ring": 0.04, "little": 0.08}
FLAT_THUMB_DEG, FLAT_THUMB_Y = 15.0, 0.006
THUMB_REACH = 0.99  # of full length: almost straight


@dataclass(frozen=True)
class Together(hm.Pose):
    """A pose drawn with TOGETHER_SPLAY; survives the oracle's dataclasses.replace()."""


@dataclass(frozen=True)
class Pressed(hm.Pose):
    """Palms together: the thumb chain is solved in the plane y = PALMS_THUMB_Y."""


class Quiet(random.Random):
    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return mu


def session_steps(suite: str) -> list[tuple[str, float, str]]:
    """STEPS from <suite>/capture/session.py, read without importing it."""
    tree = ast.parse((CTS / suite / "capture/session.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "STEPS"
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no STEPS in {suite}/capture/session.py")


def straight_thumb(deg: float, y: float, base_y: float) -> tuple[float, float, float]:
    """Tip of an almost straight thumb ``deg`` off the index, from a base at ``base_y``."""
    base = hm.THUMB_BASE
    reach = sum(hm.THUMB_BONES) * THUMB_REACH
    across = (reach**2 - (y - base_y) ** 2) ** 0.5
    a = math.radians(deg)
    return (base[0] - across * math.sin(a), y, base[2] - across * math.cos(a))


def together(pose: hm.Pose) -> float:
    """How far ``pose`` is toward Together, following the oracle's blends."""
    if isinstance(pose, Together):
        return 1.0
    target = pose.thumb_target
    if target is not None and isinstance(target[0], hm.Pose):
        a, b, t = target
        return together(a) + (together(b) - together(a)) * t
    return 0.0


def mm(v):
    return [round(c * 1000, 1) for c in v]


def centroid(points):
    return [sum(p[i] for p in points) / len(points) for i in range(3)]


def hand_data() -> dict:
    steps_in = session_steps("hand")
    if [label for label, _ in hand_oracle.STEPS] != [label for label, _, _ in steps_in]:
        raise SystemExit("hand oracle and session.py disagree on the script")
    for label, hold, _ in steps_in:
        if dict(hand_oracle.STEPS)[label] != hold:
            raise SystemExit(f"hand oracle holds {label} for a different time")

    oracle_pose_for = hand_oracle.pose_for

    def pose_for(label, options):
        pose = oracle_pose_for(label, options)
        if label == "palms_together":
            tip = straight_thumb(PALMS_THUMB_DEG, PALMS_THUMB_Y, PALMS_THUMB_Y)
            return Pressed(pose.flex, thumb_target=tip)
        if label in hand_oracle.FLAT:
            tip = straight_thumb(FLAT_THUMB_DEG, FLAT_THUMB_Y, hm.THUMB_BASE[1])
            return Together(pose.flex, thumb_target=tip)
        return pose

    oracle_solve = hm.solve

    def solve(pose):
        w = together(pose)
        splay = dict(hm.SPLAY)
        hm.SPLAY.update({f: s + (TOGETHER_SPLAY[f] - s) * w for f, s in splay.items()})
        base, open_dir = hm.THUMB_BASE, hm.THUMB_OPEN_DIR
        if isinstance(pose, Pressed):
            hm.THUMB_BASE = (base[0], PALMS_THUMB_Y, base[2])
            hm.THUMB_OPEN_DIR = (open_dir[0], 0.0, open_dir[2])
        try:
            return oracle_solve(pose)
        finally:
            hm.SPLAY.update(splay)
            hm.THUMB_BASE, hm.THUMB_OPEN_DIR = base, open_dir

    hand_oracle.pose_for = pose_for
    hm.solve = solve
    hand_oracle._sway = lambda t: (hand_oracle.IDENTITY, (0.0, 0.0, 0.0))
    hand_oracle._tremor = lambda pose, t, phase: pose
    hand_oracle._noise = lambda rng: hand_oracle.IDENTITY
    hand_oracle.random = SimpleNamespace(Random=Quiet)

    ticks, windows = hand_oracle.synthesise(hand_oracle.Options())
    period_ns = 1e9 / hand_oracle.RATE_HZ
    tip = hm.FINGER_JOINTS["middle"][-1]

    def oracle_frames(w, hold):
        """The oracle's take around window ``w``, as (left, right) joint positions."""
        lead = round(HAND_LEAD_S * HAND_FPS)
        count = lead + round(hold * HAND_FPS) + 1  # the last frame is the tick
        frames = []
        for k in range(count):
            t_ns = w.start_ns + (k - lead) * 1e9 / HAND_FPS
            i = round((t_ns - hand_oracle.T0_NS) / period_ns)
            _, left, right = ticks[min(i, len(ticks) - 1)]
            frames.append(tuple([j.position for j in h.joints] for h in (left, right)))
        return frames, lead

    def placed(pose, place):
        """Both hands for a right-hand pose and wrist placement, the left mirrored."""
        local = hm.solve(pose)
        mirrored = hand_oracle._mirror_pose(*place)
        return tuple(
            [hm.add(p, hm.quat_rotate(q, lp)) for lp, _ in joints]
            for joints, (p, q) in ((hm.mirror(local), mirrored), (local, place))
        )

    def drawn_frames(label, hold):
        target = hand_oracle.pose_for(label, hand_oracle.Options())
        place = hand_oracle.placement_for(label)
        held = placed(target, place)
        start = HAND_START[label]
        count = round(hold * HAND_FPS) + 1
        if start is None:
            return [held] * (count if label in HAND_ORBIT else 1), 0
        lead = round(HAND_MOVE_S * HAND_FPS)
        frames = []
        for k in range(lead):
            s = hand_oracle._smooth(k / lead)
            if start == "open":
                frames.append(placed(hm.blend(hm.OPEN, target, s), place))
            else:
                shift = (start * (1 - s), 0.0, 0.0)
                frames.append(placed(target, (hm.add(place[0], shift), place[1])))
        return frames + [held] * count, lead

    def offsets(frames, targets):
        return [
            hm.sub(p, targets[h])
            for pair in frames
            for h, hand in enumerate(pair)
            for p in hand
        ]

    steps = []
    for (label, hold, _), w in zip(steps_in, windows):
        if label in hand_oracle.ROLLS:
            frames, lead = oracle_frames(w, hold)
            fit = frames[lead:]  # the move in may leave the frame
        else:
            frames, lead = drawn_frames(label, hold)
            fit = frames
        held = frames[lead:]
        left_pts = [p for left, _ in held for p in left]
        right_pts = [p for _, right in held for p in right]
        apart = hm.length(hm.sub(held[0][0][tip], held[0][1][tip]))
        joint = apart < 0.12  # hands in contact share one camera
        both = mm(centroid(left_pts + right_pts))
        targets = (
            [both, both] if joint else [mm(centroid(left_pts)), mm(centroid(right_pts))]
        )
        targets_m = [tuple(c / 1000 for c in t) for t in targets]
        # Half-extents the pose must fit.
        fitted = offsets(fit, targets_m)
        extent_x = max(0.09, *(abs(d[0]) for d in fitted))
        extent_y = max(0.09, *(abs(d[1]) + 0.3 * abs(d[2]) for d in fitted))
        orbit = None
        if label in HAND_ORBIT:
            n = len(held) - 1
            orbit = [0.0] * lead + [
                round(HAND_ORBIT[label](k / n), 4) for k in range(n + 1)
            ]
            # The turning camera may see the held pose from any side.
            turning = offsets(held, targets_m)
            radial = max(math.hypot(d[0], d[2]) for d in turning)
            extent_x = max(extent_x, radial)
            extent_y = max(extent_y, *(abs(d[1]) + 0.3 * radial for d in turning))
        steps.append(
            {
                "label": label,
                "hold": hold,
                "view": HAND_VIEW[label],
                "joint": joint,
                "targets": targets,
                "extent": [round(extent_x * 1000), round(extent_y * 1000)],
                "frames": [[[mm(p) for p in hand] for hand in pair] for pair in frames],
                "hold_from": lead,
                "still": len(frames) == 1,
                "orbit": orbit,
            }
        )
    chains = {
        "thumb": list(hm.THUMB),
        **{f: list(c) for f, c in hm.FINGER_JOINTS.items()},
    }
    return {
        "steps": steps,
        "chains": chains,
        "wrist": hm.WRIST,
        "fps": HAND_FPS,
        "table_mm": hand_oracle.TABLE_Y * 1000,
    }


def body_data() -> dict:
    steps_in = session_steps("full_body")
    windows = {w["label"]: w for w in posture_script.step_windows()}
    absent = [label for label, _, _ in steps_in if label not in windows]
    if absent:
        raise SystemExit(f"the full-body oracle does not perform: {absent}")
    posture_script.random = SimpleNamespace(Random=Quiet)
    rec = posture_script.build_posture()
    rate = posture_script.RATE_HZ

    def joints_at(t: float):
        """Joint positions (mm) at oracle time t, linear between samples."""
        x = min(t * rate, len(rec.frames) - 1.0)
        i = int(x)
        a, f = rec.frames[i].joints, x - i
        b = rec.frames[min(i + 1, len(rec.frames) - 1)].joints
        return [
            [round((pa + (pb - pa) * f) * 1000) for pa, pb in zip(ja.pos, jb.pos)]
            for ja, jb in zip(a, b)
        ]

    steps = []
    for label, duration, _ in steps_in:
        w = windows[label]
        still = label in posture_script.STILL_STEPS
        # The oracle's windows are shorter than the real ones: the hold is resampled
        # to play for the session's duration. Still steps open on the oracle's blend
        # into the pose, which a performer does before pressing.
        lead_o = BODY_LEAD_S if still else 0.0
        span_o = w["end_s"] - 1 / rate - (w["start_s"] + lead_o)
        lead = round(lead_o * BODY_FPS)
        times = [w["start_s"] + k / BODY_FPS for k in range(lead)]
        count = round(duration * BODY_FPS) + 1
        times += [
            w["start_s"] + lead_o + span_o * k / (count - 1) for k in range(count)
        ]
        view, limbs = BODY_VIEW[label]
        steps.append(
            {
                "label": label,
                "duration": duration,
                "still": still,
                "view": view,
                "active": sorted(
                    {sk.J[name] for limb in limbs for name in LIMBS[limb]}
                ),
                "frames": [joints_at(t) for t in times],
                "lead": lead,
            }
        )
    side = [
        "L" if n.startswith("LEFT") else "R" if n.startswith("RIGHT") else "C"
        for n in sk.JOINT_NAMES
    ]
    return {
        "steps": steps,
        "parent": sk.PARENT,
        "side": side,
        "fps": BODY_FPS,
        "hands": [sk.J["LEFT_HAND"], sk.J["RIGHT_HAND"]],
        "head": sk.J["HEAD"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("hand", hand_data()), ("full_body", body_data())):
        html = (HERE / f"{name}.html").read_text()
        assert "/*DATA*/null" in html
        html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
        (args.out / f"{name}.html").write_text(html)
        print(
            "wrote",
            args.out / f"{name}.html",
            "steps:",
            [len(s["frames"]) for s in data["steps"]],
        )


if __name__ == "__main__":
    main()
