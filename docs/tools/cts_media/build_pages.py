# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the render-only pages capture.mjs draws the CTS step GIFs from.

    python3 docs/tools/cts_media/build_pages.py --out "$TMP/cts_media"

The motion is the oracles' synthetic reference take, with sway, tremor and noise
switched off so the GIFs stay small; the steps, holds and order come from each
suite's capture/session.py. The hand's flat pose is redrawn with the fingers together,
as the cue asks; the oracle itself is untouched.
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
HAND_LEAD_S = 1.2  # the move into the pose, before the beep
BODY_LEAD_S = 0.64  # still steps: the oracle's 0.55 s arm blend, before the press

HAND_VIEW = {
    "flat_on_table_open": "above",
    "fist": "thumb",
    "pinch_index": "palm",
    "pinch_middle": "palm",
    "pinch_ring": "palm",
    "pinch_little": "palm",
    "palms_together": "palms",
    "tips_together": "facing",
    "right_tip_roll": "facing",
    "left_tip_roll": "facing",
    "flat_on_table_close": "above",
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

# Thumbs closed alongside the index (right-hand wrist frame, metres); the oracle's open
# thumb splays away from it.
PALMS_THUMB = (-0.040, -0.002, -0.118)
# Flat on the table, fingers together: tips converge ~15 mm apart (hm.SPLAY fans them),
# thumb straight along the index, 15 degrees off it, tip just above the fingers' plane.
TOGETHER_SPLAY = {"index": -0.04, "middle": 0.0, "ring": 0.04, "little": 0.08}
FLAT_THUMB_DEG, FLAT_THUMB_REACH, FLAT_THUMB_Y = 15.0, 0.99, 0.006


@dataclass(frozen=True)
class Together(hm.Pose):
    """A pose drawn with TOGETHER_SPLAY; survives the oracle's dataclasses.replace()."""


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


def flat_thumb() -> tuple[float, float, float]:
    """Thumb tip for the flat pose: almost straight, beside the index, above the table."""
    base = hm.THUMB_BASE
    reach = sum(hm.THUMB_BONES) * FLAT_THUMB_REACH
    across = (reach**2 - (FLAT_THUMB_Y - base[1]) ** 2) ** 0.5
    a = math.radians(FLAT_THUMB_DEG)
    return (
        base[0] - across * math.sin(a),
        FLAT_THUMB_Y,
        base[2] - across * math.cos(a),
    )


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
            return hm.Pose(pose.flex, thumb_target=PALMS_THUMB)
        if label in hand_oracle.FLAT:
            return Together(pose.flex, thumb_target=flat_thumb())
        return pose

    oracle_solve = hm.solve

    def solve(pose):
        w = together(pose)
        splay = dict(hm.SPLAY)
        hm.SPLAY.update({f: s + (TOGETHER_SPLAY[f] - s) * w for f, s in splay.items()})
        try:
            return oracle_solve(pose)
        finally:
            hm.SPLAY.update(splay)

    hand_oracle.pose_for = pose_for
    hm.solve = solve
    hand_oracle._sway = lambda t: (hand_oracle.IDENTITY, (0.0, 0.0, 0.0))
    hand_oracle._tremor = lambda pose, t, phase: pose
    hand_oracle._noise = lambda rng: hand_oracle.IDENTITY
    hand_oracle.random = SimpleNamespace(Random=Quiet)

    ticks, windows = hand_oracle.synthesise(hand_oracle.Options())
    period_ns = 1e9 / hand_oracle.RATE_HZ
    tip = hm.FINGER_JOINTS["middle"][-1]
    steps = []
    for (label, hold, _), w in zip(steps_in, windows):
        lead = round(HAND_LEAD_S * HAND_FPS)
        count = lead + round(hold * HAND_FPS) + 1  # the last frame is the tick
        picked = []
        for k in range(count):
            t_ns = w.start_ns + (k - lead) * 1e9 / HAND_FPS
            i = round((t_ns - hand_oracle.T0_NS) / period_ns)
            picked.append(ticks[min(i, len(ticks) - 1)])
        frames = [
            [[mm(j.position) for j in h.joints] for h in (left, right)]
            for _, left, right in picked
        ]
        held = [(left, right) for _, left, right in picked[lead:]]
        left_pts = [j.position for left, _ in held for j in left.joints]
        right_pts = [j.position for _, right in held for j in right.joints]
        apart = hm.length(
            hm.sub(held[0][0].joints[tip].position, held[0][1].joints[tip].position)
        )
        joint = apart < 0.12  # hands in contact share one camera
        both = centroid(left_pts + right_pts)
        targets = (
            [mm(both), mm(both)]
            if joint
            else [mm(centroid(left_pts)), mm(centroid(right_pts))]
        )
        # Half-extents the held pose must fit; the move in may leave the frame.
        offsets = [
            hm.sub(j.position, tuple(c / 1000 for c in targets[h]))
            for pair in held
            for h, hand in enumerate(pair)
            for j in hand.joints
        ]
        extent_x = max(0.09, *(abs(d[0]) for d in offsets))
        extent_y = max(0.09, *(abs(d[1]) + 0.3 * abs(d[2]) for d in offsets))
        steps.append(
            {
                "label": label,
                "hold": hold,
                "view": HAND_VIEW[label],
                "joint": joint,
                "targets": targets,
                "extent": [round(extent_x * 1000), round(extent_y * 1000)],
                "frames": frames,
                "hold_from": lead,
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
