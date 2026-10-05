# SPDX-FileCopyrightText: Copyright (c) 2026 HTC CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize Hand Joints with MuJoCo
=================================

Subscribes to IsaacTeleop's ``HandTracker`` (+ ``HeadTracker`` for context) and
renders both hands live in the MuJoCo passive viewer as 26-joint skeletons:
each joint is a small sphere, and OpenXR's standard bone topology is drawn as
capsule "bones" between parents and children.

Coordinate frame
----------------
IsaacTeleop delivers poses in the OpenXR convention (right-handed, Y-up,
-Z forward); MuJoCo's default is right-handed, Z-up. Every joint position is
rotated by +90 deg about X before being handed to MuJoCo. The head mesh
follows the same rule as ``visualize_poses_mujoco_example.py``.

Usage
-----
    python visualize_hands_mujoco_example.py
    python visualize_hands_mujoco_example.py --no-head --debug
"""

import argparse
import os
import signal
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
from isaaccapture.cloudxr import CloudXRLauncher


# Route MuJoCo warnings to stderr instead of writing MUJOCO_LOG.TXT in CWD.
def _mujoco_warning(msg) -> None:
    text = msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else msg
    print(f"[MuJoCo] {text}", file=sys.stderr)


mujoco.set_mju_user_warning(_mujoco_warning)


# OpenXR -> MuJoCo world rotation (Rx +90 deg): (x, y, z) -> (x, -z, y).
R_XR_TO_MJ = R.from_euler("x", 90, degrees=True)

# Head mesh + texture (shared with the poses example) so the hands have a
# body anchor. If either file is missing the head degrades gracefully:
# mesh-only -> rgba fallback; mesh missing -> box fallback.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "vive_assets")
HEAD_MESH = "generic_hmd.obj"
HEAD_TEXTURE = "generic_hmd_color.png"
HEAD_FIX_EULER_DEG = (0.0, 0.0, 180.0)
HEAD_FALLBACK_RGBA = "0.4 0.4 0.4 1"  # used if HEAD_TEXTURE is missing
HEAD_BOX_HALF_SIZE = "0.10 0.10 0.16"  # used if HEAD_MESH is missing

# OpenXR XR_EXT_hand_tracking joint indices (also exposed as
# ``deviceio.HandJoint.*``). Listed here so the bone table is readable.
WRIST = 1
THUMB = (2, 3, 4, 5)  # metacarpal, proximal, distal, tip
INDEX = (6, 7, 8, 9, 10)  # metacarpal, proximal, intermediate, distal, tip
MIDDLE = (11, 12, 13, 14, 15)
RING = (16, 17, 18, 19, 20)
LITTLE = (21, 22, 23, 24, 25)


def _finger_chain(root: int, joints: tuple) -> list:
    """[(root,j0), (j0,j1), (j1,j2), ...] — adjacent-pair bone segments."""
    chain = [(root, joints[0])]
    for a, b in zip(joints[:-1], joints[1:]):
        chain.append((a, b))
    return chain


# Parent -> child bone list (24 segments per hand).
HAND_BONES = (
    _finger_chain(WRIST, THUMB)
    + _finger_chain(WRIST, INDEX)
    + _finger_chain(WRIST, MIDDLE)
    + _finger_chain(WRIST, RING)
    + _finger_chain(WRIST, LITTLE)
)

# Render colours: left = cyan, right = warm orange.
HAND_COLOURS = {
    "left": {"joint": (0.30, 0.85, 1.00, 1.0), "bone": (0.20, 0.55, 0.95, 1.0)},
    "right": {"joint": (1.00, 0.65, 0.30, 1.0), "bone": (0.95, 0.40, 0.20, 1.0)},
}

JOINT_RADIUS = 0.008  # m
BONE_RADIUS = 0.004  # m


# ----------------------------------------------------------------------------
# Pose helpers
# ----------------------------------------------------------------------------
def _xr_pos_to_mj(p) -> np.ndarray:
    return R_XR_TO_MJ.apply([p.x, p.y, p.z])


def _xr_pose_to_mj(position, orientation):
    pos = _xr_pos_to_mj(position)
    q = (
        R_XR_TO_MJ
        * R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
    ).as_quat()
    return pos, np.array([q[3], q[0], q[1], q[2]])  # MuJoCo wants wxyz


def _euler_to_wxyz(euler_deg) -> str:
    q = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return f"{q[3]} {q[0]} {q[1]} {q[2]}"


# ----------------------------------------------------------------------------
# MJCF: floor + optional head mocap body. Hand joints/bones are drawn at
# runtime via viewer.user_scn so we don't need 52 mocap bodies in the model.
# ----------------------------------------------------------------------------
def _build_mjcf(show_head: bool) -> str:
    head_mesh_path = os.path.join(ASSETS_DIR, HEAD_MESH)
    head_tex_path = os.path.join(ASSETS_DIR, HEAD_TEXTURE)
    quat_attr = (
        f'quat="{_euler_to_wxyz(HEAD_FIX_EULER_DEG)}" contype="0" conaffinity="0"'
    )

    if show_head and os.path.exists(head_mesh_path) and os.path.exists(head_tex_path):
        head_asset = (
            f'<mesh name="head_mesh" file="{head_mesh_path}"/>\n    '
            f'<texture name="head_tex" type="2d" file="{head_tex_path}"/>\n    '
            f'<material name="head_mat" texture="head_tex" specular="0.1" shininess="0.3"/>'
        )
        head_geom = (
            f'<geom type="mesh" mesh="head_mesh" material="head_mat" {quat_attr}/>'
        )
    elif show_head and os.path.exists(head_mesh_path):
        head_asset = f'<mesh name="head_mesh" file="{head_mesh_path}"/>'
        head_geom = f'<geom type="mesh" mesh="head_mesh" rgba="{HEAD_FALLBACK_RGBA}" {quat_attr}/>'
    elif show_head:
        head_asset = ""
        head_geom = f'<geom type="box" size="{HEAD_BOX_HALF_SIZE}" rgba="{HEAD_FALLBACK_RGBA}" {quat_attr}/>'
    else:
        head_asset = ""
        head_geom = ""

    head_body = (
        f'<body name="headset" mocap="true" pos="0 0 1.5">{head_geom}</body>'
        if show_head
        else ""
    )

    return f"""<?xml version="1.0"?>
<mujoco model="visualize_hands">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse=".7 .7 .7" ambient=".5 .5 .5" specular=".1 .1 .1"/>
    <global offwidth="1280" offheight="720" azimuth="135" elevation="-25"/>
  </visual>
  <statistic center="0 0 1.3" extent="1.0"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .55 .75" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .2 .25" rgb2=".15 .15 .2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance=".05"/>
    {head_asset}
  </asset>

  <worldbody>
    <light pos="2 0 3" dir="-0.4 0 -1" diffuse=".8 .8 .8"/>
    <light pos="-2 0 3" dir="0.4 0 -1" diffuse=".4 .4 .4"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid"/>

    <geom type="cylinder" fromto="0 0 0  0.2 0 0" size="0.005" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0 0  0 0.2 0" size="0.005" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0 0  0 0 0.2" size="0.005" rgba="0 0 1 1" contype="0" conaffinity="0"/>

    {head_body}
  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------------
# user_scn helpers — append-only renderable geoms refilled each frame.
# ----------------------------------------------------------------------------
_IDENT3 = np.eye(3).flatten()


def _add_sphere(scn, pos, radius, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=_IDENT3,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_capsule_between(scn, p0, p1, radius, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=_IDENT3,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _draw_hand(scn, hand_data, colours) -> int:
    """Append spheres + capsule bones for one hand. Returns # of valid joints."""
    if hand_data is None:
        return 0

    n = deviceio.NUM_JOINTS
    positions = [None] * n
    valid = 0
    for i in range(n):
        jp = hand_data.joints.poses(i)
        if not jp.is_valid:
            continue
        valid += 1
        positions[i] = _xr_pos_to_mj(jp.pose.position)
        _add_sphere(scn, positions[i], JOINT_RADIUS, colours["joint"])

    for a, b in HAND_BONES:
        if positions[a] is None or positions[b] is None:
            continue
        _add_capsule_between(
            scn, positions[a], positions[b], BONE_RADIUS, colours["bone"]
        )

    return valid


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def _mocap_id(model, body_name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    mid = int(model.body_mocapid[body_id])
    return mid if mid >= 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--no-head",
        action="store_true",
        help="Skip the head mesh — only draw the hands.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print per-hand valid-joint counts once per second.",
    )
    CloudXRLauncher.add_launcher_arguments(ap)
    args = ap.parse_args()

    show_head = not args.no_head
    model = mujoco.MjModel.from_xml_string(_build_mjcf(show_head))
    data = mujoco.MjData(model)
    head_mid = _mocap_id(model, "headset") if show_head else None

    head_tracker = deviceio.HeadTracker() if show_head else None
    hand_tracker = deviceio.HandTracker()
    trackers = [t for t in (head_tracker, hand_tracker) if t is not None]
    extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)

    print("Visualize Hands (MuJoCo)")
    print(f"Trackers:   {[t.get_name() for t in trackers]}")
    print(f"Extensions: {extensions}")
    print("Close the viewer window or Ctrl+C to quit.")

    with CloudXRLauncher.launch_context(args):
        with (
            oxr.OpenXRSession("VisualizeHandsMuJoCo", extensions) as oxr_session,
            deviceio.DeviceIOSession.run(
                trackers, oxr_session.get_handles()
            ) as session,
            mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            ) as viewer,
        ):
            # Restore Python's default SIGINT handler -- MuJoCo's viewer / GLFW main
            # loop installs its own that can swallow Ctrl+C, leaving the user unable
            # to exit the script. This makes Ctrl+C raise KeyboardInterrupt again.
            signal.signal(signal.SIGINT, signal.default_int_handler)

            last_debug = 0.0
            try:
                while viewer.is_running():
                    session.update()

                    if head_tracker is not None:
                        head = head_tracker.get_head(session)
                        if head is not None and head.is_valid and head_mid is not None:
                            pos, quat = _xr_pose_to_mj(
                                head.pose.position, head.pose.orientation
                            )
                            data.mocap_pos[head_mid] = pos
                            data.mocap_quat[head_mid] = quat
                            viewer.cam.lookat[:] = pos

                    viewer.user_scn.ngeom = 0
                    left = hand_tracker.get_left_hand(session)
                    right = hand_tracker.get_right_hand(session)
                    n_l = _draw_hand(viewer.user_scn, left, HAND_COLOURS["left"])
                    n_r = _draw_hand(viewer.user_scn, right, HAND_COLOURS["right"])

                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    if args.debug and time.time() - last_debug >= 1.0:
                        last_debug = time.time()
                        print(
                            f"hands: L {n_l:2d}/{deviceio.NUM_JOINTS}  "
                            f"R {n_r:2d}/{deviceio.NUM_JOINTS} valid joints"
                        )

                    time.sleep(1 / 60)
            except KeyboardInterrupt:
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
