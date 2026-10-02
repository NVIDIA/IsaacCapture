# SPDX-FileCopyrightText: Copyright (c) 2026 HTC CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize Headset + Controller Poses with MuJoCo
================================================

Subscribes to IsaacTeleop's ``HeadTracker`` and ``ControllerTracker`` and drives
three mocap bodies in a generated MJCF scene, rendered live in the MuJoCo
passive viewer. Closes when the viewer window is closed or Ctrl+C.

The bundled meshes (in ``vive_assets/``) are a generic HMD and a pair of Vive
Focus 3 controllers, each with its matching texture. If any mesh / texture is
missing, the corresponding device falls back to a coloured primitive box, so
the sample still runs without the binary assets.

Notes for adaptation
--------------------
* **Coordinate frames.** IsaacTeleop delivers poses in the OpenXR convention
  (right-handed, Y-up, -Z forward); MuJoCo's default is right-handed, Z-up. We
  rotate every incoming pose by +90 deg about X and reorder the quaternion
  from ``xyzw`` (OpenXR / scipy) to ``wxyz`` (MuJoCo).
* **Mesh-frame fix.** OBJ files are not modelled in the OpenXR pose frame;
  ``DEVICES[*]["fix_euler_deg"]`` below is a per-device static rotation
  (applied as ``<geom quat="..."/>``) that aligns each mesh with its tracked
  pose. **Tune these for your own meshes** -- the values shipped here were set
  visually for the bundled generic HMD and Vive Focus 3 controllers.
* **Camera tracking.** The free camera's ``lookat`` is rewritten to the head
  position every frame, so mouse-wheel zoom always converges on the headset.

Usage
-----
    python visualize_poses_mujoco_example.py
    python visualize_poses_mujoco_example.py --pose aim --debug
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


# ----------------------------------------------------------------------------
# Asset locations
# ----------------------------------------------------------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "vive_assets")


# ----------------------------------------------------------------------------
# Per-device configuration. Tune ``fix_euler_deg`` if a mesh looks rotated
# relative to its real motion; tune ``box_half_size`` to match the silhouette
# of your fallback geometry.
# ----------------------------------------------------------------------------
DEVICES = {
    "headset": {
        "mesh": "generic_hmd.obj",
        "texture": "generic_hmd_color.png",
        "rgba": "0.4 0.4 0.4 1",  # only used if texture missing
        "fix_euler_deg": (0.0, 0.0, 180.0),
        "init_pos": "0 0 1.5",
        "box_half_size": "0.10 0.10 0.16",
    },
    "controller_left": {
        "mesh": "vive_focus3_controller_left.obj",
        "texture": "vive_focus3_controller_left_color.png",
        "rgba": "0.3 0.5 1.0 1",  # box fallback colour
        "fix_euler_deg": (0.0, 90.0, 0.0),
        "init_pos": "-0.25 0 1.2",
        "box_half_size": "0.08 0.06 0.05",
    },
    "controller_right": {
        "mesh": "vive_focus3_controller_right.obj",
        "texture": "vive_focus3_controller_right_color.png",
        "rgba": "1.0 0.4 0.3 1",
        "fix_euler_deg": (0.0, 90.0, 0.0),
        "init_pos": "0.25 0 1.2",
        "box_half_size": "0.08 0.06 0.05",
    },
}

# OpenXR -> MuJoCo world rotation (Rx +90 deg): (x, y, z) -> (x, -z, y).
R_XR_TO_MJ = R.from_euler("x", 90, degrees=True)


# ----------------------------------------------------------------------------
# Pose / quaternion helpers
# ----------------------------------------------------------------------------
def _euler_to_wxyz(euler_deg) -> str:
    """scipy returns ``xyzw``; MJCF ``quat=`` is ``wxyz``."""
    q = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return f"{q[3]} {q[0]} {q[1]} {q[2]}"


def _convert_pose(position, orientation):
    """OpenXR pose -> MJ-frame ``(pos[3], quat_wxyz[4])``."""
    p = R_XR_TO_MJ.apply([position.x, position.y, position.z])
    q = (
        R_XR_TO_MJ
        * R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
    ).as_quat()
    return p, np.array([q[3], q[0], q[1], q[2]])


# ----------------------------------------------------------------------------
# MJCF generation -- writes one of {textured mesh, coloured mesh, box} per
# device depending on which assets are present on disk.
# ----------------------------------------------------------------------------
def _device_geom_xml(name: str, dev: dict, assets_dir: str) -> str:
    quat = _euler_to_wxyz(dev["fix_euler_deg"])
    mesh_path = os.path.join(assets_dir, dev["mesh"]) if dev["mesh"] else ""
    tex_path = os.path.join(assets_dir, dev["texture"]) if dev["texture"] else ""
    common = f'quat="{quat}" contype="0" conaffinity="0"'

    if mesh_path and os.path.exists(mesh_path):
        if tex_path and os.path.exists(tex_path):
            return (
                f'<geom type="mesh" mesh="{name}_mesh" material="{name}_mat" {common}/>'
            )
        return f'<geom type="mesh" mesh="{name}_mesh" rgba="{dev["rgba"]}" {common}/>'
    return f'<geom type="box" size="{dev["box_half_size"]}" rgba="{dev["rgba"]}" {common}/>'


def _device_asset_xml(name: str, dev: dict, assets_dir: str) -> str:
    mesh_path = os.path.join(assets_dir, dev["mesh"]) if dev["mesh"] else ""
    if not (mesh_path and os.path.exists(mesh_path)):
        return ""
    tex_path = os.path.join(assets_dir, dev["texture"]) if dev["texture"] else ""
    blocks = [f'<mesh name="{name}_mesh" file="{mesh_path}"/>']
    if tex_path and os.path.exists(tex_path):
        blocks.append(f'<texture name="{name}_tex" type="2d" file="{tex_path}"/>')
        blocks.append(
            f'<material name="{name}_mat" texture="{name}_tex" specular="0.1" shininess="0.3"/>'
        )
    return "\n    ".join(blocks)


def _build_mjcf(assets_dir: str) -> str:
    device_assets = "\n    ".join(
        filter(
            None,
            (_device_asset_xml(name, dev, assets_dir) for name, dev in DEVICES.items()),
        )
    )

    bodies = "\n".join(
        f'    <body name="{name}" mocap="true" pos="{dev["init_pos"]}">\n'
        f"      {_device_geom_xml(name, dev, assets_dir)}\n"
        f"    </body>"
        for name, dev in DEVICES.items()
    )

    return f"""<?xml version="1.0"?>
<mujoco model="visualize_poses">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse=".7 .7 .7" ambient=".5 .5 .5" specular=".1 .1 .1"/>
    <global offwidth="1280" offheight="720" azimuth="135" elevation="-25"/>
  </visual>
  <statistic center="-0.5 0.3 1.5" extent="1.2"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .55 .75" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .2 .25" rgb2=".15 .15 .2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance=".05"/>
    {device_assets}
  </asset>

  <worldbody>
    <light pos="2 0 3" dir="-0.4 0 -1" diffuse=".8 .8 .8"/>
    <light pos="-2 0 3" dir="0.4 0 -1" diffuse=".4 .4 .4"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid"/>

    <!-- World axis triad at the origin: X red, Y green, Z blue. -->
    <geom type="cylinder" fromto="0 0 0  0.2 0 0" size="0.005" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0 0  0 0.2 0" size="0.005" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0 0  0 0 0.2" size="0.005" rgba="0 0 1 1" contype="0" conaffinity="0"/>

{bodies}
  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def _mocap_id(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mid = int(model.body_mocapid[body_id])
    if mid < 0:
        raise RuntimeError(f"body '{body_name}' is not a mocap body")
    return mid


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--assets-dir",
        default=ASSETS_DIR,
        help="Directory holding the device meshes and textures "
        "(default: ./vive_assets).",
    )
    ap.add_argument(
        "--pose",
        choices=["grip", "aim"],
        default="grip",
        help="Which controller pose drives the mesh (default: grip).",
    )
    ap.add_argument(
        "--debug", action="store_true", help="Print mocap_pos values once per second."
    )
    CloudXRLauncher.add_launcher_arguments(ap)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(_build_mjcf(args.assets_dir))
    data = mujoco.MjData(model)
    mocap_ids = {nm: _mocap_id(model, nm) for nm in DEVICES}

    head_tracker = deviceio.HeadTracker()
    controller_tracker = deviceio.ControllerTracker()
    trackers = [head_tracker, controller_tracker]
    extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)

    print(f"Visualize Poses (MuJoCo)  |  assets: {os.path.normpath(args.assets_dir)}")
    print("Close the viewer window or Ctrl+C to quit.")

    with CloudXRLauncher.launch_context(args):
        with (
            oxr.OpenXRSession("VisualizePosesMuJoCo", extensions) as oxr_session,
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

                    head = head_tracker.get_head(session)
                    if head is not None and head.is_valid:
                        pos, quat = _convert_pose(
                            head.pose.position, head.pose.orientation
                        )
                        data.mocap_pos[mocap_ids["headset"]] = pos
                        data.mocap_quat[mocap_ids["headset"]] = quat
                        viewer.cam.lookat[:] = (
                            pos  # keep mouse-wheel zoom centred on head
                        )

                    for label, getter in (
                        ("controller_left", controller_tracker.get_left_controller),
                        ("controller_right", controller_tracker.get_right_controller),
                    ):
                        ctrl = getter(session)
                        if ctrl is None:
                            continue
                        cpose = ctrl.aim_pose if args.pose == "aim" else ctrl.grip_pose
                        if not cpose.is_valid:
                            continue
                        pos, quat = _convert_pose(
                            cpose.pose.position, cpose.pose.orientation
                        )
                        data.mocap_pos[mocap_ids[label]] = pos
                        data.mocap_quat[mocap_ids[label]] = quat

                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    if args.debug and time.time() - last_debug >= 1.0:
                        last_debug = time.time()
                        print("mocap_pos (MJ frame, Z-up):")
                        for nm, mid in mocap_ids.items():
                            p = data.mocap_pos[mid]
                            print(f"  {nm:18s} [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]")

                    time.sleep(1 / 60)
            except KeyboardInterrupt:
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
