<!--
SPDX-FileCopyrightText: Copyright (c) 2026 HTC CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CloudXR MuJoCo Teleop Visualization Examples

Live MuJoCo viewers for the OpenXR pose streams arriving from IsaacTeleop —
typically while a Vive headset streams through NVIDIA CloudXR. The samples
ship with a generic HMD and Vive Focus 3 (controllers) meshes and textures
in `vive_assets/`, but the pipeline they consume is runtime-agnostic and
works with any CloudXR / OpenXR-compatible HMD.

| Example | What it shows |
|---------|---------------|
| `visualize_poses_mujoco_example` | HMD + left/right controller as textured mocap bodies. |
| `visualize_hands_mujoco_example`  | Both hands as 26-joint skeletons (spheres + capsule bones), with the HMD as optional context. |

## Prerequisites

Assumes IsaacTeleop is already built and installed in this tree.

1. **[`uv`](https://docs.astral.sh/uv/)** (one-time install):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Install the example once; this pulls its Python dependencies
   (`mujoco`, `numpy`, `scipy` — see `pyproject.toml`):
   ```bash
   uv pip install -e ./examples/cloudxr_mujoco_teleop
   ```

2. **CloudXR environment loaded**, so OpenXR resolves to the CloudXR
   runtime. Source the setup script that ships with your CloudXR install:
   ```bash
   source ~/.cloudxr/run/cloudxr.env
   ```
   The CloudXR runtime should be running with the headset connected before
   launching the examples below.

Close the viewer window or press `Ctrl+C` to quit either example. No time
limit, no recording.

## Examples

### `visualize_poses_mujoco_example`

HMD + left + right controller, each rendered as a textured mesh mocap body.

```bash
python -m isaaccapture_examples.cloudxr_mujoco_teleop.visualize_poses_mujoco_example
```

| Flag | Default | Description |
|------|---------|-------------|
| `--assets-dir DIR` | `./vive_assets` | Where to find the device meshes / textures. |
| `--pose {grip,aim}` | `grip` | Which controller pose drives the mesh — natural "hold" pose or the aim ray. |
| `--debug` | off | Print `mocap_pos` for all three devices once per second. |

### `visualize_hands_mujoco_example`

Both hands as 26-joint skeletons drawn directly into `viewer.user_scn` each
frame (left = cyan, right = warm orange) — avoids paying for 52 mocap bodies
in the model. The HMD is shown as context using the same mesh + texture as
the poses example.

```bash
python -m isaaccapture_examples.cloudxr_mujoco_teleop.visualize_hands_mujoco_example
```

| Flag | Default | Description |
|------|---------|-------------|
| `--no-head` | off | Skip the head mesh — only draw the hand skeletons. |
| `--debug` | off | Print per-hand valid-joint counts once per second. |

Hand tracking obviously requires hand poses to actually arrive in the
stream. If only controllers are tracked, both hands stay empty and only the
HMD will be visible.

## Viewer controls (both examples)

- **Mouse wheel** — zoom. The free camera's `lookat` is rewritten to the
  headset position every frame, so zoom always converges on the head.
- **Left-click drag** — orbit around the headset.
- **Right-click drag** — pan (the next frame's auto-recentre snaps back).
- `[` / `]` — cycle cameras: `free` → fixed cameras (the poses example
  also ships `play_area` and a `follow_head` camera that rides behind/above
  the HMD via `mode="trackcom"`).
- Side panels are hidden by default; press `Tab` to toggle them back if you
  need MuJoCo's built-in inspector / rendering options.

## Directory layout

```
cloudxr_mujoco_teleop/
├── README.md
├── pyproject.toml                            # the distribution
└── python/isaaccapture_examples/cloudxr_mujoco_teleop/
    ├── visualize_poses_mujoco_example.py
    ├── visualize_hands_mujoco_example.py
    └── vive_assets/
        ├── generic_hmd.obj                   # HMD mesh (~500 KB)
        ├── generic_hmd_color.png             # HMD texture
        ├── vive_focus3_controller_left.obj
        ├── vive_focus3_controller_left_color.png
        ├── vive_focus3_controller_right.obj
        └── vive_focus3_controller_right_color.png
```

The bundled meshes are a generic HMD and the Vive Focus 3 controllers.
Each OBJ carries UV coordinates; the matching `*_color.png` is bound via
MJCF `<material texture="..."/>`, so missing `.mtl` files are not a
problem.

Both examples share `vive_assets/` — the hands example reuses the HMD mesh
+ texture for its context body.

If any asset is missing each example degrades gracefully: a mesh-less
device falls back to a coloured primitive box; a textureless mesh falls
back to a solid colour.

## Notes for adaptation

These are also documented at the top of each script.

- **Coordinate frames.** OpenXR is right-handed, Y-up, -Z forward; MuJoCo's
  default is right-handed, Z-up. Every incoming pose is rotated by +90°
  about X, and the quaternion (for the pose mocap bodies) is reordered
  from `xyzw` (OpenXR / scipy) to `wxyz` (MuJoCo).
- **Mesh-frame fix.** OBJ files are not modelled in the OpenXR pose frame.
  Both scripts apply a per-device static rotation (the geom's local `quat`)
  to align each mesh with its tracked pose. The shipped values
  (`(0, 0, 180)` for the HMD, `(0, 90, 0)` for the controllers) were tuned
  visually for the bundled generic HMD and Vive Focus 3 controllers —
  **re-tune them for your own meshes.**
- **Reference space.** Both samples assume a stage / floor-relative
  reference space, so the headset reports its real height (`Y ≈ 1.5 m`).
  With a local / head-relative reference space the headset always reads as
  the origin and everything will collapse to floor level.

## Trademarks

VIVE, VIVE Focus 3, and HTC are trademarks of HTC Corporation, registered
in the U.S. and other countries. The 3D models and textures in this
directory depict HTC hardware and are provided by HTC Corporation under the
Apache-2.0 license for use with the IsaacTeleop examples.
The Apache-2.0 license does NOT grant any right to use HTC's trademarks,
trade names, or product names. Use of these names in this repository is for
identification only and does not imply endorsement by HTC.
