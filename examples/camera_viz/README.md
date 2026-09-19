<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_viz

> Camera streaming + visualization on Televiz (`isaacteleop.viz`).

| Mode | What it does |
|---|---|
| **Direct** | Workstation runs the viewer with cameras attached locally. |
| **Split**  | Robot runs the sender, ships RTP H.264 to a workstation receiver. Wired Ethernet only. |

## Supported cameras

| YAML `type:` | Notes |
|---|---|
| `synthetic` | GPU test pattern — no hardware. `stereo: true` adds a `disparity_px` offset between eyes |
| `v4l2`      | USB / UVC — anything `v4l2-ctl --list-formats-ext` shows |
| `oakd`      | OAK-D RGB / LEFT / RIGHT; mono or `stereo: true` (GRAY8 over USB, GPU-broadcast to RGBA; `stereo_rgb` for color). Needs the Luxonis udev rule — see below |
| `zed`       | ZED 2 / Mini / X One; mono or `stereo: true` (per-eye SDK retrieve, zero-copy GPU) |
| `video`     | Video-file replay (anything OpenCV/FFmpeg reads) — preview / testing without a camera. Loops by default; `stereo: true` splits side-by-side files into eyes (viewer only) |

In XR mode the viewer **attaches to the CloudXR runtime + WSS proxy**, starting a background service if none is serving — nothing to start separately (`--accept-eula` for the first run; `camera_viz.py --help` for the rest). Output: XR headset (default) or desktop window (`run CONFIG --mode window`); one surface per camera — a flat plane (default), a cylinder arc, or an equirect sphere (`placements.<name>.shape`, XR only for the curved shapes). Stereo cameras render true SBS in XR; window mode shows the left eye. XR placements: `world` / `head` / `lazy` / `gimbal`.

---

## Setup (one-time)

```bash
examples/camera_viz/camera_viz.sh setup
source examples/camera_viz/.venv/bin/activate
```

`setup` creates `.venv/` via `uv` (no `--system-site-packages`) and installs `isaacteleop[cloudxr]` — which bundles Televiz — plus every other Python dep. The `cloudxr` extra is not optional here: XR is the default mode and the viewer launches the runtime itself. It then probes system packages (cairo / girepository headers, GStreamer plugins under `--with-rtp`, JetPack `cuda-nvrtc` + ld.so wiring). If anything's missing it prints the exact `apt-get` line and prompts `[y/N]` — `n` or non-interactive aborts. No need to build IsaacTeleop from source.

camera_viz needs an `isaacteleop` new enough to carry the features it uses, so `setup` works down a ladder to get one: newest **final release** meeting that minimum; else newest **release candidate** (an rc is published from every release-branch commit, and PEP 440 keeps pre-releases out of a plain minimum-version specifier); else a **source build** of this checkout, after asking. Final releases win automatically whenever one qualifies. The minimum itself lives in `scripts/_install_deps.sh`.

Flags: `--no-{v4l2,oakd}`, `--with-rtp` (split mode / `loopback`; implied by `--sender-only`), `--with-zed`, `--sender-only`, `--jetson`. Pass `--venv PATH` to install into an existing venv (symlinks `.venv` → PATH so `run` / `loopback` pick it up too).

> **OAK-D?** The camera needs a udev rule, or `depthai` reports `Insufficient permissions to communicate with X_LINK_UNBOOTED device` and never finds it. `setup` prompts to install one when an OAK-D is attached and no rule covers it; declining is not fatal. By hand:
> ```bash
> echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
> sudo udevadm control --reload-rules && sudo udevadm trigger   # then replug
> ```

> **Developing against a local build?** Pass `--wheel <path>` (e.g. `camera_viz.sh setup --wheel build/wheels/isaacteleop-*.whl`) for a wheel you already built, or `--build-from-source` to build this checkout without asking. See the [build-from-source guide](../../docs/source/getting_started/build_from_source/index.rst).

---

## Mode 1 — Direct

```bash
./camera_viz.sh run configs/v4l2.yaml                  # XR headset (default)
./camera_viz.sh run configs/v4l2.yaml --mode window    # desktop window instead
```

Set `source: local`. Swap config for `oakd.yaml`, `zed.yaml`, `realsense.yaml`, `synthetic.yaml`, `synthetic_stereo.yaml`, `synthetic_xr_3up.yaml`, `multi_camera.yaml`, `replay.yaml` (file replay — point `path:` at any recording).

> **Jetson Orin:** when the CloudXR runtime runs on Orin, set **Video Codec** to **H.264** in the CloudXR web client.

## Mode 2 — Split (robot → workstation, RTP)

> ⚠ **Wired only.** No retransmit / FEC; one lost packet = one corrupted frame until the next IDR (default 5 s).

```bash
# YAML: set source: rtp. Leave streaming.host as-is — overridden at deploy time.
$EDITOR configs/v4l2.yaml

# Export creds once per shell (keeps password out of history / argv):
export REMOTE_HOST=10.0.0.5 REMOTE_USER=nvidia
read -s REMOTE_PASSWORD && export REMOTE_PASSWORD   # if no SSH keys
export STREAMING_HOST=10.0.0.42                      # workstation IP

./camera_viz.sh deploy configs/v4l2.yaml             # full deploy + systemd
./camera_viz.sh run    configs/v4l2.yaml             # viewer on the workstation
./camera_viz.sh service-{status,logs,restart}        # operate the unit
```

What `deploy` does:

1. `rsync` source to `~/camera_viz` on the robot.
2. `ssh -t` runs `_install_deps.sh --sender-only --jetson`; the `[y/N]` prompt fires for any missing apt / CUDA wiring on the Jetson.
3. Renders `~/.config/systemd/user/camera-streamer.service`. `--streaming-host` (or `$STREAMING_HOST`) injects `--host IP` into the unit's `ExecStart`; the YAML on disk stays untouched.
4. `sudo loginctl enable-linger` (one-time) + `systemctl --user enable --now`.

`--no-service` stops after step 2. The sender retries forever (unplug, SDK errors, network blips); the service never voluntarily exits.

### Loopback

`./camera_viz.sh loopback configs/v4l2.yaml` runs sender + viewer on `127.0.0.1`. Quickest way to smoke-test the RTP path.

---

## Config

```yaml
source: local | rtp           # camera_viz only
streaming:
  host: 192.168.1.100         # workstation IP (override at deploy time)
encoder: auto | native | gstreamer

cameras:
  - name: cam
    enabled: true
    type: v4l2                # v4l2 | oakd | zed | synthetic | video
    width: 2560               # video: optional — defaults to the file's size
    height: 720
    fps: 30
    stereo: false             # zed / oakd / synthetic / video — per-eye capture + SBS XR
    # … type-specific fields (e.g. synthetic: disparity_px; video: path, loop)
    rtp:
      port: 5000              # left eye when stereo
      port_right: 5001        # required when stereo + source: rtp
      bitrate_mbps: 15
      # gop: 150              # default fps*5
      # gpu_id: 0             # multi-GPU pin

display:                      # camera_viz only
  mode: xr | window           # default: xr
  window: { width, height }
  xr:     { near_z, far_z, system_wait_seconds }   # wait: -1 forever (default), 0 fail fast
  clear_color: [r, g, b, a]
  placements:
    cam:
      lock_mode: lazy         # world | head | lazy | gimbal
      distance: 1.5
      offset_x: 0.0
      offset_y: 0.0
      # size: [w_m, h_m]
      # stereo_plane_distance_cm: 0   # stereo cams: gap between the left- and
                                      # right-eye planes; 0 = both eyes share
                                      # one plane. See "Stereo plane distance"
      # equirect_yaw_deg: 0.0  # equirect: heading the middle of the feed points at
      # shape: quad            # quad (default) | cylinder | equirect — XR only for
                               # the curved shapes
      # compositor: openxr     # openxr (default) | televiz — quads only
                               # Orin: use televiz (see Known issues)
      # cylinder_radius_m: 2.0 # cylinder: viewing distance to the arc
      # cylinder_angle_deg: 90 # cylinder: visible arc width
```

Multiple cameras → multiple `cameras:` entries; each gets its own `rtp.port` (plus `port_right` if stereo) and renders as its own plane.

## Known issues

### No video on Orin

When the CloudXR runtime runs on Jetson Orin, set **Video Codec** to **H.264** in the CloudXR web client.

### Black camera plane on Orin

With the CloudXR experimental runtime on Orin, the default OpenXR compositor displays the camera plane but its contents remain black. Set `compositor: televiz` under the camera's `display.placements` entry. The same feed displays correctly with this workaround.

```yaml
display:
  placements:
    cam:
      compositor: televiz
```

## Lock modes (XR)

| Mode | Behavior |
|---|---|
| `world`  | Placed once in front of you; stays put |
| `head`   | Follows your head every frame |
| `gimbal` | Translation follows you, yaw stays as first seen — walk and it comes along, turn your head and you look around it. For wide cylinder feeds |
| `lazy`   | World-locked, re-snaps when you look away (default) |

Lazy knobs under `placements.<name>`: `look_away_angle_deg`, `reposition_distance`, `reposition_delay_s`, `transition_duration_s`.

## Controller bindings (XR)

Retune the view without editing YAML and restarting. The right hand changes how the feed looks, the left what surface it's on. Quest and Pico report these identically.

| Input | Effect |
|---|---|
| **Right stick** ←/→ | Stereo plane gap. Stereo cameras only — on `equirect`, pans instead |
| **Right stick click** | Recenter on your view |
| **A** | Lock mode: `world` → `head` → `gimbal` → `lazy` |
| **B** | Mono / stereo. Stereo cameras only |
| **X** | Shape: `quad` → `cylinder` → `equirect` |
| **Y** | Reset everything to the YAML values |
| **Left stick** | Per shape, below |

| Shape | ←/→ | ↑/↓ |
|---|---|---|
| `quad` | size, aspect preserved | slide up / down |
| `cylinder` | arc width | slide up / down |
| `equirect` | horizontal span | vertical span |

Recentering reads per shape: an `equirect` yaws so the middle of the panorama lands dead ahead, and a placed surface re-snaps its anchor — the way back from a `world`-locked plane left in another part of the room.

Changes apply to every camera at once and appear on the status panel and an in-headset HUD that auto-hides ~2.5 s later (`hud: false` to disable). Nothing is written back to the YAML.

Neither toggle reallocates: **B** sends the left frame to both eyes, **X** flips visibility between shapes all built at startup. The cost is VRAM — two extra layers per camera, reported at startup; `shape_switching: false` keeps only the configured shape.

A quad's `distance` and a cylinder's `cylinder_radius_m` stay YAML-only. Apparent size is `2·atan((w/2)/d)` and arc width is `radius × angle`, so moving either surface further away enlarges it by the same factor and looks identical — they'd duplicate the size axis. What they *do* change is the real distance to the surface, which the stereo gap works from.

```yaml
display:
  controls:
    enabled: true                      # false disables the bindings
    hud: true                          # in-headset readout
    shape_switching: true              # keep all 3 shapes resident for X
    deadzone: 0.2                      # stick rest-position tolerance
    plane_distance_rate_cm_per_s: 2.0  # held-stick ramp rates
    size_rate_m_per_s: 0.5
    offset_rate_m_per_s: 0.5
    angle_rate_deg_per_s: 40.0
```

Limits — `plane_distance_min_cm` / `plane_distance_max_cm`, `size_range_m`, `offset_y_range_m`, `cylinder_angle_range_deg`, `equirect_h_range_deg`, `equirect_v_half_range_deg` — all default inside what the layers accept. See `configs/zed.yaml`.

### Stereo plane distance

A stereo layer draws each eye's image on its own plane; `stereo_plane_distance_cm` is the gap between them.

```yaml
display:
  placements:
    zed:
      distance: 1.0                  # how far away the planes are
      stereo_plane_distance_cm: 5.0  # how far apart they are
```

At `0` both eyes share one plane, so the whole scene — near objects and far background alike — is packed into the space between you and `distance`. Widening the gap pushes it back and lets it spread out. Applied exactly as given (×10 into the layer's `stereo_baseline_mm`); the stick moves it in 0.1 cm steps.

The HUD suggests a value from the plane distance and the headset's measured IPD, shown beside the one in use (`5.0/5.2` on the panel). Advice only — the stick sets the value, which is also how you correct a headset whose IPD setting doesn't match your eyes.

The stick can't reach **divergent parallax**: at a gap equal to your IPD the eyes' rays are parallel, and beyond it they would have to splay outward — the classic cause of stereo eye strain. The ceiling comes from the measured IPD, not the config.

**B** parks the gap at zero while mono, since one image on two separated planes would only shift its depth, and restores it on the way back.

Not applicable to `equirect`: the gap shifts each eye's surface, and the sphere sits at infinite radius where translating it does nothing, so the stick skips it there and pans instead. Curved layers have a rotation-based equivalent in Televiz (`stereo_convergence_deg`), which is uniform across the arc and works at any radius.

> **Not the camera's baseline.** That is the physical gap between the camera's two lenses — fixed in hardware, baked into the pixels, and what sets the scene's depth *scale*. This only moves where that scene sits.

> Live adjustment needs `Layer.set_stereo_baseline_mm`, newer than the released `isaacteleop` wheel. On an older wheel that one binding disables itself with a notice; A, B, X and Y still work.

### Aiming an equirect

The middle of the texture maps to the sphere pose's **−z**, and the middle row to its horizon. With no rotation that is the reference space's forward — wherever the headset faced when it last recentered — so a camera mounted facing another way puts the interesting part of the feed off to one side.

```yaml
display:
  placements:
    sky:
      shape: equirect
      equirect_yaw_deg: -90.0  # heading the middle of the feed points at
```

Positive is a turn to the left. The right stick pans it live and the stick click snaps it to your view; **Y** puts it back to the YAML value. The heading in use rides in the panel's `shape` column (`equirect +35°`).

The sphere has no lock mode: it is centred on you and follows your head everywhere, so `world` / `head` / `lazy` and the `distance` / `size` knobs are all blank for it. This heading is its whole placement.

---

## CloudXR runtime settings

The runtime is configured by environment variables, and there are three ways to set one and be silently ignored: a name nothing reads, a stale `source ~/.cloudxr/run/cloudxr.env` in your shell (which outranks both `os.environ` and `--cloudxr-device-profile`), and a boolean spelled `False` — which the runtime's exact-match parser doesn't recognise, so it reads as **true**.

camera_viz ships defaults in `cloudxr_env.DEFAULT_ENV` — pose wait off, runtime foveation on — so a config that says nothing gets both. `display.cloudxr` is a per-deployment override; list only what you want to change:

```yaml
display:
  cloudxr:
    NV_DEVICE_PROFILE: apple-vision-pro
    NV_ENABLE_POSE_WAIT: null   # null drops a camera_viz default
```

camera_viz writes these to a generated `--cloudxr-env-config` file, the one tier that outranks the process environment. Booleans are lowercased for you; the launcher-computed keys (`XR_RUNTIME_JSON`, `NV_CXR_RUNTIME_DIR`, `NV_CXR_OUTPUT_DIR`, `XRT_NO_STDIN`) are refused by name. Passing `--cloudxr-env-config` yourself wins — camera_viz won't overwrite it.

An unknown variable name warns with a suggestion before anything launches — `NV_CXR_DEVICE_PROFILE` is not a name the runtime reads, and it's one keystroke from one that is — but is still passed through, since the runtime has more knobs than camera_viz lists.

To confirm what the runtime actually resolved, read the settings dump at the top of the newest `~/.cloudxr/logs/cxr_server.*.log`. Monado's `DEBUG_GET_ONCE` options (`NV_ENABLE_POSE_WAIT` among them) aren't in that dump; run with `XRT_PRINT_OPTIONS=true` and each is logged as `NAME=value (raw)`, where `(nil)` means it never arrived.

> `NV_DEVICE_PROFILE` is device matching, not tuning. Anything other than `quest3` selects separate-frames packing, and `isFrameClientReconstructed()` bails unless packing is packed-frame — so a non-Quest profile gives up client-reconstructed streaming.

---

## Status panel

On a terminal, camera_viz redraws a snapshot in place instead of scrolling a log:

```
camera_viz  xr · local · 1 camera
────────────────────────────────────────────────────────────────────
  render 58.0 fps (target 72)   missed 0   gpu 2.1 ms

  camera      shape          lock    eyes   size m  height m  planes cm   submit/s
  zed         cylinder       lazy    stereo 1.00    +0.00     5.0/5.2     64.0

  headset IPD 63 mm

  stereo planes: 5.0 cm  ·  suggested: 5.2 cm · IPD: 63 mm
```

`planes cm` is the value in use and the suggestion. `-` means the field doesn't apply — equirect has no gap, window mode has no controls. While the panel is live it owns stderr: control messages and source lifecycle lines go through it rather than printing underneath and pushing it out of alignment. When stderr isn't a terminal (piped, or a `deploy`ed systemd unit) it falls back to one line every 5 s with the same numbers.

## Layout

```
camera_viz/
├── camera_viz.sh        — CLI: setup / loopback / run / deploy / service-*
├── camera_viz.py        — receiver / viewer (entrypoint + wiring)
├── camera_streamer.py   — robot-side RTP sender (per-camera supervisor)
├── config.py            — YAML → SourceEntry: parse + validate, no allocation
├── display.py           — VizSession + one layer per surface
├── cloudxr_env.py       — display.cloudxr → the runtime's env file
├── dashboard.py         — terminal status panel
├── controls/            — XR controller bindings, shapes, stereo geometry, HUD
├── pipeline/            — source ABC + threaded runner
├── placements/          — XR lock-mode strategies
├── sources/             — V4L2 / OAK-D / ZED / synthetic / video replay / rtp_h264
├── transports/          — RTP sender + receiver, native + GStreamer
├── codec/               — native NVENC/NVDEC pybind module
├── configs/             — one YAML per camera kind
├── test_data/           — sample replay clip (Git LFS)
└── scripts/
    ├── _install_deps.sh             — installer (setup + deploy)
    └── camera-streamer.service.in   — systemd unit template
```

---

## Sharing the XR session with TeleopSession

Only one OpenXR session is allowed per process. `VizSession` can own it and hand its live handles to `TeleopSession` / `DeviceIOSession` so they skip creating their own:

```python
import isaacteleop.viz as viz
from isaacteleop.oxr import OpenXRSessionHandles

cfg = viz.VizSessionConfig()
cfg.mode = viz.DisplayMode.kXr
# Aggregate the XR extensions downstream trackers need (e.g.
# XR_NVX1_action_context for ControllerTracker) so they're present
# on the XrInstance we're about to create.
cfg.required_extensions = DeviceIOSession.get_required_extensions(trackers)
viz_session = viz.VizSession.create(cfg)

# Pass the live handles into TeleopSession via its config.
config = TeleopSessionConfig(
    app_name="MyApp",
    pipeline=pipeline,
    oxr_handles=OpenXRSessionHandles(*viz_session.get_oxr_handles()),
)
with TeleopSession(config) as session:
    ...
```

`viz_session.get_oxr_handles()` returns `(instance, session, space, proc_addr)` as raw `uint64`s, or `None` outside `kXr`.
