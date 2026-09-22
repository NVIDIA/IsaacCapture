<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Robot Viz

A robot twin rendered stereoscopically into an Isaac Teleop Televiz XR session: a follower arm the operator drags around by hand (`--arm`, SO-101 by default), and a gripper ghost that replaces it once the clutch engages. One process, one OpenXR session, two threads — `TeleopSession` creates the session the trackers and the compositor share and runs the twin's frame loop on a thread of its own.

`isaaccapture.viz.robot` holds no `mjModel` and no `mjData`: it addresses the scene by name and publishes what moved, and `twin.py` applies the lot on the render thread. That bound is what lets the backend link against a MuJoCo the user's environment knows nothing about, and is why this example is pure Python — no compiled extension, no ABI tag, no `mujoco` pin.

## The backend is a readback, not a renderer

`mjr_render` draws into MuJoCo's offscreen framebuffer; `src/viz/robot_twin/cpp/gl_readback.cpp` blits that into a sampleable pair, runs one fullscreen pass, and reads the result into a pixel-pack buffer that CUDA imports. Every step stays in video memory, and `ProjectionLayer.submit()` is reached by CUDA pointer with no copy through host memory.

The entry point is load-bearing. `cudaGraphicsGLRegisterImage` registers no depth format and no multisampled renderbuffer, and `mjrContext.offDepthStencil` is both — that is the wall a naive port hits. `cudaGraphicsGLRegisterBuffer` has neither restriction, and `glReadPixels` into a bound `GL_PIXEL_PACK_BUFFER` is a device-to-device transfer.

The fullscreen pass exists for two conversions, both silent failures on anything short of a headset:

| | MuJoCo writes | ProjectionLayer is promised |
|---|---|---|
| row 0 | bottom (`glClipControl(GL_LOWER_LEFT, ...)`) | top |
| depth | reverse Z, near → 1 (`GL_GEQUAL`, `glClearDepth(0)`) | near → 0 |

The depth line is `1.0 - d`, which is what MuJoCo's own `mjr_readPixels` does on the CPU (`flipDepthIfRequired`, render_gl2.c); doing it in the shader keeps the host out of the loop. What it buys: every geom type, and the scene XML's materials, lights, shadows and reflections.

The backend lives in `src/viz/robot_twin/`, ships with Televiz on Linux, and ships its own MuJoCo under a private name. Whatever `mujoco` the environment has, at whatever version or none at all, is unrelated to the twin's. A Windows Televiz build omits the twin — its headless OpenGL context is EGL.

## Status

The XR half — everything downstream of the readback — has never executed anywhere. See [Not verified](#not-verified-anywhere-in-ci-or-on-a-developer-desktop).

## Scope

Renderer + MuJoCo + rig, and two scenes — one per follower in `viz.robot.PREVIEW_ARMS`, selected with `--arm`: `assets/scene.xml` (SO-101, the default) and `assets/scene_rebot.xml` (reBot DevArm, RobStride build). Each pairs its follower arm with a gripper ghost on the operator's hand, exactly one drawn at a time: the SO-101 shows its LEADER's gripper, and the reBot — which has no leader hardware — its own follower gripper. No table, no blocks, no ground plane: this is an AR scene and passthrough is the background.

Each ghost is a real mesh assembly of fetched STLs, so it exercises the `mjGEOM_MESH` path — the SO-101's is four with a hinged trigger, the reBot's seven with two rack-and-pinion fingers (`viz.robot.ghost` holds both specs). Closedness is driven by the shipped `SO101GripperRetargeter` as a graph edge — a `BaseRetargeter` node inside `_build_pipeline()`, whose closedness output reaches `mjData` and therefore the screen. Its pose is the safety harness's output, not the controller's.

Two calibrations, different in kind. `src/viz/robot_twin/cpp/frames.hpp` is a convention fixed by two specs and cannot be wrong at runtime. `EULER_HAND_FROM_GHOST_DEG` / `POS_HAND_FROM_GHOST` in `viz.robot.so101_ghost` are a measurement of how a hand holds a tool, taken on a headset and checkable nowhere else.

## Build

Nothing here is compiled; what has to be built is `isaaccapture` itself.

```bash
cmake -B build -DBUILD_VIZ=ON
cmake --build build --parallel
```

There is no flag for the twin: a Linux `BUILD_VIZ` build has it. The first configure fetches MuJoCo and its six vendored dependencies from GitHub and compiles them (~40 s on 12 cores).

To just run it, `uv pip install .` drives the same CMake and skips the `build/` tree:

```bash
uv pip install .                       # from the repo root
uv pip install -e ./examples/robot_viz # same environment
python -m isaaccapture_examples.robot_viz
```

Both must land in one environment. Beware `uv pip install isaaccapture` without a path or `--find-links`: it does not resolve at all until `isaaccapture` is published, and once it is, it resolves to a release with no robot twin in it.

Running additionally needs a GPU with EGL + CUDA, and a headset. On a multi-GPU host, pass the GPU: `SceneTwin` takes a `gl_device_index`, which indexes EGL devices and need not agree with CUDA's ordering. The renderer checks at construction and names both device numbers.

## Run

```bash
python -m isaaccapture_examples.robot_viz --help   # includes CloudXRLauncher's flags
```

The `isaaccapture` wheel has to be installed in the interpreter you launch with, not in the build venv. Picking up the wrong venv is silent, so check first — this import also fails, and says so, on an `isaaccapture` built without the twin:

```bash
python -c "import sys, isaaccapture; from isaaccapture.viz.robot import SceneTwin; print(sys.executable, isaaccapture.__file__)"
```

Against a runtime you started yourself:

```bash
python -m isaaccapture.cloudxr --accept-eula                                    # one terminal
python -m isaaccapture_examples.robot_viz --no-launch-cloudxr-runtime           # another
```

Omitting `--no-launch-cloudxr-runtime` makes the app start its own runtime, which is right when nothing else has and fatal when something has (the runtime is a host singleton on WSS port 48322). Pass it with no runtime running and the failure comes out of `VizSession.create` as an OpenXR error before any of this example's code runs — no `[robot_viz]` lines at all is the tell.

`--arm` picks the scene, from the profiles in `viz.robot.PREVIEW_ARMS`; each scene's MJCF wrappers are package data beside the module. There is no desktop or headless display mode.

## The harness the ghost renders

The ghost's pose comes from an `EePoseRateLimiter` (`viz.robot.harness`, `_build_pipeline()`), so what the operator sees is the command a follower would execute, not where their hand is. [#738](https://github.com/NVIDIA/IsaacTeleop/issues/738) reports operators losing minutes to harness interventions they could not perceive.

Two things report which band is live: the ghost lags the hand (proprioceptive, free), and the ghost changes colour — amber while clamping, red while rejecting, authored blue while passing through. Categorical, which the lag alone is not: a clamp and a refusal both read as "behind my hand".

Colour is written to the shared `leader_ghost` material, so one write recolours the whole tool. Do not switch it to `geom_rgba`: that silently wins over the material, and the four geoms would then have to be kept in step by hand. Alpha stays 1.0 in every band.

`InterventionMonitor` recovers the band by comparing the limiter's input against its output rather than by asking the limiter, at the cost of one extra `OutputCombiner` key (`COMMANDED_POSE_KEY`) that nothing draws.

The limits are chosen for this demo, not measured against an SO-101: 0.5 m/s and 2.5 rad/s clamp, 2.0 m/s and 10 rad/s reject, set so ordinary reaching is pass-through and a deliberate flick trips both bands on demand. `RateLimiterConfig` itself defaults to 0.25 m/s, which would clamp during ordinary reaching. Nothing tests these numbers, so retuning `_HARNESS` past what a hand can reach silently produces a demo that never intervenes.

The jaw is ungoverned: the trigger is one scalar the operator drives directly, not a solved output that can diverge.

## The clutch and the follower preview

Disengaged, the operator drives the follower's gripper. Position is the controller's, all three axes; yaw is the controller's own yaw, every frame, with no button held.

The joints are locked. `qpos` is written once, to `Q_HOME`, and the arm is moved as a rigid body — `PreviewArm._place` is the sole writer of both `body_pos` and `body_quat` — so the gripper's orientation in the base's frame is constant and the jaw lands at its offset from the hand with no residual. The offset starts at `GRIP_FROM_CONTROLLER_XR`: level with the hand laterally, 0.25 m ahead and 0.10 m below.

Both channels land on `GRIPPER_SITE`, and that is what makes the yaw pivot right. Upstream declares a `gripperframe` site on the `gripper` body 98.4 mm out from its origin, 3.8 mm off the closed jaw surface. The arm is placed by that point, so it is also the axis the yaw turns about: the jaw holds still and the base swings around it, 273 mm across ±90° of yaw. Place by the gripper body instead and the pinned point sits 98.4 mm short of the jaw, which then orbits it — a 15.8 mm arc over the same sweep, on the one point the operator is aiming. The body frame stays the orientation carrier and `PreviewArm.gripper_pose_mj()` returns it for that; do not read its position as the tool point.

The offset is tunable from the headset, and only while disengaged. The right thumbstick walks its two horizontal terms at 0.20 m/s of full deflection, past a 0.15 deadzone and clamped to ±0.60 m on each. Deflection is a rate, so the offset holds where the stick left it. Both terms are carried on the controller's own facing — not the base yaw, which leads it by the measured bias and would send "forward" off by exactly that much. Let go and the app prints what it settled on, to paste back into `viz.robot.preview_arm` as the new default:

```
preview arm: offset tuned to GRIP_FROM_CONTROLLER_XR = np.array([-0.31, -0.10, -0.22])
```

B (`SECONDARY_CLICK`) restores the authored value on its rising edge. While `ENGAGED` neither the stick nor the button does anything: the arm is hidden and frozen, and an offset moving under it would apply the whole excursion on the release frame.

Squeeze while the arm is green and the follower vanishes, the leader appears in the hand at the follower's rotation, and the harness colouring takes over. The offset is the preview's alone — the leader is mapped 1:1 onto the controller and an engagement does not inherit it, so the engage frame swaps one tool for another a whole offset away.

Release and the follower comes back immediately, on the release frame itself. Do not put a smoothstep back to the home pose on the release: the drag runs on every disengaged frame, so a ramp has nothing to do — the arm would reach home and be dragged straight back onto the hand on the next frame. The accepted cost is that the arm teleports from where it froze onto the hand's position and yaw. The gate's phase conjunct holds it shut for the whole engagement, so `_dwell_s` restarts at the release and no squeeze can re-latch for ~8 frames.

### `aim`, not `grip`

The app drives from the `aim` pose and `HAND_POSE` is the single constant that says so; it reaches the follower's drive, the gate's operand, the clutch's latched home and the ghost's placement through the one `HAND_POSE_KEY` channel. `grip` is the palm centroid, its `-Z` running little finger to thumb up through the fist, so a facing read off it has an arbitrary zero the operator sees and cannot tune away. `aim`'s `-Z` is the pointing ray. The cost is that `aim`'s origin is a device-specific ray origin, so the arm's position gains a lever arm that swings as the wrist turns.

`SO101ClutchRetargeter` reads the controller group directly rather than that channel, so it takes a matching `controller_pose=` argument; its orientation delta is invariant to the choice — for a fixed body-frame `T`, `(R·T)(R₀·T)⁻¹ == R·R₀⁻¹` — so that argument changes the translation pivot only.

Which axis to read is a `grip` question only; on `aim`, `-Z` is the ray by definition. Every candidate turns 1:1 with a rotation about the world vertical, so what separates them is how much wrist roll and pitch bleed into the arm's yaw. Worst leak over ±45° at the posture the gate demands, measured on `grip`:

| motion | `-Z` thumb | tool/barrel | flattened | best-fit |
|---|---|---|---|---|
| roll about the thumb axis | 0.00 | 12.89 | 21.09 | 23.40 |
| roll about the barrel | 11.75 | 0.00 | 29.60 | 31.94 |
| pitch about grip `+X` | 0.00 | 27.88 | 0.41 | 0.00 |
| pitch about horizontal-perp | 2.30 | 0.00 | 0.00 | 0.00 |

`aim`'s `-Z` is the tool/barrel column, whose leak is dominated by how far the axis sits from horizontal; the 27.88° is a near-vertical singularity reached only because that stand-in axis starts 43.7° up, and a real aim ray held level does not go near it. That is an expectation, not a measurement — the grip-to-aim transform is per-device, so re-measure the leak on a headset. `viz.robot.yaw_of_axis` takes the axis as a required argument for this reason; `viz.robot.yaw_of` keeps `-Z` and is for the head alone.

`_YAW_TRIM_DEG` (`viz.robot.clutch_preview`) should be zero, and a session that needs a large one is evidence, not a knob; it is kept only to absorb a runtime whose aim convention differs from the operator's expectation. Hold A and push the right thumbstick left or right at 20°/s — A owns the stick while held, so a trim cannot also walk the grip offset. It is applied as a constant on top of the reading, so it introduces no leakage of its own. Let go and the app prints it:

```
preview arm: yaw trim -> _YAW_TRIM_DEG = -10.0
```

### The ghost calibration

`EULER_HAND_FROM_GHOST_DEG` is solved, not measured. On `aim`, demanding "level and unrolled" means "hold the controller the way you would naturally point it", and `(270, 0, 90)` produces exactly that — 0.00° of pitch and 0.00° of roll against `Q_HOME`. Re-solve it when `Q_HOME`, or the aimed axis, moves: it is the gripper's `xquat` at `Q_HOME` and base yaw 0, carried into XR by `_xr_from_mj_quat`, as intrinsic-XYZ Euler. Do not port a `grip`-measured value: one demanded 30° of upward pitch, which is invisible on `grip` and means aiming at the ceiling on `aim`. Bearing is deliberately unpinned, hence the rounding to whole degrees.

`POS_HAND_FROM_GHOST` stays a headset measurement — no posture pins it. Porting it across a `HAND_POSE` change is per-device, so `_log_hand_frames` computes it from one frame with both poses valid and prints the replacement. Both terms of that port are needed — the origins' separation and the old offset turned by the same rotation; dropping the second leaves the ghost centimetres out while its orientation looks perfect.

The base leads the hand by a measured bias so the jaw faces it. `PreviewArm.jaw_yaw_xr` reads `GRIPPER_SITE`'s `+Z`, and `base_yaw_bias()` measures the lead at startup rather than authoring it, because how far the jaw sits off its own base yaw follows from `Q_HOME` and upstream's chain. With `J5 = -90` it is 92.79°. The jaw then faces the controller to 0.000° at every world yaw, with 0.00° of leak from wrist roll and 0.00° from pitch. The arm's reach parts company with the jaw by exactly J5's roll, so the arm's body sits 92.79° to the side of where you point.

The arm's yaw cancels the hand's, which is why no button locks one: the base carries the wrist's own yaw, so the rotation the clutch would latch is `wrist_yaw ∘ C` for a session constant `C`, and the angle the gate measures is the angle between the wrist's pitch and roll and `C` — the same number whichever way the operator faces and points. One posture to learn rather than one per reach.

### Where the arm goes

Against the measured head pose, not the reference-space origin. On the first frame carrying a usable `info.views[0].pose`, the home grip is placed `HOME_GRIP_FROM_HEAD_XR` — 0.30 m below and 0.60 m in front — of the head, yaw-projected onto the head's facing. Yaw only: a bowed or tilted head must not tip the arm toward the floor.

viz asks for no floor-origin space, so a runtime that hands back a stage-origin one puts everything authored against that origin a standing height out, which a headset run showed. The head pose is the only datum the app can trust.

The anchor is a starting pose and nothing more. Its position is only where the arm waits out the window between the first head pose and the first controller frame; its yaw only turns the arm to face the operator for that window. From the first driven frame the controller owns position, base yaw, and the frame the thumbstick offset is carried on; `PreviewArm.base_yaw_xr` reports whichever yaw is actually on the base. It does not follow the head afterwards either: the pose the clutch is about to latch must not move out from under the operator as they look around.

Placement is therefore runtime-derived, and neither tool is drawn until the anchor exists. `PreviewArm.__init__` starts the arm hidden, `ClutchPreview.__init__` starts the ghost hidden, and `ClutchPreview.after_step` returns early while `arm.anchored` is false, with the gate held shut.

Two phases, `DISENGAGED` and `ENGAGED`, and the enum never answers "is the clutch latched?". `SO101ClutchRetargeter.is_engaged` is the sole authority; the phase takes it as an input every frame and never copies it into a field.

### The engage gate

Green means all three of these hold, and `isaaccapture.viz.robot.EngageGate` returns every one that does not, which `app.py` logs on each transition:

- the hand's rotation is within the enter band of the rotation the clutch would latch,
- the rate limiter is passing through, not clamping,
- and all of that has held for a dwell.

The second is this app's, passed as `app_ok` and named `("limiter", "still catching up")` so the operator's log says what actually blocked.

There is no reach conjunct and no reach envelope. The rigid drag puts the gripper exactly at its offset from the hand every frame, so a position residual is identically zero and a limit on it would forbid nothing — the arm goes wherever the hand goes, including places no articulated SO-101 could reach. Do not add one back believing it keeps the operator inside a workspace this preview does not have.

The rotation conjunct is the point: the clutch composes orientation as a delta, so a wrist 40° from where the follower's gripper points stays 40° off for the entire engagement. Hysteresis and the dwell are not polish — the angle is recomputed every frame from a noisy controller, and a colour strobing at 72 Hz in a headset is worse than a wrong colour.

The pass-through conjunct is not optional. The leader renders the limiter's output, and the grip calibration converts hand rotation into ~4°/cm of tool motion, so 0.5 m/s of hand speed is 3.49 rad/s against a 2.5 rad/s clamp: clamping starts around 0.36 m/s, which is ordinary dragging.

`app.py` pushes `clutch.set_home_base_T_ee(pose_from_ghost_body(...))` on every non-`ENGAGED` frame, built from two sources: the position from the last usable hand pose (latched in `after_step`, because `before_step` runs a frame ahead of it) and the rotation from the follower's gripper. Neither half is interchangeable — latching the gripper's position would carry the preview's offset into a delta-composed engagement, and taking the rotation from the hand would leave the gate's rotation conjunct demanding nothing. `MEASURED_BASE_T_EE_INPUT` is deliberately left unwired: `_latch` reads it for position only.

Because the grip calibration cancels through the handoff, every geometric test passes for any value of it. Its one surviving effect is which wrist posture the gate demands, so `app.py` logs where the tool points and where the gate wants the operator's thumb, and warns past 45° on the thumb without refusing to start. Judge `EULER_HAND_FROM_GHOST_DEG` by the hand-axis direction (`pointing` on `aim`, `thumb` on `grip`); the tool direction reads the same 15° for a calibration that is right and one that is 118° wrong. Both are reported in the arm's own frame — XR axes un-yawed by `base_yaw_xr` — which is also why the log can run before the arm is anchored.

Nothing integrates. `mj_step` is never called: the follower is slid by its base, and the ghost is two mocap bodies. Upstream's six `position` actuators are therefore inert — with `ctrl = 0` and `mj_step` they would drag the arm back to `qpos0` at about 1 rad per 0.4 s. There are deliberately no `gravity="0 0 0"` or `<flag actuation="disable"/>` attributes in `scene.xml`: flags that suppress dynamics nobody runs tell the next reader that dynamics run.

`SO101ClutchRetargeter` gains one input for this, `ENGAGE_PERMITTED_INPUT`: an `OptionalType` boolean checked only where a latch is owed, so it gates the latch and never the engagement, and absent or unwired means permitted. It is an enable precondition, not a safety-rated stop.

## Frames (`robot_twin/cpp/frames.hpp`)

`R_mj_from_xr = Rz(-90) * Rx(+90)`. XR `-Z` → MuJoCo `+x`, XR `+Y` → MuJoCo `+z`, XR `+X` → MuJoCo `-y`. Testable definition: a point 1 m in front of the operator at eye height `h` lands at MuJoCo `(+1, 0, h)` before the workspace translation. It deliberately differs from `examples/cloudxr_mujoco_teleop/visualize_poses_mujoco_example.py`, which applies `Rx(+90)` only.

`kTransMjFromXr` is the lever, and it is a calibration that is routinely wrong: `(-1.0, 0.0, -0.73)`, two independent terms. `x` is operator standoff; `z` is a floor datum — MuJoCo `z = 0` is a work surface 0.73 m above the physical floor. That `z` is only right against a floor-origin reference space, and the session does not ask for one. A scene that puts static content on the work surface owns re-tuning it. Neither term may be zeroed.

It places static content only. The ghost goes out through `mj_from_xr` and the eye pose goes out through the same transform, so both constants cancel and the shipped scene is blind to a wrong value. Judging one means a scene with something world-locked in it.

There is no recentre keypress and no runtime override: changing the datum means editing the constant and rebuilding (~8 s). Stand where you intend to work, start the app on such a scene, read the `frames:` line in the startup log, compare the virtual surface against the real one, and adjust `z`. Do not add a Python-side `--workspace-offset`: applied to one of the two conversions and not the other, it would move the gripper and leave the scene put, which is the symptom this example exists to disambiguate.

## Where the ghost sits on the hand (`viz.robot.so101_ghost`)

`EULER_HAND_FROM_GHOST_DEG` and `POS_HAND_FROM_GHOST` place the leader gripper on the operator's hand. Without them the gripper's body origin — the follower's `gripper` datum, up at the wrist — lands on the hand pose and the tool hangs off at an arbitrary angle.

These are measured on a headset. The claim is about how a gripper should look in a hand that is actually holding a controller, so do not derive it from the mesh: a mesh-derived model that maps the handle loop onto the fist puts the loop centroid 56 mm from the palm and not straddling it at all.

The mesh geometry is worth knowing when reading the numbers. `Handle_SO101` is a closed loop, not a bar; the jaw assembly sits off to one side of it, and the jaws run 60.7° off the loop's long axis.

To re-tune: the rotation is degrees, intrinsic X-then-Y-then-Z, the same convention as a MuJoCo `euler=`. Change one angle in `so101_ghost.py` and relaunch — `Rz` spins the gripper about its own long axis, `Rx` / `Ry` tilt it in the hand, and `POS_HAND_FROM_GHOST` slides it along the hand-pose axes. No test asserts a posture, deliberately; the one that matters asserts the ghost is rigidly attached to the hand frame, which is true of any calibration and false if the correction is composed on the wrong side.

MuJoCo rewrites every mesh into its inertial frame, so recovering an STL's own axes needs `mesh_pos` / `mesh_quat`. Skip that and you get the handle's axis instead of the jaws', which is self-consistent, passes an axis-only check, and is wrong by 60°. The shank's own principal axis is no substitute either — a near-isotropic blob (σ₀/σ₁ = 1.26), so its principal direction is noise.

## Scene assets

The XML's materials, lights, shadows and reflections are live — this is `mjr_render`, so the scene file means what the MuJoCo docs say it means.

The lighting knob that matters is ambient, not diffuse. `scene.xml` sets `<visual><headlight ambient="0.4 0.4 0.4" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3"/>`. Ambient is direction-independent, so it is a floor on how dark a surface can get. Measured over the ghost from three directions, as a share of its material albedo:

| headlight (amb / diff / spec) | shades | dimmest | mean | below ⅓ albedo | above albedo |
|---|---|---|---|---|---|
| `0.1 / 0.4 / 0.5` (MuJoCo default) | 437 | 0.10 | 0.25 | 94.0% | 0% |
| `0.4 / 0.4 / 0.3` (shipped) | 372 | 0.40 | 0.55 | 0% | 0% |

The trade is explicit: 372 distinct shades against the default's 437, to buy a hard floor at 0.40 of albedo, which is the ambient term exactly. That floor bounds MuJoCo's smeared crease normals — one averaged normal per welded vertex, and `render_gl3.c` lights one-sided, so a face corner pointing away from its own triangle (11.4% of them on `wrist_roll`) lands on ambient rather than black. It bounds shadows the same way, so `mjRND_SHADOW` needs no attention.

Specular spends outside that budget: additive and white rather than scaled by the material `rgba`, and gated by the material as much as the light — `leader_gripper.xml` declares neither `specular` nor `shininess`, so MuJoCo's defaults (0.5 and 0.5) apply and the effective highlight is `0.3 × 0.5`. Ambient plus diffuse comes to 0.8, and that 0.2 of headroom absorbs it. Raising either term without lowering the other starts clipping.

The headlight is not head-mounted here: `mjv_updateScene` bakes it into `mjvScene.lights[0]` from the `mjvCamera` it is passed, and this app passes a fixed `mjv_defaultFreeCamera`. Scene-authored `<light>` elements are placed correctly by `mjv_updateScene` and are the supported fix.

Visibility is `model.geom_group`, from Python. Group 2 draws and group 3 does not (`mjv_defaultOption`), so hiding a tool is one write to a slice of `geom_group`. A hidden geom never becomes an `mjvGeom`, so it never writes depth — which is why this is a group switch and not an alpha.

Pass MuJoCo an absolute scene path. Measured on mujoco 3.11.0, a relative model path mis-composes an `<include>`d file's paths and fails with `Error opening file '<a path that exists>'`; with the follower's nested include it composes the directory onto itself and opens `<dir>/<dir>/so101_new_calib.xml`. `assets.ensure_so101_scene()` returns an absolute path for this reason.

The 17 STLs are fetched, not vendored: `viz.robot.assets.ensure_so101_scene()` fetches them on the first run into `~/.cache/isaaccapture/so101-assets/`, checksum-verified against a pinned commit. Set `XDG_CACHE_HOME` to relocate the cache root. Nothing fetches at build time — an isolated PEP-517 wheel build must not reach the network. Everything lands flat in one directory, because MuJoCo drops an included file's own `meshdir`; `sts3215_03a_v1.stl` is fetched twice rather than aliased, because the leader fragment names its copy `STS3215_03a.stl`. The three MJCF wrappers (`follower_arm.xml`, `leader_gripper.xml`, `scene.xml`) are tracked package data re-copied into the cache on every call, so editing one takes effect on the next launch. `joints_properties.xml` is deliberately not fetched: upstream inlines its `<default>` block rather than `<include>`ing it.

The reBot's scene works the same way, in its own cache directory (`~/.cache/isaaccapture/rebot-devarm-assets/`, `ISAACCAPTURE_REBOT_ASSETS` to move it): `ensure_rebot_devarm_scene()` fetches MuJoCo Menagerie's `seeed_rebot_devarm` — 115 meshes and its MJCF, ~15 MB — plus the same ghost meshes, since every scene draws the ghost. The set is pinned by one `REBOT_MANIFEST_SHA256` over the sorted per-file digests rather than a 117-row table or a hash of the repo tarball: content, not archive framing, and 15 MB instead of the 400 MB a tarball costs. Which meshes to fetch is read out of the MJCF, so one added upstream cannot be silently left behind — the manifest digest is what pins the set. Menagerie's own `scene.xml` is not used; it has a ground plane, skybox and haze.

The reBot's 92 collision meshes are fetched and compiled although the twin never draws them: MuJoCo must resolve every mesh the MJCF names, and the file is not edited.

`follower_arm.xml` wraps upstream's `so101_new_calib.xml`, fetched verbatim and never edited. `so101_new_calib.urdf` is pulled too, as the source of the trigger's hinge and its 0..100° travel. Three of the leader's four meshes are leader-specific print parts; the fourth is the STS3215 servo, shared with the follower and not decoration — `wrist_roll` is a C-shaped bracket that wraps it.

The ghost declares two mocap bodies, the gripper and its trigger, because the trigger articulates. A mocap body must be a jointless child of the world, so the trigger cannot be a hinged child of the gripper: its angle would live in `qpos`, which nothing here writes.

The ghost is opaque, which removes the draw-order constraint and the ghost-writes-depth-into-the-reprojection-buffer concern. `scene.xml` already `<include>`s the leader last, which is the order a translucent draw would need (`mjv_updateScene` emits in geom-id order), but nothing asserts it: that assertion belongs with the first scene that drops the alpha back.

## Not verified anywhere in CI or on a developer desktop

Everything downstream of the readback: `ProjectionLayer.submit()`, the frame loop that sequences it, the OpenXR session the compositor and the trackers share, whether the runtime accepts the depth layer, and controllers on a shared session. Since the render loop moved onto its own thread, `xrSyncActions` also runs concurrently with `xrWaitFrame` / `xrBeginFrame` / `xrEndFrame` — legal per the OpenXR spec, unverified against CloudXR. How the ghost looks is unverified, and so is the grip-to-gripper calibration.

Controllers on a shared session have no precedent elsewhere in this repository: `xrAttachSessionActionSets` is legal once per `XrSession`, Teleop sidesteps it with `XR_NVX1_action_context`, and the one existing shared-session example (`examples/oglo_tactile`) exercises only Hand and Head trackers, which use no actions. Treat that as the likeliest first-run blocker.
