<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Models for the 3D tracking view

Static models loaded once by `backend/egocap_web/viser_scene.py`. They are visual only; nothing here is tracking
calibration.

## `headset.glb` -- tracked head (generic headset)

- Source: the headset model in the WebXR samples (https://github.com/immersive-web/webxr-samples, path
  `media/gltf/headset/headset.gltf` and `headset.bin`), fetched from the `main` branch whose tip was
  `fb41c3416496974672e2c766a41c6febcba2d07e` on 2026-10-19.
- License: MIT, copyright 2018 The Immersive Web Community Group; the full text is in `LICENSE.headset.md`.
- Source file SHA-256 (gltf, bin): 6520272698fa459fd7eeb95c03f077fb716c7f3552dddee920d83b1921c59dc0 3f5edbcf8dfcdfb4dbec3c6f9ba245440ad485ac3433ca1f764d5b01143bd190
- Change: converted once to a self-contained GLB with `trimesh` (load the glTF scene, export GLB). That bakes the
  mesh node's transform (a uniform 0.015 scale and a rotation) into the geometry and drops the exporter's camera and
  lamp nodes. No other edits. The result is in meters, about 0.245 m wide, and authored in the head-pose frame
  (the WebXR sample assigns the tracked head pose directly to it), so the configured model correction is the
  identity.

## `pico-4/left.glb`, `pico-4/right.glb` -- controllers

- Source: the Pico 4 profile of the WebXR input profiles assets
  (https://github.com/immersive-web/webxr-input-profiles, `packages/assets/profiles/pico-4`), fetched from the
  `main` branch whose tip was `f4992299601614adbfefd398dc8e281556bb7444` on 2026-10-19.
- License: MIT, copyright 2019 Amazon; the full text is in `pico-4/LICENSE.md`.
- Change: the textures were downscaled from 1024x1024 to 256x256 with Pillow and the files re-exported once with
  `trimesh` (geometry and materials unchanged). This keeps each file near 240 KB instead of 770 KB, so the scene a
  browser receives when it connects stays well under 1 MiB, which some WebSocket proxies cap per message.
- SHA-256 of the files as shipped (left, right): a40aecedd7ad747c8f914c74743d9b75911808147041bf0a76e986f89d15723c 4aeb91d1059efe356e12e37b5bc0005a549a9bf62070785a0f79d6e80de80f76
- These are authored in the controller's grip space and follow the GRIP pose. They depict the Pico 4; if the
  station uses another model, the same repository has `pico-4u` and `pico-neo3` profiles.

## `pico-4-ultra/headset.glb` -- Pico 4 Ultra headset (procedural)

- Source: built by this project, not a scan or a vendor asset. No redistributable Pico 4 Ultra headset model with
  a permissive license was found, so this is a low-poly approximation composed with `trimesh` primitives from Pico's
  published dimensions (visor about 205 x 92 x 38 mm, pancake optics, two front RGB passthrough cameras and a depth
  sensor, headband with a rear battery pad). It is recognisably the device, not millimetre-accurate.
- License: same as this project; no third-party material.
- Frame: authored directly in the OpenXR head-pose frame (metres, Y up, -Z forward) with the eyes at the origin and
  the facial interface starting about 12 mm in front of them, so the configured model correction is the identity.
- Build script: `pico-4-ultra/build_headset.py` (run it with the project's Python and an output path to regenerate).

## `pico-4-ultra/left.glb`, `pico-4-ultra/right.glb` -- Pico 4 Ultra controllers

- Source: the `pico-4u` profile of the WebXR input profiles assets
  (https://github.com/immersive-web/webxr-input-profiles, `packages/assets/profiles/pico-4u`), fetched from the
  `main` branch whose tip was `f4992299601614adbfefd398dc8e281556bb7444` on 2026-10-26.
- License: MIT, copyright 2019 Amazon; the full text is in `pico-4-ultra/LICENSE.md`.
- Source file SHA-256 (left, right): febc63d7de3acb58dbbdc54759ab87a1c687ab6e1f23a76b8868af3edf97cf91
  fb3f09849cb1b9d3f4fa434240a57fb475d754fda386ea1b1d36db874f015f8c
- Change: textures downscaled from 512x512 to 256x256 with Pillow and the files re-exported once with `trimesh`
  (geometry and materials unchanged), as for the Pico 4 files.
- SHA-256 of the files as shipped (left, right): e10dff27f39f6530e6937a9d046962e5fdb4a6d53602e77a0f67402f472bac04
  e4eb175827a90dc6f450efe1b4fd8d407881405922117381b112aa2fe91f9698
- Authored in the controller's grip space and follow the GRIP pose.

## Which set is shown

`backend/egocap_web/config.py` maps a device profile name (`EGOCAP_VISER_DEVICE`, default `pico-4-ultra`) to one
headset and two controller files; `viser_scene.py` switches to another shipped profile at runtime when the capture
status names a different headset in a device description.
