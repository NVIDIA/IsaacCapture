---
name: add-new-plugin
description: >-
  Plan and implement IsaacTeleop input-device integrations, choosing native reuse, typed push,
  optional hand injection, or external bulk media per stream. Use for assessing, designing,
  building, or testing support for new input hardware; output and haptic integrations are outside
  this workflow.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add an IsaacTeleop Input Device

Add only the acquisition and data-path pieces that IsaacTeleop does not already provide. Decide
the route independently for every stream from a device; a mixed device may reuse native tracking,
push typed measurements, and carry bulk media externally at the same time.

## Boundaries

- This workflow covers input devices. Do not improvise an output/haptic implementation. If one is
  requested, state that it is unverified here and point only to `HapticSink`, `HapticCommand`, and
  `TeleopSessionConfig(sinks=[...])` for separate investigation.
- Inspection is read-only. A planning request may create or update only the plan artifact. A plan
  marked `ready` records an approved design; it does not authorize implementation edits.
- Do not execute supplied vendor code, installers, binaries, or scripts merely to inspect them.
- Do not start or stop services, accept licenses, change system configuration, or terminate
  processes unless the user explicitly authorizes that action.

## Start With Current Evidence

1. Locate the IsaacTeleop repository root and identify every path that may be touched.
2. Before editing, read the root `AGENTS.md` and every applicable `AGENTS.md` on those paths.
3. Inspect the current source, tests, CMake, repository documentation, user evidence, and official
   device/protocol documentation. Re-resolve every example path and referenced API before copying it.
4. List the device's input streams and check whether the runtime, an existing plugin, tracker, or
   source already exposes each required meaning.

If an evidence gap could change the route, schema, safety boundary, or test contract, use the short
interview loop in `references/plan-device.md`; do not guess.

## Choose One Route Per Stream

```text
Existing runtime/tracker exposes the exact required semantics?
├─ yes → reuse it; no plugin work for this stream
└─ no
   ├─ Native OpenXR hand behavior explicitly required?
   │  └─ yes → verify a truthful joint/frame/time mapping, then inject; no device-specific push schema
   ├─ Bulk video, depth, or audio?
   │  └─ yes → keep payload outside the retargeting graph; schema only required correlation metadata
   └─ typed push
      ├─ Existing schema is an exact semantic contract → reuse the schema
      └─ otherwise → create the smallest lossless typed payload and timestamped Record root

For typed push:
ordinary SchemaPusher/SchemaTracker collection → manifest-generated tracker
xrLocate, opaque channel, vendor facade, or multi-endpoint reader → current hand-written path
```

An exact schema match includes meaning, units, coordinate frame, validity, timing and freshness,
cardinality, and consumer needs. Similar fields or byte layout are not enough. Reusing a schema
also does not prove that its source or consumer fits.

## Plan

When no approved plan exists, read [references/plan-device.md](references/plan-device.md).

- Start from `assets/device-plan.template.yaml`.
- Write the plan to the requested path, defaulting to
  `src/plugins/<device>/device.spec.yaml` for an in-repository integration.
- Keep every stream-contract field and all seven pipeline nodes from the master template. Use the
  closest example under `assets/device-plan-examples/` only for its route decision and node pattern.
- Treat `pipeline` as the device-level implementation plan. For multiple streams, fold their routes
  into the same seven nodes and name the affected streams in each node's `reason` and
  `verify.expected`.
- Keep `status: draft` while assumptions, pointers, or decisions remain unresolved.
- Review the proposed routes and repository touchpoints with the user before marking it `ready`.
- Stop after planning unless implementation is explicitly authorized.

## Build

For an approved plan and an implementation request, read
[references/build-device.md](references/build-device.md).

- Implement only nodes marked `create` or `modify`. Keep `reuse` and `not_applicable` nodes in the
  plan so the complete data path remains visible, but do not edit their implementation files.
- Do not implement a plan while any node or verification action is still `undecided`.
- Match the closest maintained sibling and preserve IsaacTeleop's organized, clean, minimal, and
  readable style.
- Follow every node's `verify` block. Derive acquisition tests from device evidence; a generic
  repository test is not proof of a new device protocol or SDK path.
- For a plugin-backed typed or hand-injection route, adapt `assets/device_live.py.template` into
  `examples/<device>/python/<device>_live.py`. Fill its planned plugin identity, configuration, test
  duration, and polling interval, then adapt its three device hooks: pipeline construction, copied
  value extraction, and verification. Run bounded `test` mode only when the runtime lifecycle and
  device or simulator are available and authorized; use `live` only for human inspection.
- For CloudXR lifecycle, physical-headset-free browser emulation, headless-mode questions, or
  runtime diagnosis, read [references/troubleshoot.md](references/troubleshoot.md) only when needed.

## Handoff

Report:

- the route and schema decision for each stream;
- files and symbols changed, or why no plugin was needed;
- exact commands run and observable results;
- checks not run, why they were unavailable, and the exact next command or condition needed.

Never describe an unexecuted check as passed. Treat an unexplained runtime failure as unresolved,
not as proof that the implementation or assertion is wrong.
