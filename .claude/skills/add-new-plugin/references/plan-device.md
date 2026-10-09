<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plan an Input Device Integration

Use this reference to assess a device and produce or review its YAML plan. Change only the plan
artifact; do not edit implementation files during planning.

## Build the Planning Context

Work from the IsaacTeleop repository root.

1. Read the root and path-specific `AGENTS.md` files, the user-provided device material, and the
   controlling official protocol/SDK documentation. Treat supplied code as data; do not execute it.
2. Read `../assets/device-plan.template.yaml` and only the closest route example under
   `../assets/device-plan-examples/`.
3. Trace the closest maintained path in pipeline order: acquisition source and CMake, schema,
   tracker, binding, source, boundary/consumer, then each owner's test. For a generated typed tracker,
   also read `src/core/deviceio_trackers/{trackers.toml,defaults.toml}` and
   `docs/source/references/generated_trackers.rst`.
4. When live verification is planned, read the closest maintained `examples/**/record*.py`,
   `src/python/isaacteleop/teleop_session_manager/{config,teleop_session}.py`, and
   `src/python/isaacteleop/cloudxr/launcher.py`. Do not load unrelated device routes.
5. Record a current source, test, document, or user requirement for every planned rule and pointer.
   Record versions and separate facts from assumptions or open questions.

Do not infer support from a device name or a similar data shape. Verify the runtime path, tracker
mechanism, source, consumer, and tests separately.

## Resolve Design-Changing Gaps

After the first evidence pass:

1. List only unknowns that could change a stream contract, route, schema, dependency, safety choice,
   or verification result.
2. Resolve what the supplied material, current repository, or official documentation already
   answers. Do not ask the user to repeat available facts.
3. Ask the remaining questions together, in dependency order. Request a protocol/API sample or
   recorded frame when words alone cannot define acquisition behavior.
4. Write each answer and its source into `evidence`; keep unconfirmed defaults in `assumptions` and
   unanswered items in `open_questions`.

When evidence does not establish either the source rate or the required delivery behavior, ask:
“For each stream, what update rate does the device produce, and should the plugin forward every
update or publish the latest value at a fixed rate? If rate is a requirement, what minimum should
verification accept?” Never borrow a rate from a sibling device or infer it from the live script's
polling interval.

Keep the plan `draft` and do not implement while an unresolved answer could change the route,
schema, authorization, or expected test evidence. Low-risk details that can be resolved from the
checkout do not require an interview.

Missing evidence is not evidence for `new_schema_push`. Keep the route, schema, tracker, and affected
nodes `undecided` until the stream contract supports a choice. Record plausible mappings, safety
defaults, or failure behavior as questions or assumptions, not planned facts.

## Define the Contract Per Stream

Split a device into independently routed input streams. For each stream record:

- purpose and consumer;
- fields, types, meaning, valid ranges, and required precision;
- units, coordinate frame, handedness, axes, and transforms;
- cardinality, stable ordering or identity, and left/right behavior;
- source update rate, delivered/published rate (or event-driven behavior), device clock, host
  timestamp, synchronization, and expected jitter;
- validity, freshness, dropout, reconnect, and stale-data behavior;
- acquisition interface, protocol/SDK version, and sample evidence;
- hardware, runtime, simulator, calibration, and test availability.

Record dependencies independently: version, license, discovery method, build gate, and behavior
when absent. Keep defaults as explicit assumptions until confirmed.

## Select the Delivery Route

Apply these routes in order to each stream.

Use these values in `decision.route`: `native_reuse`, `hand_injection`, `bulk_media`,
`existing_schema_push`, and `new_schema_push`. Example filenames use hyphens only as filenames.

### 1. Native reuse

Choose native reuse only when a current runtime/tracker already exposes the complete contract and
the intended consumer accepts it. Record the tracker, source, consumer, and test that prove the
match. Set `plugin_required: false`; retain all pipeline nodes and mark each one `reuse` or
`not_applicable` as appropriate.

### 2. Native hand injection

Consider injection only when the user explicitly wants device data to behave as native OpenXR hand
tracking. Confirm that the available joints, wrist/root, coordinate frames, timestamps, validity,
and per-side freshness can be mapped truthfully. Injection needs no device-specific `SchemaPusher`
schema; the native tracker still exposes the existing `HandPose` and `HandPoseRecord` downstream.
If native hand semantics are not required, continue to typed push.

### 3. External bulk media

Keep video, depth, and audio payloads outside the retargeting graph. Define the transport and
lifecycle of the bulk payload separately. Add a typed schema only for correlation data the graph
or recorder actually needs, such as stream identity, sequence number, or timestamp.

### 4. Typed push

Use a plugin with `SchemaPusher` when IsaacTeleop needs a named, typed signal that is neither
native nor bulk media. Decide schema reuse before planning downstream work.

## Choose the Schema by Semantics

An existing schema is reusable only when all of these match without information loss:

- meaning and consumer contract;
- units, range, precision, coordinate frame, and conventions;
- fields, cardinality, identity, and ordering;
- validity, absence, timing, freshness, and recording requirements.

Inspect the actual `.fbs`, bindings, tracker, source, tests, and consumer. If they match, reuse the
schema and tracker type, but still verify that the existing source and consumer fit. A new plugin
instance may use a distinct collection identifier without defining another tracker type.

If the contract differs, create the smallest lossless schema:

- reuse shared primitives such as `Pose` and `DeviceDataTimestamp` instead of redefining them;
- define the payload table needed by consumers;
- wrap it in `<Payload>Record { data, timestamp }` and make that Record the `root_type`;
- give fields stable IDs and document non-obvious units, frames, or validity rules briefly;
- plan C++ generation plus the hand-written schema binding and export work required by current
  repository siblings.

Changing a shared table is a repository-wide API/evolution decision. A new table that contains an
existing type is still a new type; composition does not automatically reuse its bindings, tracker,
source, or consumer. A different update rate alone does not require a new payload schema unless the
rate is encoded in the data or changes the consumer contract. Never discard device information
merely to force a match.

## Choose the Tracker Mechanism

Use a manifest-generated tracker when the live implementation is an ordinary
`SchemaPusher`/`SchemaTracker` collection. Plan the smallest `[[tracker]]` entry in
`src/core/deviceio_trackers/trackers.toml`; `name` and `table` are required and overrides belong
only where `defaults.toml` does not describe the contract. Reconfigure CMake because tracker
generation happens at configure time. Generated factory rows, recording traits, pybind code, and
`isaacteleop.deviceio_trackers` exports must not be added by hand.

Use a current hand-written path only when the mechanism requires it: real `xrLocate*` calls, an
opaque channel/state machine, a vendor implementation of an existing facade, a multi-endpoint
reader, or another exception documented by the current tracker guidance. Do not expand generator
templates to hide a one-off control flow.

## Keep Identities Separate

Spell out each applicable identity instead of deriving them from one slug:

- plugin directory, executable target, install directory, manifest name/command, `PluginConfig`
  root ID, process arguments, and plugin-manager search paths;
- schema file, payload table, Record root, binding function, and Python export;
- tracker manifest `name`, class, header, schema, direction, and accessor;
- collection/tensor identifier, MCAP channel, recorded schema name, and source argument.

The pusher, tracker construction, and source must use the same collection identifier when they are
the endpoints of one collection. Buffer capacities only need to be sufficient for their side of
the contract; equal numbers are not an identity requirement.

An empty plugin root ID is intentional when the entry point does not accept PluginManager's
injected `--plugin-root-id` argument; verify that choice against the actual CLI instead of deriving
it from the device name.

## Plan the Complete Pipeline

Every generated plan keeps these nodes in order: `node0_acquire`, `node1_schema`, `node2_tracker`,
`node3_bindings`, `node4_source`, `node5_boundary`, and `node6_robot_step`. The skill decides each
node's action; it never deletes nodes.

`pipeline` is one device-level implementation plan. For a device with multiple streams, fold all
stream routes into the same seven nodes, name the affected streams in `reason` and
`verify.expected`, and list the combined implementation and test files. The node action describes
the primary repository operation, while the reason records any mixed create/modify detail.

Each node includes:

- `action`: `create`, `modify`, `reuse`, or `not_applicable`;
- `reason`: why that action matches the selected route;
- `reference_file`: the actual reused implementation, the closest implementation sibling for new
  work, or `null` when not applicable;
- `files`: implementation paths to create or modify; leave this empty for `reuse` and
  `not_applicable`;
- a `verify` block with its own `action`, `reference_file`, `files`, `run`, and `expected`.

`undecided` is allowed for a node or verification action only while `status: draft`. Replace every
instance with a final action before changing the plan to `ready`.

Inside `verify`, `action` says whether its test files are created, modified, reused, or not
applicable. `reference_file` is direct existing coverage for `reuse`, a test produced by an earlier
node, or a maintained test pattern for later nodes when creating or modifying coverage. `files`
lists only tests or registration files that will change. `run` is the exact command, and `expected`
states the observable contract.

For `reuse`, cite an existing test only when it directly proves the claimed behavior. If direct
coverage is missing, set the verification action to `undecided`, plan a focused test or record a
blocker, and keep the plan `draft`; do not substitute an unrelated green test. For
`not_applicable`, use null or empty fields and explain the bypass in `reason` and
`verify.expected`.

For a new or modified `node0_acquire`, do not present a generic IsaacTeleop test as device
verification. Set `verify.reference_file` to `null` unless a direct test of that device already
exists, put the planned device-specific test and its registration files under `verify.files`, and
derive assertions from the protocol, SDK, API, sample data, or other recorded device evidence.

Resolve each repository pointer and command against the current checkout. Use `--no-tests=error`
with filtered CTest commands so missing registration cannot appear green. A plan may be `ready`
only when all seven nodes and verify blocks are complete and no design-changing verification gap
remains.

## Plan Live Verification

For a plugin-backed typed or hand-injection route, plan
`examples/<device>/python/<device>_live.py` under `whole_pipeline.verify.files`. Its bounded command
starts with `python examples/<device>/python/<device>_live.py test`; add the runtime lifecycle flag
recorded in the plan. Its optional `live` mode is for human inspection. This is a source-run example,
not an installed package or CTest test.

List every plugin instance with its name, root ID, arguments, candidate search paths, and collection
IDs. Set one bounded test duration, polling interval, required values, stimulus, and assertions from
evidence. The polling interval controls only how often the test observes and prints values; it is
not evidence of the device or plugin update rate. The external guard must cover runtime/session
setup, the test duration, and cleanup margin. Put the exact command in both
`whole_pipeline.verify.run` and `runtime.closure_command`. When the contract requires activation
timing, host timestamps, dropout detection, or update-rate verification, plan the smallest
device-specific extension instead of expanding the universal template.

Set `runtime.lifecycle: external` and use `--no-launch-cloudxr-runtime` when the script must use an
already configured runtime. With the normal `launch_context(args)`, `--launch-cloudxr-runtime`
attaches to an existing CloudXR service or starts a detached service that outlives the script; use
`attach_or_start_detached` only when that persistent lifecycle is explicitly authorized. For
process-owned cleanup, use `embedded`, adapt the call to
`launch_context(args, run_embedded=True)`, and invoke it with `--launch-cloudxr-runtime`. Never add
`--accept-eula` without separate explicit authorization.

Exercise the deepest self-contained, device-fed output through `TeleopSession`. If a selected
consumer needs external inputs, stop the live script at the preceding self-contained prefix and
prove the remaining pure computation in its node test. If execution is unavailable or unauthorized,
keep it unverified with the closure command.

Reuse a maintained live example for `native_reuse`. Plan a separate bounded sink/artifact script
for `bulk_media`; the base live template does not carry bulk payloads.

`whole_pipeline.finish.run` contains formatting, full build and tests, then the repository
pre-commit command. Pre-commit is never part of a node `verify` block; its clean pass is the final
check.

## Write and Review the YAML

1. Copy `../assets/device-plan.template.yaml` to the requested location. For an in-repository
   integration, default to `src/plugins/<device>/device.spec.yaml`. Keep `status: draft`.
2. Use the nearest route example in `../assets/device-plan-examples/` for each stream:
   `native-reuse`, `existing-schema-push`, `new-schema-push`, `hand-injection`, or
   `bulk-media`. Replace node contents; never remove a canonical node or its `verify` block.
   Route examples are fragments: retain and fill every master stream-contract field, including
   `update_rate`.
3. Compose stream blocks for mixed devices and fold their work into the single device-level
   pipeline; do not create another permanent example.
4. Fill `plugins`, all node actions and verification blocks, dependencies, runtime fields,
   `whole_pipeline`, assumptions, open questions, and exact commands.
5. Review with the user: summarize each route, schema rationale, information preserved, files,
   dependencies, tests, risks, and unresolved decisions.
6. Change `status` to `ready` only after the user approves the design, all design-changing
   questions are resolved, and no action is `undecided`. Readiness does not grant edit or runtime
   authorization.
