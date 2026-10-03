<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Build an Approved Input Device Integration

Use this reference only when the device plan is `ready` and the user has explicitly authorized
implementation. Implement the approved touchpoints; return to planning if repository evidence
invalidates a decision.

## Preflight

1. Work from the IsaacTeleop repository root and inspect the current worktree without disturbing
   unrelated user changes.
2. List every directory the plan may change. Read the root `AGENTS.md` and every `AGENTS.md` on
   each path before editing. Read `cmake/cmake-structure.md` before touching CMake, headers, include
   paths, or layout.
3. Confirm no node or verification action is `undecided`. Resolve every planned `reference_file`,
   file, referenced API, command, and test name against the current checkout. Replace stale pointers
   in the plan before using them.
4. Confirm dependency licenses, versions, discovery, build gates, and missing-dependency behavior.
   Do not download, install, execute, or accept terms beyond the user's authorization.

## Coding Style

IsaacTeleop code is organized, clean, minimal, and readable.

- Match the closest maintained sibling before adding a layout, abstraction, target, or name.
- Keep acquisition/session orchestration thin. Isolate protocol parsing, decoding, mapping, and
  conversion in small focused functions that hardware-free tests can call.
- Prefer explicit names, typed constants, direct control flow, and narrow ownership over cleverness
  or premature generalization.
- Comment only non-obvious constraints, measurements, and traps. Do not narrate the code or its
  history.
- Follow the canonical CMake/include layout: one target per leaf, private headers beside source,
  exported headers only under `inc/`, no `../`, and repository include order.
- Mirror source ownership under `tests/`. Use SPDX headers, current repository formatting, and the
  exact checks required by applicable `AGENTS.md` files.

Avoid unrelated cleanup. Generated files are outputs, not places to patch.

## Implement the Approved Touchpoints

### Acquisition plugin

Match the closest plugin under `src/plugins/` for its target, manifest, install rule, dependency
gate, and error handling. Keep transport ownership and shutdown explicit. Extract decoding and
mapping from device I/O so malformed packets, conversions, and boundary values can be tested
without hardware, OpenXR, a daemon, or a vendor SDK.

Include the plugin leaf `CMakeLists.txt` and its guarded `add_subdirectory(...)` registration in
the repository-root `CMakeLists.txt`; plugin directories are registered explicitly, not globbed.

The plugin must fail clearly when a required dependency or device is absent. Do not silently use a
different device, collection, frame, or unit. Keep plugin, tracker, and source collection identities
consistent with the approved plan. Implement the approved source and delivery update rates; do not
copy a sibling's rate or use the live script's polling interval as the plugin rate.

### Schema and bindings

For a new typed contract, add the smallest `.fbs` payload and timestamped Record root approved in
the plan. Follow the closest schema and schema tests. Add the current hand-written schema binding,
registration, CMake wiring, and Python export; Python should expose the serialized table API, not a
parallel mutable FlatBuffers `-T` type.

Reconfigure the build after adding or changing a `.fbs`. Do not evolve a shared table, change field
IDs, or widen its meaning unless the plan explicitly treats that as a repository-wide API change.

### Tracker

For an ordinary schema collection, edit `src/core/deviceio_trackers/trackers.toml` and let the
configure-time generator create the facade, live/replay implementations, factories, recording
traits, pybind block, and `isaacteleop.deviceio_trackers` export. Add no duplicate hand-written
rows or Python compatibility exports. Inspect the resolved manifest and generated output in the
configured build tree when names or compilation disagree.

Use hand-written tracker code only for the approved special mechanism. Follow
`docs/source/references/generated_trackers.rst` plus the applicable tracker `AGENTS.md`; keep
OpenXR types out of `deviceio_trackers` and preserve the current time, recording, and snapshot
contracts in live/replay implementations.

### Source, boundary, and consumer

Add or change these only when the approved consumer cannot use a current path. Keep conversions in
the source or retargeter that owns their meaning and test them directly. An `IDeviceIOSource` is
discovered only when reachable from a declared `OutputCombiner` output; a side-effect-only source
needs an output such as the established heartbeat pattern and must be included by the combiner.
Do not confuse this with `IDeviceIOSink`, which is registered through
`TeleopSessionConfig(sinks=[...])` and is outside this workflow.

## Optional Hand Injection

Use this path only when native OpenXR hand behavior is explicitly requested and approved.

- Follow `src/plugins/plugin_utils/hand_injector.cpp` and the closest maintained injection plugin.
- Map only joints the device truly measures or can defensibly derive.
- Leave unavailable joint slots invalid; never present placeholders as tracked measurements.
- Apply the approved device-to-OpenXR frame, handedness, units, quaternion, and wrist/root policy.
- A controller or other wrist source is optional, not universal; use it only when the plan requires
  and validates it.
- Convert device/host time through the current runtime time path; do not pass clocks interchangeably.
- Update on a stable tick and track freshness independently per side. A stale side becomes inactive
  without erasing a fresh side.
- Do not keep stale poses active, exit from callbacks, or continue using invalid runtime/session
  state.
- Runtime creation, waiting, and teardown must be bounded where needed and remain within the user's
  service/process authorization.

## Tests

### Node verification

Treat the plan's seven `verify` blocks as the test plan. Process nodes in order:

1. Implement production files only when the node action is `create` or `modify`. For `reuse`, inspect
   `reference_file` but do not edit it. For `not_applicable`, make no implementation change.
2. Create or update test and registration files only when `verify.action` is `create` or `modify`.
   When it is `reuse`, run the directly relevant existing test in `verify.reference_file`. If direct
   coverage is missing, implement the focused test already planned in YAML or report the blocker.
3. Run `verify.run` and compare the result with every `verify.expected` item. Stop and diagnose a
   failure before moving on. A `not_applicable` verification has a null command and explains the
   bypass in `verify.expected`.

There is no universal unit-test template. Inspect the current tests owned by the changed subsystem
and follow their language, framework, fixtures, registration, and assertions. Use checks that
remain active in release builds. A test should prove only behavior established by the stream
contract and device evidence; do not invent packet layouts, checksums, calibration, validity,
freshness, or timing rules.

For `node0_acquire`, a new or modified device path needs a device-specific test under
`tests/cpp/plugins/<plugin>/` or the repository area that owns the acquisition logic. Derive its
fixtures and expected results from the user-provided protocol, SDK, API, sample data, or device
repository. Generic IsaacTeleop tests may show coding style, but they are not verification evidence
for an unknown device and must not be cited as such in the YAML. Include the leaf and parent test
registration files in `verify.files`, then build the test target and use CTest with
`--no-tests=error`.

For later nodes, start from the closest maintained tests only when their ownership matches:

- schema: `tests/cpp/core/schema/test_se3_tracker.cpp` or the closest schema sibling;
- tracker: `tests/cpp/core/replay_deviceio_session/test_replay_session.cpp` and
  `tests/cpp/core/mcap/test_mcap_tracker_channels.cpp`; test the generator itself only when its
  behavior changes;
- bindings: `tests/python/core/schema/test_se3_tracker.py` or the closest bound schema;
- source: `tests/python/core/retargeting_engine/test_joint_state_source.py` or
  `test_sources.py`;
- boundary: `tests/python/core/retargeting_engine/test_tensor_group.py`;
- consumer: `tests/python/core/retargeting_engine/test_joint_state_retargeter.py` or the closest
  owning retargeter test.

One test may support adjacent nodes when it contains distinct assertions for each contract, but
every node keeps its own `verify.expected` evidence and exact command.

### Live verification

For a plugin-backed typed or hand-injection route, adapt
`../assets/device_live.py.template` into
`examples/<device>/python/<device>_live.py`. It is an executable live check, not a CTest test.

Build the deepest self-contained, device-fed source/boundary/consumer path from the approved plan.
If a selected consumer requires external inputs, stop at the preceding self-contained prefix and
verify the remaining pure computation in its node test. Configure the plugin through
`TeleopSessionConfig` with `PluginConfig(required=True)` so `TeleopSession`
creates OpenXR and DeviceIO before starting the plugin and owns reverse-order cleanup. Follow the
closest maintained `*record.py` pipeline rather than adding raw tracker/session orchestration.

Fill every plugin instance's name, root ID (including a deliberate empty value), arguments,
candidate search paths, test duration, and polling interval. The polling interval controls only
observation and printing, not device acquisition or plugin publication. The scaffold starts with
one `PluginConfig(required=True)`; add another entry for each planned instance. Pass candidate
search paths directly because `TeleopSession` selects existing paths and reports a missing required
plugin. Then adapt three device hooks: pipeline construction, copied-value extraction, and
verification. If the approved device needs dynamic arguments, change `_session_config` rather than
adding route branches.

- `test` collects and prints available values for one monotonic bounded duration, fails if none
  arrive, and passes the values to the device-specific `verify` hook. The external timeout must
  exceed runtime and session setup, the test duration, and cleanup margin.
- `live` prints available values until Ctrl+C for human debugging; it is not verification evidence.
- If the contract requires activation timing, host timestamps, dropout detection, or source/delivery
  update-rate verification, extend the generated device script with only that evidence-driven check;
  use contract-defined timestamps, sequence changes, or freshness rather than polling-loop counts,
  and do not add it to the universal template.
- Do not add route switches, generic sample-only passes, simultaneous-hand requirements,
  wrong-collection probes, stop/inactive checks, MCAP output, visualization, or robot actuation
  unless the approved contract specifically requires them.

For `native_reuse`, reuse a maintained native live example instead of generating a plugin check.
For `bulk_media`, plan a device-specific bounded sink/artifact check because `TeleopSession` does
not carry the bulk payload. A required plugin-stop/inactive test remains a separate low-level check
while the reader is open.

Never start or stop a shared service, accept a license, delete runtime markers, or kill unrelated
processes. Use `--no-launch-cloudxr-runtime` for a planned external runtime. The ordinary
`--launch-cloudxr-runtime` path attaches or starts a detached service that persists after the
script; it is not process-owned cleanup. Use it only for an approved
`attach_or_start_detached` lifecycle. For approved process ownership, set `embedded` in the plan,
adapt the call to `launch_context(args, run_embedded=True)`, and invoke it with
`--launch-cloudxr-runtime`. Never add `--accept-eula` without separate explicit authorization. If
runtime or device use is unavailable or unauthorized, create the script but report it unrun with
its exact `test` command.

A failure may be in the test, implementation, runtime, hardware, or infrastructure. Preserve the
evidence and diagnose the boundary before changing code or weakening assertions.

## Repository Gates

Run the smallest relevant checks for each node, but never run pre-commit from a node `verify`
block. After every node, live check, full build/test, and formatting step is complete, run
pre-commit as the final validation command. Resolve `<build-dir>` and actual test names from the
current configuration.

```text
cmake -B <build-dir> <current required options>       # when configure/codegen inputs changed
cmake --build <build-dir> --target <target>
ctest --test-dir <build-dir> -N                       # confirm the test is discovered
ctest --test-dir <build-dir> --output-on-failure -R <actual-test-name>
# format/lint touched code as required
cmake --build <build-dir>                             # whole configured repository
ctest --test-dir <build-dir> --output-on-failure      # configured suite
SKIP=check-copyright-year pre-commit run --all-files  # final validation command
```

Use the formatting commands and exact pre-commit command required by the current applicable
`AGENTS.md`; the command above reflects the current root rule. If pre-commit changes a file, review
the change, rerun every affected earlier gate, then obtain a clean pre-commit pass last. If a
required check cannot run, report it as unverified rather than changing build policy to bypass it.

## Handoff

Report the per-stream route and schema choice, changed files and symbols, exact checks and results,
runtime evidence, and every unverified item with its blocker and closure command. Do not claim a
test passed because it exists, was discovered, compiled, or passed in a different environment.
