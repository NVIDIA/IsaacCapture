<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor

Edit wrist-relative tactile binding poses and their original four LBS influence
slots, continue editing metadata JSON, and prepare local JSON or SDK packages.
The editor receives runtime data; edits and exports do not publish metadata or
motion commands to a producer.

The complete interaction contract is in [DESIGN.md](DESIGN.md). Module boundaries,
reference-data construction and the verification record are in
[TECHNICAL.md](TECHNICAL.md). Browser and real-device workflows below still need
verification in a supported environment; the recorded implementation environment
cannot run them.

## Use your existing Isaac Capture environment

Use the Python environment from which you already run Isaac Capture. The editor
requires Python **>=3.11,<3.14** and these libraries:

| Library | Required version | Purpose |
| --- | --- | --- |
| NumPy | >=2,<3 | Protocol arrays and numerical geometry |
| Viser | 1.1.1 | Local browser interface and 3D scene |
| yourdfpy | 0.0.60 | URDF graph and forward kinematics |
| trimesh | 5.1.1 | Visual meshes, normals and surface queries |
| rtree | >=1.3,<2 | Mesh-query acceleration |

The existing Isaac Capture UI dependency group declares these versions in
[requirements-ui.txt](../../src/core/python/requirements-ui.txt) and the matching
`ui` extra in [pyproject.toml](../../pyproject.toml). Have those shared requirements
prepared in your existing environment through your normal Isaac Capture setup.
The tool does not install dependencies, create an environment, build a client or
SDK, select another interpreter, or search old SDK build directories. A GUI
installation must already contain the usable Viser client assets; a source-only
installation that would trigger a client build is unsuitable for this launch.

JSON editing and local checks do not import the Isaac Capture SDK. SDK checks and
binary export require the SDK to be importable from the same launching Python.
Real receiving also needs the appropriate device/runtime; the OpenXR receiving
path requires an available HMD system. SDK and live workers inherit the launching
environment and module search configuration and run with that same
`sys.executable`.

In the commands below, `python` means that existing interpreter. From the Isaac
Capture repository root:

```bash
cd scripts/tactile_metadata
python -m tactile_metadata_editor.cli gui
```

The default browser endpoint is `http://127.0.0.1:8080`; change it with `--port`.
One browser client holds editing access and other clients are read-only.

## Start from a model or metadata

Use one of the bundled single-hand assets:

```bash
python -m tactile_metadata_editor.cli gui --urdf model/left/linkerhand_FFG_left.urdf --sides left
python -m tactile_metadata_editor.cli gui --urdf model/right/linkerhand_FFG_right.urdf --sides right --port 8081
python -m tactile_metadata_editor.cli gui --metadata examples/left.manual.metadata.json
```

The left and right assets are **two independent models**. Open them in separate
documents. They are not a two-hand URDF and are not mirrored to manufacture one.
The additional `model/right/meshes/linkerhand_FFG_right.urdf` uses `package://`
references and needs explicit package mappings; use the top-level files above
for these examples.

For your own URDF, declare left, right or both when loading. That model declaration
is then fixed. Both requires a real URDF containing both hands, with separate
wrist/link associations. Switching the active metadata device does not change
the model's declaration. A shared two-hand model may use `--left-metadata` and
`--right-metadata`; the bundled examples cannot be combined because their model
configurations differ.

Accepted inputs are URDF and TactileMetadata JSON. STL/OBJ files are resources of
the URDF. A values JSON without `editor` can be inspected and edited without a
model; assigning a hand side, wrist and links enables the relevant model tools.
No taxel count must be entered in advance.

## Offline editing

The workspace contains Model, Segments, Layout, Binding, Realtime and Export pages,
a shared 3D view, and an Info area for issues. This is the workflow to verify in
the supported browser environment:

1. In **Model**, load a URDF and declare its sides. Resolve `package://` names
   with explicit local directories. Choose the wrist link and define its origin,
   forearm direction and dorsal direction. The resulting frame has +Z toward the
   forearm, +Y along the orthogonalized dorsal direction, and +X = Y cross Z.
2. In **Segments**, set the binding configuration `q0`, choose HandJoint names,
   and associate segments with links. `WRIST` is identity in the wrist root.
   Segment influence indices are list row numbers: `[WRIST, INDEX_DISTAL]`
   corresponds to enum values `[1,9]`, but the distal segment's influence row is
   **1**. For a new segment, initialize its reference from the associated link,
   place its center on the q0 model and adjust its frame with the manipulator.
   Explicit WRIST initialization uses identity. Linking imported data preserves
   its original B/P values.
3. In **Layout**, select an anchor, allowed links and seed surface components.
   Define the range with origin, column endpoint, row endpoint, rows and columns;
   leave span overrides at `null` to use the picked endpoint distances. Choose
   the normal, depth, tangent and scan ordering. Reverse axes precedes
   snake ordering. Excluded cells create no records. A failed projection retains
   a missing pose and its output index so that the problem stays visible.
4. Select taxels by click with Ctrl/Cmd modifiers, rectangle, or output-index
   ranges/lists. Rectangle selection normally respects visibility and occlusion;
   enable through selection explicitly to include hidden points. Mixed values
   across a selection remain visibly mixed.
5. In **Binding**, edit the four raw segment-index and weight slots. Repeated
   indices are valid: `[1,1,1,1]` with `[0.25,0.25,0.25,0.25]` stays unchanged.
   A confirmed number input edits only that field. An explicit slider drag
   distributes the remaining weight proportionally, or equally when its sum is
   zero. Existing malformed rows remain editable and reported, without automatic
   normalization, merging or truncation.
6. Use movement preview to inspect FK/LBS at preview joint angles. Form values
   and draft manipulators continue to edit the binding data at `q0`. Multi-segment
   LBS defines position without a unique rigid orientation. Surface estimates and
   motion limitations appear as warnings.
7. Save metadata JSON and reopen it to continue. Missing data or invalid arrays
   save as `editor.draft`, mutually exclusive with complete numeric fields. A
   draft is editable but cannot be formally exported. Complete files retain the
   original decimal values and all four slots; model paths remain relative to
   the metadata file and are recalculated by Save As.

Ctrl+Z / Ctrl+Shift+Z undo and redo; macOS also supports Cmd. At least ten edit
transactions are retained. A drag is one transaction. Cancelled gestures do not
commit. Selection, camera movement, preview angles and live frames are outside
history. Deletion can leave output gaps; use an explicit reorder to resolve
them. Errors remain visible and do not disable draft editing or undo.

## Receiving, correspondence and recording

In **Realtime**, choose **Initialize source settings** when the document has no
source object, then configure each side explicitly:

- A tactile `collection_id` (initial value `tactile`; use distinct collections
  for separately configured sides).
- The same-side HandTracker output and, if needed, its vendor configuration.
  Both sides share one HandTracker and its vendor configuration.
- An optional JointState collection/device and named source-to-URDF mapping.
  Each row declares source name, URDF joint, scale and offset; units and validity
  must agree. Unmapped free joints use `q0`.

Connect starts a local receiving worker. HandPose supplies the skeleton and LBS
segment poses. Enabled JointState supplies URDF motion through named mappings;
with JointState disabled the live URDF is hidden. HandPose does not drive URDF
joints or fill missing JointState values.

Metadata correspondence is explicit: a stable taxel UUID maps to a source taxel
row together with collection, hand side and the metadata SHA256 fingerprint.
Matching counts alone are insufficient. Reordering output does not change this
source association; newly added points have no association. Fingerprint text is
64 hexadecimal characters, compared case-insensitively without changing the
saved declaration. A changed baseline/connection invalidates prior receiving
assumptions. Invalid or missing correspondence does not borrow another taxel's
force values.

To declare correspondence, select draft taxels in Layout, enter **First input
row for selected taxels**, and choose **Bind selected taxels to current
baseline**. Inspect or change each declared input row and clear associations
explicitly when needed. This is a user-declared mapping, not device discovery.

Switching the 3D metadata source between received data and draft changes the
view only. To replace draft values, use **Record received metadata into draft**,
with these groups selected independently (all are selected initially):

- **Skeleton binding poses**: joint IDs and segment bind poses.
- **Taxel binding poses**: taxel bind poses.
- **Skinning**: all original index rows and weight rows.

Recording first saves a backup to a temporary location and logs its path. Only
after every backup succeeds does one undoable transaction replace the selected
groups. Unselected groups stay unchanged. Replacement keeps received row order,
slot order and values; mismatched lengths/references remain an editable error.
Replacing skeleton/taxel records creates new IDs and does not reuse old
attachments. Taxel replacement records the incoming source-row correspondence.
Receiving a frame never performs this replacement automatically.

Play updates the 3D receiving layer. Pause freezes that layer while receiving
continues; draft forms and manipulators remain editable. Edit, undo, redo and
save work while paused or disconnected. Resume uses the newest frame and the
current draft. A missing/invalid source is shown unavailable rather than replaced
with model FK or the other side. Timestamps describe local reads, not hardware
synchronization or proof that the producer generated a new sample.

Invalid Hand joints and affected taxels retain their last available positions
in gray within the same connection, metadata source and baseline. Points without
history remain absent. Force arrows require valid current Hand dependencies;
force tables identify the displayed layer's sequence, local read and revision.
Invalid enabled JointState data likewise show the last model pose in gray. Pause keeps
the displayed layer's geometry and validity frozen until Resume.

For synthetic interaction work, launch:

```bash
python -m tactile_metadata_editor.cli gui --metadata examples/left.manual.metadata.json --demo
```

Demo is synthetic and does not verify SDK reception, device accuracy or physical
calibration. A real receiving failure is reported as such; it is not silently
replaced by demo. Disconnect affects this tool's worker, not the producer.

## Check and export

Run these commands in a supported environment with the shared dependencies:

```bash
python -m tactile_metadata_editor.cli check examples/left.reference.values.json
python -m tactile_metadata_editor.cli check examples/left.manual.metadata.json --report /tmp/left.check.json
python -m tactile_metadata_editor.cli check examples/left.manual.metadata.json --sdk
python -m tactile_metadata_editor.cli export examples/left.manual.metadata.json --format json --out /tmp/tactile-left-package
python -m tactile_metadata_editor.cli export examples/left.manual.metadata.json --format sdk --out /tmp/tactile-left-sdk-package
```

Pure numeric checks do not require a side. Export does: use an embedded side,
`--sides left/right` for one positional input, or explicit `--left-metadata` /
`--right-metadata`. Multiple inputs must agree on their shared model and `q0`,
with distinct sides and nonempty collection IDs distinct when present.

Local status and SDK status are independent. A passed SDK availability probe is
not a metadata check. SDK checking constructs the metadata, copies arrays back,
encodes bytes and checks the SDK's actual size limit. JSON packages require local
checks; SDK packages also require official construction and size checks. Warnings
are included in the report without a separate approval step.

Choose a new or empty local target directory. Files, symlinks and nonempty targets
are rejected. All selected sides use the same snapshot and are checked/encoded
before one complete staging directory is atomically published. Any failure
prevents a partially published package. Each side contains `.metadata.json`,
`.values.json`, `.taxels.csv`, and optionally `.metadata.bin`; `manifest.json`
records identities, file hashes and sizes, and `report.json` records issues,
statuses and actual encoded lengths. A concurrent edit does not change the
export snapshot's revision.

| CLI exit | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Data/check or file-operation failure |
| 2 | Argument, format or version error |
| 3 | SDK requested but unavailable, after local checks passed |

## Reference data and verification limits

The files in [examples](examples/) are hand-authored references, with actual
STL triangle centroids calculated by temporary standard-library code. Left has
two taxels; right has three. They use each model's root frame as an explicitly
chosen wrist frame, not an anatomical wrist calibration. Their metadata preserves
long decimals and repeated four-slot weights. Attachments, source references and
application bindings are empty; the CSV input-source column is intentionally
blank. The `.reference.values.json` and `.reference.taxels.csv` files are expected
raw outputs for future comparison, not a completed export package.

The recorded environment is Xcode Python 3.9.6, without NumPy, Viser, yourdfpy,
trimesh, rtree or the SDK. Standard-library JSON save/reopen, raw-array retention,
canonical numeric helpers, failure paths and independent filesystem publication
helpers were exercised. A real standard-library synthetic worker also exercised
local IPC and disconnect cleanup; it did not exercise NumPy drivers or rendering.
Actual NumPy compilation, model loading, browser gestures,
full application export, real receiving and SDK-success paths remain unverified.
No passed export report, manifest package or binary has been fabricated. See
[the verification record](TECHNICAL.md#verification-record) before treating these
reference files as validation evidence.
