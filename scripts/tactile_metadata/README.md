<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile Metadata Editor

Tactile Metadata Editor is a local Viser application for creating, editing, and
previewing glove tactile metadata. It loads URDF models and
[`TactileMetadata`](../../src/core/schema/fbs/tactile_metadata.fbs) JSON files.
All edits remain local; live connections only receive Isaac Capture data.

## Set up and launch

Use an existing Isaac Capture environment with Python 3.11–3.13. Install the
shared UI requirements into that environment:

```bash
cd scripts/tactile_metadata
python -m pip install -r ../../src/core/python/requirements-ui.txt
python -m tactile_metadata_editor.cli
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). To use another port:

```bash
python -m tactile_metadata_editor.cli --port 8090
```

Offline editing uses Viser, NumPy, yourdfpy, trimesh, and rtree. Live reception
also requires the Isaac Capture SDK, a compatible OpenXR runtime, and an
available HMD system in the same Python environment.

## Data and coordinates

The editor maintains independent left- and right-hand documents. **Toggle Side**
selects the active document; each side has its own metadata path and data.
General controls are at the top of the right menu. The Selected Object panel is
always present on the left.

Three pose datasets are displayed:

- **Editing** contains editable segment and taxel binding poses plus Edit-mode
  URDF joint values.
- **Saved** is the last loaded or saved metadata snapshot. Enable **Show Saved
  Binding Poses** to render it at opacity 0.2.
- **Preview** contains current segment poses and Preview-mode URDF joint values.
  Preview taxel poses are derived from the selected binding metadata.

All UI values and scene interactions use visualization-space coordinates.
**Segments Alignment** maps metadata wrist coordinates into visualization space.
Loading applies that transform; saving applies its inverse. Changing the
alignment transforms the visible editor values without changing the underlying
metadata coordinates.

## Edit metadata

The **Edit** mode contains three tabs.

### URDF

1. Enter an absolute **URDF Path** and select **Load URDF**.
2. Use **URDF Alignment** to translate, rotate, and scale the model.
3. Expand **URDF Joints** to adjust each movable joint.

URDF configuration is session-only and is not saved in tactile metadata. Its
base opacity is 0.4, multiplied by the opacity of the dataset layer being
rendered.

### Segments

- Use **Segments Alignment** to align all metadata poses.
- Add a segment from a HandJoint and XYZ position, or select **Add with URDF
  Click** and click the model surface.
- Segments are represented by yellow spheres with radius 0.005 m.
- Expand **All Segments** to edit HandJoint, position, and XYZ rotation.
- Drag the handle above a segment to reorder it. Displayed IDs follow list order.

HandJoint values must be unique. Taxel bindings use HandJoint names internally,
so segment reordering does not change their semantic references. Removing or
renaming a referenced segment leaves the affected taxel field invalid and
outlined in red.

### Taxels

Set Rows, Columns, Area Width, Area Height, Center Position, and Rotation.
**Capture URDF Surface Pose** copies a clicked surface position and orientation
into those fields without creating taxels. **Add Taxels** then creates the
row-major planar grid. Taxels are light-grey squares with side length 0.0025 m.

To bind taxels:

1. Select **Start Binding to Segments** and select taxels in the scene.
2. Select **Select Segments** and choose one to four segments.
3. Select **Complete**. Initial active weights are equal.

Expand **All Taxels** to edit pose, four segment slots, and four weights. `None`
is an editor-only inactive slot with weight zero. Editing a weight preserves the
entered value and adjusts other active weights proportionally so the row sums to
one. If only one slot is active, its weight is one.

Dragging taxel rows changes their IDs and metadata array order.

## Scene selection

- Click an object to select it.
- Shift-click adds objects of the same type.
- Shift-drag selects a screen rectangle and temporarily disables camera dragging.
- A selection cannot contain both segments and taxels. If an initial rectangle
  contains both, taxels take priority.
- Clicking empty scene space clears the selection.

The **Selected Objects** panel edits position and rotation, deletes selected
objects, and scales multi-selection positions about their center. Edit mode
changes Editing data; Preview mode changes Preview segments. Preview taxels are
derived and cannot be selected. With no selection, the panel remains visible,
shows Type `None` and Count `0`, and disables its input fields.

## Save and load

Enter **Metadata Path**, then use **Load** or **Save**. Saving validates the
current document. If errors remain, the editor asks whether to continue; invalid
drafts can still be written.

**New** resets the active side and initializes all OpenXR HandJoint segments in
the neutral hand pose. The left-hand defaults are mirrored before conversion to
visualization space.

The JSON contains only schema fields:

- `geometry_mapping_type` and `geometry_mapping`
- `segment_source_type` and `segment_source`

Editor HandJoint references are converted to `taxel_segment_indices` when
saving and restored when loading. Pose orientations are XYZW quaternions, and
taxel array order is the tactile force order. URDF paths, alignments, joint
values, selections, and live settings are not persisted.

The tool does not provide SDK binary export, a separate project format, or
standalone check/export CLI commands.

## Preview and live data

Use **Restore to Editor Binding Values** to initialize Preview from Editing.
Choose **Binding Metadata Source**:

- **Editing** uses the current editor bindings.
- **Live** becomes available after metadata is received from the tactile stream.

The Preview Static section lists Segments before URDF Joints.

**Apply Preview Data to Editor** fully replaces Editing segments, taxels,
bindings, and URDF joint values. The button is disabled when Preview segments
are incompatible with the selected bindings.

For live reception:

1. Set **Tactile Collection ID**.
2. Optionally enable **Use JointState to Drive URDF**, set its collection ID,
   and provide mappings:

   ```json
   {"source_joint": ["urdf_joint", 1.0, 0.0]}
   ```

3. Select **Connect Live**.

HandPose updates Preview segments. Enabled JointState mappings update Preview
URDF joints. Live frames overwrite manual Static values while connected.
Tactile force magnitude maps from blue at 0 N to red at 30 N, with values above
30 N clamped. Missing or inactive force data is light grey.

For multi-segment LBS, taxel orientation uses polar decomposition of the blended
rotation. Single-segment bindings use their direct rigid transform; singular
blends fall back to the taxel bind rotation.

## Implementation

- `document.py` defines editor-native segments, taxels, validation, and per-side
  state.
- `storage.py` converts between editor data and schema JSON and performs atomic
  saves.
- `geometry.py` handles URDF loading, scene transforms, picking, grid creation,
  HandPose conversion, and LBS.
- `sdk.py` receives HandPose, TactileMetadata/TactileDataArray, and optional
  JointState in a child process started with the launching interpreter.
- `ui.py` owns Viser controls, scene state, selection, and Edit/Preview workflow.
- `editor_interactions.js` supplies list drag-and-drop and invalid-field styling.

Local source checks:

```bash
ruff format --check tactile_metadata_editor
ruff check tactile_metadata_editor
python -m compileall -q tactile_metadata_editor
```
