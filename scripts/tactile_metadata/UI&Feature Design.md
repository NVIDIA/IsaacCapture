<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

UI&Feature Design

Global standard: unless improper for specific reason, for all float or int input fields, enable up/down arrow to adjust the value. For float input field, the step should be determined by the estimated range of the value. For int input field, the step should be 1. Only when a field has a clear specific limit range, the input field should reject illegal values. For sliders, when there's a clear specific limit range, use the limit range.
* Preserve partial focused float input such as `1.` while the user edits; controlled UI refreshes must not erase the decimal point before the next digit is entered.
* When "Save", check the validation of metadata. If has error, pop up warning window "Metadata contain errors, are you sure to save?" for user to select continue or cancel.
* Every field marked `(spannable)` is fully collapsed recursively by default: the list container, each entity/joint item, and nested Taxel Binding folders all start collapsed.
* Segment, taxel, and displayed URDF-joint indices start at 0.

For field names, use title case for multi-word field names. For example, "URDF Path" instead of "urdf path".

About data cooridanates: all UI editor use visualization space. When save to metadata file, do the coordinate transformation with Segments Alignment value. When load from metadata file, use the default VISUALIZATION_FROM_HAND value as Segments Alignement (and do transformation). Modifying Segments Alignment will cause values in editors also update.
* A fixed three-axis visualization-space direction indicator is displayed in the top-right of the 3D view and updates from the render camera.
* Place the Viser icon at the bottom-left of the view, above the fixed bottom Info area.

About LBS taxel rotation: determine by Polar Decomposition algorithm. At singularity, use taxel's bind rotation. Make sure to optimize for single-segment binding case, where the rotation can be directly determined without running Polar Decomposition.

1. Use two separate menus docked on the right: General on top and Editor/Preview below. General contains Toggle side, new/save/load metadata file, reset camera, a named "Mode" section containing bold Edit and Preview checkbox labels that are mutually exclusive and cannot both be unchecked, and a toggle to enable/disable the saved data visualization "Show Saved Binding Poses".

2. This tool actually manages 3 copies of pose data (pose of segments and taxels), with different data sources.
    a) Editing dataset: the data that the user is editing in the tool. It's the binding poses of the segments and taxels under editing. Plus the URDF joint values that's under Editing mode.
    b) Saved dataset: the binding data that the metadata file stores. This doesn't contain URDF joint values.
    c) Preview dataset: the segments poses, and a set of URDF joint values. It's not a binding data and can't be saved to the metadata file.

3. When toggle the "Show Saved Binding Poses" on, the 3D scene will display the saved binding poses of the segments and taxels, at a opacity of 0.2.

4. Two modes use different source of pose data: When switch to Edit mode, the Editing dataset will be used to render in the 3D scene, at the opacity of 1.0. When switch to Preview mode, the Preview dataset will be used to render in the 3D scene, at the opacity of 1.0.

6. Two modes use different UI menu: When switch to Edit mode, enable "URDF", "Segments", "Taxels" 3 menu tabs. When switch to Preview mode, there won't be tabs. These tabs will be defined explicitly at following.

7. URDF tab: contains following fields:
    a) URDF Path: a text field to input the absolute path of the URDF file.
    b) "Load" button, to load the URDF file at URDF Path in scene.
    c) URDF Alignment: 3 float input fields to shift the position, 3 float slider fields to rotate the orientation, and one float input field to scale the URDF model.
    c) A field to modify each joint value in the loaded URDF file. Current UI field "URDF free joint" and "Preview joint" provide this function, but I want you to use a different UI that shows a spannable full list of URDF joints and their input sliders for the user to slide.

8. Segments tab:
    a) Segments Alignment: 3 float input fields to shift the position, 3 float slider fields to rotate the orientation.
    b) Add new segments: input fields of segment name (HandJoint enum, dropdown selection), position (3 float input fields). An "Add with Input" button.
    c) Add new segments by 3D scene interaction: a button to "Add with URDF Click" button. After clicking this button, the button shows a "pressed" shape, the 3D scene will grey out, with only the URDF (of the Editing dataset) highlighted. The user can click on the URDF to add a new segment on the URDF surface position. Clicking on any other area will cancel the operation.
    d) (spannable) A full list of all segments, with an id number (read-only, starting at 0, it's the index of the segment in the list), their name (dropdown list of HandJoint, display in enum name), editable position (3 float input), editable rotation (3 float slider). This is to modify the binding poses of the segments, display this info above this section.
* It's invalid to have more than one segments to select the same HandJoint. At error, outline the repeated fields in red.
* Some fields in Taxels tab refer to segments in this Segments tab, and modifying segments may result in invalid taxel binding that's revealed in Taxels tab.
* Segments are displayed as flat-shaded cubes with side length 0.004m, without smooth color gradients or cast/received shadows. The three axes of each segment are not displayed by default. Clicking a segment on the 3D scene while in Edit mode will select the segment, highlight it, and reveal its 3-axis translation control. When a segment is newly added, it is selected and the previous selected objects are deselected. Clicking any other area in the 3D scene deselects the segment; dragging the 3D scene or clicking menu/UI areas does not. Clicking the same selected segment does nothing.

9. Taxels tab:
* Taxels don't have their onw alignment, it uses the alignment of the segments.
    a) Add new taxels: input fields of columns (int input), rows (int input), area width and height (float inputs in m), center position (3 float inputs), rotation (3 float sliders), and a left-labeled four-slot Segment/Weight binding grid matching each Taxel row. Each binding grid has no gaps between cells. Leave standard vertical spacing between this grid and "Capture URDF Surface Pose" without a visible spacer element. "Capture URDF Surface Pose" enters a pressed state and lets the user click the URDF to replace these Position and Rotation fields with the clicked surface pose without creating taxels. "Capture Segment Pose" does the same using the clicked Editing segment's complete pose. Clicking elsewhere cancels capture. The following "Add Taxels" button creates a row-major rows*columns planar grid from the current fields and copies the binding grid into every new Taxel; newly created taxels are selected and previous objects are deselected.
    c) Vertically align "Start Binding to Segments", "Select Segments", and "Complete", with a centered down arrow between each adjacent pair. At most one is clickable: initially Start; after Start, no button until taxels are selected; then Select Segments; after it is clicked, no button until segments are selected; then Complete. The first selected Segment receives weight 1 and later selected Segments receive weight 0. Clicking elsewhere cancels and re-enables Start. More than four selected segments cancels the operation, reports the error in the Info panel at the bottom of the 3D display, and re-enables Start.
* Same logic as 3D scene segments editing: dragging in 3D scene will not cancel the action.
    d) (spannable) A paginated list of all taxels, showing 25 rows per page with global zero-based Taxel indices. Above the rows, show page/total and range/total status. The next line contains an unlabeled integer input and "Jump"; the following line contains Last Page (previous) and Next Page. Remove the NumberInput label cell entirely. Repeat the same two control lines below the rows. Build one fixed pool of 25 row controls and retarget their labels/values on page changes; never remove or rebuild the page container. Each visible row contains position (3 float input), editable rotation (3 float slider), editable bound segments (4 dropdowns by HandJoint name, with "None" for inactive slots), and four binding weights. Selecting the first non-None Segment sets that slot's weight to 1; later Segment selections start at 0. Editing a weight changes only that slot and never modifies the other weights. Weight fields reject numbers out of 0~1, and row validation reports when the total is not 1.
* "None" is only for the UI display. When saving to metadata file, it's saved as any of the valid segment in the list, and its weight is 0.
* Taxels data can be invalid for multiple reasons. When invalid, line the invalid field in red. Reasons including:
    a) the taxel is not bound to any segment or segment doesn't exist (caused by Segments tab modification). - segment selection fields.
    b) the weights don't sum to 1. (Can happen when a segment is removed from Segments tab or by Preview apply to editor.) - weights input fields.
* Taxels are displayed as flat squares of size 0.002m*0.002m. A taxel with any validation error uses ERROR_COLOR unless selected; selection uses SELECTED_COLOR with higher priority. The three axes of each taxel are not displayed by default. Clicking a taxel on the 3D scene while in Edit mode will select the taxel, highlight it, and reveal its 3-axis translation control. The 3D scene modification rules are the same as segments.
* Multi-selection in 3D scene: While holding Shift and clicking on an object in 3D scene, the object will be added to the selection without de-selecting the previous selected object(s). It's not allowed to select both segments and taxels at the same time. So when taxels are already selected, Shift-clicking on a segment will not do anything, vise versa. When holding Shift, it's also allowed to left-button drag on the 3D scene to select a rectangular area of objects. If taxels are previously already selected, only taxels in the selected area will be selected, vise versa. When Shift is holding, dragging 3D scene won't change camera as it does otherwise. When nothing is selected, if a Shift dragging covers both segment and taxels, select the taxels but not segments.
* Selection also highlights related objects with the separate configured `SIDE_HIGHLIGHT_COLOR`: selecting segment(s) highlights taxels with a positive binding weight to any selected HandJoint; selecting taxel(s) highlights segments referenced by a positive binding weight. Zero-weight slots never produce this highlight, and selected objects retain `SELECTED_COLOR`.
* The 3-axis translation control is draggable only along its X/Y/Z axes and moves every selected object by the same translation. For one object it follows that object's orientation; for multiple objects it is centered on the selection with axes aligned to visualization-space coordinates. Rotation, plane translation, and scaling handles are disabled.
* An additional floating menu on the left side of the view is always displayed for selected object(s)' modification; it is not docked to the viewport edge. It applies also to single-object selection. With no selection it shows Type: None and Count: 0 and disables every input field:
a) (read only) selected elements type and count (multi selection is stated in following part)
b) (editable) current position (defined above)
c) (editable) current rotation (defined above)
d) (editable single float) scaling (only for multi-selection, scale up/down the selected objects' position in space, based on the area center. For example, if one object was at (1, 2, 3) and the area center is at (1, 1, 1), scaling the area up to 2x will move the object to (1, 3, 5).)
e) When exactly one Segment is selected, show a Segment Editor for its HandJoint name.
f) When exactly one Taxel is selected, show a Taxel Editor for its four Segment/Weight binding slots.
g) Delete

10. Preview menu:
    a) A named "Binding Metadata Source" section containing bold Editor and Live checkbox labels. They are mutually exclusive and cannot both be unchecked; Editor is selected by default. Live remains disabled/grey until the first Live metadata is received. This selection affects rendered taxel poses because they are calculated from segment poses and the selected binding metadata.
    b) A checkbox toggling "Show Binding Poses". When checked, show the segments and taxels (without URDF) from the Editing dataset in the 3D scene at a opacity of 0.5.
    c) A text input field "Tactile Collection ID", default to "tactile".
    d) Four buttons:
       - "Apply Segments" replaces only Editing HandJoint/segment poses from Preview.
       - "Apply Taxels" replaces Editing taxels with exactly the valid derived Preview taxels currently rendered in the 3D scene; invalid taxels are omitted and do not block this operation.
       - "Apply URDF" replaces only Editing URDF joint values.
       - "Apply ALL" applies Preview Segments and URDF values plus exactly the same valid rendered taxel subset used by Apply Taxels.
       Invalid Preview taxels do not disable any Apply operation; they are reported in Info and omitted from taxel application.
* When deffering taxel poses or applying binding relationships to the Edit dataset, segments should be referred to by their HandJoint enum name instead of indice. It means received Metadata requires one layer of translation to turn the binding indice into HandJoint.
* If the segment list is incompatible with the metadata's taxel bindings, show the missing HandJoint reason in the fixed Info area. Apply Taxels and Apply ALL remain available and use only the valid rendered subset.
* Preview deformation is independent per taxel. Invalid taxels are omitted and reported by row without preventing valid taxels from rendering.
Live section:
    a) Checkbox "Use JointState to Drive URDF". When checked, enable the "JointState collection id" input field, and a Joint Mappings text field to receive a long json string to map the joint names in JointState to the URDF. The current field "Joint mappings JSON: source → [target, scale, offset]" is reusable.
    b) A button "Connect Live" button. When clicked, replace the Preview segment list and poses with only the `is_valid=True` HandJoint entries from received HandPose, kept in HandJoint order. The selected binding metadata never seeds, filters, or overwrites Preview segments; it is used only to derive taxel poses from the valid HandPose segments. Manage a field to clearly state the connection state: Live or Disconnected. After clicking "Connect Live", the button changes text to "Disconnect Live". When disconnected, stop updating segment poses or URDF joint values. (If JointState is used, Live also updates URDF joint values.) Show warning: Live will override values in Static section.
* Invalid HandPose joints are absent from Preview. If WRIST is invalid, Preview has no live segments because wrist-relative conversion is unavailable.
* If the metadata
* While Live is connected, if TactileDataArray available, render the taxel force (norm) with corresponding color: 0 is blue, 30 (clamp) is red, in the middle do color interpolation. When force missing, show as light grey (default color). When disconnect, colors go back to light grey.
Static section:
    a) A button "Reset Values" that resets Preview segments and URDF joint values from the Editing dataset binding poses and URDF values.
    b) (spannable) A full list of segments to adjust values, fields are same as in Segments tab, but the data is for the Preview dataset. Make joint_name read-only. Above the section, show the information that this is for preview only. This field is in spannable style. This value reflects the preview segment poses, so will be updated with the Live mode streaming if hand data is received.
    c) (spannable) A full list of URDF joints to adjust values, fields are same as in URDF tab, but the data is for the Preview dataset. Above the section, show the information that this is for preview only. This field is in spannable style. This values reflects real-time URDF joint values of Preview dataset, so will be updated with the Live mode streaming if JointState is applied.
* Segments are selectable and editable from interactive 3D scene under Preview mode, it's technicaly equal to modifying in the Static section. Taxels are not selectable or editable in Preview mode. Preview mode's taxel poses are always derived from its segments poses and the binding metadata it uses.

Button info display: show error messages in a fixed, non-movable Info area occupying the bottom of the 3D display when an action fails.
