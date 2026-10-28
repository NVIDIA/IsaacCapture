UI&Feature Design

Global standard: unless improper for specific reason, for all float or int input fields, enable up/down arrow to adjust the value. For float input field, the step should be determined by the estimated range of the value. For int input field, the step should be 1. Only when a field has a clear specific limit range, the input field should reject illegal values. For sliders, when there's a clear specific limit range, use the limit range.
* When "Save", check the validation of metadata. If has error, pop up warning window "Metadata contain errors, are you sure to save?" for user to select continue or cancel.

For field names, use title case for multi-word field names. For example, "URDF Path" instead of "urdf path".

About data cooridanates: all UI editor use visualization space. When save to metadata file, do the coordinate transformation with Segments Alignment value. When load from metadata file, use the default VISUALIZATION_FROM_HAND value as Segments Alignement (and do transformation). Modifying Segments Alignment will cause values in editors also update.

About LBS taxel rotation: determine by Polar Decomposition algorithm. At singularity, use taxel's bind rotation. Make sure to optimize for single-segment binding case, where the rotation can be directly determined without running Polar Decomposition.

1. At top, always display general function: Toggle side, new/save/load metadata file, reset camera. A Mode single seletion from: Edit, Preview. And a toggle to enable/disable the saved data visualization "Show Saved Binding Poses".

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
    d) (spannable) A full list of all segments, with an id number (read-only, starting at 0, it's the index of the segment in the list), their name (dropdown list of HandJoint, display in enum name), editable position (3 float input), editable rotation (3 float slider). This is to modify the binding poses of the segments, display this info above this section. Each segment has a grabber on the side that can be used to drag them and reorder, this will change the id numbers.
* It's invalid to have more than one segments to select the same HandJoint. At error, outline the repeated fields in red.
* Some fields in Taxels tab refer to segments in this Segments tab, and modifying segments may result in invalid taxel binding that's revealed in Taxels tab.
* Segments are displayed as yellow spheres of radius 0.01m. The three axes of each segment are not displayed by default.Clicking an segment on the 3D scene while in Edit mode will select the segment, highlight it, reveal its 3 axes. When a segment is newly added, it's selected, and deselect the previous selected object(s). Clicking on any other area in the 3D scene will deselect the segment; dragging 3D scene or clicking on the menu/other UI area will not deselect the segment. Clicking on the same selected segment will not do anything.

9. Taxels tab:
* Taxels don't have their onw alignment, it uses the alignment of the segments.
    a) Add new taxels: input fields of columns (int input), rows (int input), intersection width (float input, declare the unit is m), height (float input, declare the unit is m), taxels center position (3 float input fields). An "Add with Input" button. When adding taxels, it will generate columns*rows taxels, which aligned in a grid of the inputted area width and height, with the grid center at the inputted taxels center position, the grid orientation facing +Z axis of visualization space. Each taxel's orientation is originally identity.
    b) Add new taxels by 3D scene interaction: a button to "Add with URDF Click" button with columns and rows int input fields. After clicking this button, the button shows a "pressed" shape, the 3D scene will grey out, with only the URDF (of the Editing dataset) highlighted. The user can click on the URDF to add a new array of taxel on the URDF surface, this will use the clicked URDF position as grid center position, and the clicked URDF normal (pointing out of the URDF surface) as grid orientation. Otherwise the grid-adding rule is the same as Add with Input.Clicking on any other area will cancel the operation. When taxels are newly created, they're selected and the previous selected objects are de-selected.
    c) A series of buttons "Start Binding to Segments" "Select Segments" "Complete". At first, only "Start Binding to Segments" is clicable. When clicking this button, 3D scene will grey out except for taxels (from Edit dataset) is displayed normally (without highlight), the instruction field under the button will print: "Select taxel(s) to bind in 3D scene. Hold Shift to multi-select. Click "Select Segments" after done selecting taxels. Click on any other area in 3D scene to cancel.". When at least one taxel is selected, the button "Select Segments" becomes clickcable. The 3D scene stays greyed out and only segments (from Edit dataset) are displayed normally (without highlight). The instruction field changes to print: "Select 1-4 segment(s) to bind to the selected taxel(s) in 3D scene. Hold Shift to multi-select. Click "Complete" after done selecting segments. Click on any other area in 3D scene to cancel.". When at least one segment is selected, the "Complete" button becomes clickable. Clicking "Complete" button will update the Edit dataset's binding, each original weight is average (if selected 4 segments, then 0.25 each; if selected 3 segments, then 1/3 each.) (also update the following section). If more than 4 segments are selected, don't update the binding, and print in instruction field: "Too many segments selected, please select 1-4 segments. Click "Start Binding to Segments" to start over.".
* Same logic as 3D scene segments editing: dragging in 3D scene will not cancel the action.
    d) (spannable) A full list of all taxels, with an id number (read-only, starting at 0, it's the index of the taxel in the list), their position (3 float input), editable rotation (3 float slider), editable binded segments (4 * dropdown list from the segments list, displayed as segment name (HandJoint enum name). "None" if not selected), binding weights (4 * float input. Grey out for non-selected segments.). This is to modify the binding of the taxels, display this info above this section. Each taxel has a grabber on the side that can be used to drag them and reorder, this will change the id numbers. After modifying a weight here, automatically update the other weights for the same taxel by their proportion to sum to 1 (unify them without changing just edited value. If all other weights are 0, treat them as equal proportion). Weight field reject numbers out of 0~1. If the modified weight is the only weight, adjust it to 1.
* "None" is only for the UI display. When saving to metadata file, it's saved as any of the valid segment in the list, and its weight is 0.
* Taxels data can be invalid for multiple reasons. When invalid, line the invalid field in red. Reasons including:
    a) the taxel is not bound to any segment or segment doesn't exist (caused by Segments tab modification). - segment selection fields.
    b) the weights don't sum to 1. (Can happen when a segment is removed from Segments tab or by Preview apply to editor.) - weights input fields.
    c) the weight is 0 for a non-None segment - weight input field.
* Taxels are displayed as light grey flat square of size 0.01m*0.01m. The three axes of each taxel are not displayed by default. Clicking an taxel on the 3D scene while in Edit mode will select the taxel, highlight it, reveal its 3 axes. The 3D scene modification rules are the same as segments.
* Multi-selection in 3D scene: While holding Shift and clicking on an object in 3D scene, the object will be added to the selection without de-selecting the previous selected object(s). It's not allowed to select both segments and taxels at the same time. So when taxels are already selected, Shift-clicking on a segment will not do anything, vise versa. When holding Shift, it's also allowed to left-button drag on the 3D scene to select a rectangular area of objects. If taxels are previously already selected, only taxels in the selected area will be selected, vise versa. When Shift is holding, dragging 3D scene won't change camera as it does otherwise. When nothing is selected, if a Shift dragging covers both segment and taxels, select the taxels but not segments.
* When multiple objects are selected, the 3-axis visualizer should show in the center of the selected objects' area, with axis directions align with the visualiation space coordinates.
* An additional small menu for selected object(s)' modification, apply also to single object selection:
a) (read only) selected elements type and count (multi selection is stated in following part)
b) (editable) current position (defined above)
c) (editable) current rotation (defined above)
d) (editable single float) scaling (only for multi-selection, scale up/down the selected objects' position in space, based on the area center. For example, if one object was at (1, 2, 3) and the area center is at (1, 1, 1), scaling the area up to 2x will move the object to (1, 3, 5).)
e) Delete

10. Preview menu:
    a) A selection field to select the Binding Metadata Source from "Live" and "Editing", by default "Editing". "Live" is only available after the first live metadata is received. This selection affects the taxels pose rendered, because it's calculated based on the segments poses and binding metadata.
    b) A checkbox toggling "Show Binding Poses". When checked, show the segments and taxels (without URDF) from the Editing dataset in the 3D scene at a opacity of 0.5.
    c) A text input field "Tactile Collection ID", default to "tactile".
    d) A button "Apply Preview Data to Editor". When clicked, apply the Preview dataset's segment poses and deferred taxel poses (segment poses + selected metadata binding) and the binding relationships, to the Editing dataset's segment binding poses and taxel binding data. Also update the urdf joint values to Editing dataset. Remind the user that this action will fully override segments and taxels.
* When deffering taxel poses or applying binding relationships to the Edit dataset, segments should be referred to by their HandJoint enum name instead of indice. It means received Metadata requires one layer of translation to turn the binding indice into HandJoint.
* If the segment list is incompatible to the metadata's taxels binding (e.g. taxels refer to a segment that's not in list), an error should show up at the top of Preview menu including information like the missing segment enum name, and "Apply Preview Data to Editor" should grey out not clickable.
Live section:
    a) Checkbox "Use JointState to Drive URDF". When checked, enable the "JointState collection id" input field, and a Joint Mappings text field to receive a long json string to map the joint names in JointState to the URDF. The current field "Joint mappings JSON: source → [target, scale, offset]" is reusable. 
    b) A button "Connect Live" button. When clicked, update the Preview dataset's segments poses with received hand data in real time, and update the taxels poses based on the segments poses and the binding metadata (selected in above selection field). Manage a field to clearly state the connection statue: Live or Disconnected. After clicking "Connect Live", the button changes text to "Disconnect Live". When disconnected, stops updating the segments poses or URDF joint values. (If JointState is used, Live mode also updates the URDF joint values.) Show warning: Live will override values in Static section.
* If the metadata 
* While Live is connected, if TactileDataArray available, render the taxel force (norm) with corresponding color: 0 is blue, 30 (clamp) is red, in the middle do color interpolation. When force missing, show as light grey (default color). When disconnect, colors go back to light grey.
Static section:
    a) A button "Restore to Editor Binding Values"
    b) (spannable) A full list of URDF joints to adjust values, fields are same as in URDF tab, but the data is for the Preview dataset. Above the section, show the information that this is for preview only. This field is in spannable style. This values reflects real-time URDF joint values of Preview dataset, so will be updated with the Live mode streaming if JointState is applied.
    c) (spannable) A full list of segments to adjust values, fields are same as in Segments tab, but the data is for the Preview dataset. Exclude the reordering grabber UI. Make joint_name read-only. Above the section, show the information that this is for preview only. This field is in spannable style. This value reflects the preview segment poses, so will be updated with the Live mode streaming if hand data is received.
* Segments are selectable and editable from interactive 3D scene under Preview mode, it's technicaly equal to modifying in the Static section. Taxels are not selectable or editable in Preview mode. Preview mode's taxel poses are always derived from its segments poses and the binding metadata it uses. 

Button info display: show error messages at button when action fails.