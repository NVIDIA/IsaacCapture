<!--
SPDX-FileCopyrightText: 2026 OneRobotics
SPDX-License-Identifier: CC-BY-4.0
-->

# OneRobotics A1 Asset License

License declaration date: 2026-08-14

Asset inventory updated: 2026-08-28

Copyright holder and licensor: **OneRobotics**

License: **Creative Commons Attribution 4.0 International** (`CC-BY-4.0`)

`ASSET_LICENSE_DECISION = LICENSED_CC_BY_4_0`

## License grant

OneRobotics makes the asset material listed below available under the Creative Commons Attribution 4.0 International license. You may share and adapt the material for any purpose, including commercial use, provided that you comply with the license terms.

The complete legal code is included at [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt). The official license page is <https://creativecommons.org/licenses/by/4.0/>.

## Covered material

The CC BY 4.0 asset license applies to:

- all 66 binary STL mesh files: 33 under
  `source/h1_reach/h1_reach/assets/meshes/`, 16 under
  `source/h1_reach/h1_reach/assets/urdf/A1_2026/meshes/`, and 17 under
  `source/h1_reach/h1_reach/assets/urdf/A1_2026/bimanual_stand/meshes/`;
- all six URDF robot-model files under
  `source/h1_reach/h1_reach/assets/urdf/`, including the A1 2026 right- and
  left-arm source models `A1_2026/a1_r.urdf` and `A1_2026/a1_l.urdf`, and the
  whole-robot source model
  `A1_2026/bimanual_stand/a1_bimanual_stand.urdf`;
- all five MuJoCo/MJCF XML robot-model files under `source/h1_reach/h1_reach/assets/mjcf/`;
- the A1 2026 model and control metadata in
  `source/h1_reach/h1_reach/assets/urdf/A1_2026/model_parameters.yaml`;
- USD, USDA, USDC, or USDZ conversions that OneRobotics publishes from the covered model files and explicitly identifies as covered by this declaration, excluding incorporated third-party material that OneRobotics is not authorized to license.

The current source tree contains 77 covered robot-model files: 66 STL files,
six URDF files, and five MJCF/XML files. Together with the covered A1 2026
metadata YAML, the current covered asset inventory is 78 files. The A1 2026
inventory contains two independent single-arm URDF models with 16 meshes and a
distinct whole-robot bimanual stand URDF with 17 meshes, for 36 A1 2026
robot-model files. The revisions are separate CAD sources and do not include a
gripper. No USD-format asset is currently tracked.

In built wheel and source-distribution artifacts, these files appear under
`h1_reach/assets/`; that packaging-path change does not alter the license scope.

## Required attribution

When redistributing the covered material or an adaptation, retain the copyright and license information, link to CC BY 4.0, identify OneRobotics as the creator, and indicate whether changes were made. A suitable attribution is:

> OneRobotics A1 robot assets © 2026 OneRobotics, licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). Source: <https://github.com/katazen/onerobot_h1>. Changes: none.

Replace `none` with a short description when redistributing modified assets. The OneRobotics website is <http://www.onerobot.com/>.

For the bimanual stand publication candidate in this repository, the indicated
changes are: mesh URI and MuJoCo `meshdir` references adapted to the portable
repository layout, normalized line endings, and removal of the exporter's
contact address from the generated comment; physical model fields unchanged.

## Scope boundaries

- The project source code is not licensed by this file. It uses BSD-3-Clause as described in the root `LICENSE`, except where a source file retains a separate upstream notice.
- Files copied or substantially derived from Isaac Lab retain their existing Isaac Lab Project Developers copyright and BSD-3-Clause notices. See `THIRD_PARTY_NOTICES.md`.
- The SolidWorks to URDF Exporter attribution embedded in the URDF comments identifies the generating tool; the tool is not vendored and its software license does not replace this asset license.
- The CC BY 4.0 grant does not cover third-party Isaac Lab, Isaac Sim, NVIDIA scene, user-interface, grid, table, or marker elements visible in `media/onerobotics-a1-zero-pose.png`. The covered A1 model remains CC BY 4.0; the screenshot is included only as documentation and is not relicensed as a whole by this file.
- CC BY 4.0 does not grant rights in OneRobotics names, logos, or trademarks beyond the use reasonably necessary for attribution, and reuse must not imply endorsement by OneRobotics or Isaac Lab.
- Third-party runtime software and dependencies remain under their own licenses.

This declaration is the controlling repository license record for the exact covered material above.
