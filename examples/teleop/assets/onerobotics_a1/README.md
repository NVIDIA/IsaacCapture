# OneRobotics A1 assets

This directory packages the OneRobotics A1 source assets used by the Isaac Teleop
joint-space example. The package contains the two independent 7-DoF arm URDFs,
the bimanual stand URDF, their STL meshes, the supplied MJCF models, and the
source license records.

The runnable example uses `mjcf/A1/a1_right_position.xml`. This is the supplied
MuJoCo right-arm model; it is not a URDF conversion. The independent-arm and
bimanual URDFs remain available under `urdf/A1_2026/` for Isaac Sim/Isaac Lab
integration.

## Provenance

- Source repository: <https://github.com/katazen/onerobot_h1>
- Source revision: `ecf530911284ba0e559f7a24dc222fd8e60d31ed`
- Asset license: CC BY 4.0; see `CC-BY-4.0.txt` and `ASSET_LICENSE_STATUS.md`.
- Project-code license text retained as `LICENSE`; it does not replace the asset
  license.

The A1 asset attribution is:

> OneRobotics A1 robot assets © 2026 OneRobotics, licensed under CC BY 4.0
> (<https://creativecommons.org/licenses/by/4.0/>). Source:
> <https://github.com/katazen/onerobot_h1>. Changes: mesh paths are packaged
> under this directory; robot geometry and inertial data are unchanged.

## Verification

From the asset root:

```bash
cd urdf/A1_2026 && sha256sum --check SHA256SUMS
```

The example also checks that the right-arm MJCF compiles and that its seven
joint names match the Isaac Teleop mapping.
