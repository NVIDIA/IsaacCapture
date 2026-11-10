<a id="license"></a>

# License

Unless otherwise noted, the code in this repository is licensed under the
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

## Known exceptions

- EgoPost in `src/python/isaaccapture_egopost/` is licensed under
  [OpenMDW-1.1](https://github.com/NVIDIA/IsaacCapture/blob/main/LICENSES/LicenseRef-OpenMDW-1.1.txt),
  with explicit third-party exceptions preserved.
- Certain headers from the [OpenXR SDK](https://github.com/KhronosGroup/OpenXR-SDK) or related
  [extensions](https://github.com/NVIDIA/IsaacCapture/tree/main/deps/cloudxr/openxr_extensions)
  are imported from their original source and are licensed under the
  [MIT license](https://opensource.org/licenses/MIT) or [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt).
- CloudXR SDK is NVIDIA’s proprietary software, licensed under the [NVIDIA CloudXR License](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE).
- The documentation is built using [NVIDIA Sphinx Theme](https://pypi.org/project/nvidia-sphinx-theme/).

## License headers

Pre-commit automatically checks and updates license headers with
[REUSE](https://reuse.software/). It selects `Apache-2.0` by default and
`LicenseRef-OpenMDW-1.1` for EgoPost, preserving existing attribution and
explicit third-party exceptions. Review and stage any automatic edits before
committing. Checked-in third-party files record upstream’s terms; see *What stops
the Release build* in [scripts/sbom/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/scripts/sbom/README.md).

You can also run `reuse spdx -o project.spdx` to generate an SPDX report for the project.
