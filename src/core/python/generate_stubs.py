# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Python stub file (.pyi) for a single Isaac Teleop pybind11 module.

This script uses pybind11-stubgen's Python API to generate type stubs for IDE intellisense.

Usage:
    python generate_stubs.py <module_name> <package_dir>

Example:
    python generate_stubs.py isaaccapture.deviceio._deviceio /path/to/python_package
"""

import sys
import traceback
from pathlib import Path


def _err(message: str) -> None:
    """Report a failure on stderr.

    Diagnostics go to stderr so they stay adjacent to each other: stdout is
    block-buffered under a pipe (as in a CMake custom command), which otherwise
    strands related lines dozens apart in the build log.
    """
    print(message, file=sys.stderr)


def generate_stub(module_name: str, package_dir: Path) -> bool:
    """Generate stub file for a single pybind11 module.

    Args:
        module_name: Fully qualified module name (e.g., "isaaccapture.deviceio._deviceio")
        package_dir: Path to the python_package directory containing isaaccapture.

    Returns:
        True if successful, False otherwise.
    """
    # Add package_dir to sys.path so pybind11-stubgen can import the module
    sys.path.insert(0, str(package_dir))

    try:
        from pybind11_stubgen import (
            CLIArgs,
            Printer,
            Writer,
            run,
            stub_parser_from_args,
            to_output_and_subdir,
        )
    except ImportError:
        _err("Error: pybind11-stubgen not installed")
        _err("This script should be run via: uv run --with pybind11-stubgen")
        return False

    print(f"Generating stubs for {module_name}...")

    # Configure stubgen using the proper API
    args = CLIArgs(
        module_name=module_name,
        output_dir=str(package_dir),
        root_suffix="",
        ignore_invalid_expressions=None,
        ignore_invalid_identifiers=None,
        ignore_unresolved_names=None,
        ignore_all_errors=True,  # Continue even if some signatures fail
        enum_class_locations=[],
        numpy_array_wrap_with_annotated=False,
        numpy_array_use_type_var=False,
        numpy_array_remove_parameters=False,
        print_invalid_expressions_as_is=False,
        print_safe_value_reprs=None,
        exit_code=True,
        dry_run=False,
        stub_extension="pyi",
    )

    try:
        parser = stub_parser_from_args(args)
        printer = Printer(
            invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is
        )
        out_dir, sub_dir = to_output_and_subdir(
            output_dir=args.output_dir,
            module_name=args.module_name,
            root_suffix=args.root_suffix,
        )
        writer = Writer(stub_ext=args.stub_extension)

        run(
            parser=parser,
            printer=printer,
            module_name=args.module_name,
            out_dir=out_dir,
            sub_dir=sub_dir,
            dry_run=args.dry_run,
            writer=writer,
        )
        print(f"  Generated stubs for {module_name}")
        return True

    except Exception as e:
        # Headline the failure, then print the whole chain. str(e) alone is not
        # enough: importing a pybind11 module runs PYBIND11_MODULE's init, which
        # re-raises any failure as a bare "ImportError: initialization failed"
        # and hangs the real cause off __cause__.
        _err(f"Error: stubgen failed for {module_name}: {e}\n{traceback.format_exc()}")
        return False


def main() -> int:
    """Main entry point."""
    if len(sys.argv) != 3:
        _err(f"Usage: {sys.argv[0]} <module_name> <package_dir>")
        _err(
            f"Example: {sys.argv[0]} isaaccapture.deviceio._deviceio /path/to/python_package"
        )
        return 1

    module_name = sys.argv[1]
    package_dir = Path(sys.argv[2]).resolve()

    if not package_dir.exists():
        _err(f"Error: Package directory does not exist: {package_dir}")
        return 1

    if generate_stub(module_name, package_dir):
        return 0
    else:
        _err("Stub generation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
