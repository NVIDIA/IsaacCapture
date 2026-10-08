# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dex BiManual Retargeting Example

Demonstrates using DexBiManualRetargeter to control two hands simultaneously.
"""

import sys
import time
import argparse
import contextlib
from types import SimpleNamespace
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import HandsSource
from isaaccapture.retargeters import (
    DexHandRetargeter,
    DexHandRetargeterConfig,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig
from isaaccapture.retargeting_engine_ui import MultiRetargeterTuningUIImGui


def main():
    print("\n" + "=" * 80)
    print("  Dex BiManual Retargeting Example")
    print("=" * 80)
    print("Requires dex-retargeting library and config files.")
    print("This example assumes a robot with both left and right hands.")
    print("=" * 80 + "\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dex BiManual Retargeting Example")
    parser.add_argument("--left-urdf", type=str, help="Path to left hand URDF")
    parser.add_argument("--right-urdf", type=str, help="Path to right hand URDF")
    parser.add_argument(
        "--left-config", type=str, help="Path to left hand retargeting config (YAML)"
    )
    parser.add_argument(
        "--right-config", type=str, help="Path to right hand retargeting config (YAML)"
    )
    parser.add_argument(
        "--left-params", type=str, help="Path to left hand tunable parameters (JSON)"
    )
    parser.add_argument(
        "--right-params", type=str, help="Path to right hand tunable parameters (JSON)"
    )
    parser.add_argument(
        "--enable-tuning", action="store_true", help="Enable retargeting tuning UI"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    # Check for config files
    # Note: These paths are placeholders. In a real scenario, you'd point to your robot's config.
    # We'll use the example configs from dex_hand_retargeting_example.py logic if they existed,
    # but since this is an example, we'll mock the paths or expect them relative to this script.

    # We'll assume the user has set up the configs similar to the single hand example
    config_dir = Path(__file__).parent / "config" / "dex_retargeting"

    # Determine paths based on args or defaults
    left_yaml = (
        Path(args.left_config)
        if args.left_config
        else config_dir / "hand_left_config.yml"
    )
    right_yaml = (
        Path(args.right_config)
        if args.right_config
        else config_dir / "hand_right_config.yml"
    )
    left_urdf_path = (
        Path(args.left_urdf) if args.left_urdf else config_dir / "left_robot_hand.urdf"
    )
    right_urdf_path = (
        Path(args.right_urdf)
        if args.right_urdf
        else config_dir / "right_robot_hand.urdf"
    )

    # Check if files exist (soft check)
    if not left_yaml.exists():
        print(f"Warning: Config file {left_yaml} not found. Example may fail.")

    # ==================================================================
    # Build Retargeting Pipeline
    # ==================================================================

    # Create source (tracker is internal)
    hands = HandsSource(name="hands")

    # Define parameter config paths for persistence
    # Use provided args or default to /tmp/dex_bimanual_example_{side}_params.json
    left_param_path = (
        args.left_params
        if args.left_params
        else "/tmp/dex_bimanual_example_left_params.json"
    )
    right_param_path = (
        args.right_params
        if args.right_params
        else "/tmp/dex_bimanual_example_right_params.json"
    )

    print(
        f"Parameter persistence enabled. Configs will be saved to:\n  - {left_param_path}\n  - {right_param_path}"
    )

    # Left Config
    left_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=str(left_yaml),
        hand_urdf=str(left_urdf_path),
        hand_side="left",
        handtracking_to_baselink_frame_transform=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        parameter_config_path=left_param_path,
    )

    # Right Config
    right_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=str(right_yaml),
        hand_urdf=str(right_urdf_path),
        hand_side="right",
        handtracking_to_baselink_frame_transform=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        parameter_config_path=right_param_path,
    )

    # Instantiate separate retargeters
    left_retargeter = DexHandRetargeter(left_cfg, name="left_hand")
    right_retargeter = DexHandRetargeter(right_cfg, name="right_hand")

    # Connect them
    connected_left = left_retargeter.connect(
        {HandsSource.LEFT: hands.output(HandsSource.LEFT)}
    )
    connected_right = right_retargeter.connect(
        {HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}
    )

    # Combine outputs
    pipeline = OutputCombiner(
        {
            "left_hand_joints": connected_left.output("hand_joints"),
            "right_hand_joints": connected_right.output("hand_joints"),
        }
    )

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    with CloudXRLauncher.launch_context(args):
        session_config = TeleopSessionConfig(
            app_name="DexBiManualExample",
            trackers=[],  # Auto-discovered from pipeline
            pipeline=pipeline,
        )

        # Access the internal retargeters for tuning
        retargeters_to_tune = [left_retargeter, right_retargeter]

        # Open the UI using the context manager
        if args.enable_tuning:
            print("Opening Retargeting UI...")
            ui_context = MultiRetargeterTuningUIImGui(
                retargeters_to_tune, title="Hand Retargeting Tuning"
            )
        else:
            ui_context = contextlib.nullcontext(
                SimpleNamespace(is_running=lambda: True)
            )

        with ui_context as ui:
            with TeleopSession(session_config) as session:
                start_time = time.time()

                while time.time() - start_time < 360.0 and ui.is_running():
                    result = session.step()

                    # Output: Combined joint angles
                    # result["left_hand_joints"] and result["right_hand_joints"] are TensorGroups
                    left_vals = list(result["left_hand_joints"])
                    right_vals = list(result["right_hand_joints"])

                    if session.frame_count % 30 == 0:
                        elapsed = session.get_elapsed_time()
                        # Print first few joints from left and right parts
                        l_print = left_vals[: min(3, len(left_vals))]
                        r_print = right_vals[: min(3, len(right_vals))]

                        print(f"[{elapsed:5.1f}s] L: {l_print} ... R: {r_print} ...")

                    time.sleep(0.016)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
