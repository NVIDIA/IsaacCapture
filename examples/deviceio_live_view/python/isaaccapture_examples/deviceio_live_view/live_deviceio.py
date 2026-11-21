# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize live DeviceIO human tracking in real time with viser.

Registers every human-related DeviceIO source (hands, head, controllers, full
body) and draws whichever trackers are currently active. Inactive or absent
trackers are hidden rather than shown in an error color.

``CloudXRLauncher`` starts the CloudXR runtime and WSS proxy automatically.
Open the URL viser prints in a browser. Binds all interfaces by default, so
another machine on the network can reach it at http://<this-host>:8080.

Usage:
    python -m isaaccapture_examples.deviceio_live_view [--port 8080] [--host 127.0.0.1] [--accept-eula]

Press Ctrl+C to stop.
"""

import argparse
import time

import viser

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaBodyRepresentation,
    SomaHandRepresentation,
)

from .body_pipeline import BodySchema, create_body_view_pipeline, resolve_body_schema
from .deviceio_pipeline import build_all_human_pipeline
from .deviceio_viser import HumanDeviceIOViz, setup_scene
from .hand_pipeline import HandSchema, create_hand_view_pipeline


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viser HTTP bind address (default: 0.0.0.0, all interfaces; pass 127.0.0.1 to keep it local)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    parser.add_argument(
        "--body-schema",
        choices=[schema.value for schema in BodySchema],
        help="Input body schema (default: full-body-pose)",
    )
    parser.add_argument(
        "--soma-body-collection-id",
        default="soma_body_demo",
        help="SOMA body publisher's tensor collection ID",
    )
    parser.add_argument(
        "--soma-body-representation",
        choices=[representation.value for representation in SomaBodyRepresentation],
        default=SomaBodyRepresentation.JOINT_ROTATIONS.value,
        help="SOMA FlatBuffer profile consumed by the viewer",
    )
    parser.add_argument(
        "--hand-schema",
        choices=[schema.value for schema in HandSchema],
        default=HandSchema.OPENXR_HAND_POSE.value,
        help="Input hand schema (default: openxr-hand-pose)",
    )
    parser.add_argument(
        "--soma-left-hand-collection-id",
        default="soma_hand_left_demo",
        help="Left SOMA hand publisher's tensor collection ID",
    )
    parser.add_argument(
        "--soma-right-hand-collection-id",
        default="soma_hand_right_demo",
        help="Right SOMA hand publisher's tensor collection ID",
    )
    parser.add_argument(
        "--soma-hand-representation",
        choices=[representation.value for representation in SomaHandRepresentation],
        default=SomaHandRepresentation.JOINT_ROTATIONS.value,
        help="SOMA hand FlatBuffer profile consumed by the viewer",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])

    try:
        body_schema = resolve_body_schema(args.body_schema)
        body = create_body_view_pipeline(
            body_schema=body_schema,
            soma_body_collection_id=args.soma_body_collection_id,
            soma_body_representation=args.soma_body_representation,
        )
        hands = create_hand_view_pipeline(
            hand_schema=args.hand_schema,
            soma_left_collection_id=args.soma_left_hand_collection_id,
            soma_right_collection_id=args.soma_right_hand_collection_id,
            soma_hand_representation=args.soma_hand_representation,
        )
        pipeline = build_all_human_pipeline(body=body, hands=hands)
    except ValueError as error:
        parser.error(str(error))

    server = viser.ViserServer(host=args.host, port=args.port)
    ground = setup_scene(server)
    config = TeleopSessionConfig(
        app_name="LiveDeviceIOExample",
        pipeline=pipeline,
    )

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(f"[live] CloudXR runtime started (WSS log: {launcher.wss_log_path})")
        print("[live] waiting for headset connection… (Ctrl+C to stop)")

        with TeleopSession(config) as session:
            viz = HumanDeviceIOViz(server, ground, body.layout, hands.layout)
            print(
                f"[live] viser listening on {args.host}:{args.port} "
                f"(http://localhost:{args.port})"
            )
            print(f"[live] body schema: {body_schema.value}")
            print(f"[live] hand schema: {args.hand_schema}")
            try:
                while True:
                    result = session.step()
                    active = viz.update(result)

                    if session.frame_count % 60 == 0:
                        body_joints = active["body_joints"]
                        print(
                            f"[live] frame={session.frame_count}  "
                            f"hands(L/R)={'Y' if active['hand_left'] else '-'}/"
                            f"{'Y' if active['hand_right'] else '-'}  "
                            f"head={'Y' if active['head'] else '-'}  "
                            f"ctrl(L/R)={'Y' if active['controller_left'] else '-'}/"
                            f"{'Y' if active['controller_right'] else '-'}  "
                            f"body={body_joints:02d}/{len(body.layout.joint_names)}"
                        )
                    time.sleep(1 / 60)
            except KeyboardInterrupt:
                pass

    print("[live] stopped")
    return 0
