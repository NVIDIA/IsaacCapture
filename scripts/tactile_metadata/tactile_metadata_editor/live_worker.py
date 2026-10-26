# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local receive-only SDK session, or explicitly synthetic standard-library demo."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import socket
import sys
import threading
import time

from .compiler import canonical_values, metadata_fingerprint
from .document import FORMAT, VERSION
from .domain import thaw
from .live import (
    ConnectConfig, _finite_float, _number, _reject_constant, _rows,
    receive_message, send_message,
)
from .profiles.hand import JOINT_NAMES

LOGGER = logging.getLogger("isaaccapture.tools.tactile_metadata.live_worker")


def _invalid(reason):
    return {"valid": False, "reason": reason}


def _demo_metadata():
    identity = [0, 0, 0, 0, 0, 0, 1]
    poses = [[0.025 + column * 0.003, 0.005 + row * 0.003, 0, 0, 0, 0, 1]
             for row in range(4) for column in range(4)]
    return canonical_values({
        "format": FORMAT, "version": VERSION,
        "segment_source": {
            "type": "HandSkeletonRef", "joint_ids": ["WRIST", "INDEX_DISTAL"],
        },
        "geometry_mapping": {
            "type": "LBS",
            "segment_bind_poses": [identity, [0.02, 0.01, 0, 0, 0, 0, 1]],
            "taxel_bind_poses": poses,
            "taxel_segment_indices": [[1, 1, 1, 1] for _ in poses],
            "taxel_segment_weights": [[0.25, 0.25, 0.25, 0.25] for _ in poses],
        },
    })


def _demo_hand(elapsed, side):
    base = (0.4, -0.2 if side == "left" else 0.2, 0.1)
    root_angle = 0.2 * math.sin(elapsed)
    angle = 0.5 * math.sin(elapsed * 1.3)
    cosine, sine = math.cos(root_angle), math.sin(root_angle)
    positions, orientations = [], []
    for index, name in enumerate(JOINT_NAMES):
        if name == "WRIST":
            x, y = 0, 0
        elif name == "INDEX_DISTAL":
            x, y = 0.02, 0.01
        else:
            x, y = 0.006 * (index % 5), 0.008 * (index // 5 + 1)
        positions.append([base[0] + cosine*x - sine*y,
                          base[1] + sine*x + cosine*y, base[2]])
        theta = root_angle if name == "WRIST" else root_angle + angle
        orientations.append([0, 0, math.sin(theta/2), math.cos(theta/2)])
    return {
        "valid": True, "positions": positions,
        "orientations_xyzw": orientations, "radii": [0.005] * len(JOINT_NAMES),
        "is_valid": [True] * len(JOINT_NAMES), "reason": None,
    }


def _control(stream, stop):
    try:
        while not stop.is_set():
            message = receive_message(stream, stop)
            if message != {"type": "stop"}:
                raise ValueError("Unsupported receiving control message")
            stop.set()
    except (EOFError, OSError, ValueError):
        stop.set()


def _send_metadata(stream, epoch, collection, values):
    send_message(stream, {
        "type": "metadata", "epoch": epoch, "collection_id": collection,
        "values": values, "fingerprint": metadata_fingerprint(values),
    })


def _demo(stream, config, epoch, stop):
    baselines = {}
    for binding in config.bindings.values():
        collection = binding["tactile_collection_id"]
        values = config.demo_values.get(collection)
        baselines[collection] = thaw(values) if values is not None else _demo_metadata()
        _send_metadata(stream, epoch, collection, baselines[collection])
    start, seq = time.monotonic(), 0
    while not stop.is_set():
        tick = time.monotonic()
        elapsed = tick - start
        sources = {"tactile": {}, "hands": {}, "joints": {}}
        for side, binding in config.bindings.items():
            collection = binding["tactile_collection_id"]
            count = len(baselines[collection]["geometry_mapping"]["taxel_bind_poses"])
            sources["tactile"][collection] = {
                "valid": True, "reason": None,
                "forces": [[0, 0, 15 + 15 * math.sin(elapsed + index*0.1)]
                           for index in range(count)],
            }
            sources["hands"][side] = _demo_hand(elapsed, side)
            joint = binding.get("joint_source")
            if joint is not None:
                names = {row["source_joint"] for item in config.bindings.values()
                         if item.get("joint_source")
                         and item["joint_source"]["collection_id"]
                         == joint["collection_id"]
                         for row in item.get("joint_mapping", ())}
                sources["joints"][joint["collection_id"]] = {
                    "valid": True, "reason": None, "device_id": joint["device_id"],
                    "joints": [{"name": name, "position": 0.25*math.sin(elapsed),
                                "velocity": 0.25*math.cos(elapsed), "effort": 0,
                                "valid": True} for name in sorted(names)],
                    "has_velocity": True, "has_effort": False,
                }
        send_message(stream, {
            "type": "frame", "epoch": epoch, "seq": seq,
            "read_monotonic_ns": time.monotonic_ns(), "sources": sources,
        })
        seq += 1
        stop.wait(max(0, 1/30 - (time.monotonic() - tick)))


def _metadata(sample, np):
    source, mapping = sample.segment_source, sample.geometry_mapping
    if source is None or mapping is None:
        raise ValueError("Received tactile metadata has no supported source/mapping")
    names = [joint.name for joint in source.joint_ids]
    fields = {}
    for name in ("segment_bind_poses", "taxel_bind_poses",
                 "taxel_segment_indices", "taxel_segment_weights"):
        fields[name] = np.array(getattr(mapping, name), copy=True, order="C").tolist()
    return canonical_values({
        "format": FORMAT, "version": VERSION,
        "segment_source": {"type": "HandSkeletonRef", "joint_ids": names},
        "geometry_mapping": {"type": "LBS", **fields},
    })


def _hand(sample, np):
    if sample is None or sample.joints is None:
        return _invalid("No HandPose sample")
    joints = sample.joints
    fields = {}
    for name, attribute in (("positions", "positions"),
                            ("orientations_xyzw", "orientations"),
                            ("radii", "radii"), ("is_valid", "is_valid")):
        fields[name] = np.array(getattr(joints, attribute), copy=True, order="C")
    count = len(JOINT_NAMES)
    if (
        fields["positions"].shape != (count, 3)
        or fields["orientations_xyzw"].shape != (count, 4)
        or fields["radii"].shape != (count,)
        or fields["is_valid"].shape != (count,)
    ):
        return _invalid("HandPose contains invalid arrays")
    result = {name: value.tolist() for name, value in fields.items()}
    result["is_valid"] = [bool(value) for value in result["is_valid"]]
    # Null geometry stays joint-local without changing producer validity flags.
    for name, width in (("positions", 3), ("orientations_xyzw", 4)):
        result[name] = [row if _rows([row], width) else None for row in result[name]]
    result["radii"] = [value if _number(value) else None for value in result["radii"]]
    return {"valid": True, "reason": None, **result}


def _joint(sample):
    if sample is None:
        return _invalid("No JointState sample")
    rows = []
    for joint in sample.joints:
        # Preserve the SDK validity flag; absent numeric values remain unusable.
        rows.append({
            "name": joint.name,
            "position": joint.position if _number(joint.position) else None,
            "velocity": joint.velocity if _number(joint.velocity) else None,
            "effort": joint.effort if _number(joint.effort) else None,
            "valid": bool(joint.valid),
        })
    return {
        "valid": True, "reason": None, "device_id": sample.device_id,
        "joints": rows, "has_velocity": bool(sample.has_velocity),
        "has_effort": bool(sample.has_effort),
    }


def _real(stream, config, epoch, stop):
    import numpy as np
    from isaaccapture.deviceio_session import (
        DeviceIOSession, TrackerVendor, VendorConfig,
    )
    from isaaccapture.deviceio_trackers import (
        HandTracker, JointStateTracker, TactileTracker,
    )
    from isaaccapture.oxr import OpenXRSession

    tactile = {
        binding["tactile_collection_id"]: TactileTracker(
            binding["tactile_collection_id"]
        ) for binding in config.bindings.values()
    }
    joints = {binding["joint_source"]["collection_id"]: None
              for binding in config.bindings.values() if binding.get("joint_source")}
    joints = {collection: JointStateTracker(collection) for collection in joints}
    hand = HandTracker()
    trackers = [*tactile.values(), hand, *joints.values()]
    hand_config = next(iter(config.bindings.values()))["hand_source"]
    vendor = hand_config.get("vendor_id")
    pairs = [(hand, TrackerVendor(
        vendor, thaw(hand_config.get("vendor_params", {})),
    ))] if vendor else []
    vendor_config = VendorConfig(pairs)
    extensions = DeviceIOSession.get_required_extensions(trackers, vendor_config)
    baselines, seq = {}, 0
    with OpenXRSession("TactileMetadataEditor", extensions) as oxr:
        with DeviceIOSession.run(
            trackers, oxr.get_handles(), vendor_config=vendor_config,
        ) as session:
            while not stop.is_set():
                tick = time.monotonic()
                session.update()
                sources = {"tactile": {}, "hands": {}, "joints": {}}
                for collection, tracker in tactile.items():
                    metadata = tracker.get_metadata(session)
                    if metadata is not None:
                        values = _metadata(metadata, np)
                        fingerprint = metadata_fingerprint(values)
                        if collection not in baselines:
                            baselines[collection] = (fingerprint, len(
                                values["geometry_mapping"]["taxel_bind_poses"]
                            ))
                            _send_metadata(stream, epoch, collection, values)
                        elif baselines[collection][0] != fingerprint:
                            raise ValueError(
                                "Received tactile metadata changed within the session"
                            )
                    data = tracker.get_data(session)
                    if data is None:
                        sources["tactile"][collection] = _invalid("No tactile sample")
                    else:
                        forces = np.array(
                            data.taxel_forces, copy=True, order="C",
                        ).tolist()
                        if (
                            collection not in baselines
                            or len(forces) != baselines[collection][1]
                        ):
                            raise ValueError(
                                "Tactile frame count differs from producer metadata"
                            )
                        sources["tactile"][collection] = (
                            {"valid": True, "forces": forces, "reason": None}
                            if _rows(forces, 3)
                            else _invalid("Invalid tactile force values")
                        )
                for side in config.bindings:
                    sample = (
                        hand.get_left_hand(session) if side == "left"
                        else hand.get_right_hand(session)
                    )
                    sources["hands"][side] = _hand(sample, np)
                for collection, tracker in joints.items():
                    sources["joints"][collection] = _joint(tracker.get_data(session))
                send_message(stream, {
                    "type": "frame", "epoch": epoch, "seq": seq,
                    "read_monotonic_ns": time.monotonic_ns(), "sources": sources,
                })
                seq += 1
                stop.wait(max(0, 1/30 - (time.monotonic() - tick)))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        return 2
    stream, stop = None, threading.Event()
    try:
        with Path(args[0]).open(encoding="utf-8") as source:
            request = json.load(
                source, parse_constant=_reject_constant, parse_float=_finite_float,
            )
        if (
            type(request.get("version")) is not int or request["version"] != 1
            or type(request.get("port")) is not int or not 0 < request["port"] < 65536
            or not isinstance(request.get("token"), str)
            or not isinstance(request.get("epoch"), str)
        ):
            raise ValueError("Invalid local receiving request")
        config = ConnectConfig(
            request["bindings"], request.get("demo", False),
            request.get("demo_values", {}),
        )
        stream = socket.create_connection(("127.0.0.1", request["port"]), timeout=5)
        stream.settimeout(0.2)
        send_message(stream, {
            "version": 1, "type": "hello", "token": request["token"],
            "epoch": request["epoch"],
        })
        accepted = receive_message(stream, deadline=time.monotonic() + 5)
        if accepted != {"type": "status", "status": "accepted", "version": 1}:
            raise ValueError("Local receiving parent did not accept the handshake")
        control = threading.Thread(target=_control, args=(stream, stop), daemon=True)
        control.start()
        send_message(stream, {
            "type": "status", "status": "connecting", "epoch": request["epoch"],
        })
        if config.demo:
            _demo(stream, config, request["epoch"], stop)
        else:
            _real(stream, config, request["epoch"], stop)
        return 0
    except Exception as error:
        if stop.is_set() and isinstance(error, (OSError, EOFError)):
            return 0
        if stream is not None and not stop.is_set():
            try:
                send_message(stream, {
                    "type": "error", "epoch": request["epoch"],
                    "message": f"{type(error).__name__}: {error}",
                })
            except (OSError, ValueError):
                pass
        else:
            LOGGER.error(
                "Receiving worker could not start: %s: %s",
                type(error).__name__, error,
            )
        return 1
    finally:
        stop.set()
        if stream is not None:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
