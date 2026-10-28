# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit EGO recording clocks from annotated physical events and held-out evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaccapture_examples.orbbec_ego.calibration import (
    PairedTimeEvent,
    evidence_sha256,
    fit_time_alignment,
    make_recording_alignment,
)


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _file(base: Path, value, name: str) -> Path:
    path = (base / _text(value, name)).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must name a regular file: {path}")
    return path


def calibrate_manifest(manifest_path: str | Path) -> dict:
    """Resolve annotation evidence paths against the manifest, without opening devices."""
    path = Path(manifest_path).resolve(strict=True)
    manifest = _object(
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        ),
        "manifest",
    )
    if _integer(manifest["version"], "version") != 1:
        raise ValueError("Only version-1 event manifests are supported")
    model_id = _text(manifest["model_id"], "model_id")
    identity = _object(manifest["identity"], "identity")
    profile = _object(manifest["profile"], "profile")
    fps = _integer(_object(profile["video"], "profile.video")["fps"], "fps")
    recording = _file(path.parent, manifest["recording"], "recording")
    media = [
        _file(path.parent, value, "media_paths")
        for value in _array(manifest.get("media_paths", []), "media_paths")
    ]
    hashes: dict[Path, str] = {}

    def events(values, name):
        result = []
        for value in _array(values, name):
            value = _object(value, name)
            evidence = _file(path.parent, value["evidence"], "event.evidence")
            if evidence not in hashes:
                hashes[evidence] = "sha256:" + evidence_sha256(evidence)
            result.append(
                PairedTimeEvent(
                    _text(value["event_id"], "event_id"),
                    _integer(value["source_value"], "source_value"),
                    _integer(value["reference_time_ns"], "reference_time_ns"),
                    _integer(value["uncertainty_ns"], "uncertainty_ns"),
                    hashes[evidence],
                )
            )
        return result

    def fit(value, *, begin, end, audio=False):
        training = events(value["calibration_events"], "calibration_events")
        held_out = events(value["validation_events"], "validation_events")
        validation_start = _integer(value["validation_start_ns"], "validation_start_ns")
        validation_end = _integer(value["validation_end_ns"], "validation_end_ns")
        if not audio and not validation_start <= begin < end <= validation_end:
            raise ValueError(
                "Effective epoch interval exceeds independent validation bounds"
            )
        if audio:
            first = _integer(value["first_sample_index"], "first_sample_index")
            count = _integer(value["sample_count"], "sample_count")
            if any(
                not first <= event.source_value < first + count
                for event in (*training, *held_out)
            ):
                raise ValueError("Audio event is outside its declared sample segment")
        return fit_time_alignment(
            training,
            held_out,
            fps=fps,
            validation_start_ns=validation_start,
            validation_end_ns=validation_end,
            source_unit="samples" if audio else "ns",
            sample_rate_hz=(
                _integer(
                    _object(profile["audio"], "profile.audio")["sample_rate_hz"],
                    "sample_rate_hz",
                )
                if audio
                else None
            ),
        )

    epochs = []
    for epoch in _array(manifest["epochs"], "epochs"):
        epoch = _object(epoch, "epoch")
        begin = _integer(epoch["valid_from_ns"], "valid_from_ns")
        end = _integer(epoch["valid_to_ns"], "valid_to_ns")
        streams = {}
        for name, value in _object(epoch["streams"], "streams").items():
            value = _object(value, name)
            if name == "audio":
                segments = []
                for segment in _array(value["segments"], "audio.segments"):
                    segment = _object(segment, "audio segment")
                    continuity = segment.get("continuity_verified", False)
                    if type(continuity) is not bool:
                        raise ValueError("continuity_verified must be a boolean")
                    segments.append(
                        {
                            "segment_id": _integer(segment["segment_id"], "segment_id"),
                            "first_sample_index": _integer(
                                segment["first_sample_index"], "first_sample_index"
                            ),
                            "sample_count": _integer(
                                segment["sample_count"], "sample_count"
                            ),
                            "continuity_verified": continuity,
                            "fit": fit(segment, begin=begin, end=end, audio=True),
                        }
                    )
                streams[name] = segments
            elif name in ("right_rgb", "accel", "gyro"):
                streams[name] = fit(value, begin=begin, end=end)
            else:
                raise ValueError(f"Unsupported fitted stream: {name}")
        epochs.append(
            {
                "epoch": _integer(epoch["epoch"], "epoch"),
                "valid_from_ns": begin,
                "valid_to_ns": end,
                "streams": streams,
            }
        )
    return make_recording_alignment(
        recording,
        model_id=model_id,
        identity=identity,
        profile=profile,
        epochs=epochs,
        media_paths=media,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit EGO time alignment from manually annotated physical events. "
            "This command does not detect events or access hardware."
        )
    )
    parser.add_argument("manifest", type=Path, help="Version-1 event manifest JSON")
    parser.add_argument(
        "--output", required=True, type=Path, help="New alignment JSON path"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite alignment output: {args.output}"
            )
        document = calibrate_manifest(args.manifest)
        encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f"EGO alignment status={document['status']}; output={args.output}")
    return 0 if document["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
