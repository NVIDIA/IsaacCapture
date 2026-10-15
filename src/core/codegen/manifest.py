# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve tracker manifest entries with defaults.toml placeholder expansion."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REQUIRED_KEYS = frozenset({"name", "table"})
VALID_DIRECTIONS = frozenset({"pull", "push"})

# Keys that must be present after resolution for code generation.
RESOLVED_REQUIRED = frozenset(
    {
        "name",
        "direction",
        "shape",
        "schema",
        "table",
        "class",
        "tensor_identifier",
        "localized_name",
        "channel",
        "traits",
        "python_accessor",
        "max_flatbuffer_size",
        "record",
        "facade_tensor_constant",
        "metadata",
        "metadata_schema",
        "metadata_channel",
    }
)

_PLACEHOLDER = re.compile(
    r"%(?P<key>[a-zA-Z0-9_]+?)(?P<transform>_(?:CamelCase|snake_case|UPPER))?%|(?P<literal>%%)"
)


def snake_to_camel(value: str) -> str:
    """Convert snake_case to CamelCase, capitalizing after digits (3axis -> 3Axis)."""
    parts = value.split("_")
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part[0].isdigit():
            # Capitalize first alpha after leading digits.
            idx = 0
            while idx < len(part) and part[idx].isdigit():
                idx += 1
            if idx < len(part):
                out.append(part[:idx] + part[idx].upper() + part[idx + 1 :])
            else:
                out.append(part)
        else:
            out.append(part[0].upper() + part[1:])
    return "".join(out)


def camel_to_snake(value: str) -> str:
    """Convert CamelCase to snake_case (TactileShape -> tactile_shape)."""
    return re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", value
    ).lower()


def apply_transform(value: str, transform: str | None) -> str:
    if transform == "_CamelCase":
        return snake_to_camel(value)
    if transform == "_snake_case":
        return camel_to_snake(value)
    if transform == "_UPPER":
        return value.upper()
    return value


def expand_string(template: str, values: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        if match.group("literal"):
            return "%"
        key = match.group("key")
        transform = match.group("transform")
        if key not in values:
            raise KeyError(key)
        raw = values[key]
        if isinstance(raw, (list, dict, bool, int, float)):
            raise TypeError(
                f"placeholder %{key}% requires a string value, got {type(raw).__name__}"
            )
        return apply_transform(str(raw), transform)

    return _PLACEHOLDER.sub(repl, template)


def _expand_value(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return expand_string(value, values)
    if isinstance(value, list):
        return [_expand_value(item, values) for item in value]
    if isinstance(value, dict):
        return {k: _expand_value(v, values) for k, v in value.items()}
    return value


def _merge_defaults(
    base: dict[str, Any],
    direction: str,
    defaults_doc: dict[str, Any],
) -> dict[str, Any]:
    root = dict(defaults_doc.get("defaults", {}))
    overlay = root.pop(direction, {})
    if not isinstance(overlay, dict):
        overlay = {}
    for other_direction in VALID_DIRECTIONS:
        root.pop(other_direction, None)
    merged = root
    merged.update(overlay)
    merged.update(base)
    return merged


def resolve_tracker_entry(
    raw: dict[str, Any],
    defaults_doc: dict[str, Any],
) -> dict[str, Any]:
    if "name" not in raw:
        raise ValueError("tracker entry missing required key 'name'")
    if "table" not in raw:
        raise ValueError(f"tracker '{raw['name']}': missing required key 'table'")
    if "record_type" in raw:
        raise ValueError(
            f"tracker '{raw['name']}': record_type is no longer supported; "
            "generated trackers always record <Table>Record"
        )

    direction = str(
        raw.get("direction", defaults_doc.get("defaults", {}).get("direction", "pull"))
    )
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"tracker '{raw['name']}': invalid direction '{direction}' "
            f"(expected one of: {', '.join(sorted(VALID_DIRECTIONS))})"
        )
    working = _merge_defaults(raw, direction, defaults_doc)

    resolved: dict[str, Any] = {"name": raw["name"], "table": raw["table"]}
    pending = {k: v for k, v in working.items() if k not in ("name", "table")}

    # Seed with explicit name/table so %name% / %table% work immediately.
    resolved["name"] = raw["name"]
    resolved["table"] = raw["table"]

    max_passes = len(pending) + 8
    for _ in range(max_passes):
        progressed = False
        for key in list(pending.keys()):
            value = pending[key]
            try:
                new_value = _expand_value(value, resolved)
            except KeyError:
                continue
            resolved[key] = new_value
            del pending[key]
            progressed = True
        if not pending:
            break
        if not progressed:
            missing = ", ".join(sorted(pending))
            raise ValueError(
                f"tracker '{raw['name']}': could not resolve keys (cycle or missing reference): {missing}"
            )

    missing_required = sorted(RESOLVED_REQUIRED - resolved.keys())
    if missing_required:
        raise ValueError(
            f"tracker '{raw['name']}': missing required keys after resolution: {', '.join(missing_required)}"
        )

    record = resolved["record"]
    if not isinstance(record, bool):
        raise ValueError(
            f"tracker '{raw['name']}': record must be a boolean, got {type(record).__name__}"
        )

    if resolved["metadata"] and resolved["direction"] != "pull":
        raise ValueError(
            f"tracker '{raw['name']}': metadata is supported only for direction=pull"
        )

    if resolved["direction"] == "pull" and record:
        for key in ("mcap_channels", "replay_channels"):
            if key not in resolved:
                raise ValueError(
                    f"tracker '{raw['name']}': missing required key '{key}' for direction=pull"
                )

    # single_collection templates always emit MCAP channel/viewer constructors; the
    # factory only omits them when record=false. Reject the mismatch rather than
    # generating sources that cannot link against the factory methods.
    if (
        resolved["direction"] == "pull"
        and resolved.get("shape") == "single_collection"
        and not record
    ):
        raise ValueError(
            f"tracker '{raw['name']}': shape=single_collection requires record=true "
            "(non-recording multi-sample readers such as HapticCommandReaderTracker stay hand-written)"
        )

    return resolved


def load_defaults(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_manifest(manifest_path: Path, defaults_path: Path) -> list[dict[str, Any]]:
    defaults_doc = load_defaults(defaults_path)
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    entries = manifest.get("tracker", [])
    if not isinstance(entries, list):
        raise ValueError("trackers.toml: 'tracker' must be an array")
    return [resolve_tracker_entry(entry, defaults_doc) for entry in entries]


def print_resolved(manifest_path: Path, defaults_path: Path) -> None:
    for entry in load_manifest(manifest_path, defaults_path):
        print(json.dumps(entry, indent=2, sort_keys=True))
        print()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Resolve tracker manifest entries")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--print-resolved", action="store_true")
    args = parser.parse_args(argv)

    if args.print_resolved:
        print_resolved(args.manifest, args.defaults)
        return 0

    load_manifest(args.manifest, args.defaults)
    return 0


if __name__ == "__main__":
    sys.exit(main())
