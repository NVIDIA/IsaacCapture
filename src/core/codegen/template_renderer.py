# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load `templates/` bodies and substitute @KEY@ placeholders (stdlib-only)."""

from __future__ import annotations

from pathlib import Path

from templates import TrackerGenContext

TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


def _strip_spdx_header(text: str) -> str:
    """Drop leading // SPDX-* lines (and the blank line after) so emit-time COPYRIGHT is sole banner."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("// SPDX-"):
        i += 1
    if i > 0 and i < len(lines) and lines[i].strip() == "":
        i += 1
    return "".join(lines[i:])


def render_template(rel_path: str, values: dict[str, str]) -> str:
    text = _strip_spdx_header((TEMPLATE_ROOT / rel_path).read_text(encoding="utf-8"))
    # Longer keys first so @NAME@ does not corrupt @SCHEMA_NAME@ / @LIVE_IMPL_FILE@ / etc.
    for key, value in sorted(values.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(f"@{key}@", value)
    return text


def _tensor_identifier_expr(ctx: TrackerGenContext) -> str:
    if ctx.facade_tensor_constant:
        return f"std::string({ctx.cls}::TENSOR_IDENTIFIER)"
    return f'"{ctx.tensor_identifier}"'


def template_values(ctx: TrackerGenContext) -> dict[str, str]:
    """Flat @KEY@ map for per-tracker C++ templates."""
    values: dict[str, str] = {
        "CLASS": ctx.cls,
        "IFACE": ctx.iface,
        "NAME": ctx.name,
        "HEADER": ctx.header,
        "BASE_HEADER": ctx.base_header,
        "SCHEMA": ctx.schema,
        "DATA_TYPE": ctx.data_type,
        "RECORD_TYPE": ctx.record_type,
        "FB_TABLE": ctx.fb_table,
        "LIVE_IMPL": ctx.live_impl,
        "REPLAY_IMPL": ctx.replay_impl,
        "LIVE_IMPL_FILE": ctx.live_impl_file,
        "REPLAY_IMPL_FILE": ctx.replay_impl_file,
        "MCAP_CHANNELS_TYPE": ctx.mcap_channels_type,
        "SCHEMA_TRACKER_TYPE": ctx.schema_tracker_type,
        "MCAP_VIEWERS_TYPE": ctx.mcap_viewers_type,
        "TRAITS": ctx.traits,
        "MAX_FLATBUFFER_SIZE": str(ctx.max_flatbuffer_size),
        "TENSOR_IDENTIFIER": ctx.tensor_identifier,
        "LOCALIZED_NAME": ctx.localized_name,
        "PYTHON_ACCESSOR": ctx.python_accessor,
        "TENSOR_IDENTIFIER_EXPR": _tensor_identifier_expr(ctx),
        "HAS_TENSOR_IDENTIFIER": "1" if ctx.facade_tensor_constant else "0",
        "HAS_METADATA": "1" if ctx.metadata else "0",
        "METADATA": ctx.metadata or "",
        "METADATA_SCHEMA": ctx.metadata_schema,
        "METADATA_CHANNEL": ctx.metadata_channel,
    }
    if ctx.mcap_channels is not None:
        values["RECORDING_CHANNELS"] = ", ".join(f'"{c}"' for c in ctx.mcap_channels)
    if ctx.replay_channels is not None:
        values["REPLAY_CHANNELS"] = ", ".join(f'"{c}"' for c in ctx.replay_channels)
    return values


def render_with_copyright(rel_path: str, ctx: TrackerGenContext, copyright: str) -> str:
    return copyright + render_template(rel_path, template_values(ctx))
