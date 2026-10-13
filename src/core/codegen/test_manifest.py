# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tracker manifest placeholder resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from manifest import load_defaults, load_manifest, resolve_tracker_entry, snake_to_camel


CORE_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = CORE_ROOT / "deviceio_trackers" / "defaults.toml"


class SnakeToCamelTest(unittest.TestCase):
    def test_generic_pedal(self) -> None:
        self.assertEqual(snake_to_camel("generic_3axis_pedal"), "Generic3AxisPedal")

    def test_se3(self) -> None:
        self.assertEqual(snake_to_camel("se3_tracker"), "Se3Tracker")


class ManifestResolverTest(unittest.TestCase):
    def test_cycle_is_rejected(self) -> None:
        defaults = {"defaults": {"a": "%b%", "b": "%a%"}}
        with self.assertRaises(ValueError) as ctx:
            resolve_tracker_entry(
                {"name": "x", "table": "T", "a": "%b%", "b": "%a%"}, defaults
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_missing_reference(self) -> None:
        defaults = {"defaults": {"tensor_identifier": "%missing%"}}
        with self.assertRaises(ValueError):
            resolve_tracker_entry({"name": "x", "table": "T"}, defaults)

    def test_joint_state_defaults(self) -> None:
        entry = resolve_tracker_entry(
            {
                "name": "joint_state",
                "table": "JointStateOutput",
                "max_flatbuffer_size": 4096,
            },
            load_defaults(DEFAULTS_PATH),
        )
        self.assertEqual(entry["class"], "JointStateTracker")
        self.assertEqual(entry["schema"], "joint_state")
        self.assertEqual(entry["traits"], "JointStateRecordingTraits")
        self.assertEqual(entry["mcap_channels"], ["joint_state", "joint_state_tracked"])

    def test_se3_class_override(self) -> None:
        entry = resolve_tracker_entry(
            {
                "name": "se3_tracker",
                "table": "Se3TrackerPose",
                "class": "Se3Tracker",
                "max_flatbuffer_size": 256,
            },
            load_defaults(DEFAULTS_PATH),
        )
        self.assertEqual(entry["class"], "Se3Tracker")
        self.assertEqual(entry["tensor_identifier"], "se3_tracker")

    def test_plugin_device_status_contract(self) -> None:
        entries = load_manifest(
            CORE_ROOT / "deviceio_trackers" / "trackers.toml", DEFAULTS_PATH
        )
        entry = next(item for item in entries if item["name"] == "plugin_device_status")
        self.assertEqual(entry["table"], "PluginDeviceStatusSnapshot")
        self.assertEqual(entry["class"], "PluginDeviceStatusTracker")
        self.assertEqual(entry["max_flatbuffer_size"], 4096)
        self.assertTrue(entry["facade_tensor_constant"])
        self.assertEqual(entry["python_accessor"], "get_device_status_snapshot")

    def test_pedal_schema_and_traits_override(self) -> None:
        entry = resolve_tracker_entry(
            {
                "name": "generic_3axis_pedal",
                "table": "Generic3AxisPedalOutput",
                "schema": "pedals",
                "channel": "pedals",
                "traits": "PedalRecordingTraits",
                "max_flatbuffer_size": 256,
                "python_accessor": "get_pedal_data",
            },
            load_defaults(DEFAULTS_PATH),
        )
        self.assertEqual(entry["schema"], "pedals")
        self.assertEqual(entry["traits"], "PedalRecordingTraits")
        self.assertEqual(entry["python_accessor"], "get_pedal_data")

    def test_push_direction_defaults(self) -> None:
        entry = resolve_tracker_entry(
            {
                "name": "haptic_command",
                "direction": "push",
                "table": "HapticCommand",
                "tensor_identifier": "haptic_command",
            },
            load_defaults(DEFAULTS_PATH),
        )
        self.assertEqual(entry["class"], "HapticCommandPushTracker")
        self.assertFalse(entry["record"])
        self.assertEqual(entry["python_accessor"], "push")
        self.assertNotIn("pull", entry)
        self.assertNotIn("mcap_channels", entry)

    def test_pull_direction_has_no_push_overlay_keys(self) -> None:
        entry = resolve_tracker_entry(
            {"name": "joint_state", "table": "JointStateOutput"},
            load_defaults(DEFAULTS_PATH),
        )
        self.assertEqual(entry["direction"], "pull")
        self.assertNotIn("push", entry)

    def test_unknown_direction_is_rejected_before_defaults_merge(self) -> None:
        # Only ``defaults.pull`` carries required keys. Merging under direction="bogus"
        # would either hang on missing placeholders or invent a silent overlay — so
        # rejection must happen before ``_merge_defaults``.
        defaults = {
            "defaults": {
                "direction": "pull",
                "pull": {
                    "shape": "single_collection",
                    "schema": "%name%",
                    "class": "%name_CamelCase%Tracker",
                    "tensor_identifier": "%name%",
                    "localized_name": "%class%",
                    "channel": "%name%",
                    "traits": "%channel_CamelCase%RecordingTraits",
                    "python_accessor": "get_data",
                    "max_flatbuffer_size": 512,
                    "record": True,
                    "facade_tensor_constant": False,
                    "mcap_channels": ["x"],
                    "replay_channels": ["x"],
                },
            }
        }
        with self.assertRaises(ValueError) as ctx:
            resolve_tracker_entry(
                {"name": "x", "table": "T", "direction": "bogus"}, defaults
            )
        message = str(ctx.exception).lower()
        self.assertIn("invalid direction", message)
        self.assertIn("bogus", message)
        # Must not fall through to placeholder / missing-key errors from a bad merge.
        self.assertNotIn("could not resolve", message)
        self.assertNotIn("missing required", message)

    def test_single_collection_without_record_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_tracker_entry(
                {
                    "name": "x",
                    "table": "T",
                    "shape": "single_collection",
                    "record": False,
                },
                load_defaults(DEFAULTS_PATH),
            )
        message = str(ctx.exception).lower()
        self.assertIn("single_collection", message)
        self.assertIn("record=true", message)

    def test_record_must_be_boolean(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_tracker_entry(
                {"name": "x", "table": "T", "record": "false"},
                load_defaults(DEFAULTS_PATH),
            )
        message = str(ctx.exception).lower()
        self.assertIn("record", message)
        self.assertIn("boolean", message)

    def test_load_manifest_file(self) -> None:
        manifest = CORE_ROOT / "deviceio_trackers" / "trackers.toml"
        self.assertTrue(manifest.is_file(), f"missing required manifest: {manifest}")
        entries = load_manifest(manifest, DEFAULTS_PATH)
        self.assertGreaterEqual(len(entries), 1)

    def test_header_override_preserves_file_stems(self) -> None:
        from templates import enrich_context

        entry = resolve_tracker_entry(
            {
                "name": "frame_metadata_oak",
                "table": "FrameMetadataOak",
                "class": "FrameMetadataTrackerOak",
                "schema": "oak",
                "tensor_identifier": "frame_metadata",
                "channel": "oak",
                "header": "frame_metadata_tracker_oak",
                "max_flatbuffer_size": 128,
            },
            load_defaults(DEFAULTS_PATH),
        )
        ctx = enrich_context(entry)
        self.assertEqual(ctx.header, "frame_metadata_tracker_oak")
        self.assertEqual(ctx.base_header, "frame_metadata_tracker_oak_base")
        self.assertEqual(ctx.live_impl_file, "live_frame_metadata_tracker_oak_impl")
        self.assertEqual(ctx.replay_impl_file, "replay_frame_metadata_tracker_oak_impl")
        self.assertEqual(ctx.cls, "FrameMetadataTrackerOak")
        self.assertEqual(ctx.traits, "OakRecordingTraits")


class RecordTypeTest(unittest.TestCase):
    def _resolve(self, **overrides) -> dict:
        raw = {"name": "tactile_shape", "table": "TactileShape", **overrides}
        return resolve_tracker_entry(raw, load_defaults(DEFAULTS_PATH))

    def test_defaults_to_the_record_wrapper(self) -> None:
        self.assertEqual(self._resolve()["record_type"], "TactileShapeRecord")

    def test_table_recorded_as_itself(self) -> None:
        from generate_trackers import render_for_entry
        from templates import enrich_context

        files = render_for_entry(
            enrich_context(self._resolve(record_type="TactileShape"))
        )
        live = files["live_trackers/live_tactile_shape_tracker_impl.hpp"]
        self.assertIn("SchemaTracker<TactileShape, TactileShape>", live)
        self.assertIn("McapTrackerChannels<TactileShape>", live)
        replay = files["replay_trackers/replay_tactile_shape_tracker_impl.hpp"]
        self.assertIn("McapTrackerViewers<TactileShape>", replay)


if __name__ == "__main__":
    unittest.main()
