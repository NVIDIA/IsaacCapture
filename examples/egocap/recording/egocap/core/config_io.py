# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration I/O - Load and save YAML files"""

import os

import yaml

from egocap.core.data_root import data_root

try:
    from ruamel.yaml import YAML

    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False


class ConfigIO:
    """Handle YAML configuration file operations"""

    def __init__(self, config_path):
        self.config_path = config_path

    def load(self):
        """Load YAML configuration from file"""
        if not os.path.exists(self.config_path):
            # Return default structure if file doesn't exist
            return self._get_default_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config if config else self._get_default_config()

    def _dump_yaml(self, stream, config_data):
        """Serialize config_data to a stream with the recording YAML formatting."""

        # Use a custom Dumper class to handle specific formatting.
        # ignore_aliases prevents repeated values (e.g. camera_resolution [640, 400])
        # from being written as YAML anchors (&id001 / *id001).
        class CustomDumper(yaml.Dumper):
            def ignore_aliases(self, data):
                return True

        def represent_list(dumper, data):
            # Only make numeric lists with 2-4 items inline (like coordinates/resolutions)
            # Check if all items are numbers
            if len(data) <= 4 and all(isinstance(x, (int, float)) for x in data):
                return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

        def represent_str(dumper, data):
            # Force quotes on path-like strings (directories).
            if "/" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        CustomDumper.add_representer(list, represent_list)
        CustomDumper.add_representer(str, represent_str)

        yaml.dump(
            config_data,
            stream,
            Dumper=CustomDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )

    def save(self, config_data):
        """Save configuration to YAML file with inline lists for coordinate arrays"""
        # Ensure directory exists
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(self.config_path, "w") as f:
            self._dump_yaml(f, config_data)

    def load_with_comments(self):
        """Load YAML configuration preserving comments and formatting

        Uses ruamel.yaml to preserve comments, whitespace, and structure.
        Falls back to standard yaml.safe_load if ruamel.yaml is not available.
        """
        if not os.path.exists(self.config_path):
            return self._get_default_config()

        if RUAMEL_AVAILABLE:
            yaml_loader = YAML()
            yaml_loader.preserve_quotes = True
            yaml_loader.default_flow_style = False
            with open(self.config_path, "r") as f:
                config = yaml_loader.load(f)
            return config if config else self._get_default_config()
        else:
            # Fallback to standard load
            return self.load()

    def save_with_comments(self, config_data):
        """Save configuration preserving comments and formatting

        Uses ruamel.yaml to preserve comments, whitespace, and structure.
        Falls back to standard yaml.dump if ruamel.yaml is not available.
        """
        # Ensure directory exists
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if RUAMEL_AVAILABLE:
            yaml_saver = YAML()
            yaml_saver.preserve_quotes = True
            yaml_saver.default_flow_style = False
            yaml_saver.indent(mapping=2, sequence=2, offset=2)
            # Prevent repeated values (e.g. camera_resolution [640, 400]) from
            # being written as YAML anchors (&id001 / *id001). Assigned as an
            # instance attribute, not patched onto the class, so ruamel calls
            # it as `ignore_aliases(data)` with no implicit `self` -- one arg.
            yaml_saver.representer.ignore_aliases = lambda data: True
            with open(self.config_path, "w") as f:
                yaml_saver.dump(config_data, f)
        else:
            # Fallback to standard save
            self.save(config_data)

    @staticmethod
    def _get_default_config():
        """Return default configuration structure: the recorded-dataset source
        schema the downstream data loader and the training conversion pipeline
        require (see ../../README.md)."""
        return {
            "recording": {
                "data_dir": str(data_root() / "recordings"),
                "record_fps": 60,
                "stream_fps": 60,
                "verbose": False,
                "video_frame_type": "RGBCameraFrame",
                "numeric_frame_type": "NumericFrame",
            },
            "video_sources": [],
            "numeric_sources": [],
        }
