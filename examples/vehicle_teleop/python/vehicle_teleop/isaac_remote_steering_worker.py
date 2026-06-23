# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager

import isaacteleop.deviceio as deviceio
import isaacteleop.oxr as oxr
import yaml
import zmq
from isaacteleop.retargeters import (
    VehicleControlRetargeter,
    VehicleControlRetargeterConfig,
)
from isaacteleop.schema import SteeringWheelOutput

from vehicle_teleop.command_log import McapCommandLogger, SteeringWheelSample
from vehicle_teleop.vehicle_command import VehicleControlCommand


DEFAULT_BIND = "tcp://*:5555"
DEFAULT_TOPIC = "kia_control"
DEFAULT_COLLECTION_ID = "steering_wheel"
DEFAULT_DEVICE_PATH = "/dev/input/js0"
DEFAULT_FIRST_SAMPLE_TIMEOUT_S = 5.0
DEFAULT_AXIS_MAPPING = {
    "steering_axis": 0,
    "throttle_axis": 1,
    "brake_axis": 2,
    "clutch_axis": -1,
}


def example_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return example_root() / "config" / "steering_wheel_config.yaml"


def default_plugin_binary() -> Path:
    root = example_root()
    candidates = [
        root.parents[1]
        / "build"
        / "src"
        / "plugins"
        / "steering_wheel"
        / "steering_wheel_plugin",
        root.parents[1] / "plugins" / "steering_wheel" / "steering_wheel_plugin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class NativeSteeringWheelPlugin:
    def __init__(
        self,
        *,
        binary: Path,
        device_path: str,
        collection_id: str,
        steering_axis: int,
        throttle_axis: int,
        brake_axis: int,
        clutch_axis: int,
    ) -> None:
        self._command = [
            str(binary),
            device_path,
            collection_id,
            str(steering_axis),
            str(throttle_axis),
            str(brake_axis),
            str(clutch_axis),
        ]
        self._process: subprocess.Popen | None = None

    @property
    def command(self) -> list[str]:
        return list(self._command)

    def __enter__(self) -> "NativeSteeringWheelPlugin":
        self._process = subprocess.Popen(self._command)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None


def load_axis_mapping_from_config(path: str | Path) -> dict[str, int]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Steering wheel config must be a YAML mapping: {config_path}")
    wheel = data.get("g920_racing_wheel")
    if not isinstance(wheel, dict):
        raise KeyError(
            f"Steering wheel config missing g920_racing_wheel section: {config_path}"
        )

    mapping = dict(DEFAULT_AXIS_MAPPING)
    mapping["steering_axis"] = int(wheel["steering_wheel"])
    mapping["throttle_axis"] = int(wheel["throttle"])
    mapping["brake_axis"] = int(wheel["brake"])
    if "clutch" in wheel:
        mapping["clutch_axis"] = int(wheel["clutch"])
    return mapping


def resolve_axis_mapping(args: argparse.Namespace) -> dict[str, int]:
    mapping = dict(DEFAULT_AXIS_MAPPING)
    if args.config:
        mapping.update(load_axis_mapping_from_config(args.config))
    for arg_name in ("steering_axis", "throttle_axis", "brake_axis", "clutch_axis"):
        value = getattr(args, arg_name)
        if value is not None:
            mapping[arg_name] = int(value)
    return mapping


class IsaacRemoteSteeringWorker:
    def __init__(self, args: argparse.Namespace):
        self._bind = args.bind
        self._topic = args.topic
        self._rate_hz = args.rate_hz
        self._collection_id = args.collection_id
        self._device_path = args.device
        self._plugin_binary = Path(args.plugin_binary)
        self._axis_args = resolve_axis_mapping(args)
        self._start_plugin = not args.no_start_plugin
        self._first_sample_timeout_s = args.first_sample_timeout_s
        self._verbose = args.verbose
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._tracker = deviceio.SteeringWheelTracker(self._collection_id)
        self._deviceio_session = None
        self._retargeter = VehicleControlRetargeter(
            VehicleControlRetargeterConfig(steer_scale=-1.0)
        )
        self._log_mcap = args.log_mcap
        self._logger = None
        self._sequence = 0
        self._running = True

    def run(self) -> None:
        self._register_signal_handlers()
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind)

        plugin_ctx = self._plugin_context()
        trackers = [self._tracker]
        required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
        logger_ctx: ContextManager = (
            McapCommandLogger(self._log_mcap) if self._log_mcap else nullcontext()
        )
        try:
            with (
                plugin_ctx,
                oxr.OpenXRSession(
                    "VehicleTeleopSteeringWorker", required_extensions
                ) as oxr_session,
            ):
                handles = oxr_session.get_handles()
                with deviceio.DeviceIOSession.run(
                    trackers, handles
                ) as self._deviceio_session:
                    with logger_ctx as self._logger:
                        self._capture_steering_neutral()
                        period_s = 1.0 / self._rate_hz
                        print(
                            f"Publishing IsaacTeleop steering commands on {self._bind} "
                            f"topic={self._topic!r} at {self._rate_hz:.1f} Hz"
                        )
                        print(
                            f"Reading steering wheel through IsaacTeleop DeviceIO collection "
                            f"{self._collection_id!r}"
                        )
                        self._run_loop(period_s)
        finally:
            self._publish_neutral()
            self._socket.close()
            self._context.term()
            print("\nIsaacTeleop vehicle steering worker stopped.")

    def _plugin_context(self) -> ContextManager:
        if not self._start_plugin:
            return nullcontext()
        if not self._plugin_binary.exists():
            raise FileNotFoundError(
                f"IsaacTeleop steering wheel plugin not found at {self._plugin_binary}. "
                "Build IsaacTeleop first, or pass --plugin-binary/--no-start-plugin."
            )
        plugin = NativeSteeringWheelPlugin(
            binary=self._plugin_binary,
            device_path=self._device_path,
            collection_id=self._collection_id,
            **self._axis_args,
        )
        print("Starting native IsaacTeleop steering wheel plugin:")
        print(" ".join(plugin.command))
        return plugin

    def _run_loop(self, period_s: float) -> None:
        while self._running:
            started = time.monotonic()
            self._publish_next_command()
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    def _publish_next_command(self) -> None:
        sample = self._read_isaac_sample()
        command = isaac_command_to_wire_command(
            self._retargeter.retarget(sample, sequence=self._sequence),
            timestamp_ns=time.time_ns(),
        )
        self._publish(command)
        if self._logger is not None:
            self._logger.write(
                sample=wire_sample_from_isaac_sample(
                    sample, timestamp_ns=command.timestamp_ns
                ),
                command=command,
            )
        if self._verbose:
            self._print_status_line(
                f"seq={command.sequence} accel={command.accel:+.3f} steer={command.steer:+.3f} "
                f"throttle={command.throttle:.3f} brake={command.brake:.3f}"
            )
        self._sequence += 1

    @staticmethod
    def _print_status_line(text: str) -> None:
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        if columns > 1:
            text = text[: columns - 1]
        print(f"\r\033[K{text}", end="", flush=True)

    def _read_isaac_sample(self) -> SteeringWheelOutput:
        if self._deviceio_session is None:
            raise RuntimeError("DeviceIO session is not initialized")
        self._deviceio_session.update()
        tracked = self._tracker.get_wheel_data(self._deviceio_session)
        if tracked.data is None:
            raise RuntimeError(
                "No steering wheel data available from IsaacTeleop. "
                "Check that the native steering_wheel_plugin is running and publishing "
                f"collection {self._collection_id!r}."
            )
        return tracked.data

    def _wait_for_first_sample(self) -> SteeringWheelOutput:
        deadline = time.monotonic() + self._first_sample_timeout_s
        last_error: RuntimeError | None = None
        while self._running and time.monotonic() < deadline:
            try:
                return self._read_isaac_sample()
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Stopped before receiving steering wheel data")

    def _publish_neutral(self) -> None:
        self._publish(VehicleControlCommand.neutral(sequence=self._sequence))

    def _publish(self, command: VehicleControlCommand) -> None:
        payload = json.dumps(command.to_dict(), separators=(",", ":"))
        self._socket.send_string(f"{self._topic} {payload}")

    def _capture_steering_neutral(self) -> None:
        sample = self._wait_for_first_sample()
        self._retargeter.calibrate_neutral(sample)
        print(f"Steering neutral offset: {self._retargeter.steering_neutral:+.3f}")

    def stop(self, _signum=None, _frame=None) -> None:
        self._running = False

    def _register_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)


def isaac_command_to_wire_command(
    command, *, timestamp_ns: int
) -> VehicleControlCommand:
    return VehicleControlCommand(
        sequence=int(command.sequence),
        timestamp_ns=timestamp_ns,
        steer=float(command.steer),
        accel=float(command.accel),
        throttle=float(command.throttle),
        brake=float(command.brake),
    )


def wire_sample_from_isaac_sample(
    sample: SteeringWheelOutput, *, timestamp_ns: int
) -> SteeringWheelSample:
    return SteeringWheelSample(
        steering=float(sample.steering),
        accel_axis=float(sample.throttle),
        brake_axis=float(sample.brake),
        timestamp_ns=timestamp_ns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read steering wheel input through IsaacTeleop and publish Kia "
            "vehicle commands over ZMQ."
        )
    )
    parser.add_argument("--bind", default=DEFAULT_BIND, help="ZMQ PUB bind address.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ZMQ topic prefix.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Publish rate.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE_PATH,
        help="Linux joystick device path for the native plugin.",
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help=(
            "Steering wheel YAML config used to resolve axis indexes. Pass an "
            "empty string to use plugin defaults."
        ),
    )
    parser.add_argument(
        "--collection-id",
        default=DEFAULT_COLLECTION_ID,
        help="IsaacTeleop tensor collection id.",
    )
    parser.add_argument(
        "--plugin-binary",
        default=str(default_plugin_binary()),
        help="Path to steering_wheel_plugin.",
    )
    parser.add_argument(
        "--steering-axis",
        type=int,
        default=None,
        help="Override joystick axis index for steering.",
    )
    parser.add_argument(
        "--throttle-axis",
        type=int,
        default=None,
        help="Override joystick axis index for throttle.",
    )
    parser.add_argument(
        "--brake-axis",
        type=int,
        default=None,
        help="Override joystick axis index for brake.",
    )
    parser.add_argument(
        "--clutch-axis",
        type=int,
        default=None,
        help="Override joystick axis index for clutch, or -1 to disable.",
    )
    parser.add_argument(
        "--no-start-plugin",
        action="store_true",
        help="Do not launch steering_wheel_plugin; read from an already running publisher.",
    )
    parser.add_argument(
        "--first-sample-timeout-s",
        type=float,
        default=DEFAULT_FIRST_SAMPLE_TIMEOUT_S,
        help="Seconds to wait for the first IsaacTeleop steering sample before failing.",
    )
    parser.add_argument(
        "--log-mcap",
        default=None,
        help="Record raw-axis samples and commands to an MCAP log.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print live control values."
    )
    return parser


def main() -> None:
    IsaacRemoteSteeringWorker(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
