# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import select
import signal
import sys
import termios
import time
from dataclasses import dataclass

import zmq
from isaacteleop.retargeters import (
    VehicleControlRetargeter,
    VehicleControlRetargeterConfig,
)
from isaacteleop.schema import SteeringWheelOutput

from vehicle_teleop.vehicle_command import VehicleControlCommand, clamp


DEFAULT_BIND = "tcp://*:5555"
DEFAULT_TOPIC = "kia_control"


@dataclass
class IsaacKeyboardControlState:
    axis_increment: float = 0.05
    gas_brake: float = 0.0
    steer: float = 0.0
    quit_requested: bool = False

    def apply_key(self, key: str) -> bool:
        normalized = key.lower()
        if normalized == "w":
            self.gas_brake = clamp(self.gas_brake + self.axis_increment, -1.0, 1.0)
        elif normalized == "s":
            self.gas_brake = clamp(self.gas_brake - self.axis_increment, -1.0, 1.0)
        elif normalized == "a":
            self.steer = clamp(self.steer + self.axis_increment, -1.0, 1.0)
        elif normalized == "d":
            self.steer = clamp(self.steer - self.axis_increment, -1.0, 1.0)
        elif normalized in ("r", "c"):
            self.reset()
        elif normalized in ("q", "\x1b"):
            self.quit_requested = True
        else:
            return False
        return True

    def reset(self) -> None:
        self.gas_brake = 0.0
        self.steer = 0.0

    def to_isaac_sample(self) -> SteeringWheelOutput:
        accel = clamp(self.gas_brake, -1.0, 1.0)
        return SteeringWheelOutput(
            clamp(self.steer, -1.0, 1.0),
            pedal_to_inverted_axis(max(0.0, accel)),
            pedal_to_inverted_axis(max(0.0, -accel)),
            0.0,
        )


def pedal_to_inverted_axis(value: float) -> float:
    return 1.0 - 2.0 * clamp(value, 0.0, 1.0)


class TerminalKeyReader:
    def __init__(self, input_stream=sys.stdin) -> None:
        self._input_stream = input_stream
        self._fd = input_stream.fileno()
        self._old_term: list | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if not self._input_stream.isatty():
            raise RuntimeError(
                "Keyboard worker must be run from an interactive terminal."
            )
        self._old_term = termios.tcgetattr(self._fd)
        new_term = self._old_term.copy()
        new_term[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(self._fd, termios.TCSAFLUSH, new_term)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        if self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._old_term)
            self._old_term = None

    def read_key(self) -> str | None:
        ready, _, _ = select.select([self._input_stream], [], [], 0)
        if not ready:
            return None
        return self._input_stream.read(1)


class IsaacKeyboardControlWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self._bind = args.bind
        self._topic = args.topic
        self._rate_hz = args.rate_hz
        self._verbose = args.verbose
        self._state = IsaacKeyboardControlState(axis_increment=args.axis_increment)
        self._retargeter = VehicleControlRetargeter(
            VehicleControlRetargeterConfig(steer_scale=args.steer_scale)
        )
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._sequence = 0
        self._running = True

    def run(self) -> None:
        self._register_signal_handlers()
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind)
        period_s = 1.0 / self._rate_hz

        print(
            f"Publishing IsaacTeleop keyboard commands on {self._bind} "
            f"topic={self._topic!r} at {self._rate_hz:.1f} Hz"
        )
        print("Controls: W/S gas-brake, A/D steer, R reset, C neutral, Q or Esc quit")
        try:
            with TerminalKeyReader() as reader:
                self._run_loop(reader, period_s)
        finally:
            self._publish_neutral()
            self._socket.close()
            self._context.term()
            print("\nIsaacTeleop keyboard control worker stopped.")

    def stop(self, _signum=None, _frame=None) -> None:
        self._running = False

    def _register_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def _run_loop(self, reader: TerminalKeyReader, period_s: float) -> None:
        while self._running and not self._state.quit_requested:
            started = time.monotonic()
            key = reader.read_key()
            if key is not None:
                self._state.apply_key(key)
            self._publish_next_command()
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    def _publish_next_command(self) -> None:
        timestamp_ns = time.time_ns()
        isaac_command = self._retargeter.retarget(
            self._state.to_isaac_sample(),
            sequence=self._sequence,
        )
        command = isaac_command_to_wire_command(
            isaac_command, timestamp_ns=timestamp_ns
        )
        self._publish(command)
        if self._verbose:
            print(
                f"seq={command.sequence} accel={command.accel:+.3f} "
                f"steer={command.steer:+.3f} throttle={command.throttle:.3f} "
                f"brake={command.brake:.3f}",
                end="\r",
                flush=True,
            )
        self._sequence += 1

    def _publish_neutral(self) -> None:
        self._publish(VehicleControlCommand.neutral(sequence=self._sequence))

    def _publish(self, command: VehicleControlCommand) -> None:
        payload = json.dumps(command.to_dict(), separators=(",", ":"))
        self._socket.send_string(f"{self._topic} {payload}")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read WASD keyboard input through IsaacTeleop retargeting and publish Kia teleop commands over ZMQ."
    )
    parser.add_argument("--bind", default=DEFAULT_BIND, help="ZMQ PUB bind address.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ZMQ topic prefix.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Publish rate.")
    parser.add_argument(
        "--axis-increment",
        type=float,
        default=0.05,
        help="Axis delta applied for each key press.",
    )
    parser.add_argument(
        "--steer-scale",
        type=float,
        default=1.0,
        help="Scale or invert steering after IsaacTeleop retargeting.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print live control values."
    )
    return parser


def main() -> None:
    IsaacKeyboardControlWorker(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
