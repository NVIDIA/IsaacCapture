# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import signal
import time

import zmq

from vehicle_teleop.vehicle_command import VehicleControlCommand, clamp


DEFAULT_CONNECT = "tcp://127.0.0.1:5555"
DEFAULT_TOPIC = "kia_control"


class KiaPandaWorker:
    def __init__(self, args: argparse.Namespace):
        self._connect = args.connect
        self._topic = args.topic
        self._command_timeout = args.command_timeout
        self._poll_ms = args.poll_ms
        self._steer_sign = args.steer_sign
        self._dry_run = args.dry_run
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._poller = zmq.Poller()
        self._panda_runner = None
        self._panda = None
        self._car_control = None
        self._last_command = VehicleControlCommand.neutral()
        self._last_received = 0.0
        self._last_printed = None
        self._neutral_sent_after_timeout = False
        self._running = True

    def run(self) -> None:
        self._register_signal_handlers()
        self._open_socket()
        self._open_panda()

        print(f"Receiving steering commands from {self._connect} topic={self._topic!r}")
        try:
            while self._running:
                command = self._next_command_to_apply()
                if command is not None:
                    if self._dry_run:
                        self._print_command(command)
                    else:
                        self._apply_command(command)
        finally:
            if self._panda is not None:
                self._apply_command(VehicleControlCommand.neutral())
            if self._panda_runner is not None:
                self._panda_runner.__exit__(None, None, None)
            self._socket.close()
            self._context.term()
            print("\nVehicle panda worker stopped.")

    def stop(self, _signum=None, _frame=None) -> None:
        self._running = False

    def _register_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def _open_socket(self) -> None:
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(self._connect)
        self._poller.register(self._socket, zmq.POLLIN)

    def _open_panda(self) -> None:
        if self._dry_run:
            return

        from opendbc.car.panda_runner import PandaRunner
        from opendbc.car.structs import CarControl

        self._car_control = CarControl(enabled=False)
        self._panda_runner = PandaRunner()
        self._panda = self._panda_runner.__enter__()

    def _poll_once(self) -> bool:
        events = dict(self._poller.poll(timeout=self._poll_ms))
        if self._socket not in events:
            return False

        message = self._socket.recv_string()
        _topic, payload = message.split(" ", 1)
        self._last_command = VehicleControlCommand.from_dict(json.loads(payload))
        self._last_received = time.monotonic()
        self._neutral_sent_after_timeout = False
        return True

    def _next_command_to_apply(self) -> VehicleControlCommand | None:
        if self._poll_once():
            return self._last_command
        if self._last_received == 0.0:
            return None
        if time.monotonic() - self._last_received <= self._command_timeout:
            return None
        if self._neutral_sent_after_timeout:
            return None

        self._neutral_sent_after_timeout = True
        return VehicleControlCommand(
            sequence=self._last_command.sequence,
            timestamp_ns=time.time_ns(),
            steer=0.0,
            accel=0.0,
            throttle=0.0,
            brake=0.0,
        )

    def _print_command(self, command: VehicleControlCommand) -> None:
        printable = (command.sequence, round(command.accel, 3), round(command.steer, 3))
        if printable == self._last_printed:
            return
        print(
            f"seq={command.sequence} accel={command.accel:+.3f} steer={command.steer:+.3f}"
        )
        self._last_printed = printable

    def _apply_command(self, command: VehicleControlCommand) -> None:
        self._car_control.actuators.accel = float(4.0 * clamp(command.accel, -1.0, 1.0))
        self._car_control.actuators.torque = float(
            self._steer_sign * clamp(command.steer, -1.0, 1.0)
        )

        self._panda.read()
        self._car_control.enabled = True
        self._car_control.latActive = True
        self._car_control.longActive = True
        self._panda.write(self._car_control)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive Kia teleop commands over ZMQ and write them to PandaRunner."
    )
    parser.add_argument(
        "--connect", default=DEFAULT_CONNECT, help="ZMQ PUB endpoint to connect to."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ZMQ topic prefix.")
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=0.25,
        help="Seconds before stale commands go neutral.",
    )
    parser.add_argument(
        "--poll-ms", type=int, default=10, help="ZMQ poll interval in milliseconds."
    )
    parser.add_argument(
        "--steer-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Invert steering torque if needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Receive and print commands without opening PandaRunner.",
    )
    return parser


def main() -> None:
    KiaPandaWorker(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
