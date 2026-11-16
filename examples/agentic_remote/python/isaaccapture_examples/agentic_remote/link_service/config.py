# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot link service settings: a frozen dataclass, checked when loaded from YAML."""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import yaml

LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(ValueError):
    """A setting is missing, unknown, of the wrong kind or out of range."""


@dataclass(frozen=True)
class LinkServiceConfig:
    """Settings of the robot link service."""

    listen_address: str = "127.0.0.1"
    listen_port: int = 47610
    allowed_peers: tuple[str, ...] = ("127.0.0.1",)
    socket_path: str | None = None  # None: the default path
    log_level: str = "info"
    max_clients: int = 16
    hello_timeout_s: float = 2.0
    send_timeout_s: float = 2.0
    queue_messages: int = 256
    queue_bytes: int = 8 * 1024 * 1024
    # The typed fallback: the operator console may ask for a takeover (a person
    # becomes the control owner) and for the hand-back that ends it.
    typed_takeover: bool = False

    def __post_init__(self) -> None:
        _text("listen.address", self.listen_address)
        _whole("listen.port", self.listen_port, 0, 65535)
        peers = self.allowed_peers
        if not isinstance(peers, tuple) or not peers:
            raise ConfigError(f"allowed_peers must be a non-empty list, got {peers!r}")
        spelled: list[str] = []
        for index, peer in enumerate(peers):
            _text(f"allowed_peers[{index}]", peer)
            try:
                address = ipaddress.ip_address(peer.strip())
            except ValueError:
                raise ConfigError(
                    f"allowed_peers[{index}] must be an IP address, got {peer!r}"
                ) from None
            if _needs_interface_name(address):
                raise ConfigError(
                    f"allowed_peers[{index}] is a link-local address: write it "
                    "with the name of this workstation's interface, as in "
                    f"fe80::1%eth0, got {peer!r}"
                )
            spelled.append(str(address))
        # One spelling per address, as the service compares a dialing peer's.
        object.__setattr__(self, "allowed_peers", tuple(spelled))
        if self.socket_path is not None:
            _text("socket_path", self.socket_path)
        if self.log_level not in LOG_LEVELS:
            raise ConfigError(
                f"log_level must be one of {', '.join(LOG_LEVELS)}, got "
                f"{self.log_level!r}"
            )
        _whole("local.max_clients", self.max_clients, 1, 1024)
        _positive("local.hello_timeout_s", self.hello_timeout_s)
        _positive("local.send_timeout_s", self.send_timeout_s)
        _whole("local.queue_messages", self.queue_messages, 8, 1 << 20)
        _whole("local.queue_bytes", self.queue_bytes, 1 << 16, 1 << 32)
        _flag("operator.typed_takeover", self.typed_takeover)

    @property
    def listen_endpoint(self) -> str:
        """The address the service listens on, as "host:port"."""
        return f"{self.listen_address}:{self.listen_port}"

    def service_options(self) -> dict[str, Any]:
        """The keyword arguments of RobotLinkService that come from this file."""
        return {
            "socket_path": self.socket_path,
            "allowed_peers": self.allowed_peers,
            "max_clients": self.max_clients,
            "hello_timeout_s": self.hello_timeout_s,
            "send_timeout_s": self.send_timeout_s,
            "queue_messages": self.queue_messages,
            "queue_bytes": self.queue_bytes,
            "typed_takeover": self.typed_takeover,
        }


def load_link_service_config(path: str | os.PathLike[str]) -> LinkServiceConfig:
    """The settings in a YAML file; ConfigError names the setting that is wrong.

    A setting the file leaves out keeps its default; one it does not know is an
    error.
    """
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read {os.fspath(path)}: {error}") from error
    return link_service_config(data)


def link_service_config(data: object) -> LinkServiceConfig:
    """The settings in a mapping as the YAML file holds it."""
    top = _section("the file", {} if data is None else data)
    listen = _section("listen", top.pop("listen", {}))
    local = _section("local", top.pop("local", {}))
    operator = _section("operator", top.pop("operator", {}))
    values: dict[str, Any] = {}
    _move(listen, "address", values, "listen_address")
    _move(listen, "port", values, "listen_port")
    for name in ("log_level", "socket_path"):
        _move(top, name, values, name)
    if not values.get("socket_path"):
        values.pop("socket_path", None)  # empty means the default path
    if "allowed_peers" in top:
        peers = top.pop("allowed_peers")
        if not isinstance(peers, (list, tuple)):
            raise ConfigError(f"allowed_peers must be a list, got {peers!r}")
        values["allowed_peers"] = tuple(peers)
    for name in (
        "max_clients",
        "hello_timeout_s",
        "send_timeout_s",
        "queue_messages",
        "queue_bytes",
    ):
        _move(local, name, values, name)
    _move(operator, "typed_takeover", values, "typed_takeover")
    sections = (
        ("", top),
        ("listen.", listen),
        ("local.", local),
        ("operator.", operator),
    )
    for where, left in sections:
        if left:
            unknown = ", ".join(f"{where}{key}" for key in sorted(map(str, left)))
            raise ConfigError(f"unknown setting(s): {unknown}")
    return LinkServiceConfig(**values)


def _needs_interface_name(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Whether an allowed peer is an IPv6 link-local address without its interface.

    The remote link reports such a peer as "fe80::1%eth0", with the name of
    the interface it dialed in on: an entry without a zone, or with the
    interface's number, never equals it. An IPv4 address has no zone.
    """
    if address.version != 6 or address.ipv4_mapped is not None:
        return False
    zone = address.scope_id
    return address.is_link_local and (zone is None or zone.isdigit())


def _section(where: str, value: object) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {value!r}")
    return dict(value)


def _move(source: dict[Any, Any], key: str, values: dict[str, Any], name: str) -> None:
    if key in source:
        values[name] = source.pop(key)


def _text(where: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string, got {value!r}")


def _flag(where: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {value!r}")


def _whole(where: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigError(f"{where} must be a whole number, got {value!r}")
    if not low <= value <= high:
        raise ConfigError(f"{where} must be from {low} to {high}, got {value!r}")


def _positive(where: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    if not (math.isfinite(value) and value > 0.0):
        raise ConfigError(f"{where} must be above 0, got {value!r}")
