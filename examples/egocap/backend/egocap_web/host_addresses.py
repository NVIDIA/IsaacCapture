# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The capture host address an operator enters in the headset.

The headset connects to the capture host over the local network, so the operator needs that host's address. When
CaptureServer runs on this machine the candidates are this machine's own IPv4 addresses, wireless first: a
headset is on the wireless network, and a wired link is often a point-to-point cable to another device. When
CaptureServer runs elsewhere its own host is the only candidate.
"""

from __future__ import annotations

import fcntl
import ipaddress
import socket
import struct
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Linux ``ioctl`` request that returns an interface's IPv4 address.
_SIOCGIFADDR = 0x8915

WIRELESS, WIRED, OTHER, CAPTURE_SERVER = "wireless", "wired", "other", "capture_server"

#: Interface name prefixes, by the kind of link they name on Linux.
_KIND_PREFIXES = ((WIRELESS, ("wl",)), (WIRED, ("eth", "en")))

#: Wireless first; see the module docstring.
_KIND_ORDER = (WIRELESS, WIRED, OTHER)


@dataclass(frozen=True)
class HostAddress:
    """One address the headset could be given, and where it comes from."""

    address: str
    #: The network interface holding the address, or None when it is not one of this machine's.
    interface: str | None
    kind: str


def interface_kind(name: str) -> str:
    """Whether an interface name is a wireless link, a wired one, or neither."""
    for kind, prefixes in _KIND_PREFIXES:
        if name.startswith(prefixes):
            return kind
    return OTHER


def _interface_address(name: str) -> str | None:
    """The IPv4 address of one interface through the Linux ioctl, or None when it has none."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            packed = fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, struct.pack("256s", name.encode()[:15]))
        except OSError:
            return None
    return socket.inet_ntoa(packed[20:24])


def _system_interfaces() -> list[tuple[str, str]]:
    """Every interface of this machine with an IPv4 address, as (name, address); empty off Linux."""
    # The request number below is Linux's; elsewhere it could mean something else, so it is never sent.
    if not sys.platform.startswith("linux"):
        return []
    found = []
    for _index, name in socket.if_nameindex():
        address = _interface_address(name)
        if address is not None:
            found.append((name, address))
    return found


def _route_address() -> str | None:
    """The address this machine would send from, found without sending anything; None when it has no route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            # Connecting a datagram socket only selects a route; no packet leaves the machine.
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
        except OSError:
            return None


def local_addresses(
    interfaces: Callable[[], Iterable[tuple[str, str]]] = _system_interfaces,
    route_address: Callable[[], str | None] = _route_address,
) -> list[HostAddress]:
    """This machine's usable IPv4 addresses, wireless first, then in the order the system lists them.

    Where interfaces cannot be listed (the ioctl is Linux only), the outbound route's address is the one
    candidate.
    """
    try:
        listed = list(interfaces())
    except OSError:
        listed = []
    usable = [
        HostAddress(address, name, interface_kind(name))
        for name, address in listed
        # Loopback and link-local addresses are unreachable from a headset on the network.
        if not (ipaddress.ip_address(address).is_loopback or ipaddress.ip_address(address).is_link_local)
    ]
    if usable:
        return sorted(usable, key=lambda candidate: _KIND_ORDER.index(candidate.kind))
    fallback = route_address()
    if fallback is None or ipaddress.ip_address(fallback).is_loopback:
        return []
    return [HostAddress(fallback, None, OTHER)]


def _is_this_machine(host: str) -> bool:
    """Whether a CaptureServer host names this machine: ``localhost`` or a loopback address."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _resolved(host: str) -> str | None:
    """A host as an IP literal: itself when it already is one, else what its name resolves to, else None."""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def headset_addresses(
    capture_server_url: str,
    local: Callable[[], list[HostAddress]] = local_addresses,
    resolve: Callable[[str], str | None] = _resolved,
) -> list[HostAddress]:
    """The addresses to offer for the headset, best first; empty when none can be determined.

    Always IP literals: a headset is given an address, not a name, and a station remembers one.
    """
    host = urlsplit(capture_server_url).hostname or ""
    if host == "" or _is_this_machine(host):
        return local()
    # A capture host on another machine is where the headset must connect, whatever this machine's addresses are.
    address = resolve(host)
    return [HostAddress(address, None, CAPTURE_SERVER)] if address is not None else []


def address_changed(saved: str | None, candidates: list[HostAddress]) -> bool:
    """Whether the address saved for a station is no longer one the host has, so the headset needs updating."""
    return saved is not None and bool(candidates) and saved not in {candidate.address for candidate in candidates}
