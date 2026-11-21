# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The capture host address offered for the headset: which addresses, in which order, and when one is stale."""

from __future__ import annotations

import socket
import struct

import pytest

from egocap_web import host_addresses
from egocap_web.host_addresses import (
    HostAddress,
    address_changed,
    headset_addresses,
    interface_kind,
    local_addresses,
)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("wlan0", "wireless"), ("wlp2s0", "wireless"), ("eth0", "wired"), ("enp3s0", "wired"), ("docker0", "other")],
)
def test_an_interface_is_classified_by_its_name(name, kind):
    assert interface_kind(name) == kind


def test_wireless_addresses_come_first_and_unreachable_ones_are_dropped():
    listed = [
        ("lo", "127.0.0.1"),
        ("enp3s0", "192.168.1.10"),
        ("usb0", "169.254.7.7"),
        ("docker0", "172.17.0.1"),
        ("wlp2s0", "10.0.0.42"),
    ]
    assert local_addresses(lambda: listed, lambda: "203.0.113.9") == [
        HostAddress("10.0.0.42", "wlp2s0", "wireless"),
        HostAddress("192.168.1.10", "enp3s0", "wired"),
        HostAddress("172.17.0.1", "docker0", "other"),
    ]


def _failing_interfaces():
    raise OSError("interfaces cannot be listed")


@pytest.mark.parametrize("interfaces", [lambda: [], lambda: [("lo", "127.0.0.1")], _failing_interfaces])
def test_the_outbound_route_is_the_fallback_when_no_interface_is_usable(interfaces):
    assert local_addresses(interfaces, lambda: "192.168.1.10") == [HostAddress("192.168.1.10", None, "other")]
    assert local_addresses(interfaces, lambda: None) == []
    assert local_addresses(interfaces, lambda: "127.0.0.1") == []


@pytest.mark.parametrize("url", ["ws://127.0.0.1:8082", "ws://localhost:8082", "ws://[::1]:8082", "not a url"])
def test_a_capture_server_on_this_machine_offers_this_machines_addresses(url):
    here = [HostAddress("10.0.0.42", "wlp2s0", "wireless")]
    assert headset_addresses(url, lambda: here) == here


def test_a_capture_server_elsewhere_is_the_only_address_offered_and_always_as_an_address():
    here = [HostAddress("10.0.0.42", "wlp2s0", "wireless")]
    assert headset_addresses("ws://192.168.1.50:8082", lambda: here) == [
        HostAddress("192.168.1.50", None, "capture_server")
    ]
    # A capture host given by name is resolved: a headset is given an address, and a station remembers one.
    names = {"capture.example": "192.168.1.60"}
    assert headset_addresses("wss://capture.example:8082", lambda: here, names.get) == [
        HostAddress("192.168.1.60", None, "capture_server")
    ]
    assert headset_addresses("wss://unknown.example:8082", lambda: here, names.get) == []


def test_a_host_name_is_resolved_and_an_unknown_one_is_not_guessed(monkeypatch):
    assert host_addresses._resolved("192.168.1.50") == "192.168.1.50"
    monkeypatch.setattr(host_addresses.socket, "gethostbyname", lambda host: "192.168.1.60")
    assert host_addresses._resolved("capture.example") == "192.168.1.60"

    def unknown(host):
        raise OSError("name or service not known")

    monkeypatch.setattr(host_addresses.socket, "gethostbyname", unknown)
    assert host_addresses._resolved("capture.example") is None


def test_a_saved_address_is_stale_only_when_the_host_no_longer_has_it():
    candidates = [HostAddress("10.0.0.42", "wlp2s0", "wireless"), HostAddress("192.168.1.10", "enp3s0", "wired")]
    assert not address_changed("192.168.1.10", candidates)  # a wired address the host still has is not stale
    assert address_changed("10.0.0.7", candidates)
    assert not address_changed(None, candidates)
    assert not address_changed("10.0.0.7", [])  # nothing to compare against is not evidence of a change


def test_interfaces_are_read_through_the_linux_ioctl_only_on_linux(monkeypatch):
    def fake_ioctl(_fd, _request, packed):
        name = packed.rstrip(b"\x00").decode()
        if name == "down0":
            raise OSError("no address")
        return struct.pack("20s4s232s", b"", socket.inet_aton("10.0.0.42"), b"")

    monkeypatch.setattr(host_addresses.socket, "if_nameindex", lambda: [(1, "wlp2s0"), (2, "down0")])
    monkeypatch.setattr(host_addresses.fcntl, "ioctl", fake_ioctl)
    monkeypatch.setattr(host_addresses.sys, "platform", "linux")
    assert host_addresses._system_interfaces() == [("wlp2s0", "10.0.0.42")]
    monkeypatch.setattr(host_addresses.sys, "platform", "darwin")
    assert host_addresses._system_interfaces() == []


class _Probe:
    """A stand-in datagram socket: ``route`` is the address it would send from, or None for no route."""

    route: str | None = None

    def __init__(self, *_args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def connect(self, _target):
        if self.route is None:
            raise OSError("network is unreachable")

    def getsockname(self):
        return (self.route, 0)


def test_the_route_address_is_found_without_sending_and_absent_without_a_route(monkeypatch):
    monkeypatch.setattr(host_addresses.socket, "socket", _Probe)
    assert host_addresses._route_address() is None
    monkeypatch.setattr(_Probe, "route", "192.168.1.10")
    assert host_addresses._route_address() == "192.168.1.10"
