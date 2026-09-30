# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge the existing SONIC ZMQ wire format over a LiveKit data session."""

import asyncio
import json
import logging
import time
from collections import Counter, deque

import zmq

from .data import MAX_MESSAGE, DataSession

logger = logging.getLogger("isaaccapture.remote_assistance.bridge")


def sonic_stop_message():
    """SONIC command wire format: topic + 1280-byte JSON header + three u8 fields."""
    header = {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": [
            {"name": name, "dtype": "u8", "shape": [1]}
            for name in ("start", "stop", "planner")
        ],
    }
    return (
        b"command"
        + json.dumps(header, separators=(",", ":")).encode().ljust(1280, b"\0")
        + bytes((0, 1, 0))
    )


class Outbox:
    """Coalesce state by topic, preserve discrete commands, and expire queued work."""

    def __init__(self, max_age_s):
        self.max_age_s = max_age_s
        self.states = {}
        self.pending = deque()
        self.event_count = 0

    def clear(self):
        self.states.clear()
        self.pending.clear()
        self.event_count = 0

    def put(self, route, payload, now):
        if len(payload) > MAX_MESSAGE:
            raise ValueError(f"{route['topic']}: message exceeds {MAX_MESSAGE} bytes")
        topic = route["topic"]
        item = [route, payload, now]
        if route.get("latest", True) and topic in self.states:
            self.states[topic][:] = item
            return
        if len(self.pending) >= 128 or (
            not route.get("latest", True) and self.event_count >= 32
        ):
            raise RuntimeError(
                "Command queue full; closing transport instead of replaying a backlog"
            )
        self.pending.append(item)
        if route.get("latest", True):
            self.states[topic] = item
        else:
            # The manager sends targets before mode switches. Coalescing must
            # not replace those targets with samples from after the command.
            self.states.clear()
            self.event_count += 1

    def pop(self, now):
        while self.pending:
            item = self.pending.popleft()
            route = item[0]
            if not route.get("latest", True):
                self.event_count -= 1
            if self.states.get(route["topic"]) is item:
                del self.states[route["topic"]]
            if now - item[2] < self.max_age_s:
                return item[:2]
        return None


async def run_bridge(config, role, stop):
    local = config[role]
    other = config["robot" if role == "edge" else "edge"]
    routes = config["routes"]
    outgoing = [r for r in routes if r["from"] == role]
    incoming = {r["topic"]: r for r in routes if r["from"] != role}
    if len({r["topic"] for r in routes}) != len(routes):
        raise ValueError("Route topics must be unique")
    for route in routes:
        if not route["prefix"] or not route["topic"].startswith("ra/v1/"):
            raise ValueError("Each route needs a nonempty ZMQ prefix and ra/v1/ topic")
    context = zmq.Context()
    subscriber, publisher = context.socket(zmq.SUB), context.socket(zmq.PUB)
    for sock in (subscriber, publisher):
        sock.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVHWM, 32)
    subscriber.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE)
    publisher.setsockopt(zmq.SNDHWM, 8)
    settings = {**config["livekit"], **local.get("livekit", {})}
    age_s = float(settings.get("max_age_ms", 300)) / 1000
    outbox = Outbox(age_s)
    forwarded = Counter()
    last_control = None
    suspended = False
    monitor_control = role == "robot"

    def receive(topic, payload):
        nonlocal last_control, suspended
        route = incoming[topic]
        if not payload.startswith(route["prefix"].encode()):
            raise ValueError("LiveKit payload does not match its configured ZMQ topic")
        publisher.send(payload, flags=zmq.DONTWAIT)
        forwarded["to_zmq/" + route["prefix"]] += 1
        if route["prefix"] in ("pose", "planner"):
            last_control = time.monotonic()
            if suspended:
                suspended = False
                logger.info("Fresh control targets received; forwarding resumed")

    session = DataSession(
        settings, local["identity"], other["identity"], incoming, receive
    )
    try:
        for route in outgoing:
            subscriber.setsockopt(zmq.SUBSCRIBE, route["prefix"].encode())
        subscriber.connect(local["subscribe"])
        publisher.bind(local["publish"])
        await session.start()
        logger.info(
            "%s bridge: %s -> LiveKit -> %s", role, local["subscribe"], local["publish"]
        )
        next_report = time.monotonic() + 5
        while not stop.is_set() and not session.closed.is_set():
            now = time.monotonic()
            if not session.ready:
                outbox.clear()
            for _ in range(64):
                try:
                    parts = subscriber.recv_multipart(flags=zmq.DONTWAIT)
                except zmq.Again:
                    break
                if len(parts) != 1:
                    raise ValueError("SONIC adapter expects single-frame ZMQ messages")
                if not session.ready:
                    continue
                payload = parts[0]
                route = next(
                    (r for r in outgoing if payload.startswith(r["prefix"].encode())),
                    None,
                )
                if route:
                    outbox.put(route, payload, now)
            item = outbox.pop(time.monotonic())
            if item:
                route, payload = item
                sent = await session.send(
                    route["topic"], payload, reliable=bool(route.get("reliable", False))
                )
                if sent:
                    forwarded["to_livekit/" + route["prefix"]] += 1
            if (
                monitor_control
                and not suspended
                and last_control is not None
                and time.monotonic() - last_control > age_s
            ):
                # SONIC owns pose hold/planner idle. stop=true exits its policy
                # loop and must never be synthesized for missing headset input.
                suspended = True
                forwarded["input_suspensions"] += 1
                logger.warning(
                    "Control targets absent; transport suspended. "
                    "SONIC remains responsible for local hold/idle; "
                    "fresh targets will resume forwarding."
                )
            if now >= next_report:
                logger.info(
                    "%s transport ready=%s control=%s counters=%s forwarded=%s",
                    role,
                    session.ready,
                    "suspended"
                    if suspended
                    else "active"
                    if last_control is not None
                    else "waiting",
                    dict(session.stats),
                    dict(forwarded),
                )
                next_report = now + 5
            await asyncio.sleep(0.001)
        if session.error:
            raise RuntimeError("LiveKit data session failed") from session.error
    finally:
        # Transport lifetime must not terminate the controller's balance loop.
        await session.stop()
        subscriber.close()
        publisher.close()
        context.term()
