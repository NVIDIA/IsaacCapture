# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-browser USB-local topology, transfer, and private report helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shlex
import stat
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

USB_NODE = re.compile(r"^\d+-\d+(?:\.\d+)*$")
PCI_NODE = re.compile(r"^\d{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
CURL_LINE = re.compile(
    r"^B:(\d{1,12}):(\d{1,4}(?:\.\d{1,9})?):(\d{1,4}(?:\.\d{1,9})?)$"
)
TIME_LINE = re.compile(r"^T:(\d+):(\d+)$")
SCHEMA = "isaac-capture.usb-health/v1"


def _read_serial_salt(path: Path) -> bytes:
    """Reject symlinks and incomplete salts before hashing device identifiers."""

    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        salt = stream.read(33)
    if len(salt) != 32:
        raise OSError("USB health serial salt is not a complete 32-byte value")
    return salt


def _load_or_create_serial_salt(directory: Path) -> bytes:
    """Publish a complete per-install salt before concurrent readers can see it."""
    salt_path = directory / ".serial-salt"
    try:
        return _read_serial_salt(salt_path)
    except FileNotFoundError:
        # No salt is published yet; race to publish a complete candidate below.
        pass
    temporary = directory / f".serial-salt.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(secrets.token_bytes(32))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, salt_path)
        except FileExistsError:
            # Another creator published its complete salt first; use that one.
            pass
    finally:
        temporary.unlink(missing_ok=True)
    return _read_serial_salt(salt_path)


def _read(path: Path) -> str | None:
    """Read bounded sysfs text, returning unknown for inaccessible fields."""
    try:
        return path.read_text(encoding="utf-8").strip()[:160]
    except (OSError, UnicodeError):
        return None


def _number(path: Path) -> float | None:
    """Parse an optional numeric sysfs field without assuming it exists."""
    try:
        return float(_read(path))
    except (TypeError, ValueError):
        return None


def _pci(path: Path) -> str | None:
    """Find the PCI controller that owns a resolved USB sysfs node."""
    return next(
        (p.name for p in path.resolve().parents if PCI_NODE.fullmatch(p.name)), None
    )


def _node(path: Path) -> dict:
    """Capture the non-secret sysfs attributes of one USB topology node."""
    return {
        "node": path.name,
        "speedMbps": _number(path / "speed"),
        "usbVersion": _read(path / "version"),
        "vendorId": _read(path / "idVendor"),
        "productId": _read(path / "idProduct"),
        "manufacturer": _read(path / "manufacturer"),
        "product": _read(path / "product"),
        "authorized": _read(path / "authorized"),
        "maxPower": _read(path / "bMaxPower"),
    }


def inspect_link(
    serial: str, listing: str, root: Path = Path("/sys/bus/usb/devices")
) -> dict:
    """Correlate the already-selected ADB transport with Linux's negotiated link."""
    result = {
        "probeSucceeded": False,
        "negotiatedSpeedMbps": None,
        "isBetterPortAvailable": None,
        "betterPortCandidates": [],
        "parentChain": [],
        "errors": [],
    }
    fields = next(
        (line.split() for line in listing.splitlines() if line.split()[:1] == [serial]),
        None,
    )
    if not fields or len(fields) < 2 or fields[1] != "device":
        result["errors"].append("Selected ADB device is absent or not ready")
        return result
    props = dict(part.split(":", 1) for part in fields[2:] if ":" in part)
    result["adbState"] = fields[1]
    result["model"] = props.get("model", "")[:80]
    result["product"] = props.get("product", "")[:80]
    usb_path = props.get("usb")
    device = root / usb_path if usb_path and USB_NODE.fullmatch(usb_path) else None
    if device is None or not device.exists():
        # The descriptor fallback covers ADB versions that omit usb:<path>.
        matches = [
            entry.parent
            for entry in root.glob("*/serial")
            if USB_NODE.fullmatch(entry.parent.name) and _read(entry) == serial
        ]
        if len(matches) != 1:
            result["errors"].append(
                "ADB transport could not be uniquely mapped to sysfs"
            )
            return result
        device = matches[0]
    speed = _number(device / "speed")
    if speed is None:
        result["errors"].append("Negotiated speed is unavailable in sysfs")
        return result
    chain = [
        _node(node)
        for node in (device.resolve(), *device.resolve().parents)
        if USB_NODE.fullmatch(node.name) or re.fullmatch(r"usb\d+", node.name)
    ]
    controller = _pci(device)
    root_hub = next(
        (node["node"] for node in chain if node["node"].startswith("usb")), None
    )
    candidates = [
        _node(other)
        for other in sorted(root.glob("usb*"))
        if re.fullmatch(r"usb\d+", other.name)
        and other.name != root_hub
        and controller is not None
        and _pci(other) == controller
        and (_number(other / "speed") or 0) > speed
    ]
    result.update(
        probeSucceeded=True,
        negotiatedSpeedMbps=speed,
        speedClass="usb3" if speed >= 5000 else "usb2_or_lower",
        isBetterPortAvailable=bool(candidates) if controller else None,
        betterPortCandidates=candidates,
        betterPortInference="same-controller faster root hub; physical connector unverified",
        deviceNode=device.name,
        busNumber=_read(device / "busnum"),
        deviceNumber=_read(device / "devnum"),
        controller=controller,
        parentChain=chain,
        power={
            "advertisedMaxPower": _read(device / "bMaxPower"),
            "runtimeStatus": _read(device / "power/runtime_status"),
            "control": _read(device / "power/control"),
        },
    )
    return result


async def _adb_shell(serial: str, script: str, timeout: float) -> tuple[int, str]:
    """Run pinned ADB shell, preserving timeout/cancellation after bounded cleanup."""
    child = await asyncio.create_subprocess_exec(
        "adb",
        "-s",
        serial,
        "shell",
        "-T",
        "sh -c " + shlex.quote(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await asyncio.wait_for(child.communicate(), timeout)
        return child.returncode, output[:65536].decode("ascii", "replace")
    except (asyncio.CancelledError, TimeoutError) as original_failure:
        cleanup = asyncio.create_task(_terminate_adb_shell_child(child))
        later_cancellation = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                # A later cancellation must not abandon the owned child.
                later_cancellation = exc
        # Cleanup failure cannot replace the original timeout or cancellation.
        with suppress(Exception):
            cleanup.result()
        if later_cancellation is not None:
            raise later_cancellation from None
        raise original_failure from None


async def _terminate_adb_shell_child(child) -> bool:
    """Attempt bounded drain and reap, even when a descendant holds stdout."""
    # ADB may exit before either signal; communicate still drains its pipe.
    with suppress(ProcessLookupError):
        child.terminate()
    try:
        await asyncio.wait_for(child.communicate(), 0.5)
        return True
    except TimeoutError:
        with suppress(ProcessLookupError):
            child.kill()
        try:
            await asyncio.wait_for(child.communicate(), 0.5)
            return True
        except TimeoutError:
            # A descendant may hold stdout open after ADB itself exits.
            try:
                await asyncio.wait_for(child.wait(), 0.5)
                return True
            except TimeoutError:
                return False


async def measure_transfer(serial: str, port: int, asset_size: int) -> dict:
    """Measure three seconds over the existing HTTPS listener and reverse rule."""
    started = time.monotonic()
    result = {
        "attempted": True,
        "completed": False,
        "requestedMeasurementDurationMs": 3000,
        "effectiveMeasurementDurationMs": None,
        "hostToHeadsetMbps": None,
        "hostToHeadsetResponseBodyMbps": None,
        "headsetToHostMbps": None,
        "completedBytes": 0,
        "completedRequests": 0,
        "method": None,
        "pathCoverage": None,
        "directionCoverage": "host_to_headset",
        "browserRequired": False,
        "newListenerCreated": False,
        "newReverseRuleCreated": False,
        "childCleanup": {"reaped": True},
        "measurementRevision": 2,
        "errors": [],
    }
    try:
        code, capabilities = await _adb_shell(
            serial,
            "command -v curl; command -v timeout; date +%s%N; command -v cat",
            2,
        )
        lines = capabilities.splitlines()
        curl_ready = code == 0 and any("curl" in line for line in lines)
        timer_ready = any(re.fullmatch(r"\d{16,20}", line) for line in lines)
        timeout_ready = any("timeout" in line for line in lines)
        result["prerequisites"] = {
            "deviceCurl": curl_ready,
            "deviceTimeout": timeout_ready,
            "nanosecondDate": timer_ready,
            "bundleSizeBytes": asset_size,
            "existingProxyReverseVerified": True,
        }
        if curl_ready and timeout_ready and timer_ready and asset_size >= 262144:
            result["method"] = "device_curl_static_asset_loop"
            result["pathCoverage"] = "https_static_over_adb_reverse"
            # A line is emitted only after a complete curl response; the cut-off
            # request contributes no bytes or response-body timing.
            script = (
                "start=$(date +%s%N); "
                "timeout 3 sh -c 'while :; do "
                "n=$(curl -fksS --max-time 2 -o /dev/null "
                '-w "%{size_download}:%{time_starttransfer}:%{time_total}" '
                f"https://127.0.0.1:{port}/client/bundle.js); "
                'rc=$?; if test "$rc" = 0; then printf "B:%s\\n" "$n"; fi; '
                'done\'; end=$(date +%s%N); printf \'T:%s:%s\\n\' "$start" "$end"'
            )
            code, output = await _adb_shell(serial, script, 5)
            samples = [
                (int(match.group(1)), float(match.group(2)), float(match.group(3)))
                for line in output.splitlines()
                if (match := CURL_LINE.fullmatch(line))
            ]
            malformed_samples = [
                line
                for line in output.splitlines()
                if line.startswith("B:") and not CURL_LINE.fullmatch(line)
            ]
            byte_counts = [size for size, _, _ in samples]
            timing = next(
                (
                    match
                    for line in output.splitlines()
                    if (match := TIME_LINE.fullmatch(line))
                ),
                None,
            )
            if timing is None:
                raise ValueError("Device did not return measurement timing")
            duration = (int(timing.group(2)) - int(timing.group(1))) / 1e6
            result["effectiveMeasurementDurationMs"] = round(duration, 2)
            result["timingValid"] = 2950 <= duration <= 3250
            result["completedBytes"] = sum(byte_counts)
            result["completedRequests"] = len(byte_counts)
            result["partialTailExcluded"] = True
            result["responseBodyTimingMethod"] = "curl_total_minus_starttransfer"
            if (
                code != 0
                or not result["timingValid"]
                or not byte_counts
                or malformed_samples
                or any(count != asset_size for count in byte_counts)
            ):
                raise ValueError("Device transfer window was invalid")
            result["hostToHeadsetMbps"] = round(
                sum(byte_counts) * 8000 / duration / 1e6, 2
            )
            # Each curl starts a new TLS connection; do not substitute the
            # response-body-only rate for three-second wall-clock goodput.
            phase_valid = all(0 <= first < total <= 2.5 for _, first, total in samples)
            result["responseBodyTimingValid"] = phase_valid
            if phase_valid:
                first_byte_wait_ms = sum(first for _, first, _ in samples) * 1000
                body_ms = sum(total - first for _, first, total in samples) * 1000
                result["completedFirstByteWaitDurationMs"] = round(
                    first_byte_wait_ms, 2
                )
                result["completedResponseBodyDurationMs"] = round(body_ms, 2)
                result["hostToHeadsetResponseBodyMbps"] = round(
                    sum(byte_counts) * 8000 / body_ms / 1e6, 2
                )
            else:
                result["responseBodyTimingUnavailableReason"] = (
                    "curl returned invalid per-request timing"
                )
            result["completed"] = True
        else:
            result["method"] = "adb_shell_sink"
            result["pathCoverage"] = "adb_transport_only"
            result["fallbackReason"] = (
                "Device curl/timing unavailable or static asset too small"
            )
            await _measure_fallback(serial, result)
    except asyncio.CancelledError:
        result["errors"].append("Measurement cancelled")
        raise
    except (OSError, TimeoutError, ValueError) as exc:
        result["errors"].append(str(exc)[:160])
    result["wholeStageDurationMs"] = round((time.monotonic() - started) * 1000, 2)
    result["outcome"] = "pass" if result["completed"] else "unsupported"
    return result


async def _terminate_fallback_child(child) -> bool:
    """Bound termination and reaping of the ADB sink child."""
    if child.returncode is None:
        # The child may exit between the returncode check and the signal.
        with suppress(ProcessLookupError):
            child.terminate()
    try:
        await asyncio.wait_for(child.wait(), 0.5)
        return True
    except TimeoutError:
        if child.returncode is None:
            with suppress(ProcessLookupError):
                child.kill()
        try:
            await asyncio.wait_for(child.wait(), 0.5)
            return True
        except TimeoutError:
            return False


async def _reap_fallback_child(child) -> bool:
    """Finish bounded reaping even if cancellation recurs during cleanup."""
    cleanup = asyncio.create_task(_terminate_fallback_child(child))
    cancellation = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as exc:
            cancellation = exc
    if cancellation is not None:
        raise cancellation
    return cleanup.result()


async def _measure_fallback(serial: str, result: dict) -> None:
    """Bound a host-to-device ADB-only sink when device curl is unavailable."""
    child = await asyncio.create_subprocess_exec(
        "adb",
        "-s",
        serial,
        "shell",
        "-T",
        "cat | wc -c",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    start = time.monotonic()
    written = 0
    failure: BaseException | None = None
    output = b""
    try:
        payload = b"\0" * 65536
        while time.monotonic() - start < 3:
            child.stdin.write(payload)
            await asyncio.wait_for(child.stdin.drain(), 0.5)
            written += len(payload)
        duration = (time.monotonic() - start) * 1000
        result["effectiveMeasurementDurationMs"] = round(duration, 2)
        result["timingValid"] = 2950 <= duration <= 3250
        result["hostBytesSubmitted"] = written
    # drain() can raise cancellation or a non-Exception signal; retain it until
    # the owned ADB child is reaped, then re-raise that same failure below.
    except BaseException as exc:
        failure = exc
    try:
        child.stdin.close()
    except Exception as exc:
        if failure is None:
            failure = exc
    if failure is None:
        try:
            output = await asyncio.wait_for(child.stdout.read(64), 0.5)
            await asyncio.wait_for(child.wait(), 0.5)
        # The final read/wait can also be interrupted; Exception would miss
        # CancelledError and skip reaping before the original signal propagates.
        except BaseException as exc:
            failure = exc
    if failure is not None:
        try:
            result["childCleanup"] = {"reaped": await _reap_fallback_child(child)}
        except asyncio.CancelledError as exc:
            failure = exc
        except Exception:
            result["childCleanup"] = {"reaped": False}
        # Cleanup failure must not turn shutdown cancellation into warn-mode success.
        raise failure
    result["childCleanup"] = {"reaped": True}
    if child.returncode != 0 or not output.strip().isdigit():
        raise ValueError("ADB sink did not confirm received byte count")
    received = int(output.strip())
    result["completedBytes"] = received
    result["hostToHeadsetMbps"] = round(received * 8000 / duration / 1e6, 2)
    result["completed"] = result["timingValid"]


class UsbHealthReport:
    """Own one private, atomically replaced JSON report for a clean start."""

    def __init__(
        self,
        logs_dir: Path,
        session_id: str,
        generation: int,
        serial: str | None,
        policy: str = "warn",
    ) -> None:
        """Prepare private report state and a salted identity for this session."""
        self.directory = logs_dir / "usb-health"
        if self.directory.is_symlink():
            raise OSError("USB health report directory cannot be a symlink")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = (
            self.directory
            / f"usb-health-{stamp}-{secrets.token_hex(4)}-g{generation}.json"
        )
        self._serial_salt = _load_or_create_serial_salt(self.directory)
        now = datetime.now(timezone.utc).isoformat()
        self.payload = {
            "schema": SCHEMA,
            "createdAt": now,
            "updatedAt": now,
            "context": {
                "mode": "usb-local",
                "policy": policy,
                "generation": generation,
                "sessionIdPrefix": session_id[:8],
                "requestedMeasurementDurationMs": 3000,
                "reportPath": str(Path("logs/usb-health") / self.path.name),
            },
            "device": {
                "selectedBy": "oobLifecycle",
                "serialHash": None,
                "serialPersistedRaw": False,
            },
            "topology": None,
            "transferTest": None,
            "decision": {"overallOutcome": "pending", "browserLaunchAllowed": None},
            "events": [],
            "errors": [],
            "redaction": {
                "rawSerial": False,
                "rawUrlOrQuery": False,
                "subprocessArgv": False,
                "controlOrTurnCredential": False,
            },
        }
        if serial is not None:
            self.set_device(serial)

    def set_device(self, serial: str) -> None:
        """Store an installation-salted serial hash, never the raw serial."""
        digest = hashlib.sha256(self._serial_salt + serial.encode()).hexdigest()
        self.payload["device"]["serialHash"] = f"sha256-install-salted:{digest}"
        self.update()

    def event(self, state: str) -> None:
        """Append one timestamped report state transition."""
        self.payload["events"].append(
            {"at": datetime.now(timezone.utc).isoformat(), "state": state}
        )
        self.update()

    def update(self, **sections) -> None:
        """Replace whole top-level sections, then synchronously persist and prune.

        I/O failures propagate to the caller.
        """
        self.payload.update(sections)
        self.payload["updatedAt"] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_name(
            self.path.name + "." + secrets.token_hex(4) + ".tmp"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.payload, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._prune()
        finally:
            temporary.unlink(missing_ok=True)

    def _prune(self) -> None:
        """Keep at most 20 reports, dropping retained files older than 14 days."""
        files = sorted(
            (
                p
                for p in self.directory.glob("usb-health-*.json")
                if p.is_file() and not p.is_symlink() and stat.S_ISREG(p.stat().st_mode)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[20:]:
            old.unlink()
        cutoff = time.time() - 14 * 86400
        for old in files[:20]:
            if old != self.path and old.stat().st_mtime < cutoff:
                old.unlink()
