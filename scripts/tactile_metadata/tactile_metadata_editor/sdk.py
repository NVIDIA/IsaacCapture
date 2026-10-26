# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK checks in a short local process using the launching Python environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

from .compiler import ValuePreparationError, canonical_values
from .domain import FrozenMap

_DIAGNOSTIC_LIMIT = 8192
_REPORT_LIMIT = 1024 * 1024
_TYPES = ("HandSkeletonRef", "LBS", "TactileMetadata")
_EXIT_CODES = {"passed": 0, "failed": 1, "unavailable": 3}
_BOOTSTRAP = """
import json, runpy, sys
request_path = sys.argv[1]
with open(request_path, encoding="utf-8") as stream:
    request = json.load(stream)
sys.path[:] = request["module_search_path"]
sys.argv[:] = ["tactile_metadata_editor.sdk_worker", request_path]
runpy.run_module("tactile_metadata_editor.sdk_worker", run_name="__main__")
"""


@dataclass(frozen=True)
class SdkResult:
    status: str
    operation: str
    report: FrozenMap
    encoded: bytes | None = None


def _report(operation, status, code, message, *, issues=None, diagnostics=""):
    return SdkResult(
        status,
        operation,
        FrozenMap({
            "version": 1,
            "operation": operation,
            "status": status,
            "sdk_version": None,
            "supported_types": [],
            "joint_names_values": {},
            "metadata_max_size": None,
            "encoded_size": None,
            "issues": issues or [{
                "severity": "error", "code": code, "message": message,
                "field": "sdk",
            }],
            "diagnostics": diagnostics,
        }),
    )


def _write_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def _wait(process, timeout):
    """Drain diagnostics while retaining a bounded tail, without a log file."""
    tail = bytearray()
    deadline = time.monotonic() + timeout
    timed_out = False
    stream = process.stdout
    try:
        os.set_blocking(stream.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ)
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    chunk = os.read(key.fd, _DIAGNOSTIC_LIMIT)
                    if chunk:
                        tail.extend(chunk)
                        del tail[:-_DIAGNOSTIC_LIMIT]
                    else:
                        selector.unregister(stream)
            # Reaped children can leave several pipe buffers; retain the final tail.
            while selector.get_map():
                try:
                    chunk = os.read(stream.fileno(), _DIAGNOSTIC_LIMIT)
                except BlockingIOError:
                    break
                if not chunk:
                    selector.unregister(stream)
                    break
                tail.extend(chunk)
                del tail[:-_DIAGNOSTIC_LIMIT]
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        stream.close()
    return process.returncode, tail.decode("utf-8", errors="replace"), timed_out


def _validate_report(report, operation, returncode):
    if not isinstance(report, dict):
        raise ValueError("SDK report must be an object")
    required = {
        "version", "operation", "status", "sdk_version", "supported_types",
        "joint_names_values", "metadata_max_size", "encoded_size", "issues",
    }
    if not required.issubset(report):
        raise ValueError("SDK report is missing required fields")
    if type(report.get("version")) is not int or report["version"] != 1:
        raise ValueError("Unsupported SDK report version")
    if report.get("operation") != operation:
        raise ValueError("SDK report operation differs from the request")
    status = report.get("status")
    if not isinstance(status, str) or status not in _EXIT_CODES:
        raise ValueError("Invalid SDK report status")
    if returncode != _EXIT_CODES[status]:
        raise ValueError("SDK report status and process exit code disagree")
    if report["sdk_version"] is not None and not isinstance(
        report["sdk_version"], str
    ):
        raise ValueError("Invalid SDK version")
    types = report["supported_types"]
    if not isinstance(types, list) or any(not isinstance(name, str) for name in types):
        raise ValueError("Invalid supported metadata types")
    joints = report["joint_names_values"]
    if (
        not isinstance(joints, dict)
        or any(not isinstance(name, str) or type(value) is not int
               for name, value in joints.items())
    ):
        raise ValueError("Invalid joint names and values")
    maximum = report["metadata_max_size"]
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("Invalid metadata size limit")
    issues = report.get("issues")
    if not isinstance(issues, list):
        raise ValueError("SDK report issues must be an array")
    for issue in issues:
        if (
            not isinstance(issue, dict)
            or issue.get("severity") not in ("error", "warning", "info")
            or not isinstance(issue.get("code"), str)
            or not isinstance(issue.get("message"), str)
        ):
            raise ValueError("Invalid SDK issue")
    if status != "passed":
        if (
            not any(issue["severity"] == "error" for issue in issues)
            or report.get("encoded_size") is not None
        ):
            raise ValueError("An unsuccessful SDK result needs issues and no binary")
        return
    if any(issue["severity"] == "error" for issue in issues):
        raise ValueError("A successful SDK report contains errors")
    if not isinstance(report.get("sdk_version"), str):
        raise ValueError("Successful SDK report is missing the SDK version")
    if not all(name in types for name in _TYPES):
        raise ValueError("SDK report is missing supported metadata types")
    if not joints:
        raise ValueError("SDK report is missing joint names and values")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("SDK report is missing the metadata size limit")
    size = report.get("encoded_size")
    if operation == "probe":
        if size is not None:
            raise ValueError("SDK probe cannot claim a constructed binary")
    elif type(size) is not int or not 0 < size <= maximum:
        raise ValueError("SDK encoded size is invalid or exceeds its limit")


def _run(operation, values, timeout):
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("SDK timeout must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="tactile-sdk-", dir="/tmp") as name:
        directory = Path(name)
        request_path = directory / "request.json"
        values_path = directory / "values.json"
        report_path = directory / "report.json"
        binary_path = directory / "metadata.bin"
        if values is not None:
            _write_json(values_path, values)
        _write_json(request_path, {
            "version": 1,
            "operation": operation,
            "values_file": str(values_path),
            "report_file": str(report_path),
            "binary_file": str(binary_path),
            "module_search_path": list(sys.path),
        })
        environment = os.environ.copy()
        environment["ISAACCAPTURE_LOGGING"] = "off"
        try:
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", _BOOTSTRAP, str(request_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=environment,
            )
            returncode, diagnostics, timed_out = _wait(process, timeout)
        except OSError as error:
            return _report(operation, "unavailable", "sdk_process", str(error))
        if timed_out:
            return _report(
                operation, "unavailable", "sdk_timeout",
                f"SDK {operation} exceeded {timeout:g} seconds",
                diagnostics=diagnostics,
            )
        try:
            if report_path.stat().st_size > _REPORT_LIMIT:
                raise ValueError("SDK report exceeds the size limit")
            with report_path.open(encoding="utf-8") as stream:
                report = json.load(
                    stream, parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
            _validate_report(report, operation, returncode)
            encoded = None
            if report["status"] == "passed" and operation == "encode":
                if binary_path.stat().st_size != report["encoded_size"]:
                    raise ValueError("SDK binary length differs from its report")
                encoded = binary_path.read_bytes()
                if len(encoded) != report["encoded_size"]:
                    raise ValueError("SDK binary changed while being read")
            elif binary_path.exists():
                raise ValueError("SDK returned an unexpected binary")
            report["diagnostics"] = diagnostics
            return SdkResult(report["status"], operation, FrozenMap(report), encoded)
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as error:
            return _report(
                operation, "failed", "sdk_protocol", str(error),
                diagnostics=diagnostics,
            )


def probe(timeout=30) -> SdkResult:
    """A passed probe means SDK availability, not official metadata validation."""
    return _run("probe", None, timeout)


def encode(values: Mapping, timeout=30) -> SdkResult:
    """Construct, read back and encode one unmodified target-dtype layout."""
    try:
        values = canonical_values(values)
    except ValuePreparationError as error:
        return _report(
            "encode", "failed", "invalid_values", str(error),
            issues=[asdict(issue) for issue in error.issues],
        )
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return _report("encode", "failed", "invalid_values", str(error))
    return _run("encode", values, timeout)
