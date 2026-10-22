# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read EGO record wrappers using the BFBS schema stored in their MCAP channel."""

from dataclasses import dataclass
import struct

import numpy as np


class _Buffer:
    def __init__(self, data):
        self.data = data

    def check(self, offset, size):
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("Truncated or invalid FlatBuffer")

    def scalar(self, offset, fmt):
        self.check(offset, struct.calcsize("<" + fmt))
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def indirect(self, offset):
        target = offset + self.scalar(offset, "I")
        self.check(target, 1)
        return target

    def field(self, table, field_id):
        vtable = table - self.scalar(table, "i")
        size = self.scalar(vtable, "H")
        self.check(vtable, size)
        if size < 4 or size % 2:
            raise ValueError("Invalid FlatBuffer vtable")
        entry = 4 + 2 * field_id
        relative = self.scalar(vtable + entry, "H") if entry < size else 0
        return table + relative if relative else None

    def value(self, table, field_id, fmt, default=0):
        offset = self.field(table, field_id)
        return default if offset is None else self.scalar(offset, fmt)

    def vector(self, offset, stride):
        start = self.indirect(offset)
        count = self.scalar(start, "I")
        self.check(start + 4, count * stride)
        return start + 4, count

    def string(self, offset):
        start, size = self.vector(offset, 1)
        self.check(start + size, 1)
        if self.data[start + size] != 0:
            raise ValueError("Invalid FlatBuffer string terminator")
        return self.data[start : start + size].decode("utf-8")

    def text(self, table, field_id):
        offset = self.field(table, field_id)
        return self.string(offset) if offset is not None else ""

    def tables(self, table, field_id):
        offset = self.field(table, field_id)
        if offset is None:
            return []
        start, count = self.vector(offset, 4)
        return [self.indirect(start + 4 * i) for i in range(count)]


@dataclass(frozen=True)
class ByteVector:
    offset: int
    size: int


_SCALARS = {
    1: "B",
    2: "?",
    3: "b",
    4: "B",
    5: "h",
    6: "H",
    7: "i",
    8: "I",
    9: "q",
    10: "Q",
    11: "f",
    12: "d",
}


class RecordDecoder:
    """Small reflection reader; media vectors remain references into the message."""

    def __init__(self, bfbs, root_name):
        schema = _Buffer(bfbs)
        if len(bfbs) < 8 or bfbs[4:8] != b"BFBS":
            raise ValueError("EGO synchronization requires an embedded BFBS schema")
        root = schema.indirect(0)
        self.objects = []
        for obj in schema.tables(root, 0):
            fields = []
            for field in schema.tables(obj, 1):
                field_type = schema.field(field, 1)
                if field_type is None:
                    raise ValueError("BFBS field has no type")
                field_type = schema.indirect(field_type)
                fields.append(
                    {
                        "name": schema.text(field, 0),
                        "base": schema.value(field_type, 0, "b"),
                        "element": schema.value(field_type, 1, "b"),
                        "index": schema.value(field_type, 2, "i", -1),
                        "offset": schema.value(field, 3, "H"),
                        "integer": schema.value(field, 4, "q"),
                        "real": schema.value(field, 5, "d"),
                        "optional": schema.value(field, 11, "?"),
                    }
                )
            self.objects.append(
                {
                    "name": schema.text(obj, 0),
                    "struct": schema.value(obj, 2, "?"),
                    "size": schema.value(obj, 4, "i"),
                    "fields": fields,
                }
            )
        matches = [i for i, obj in enumerate(self.objects) if obj["name"] == root_name]
        if (
            len(matches) != 1
            or not root_name.startswith("core.OrbbecEgo")
            or not root_name.endswith("Record")
        ):
            raise ValueError(f"Unsupported EGO record schema: {root_name}")
        self.root = matches[0]
        self.struct_dtypes = {}
        if {f["name"] for f in self.objects[self.root]["fields"]} != {
            "data",
            "timestamp",
        }:
            raise ValueError("EGO record must contain the full data/timestamp wrapper")

    def decode(self, data):
        buf = _Buffer(data)
        result = self._object(buf, buf.indirect(0), self.root, 0)
        if not result.get("data") or not result.get("timestamp"):
            raise ValueError("EGO record is missing data or its capture timestamp")
        return result

    def _object(self, buf, start, index, depth):
        if depth > 20 or not 0 <= index < len(self.objects):
            raise ValueError("Invalid or recursive BFBS object")
        obj = self.objects[index]
        if obj["struct"]:
            buf.check(start, obj["size"])
        result = {}
        for field in obj["fields"]:
            offset = (
                start + field["offset"]
                if obj["struct"]
                else buf.field(start, (field["offset"] - 4) // 2)
            )
            base = field["base"]
            if offset is None:
                value = None
                if base in _SCALARS and not field["optional"]:
                    value = field["real"] if base in (11, 12) else field["integer"]
                elif base == 13:
                    value = ""
                elif base == 14:
                    value = []
            else:
                value = self._value(buf, offset, base, field, depth + 1)
            result[field["name"]] = value
        return result

    def _value(self, buf, offset, base, field, depth):
        if base in _SCALARS:
            return buf.scalar(offset, _SCALARS[base])
        if base == 13:
            return buf.string(offset)
        if base == 15:
            obj = self.objects[field["index"]]
            target = offset if obj["struct"] else buf.indirect(offset)
            return self._object(buf, target, field["index"], depth)
        if base == 14:
            element = field["element"]
            stride = 4
            if element in _SCALARS:
                stride = struct.calcsize(_SCALARS[element])
            elif element == 15 and self.objects[field["index"]]["struct"]:
                stride = self.objects[field["index"]]["size"]
            if stride <= 0:
                raise ValueError("Invalid BFBS vector stride")
            start, count = buf.vector(offset, stride)
            if element == 4:
                return ByteVector(start, count)
            if (
                element == 15
                and self.objects[field["index"]]["name"] == "core.OrbbecEgoImuSample"
            ):
                index = field["index"]
                if index not in self.struct_dtypes:
                    obj = self.objects[index]
                    if not obj["struct"] or any(
                        value["base"] not in _SCALARS
                        or value["offset"] + struct.calcsize(_SCALARS[value["base"]])
                        > stride
                        for value in obj["fields"]
                    ):
                        raise ValueError("Invalid IMU sample struct")
                    self.struct_dtypes[index] = np.dtype(
                        {
                            "names": [value["name"] for value in obj["fields"]],
                            "formats": [
                                "<" + _SCALARS[value["base"]] for value in obj["fields"]
                            ],
                            "offsets": [value["offset"] for value in obj["fields"]],
                            "itemsize": stride,
                        }
                    )
                # Keep millions of IMU samples as immutable views of their record
                # bytes rather than retaining one Python dictionary per sample.
                samples = np.frombuffer(
                    buf.data, self.struct_dtypes[index], count, start
                )
                samples.setflags(write=False)
                return samples
            return [
                self._value(buf, start + i * stride, element, field, depth)
                for i in range(count)
            ]
        raise ValueError(f"Unsupported BFBS field type {base}")
