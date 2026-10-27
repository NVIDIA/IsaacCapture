# AVS1 wire format

AVS1 means **Argus Video Streams, version 1**. It carries one or more logical
mono/stereo streams, with HEVC or raw NV12 samples, over one reliable byte
stream. The TCP stream and `network.dump_wire` file contain the same repeated
records and have no file-level header or footer.

All integers are unsigned and encoded in network byte order (big-endian). All
reserved fields must be zero. Sizes and offsets are bytes.

## TCP record

| Offset | Size | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | `record_length` | Bytes after this prefix |
| 4 | `record_length` | `avs1_record` | One complete bundle |

Read exactly four bytes, decode `record_length`, then read exactly that many
bytes. The maximum representable record is `2^32 - 1` bytes.

## Bundle header

Every `avs1_record` starts with this 64-byte header:

| Offset | Size | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | `magic` | ASCII `AVS1` |
| 4 | 2 | `version` | `1` |
| 6 | 2 | `flags` | Bundle flags |
| 8 | 4 | `header_bytes` | `64` |
| 12 | 2 | `stream_count` | Logical stream descriptor count |
| 14 | 2 | `sample_count` | Total camera sample descriptor count |
| 16 | 8 | `bundle_sequence` | Starts at 1 and increases each base-rate tick |
| 24 | 8 | `session_id` | Nonzero ID; changes when the pipeline is rebuilt |
| 32 | 8 | `sender_capture_ready_ns` | Sender clock after due captures are acquired |
| 40 | 8 | `sender_tcp_enqueue_ns` | Sender clock before the completed bundle is queued |
| 48 | 4 | `descriptor_bytes` | Stream plus sample descriptor bytes |
| 52 | 4 | `payload_bytes` | Combined sample payload bytes |
| 56 | 8 | `sender_capture_ready_tsc_ns` | Sender Tegra TSC read at the `sender_capture_ready_ns` instant; zero unless `SENDER_TSC_VALID` |

Bundle flags:

| Mask | Name | Meaning |
| ---: | --- | --- |
| `0x0001` | `SENDER_CLOCK_VALID` | Sender timing fields are valid |
| `0x0002` | `SENDER_CLOCK_TAI` | Valid sender fields use `CLOCK_TAI`; otherwise `CLOCK_REALTIME` |
| `0x0004` | `SENDER_TSC_VALID` | `sender_capture_ready_tsc_ns` is valid |

`SENDER_CLOCK_TAI` is invalid unless `SENDER_CLOCK_VALID` is also set.

`SENDER_TSC_VALID` is invalid unless `SENDER_CLOCK_VALID` is also set. When it
is set, `sender_capture_ready_tsc_ns` must be nonzero; when it is clear, the
field must be zero. Parsers reject either violation. Offset 56 was a reserved
zero before this flag existed, so a sender that predates it (flag clear, field
zero) is still parsed unchanged by current receivers. The reverse is not true: a
receiver that predates the flag rejects every bundle that sets it as an unknown
bundle flag, so deploy the receiver before (or together with) a sender that
emits `SENDER_TSC_VALID`.

The record layout is:

```text
64-byte bundle header
stream_count * 40-byte stream descriptors
sample_count * 72-byte sample descriptors
payload_bytes of sample data
```

Therefore:

```text
descriptor_bytes = stream_count * 40 + sample_count * 72
record_length = 64 + descriptor_bytes + payload_bytes
```

## Stream descriptor

There is one 40-byte descriptor for every configured logical stream, in YAML
`streams` order. A descriptor remains in every bundle even when that stream is
not due at the current mixed-rate tick.

| Offset | Size | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `stream_index` | Zero-based config index |
| 2 | 2 | `flags` | Stream flags |
| 4 | 1 | `codec` | `0` raw NV12, `1` HEVC |
| 5 | 1 | `pixel_format` | `1` NV12 |
| 6 | 2 | `rotation` | `0` or `180`, presentation rotation applied by the receiver |
| 8 | 4 | `width` | Coded/output width |
| 12 | 4 | `height` | Coded/output height |
| 16 | 4 | `fps_numerator` | Configured/resolved stream output rate |
| 20 | 4 | `fps_denominator` | Currently `1`; never zero |
| 24 | 8 | `stream_sequence` | Increases only when this stream is present; zero when absent |
| 32 | 2 | `first_sample` | Flat sample-descriptor index |
| 34 | 2 | `sample_count` | `1` mono, `2` stereo |
| 36 | 4 | `reserved` | Zero |

Stream flags:

| Mask | Name | Meaning |
| ---: | --- | --- |
| `0x0001` | `PRESENT` | Samples and payloads are present in this bundle |
| `0x0002` | `STEREO` | Two ordered samples: left, then right |

For a lower-rate stream on an absent tick, `PRESENT` is clear,
`stream_sequence` is zero, and both stereo samples are absent atomically. The
descriptor still reports codec, dimensions, FPS, and camera count.

## Sample descriptor

There is one 72-byte descriptor per physical camera. Samples are flattened in
stream order, then camera order. Stereo order is always left then right.

| Offset | Size | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `stream_index` | Owning logical stream |
| 2 | 1 | `camera_index` | Index inside stream: `0`, or `0/1` for stereo |
| 3 | 1 | `role` | `0` mono, `1` left, `2` right |
| 4 | 2 | `flags` | Sample validity flags |
| 6 | 2 | `reserved` | Zero |
| 8 | 4 | `device_index` | Argus device index from sender config |
| 12 | 4 | `sensor_mode_index` | Argus mode index from sender config |
| 16 | 8 | `frame_number` | Argus capture frame number |
| 24 | 8 | `sof_tsc_ns` | VI hardware start-of-frame TSC timestamp |
| 32 | 8 | `frame_timestamp_ns` | Argus frame timestamp |
| 40 | 8 | `sender_capture_ready_ns` | Sender clock when due captures were ready |
| 48 | 8 | `sender_process_submit_ns` | Sender clock before copy/encode submission |
| 56 | 8 | `sender_process_done_ns` | Sender clock when raw/encoded payload was ready |
| 64 | 4 | `payload_offset` | Offset within the payload area |
| 68 | 4 | `payload_length` | This sample's payload bytes |

Sample flags:

| Mask | Name | Meaning |
| ---: | --- | --- |
| `0x0001` | `PRESENT` | Payload is present |
| `0x0002` | `SOF_TSC_VALID` | `sof_tsc_ns` is valid |
| `0x0004` | `FRAME_TIMESTAMP_VALID` | `frame_timestamp_ns` is valid |
| `0x0008` | `SENDER_TIMING_VALID` | The three per-sample sender fields are valid |

An absent sample has all sample flags clear, zero timing/frame fields, and a
zero payload length. A present sample has a nonzero payload. Stream and sample
presence must agree; stereo eye presence is atomic.

`sof_tsc_ns` is from the Tegra-wide VI TSC domain, not wall clock. Signed
`right.sof_tsc_ns - left.sof_tsc_ns` is meaningful for a stereo stream on one
sender. Sender wall-clock values can be compared across hosts only to the extent
that their clocks are synchronized.

### Sensor SOF wall time

`sender_capture_ready_ns` is read after Argus has delivered every due capture,
so it trails the exposure by a jittery 17-25 ms. `sender_capture_ready_tsc_ns`
is the sender's Tegra TSC read at that same instant, which places each sample's
VI start-of-frame on the sender wall clock:

```text
sensor_sof_ns = sender_capture_ready_ns
              - (sender_capture_ready_tsc_ns - sof_tsc_ns)
```

Use it only when `SENDER_CLOCK_VALID`, `SENDER_TSC_VALID`, and the sample's
`SOF_TSC_VALID` are all set and the lag
`sender_capture_ready_tsc_ns - sof_tsc_ns` lies in `[0, 500 ms]`; otherwise
fall back to `sender_capture_ready_ns`. The sender reads the TSC from the ARM
generic timer scaled by `cntfrq_el0` (the domain that stamps VI SOF) and sets
`SENDER_TSC_VALID` only when that lag check passed for the bundle's due
captures. `frame_timestamp_ns` is in a different domain and must not be used as
the exposure anchor.

## Payload area

Payloads are concatenated in stream order and camera order. `payload_offset` is
relative to the first byte after all descriptors.

- HEVC (`codec = 1`): one opaque Annex-B access unit returned by NVENC, including
  start-code-prefixed NAL units.
- Raw (`codec = 0`): tight NV12 with `width * height` Y bytes followed by
  `width * height / 2` interleaved UV bytes. Its length must equal
  `width * height * 3 / 2`.

Receivers must validate every payload range against `payload_bytes` and reject
overlap, gaps, unknown flags, unsupported versions, or inconsistent presence.

## Minimal Python framing

The standalone parser is
`televiz/camera_receiver/avs_wire.py` relative to the example root. This fragment shows the outer framing
and fixed descriptor structs:

```python
import struct

HEADER = struct.Struct("!4sHHIHHQQQQIIQ")   # 64 bytes
STREAM = struct.Struct("!HHBBHIIIIQHHI")    # 40 bytes
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII") # 72 bytes


def recv_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    done = 0
    while done < size:
        count = sock.recv_into(view[done:])
        if count == 0:
            raise EOFError("truncated AVS1 record")
        done += count
    return bytes(data)


def recv_record(sock, max_record_bytes=256 * 1024 * 1024):
    (record_length,) = struct.unpack("!I", recv_exact(sock, 4))
    if not 0 < record_length <= max_record_bytes:
        raise ValueError(f"invalid AVS1 record length: {record_length}")
    record = recv_exact(sock, record_length)
    header = HEADER.unpack_from(record)
    if header[0] != b"AVS1" or header[1] != 1 or header[3] != HEADER.size:
        raise ValueError("unsupported AVS1 record")
    return record
```

A clean dump ends on a record boundary. An interrupted process can leave a
partial final record; an offline parser should ignore or report that final tail.
