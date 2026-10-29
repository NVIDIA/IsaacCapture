# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda_ipc_consumer"))
from consumer import CudaIpcClient  # noqa: E402


@pytest.mark.parametrize("failure", ["second_handle", "array"])
def test_connect_failure_closes_socket_and_partial_mappings(monkeypatch, failure):
    import consumer

    runtime = SimpleNamespace(
        cudaIpcMemLazyEnablePeerAccess=1,
        ipcOpenMemHandle=Mock(side_effect=[123, RuntimeError("mapping failed")]),
        ipcCloseMemHandle=Mock(),
    )
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=runtime,
            Device=lambda _: contextlib.nullcontext(),
            UnownedMemory=Mock(),
            MemoryPointer=Mock(),
        ),
        uint8=object(),
        ndarray=Mock(
            side_effect=RuntimeError("array failed") if failure == "array" else None
        ),
    )
    sock = Mock()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(consumer.socket, "socket", Mock(return_value=sock))
    slots = [
        dict(
            slot=i,
            handle=base64.b64encode(b"handle").decode(),
            nbytes=4,
            shape=[2, 2],
            strides=[2, 1],
        )
        for i in range(2)
    ]
    client = CudaIpcClient("/unused.sock")
    client._read_json_blocking = Mock(
        side_effect=[
            {"type": "hello"},
            {"type": "subscribed", "streams": [{"name": "head.left", "slots": slots}]},
        ]
    )
    with pytest.raises(RuntimeError, match="failed"):
        client.connect()
    runtime.ipcCloseMemHandle.assert_called_once_with(123)
    sock.close.assert_called_once()
    assert client.sock is None
    assert not client.slots and not client.stream_order and not client.stream_info
    client.close()
    runtime.ipcCloseMemHandle.assert_called_once_with(123)
