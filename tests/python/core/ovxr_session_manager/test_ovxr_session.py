# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused ownership and lifecycle tests for external OVXR presentation."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from isaaccapture.ovxr_session_manager import camera_panels
from isaaccapture.ovxr_session_manager.camera_panels import (
    CameraPanelConfig,
    CameraPanelConsumer,
)
import isaaccapture.ovxr_session_manager.ovxr_session as ovxr_session
from isaaccapture.ovxr_session_manager import frame_pump, runtime_owner


class _FakeOvxrError(RuntimeError):
    pass


class _WorkerAbort(BaseException):
    pass


class _FakeContext:
    operations: ClassVar[list[tuple[str, int]]] = []
    create_error: BaseException | None = None
    bind_error: BaseException | None = None
    destroy_error: BaseException | None = None
    request_exit_error: BaseException | None = None
    request_exit_callback = None
    owner_lane_threads: ClassVar[set[int]] = set()
    ordinal_ledger = None
    bind_ordinals: ClassVar[list[int]] = []
    destroy_ordinals: ClassVar[list[int]] = []
    seal_bind_results: ClassVar[set[int]] = {0, 1}
    last_desc = None
    selected_products = None

    def __init__(self, desc):
        assert threading.get_native_id() in self.__class__.owner_lane_threads
        self.desc = desc
        self.__class__.last_desc = desc
        self.last_error = ""
        self.bind_results = [0]
        if self.__class__.create_error is not None:
            raise self.__class__.create_error
        self.__class__.operations.append(("create", threading.get_native_id()))

    def get_openxr_handles(self):
        return SimpleNamespace(
            xr_instance=11,
            xr_session=22,
            xr_space=33,
            xr_get_instance_proc_addr=44,
        )

    def set_frame_render_products(self, products):
        assert threading.get_native_id() in self.__class__.owner_lane_threads
        self.__class__.selected_products = tuple(products)

    def get_vulkan_handles(self):
        return SimpleNamespace()

    def submit_openxr_blank_frame(self):
        return 0, SimpleNamespace()

    def begin_frame(self):
        self.__class__.operations.append(("begin", threading.get_native_id()))
        return 0, SimpleNamespace(step_result=0)

    def bind_frame(self, _plan, _time, *, stage_ordinal):
        self.__class__.operations.append(("bind", threading.get_native_id()))
        self.__class__.bind_ordinals.append(stage_ordinal)
        if self.__class__.bind_error is not None:
            raise self.__class__.bind_error
        if len(self.bind_results) > 1:
            result = self.bind_results.pop(0)
        else:
            result = self.bind_results[0]
        if result in self.__class__.seal_bind_results:
            self.__class__.ordinal_ledger.seal_external(stage_ordinal)
        return result, result in self.__class__.seal_bind_results

    def submit_frame(self, _plan, *, additional_layers=()):
        self.__class__.operations.append(("submit", threading.get_native_id()))
        return 0, SimpleNamespace(submitted=True)

    def cancel_frame(self, _plan):
        self.__class__.operations.append(("cancel", threading.get_native_id()))

    def request_exit(self):
        self.__class__.operations.append(("request-exit", threading.get_native_id()))
        if self.__class__.request_exit_error is not None:
            raise self.__class__.request_exit_error
        if self.__class__.request_exit_callback is not None:
            self.__class__.request_exit_callback()

    def destroy(self, *, stage_ordinal):
        assert threading.get_native_id() in self.__class__.owner_lane_threads
        self.__class__.destroy_ordinals.append(stage_ordinal)
        if self.__class__.destroy_error is not None:
            raise self.__class__.destroy_error
        self.__class__.ordinal_ledger.seal_external(stage_ordinal)
        self.__class__.operations.append(("destroy", threading.get_native_id()))


class _FakeContextDesc:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeOrdinalLedger:
    def __init__(self):
        self.floor = 9
        self.active = None
        self.issued = []
        self.published = []
        self.external_resolutions = []
        self.abandoned = []
        self.resolve_error = None

    def reserve(self):
        assert self.active is None
        ordinal = self.floor + 1
        self.active = ordinal
        self.issued.append(ordinal)
        return ordinal

    def publish(self, ordinal):
        assert self.active == ordinal
        self.published.append(ordinal)
        self.floor = ordinal
        self.active = None

    def seal_external(self, ordinal):
        assert self.active == ordinal
        self.published.append(ordinal)
        self.floor = ordinal

    def resolve_external(self, ordinal, *, expected_published):
        assert self.active == ordinal
        if self.resolve_error is not None:
            raise self.resolve_error
        published = ordinal in self.published
        assert expected_published is None or published == expected_published
        self.external_resolutions.append((ordinal, expected_published, published))
        self.active = None
        return published

    def abandon(self, ordinal):
        assert self.active == ordinal
        self.abandoned.append(ordinal)
        self.active = None


def _fake_ovxr():
    return SimpleNamespace(
        Context=_FakeContext,
        OvxrError=_FakeOvxrError,
        ContextDesc=_FakeContextDesc,
        Result=SimpleNamespace(SUCCESS=0, TIMEOUT=1, NOT_READY=2, ERROR=3),
        StageOrdinalMode=SimpleNamespace(PRIVATE=0, APPLICATION=1),
        DisplayMode=SimpleNamespace(STEREO=3),
        SessionMode=SimpleNamespace(VR=4, AR=5),
        StageUpAxis=SimpleNamespace(Z=4, Y=5),
        SceneInfoPolicy=SimpleNamespace(AUTO=5),
        RasterPanelSpace=SimpleNamespace(VIEWER_START=0, STAGE_WORLD=1, HEAD_LOCKED=2),
        result_name=str,
    )


class _FakeHost:
    def __init__(self):
        self.renderer = object()
        self.stage = object()
        self.ordinal_ledger = _FakeOrdinalLedger()
        self.guard = threading.Lock()
        self.offer_thread = None
        self.attach_thread = None
        self.publish_threads = []
        self.validated = []
        self.events = []
        self.publication_closed = False
        self.owner_lane_entries = []
        self.frame_consumers = ()

    def offer_render_owner_transfer(self):
        self.offer_thread = threading.get_native_id()
        return object()

    def attach_on_render_thread(self, _capability):
        self.attach_thread = threading.get_native_id()
        self.events.append("attach")
        _FakeContext.ordinal_ledger = self.ordinal_ledger
        return ovxr_session.OvxrSessionBinding(
            self.renderer,
            self.stage,
            self.ordinal_ledger,
            frame_consumers=self.frame_consumers,
        )

    @contextmanager
    def owner_lane(self):
        thread_id = threading.get_native_id()
        self.owner_lane_entries.append(thread_id)
        _FakeContext.owner_lane_threads.add(thread_id)
        try:
            yield
        finally:
            _FakeContext.owner_lane_threads.remove(thread_id)

    @contextmanager
    def publication_guard(self):
        with self.guard:
            yield

    def publish_complete_scene(self, publication_id):
        assert not self.publication_closed
        self.publish_threads.append(threading.get_native_id())
        self.events.append(("publish", publication_id))
        ordinal = self.ordinal_ledger.reserve()
        self.ordinal_ledger.publish(ordinal)
        return ordinal

    def validate_first_published_scene(self, ordinal):
        self.validated.append(ordinal)

    def get_target_frame_T_world(self, prim_path):
        return ("target", prim_path)

    def close_publication(self):
        self.publication_closed = True
        self.events.append("publication-close")

    def close_on_render_thread(self):
        self.events.append(("host-close", threading.get_native_id()))

    def close_environment_on_render_thread(self):
        self.events.append(("environment-close", threading.get_native_id()))

    def abort(self):
        self.events.append("abort")


def _session_with_consumers(host, config, *, frame_consumers):
    host.frame_consumers = frame_consumers
    return ovxr_session.OvxrSession(host, config)


@pytest.fixture(autouse=True)
def _runtime_stubs(monkeypatch):
    _FakeContext.operations = []
    _FakeContext.create_error = None
    _FakeContext.bind_error = None
    _FakeContext.destroy_error = None
    _FakeContext.request_exit_error = None
    _FakeContext.request_exit_callback = None
    _FakeContext.owner_lane_threads = set()
    _FakeContext.ordinal_ledger = None
    _FakeContext.bind_ordinals = []
    _FakeContext.destroy_ordinals = []
    _FakeContext.seal_bind_results = {0, 1}
    _FakeContext.last_desc = None
    _FakeContext.selected_products = None
    monkeypatch.setattr(frame_pump, "_import_ovxr", _fake_ovxr)
    monkeypatch.setattr(
        camera_panels,
        "OpenXRCameraPanels",
        lambda panels: SimpleNamespace(
            supports_gpu=False,
            finish_frame=lambda: None,
            attach=lambda _xr, _vk: None,
            layer=lambda _index: None,
            publish=lambda _index, _image, **kw: True,
            close=lambda: None,
        ),
    )
    oxr = ModuleType("isaaccapture.oxr")
    oxr.OpenXRSessionHandles = lambda *values: values
    monkeypatch.setitem(__import__("sys").modules, "isaaccapture.oxr", oxr)


def _config(**kwargs):
    return ovxr_session.OvxrSessionConfig(
        camera_paths=("/left", "/right"), auto_launch_cloudxr=False, **kwargs
    )


@pytest.mark.parametrize(
    "extra,required,expected",
    [
        (None, (), []),
        (
            ("XR_EXT_hand_tracking", "XR_NV_opaque_data_channel"),
            ("XR_KHR_convert_timespec_time", "XR_NV_opaque_data_channel"),
            [
                "XR_EXT_hand_tracking",
                "XR_NV_opaque_data_channel",
                "XR_KHR_convert_timespec_time",
            ],
        ),
        (
            (),
            ("XR_KHR_win32_convert_performance_counter_time",),
            ["XR_KHR_win32_convert_performance_counter_time"],
        ),
    ],
)
def test_session_enables_only_device_requirements_and_application_extras(
    extra, required, expected
):
    config = _config() if extra is None else _config(extra_openxr_extensions=extra)
    owner = ovxr_session.OvxrSession(
        _FakeHost(), config, required_openxr_extensions=required
    )
    with owner:
        owner.start()
        assert owner._pump._context.desc.extra_openxr_extensions == expected


def _wait_for_operation(name: str) -> None:
    deadline = time.monotonic() + 1.0
    while name not in [operation for operation, _thread in _FakeContext.operations]:
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {name}")
        time.sleep(0.001)


class _ControlledLauncher:
    def __init__(self, *, block_stop=False, stop_error=None):
        self.block_stop = block_stop
        self.stop_error = stop_error
        self.stop_started = threading.Event()
        self.allow_stop = threading.Event()
        self.events = []

    def interrupt_runtime(self):
        self.events.append(("interrupt", threading.get_native_id()))

    def stop(self):
        self.events.append(("stop-start", threading.get_native_id()))
        self.stop_started.set()
        if self.block_stop:
            self.allow_stop.wait(timeout=1.0)
        if self.stop_error is not None:
            raise self.stop_error
        self.events.append(("stop-end", threading.get_native_id()))


def _call_in_thread(callback):
    errors = []

    def call():
        try:
            callback()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    return thread, errors


@pytest.mark.parametrize("wait_phase", ["bootstrap", "xr"])
def test_interrupted_start_cancels_worker_before_retirement(monkeypatch, wait_phase):
    entered = threading.Event()
    release = threading.Event()
    failure = _WorkerAbort("startup interrupted")

    def block_startup():
        entered.set()
        assert release.wait(timeout=2.0)

    class Host(_FakeHost):
        def attach_on_render_thread(self, capability):
            if wait_phase == "bootstrap":
                block_startup()
            return super().attach_on_render_thread(capability)

    class InterruptedCondition(threading.Condition):
        def wait_for(self, predicate, timeout=None):
            if (timeout is None) == (wait_phase == "xr"):
                assert entered.wait(timeout=1.0)
                raise failure
            return super().wait_for(predicate, timeout)

    owner = ovxr_session.OvxrSession(Host(), _config())
    owner._worker_condition = InterruptedCondition()
    if wait_phase == "xr":
        monkeypatch.setattr(owner._pump, "bootstrap", block_startup)
    try:
        with pytest.raises(_WorkerAbort) as caught:
            owner.start()
        assert caught.value is failure
        assert owner.shutdown_requested
    finally:
        release.set()
        owner.close()

    assert not owner._worker.is_alive()
    assert owner.retirement_complete


def test_bootstrap_timeout_cancels_slow_attach_without_creating_context():
    """An expired attach retains its host until the same worker can retire it."""
    attach_entered = threading.Event()
    allow_attach = threading.Event()

    class SlowHost(_FakeHost):
        def attach_on_render_thread(self, capability):
            attach_entered.set()
            assert allow_attach.wait(timeout=2.0)
            return super().attach_on_render_thread(capability)

    host = SlowHost()
    owner = ovxr_session.OvxrSession(host, _config())
    start_thread, start_errors = _call_in_thread(lambda: owner.start(timeout_s=0.02))
    close_thread = None
    close_errors = []
    try:
        assert attach_entered.wait(timeout=1.0)
        start_thread.join(timeout=1.0)
        assert not start_thread.is_alive()
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], TimeoutError)
        close_thread, close_errors = _call_in_thread(owner.close)
        assert owner._borrowers_quiesced.wait(timeout=1.0)
        assert owner.owns_environment
        assert not owner.retirement_complete
        assert not any(isinstance(event, tuple) for event in host.events)
    finally:
        allow_attach.set()
        start_thread.join(timeout=1.0)
        if close_thread is not None:
            close_thread.join(timeout=1.0)
        owner.close()

    assert close_errors == []
    assert not owner._worker.is_alive()
    assert owner.retirement_complete
    assert "create" not in [operation for operation, _thread in _FakeContext.operations]
    assert [event for event in host.events if isinstance(event, tuple)] == [
        ("host-close", owner.render_worker_native_thread_id),
        ("environment-close", owner.render_worker_native_thread_id),
    ]


@pytest.mark.parametrize(
    "timeout_s", [0, -1, float("nan"), float("inf"), threading.TIMEOUT_MAX * 2]
)
def test_invalid_bootstrap_budget_is_rejected_before_owner_transfer(timeout_s):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    with pytest.raises(ValueError, match="bootstrap timeout must be finite"):
        owner.start(timeout_s=timeout_s)
    assert host.offer_thread is None
    assert owner._worker is None
    assert not _FakeContext.operations
    owner.close()


def test_scene_publication_is_admitted_once_per_render_demand(monkeypatch):
    host = _FakeHost()
    frame_entered = threading.Event()
    release_frame = threading.Event()

    class BlockingSubmitContext(_FakeContext):
        def submit_frame(self, plan, **kwargs):
            frame_entered.set()
            assert release_frame.wait(timeout=1.0)
            return super().submit_frame(plan, **kwargs)

    fake_ovxr = _fake_ovxr()
    fake_ovxr.Context = BlockingSubmitContext
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: fake_ovxr)
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()

    try:
        first_ordinal = owner.request_render("step-1")
        assert frame_entered.wait(timeout=1.0)

        second_ordinal = owner.request_render("step-2")
        assert second_ordinal > first_ordinal
        assert owner._worker_publication_requested is False
        assert owner.request_render("step-3") is None

        assert owner.request_render("reset-1", force=True) > second_ordinal
        assert [
            event
            for event in host.events
            if isinstance(event, tuple) and event[0] == "publish"
        ] == [
            ("publish", "step-1"),
            ("publish", "step-2"),
            ("publish", "reset-1"),
        ]
    finally:
        release_frame.set()
        owner.close()


@pytest.mark.parametrize("force", [False, True])
def test_shutdown_rejects_late_publication_without_error(force):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    with owner:
        owner.request_shutdown()
        assert owner.request_render("late-step", force=force) is None
        assert host.publish_threads == []
        assert owner._worker_pending_ordinal is None
    assert owner.retirement_complete


def test_shutdown_during_publication_does_not_enqueue_a_frame(monkeypatch):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    original_publish = host.publish_complete_scene

    def publish_then_shutdown(publication_id):
        ordinal = original_publish(publication_id)
        owner.request_shutdown()
        return ordinal

    monkeypatch.setattr(host, "publish_complete_scene", publish_then_shutdown)
    owner.start()
    with owner:
        assert owner.request_render("last-step") is None
        assert host.publish_threads == [threading.get_native_id()]
        assert owner._worker_pending_ordinal is None
        assert not host.publication_closed
    assert owner.retirement_complete
    assert "submit" not in [name for name, _thread in _FakeContext.operations]


@pytest.mark.parametrize("during_publication", [False, True])
def test_shutdown_does_not_suppress_a_worker_failure(monkeypatch, during_publication):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    failure = RuntimeError("first worker failure")
    if during_publication:
        original_publish = host.publish_complete_scene

        def publish_then_fail(publication_id):
            ordinal = original_publish(publication_id)
            owner._record_worker_failure(failure)
            return ordinal

        monkeypatch.setattr(host, "publish_complete_scene", publish_then_fail)
    else:
        owner._record_worker_failure(failure)
    try:
        with pytest.raises(RuntimeError) as caught:
            owner.request_render("late-step")
        assert caught.value is failure
        assert len(host.publish_threads) == int(during_publication)
        assert owner._worker_pending_ordinal is None
    finally:
        with pytest.raises(RuntimeError) as caught:
            owner.close()
        assert caught.value is failure


def test_publication_before_start_and_after_close_remains_an_error():
    owner = ovxr_session.OvxrSession(_FakeHost(), _config())
    with pytest.raises(RuntimeError, match="not accepting requests"):
        owner.request_render("before-start")
    owner.start()
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.request_render("after-close")


@pytest.mark.parametrize("timeout_s", [0.2, 300.0])
def test_non_isaaclab_host_lends_exact_pair_to_one_owner_thread(timeout_s):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    caller_thread = threading.get_native_id()

    owner.start(timeout_s=timeout_s)
    assert host.offer_thread == caller_thread
    assert host.attach_thread == owner.render_worker_native_thread_id
    assert host.attach_thread != caller_thread
    context = owner._pump._context
    assert context.desc.ovrtx_renderer is host.renderer
    assert context.desc.ovstage is host.stage
    assert context.desc.ordinal_mode == 1
    assert context.desc.host_wait_render_output is False
    assert owner.external_oxr_handles() == (11, 22, 33, 44)

    scene_ordinal = owner.request_render("step-1")
    _wait_for_operation("submit")
    owner.close()

    assert host.publish_threads == [caller_thread]
    render_operations = [
        thread_id
        for name, thread_id in _FakeContext.operations
        if name != "request-exit"
    ]
    assert set(render_operations) == {host.attach_thread}
    request_threads = [
        thread_id
        for name, thread_id in _FakeContext.operations
        if name == "request-exit"
    ]
    assert caller_thread in request_threads
    assert host.attach_thread in request_threads
    assert host.owner_lane_entries == [host.attach_thread, host.attach_thread]
    assert host.validated == [scene_ordinal]
    assert _FakeContext.bind_ordinals
    assert _FakeContext.bind_ordinals[0] > scene_ordinal
    assert _FakeContext.destroy_ordinals
    assert host.events[-2:] == [
        ("host-close", host.attach_thread),
        ("environment-close", host.attach_thread),
    ]


def test_host_wait_render_output_is_forwarded_to_ovxr():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config(host_wait_render_output=True))

    owner.start()
    with owner:
        assert owner._pump._context.desc.host_wait_render_output is True


def test_bind_timeout_reconciles_a_sealed_ordinal_then_reserves_a_fresh_one():
    host = _FakeHost()
    _FakeContext.seal_bind_results = {0, 1}
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner._pump._context.bind_results = [1, 0]

    scene_ordinal = owner.request_render("step-1")
    _wait_for_operation("cancel")
    deadline = time.monotonic() + 1.0
    while [name for name, _thread in _FakeContext.operations].count("submit") < 1:
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for retry submission")
        time.sleep(0.001)
    owner.close()

    names = [name for name, _thread in _FakeContext.operations]
    assert names.count("bind") >= 2
    assert names.index("cancel") < names.index("submit")
    assert host.validated == [scene_ordinal]
    assert len(_FakeContext.bind_ordinals) >= 2
    first, second = _FakeContext.bind_ordinals[:2]
    assert second > first
    assert (first, True, True) in host.ordinal_ledger.external_resolutions


def test_frame_scene_callback_seals_before_the_next_bind_ordinal():
    class Host(_FakeHost):
        def attach_on_render_thread(self, capability):
            binding = super().attach_on_render_thread(capability)

            def prepare():
                ordinal = self.ordinal_ledger.reserve()
                self.ordinal_ledger.publish(ordinal)
                self.events.append(("frame-scene", ordinal))

            return replace(binding, prepare_frame_scene=prepare)

    host = Host()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner.request_render("physics")
    _wait_for_operation("submit")
    owner.close()
    prepared = [
        e[1] for e in host.events if isinstance(e, tuple) and e[0] == "frame-scene"
    ]
    assert prepared
    assert _FakeContext.bind_ordinals == [ordinal + 1 for ordinal in prepared]


def test_shutdown_wait_wakes_on_the_existing_lifecycle_event():
    owner = ovxr_session.OvxrSession(_FakeHost(), _config())
    woke = threading.Event()
    thread = threading.Thread(target=lambda: (owner.wait_for_shutdown(30), woke.set()))
    thread.start()
    try:
        owner.request_shutdown()
        assert woke.wait(1)
        assert owner.wait_for_shutdown(30)
    finally:
        thread.join(timeout=1)
    assert not thread.is_alive()


def test_unsupported_provider_rejects_before_host_transfer(monkeypatch):
    runtime = _fake_ovxr()
    runtime.Context = type(
        "OldContext", (_FakeContext,), {"set_frame_render_products": None}
    )
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: runtime)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    with pytest.raises(
        RuntimeError, match="required Context methods: set_frame_render_products"
    ):
        owner.start()
    assert host.offer_thread is None


def test_missing_core_phase_method_still_rejects_before_host_transfer(monkeypatch):
    runtime = _fake_ovxr()
    runtime.Context = type("BrokenContext", (_FakeContext,), {"cancel_frame": None})
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: runtime)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    with pytest.raises(RuntimeError, match="required Context methods: cancel_frame"):
        owner.start()
    assert host.offer_thread is None


def test_bind_timeout_reconciles_an_unsealed_ordinal_then_reuses_it():
    host = _FakeHost()
    _FakeContext.seal_bind_results = {0}
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner._pump._context.bind_results = [1, 0]

    owner.request_render("step-1")
    _wait_for_operation("cancel")
    deadline = time.monotonic() + 1.0
    while [name for name, _thread in _FakeContext.operations].count("submit") < 1:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    owner.close()

    assert len(_FakeContext.bind_ordinals) >= 2
    assert _FakeContext.bind_ordinals[1] == _FakeContext.bind_ordinals[0]
    assert (
        _FakeContext.bind_ordinals[0],
        False,
        False,
    ) in host.ordinal_ledger.external_resolutions


def test_bind_not_ready_without_writes_retries_the_unsealed_ledger_ordinal():
    host = _FakeHost()
    _FakeContext.seal_bind_results = {0}
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner._pump._context.bind_results = [2, 0]

    owner.request_render("step-1")
    _wait_for_operation("cancel")
    deadline = time.monotonic() + 1.0
    while [name for name, _thread in _FakeContext.operations].count("submit") < 1:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    owner.close()

    assert len(_FakeContext.bind_ordinals) >= 2
    assert _FakeContext.bind_ordinals[1] == _FakeContext.bind_ordinals[0]
    assert (
        _FakeContext.bind_ordinals[0],
        False,
        False,
    ) in host.ordinal_ledger.external_resolutions


@pytest.mark.parametrize("sealed", [False, True])
def test_fatal_bind_resolves_publication_before_reporting_failure(sealed):
    host = _FakeHost()
    _FakeContext.seal_bind_results = {0, 3} if sealed else {0}
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner._pump._context.bind_results = [3]
    owner.request_render("step-1")
    _wait_for_operation("cancel")
    with pytest.raises(RuntimeError, match="OVXR bind_frame failed"):
        owner.close()
    first = _FakeContext.bind_ordinals[0]
    assert (first, sealed, sealed) in host.ordinal_ledger.external_resolutions
    assert "submit" not in [name for name, _thread in _FakeContext.operations]
    assert host.ordinal_ledger.active is None


def test_create_failure_does_not_reserve_or_publish_an_ordinal():
    host = _FakeHost()
    _FakeContext.create_error = RuntimeError("create failed")
    owner = ovxr_session.OvxrSession(host, _config())

    with pytest.raises(RuntimeError, match="create failed"):
        owner.start()

    assert host.ordinal_ledger.issued == []
    assert host.ordinal_ledger.external_resolutions == []
    assert host.ordinal_ledger.abandoned == []


@pytest.mark.parametrize("before_attach", [False, True])
def test_shutdown_unblocks_start_while_context_is_still_waiting(
    monkeypatch, before_attach
):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    entered = threading.Event()
    resume = threading.Event()
    original = _FakeContext.__init__

    def create(context, desc):
        entered.set()
        assert resume.wait(timeout=2.0)
        original(context, desc)

    if before_attach:
        attach = host.attach_on_render_thread

        def blocked_attach(capability):
            entered.set()
            assert resume.wait(timeout=2.0)
            return attach(capability)

        monkeypatch.setattr(host, "attach_on_render_thread", blocked_attach)
    else:
        monkeypatch.setattr(_FakeContext, "__init__", create)
    results = []
    thread, errors = _call_in_thread(lambda: results.append(owner.start()))
    try:
        assert entered.wait(timeout=1.0)
        owner.request_shutdown()
        thread.join(timeout=0.2)
        assert not thread.is_alive()
        assert errors == []
        assert results == [False]
    finally:
        resume.set()
        thread.join(timeout=1.0)
        owner.close()


def test_close_wakes_context_before_releasing_ready_worker():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    caller = threading.get_native_id()
    wakes = []

    def wake():
        if threading.get_native_id() == caller:
            wakes.append(True)
            assert not owner._borrowers_quiesced.is_set()

    _FakeContext.request_exit_callback = wake
    owner.request_shutdown()
    assert owner._phase is not ovxr_session._Phase.CLOSED
    assert "destroy" not in [name for name, _thread in _FakeContext.operations]
    owner.close()
    assert wakes == [True]
    assert owner.retirement_complete


def test_successful_shutdown_waits_for_caller_quiescence_before_retirement():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    waiting = threading.Event()

    class ObservedQuiescence(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    owner._borrowers_quiesced = ObservedQuiescence()
    owner.start()
    context = owner._pump._context
    with owner:
        owner.request_shutdown()
        assert waiting.wait(timeout=1.0)
        assert owner._worker.is_alive()
        assert owner._pump._context is context
        assert owner._phase is not ovxr_session._Phase.CLOSED
        assert not owner.retirement_complete
        assert "destroy" not in [name for name, _thread in _FakeContext.operations]
        assert not any(
            isinstance(event, tuple) and event[0] in {"host-close", "environment-close"}
            for event in host.events
        )

    assert not owner._worker.is_alive()
    assert owner.retirement_complete
    assert _FakeContext.operations.count(("destroy", host.attach_thread)) == 1


def test_close_times_out_without_blocking_on_startup_teardown_lock(monkeypatch):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    entered = threading.Event()
    resume = threading.Event()
    destroy = _FakeContext.destroy
    failure = RuntimeError("startup failed")

    def blocked_destroy(context, *, stage_ordinal):
        entered.set()
        assert resume.wait(timeout=2.0)
        destroy(context, stage_ordinal=stage_ordinal)

    def fail_start():
        raise failure

    monkeypatch.setattr(_FakeContext, "destroy", blocked_destroy)
    monkeypatch.setattr(owner._pump, "bootstrap", fail_start)
    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    try:
        with pytest.raises(RuntimeError) as caught:
            owner.start()
        assert caught.value is failure
        assert entered.wait(timeout=1.0)
        closing, errors = _call_in_thread(owner.close)
        closing.join(timeout=0.5)
        assert not closing.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
        assert owner._pump._context is not None
        assert not owner.retirement_complete
    finally:
        resume.set()
        owner._worker.join(timeout=1.0)
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is failure


@pytest.mark.parametrize("abort_error_type", [None, RuntimeError, KeyboardInterrupt])
def test_worker_thread_start_failure_rolls_back_host_and_preserves_first_error(
    monkeypatch, abort_error_type, caplog
):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    start_error = RuntimeError("native thread creation failed")
    original_cause = OSError("thread resources unavailable")
    start_error.add_note("Original thread diagnostic")
    original_notes = list(start_error.__notes__)
    abort_error = abort_error_type("host abort failed") if abort_error_type else None

    def fail_start(_thread):
        raise start_error from original_cause

    def abort():
        host.events.append("abort")
        if abort_error is not None:
            raise abort_error

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    monkeypatch.setattr(host, "abort", abort)
    interrupted = abort_error_type is KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError) as error:
        owner.start()
    assert error.value is (abort_error if interrupted else start_error)
    assert start_error.__cause__ is original_cause
    assert start_error.__notes__ == original_notes
    diagnostics = [
        record
        for record in caplog.records
        if record.getMessage() == "Cleanup after failed render-worker start also failed"
    ]
    assert [record.exc_info[1] for record in diagnostics] == (
        [abort_error] if abort_error is not None and not interrupted else []
    )
    assert not owner.owns_environment
    assert host.events == ["abort"]
    assert not _FakeContext.operations
    owner.close()
    assert host.events == ["abort"]


def test_close_before_start_aborts_host_without_creating_context():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())

    owner.close()

    assert host.events == ["abort"]
    assert _FakeContext.operations == []


def test_binding_rejects_an_incomplete_borrowed_tuple():
    with pytest.raises(ValueError, match="renderer, Stage, and ordinal ledger"):
        ovxr_session.OvxrSessionBinding(object(), object(), None)


def test_camera_panel_registers_one_borrowed_product_without_a_second_frame_source():
    host = _FakeHost()
    path = "/Camera/Head"
    config = _config()
    panel = CameraPanelConfig(
        path,
        max_image_bytes=720 * 450 * 4,
        panel_width_m=0.5,
        position=(0.1, -0.2, -0.8),
        space="head_locked",
    )
    host.frame_consumers = (CameraPanelConsumer((panel,)),)
    owner = ovxr_session.OvxrSession(host, config)

    owner.start()
    desc = _FakeContext.last_desc
    owner.close()

    assert _FakeContext.selected_products == (path,)
    assert not hasattr(desc, "enable_raster_layer")
    assert not hasattr(desc, "auxiliary_render_product")
    assert [name for name, _thread in _FakeContext.operations].count("submit") == 0


def test_auxiliary_panel_space_remains_application_policy():
    host = _FakeHost()
    path = "/Camera/Head"
    config = _config()
    panel = CameraPanelConfig(
        path,
        max_image_bytes=720 * 450 * 4,
        space="head_locked",
    )
    host.frame_consumers = (CameraPanelConsumer((panel,)),)
    owner = ovxr_session.OvxrSession(host, config)

    owner.start()
    owner.close()

    assert panel.space == "head_locked"
    assert _FakeContext.selected_products == (path,)
    assert not hasattr(_FakeContext.last_desc, "auxiliary_panel_space")


@pytest.mark.parametrize("space", ["stage", "viewer_start", "world"])
def test_camera_panel_rejects_unsupported_panel_space(space):
    with pytest.raises(ValueError, match="space"):
        CameraPanelConfig("/Camera/Head", max_image_bytes=720 * 450 * 4, space=space)


def test_camera_panel_has_bounded_public_defaults():
    config = CameraPanelConfig(
        "/Camera/Head",
    )

    assert config.max_image_bytes == 64 * 1024 * 1024


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_image_bytes": 0}, "max_image_bytes"),
        ({"panel_width_m": 0.0}, "panel_width_m"),
        ({"orientation_xyzw": (0.0, 0.0, 0.0, 0.0)}, "orientation"),
    ],
)
def test_camera_panel_rejects_invalid_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CameraPanelConfig("/Camera/Head", **kwargs)


def test_camera_panel_requires_an_absolute_host_product():
    with pytest.raises(ValueError, match="absolute"):
        CameraPanelConfig("relative")


def test_cloudxr_and_ovxr_are_created_and_destroyed_on_the_owner_thread(monkeypatch):
    host = _FakeHost()
    launcher_events = []

    class FakeLauncher:
        def __init__(self):
            launcher_events.append(("create", threading.get_native_id()))

        def interrupt_runtime(self):
            launcher_events.append(("interrupt", threading.get_native_id()))

        def stop(self):
            launcher_events.append(("stop", threading.get_native_id()))

    def launcher_factory(_path, *, accept_eula=False):
        return FakeLauncher()

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )

    owner.start()
    owner.close()

    assert launcher_events == [
        ("create", owner.render_worker_native_thread_id),
        ("stop", owner.render_worker_native_thread_id),
    ]


def test_injected_consumers_follow_frame_and_retirement_order(monkeypatch):
    events = []
    threads = []
    outputs = SimpleNamespace(contents=SimpleNamespace(output_count=0, outputs=[]))
    caller_thread = threading.get_native_id()
    host = _FakeHost()

    class Context(_FakeContext):
        def begin_frame(self):
            result, plan = super().begin_frame()
            plan.step_result = 1
            return result, plan

    class Consumer(ovxr_session.OvxrFrameConsumer):
        def __init__(self, name):
            self.name = name

        def record(self, event):
            events.append((event, self.name))
            threads.append(threading.get_native_id())

        def on_session_started(self, _binding, _config, request_shutdown):
            assert any(name == "create" for name, _ in _FakeContext.operations)
            self.request_shutdown = request_shutdown
            self.record("start")

        def on_outputs(self, frame):
            assert frame.latest_color("/missing") is None
            self.record("outputs")

        def finish_frame(self):
            self.record("finish")

        def on_frame_submitted(self):
            assert any(name == "submit" for name, _ in _FakeContext.operations)
            self.record("submitted")
            self.request_shutdown()

        def close(self):
            assert not any(name == "destroy" for name, _ in _FakeContext.operations)
            self.record("close")

    api = _fake_ovxr()
    api.Context = Context
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: api)
    monkeypatch.setattr(frame_pump, "native_renderer_handle", lambda _renderer: 7)
    monkeypatch.setattr(frame_pump, "completed_outputs", lambda *_args: outputs)
    owner = _session_with_consumers(
        host, _config(), frame_consumers=(Consumer("first"), Consumer("second"))
    )
    with owner:
        owner.start()
        owner.request_render("step-1")
        _wait_for_operation("submit")
    assert events == [
        ("start", "first"),
        ("start", "second"),
        ("outputs", "first"),
        ("outputs", "second"),
        ("finish", "first"),
        ("finish", "second"),
        ("submitted", "first"),
        ("submitted", "second"),
        ("close", "second"),
        ("close", "first"),
    ]
    assert caller_thread != owner.render_worker_native_thread_id
    assert set(threads) == {owner.render_worker_native_thread_id}
    assert not any(name == "cancel" for name, _ in _FakeContext.operations)


def test_consumer_start_failure_retires_borrowed_owners():
    host = _FakeHost()
    failure = RuntimeError("consumer start failed")

    class Consumer(ovxr_session.OvxrFrameConsumer):
        def on_session_started(self, *_args):
            raise failure

        def on_outputs(self, frame):
            pass

        def close(self):
            host.events.append("consumer-close")

    owner = _session_with_consumers(host, _config(), frame_consumers=(Consumer(),))
    with pytest.raises(RuntimeError) as caught:
        owner.start()
    assert caught.value is failure
    with pytest.raises(RuntimeError):
        owner.close()
    assert "consumer-close" in host.events
    assert "destroy" in [name for name, _ in _FakeContext.operations]


def test_quad_detach_failure_reaches_close_and_retains_borrowed_owners(monkeypatch):
    host = _FakeHost()
    panel = CameraPanelConsumer((CameraPanelConfig("/Render/Head"),))
    owner = _session_with_consumers(host, _config(), frame_consumers=(panel,))
    owner.start()
    context = owner._pump._context
    quad = panel._native
    launcher = _ControlledLauncher()
    owner._pump._runtime._launcher = launcher
    failure = RuntimeError("native quad detach failed")

    def fail_detach():
        raise failure

    quad.close = fail_detach
    with pytest.raises(RuntimeError, match="native quad detach failed") as caught:
        owner.close()

    assert caught.value is failure
    assert not owner._worker.is_alive()
    assert owner._shutdown_requested.is_set()
    assert owner._phase is ovxr_session._Phase.CLOSED
    assert not owner.retirement_complete
    with pytest.raises(RuntimeError) as repeated:
        owner.close()
    assert repeated.value is failure
    assert owner._pump._context is context
    assert panel._native is quad
    assert owner._pump._runtime._launcher is launcher
    assert launcher.events == []
    assert "destroy" not in [name for name, _thread in _FakeContext.operations]
    assert not any(
        event[0] in {"host-close", "environment-close"}
        for event in host.events
        if isinstance(event, tuple)
    )


def test_shutdown_timeout_preserves_runtime_until_frame_worker_retires(monkeypatch):
    host = _FakeHost()
    frame_wait_entered = threading.Event()
    release_frame = threading.Event()
    launcher = _ControlledLauncher()

    class BlockingContext(_FakeContext):
        def begin_frame(self):
            frame_wait_entered.set()
            assert release_frame.wait(timeout=2.0)
            return super().begin_frame()

    fake_ovxr = _fake_ovxr()
    fake_ovxr.Context = BlockingContext
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: fake_ovxr)

    def launcher_factory(*args, **kwargs):
        return launcher

    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()
    assert owner.request_render("step-1") == 10
    assert frame_wait_entered.wait(timeout=1.0)
    context = owner._pump._context
    failure = RuntimeError("request exit failed")
    _FakeContext.request_exit_error = failure
    try:
        with pytest.raises(
            TimeoutError, match="OVXR session worker did not stop"
        ) as error:
            owner.close()
        assert error.value.__cause__ is failure
        assert owner._pump._context is context
        assert owner._worker.is_alive()
        assert not owner.retirement_complete
        assert launcher.events == []
        assert "destroy" not in [name for name, _thread in _FakeContext.operations]
    finally:
        release_frame.set()
        owner._worker.join(timeout=1.0)
    assert not owner._worker.is_alive()
    with pytest.raises(RuntimeError) as error:
        owner.close()
    assert error.value is failure
    assert [name for name, _thread in launcher.events] == ["stop-start", "stop-end"]


def test_cloudxr_stop_can_finish_without_runtime_interruption(monkeypatch, caplog):
    host = _FakeHost()
    launcher = _ControlledLauncher(block_stop=True)

    def launcher_factory(_path, *, accept_eula=False):
        return launcher

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()

    close_thread, close_errors = _call_in_thread(owner.close)
    try:
        assert launcher.stop_started.wait(timeout=1.0)
        assert owner._pump._runtime._launcher is launcher
        time.sleep(0.03)
        assert close_thread.is_alive()
        assert "interrupt" not in [name for name, _thread in launcher.events]
        assert "interrupting the owned OpenXR runtime" not in caplog.text
    finally:
        launcher.allow_stop.set()
        close_thread.join(timeout=1.0)

    assert not close_thread.is_alive()
    assert close_errors == []
    assert owner._pump._runtime._launcher is None
    assert launcher.events == [
        ("stop-start", owner.render_worker_native_thread_id),
        ("stop-end", owner.render_worker_native_thread_id),
    ]


def test_destroy_failure_retains_borrowed_providers_and_forbids_retry(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher()

    def launcher_factory(_path, *, accept_eula=False):
        return launcher

    failure = RuntimeError("destroy failed")
    _FakeContext.destroy_error = failure
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()
    context, binding = owner._pump._context, owner._pump._binding

    with pytest.raises(RuntimeError, match="destroy failed") as error:
        owner.close()

    assert error.value is failure
    assert owner._pump._context is context
    assert owner._pump._binding is binding
    assert owner._pump._runtime._launcher is launcher
    assert not owner.retirement_complete
    assert launcher.events == []
    assert not any(
        isinstance(event, tuple) and event[0] in ("host-close", "environment-close")
        for event in host.events
    )
    destroy_attempts = list(_FakeContext.destroy_ordinals)
    with pytest.raises(RuntimeError, match="destroy failed"):
        owner.close()
    assert _FakeContext.destroy_ordinals == destroy_attempts


@pytest.mark.parametrize("failure", ["native-result", "ffi", "ordinal", "success"])
def test_actual_public_destroy_wrapper_distinguishes_consumed_context(
    monkeypatch, failure
):
    ovxr = pytest.importorskip("ovxr")
    host = _FakeHost()
    calls = []
    invalidations = []
    native = ovxr.Context.__new__(ovxr.Context)
    native._handle = 123
    native._uses_explicit_stage_ordinals = True
    native._frame_lifetime = SimpleNamespace(
        invalidate_context=lambda: invalidations.append(True)
    )

    def destroy(_handle, ordinal):
        calls.append(ordinal)
        if failure == "ffi":
            raise RuntimeError("FFI before consumption")
        host.ordinal_ledger.seal_external(ordinal)
        return ovxr.Result.ERROR if failure == "native-result" else ovxr.Result.SUCCESS

    native._bindings = SimpleNamespace(
        destroy=destroy, get_last_error=lambda: "release failed"
    )

    class WrappedContext(_FakeContext):
        def destroy(self, *, stage_ordinal):
            native.destroy(stage_ordinal=0 if failure == "ordinal" else stage_ordinal)

    runtime = _fake_ovxr()
    runtime.Context = WrappedContext
    runtime.OvxrError = ovxr.OvxrError
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: runtime)
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    if failure == "success":
        owner.close()
    else:
        with pytest.raises(RuntimeError):
            owner.close()
    consumed = failure in ("native-result", "success")
    assert (owner._pump._context is None) == consumed
    assert bool(invalidations) == consumed
    assert (native._handle is None) == consumed
    assert (
        any(isinstance(e, tuple) and e[0] == "environment-close" for e in host.events)
        == consumed
    )
    first_calls = calls.copy()
    if failure != "success":
        with pytest.raises(RuntimeError):
            owner.close()
    assert calls == first_calls
    # Avoid destructor warnings for deliberately quarantined fixture handles.
    native._handle = None


@pytest.mark.parametrize("accepted", [False, True])
def test_explicit_eula_consent_reaches_generic_cloudxr_launcher(monkeypatch, accepted):
    import isaaccapture.cloudxr

    captured = []
    monkeypatch.setattr(
        isaaccapture.cloudxr,
        "CloudXRLauncher",
        lambda **kwargs: captured.append(kwargs),
    )
    runtime_owner._create_cloudxr_launcher("profile.env", accept_eula=accepted)
    assert captured[0]["accept_eula"] is accepted


def test_failed_exit_request_does_not_prevent_proven_context_retirement():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    _FakeContext.request_exit_error = RuntimeError("request exit failed")
    with pytest.raises(RuntimeError, match="request exit failed"):
        owner.close()
    assert owner._pump._context is None
    assert [event[0] for event in host.events if isinstance(event, tuple)][-2:] == [
        "host-close",
        "environment-close",
    ]


def test_cloudxr_stop_failure_is_reported_without_runtime_interrupt(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher(stop_error=RuntimeError("cloudxr stop failed"))

    def launcher_factory(_path, *, accept_eula=False):
        return launcher

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()

    with pytest.raises(RuntimeError, match="cloudxr stop failed"):
        owner.close()

    assert owner._pump._runtime._launcher is launcher
    assert launcher.events == [("stop-start", owner.render_worker_native_thread_id)]

    launcher.stop_error = None
    with pytest.raises(RuntimeError, match="cloudxr stop failed"):
        owner.close()
    assert owner._pump._runtime._launcher is None
    assert [name for name, _thread in launcher.events] == [
        "stop-start",
        "stop-start",
        "stop-end",
    ]


def test_partial_startup_failure_retires_context_before_cloudxr_stop(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher(block_stop=True)

    def launcher_factory(_path, *, accept_eula=False):
        return launcher

    _FakeContext.create_error = RuntimeError("create failed")
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )

    start_thread, start_errors = _call_in_thread(owner.start)
    assert launcher.stop_started.wait(timeout=1.0)
    close_thread, close_errors = _call_in_thread(owner.close)
    try:
        time.sleep(0.03)
        assert "interrupt" not in [name for name, _thread in launcher.events]
    finally:
        launcher.allow_stop.set()
        start_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)

    assert not start_thread.is_alive()
    assert not close_thread.is_alive()
    assert [str(error) for error in start_errors] == ["create failed"]
    assert [str(error) for error in close_errors] == ["create failed"]
    assert launcher.events == [
        ("stop-start", owner.render_worker_native_thread_id),
        ("stop-end", owner.render_worker_native_thread_id),
    ]


@pytest.mark.parametrize("failure_type", [RuntimeError, _WorkerAbort])
def test_render_failure_waits_for_borrowers_and_preserves_first_error(
    monkeypatch, failure_type
):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    failure = failure_type("bind failed")
    _FakeContext.bind_error = failure
    failed = threading.Event()
    original = owner._record_worker_failure

    def record(error):
        original(error)
        failed.set()

    monkeypatch.setattr(owner, "_record_worker_failure", record)

    owner.request_render("step-1")
    try:
        assert failed.wait(timeout=1.0)
        assert owner._phase is not ovxr_session._Phase.CLOSED
        assert "destroy" not in [name for name, _thread in _FakeContext.operations]
        assert not any(
            isinstance(event, tuple) and event[0] in {"host-close", "environment-close"}
            for event in host.events
        )
        for read in (
            owner.external_oxr_handles,
            lambda: owner.request_render("step-2"),
        ):
            with pytest.raises(failure_type) as caught:
                read()
            assert caught.value is failure
    finally:
        with pytest.raises(failure_type) as caught:
            owner.close()
        assert caught.value is failure
    with pytest.raises(failure_type) as caught:
        owner.close()
    assert caught.value is failure
    assert host.publication_closed
    assert host.events[-2:] == [
        ("host-close", host.attach_thread),
        ("environment-close", host.attach_thread),
    ]


def test_bind_failure_remains_primary_when_ordinal_reconciliation_also_fails():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    _FakeContext.bind_error = RuntimeError("bind failed")
    host.ordinal_ledger.resolve_error = RuntimeError("reconciliation failed")

    owner.request_render("step-1")
    deadline = time.monotonic() + 1.0
    while not owner.shutdown_requested:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    with pytest.raises(RuntimeError, match="bind failed") as error:
        owner.close()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "reconciliation failed"


def test_destroy_failure_remains_primary_when_ordinal_reconciliation_also_fails():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    _FakeContext.destroy_error = RuntimeError("destroy failed")
    host.ordinal_ledger.resolve_error = RuntimeError("reconciliation failed")

    with pytest.raises(RuntimeError, match="destroy failed") as error:
        owner.close()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "reconciliation failed"


@pytest.mark.parametrize(
    "bind_result", [0, 1, 2], ids=["submit", "timeout", "not-ready"]
)
def test_paused_frames_use_runtime_pacing_without_an_added_idle_wait(
    monkeypatch, bind_result
):
    fourth_frame = threading.Event()
    release_frame = threading.Event()
    idle_waits = []

    class PacedContext(_FakeContext):
        frames = 0

        def begin_frame(self):
            self.frames += 1
            # The runtime owns the next-frame wait, including while physics is
            # paused. Block in that boundary to check shutdown as well as pace.
            if self.frames == 4:
                fourth_frame.set()
                assert release_frame.wait(1.0)
            return super().begin_frame()

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.get_native_id() == owner.render_worker_native_thread_id:
                idle_waits.append((owner._pump._last_ordinal, timeout))
            return super().wait(timeout)

    fake = _fake_ovxr()
    fake.Context = PacedContext
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: fake)
    _FakeContext.request_exit_callback = release_frame.set
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._worker_condition = ObservedCondition()
    with owner:
        owner.start()
        owner._pump._context.bind_results = [bind_result]
        owner.request_render("paused-scene")
        assert fourth_frame.wait(1.0)

    assert not [wait for wait in idle_waits if wait[0] is not None]
    assert [
        event
        for event in host.events
        if isinstance(event, tuple) and event[0] == "publish"
    ] == [("publish", "paused-scene")]
    assert owner._pump._context is None


@pytest.mark.parametrize(
    "begin_result", [None, 1, 2], ids=["no-scene", "timeout", "not-ready"]
)
def test_idle_or_unready_frames_keep_a_wakeable_backoff(monkeypatch, begin_result):
    backed_off = threading.Event()
    waits = []

    class UnreadyContext(_FakeContext):
        def begin_frame(self):
            assert begin_result is not None
            self.__class__.operations.append(("begin", threading.get_native_id()))
            return begin_result, None

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.get_native_id() == owner.render_worker_native_thread_id:
                waits.append(timeout)
                if len(waits) == 3:
                    backed_off.set()
            return super().wait(timeout)

    fake = _fake_ovxr()
    fake.Context = UnreadyContext
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: fake)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._worker_condition = ObservedCondition()
    with owner:
        owner.start()
        if begin_result is not None:
            owner.request_render("unready-scene")
        assert backed_off.wait(1.0)

    assert len(waits) >= 3
    assert all(timeout == 0.01 for timeout in waits)
    assert "submit" not in [name for name, _thread in _FakeContext.operations]
    assert not owner._worker.is_alive()


@pytest.mark.parametrize(
    "publish_scene", [False, True], ids=["idle-owner", "rendering-owner"]
)
def test_late_peer_exposes_input_handles_after_bootstrap(monkeypatch, publish_scene):
    peer_ready = threading.Event()
    missing_observed = threading.Event()
    handles_observed = threading.Event()
    get_threads = []

    class LatePeerContext(_FakeContext):
        def get_openxr_handles(self):
            get_threads.append(threading.get_native_id())
            if not peer_ready.is_set():
                missing_observed.set()
                return None
            handles_observed.set()
            return super().get_openxr_handles()

    fake = _fake_ovxr()
    fake.Context = LatePeerContext
    monkeypatch.setattr(frame_pump, "_import_ovxr", lambda: fake)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config(), max_blank_frames=1)
    with owner:
        owner.start()
        assert missing_observed.wait(1.0)
        assert owner.external_oxr_handles() is None
        if publish_scene:
            owner.request_render("paused-scene")
            _wait_for_operation("submit")
        # Scene rendering may proceed before DeviceIO handles exist, exactly
        # as for an initial failed peer followed by a later headset connection.
        peer_ready.set()
        with owner._worker_condition:
            ready = owner._worker_condition.wait_for(
                lambda: owner._worker_handles is not None, timeout=1.0
            )
        assert ready, "Late XR session never reached the Teleop input borrower"
        assert handles_observed.is_set()
        assert owner.external_oxr_handles() == (11, 22, 33, 44)
        assert set(get_threads) == {host.attach_thread}


@pytest.mark.parametrize("mode,expected", [("ar", 5), ("vr", 4)])
def test_session_mode_is_forwarded_to_ovxr(mode, expected):
    owner = ovxr_session.OvxrSession(_FakeHost(), _config(session_mode=mode))
    with owner:
        owner.start()
        assert owner._pump._context.desc.session_mode == expected


def test_config_rejects_unknown_session_mode():
    with pytest.raises(ValueError, match="session_mode must be 'ar' or 'vr'"):
        _config(session_mode="automatic")


def test_three_products_keep_selection_order_and_one_native_owner(monkeypatch):
    events = []
    paths = ("/Render/Head", "/Render/Left", "/Render/Right")
    panels = tuple(
        CameraPanelConfig(path, label=label)
        for path, label in zip(paths, ("Head", "Left", "Right"))
    )

    def create(configs):
        assert configs == panels
        return SimpleNamespace(
            supports_gpu=False,
            finish_frame=lambda: None,
            attach=lambda *_a: events.append("attach"),
            close=lambda: events.append("close"),
        )

    monkeypatch.setattr(camera_panels, "OpenXRCameraPanels", create)
    consumer = CameraPanelConsumer(panels)
    owner = _session_with_consumers(_FakeHost(), _config(), frame_consumers=(consumer,))
    owner.start()
    assert _FakeContext.selected_products == paths
    assert tuple(box.product_path for box in consumer._mailboxes) == paths
    owner.close()
    assert events[-1] == "close"


def test_camera_products_reject_duplicate_identity():
    with pytest.raises(ValueError, match="unique"):
        CameraPanelConsumer((CameraPanelConfig("/A"), CameraPanelConfig("/A")))


@pytest.mark.parametrize("release_fails", [False, True])
def test_bound_output_reads_finish_before_submit_or_cancel(monkeypatch, release_fails):
    host = _FakeHost()
    import ctypes

    host.renderer = SimpleNamespace(_handle=ctypes.c_void_p(123))
    owner = ovxr_session.OvxrSession(host, _config())
    owner._pump._binding = host.attach_on_render_thread(object())
    plan = SimpleNamespace(step_result=77)
    events = []

    def bind(_plan, _time, *, stage_ordinal):
        host.ordinal_ledger.seal_external(stage_ordinal)
        events.append("bind")
        return 0, True

    def release():
        events.append("unmap")
        if release_fails:
            raise RuntimeError("source read completion unproved")

    def submit(_plan, *, additional_layers):
        assert events == ["bind", "observe", "copy", "unmap"]
        assert additional_layers == [123]
        events.append("submit")
        return 0, SimpleNamespace(submitted=True)

    owner._pump._context = SimpleNamespace(
        begin_frame=lambda: (0, plan),
        bind_frame=bind,
        submit_frame=submit,
        cancel_frame=lambda _plan: events.append("cancel"),
    )

    class Consumer(ovxr_session.OvxrFrameConsumer):
        def on_outputs(self, frame):
            events.extend(["observe", "copy"])

        def composition_layers(self):
            return (123,)

        def finish_frame(self):
            release()

        def close(self):
            release()

    owner._pump._frame_consumers = (Consumer(),)

    def fetch(renderer, result):
        assert renderer == 123 and result == 77
        return object()

    monkeypatch.setattr(frame_pump, "completed_outputs", fetch)
    if release_fails:
        with pytest.raises(RuntimeError, match="completion unproved"):
            owner._pump.render(9)
        assert "submit" not in events and "cancel" not in events
        assert owner._pump._pending_plan is plan
    else:
        owner._pump.render(9)
        assert events[-1] == "submit" and owner._pump._pending_plan is None


def test_no_render_plan_does_not_read_outputs_or_append_layers(monkeypatch):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._pump._binding = host.attach_on_render_thread(object())
    plan = SimpleNamespace(step_result=0)
    submitted = []

    def bind(_plan, _time, *, stage_ordinal):
        return 0, False

    def unexpected(*args):
        pytest.fail("no-render frame attempted an output read or quad operation")

    owner._pump._context = SimpleNamespace(
        begin_frame=lambda: (0, plan),
        bind_frame=bind,
        submit_frame=lambda _plan, **kw: (
            submitted.append(kw) or 0,
            SimpleNamespace(submitted=False),
        ),
        cancel_frame=unexpected,
    )

    class Consumer(ovxr_session.OvxrFrameConsumer):
        on_outputs = unexpected
        composition_layers = unexpected

        def close(self):
            pass

    owner._pump._frame_consumers = (Consumer(),)
    monkeypatch.setattr(frame_pump, "completed_outputs", unexpected)
    owner._pump.render(9)
    assert submitted == [{"additional_layers": []}]


@pytest.mark.parametrize("cleanup_boundary", ["outputs", "cancel"])
def test_frame_cleanup_preserves_the_original_bind_error(cleanup_boundary):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._pump._binding = host.attach_on_render_thread(object())

    failure = RuntimeError("original bind failure")
    plan = SimpleNamespace(step_result=0)

    def bind(*_args, **_kwargs):
        raise failure

    def cleanup():
        raise RuntimeError("cleanup failed")

    owner._pump._context = SimpleNamespace(
        begin_frame=lambda: (0, plan),
        bind_frame=bind,
        cancel_frame=lambda _plan: cleanup() if cleanup_boundary == "cancel" else None,
    )
    if cleanup_boundary == "outputs":
        owner._pump._finish_frame_outputs = cleanup
    with pytest.raises(RuntimeError, match="original bind failure") as error:
        owner._pump.render(9)
    assert error.value is failure
    assert "cleanup failed" in " ".join(error.value.__notes__)
    assert owner._pump._pending_plan is plan


def test_consumer_retirement_failure_preserves_earlier_publication_error():
    host = _FakeHost()
    failure = RuntimeError("publication close failed")

    def close_publication():
        raise failure

    def close_map():
        raise RuntimeError("map retirement failed")

    host.close_publication = close_publication
    owner = ovxr_session.OvxrSession(host, _config())
    owner._pump._frame_consumers = (SimpleNamespace(close=close_map),)
    with pytest.raises(RuntimeError, match="publication close failed") as error:
        owner._pump.close()
    assert error.value is failure
    assert "map retirement failed" in " ".join(error.value.__notes__)
    assert not host.events  # Borrowed objects must not retire past a live map.


@pytest.mark.parametrize("hide_fails", [False, True])
@pytest.mark.parametrize("failure_site", ["fetch", "publish"])
def test_failed_output_hides_cached_quad_until_successful_refresh(
    monkeypatch, hide_fails, failure_site
):
    import ctypes

    host = _FakeHost()
    host.renderer = SimpleNamespace(_handle=ctypes.c_void_p(123))
    owner = ovxr_session.OvxrSession(host, _config())
    owner._pump._binding = host.attach_on_render_thread(object())
    events = []

    def bind(_plan, _time, *, stage_ordinal):
        host.ordinal_ledger.seal_external(stage_ordinal)
        return 0, True

    failing = [True]

    def fetch(*_args):
        if failing[0] and failure_site == "fetch":
            raise RuntimeError("fetch failed")
        return object()

    owner._pump._context = SimpleNamespace(
        begin_frame=lambda: (0, SimpleNamespace(step_result=77)),
        bind_frame=bind,
        submit_frame=lambda *_a, **kw: (
            events.append(kw["additional_layers"]) or 0,
            SimpleNamespace(submitted=True),
        ),
    )
    visible = [True]

    def hide():
        if hide_fails:
            raise RuntimeError("optional hide failed")
        visible[0] = False

    def publish(*_args, **_kwargs):
        if failing[0] and failure_site == "publish":
            raise RuntimeError("publish failed")
        visible[0] = True

    consumer = CameraPanelConsumer((CameraPanelConfig("/Camera/Head"),))
    consumer._enabled = True
    consumer._native = SimpleNamespace(
        finish_frame=lambda: None,
        hide=lambda _index: hide(),
        publish=publish,
        layer=lambda _index: 123 if visible[0] else 0,
    )
    consumer._mailboxes = [
        SimpleNamespace(
            on_outputs=lambda *_a, **_kw: None,
            close=lambda: None,
            snapshot=object,
            error=lambda: None,
        )
    ]
    owner._pump._frame_consumers = (consumer,)
    monkeypatch.setattr(frame_pump, "completed_outputs", fetch)
    owner._pump.render(9)
    assert events == [[]]
    failing[0] = False
    owner._pump.render(9)
    assert events == [[], [123]]


@pytest.mark.parametrize(
    "scale", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
)
def test_world_scale_rejects_nonfinite_or_nonpositive_values(scale):
    with pytest.raises(ValueError):
        _config(meters_per_unit=scale)


def test_slow_context_retirement_keeps_owned_runtime_alive(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher()
    entered = threading.Event()
    resume = threading.Event()
    destroy = _FakeContext.destroy

    def slow_destroy(context, *, stage_ordinal):
        entered.set()
        assert resume.wait(timeout=3.0)
        destroy(context, stage_ordinal=stage_ordinal)

    monkeypatch.setattr(_FakeContext, "destroy", slow_destroy)

    def launcher_factory(*args, **kwargs):
        return launcher

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()
    closing, errors = _call_in_thread(owner.close)
    try:
        assert entered.wait(timeout=1.0)
        # Slow native retirement must keep its runtime alive until completion.
        time.sleep(1.1)
        assert closing.is_alive()
        assert launcher.events == []
    finally:
        resume.set()
        closing.join(timeout=2.0)
    assert not closing.is_alive()
    assert errors == []
    assert owner.retirement_complete
    assert [name for name, _thread in launcher.events] == ["stop-start", "stop-end"]


@pytest.mark.parametrize("native_error", [False, True])
def test_retired_context_is_unpublished_before_lifecycle_unlock(
    monkeypatch, native_error
):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    context = owner._pump._context
    lock = threading.Lock()
    observed = []

    class ObservedLock:
        def acquire(self, blocking=True):
            return lock.acquire(blocking=blocking)

        def release(self):
            lock.release()

        def __enter__(self):
            lock.acquire()

        def __exit__(self, *_args):
            lock.release()
            observed.append(owner._pump._context)
            # The real close wake can enter immediately after unlock.
            owner._pump.request_exit()

    owner._pump._context_lifecycle_lock = ObservedLock()
    if native_error:
        _FakeContext.destroy_error = _FakeOvxrError("native result error")
        with pytest.raises(_FakeOvxrError):
            owner.close()
    else:
        owner.close()
    assert observed == [None]
    assert owner._pump._context is None
    assert context is not None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_cancel_failure_does_not_skip_context_and_service_cleanup(
    monkeypatch, error_type
):
    host = _FakeHost()
    launcher = _ControlledLauncher()

    def launcher_factory(*args, **kwargs):
        return launcher

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    failure = error_type("cancel failed")

    def cancel(_self, _plan):
        raise failure

    monkeypatch.setattr(_FakeContext, "cancel_frame", cancel)
    owner.start()
    owner._pump._pending_plan = SimpleNamespace(step_result=0)
    with pytest.raises(error_type) as caught:
        owner.close()
    assert caught.value is failure
    assert "destroy" in [name for name, _thread in _FakeContext.operations]
    assert owner._pump._context is None
    assert owner._pump._pending_plan is None
    assert [name for name, _thread in launcher.events] == ["stop-start", "stop-end"]
    assert [event[0] for event in host.events if isinstance(event, tuple)][-2:] == [
        "host-close",
        "environment-close",
    ]
    assert not owner.retirement_complete  # Cleanup does not erase the original error.


def test_owner_lane_exit_failure_after_destroy_still_releases_retired_owners(
    monkeypatch,
):
    host = _FakeHost()
    launcher = _ControlledLauncher()

    def launcher_factory(*args, **kwargs):
        return launcher

    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(
        host, config, runtime_launcher_factory=launcher_factory
    )
    owner.start()
    original_lane = host.owner_lane

    @contextmanager
    def failing_lane():
        with original_lane():
            yield
        raise RuntimeError("owner lane exit failed")

    host.owner_lane = failing_lane
    with pytest.raises(RuntimeError, match="owner lane exit failed"):
        owner.close()
    assert owner._pump._context is None
    assert [name for name, _thread in launcher.events] == ["stop-start", "stop-end"]
    assert [event[0] for event in host.events if isinstance(event, tuple)][-2:] == [
        "host-close",
        "environment-close",
    ]


@pytest.mark.parametrize("axis, expected", [("Z", 4), ("Y", 5)])
def test_host_up_axis_reaches_openxr_context(axis, expected):
    owner = ovxr_session.OvxrSession(_FakeHost(), _config(up_axis=axis))
    try:
        owner.start()
        assert owner._pump._context.desc.up_axis == expected
    finally:
        owner.close()


def test_stereo_camera_paths_are_required_and_axis_is_validated():
    with pytest.raises(TypeError, match="camera_paths"):
        ovxr_session.OvxrSessionConfig()
    with pytest.raises(ValueError, match="up_axis"):
        _config(up_axis="X")


def test_failure_after_worker_ready_before_start_returns_retires_without_close(
    monkeypatch,
):
    """A failed start never transfers quiescence responsibility to its caller."""
    may_finish_bootstrap = threading.Event()
    environment_closed = threading.Event()
    failure = RuntimeError("first frame failed")

    class Host(_FakeHost):
        def close_environment_on_render_thread(self):
            super().close_environment_on_render_thread()
            environment_closed.set()

    class StartupCondition(threading.Condition):
        def wait_for(self, predicate, timeout=None):
            if timeout is None:
                may_finish_bootstrap.set()
                return super().wait_for(lambda: owner.shutdown_requested, timeout=1.0)
            return super().wait_for(predicate, timeout)

    def bootstrap(_pump):
        assert may_finish_bootstrap.wait(timeout=1.0)

    def fail_frame(_pump):
        raise failure

    monkeypatch.setattr(frame_pump.OvxrFramePump, "bootstrap", bootstrap)
    monkeypatch.setattr(frame_pump.OvxrFramePump, "render_reused", fail_frame)
    owner = ovxr_session.OvxrSession(Host(), _config())
    owner._worker_condition = StartupCondition()
    try:
        with pytest.raises(RuntimeError) as caught:
            owner.start()
        assert caught.value is failure
        assert environment_closed.wait(timeout=1.0)
        owner._worker.join(timeout=1.0)
        assert not owner._worker.is_alive()
        assert owner.retirement_complete
    finally:
        with pytest.raises(RuntimeError):
            owner.close()


@pytest.mark.parametrize("failure", ["vulkan_missing", "external_memory", "attach"])
def test_unavailable_camera_panels_do_not_block_input_handles(
    monkeypatch, tmp_path, caplog, failure
):
    import ctypes
    from unittest.mock import Mock
    from isaaccapture.ovxr_session_manager.openxr_quad import OpenXRCameraPanels

    library = tmp_path / "panels.so"
    library.touch()
    native = SimpleNamespace(
        **{
            "isaacteleop_pip_" + name: Mock(return_value=value)
            for name, value in {
                "create": 123,
                "max_cameras": 3,
                "max_image_dimension": 4096,
                "supports_cuda": 1,
                "attach": -3 if failure == "attach" else 0,
                "destroy": 0,
                "wait_cuda": 0,
                "layer": 0,
                "configure": 0,
                "publish_rgba": 0,
                "publish_cuda_array": 0,
            }.items()
        }
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: native)
    monkeypatch.setattr(
        camera_panels,
        "OpenXRCameraPanels",
        lambda panels: OpenXRCameraPanels(panels, library_path=str(library)),
    )
    vulkan = (
        None
        if failure == "vulkan_missing"
        else SimpleNamespace(
            external_memory_handle_types=0 if failure == "external_memory" else 1,
            vk_physical_device=1,
            vk_device=2,
            vk_queue=3,
            queue_family_index=0,
        )
    )
    attempts = []

    def get_vulkan(self):
        attempts.append(None)
        if len(attempts) > 3:
            raise RuntimeError("optional panels stalled core readiness")
        return vulkan

    monkeypatch.setattr(_FakeContext, "get_vulkan_handles", get_vulkan)
    consumer = CameraPanelConsumer((CameraPanelConfig("/Camera/Head"),))
    owner = _session_with_consumers(_FakeHost(), _config(), frame_consumers=(consumer,))
    try:
        assert owner.start()
        assert owner.external_oxr_handles() == (11, 22, 33, 44)
        assert consumer.composition_layers() == ()
        assert "Camera panels disabled" in caplog.text
    finally:
        owner.close()
    assert native.isaacteleop_pip_attach.call_count == int(failure == "attach")
    native.isaacteleop_pip_destroy.assert_called_once()


@pytest.mark.parametrize("native_fails", [False, True])
def test_camera_completion_failure_still_retires_independent_mailboxes(native_fails):
    events = []
    pending = [True]

    def finish_native():
        events.append("wait")
        if native_fails and pending[0]:
            raise RuntimeError("native completion failed")

    def blocked():
        events.append("blocked")
        if pending[0]:
            raise RuntimeError("map still busy")

    consumer = CameraPanelConsumer((CameraPanelConfig("/Camera/Head"),))
    consumer._native = SimpleNamespace(
        finish_frame=finish_native, close=lambda: events.append("native close")
    )
    consumer._mailboxes = [
        SimpleNamespace(close=blocked),
        SimpleNamespace(close=lambda: events.append("independent")),
    ]
    with pytest.raises(RuntimeError):
        consumer.close()
    assert events == ["wait", "blocked", "independent"]
    assert consumer._native is not None
    pending[0] = False
    consumer.close()
    assert events[-4:] == ["wait", "blocked", "independent", "native close"]
