# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused ownership and lifecycle tests for external OVXR presentation."""

from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import ClassVar
import pytest
import isaaccapture.teleop_session_manager.ovxr_session as ovxr_session


class _FakeOvxrError(RuntimeError):
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
            xr_instance=11, xr_session=22, xr_space=33, xr_get_instance_proc_addr=44
        )

    def set_frame_render_products(self, products):
        assert threading.get_native_id() in self.__class__.owner_lane_threads
        self.__class__.selected_products = tuple(products)

    def get_vulkan_handles(self):
        return SimpleNamespace()

    def submit_openxr_blank_frame(self):
        return (0, SimpleNamespace())

    def begin_frame(self):
        self.__class__.operations.append(("begin", threading.get_native_id()))
        return (0, SimpleNamespace(step_result=0))

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
        return (result, result in self.__class__.seal_bind_results)

    def submit_frame(self, _plan, *, additional_layers=()):
        self.__class__.operations.append(("submit", threading.get_native_id()))
        return (0, SimpleNamespace(submitted=True))

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
        StageUpAxis=SimpleNamespace(Z=4),
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

    def offer_render_owner_transfer(self):
        self.offer_thread = threading.get_native_id()
        return object()

    def attach_on_render_thread(self, _capability):
        self.attach_thread = threading.get_native_id()
        self.events.append("attach")
        _FakeContext.ordinal_ledger = self.ordinal_ledger
        return ovxr_session.OvxrSessionBinding(
            self.renderer, self.stage, self.ordinal_ledger
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
    monkeypatch.setattr(ovxr_session, "_import_ovxr", _fake_ovxr)
    oxr = ModuleType("isaaccapture.oxr")
    oxr.OpenXRSessionHandles = lambda *values: values
    monkeypatch.setitem(__import__("sys").modules, "isaaccapture.oxr", oxr)


def _config(**kwargs):
    return ovxr_session.OvxrSessionConfig(
        camera_paths=("/left", "/right"), auto_launch_cloudxr=False, **kwargs
    )


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
    return (thread, errors)


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
        assert not any((isinstance(event, tuple) for event in host.events))
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
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: fake_ovxr)
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
        ] == [("publish", "step-1"), ("publish", "step-2"), ("publish", "reset-1")]
    finally:
        release_frame.set()
        owner.close()


@pytest.mark.parametrize("force", [False, True])
def test_shutdown_rejects_late_publication_without_error(force):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    try:
        owner.request_shutdown()
        assert owner.request_render("late-step", force=force) is None
        assert host.publish_threads == []
        assert owner._worker_pending_ordinal is None
    finally:
        owner.close()
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
    try:
        assert owner.request_render("last-step") is None
        assert host.publish_threads == [threading.get_native_id()]
        assert owner._worker_pending_ordinal is None
        assert not host.publication_closed
    finally:
        owner.close()
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
    context = owner._context
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
    try:
        assert owner._context.desc.host_wait_render_output is True
    finally:
        owner.close()


def test_bind_timeout_reconciles_a_sealed_ordinal_then_reserves_a_fresh_one():
    host = _FakeHost()
    _FakeContext.seal_bind_results = {0, 1}
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    owner._context.bind_results = [1, 0]
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


def test_missing_core_phase_method_still_rejects_before_host_transfer(monkeypatch):
    runtime = _fake_ovxr()
    runtime.Context = type("BrokenContext", (_FakeContext,), {"cancel_frame": None})
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: runtime)
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
    owner._context.bind_results = [1, 0]
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
    owner._context.bind_results = [2, 0]
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
    owner._context.bind_results = [3]
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
    assert not owner._closed
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
    context = owner._context
    try:
        owner.request_shutdown()
        assert waiting.wait(timeout=1.0)
        assert owner._worker.is_alive()
        assert owner._context is context
        assert not owner._closed
        assert not owner.retirement_complete
        assert "destroy" not in [name for name, _thread in _FakeContext.operations]
        assert not any(
            (
                isinstance(event, tuple)
                and event[0] in {"host-close", "environment-close"}
                for event in host.events
            )
        )
    finally:
        owner.close()
    assert not owner._worker.is_alive()
    assert owner.retirement_complete
    assert _FakeContext.operations.count(("destroy", host.attach_thread)) == 1


def test_close_does_not_block_on_startup_teardown_lock(monkeypatch):
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
    monkeypatch.setattr(owner, "_bring_up_openxr", fail_start)
    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(owner, "_interrupt_owned_openxr_runtime", resume.set)
    try:
        with pytest.raises(RuntimeError) as caught:
            owner.start()
        assert caught.value is failure
        assert entered.wait(timeout=1.0)
        closing, errors = _call_in_thread(owner.close)
        closing.join(timeout=0.5)
        assert not closing.is_alive()
        assert errors == [failure]
    finally:
        resume.set()
        owner._worker.join(timeout=1.0)


@pytest.mark.parametrize("abort_fails", [False, True])
def test_worker_thread_start_failure_rolls_back_host_and_preserves_first_error(
    monkeypatch, abort_fails, caplog
):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    start_error = RuntimeError("native thread creation failed")
    original_cause = OSError("thread resources unavailable")
    if hasattr(start_error, "add_note"):
        start_error.add_note("Original thread diagnostic")
    original_notes = getattr(start_error, "__notes__", None)
    start_error.add_note = None
    abort_error = RuntimeError("host abort failed")

    def fail_start(_thread):
        raise start_error from original_cause

    def abort():
        host.events.append("abort")
        if abort_fails:
            raise abort_error

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    monkeypatch.setattr(host, "abort", abort)
    with pytest.raises(RuntimeError) as error:
        owner.start()
    assert error.value is start_error
    assert error.value.__cause__ is original_cause
    assert error.value.__notes__ == original_notes
    diagnostics = [
        record
        for record in caplog.records
        if record.getMessage() == "Cleanup after failed render-worker start also failed"
    ]
    assert [record.exc_info[1] for record in diagnostics] == (
        [abort_error] if abort_fails else []
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

    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: FakeLauncher(),
    )
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    owner.start()
    owner.close()
    assert launcher_events == [
        ("create", owner.render_worker_native_thread_id),
        ("stop", owner.render_worker_native_thread_id),
    ]


def test_shutdown_falls_back_only_when_request_exit_cannot_wake_frame_wait(monkeypatch):
    host = _FakeHost()
    frame_wait_entered = threading.Event()
    runtime_stopped = threading.Event()
    launcher_events = []

    class BlockingContext(_FakeContext):
        def begin_frame(self):
            frame_wait_entered.set()
            runtime_stopped.wait(timeout=1.0)
            return super().begin_frame()

    class FakeLauncher:
        def interrupt_runtime(self):
            launcher_events.append(("interrupt", threading.get_native_id()))
            runtime_stopped.set()
            raise RuntimeError("runtime interrupt failed")

        def stop(self):
            launcher_events.append(("stop", threading.get_native_id()))

    fake_ovxr = _fake_ovxr()
    fake_ovxr.Context = BlockingContext
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: fake_ovxr)
    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: FakeLauncher(),
    )
    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    _FakeContext.request_exit_error = RuntimeError("request exit failed")
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    owner.start()
    assert owner.request_render("step-1") == 10
    assert frame_wait_entered.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="request exit failed") as error:
        owner.close()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "runtime interrupt failed"
    assert [name for name, _thread in launcher_events].count("interrupt") == 1
    assert launcher_events[-1] == ("stop", owner.render_worker_native_thread_id)
    with pytest.raises(RuntimeError, match="request exit failed"):
        owner.close()
    assert [name for name, _thread in launcher_events].count("interrupt") == 1


def test_context_teardown_completion_prevents_interrupt_during_cloudxr_stop(
    monkeypatch, caplog
):
    host = _FakeHost()
    launcher = _ControlledLauncher(block_stop=True)
    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: launcher,
    )
    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    owner.start()
    close_thread, close_errors = _call_in_thread(owner.close)
    try:
        assert launcher.stop_started.wait(timeout=1.0)
        assert owner._ovxr_teardown_complete.is_set()
        assert not owner._cloudxr_stop_complete.is_set()
        time.sleep(0.03)
        assert close_thread.is_alive()
        assert "interrupt" not in [name for name, _thread in launcher.events]
        assert "interrupting the owned OpenXR runtime" not in caplog.text
    finally:
        launcher.allow_stop.set()
        close_thread.join(timeout=1.0)
    assert not close_thread.is_alive()
    assert close_errors == []
    assert owner._cloudxr_stop_complete.is_set()
    assert launcher.events == [
        ("stop-start", owner.render_worker_native_thread_id),
        ("stop-end", owner.render_worker_native_thread_id),
    ]


def test_destroy_failure_retains_borrowed_providers_and_forbids_retry(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher()
    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: launcher,
    )
    failure = RuntimeError("destroy failed")
    _FakeContext.destroy_error = failure
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    owner.start()
    context, binding = (owner._context, owner._binding)
    with pytest.raises(RuntimeError, match="destroy failed") as error:
        owner.close()
    assert error.value is failure
    assert owner._context is context
    assert owner._binding is binding
    assert owner._cloudxr_launcher is launcher
    assert not owner.retirement_complete
    assert owner._ovxr_teardown_complete.is_set()
    assert launcher.events == []
    assert not any(
        (
            isinstance(event, tuple) and event[0] in ("host-close", "environment-close")
            for event in host.events
        )
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
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: runtime)
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    if failure == "success":
        owner.close()
    else:
        with pytest.raises(RuntimeError):
            owner.close()
    consumed = failure in ("native-result", "success")
    assert (owner._context is None) == consumed
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
    ovxr_session._create_cloudxr_launcher("profile.env", accept_eula=accepted)
    assert captured[0]["accept_eula"] is accepted


def test_failed_exit_request_does_not_prevent_proven_context_retirement():
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    _FakeContext.request_exit_error = RuntimeError("request exit failed")
    with pytest.raises(RuntimeError, match="request exit failed"):
        owner.close()
    assert owner._context is None
    assert [event[0] for event in host.events if isinstance(event, tuple)][-2:] == [
        "host-close",
        "environment-close",
    ]


def test_cloudxr_stop_failure_is_reported_without_runtime_interrupt(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher(stop_error=RuntimeError("cloudxr stop failed"))
    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: launcher,
    )
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    owner.start()
    with pytest.raises(RuntimeError, match="cloudxr stop failed"):
        owner.close()
    assert owner._ovxr_teardown_complete.is_set()
    assert owner._cloudxr_stop_complete.is_set()
    assert launcher.events == [("stop-start", owner.render_worker_native_thread_id)]


def test_partial_startup_failure_marks_ovxr_teardown_before_cloudxr_stop(monkeypatch):
    host = _FakeHost()
    launcher = _ControlledLauncher(block_stop=True)
    monkeypatch.setattr(
        ovxr_session,
        "_create_cloudxr_launcher",
        lambda _path, *, accept_eula=False: launcher,
    )
    monkeypatch.setattr(ovxr_session, "_WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    _FakeContext.create_error = RuntimeError("create failed")
    config = _config()
    config.cloudxr_env_file = "/cloudxr.env"
    config.auto_launch_cloudxr = True
    owner = ovxr_session.OvxrSession(host, config)
    start_thread, start_errors = _call_in_thread(owner.start)
    assert launcher.stop_started.wait(timeout=1.0)
    assert owner._ovxr_teardown_complete.is_set()
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


def test_render_failure_waits_for_borrowers_and_preserves_first_error(monkeypatch):
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner.start()
    failure = RuntimeError("bind failed")
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
        assert not owner._closed
        assert "destroy" not in [name for name, _thread in _FakeContext.operations]
        assert not any(
            (
                isinstance(event, tuple)
                and event[0] in {"host-close", "environment-close"}
                for event in host.events
            )
        )
        for read in (
            owner.external_oxr_handles,
            lambda: owner.request_render("step-2"),
        ):
            with pytest.raises(RuntimeError) as caught:
                read()
            assert caught.value is failure
    finally:
        with pytest.raises(RuntimeError) as caught:
            owner.close()
        assert caught.value is failure
    with pytest.raises(RuntimeError) as caught:
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
            if self.frames == 4:
                fourth_frame.set()
                assert release_frame.wait(1.0)
            return super().begin_frame()

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.get_native_id() == owner.render_worker_native_thread_id:
                idle_waits.append((owner._worker_last_ordinal, timeout))
            return super().wait(timeout)

    fake = _fake_ovxr()
    fake.Context = PacedContext
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: fake)
    _FakeContext.request_exit_callback = release_frame.set
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._worker_condition = ObservedCondition()
    try:
        owner.start()
        owner._context.bind_results = [bind_result]
        owner.request_render("paused-scene")
        assert fourth_frame.wait(1.0)
    finally:
        owner.close()
    assert not [wait for wait in idle_waits if wait[0] is not None]
    assert [
        event
        for event in host.events
        if isinstance(event, tuple) and event[0] == "publish"
    ] == [("publish", "paused-scene")]
    assert owner._context is None


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
            return (begin_result, None)

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.get_native_id() == owner.render_worker_native_thread_id:
                waits.append(timeout)
                if len(waits) == 3:
                    backed_off.set()
            return super().wait(timeout)

    fake = _fake_ovxr()
    fake.Context = UnreadyContext
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: fake)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config())
    owner._worker_condition = ObservedCondition()
    try:
        owner.start()
        if begin_result is not None:
            owner.request_render("unready-scene")
        assert backed_off.wait(1.0)
    finally:
        owner.close()
    assert len(waits) >= 3
    assert all((timeout == 0.01 for timeout in waits))
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
    monkeypatch.setattr(ovxr_session, "_import_ovxr", lambda: fake)
    host = _FakeHost()
    owner = ovxr_session.OvxrSession(host, _config(), max_blank_frames=1)
    try:
        owner.start()
        assert missing_observed.wait(1.0)
        assert owner.external_oxr_handles() is None
        if publish_scene:
            owner.request_render("paused-scene")
            _wait_for_operation("submit")
        peer_ready.set()
        with owner._worker_condition:
            ready = owner._worker_condition.wait_for(
                lambda: owner._worker_handles is not None, timeout=1.0
            )
        assert ready, "Late XR session never reached the Teleop input borrower"
        assert handles_observed.is_set()
        assert owner.external_oxr_handles() == (11, 22, 33, 44)
        assert set(get_threads) == {host.attach_thread}
    finally:
        owner.close()


@pytest.mark.parametrize("mode,expected", [("ar", 5), ("vr", 4)])
def test_session_mode_is_forwarded_to_ovxr(mode, expected):
    owner = ovxr_session.OvxrSession(_FakeHost(), _config(session_mode=mode))
    try:
        owner.start()
        assert owner._context.desc.session_mode == expected
    finally:
        owner.close()


def test_config_rejects_unknown_session_mode():
    with pytest.raises(ValueError, match="session_mode must be 'ar' or 'vr'"):
        _config(session_mode="automatic")


@pytest.mark.parametrize(
    "scale", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
)
def test_world_scale_rejects_nonfinite_or_nonpositive_values(scale):
    with pytest.raises(ValueError):
        _config(meters_per_unit=scale)
