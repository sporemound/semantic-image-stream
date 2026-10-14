"""No native models: controlled calls exercise real daemon worker boundaries."""
import asyncio
import gc
import threading
import time
import weakref

import pytest

from semantic_image_stream.native_workers import NativeWorker, NativeWorkerClosed


async def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "The fake worker did not reach its expected boundary"
        await asyncio.sleep(.005)


async def finish(worker, release=None):
    worker.retire(release)
    await until(worker.drained.is_set)


@pytest.mark.asyncio
async def test_calls_reuse_one_serial_daemon_thread():
    worker = NativeWorker("sis-test-serial")
    active, maximum = 0, 0
    lock = threading.Lock()

    def work(value):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.005)
        with lock:
            active -= 1
        return value, threading.get_ident(), threading.current_thread().daemon

    try:
        results = await asyncio.gather(*(worker.run(work, index) for index in range(12)))
        assert [value for value, _, _ in results] == list(range(12))
        assert len({identity for _, identity, _ in results}) == 1
        assert all(daemon for _, _, daemon in results) and maximum == 1
    finally:
        await finish(worker)


@pytest.mark.asyncio
async def test_cancelled_pending_call_never_enters_native_code():
    worker = NativeWorker()
    entered, release = threading.Event(), threading.Event()
    invoked = []

    def blocking():
        entered.set()
        assert release.wait(3)
        return "first"

    active = asyncio.create_task(worker.run(blocking))
    try:
        await until(entered.is_set)
        pending = asyncio.create_task(worker.run(lambda: invoked.append(True)))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        assert await active == "first"
        assert await worker.run(lambda: "after cancellation") == "after cancellation"
        assert invoked == []
    finally:
        release.set()
        await asyncio.gather(active, return_exceptions=True)
        await finish(worker)


@pytest.mark.asyncio
async def test_retirement_waits_for_actual_cancelled_call_before_releasing_once():
    worker = NativeWorker()
    entered, release = threading.Event(), threading.Event()
    order = []

    def blocking():
        entered.set()
        assert release.wait(3)
        order.append(("native finished", threading.get_ident()))

    task = asyncio.create_task(worker.run(blocking))
    try:
        await until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        drained = worker.retire(lambda: order.append(("released", threading.get_ident())))
        assert drained is worker.drained and not drained.is_set()
        assert worker.retired
        worker.close(lambda: order.append(("duplicate", 0)))
        release.set()
        await until(drained.is_set)
        assert [name for name, _ in order] == ["native finished", "released"]
        assert order[0][1] == order[1][1]
        with pytest.raises(NativeWorkerClosed):
            await worker.run(lambda: None)
    finally:
        release.set()
        await finish(worker)


@pytest.mark.asyncio
async def test_retirement_rejects_queued_work_without_running_it():
    worker = NativeWorker()
    entered, release = threading.Event(), threading.Event()
    invoked = []

    def blocking():
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(worker.run(blocking))
    try:
        await until(entered.is_set)
        pending = asyncio.create_task(worker.run(lambda: invoked.append(True)))
        await asyncio.sleep(0)
        worker.retire()
        release.set()
        await active
        with pytest.raises(NativeWorkerClosed):
            await pending
        await until(worker.drained.is_set)
        assert invoked == []
    finally:
        release.set()
        await asyncio.gather(active, return_exceptions=True)
        await finish(worker)


@pytest.mark.asyncio
async def test_native_failure_does_not_kill_worker_or_echo_diagnostics():
    worker = NativeWorker()

    def failure():
        raise MemoryError("private input must not be logged")

    try:
        with pytest.raises(MemoryError):
            await worker.run(failure)
        assert await worker.run(lambda: "recovered") == "recovered"
    finally:
        await finish(worker)


@pytest.mark.asyncio
async def test_cleanup_error_still_drains_and_logs_only_type(caplog):
    worker = NativeWorker()

    def failure():
        raise RuntimeError("private model path")

    await finish(worker, failure)
    assert worker.release_error_type == "RuntimeError"
    assert "RuntimeError" in caplog.text and "private model path" not in caplog.text


@pytest.mark.asyncio
async def test_idle_worker_releases_last_bound_function_and_result():
    worker = NativeWorker()

    class OwnedModel:
        def use(self):
            return self

    model = OwnedModel()
    reference = weakref.ref(model)
    try:
        result = await worker.run(model.use)
        assert result is model
        del result, model
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None
    finally:
        await finish(worker)


def test_finished_work_can_retire_after_owning_loop_is_closed():
    worker = NativeWorker()
    entered, release = threading.Event(), threading.Event()
    cleaned = []

    def blocking():
        entered.set()
        assert release.wait(3)
        return object()

    async def run_then_close_loop():
        task = asyncio.create_task(worker.run(blocking))
        await until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_then_close_loop())
    worker.retire(lambda: cleaned.append(True))
    release.set()
    assert worker.drained.wait(3)
    assert cleaned == [True]
