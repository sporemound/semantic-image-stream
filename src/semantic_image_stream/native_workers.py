"""Serial daemon workers whose native operations outlive async cancellation safely."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import queue
import threading
from typing import Callable

log = logging.getLogger(__name__)


class NativeWorkerClosed(RuntimeError):
    """Controlled rejection after a native worker has been retired."""


@dataclass
class _Job:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    function: Callable
    args: tuple


@dataclass
class _Release:
    callback: Callable | None


class NativeWorker:
    """Reuse one native thread and release its resources after actual work ends.

    Cancelling an awaiting task cannot interrupt native code. Pending cancelled
    jobs are skipped; retirement also rejects queued work, then runs optional
    cleanup on this same worker. ``drained`` is set only after cleanup finishes.
    Callers can coordinate a later worker's model loading using that event.
    """

    def __init__(self, name="sis-native-work"):
        self.drained = threading.Event()
        self.release_error_type = ""
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._retired = False
        self._thread = threading.Thread(target=self._work, daemon=True, name=name)
        self._thread.start()

    @property
    def retired(self):
        with self._lock:
            return self._retired

    async def run(self, function, *args):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if self._retired:
                raise NativeWorkerClosed("The native worker has stopped.")
            self._queue.put_nowait(_Job(loop, future, function, args))
        return await future

    def retire(self, release=None):
        """Queue cleanup once without waiting on native work or the caller's UI."""
        with self._lock:
            if not self._retired:
                self._retired = True
                self._queue.put_nowait(_Release(release))
        return self.drained

    def close(self, release=None):
        return self.retire(release)

    @staticmethod
    def _finish(future, value, error):
        if future.done():
            return
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)

    def _deliver(self, job, value=None, error=None):
        if job.loop.is_closed():
            return
        try:
            job.loop.call_soon_threadsafe(self._finish, job.future, value, error)
        except RuntimeError:
            pass  # The owning loop may close while real native work finishes.

    def _execute(self, job):
        try:
            value = job.function(*job.args)
        except BaseException as error:
            if isinstance(error, StopIteration):
                error = RuntimeError("The native operation could not complete.")
            self._deliver(job, error=error)
        else:
            self._deliver(job, value=value)

    def _work(self):
        try:
            while True:
                job = self._queue.get()
                if isinstance(job, _Release):
                    try:
                        if job.callback is not None:
                            job.callback()
                    except BaseException as error:
                        self.release_error_type = type(error).__name__
                        log.warning("Native worker cleanup failed (%s)", self.release_error_type)
                    return
                with self._lock:
                    retired = self._retired
                if job.future.cancelled():
                    pass
                elif retired:
                    self._deliver(job, error=NativeWorkerClosed("The native worker has stopped."))
                else:
                    self._execute(job)
                # An idle persistent thread must not retain the last model,
                # input tensors, transcript snapshot, result or exception.
                del job
        finally:
            self.drained.set()
