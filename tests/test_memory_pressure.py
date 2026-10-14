"""Memory diagnostics are mocked; these checks never load a native Windows DLL."""
import ctypes
from types import SimpleNamespace

import pytest

from semantic_image_stream import memory_pressure as pressure


class MockMemoryApi:
    def __init__(self, available=512 * 1024 * 1024, total=8 * 1024 * 1024 * 1024,
                 result=1, memory_load=50, header=None):
        self.available, self.total = available, total
        self.result, self.memory_load, self.header = result, memory_load, header
        self.calls = []

    def __call__(self, pointer):
        status = ctypes.cast(pointer, ctypes.POINTER(pressure._MemoryStatusEx)).contents
        self.calls.append(status.dwLength)
        status.dwMemoryLoad = self.memory_load
        status.ullTotalPageFile, status.ullAvailPageFile = self.total, self.available
        if self.header is not None:
            status.dwLength = self.header
        return self.result


@pytest.fixture(autouse=True)
def mock_windows(monkeypatch):
    api = MockMemoryApi()
    loads = []

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return SimpleNamespace(GlobalMemoryStatusEx=api)

    # Replace only this module's ctypes/sys bindings, leaving the test process
    # and other modules' actual platform/native libraries untouched.
    monkeypatch.setattr(pressure, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(pressure, "ctypes", SimpleNamespace(
        WinDLL=load, POINTER=ctypes.POINTER, c_int=ctypes.c_int,
        sizeof=ctypes.sizeof, byref=ctypes.byref))
    return api, loads


def test_windows_memory_snapshot_reads_commit_not_physical_ram(mock_windows):
    api, loads = mock_windows
    assert pressure.available_commit_bytes() == 512 * 1024 * 1024
    assert ctypes.sizeof(pressure._MemoryStatusEx) == 64
    assert api.calls == [64]
    assert loads == [("kernel32", {"use_last_error": True})]
    assert api.restype is ctypes.c_int
    assert api.argtypes == [ctypes.POINTER(pressure._MemoryStatusEx)]


def test_pixel_memory_lock_allows_nested_operations_in_the_same_thread():
    with pressure.PIXEL_MEMORY_LOCK:
        assert pressure.PIXEL_MEMORY_LOCK.acquire(blocking=False)
        pressure.PIXEL_MEMORY_LOCK.release()


def test_non_windows_has_no_native_library_load(monkeypatch, mock_windows):
    api, loads = mock_windows
    monkeypatch.setattr(pressure, "sys", SimpleNamespace(platform="linux"))
    assert pressure.available_commit_bytes() is None
    assert not loads and not api.calls


@pytest.mark.parametrize("result", [0, None, "success", .5, float("nan")])
def test_failed_or_malformed_native_status_is_unavailable(mock_windows, result):
    api, _ = mock_windows
    api.result = result
    assert pressure.available_commit_bytes() is None


@pytest.mark.parametrize("changes", [{"total": 0}, {"total": -1}, {"available": -1},
                                      {"available": 10 * 1024 * 1024 * 1024},
                                      {"memory_load": 101}, {"header": 0},
                                      {"available": float("inf")}, {"total": float("nan")}])
def test_malformed_memory_structures_are_not_used(mock_windows, changes):
    api, _ = mock_windows
    for name, value in changes.items():
        setattr(api, name, value)
    assert pressure.available_commit_bytes() is None


def test_unavailable_windows_api_is_nonfatal(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("API unavailable")

    monkeypatch.setattr(pressure.ctypes, "WinDLL", unavailable)
    assert pressure.available_commit_bytes() is None
    pressure.require_memory(1024)


def test_zero_available_commit_is_real_pressure(mock_windows):
    api, _ = mock_windows
    api.available = 0
    assert pressure.available_commit_bytes() == 0
    with pytest.raises(pressure.MemoryPressureError):
        pressure.require_memory(1, reserve_bytes=0)


def test_allocation_threshold_includes_reserve_and_accepts_exact_headroom(mock_windows):
    api, _ = mock_windows
    api.available = 512 + 128
    pressure.require_memory(512, reserve_bytes=128)
    api.available -= 1
    with pytest.raises(pressure.MemoryPressureError) as error:
        pressure.require_memory(512, reserve_bytes=128)
    assert str(error.value) == pressure.MEMORY_PRESSURE_MESSAGE
    assert "Close unused apps" in str(error.value) and "retry automatically" in str(error.value)


def test_default_reserve_is_128_mib(mock_windows):
    api, _ = mock_windows
    api.available = 1 + 128 * 1024 * 1024
    pressure.require_memory(1)
    api.available -= 1
    with pytest.raises(pressure.MemoryPressureError):
        pressure.require_memory(1)


@pytest.mark.parametrize("available", [None, -1, "unknown", True, float("inf"), float("nan")])
def test_unavailable_or_malformed_snapshot_does_not_create_a_platform_failure(monkeypatch, available):
    monkeypatch.setattr(pressure, "available_commit_bytes", lambda: available)
    pressure.require_memory(1024)


@pytest.mark.parametrize("estimate,reserve", [(0, 0), (-1, 0), (True, 0), (1024, -1),
                                               (1024, False), (1.5, 0), (float("inf"), 0),
                                               (float("nan"), 0), (1 << 64, 0)])
def test_allocation_estimates_need_supported_integer_bytes(estimate, reserve):
    with pytest.raises(ValueError):
        pressure.require_memory(estimate, reserve_bytes=reserve)


@pytest.mark.parametrize("error", [MemoryError(), pressure.MemoryPressureError(),
    RuntimeError("mkl_malloc: failed to allocate memory"),
    RuntimeError("mki_malloc failedallocatememory"),
    RuntimeError("Intel MKL: Memory allocation failed"),
    RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 4096 bytes"),
    RuntimeError("std::bad_alloc"), RuntimeError("Failed to allocate memory")])
def test_known_native_allocation_failures_are_classified(error):
    assert pressure.is_memory_allocation_error(error)


@pytest.mark.parametrize("error", [RuntimeError("Private scene: memory of a coastline."),
    RuntimeError("MKL: invalid matrix dimensions"), RuntimeError("DefaultCPUAllocator: invalid alignment"),
    RuntimeError("Could not download an image"), ValueError("Invalid model settings"),
    "failed to allocate memory", None])
def test_other_errors_and_user_text_are_not_classified(error):
    assert not pressure.is_memory_allocation_error(error)


def test_wrapped_allocation_error_is_recognized_without_exposing_error_text():
    outer = RuntimeError("The selected image model could not load.")
    outer.__cause__ = MemoryError("Private source text must never be logged by classification")
    assert pressure.is_memory_allocation_error(outer)


def test_cyclic_and_unprintable_exception_chains_are_safe():
    cyclic = RuntimeError("Unrelated failure")
    cyclic.__cause__ = cyclic
    assert not pressure.is_memory_allocation_error(cyclic)

    class UnprintableError(RuntimeError):
        def __str__(self):
            raise ValueError("Cannot format error")

    assert not pressure.is_memory_allocation_error(UnprintableError())
