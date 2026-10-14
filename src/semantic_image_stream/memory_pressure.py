"""Small, read-only Windows allocation checks; no models or process inspection.

GlobalMemoryStatusEx's ullAvailPageFile is the memory this process can still
commit, which may be smaller than system-wide available commit:
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
A successful check is a volatile snapshot, not an allocation guarantee.
"""
from __future__ import annotations

import ctypes
import re
import sys
import threading

DEFAULT_RESERVE_BYTES = 128 * 1024 * 1024
PIXEL_MEMORY_LOCK = threading.RLock()
_MAX_BYTES = (1 << 63) - 1
MEMORY_PRESSURE_MESSAGE = (
    "Windows is low on available memory. Close unused apps; "
    "image monitoring will retry automatically."
)


class MemoryPressureError(RuntimeError):
    """A large image allocation should wait for available commit headroom."""


class _MemoryStatusEx(ctypes.Structure):
    # Explicit widths preserve the Windows ABI even in mocked Linux tests.
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def available_commit_bytes() -> int | None:
    """Return process-available Windows commit, or None when it is unavailable."""
    if sys.platform != "win32":
        return None
    try:
        api = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        api.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        api.restype = ctypes.c_int
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        result = api(ctypes.byref(status))
        if not isinstance(result, (int, bool)) or not result:
            return None
        total, available = int(status.ullTotalPageFile), int(status.ullAvailPageFile)
        if (status.dwLength != ctypes.sizeof(status) or status.dwMemoryLoad > 100
                or not 0 < total <= _MAX_BYTES or not 0 <= available <= total):
            return None
        return available
    except Exception:
        # Unavailable diagnostics must not introduce a new platform failure.
        return None


def require_memory(estimated_bytes: int, reserve_bytes: int = DEFAULT_RESERVE_BYTES) -> None:
    """Reject a large allocation only when a valid snapshot lacks headroom."""
    for value, minimum in ((estimated_bytes, 1), (reserve_bytes, 0)):
        if type(value) is not int or not minimum <= value <= _MAX_BYTES:
            raise ValueError("Memory estimates must be finite byte counts; the estimate must be positive.")
    available = available_commit_bytes()
    if type(available) is int and 0 <= available <= _MAX_BYTES and available < estimated_bytes + reserve_bytes:
        raise MemoryPressureError(MEMORY_PRESSURE_MESSAGE)


_ALLOCATION_FAILURE = re.compile(
    r"(?:\b(?:mkl|mki)_malloc\b.{0,160}(?:fail|unable|cannot|returned\s+(?:null|0))|"
    r"\b(?:intel\s+)?mkl\b.{0,160}(?:memory\s+allocation\s+failed|out\s+of\s+memory)|"
    r"\bdefaultcpuallocator\b.{0,160}(?:not\s+enough\s+memory|out\s+of\s+memory|"
    r"(?:failed|cannot|unable).{0,40}allocat|allocat\w*.{0,40}failed)|"
    r"\bstd::bad_alloc\b|\bfailed\s+to\s+allocate\s+memory\b)",
    re.IGNORECASE | re.DOTALL,
)


def is_memory_allocation_error(error: BaseException) -> bool:
    """Recognize known allocators without returning or logging exception text."""
    visited = set()
    while isinstance(error, BaseException) and id(error) not in visited and len(visited) < 8:
        visited.add(id(error))
        if isinstance(error, (MemoryError, MemoryPressureError)):
            return True
        try:
            if _ALLOCATION_FAILURE.search(str(error)[:8192]):
                return True
        except Exception:
            return False
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return False
