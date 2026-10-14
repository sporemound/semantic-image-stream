"""One bounded DuckDuckGo request; the parent can terminate native I/O safely."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import math
import os
import sys

MAX_INPUT_BYTES = 4096
MAX_OUTPUT_BYTES = 2_000_000


def _pipe_stream(current, standard_handle: int, mode: str):
    if current is not None:
        return getattr(current, "buffer", current), False
    if sys.platform != "win32":
        raise OSError("Worker pipe unavailable")
    # PyInstaller windowed executables replace Python stdio with None. The
    # redirected OS pipe handles still exist; duplicate them before ownership
    # passes to the CRT so closing this stream never closes the original.
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                                      ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                      wintypes.BOOL, wintypes.DWORD]
    kernel.DuplicateHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.GetStdHandle(standard_handle & 0xffffffff)
    if not handle or handle == wintypes.HANDLE(-1).value:
        raise OSError("Worker pipe unavailable")
    duplicate = wintypes.HANDLE()
    process = kernel.GetCurrentProcess()
    if not kernel.DuplicateHandle(process, handle, process, ctypes.byref(duplicate), 0, False, 2):
        raise OSError("Worker pipe unavailable")
    try:
        flags = os.O_BINARY | (os.O_RDONLY if mode == "rb" else os.O_WRONLY)
        descriptor = msvcrt.open_osfhandle(duplicate.value, flags)
    except BaseException:
        kernel.CloseHandle(duplicate)
        raise
    return os.fdopen(descriptor, mode), True


def _load_factory():
    from .search_providers import _duckduckgo_client_factory
    return _duckduckgo_client_factory()


def _run_request(request):
    required = {"query", "limit", "timeout"}
    if (not isinstance(request, dict) or not required <= set(request)
            or not set(request) <= required | {"allow_nsfw"}):
        raise ValueError("Invalid worker request")
    query, limit, timeout = request["query"], request["limit"], request["timeout"]
    allow_nsfw = request.get("allow_nsfw", False)
    if (not isinstance(query, str) or not query.strip() or len(query) > 180
            or type(limit) is not int or not 1 <= limit <= 200
            or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0 or type(allow_nsfw) is not bool):
        raise ValueError("Invalid worker request")
    from .search_providers import _dimension, _generated_status, _mature_status, _rows, plain_text, sanitize_query
    query = sanitize_query(query)
    if not query:
        raise ValueError("Invalid worker request")
    engine = _load_factory()(timeout=timeout)
    rows = engine.images(query, max_results=limit, safesearch="off" if allow_nsfw else "moderate", backend="duckduckgo")
    if not isinstance(rows, list):
        raise ValueError("Invalid worker result")
    # Native source dictionaries may contain extra or excessive fields. Only
    # bounded discovery metadata crosses the child-process boundary.
    result = []
    for row in _rows(rows)[:limit]:
        bounded = {key: plain_text(row.get(key), max_length=length)
                   for key, length in {"title": 300, "image": 4096, "url": 4096,
                                       "source": 300, "description": 2000,
                                       "_sis_backend": 40}.items()}
        bounded.update(width=_dimension(row.get("width")), height=_dimension(row.get("height")))
        generated = _generated_status(row, row.get("title"), row.get("source"), row.get("description"))
        if generated is not None:
            bounded["ai_generated"] = generated
        mature = _mature_status(row)
        if mature is not None:
            bounded["mature_content"] = mature
        result.append(bounded)
    return result


def main() -> int:
    input_stream = output_stream = None
    own_input = own_output = False
    previous_log_disable = logging.root.manager.disable
    try:
        input_stream, own_input = _pipe_stream(sys.stdin, -10, "rb")
        output_stream, own_output = _pipe_stream(sys.stdout, -11, "wb")
        logging.disable(logging.CRITICAL)
        data = input_stream.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("Worker request too large")
        request = json.loads(data)
        # Suppress library chatter: the protocol never prints inputs, URLs,
        # file paths, model data, or exception messages.
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            rows = _run_request(request)
        response = {"ok": True, "rows": rows}
    except Exception as error:
        response = {"ok": False, "error": type(error).__name__[:80]}
    try:
        if output_stream is None:
            return 1
        data = json.dumps(response, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_OUTPUT_BYTES:
            data = b'{"ok":false,"error":"ValueError"}'
        output_stream.write(data)
        output_stream.flush()
        return 0
    except (OSError, ValueError):
        return 1
    finally:
        logging.disable(previous_log_disable)
        if own_input and input_stream is not None:
            input_stream.close()
        if own_output and output_stream is not None:
            output_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
