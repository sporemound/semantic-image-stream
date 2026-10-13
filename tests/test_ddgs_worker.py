"""Offline real child-process timeout, cancellation and bounded IPC contracts."""
import asyncio
import json
import logging
from pathlib import Path
import sys
import textwrap

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream import ddgs_worker, search_providers
from semantic_image_stream.search_providers import DuckDuckGoProvider, ProviderError


def worker_fixture(tmp_path, monkeypatch, body, *, windowed=False):
    script = tmp_path / "controlled_ddgs_worker.py"
    script.write_text(textwrap.dedent("""\
        import sys, time
        from semantic_image_stream import ddgs_worker as worker
        class Client:
            def __init__(self, **kwargs): pass
            def images(self, query, **kwargs):
                assert kwargs['backend'] == 'duckduckgo'
                assert kwargs['safesearch'] == 'moderate'
    """) + textwrap.indent(textwrap.dedent(body), "        ") + "\nworker._load_factory = lambda: Client\n"
                      + ("sys.stdin = sys.stdout = sys.stderr = None\n" if windowed else "")
                      + "raise SystemExit(worker.main())\n", encoding="utf-8")
    command = [sys.executable, str(script)]
    monkeypatch.setenv("PYTHONPATH", str(Path(search_providers.__file__).resolve().parents[1]))
    monkeypatch.setattr(search_providers, "_duckduckgo_worker_command", lambda: command)
    return command


def ready_body():
    return """
        return [{'title': 'Coastal cave', 'image': 'https://photos.example.org/cave.jpg',
                 'url': 'https://museum.example.org/cave', 'width': 800, 'height': 600}]
    """


async def wait_started(provider):
    for _ in range(200):
        if provider._worker_process is not None:
            return provider._worker_process
        await asyncio.sleep(.01)
    raise AssertionError("Controlled worker did not start")


@pytest.mark.asyncio
async def test_stuck_worker_is_reaped_before_next_request_then_source_recovers(tmp_path, monkeypatch):
    command = worker_fixture(tmp_path, monkeypatch, """
        if query == 'hang': time.sleep(60)
    """ + ready_body())
    settings = Settings(cache_dir=str(tmp_path), download_timeout=.6)
    provider = DuckDuckGoProvider(settings)
    task = asyncio.create_task(provider.search("hang", 1))
    process = await wait_started(provider)
    with pytest.raises(ProviderError, match="TimeoutError"):
        await task
    assert process.returncode is not None
    assert provider._worker_process is None and not provider._inflight
    assert not any("hang" in argument for argument in command)
    settings.download_timeout = 5
    images = await provider.search("coastal cave", 1)
    assert len(images) == 1 and images[0].title == "Coastal cave"
    assert provider._worker_process is None and not provider._inflight


@pytest.mark.asyncio
async def test_cancelled_worker_is_killed_and_reaped(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, "time.sleep(60)\nreturn []")
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=30))
    task = asyncio.create_task(provider.search("coastal cave", 1))
    process = await wait_started(provider)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode is not None
    assert provider._worker_process is None and not provider._inflight


@pytest.mark.asyncio
async def test_repeated_stop_during_reap_finishes_cleanup_before_latch_clears(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, "time.sleep(60)\nreturn []")
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=30))
    task = asyncio.create_task(provider.search("coastal cave", 1))
    process = await wait_started(provider)
    original_wait = process.wait
    reaping, release = asyncio.Event(), asyncio.Event()

    async def delayed_wait():
        reaping.set()
        await release.wait()
        return await original_wait()

    monkeypatch.setattr(process, "wait", delayed_wait)
    task.cancel()
    await asyncio.wait_for(reaping.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert provider._inflight and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert process.returncode is not None
    assert provider._worker_process is None and not provider._inflight


@pytest.mark.asyncio
async def test_close_reaps_current_worker(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, "time.sleep(60)\nreturn []")
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=30))
    task = asyncio.create_task(provider.search("coastal cave", 1))
    process = await wait_started(provider)
    await provider.close()
    with pytest.raises(ProviderError):
        await task
    assert process.returncode is not None and not provider._inflight


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="Windows windowed standard-pipe handles")
async def test_windowed_none_stdio_uses_real_inherited_os_pipes(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, ready_body(), windowed=True)
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=5))
    images = await provider.search("coastal cave", 1)
    assert len(images) == 1 and images[0].width == 800 and images[0].height == 600


@pytest.mark.asyncio
async def test_native_worker_error_echoes_only_class_not_private_inputs(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, "raise RuntimeError(query + ' private path/secret')")
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=5))
    with pytest.raises(ProviderError, match=r"search failed \(RuntimeError\)") as caught:
        await provider.search("private story", 1)
    assert "private" not in str(caught.value) and not provider._inflight


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"not JSON", b"[]", b'{"ok":true,"rows":"bad"}',
                                        b'{"ok":true,"rows":[false]}',
                                        b'{"ok":false,"error":"private story/path"}',
                                        b"x" * (ddgs_worker.MAX_OUTPUT_BYTES + 1)],
                         ids=["not-json", "wrong-root", "wrong-rows", "wrong-row", "private-error", "oversized"])
async def test_malformed_or_oversized_worker_protocol_is_rejected_and_reaped(tmp_path, monkeypatch, payload):
    script = tmp_path / "malformed_worker.py"
    script.write_text("import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(" + repr(payload)
                      + ")\nsys.stdout.buffer.flush()\n", encoding="utf-8")
    monkeypatch.setattr(search_providers, "_duckduckgo_worker_command", lambda: [sys.executable, str(script)])
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), download_timeout=5))
    with pytest.raises(ProviderError):
        await provider.search("coastal cave", 1)
    assert provider._worker_process is None and not provider._inflight


@pytest.mark.asyncio
async def test_worker_bounds_native_metadata_and_keeps_explicit_ai_flag(tmp_path, monkeypatch):
    worker_fixture(tmp_path, monkeypatch, """
        return [{'title': '<b>Coastal cave</b>', 'image': 'https://photos.example.org/cave.jpg',
                 'url': 'https://museum.example.org/cave', 'description': 'x' * 100_000,
                 'is_generated': True, 'unused_private_blob': 'secret'}]
    """)
    settings = Settings(cache_dir=str(tmp_path), download_timeout=5, exclude_ai_generated=False)
    provider = DuckDuckGoProvider(settings)
    images = await provider.search("coastal cave", 1)
    assert len(images) == 1 and images[0].ai_generated is True
    assert images[0].title == "Coastal cave"
    settings.exclude_ai_generated = True
    assert await provider.search("coastal cave", 1) == []


def test_worker_rejects_oversized_input_without_loading_optional_library(monkeypatch):
    import io
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", io.BytesIO(b"x" * (ddgs_worker.MAX_INPUT_BYTES + 1)))
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(ddgs_worker, "_load_factory", lambda: pytest.fail("Invalid input cannot load native code"))
    assert ddgs_worker.main() == 0
    assert json.loads(output.getvalue()) == {"ok": False, "error": "ValueError"}


@pytest.mark.parametrize("valid", [True, False])
def test_direct_worker_restores_parent_logging_level_on_success_or_error(monkeypatch, valid):
    import io
    request = {"query": "coastal cave", "limit": 1, "timeout": 5}
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", io.BytesIO(json.dumps(request).encode() if valid else b"not JSON"))
    monkeypatch.setattr(sys, "stdout", output)

    class Client:
        def __init__(self, **kwargs):
            pass

        def images(self, *args, **kwargs):
            return []

    monkeypatch.setattr(ddgs_worker, "_load_factory", lambda: Client)
    original_level = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        assert ddgs_worker.main() == 0
        assert logging.root.manager.disable == logging.WARNING
        assert json.loads(output.getvalue())["ok"] is valid
    finally:
        logging.disable(original_level)


def test_frozen_command_uses_hidden_dispatch_without_query(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert search_providers._duckduckgo_worker_command() == [sys.executable, "--ddgs-worker"]


def test_hidden_dispatch_runs_before_ui_or_audio_imports(monkeypatch):
    from semantic_image_stream import __main__
    monkeypatch.setattr(ddgs_worker, "main", lambda: 17)
    assert __main__.main(["--ddgs-worker"]) == 17
