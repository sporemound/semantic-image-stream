"""Bounded native/pixel memory without changing recorded context or image quality."""
import asyncio
import io
import queue
import threading
from types import SimpleNamespace
import weakref

from PIL import Image, ImageOps, PngImagePlugin
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.image_download import DownloadError, ImageDownloader, IMAGE_DECODE_LOCK
from semantic_image_stream.memory_pressure import PIXEL_MEMORY_LOCK
from semantic_image_stream.models import TranscriptSegment
from semantic_image_stream.native_workers import NativeWorker


@pytest.mark.asyncio
async def test_production_native_calls_reuse_one_thread_until_retirement(tmp_path):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    worker = NativeWorker("test-ranking-reuse")
    pipeline._native_workers = {"ranking": worker}
    identities = []

    def rank(index):
        identities.append(threading.get_ident())
        return index

    try:
        values = [await pipeline._native_background("ranking", rank, index) for index in range(100)]
        assert values == list(range(100)) and len(set(identities)) == 1
        pipeline.stop()
        while not worker.drained.is_set():
            await asyncio.sleep(.001)
        assert worker.retired and not pipeline._native_workers
    finally:
        worker.retire()


@pytest.mark.asyncio
async def test_next_monitoring_waits_for_actual_old_decode_and_model_release(tmp_path):
    entered, release = threading.Event(), threading.Event()
    order = []
    state = SessionState(segments=[TranscriptSegment(0, 6, "A retained recorded passage.")])
    old = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None, state)
    worker = NativeWorker("test-speech-retirement")
    old._native_workers = {"speech": worker}

    class Model:
        pass

    class Transcriber:
        def __init__(self):
            self.model = Model()

        def transcribe(self):
            entered.set()
            assert release.wait(3)
            order.append("decode finished")

        def stop(self):
            order.append("stop requested")

        def close(self):
            self.model = None
            order.append("model released")

    transcriber = Transcriber()
    reference = weakref.ref(transcriber.model)
    old._transcriber = transcriber
    active = asyncio.create_task(old._native_background("speech", transcriber.transcribe))
    while not entered.is_set():
        await asyncio.sleep(.001)
    old.stop()
    new = Pipeline(old.settings, lambda *args: None, state)
    waiting = asyncio.create_task(new._wait_for_native_cleanup())
    try:
        await asyncio.sleep(.01)
        assert not waiting.done() and not worker.drained.is_set() and reference() is not None
        assert old._transcriber is None
        release.set()
        await asyncio.wait_for(active, 2)
        assert await asyncio.wait_for(waiting, 2)
        assert order == ["stop requested", "decode finished", "model released"]
        assert reference() is None and new.session_state is state and len(state.segments) == 1
    finally:
        release.set()
        new.stop_event.set()
        worker.retire()
        await asyncio.gather(active, waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_mkl_failure_retries_same_audio_chunk_without_losing_prior_text(tmp_path, monkeypatch):
    clock = [0.0]
    sleep = asyncio.sleep

    async def advance(seconds):
        clock[0] += seconds
        await sleep(0)

    monkeypatch.setattr("semantic_image_stream.engine.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("semantic_image_stream.engine.asyncio.sleep", advance)
    previous = TranscriptSegment(0, 6, "The earlier passage is retained.")
    state = SessionState(segments=[previous])
    events, attempts = [], []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None, state)
    pipeline.audio_queue = queue.Queue()
    chunk = object()
    pipeline.audio_queue.put_nowait(chunk)

    class Transcriber:
        def transcribe(self, received):
            attempts.append(received)
            if len(attempts) == 1:
                raise RuntimeError("mkl_malloc: failed to allocate memory; private path")
            return [TranscriptSegment(6, 12, "The pending chunk succeeds once memory is available.")]

    pipeline._transcriber = Transcriber()

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "transcript":
            pipeline.stop_event.set()

    pipeline.on_event = receive
    await asyncio.wait_for(pipeline._transcribe(), 2)
    assert attempts == [chunk, chunk] and clock[0] >= 10
    assert state.segments[0] is previous and len(state.segments) == 2
    assert not any(kind == "error" for kind, _ in events)
    assert "private path" not in str(events)


def test_actual_decode_operations_share_the_ranking_pixel_lock(tmp_path, monkeypatch):
    downloader = ImageDownloader(Settings(cache_dir=str(tmp_path)))
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    counts = []

    def decode(data):
        counts.append(data)
        if data == b"first":
            entered.set()
            assert release.wait(2)
        else:
            second_entered.set()
        return data

    monkeypatch.setattr(downloader, "_decode_owned", decode)
    first = threading.Thread(target=downloader._decode, args=(b"first",), daemon=True)
    second = threading.Thread(target=downloader._decode, args=(b"second",), daemon=True)
    try:
        assert IMAGE_DECODE_LOCK is PIXEL_MEMORY_LOCK
        first.start()
        assert entered.wait(1)
        second.start()
        assert not second_entered.wait(.02)
        release.set()
        first.join(1)
        second.join(1)
        assert not first.is_alive() and not second.is_alive() and counts == [b"first", b"second"]
    finally:
        release.set()
        asyncio.run(downloader.close())


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
@pytest.mark.filterwarnings("ignore:Palette images with Transparency expressed in bytes:UserWarning")
def test_low_copy_decode_preserves_pixels_and_strips_source_metadata(tmp_path, monkeypatch, mode):
    monkeypatch.setattr("semantic_image_stream.memory_pressure.available_commit_bytes", lambda: None)
    original = Image.new(mode, (120, 80), (40, 90, 30, 80) if mode == "RGBA" else 1 if mode == "P" else (40, 90, 30))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Public synthetic raster metadata")
    source = io.BytesIO()
    original.save(source, format="PNG", pnginfo=metadata, **({"transparency": 1} if mode == "P" else {}))
    downloader = ImageDownloader(Settings(cache_dir=str(tmp_path), min_width=32, min_height=24))
    try:
        encoded, extension, width, height, phash, generated = downloader._decode(source.getvalue())
        with Image.open(io.BytesIO(source.getvalue())) as expected, Image.open(io.BytesIO(encoded)) as output:
            expected = expected.convert("RGBA" if mode in {"RGBA", "P"} else "RGB")
            assert output.tobytes() == expected.tobytes() and not output.info
            expected.close()
        assert extension == ".png" and (width, height) == (120, 80) and len(phash) == 16 and not generated
    finally:
        original.close()
        asyncio.run(downloader.close())


def test_orientation_is_preserved_without_a_second_transposed_full_raster(tmp_path, monkeypatch):
    monkeypatch.setattr("semantic_image_stream.memory_pressure.available_commit_bytes", lambda: None)
    original = Image.new("RGB", (120, 80), "green")
    exif = original.getexif()
    exif[274] = 6
    source = io.BytesIO()
    original.save(source, format="JPEG", exif=exif)
    downloader = ImageDownloader(Settings(cache_dir=str(tmp_path), min_width=32, min_height=24))
    try:
        encoded, _, width, height, _, _ = downloader._decode(source.getvalue())
        with Image.open(io.BytesIO(source.getvalue())) as reference:
            reference = ImageOps.exif_transpose(reference)
            assert (width, height) == reference.size == (80, 120)
            reference.close()
        with Image.open(io.BytesIO(encoded)) as output:
            assert not output.getexif() and not output.info.get("exif")
    finally:
        original.close()
        asyncio.run(downloader.close())


def test_decoder_memory_failure_has_a_transient_reason_without_private_details(tmp_path, monkeypatch):
    downloader = ImageDownloader(Settings(cache_dir=str(tmp_path)))

    def failed(data):
        raise MemoryError("Private source URL or metadata")

    monkeypatch.setattr(downloader, "_decode_owned", failed)
    try:
        with pytest.raises(DownloadError) as caught:
            downloader._decode(b"synthetic")
        assert caught.value.reason == "Low system memory" and "Private" not in str(caught.value)
    finally:
        asyncio.run(downloader.close())
