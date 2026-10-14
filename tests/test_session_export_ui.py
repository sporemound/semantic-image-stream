"""Offscreen integration tests; no audio device, network or model is started."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QCloseEvent, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from semantic_image_stream.configuration import Settings  # noqa: E402
from semantic_image_stream.models import ImageCandidate, Topic  # noqa: E402
from semantic_image_stream import session_export_ui as ui  # noqa: E402


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui.MonitoringWindow, "_load_devices", lambda *_: None)
    monkeypatch.setattr(ui.MonitoringWindow, "_restore_displayed_ledger", lambda *_: None)
    monkeypatch.setattr(ui.MonitoringWindow, "_refresh_training_status", lambda *_: None)
    settings = Settings()
    settings.cache_dir = tmp_path / "cache"
    settings.retain_context = False
    settings.audio_mode = "file"
    value = ui.MainWindow(settings)
    yield value
    value._monitoring = value._stopping = False
    value.close()
    if value.session_images is not None:
        value.session_images.close()
    value.deleteLater()
    app.processEvents()


def candidate(tmp_path, score=.85, name="one"):
    path = tmp_path / f"{name}.png"
    image = QImage(32, 32, QImage.Format.Format_RGB32)
    image.fill(0xFF808080)
    assert image.save(str(path))
    return ImageCandidate(image_url=f"https://images.example.org/{name}.png",
                          source_url=f"https://example.org/{name}", title="Forest",
                          provider="deviantart", creator="Artist", score=score,
                          width=32, height=32, local_path=str(path))


def emit_images(window, *images):
    window._handle_event(window._session, "images", list(images))


def choose_path(monkeypatch, path, accepted=True):
    class Dialog(QFileDialog):
        def exec(self):
            assert self.defaultSuffix() == "csv"
            assert self.acceptMode() == QFileDialog.AcceptMode.AcceptSave
            return 1 if accepted else 0

        def selectedFiles(self):
            return [str(path)]

    monkeypatch.setattr(ui, "QFileDialog", Dialog)


def test_export_panel_counts_accepted_batches_and_refilters(window, tmp_path):
    assert "Export" in [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert not window.export_csv_button.isEnabled()
    window._feedback_topic = Topic("forest", "private spoken context", visual_subjects=("trees",))
    emit_images(window, candidate(tmp_path, .85), candidate(tmp_path, .6, "two"))
    assert window.session_images.count() == 2
    window.export_threshold.setValue(.8)
    assert window.export_count_label.text().startswith("1 of 2")
    assert window.export_csv_button.isEnabled()
    window.export_threshold.setValue(.9)
    assert not window.export_csv_button.isEnabled()
    assert window.session_images.count() == 2  # Refiltering never deletes recorded links.


def test_stale_frozen_and_closed_events_do_not_leak_into_export(window, tmp_path):
    image = candidate(tmp_path)
    window._handle_event(window._session - 1, "images", [image])
    window._context_frozen = True
    emit_images(window, image)
    window._context_frozen = False
    window._closing = True
    emit_images(window, image)
    window._closing = False
    assert window.session_images.count() == 0


def test_stop_resume_retains_links_but_new_book_clears_them(window, tmp_path, monkeypatch):
    emit_images(window, candidate(tmp_path))
    original = window.session_images.session_id
    window._handle_event(window._session, "stopped", None)
    assert window.session_images.count() == 1
    window._session += 1
    window._context_frozen = False
    emit_images(window, candidate(tmp_path, name="two"))
    assert window.session_images.count() == 2
    assert window.session_images.session_id == original
    monkeypatch.setattr(window, "_clear_saved_story", lambda: None)
    window.new_book()
    assert window.session_images.count() == 0
    assert window.session_images.session_id != original
    assert not window.export_csv_button.isEnabled()


def test_delete_history_clears_export_but_cache_clear_does_not(window, tmp_path, monkeypatch):
    emit_images(window, candidate(tmp_path))
    monkeypatch.setattr(window, "_maintenance_pipeline", lambda: SimpleNamespace(
        clear_cache=lambda: None, clear_history=lambda: None))
    monkeypatch.setattr(window, "_clear_saved_story", lambda: None)
    window.clear_cache()
    assert window.session_images.count() == 1
    window.clear_history()
    assert window.session_images.count() == 0


def test_csv_dialog_save_and_cancel_preserve_session(window, tmp_path, monkeypatch):
    emit_images(window, candidate(tmp_path))
    path = tmp_path / "my-links.csv"
    choose_path(monkeypatch, path, accepted=False)
    window.export_session_csv()
    assert not path.exists()
    choose_path(monkeypatch, path)
    window.export_session_csv()
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "Exported 1 unique image links" in window.export_status_label.text()
    assert window.session_images.count() == 1


def test_csv_write_failure_is_visible_and_retryable(window, tmp_path, monkeypatch):
    emit_images(window, candidate(tmp_path))
    choose_path(monkeypatch, tmp_path / "links.csv")

    def fail(*_args):
        raise PermissionError("locked")

    monkeypatch.setattr(window.session_images, "export_csv", fail)
    window.export_session_csv()
    assert "Could not save the CSV" in window.export_status_label.text()
    assert window.export_csv_button.isEnabled()
    assert window.session_images.count() == 1


def test_record_failure_does_not_interrupt_image_event(window, tmp_path, monkeypatch):
    image = candidate(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(window.session_images, "record", fail)
    emit_images(window, image)
    assert "may be incomplete" in window.export_status_label.text()
    assert window.slideshow.current is image


def test_rejected_close_retains_log_and_accepted_close_releases_it(window, tmp_path, monkeypatch):
    emit_images(window, candidate(tmp_path))
    with monkeypatch.context() as context:
        context.setattr(ui.MonitoringWindow, "closeEvent", lambda self, event: event.ignore())
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        assert window.session_images.count() == 1
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert window.session_images is None


def test_cli_launches_export_enabled_window(monkeypatch):
    import sys
    from semantic_image_stream import __main__ as cli
    calls = []
    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(cli.Settings, "load", lambda *_: Settings())
    monkeypatch.setitem(sys.modules, "semantic_image_stream.session_export_ui", SimpleNamespace(
        run=lambda settings, config: calls.append((settings.whisper_device, config)) or 7))
    assert cli.main(["--cpu"]) == 7
    assert calls == [("cpu", None)]
