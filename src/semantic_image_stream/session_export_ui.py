"""Session-link export presentation, separate from the monitoring/fullscreen UI."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFormLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from .configuration import Settings
from .desktop_ui import MainWindow as MonitoringWindow
from .session_image_export import SessionImageLog, default_export_threshold


class MainWindow(MonitoringWindow):
    """Keep accepted image links independent of the bounded slideshow history."""

    def __init__(self, settings: Settings, config_path: str | Path | None = None):
        # Base construction can dispatch restored context to overridden hooks.
        self.session_images = None
        self._export_error = ""
        super().__init__(settings, config_path)
        self._new_image_log()

    def _build_ui(self) -> None:
        super()._build_ui()
        panel = QWidget()
        column = QVBoxLayout(panel)
        explanation = QLabel(
            "Export unique image links accepted during this open session, including queued images. "
            "Each link keeps its highest relevance score. Scores are not probabilities."
        )
        explanation.setWordWrap(True)
        column.addWidget(explanation)
        form = QFormLayout()
        self.export_threshold = QDoubleSpinBox()
        self.export_threshold.setRange(0.0, 1.0)
        self.export_threshold.setDecimals(2)
        self.export_threshold.setSingleStep(0.05)
        self.export_threshold.setValue(default_export_threshold(self.settings.ranking))
        self.export_threshold.setToolTip(
            "Inclusive cutoff on the recorded score; changing it does not affect playback. "
            "Default: 0.70, or 0.35 for raw CLIP. Different ranking modes use different scales."
        )
        form.addRow("Minimum relevance score", self.export_threshold)
        column.addLayout(form)
        self.export_count_label = QLabel("No image links recorded yet.")
        self.export_count_label.setWordWrap(True)
        column.addWidget(self.export_count_label)
        self.export_csv_button = QPushButton("Export high-relevance CSV…")
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.clicked.connect(self.export_session_csv)
        column.addWidget(self.export_csv_button)
        self.export_status_label = QLabel()
        self.export_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.export_status_label.setWordWrap(True)
        column.addWidget(self.export_status_label)
        privacy = QLabel(
            "Includes source links, credits, licenses, search query and visual subjects; "
            "no audio, transcript, full narrative, manual brief or local paths. "
            "Temporary local records survive Stop/resume but are cleared by New book, "
            "Delete session history, or closing this window. Export before clearing. "
            "Saved CSV files are not deleted."
        )
        privacy.setWordWrap(True)
        privacy.setObjectName("privacy")
        column.addWidget(privacy)
        column.addStretch()
        self.tabs.addTab(panel, "Export")
        self.export_threshold.valueChanged.connect(self._refresh_export_count)

    def _new_image_log(self) -> None:
        if self.session_images is not None:
            self.session_images.close()
        self.session_images = None
        self._export_error = ""
        self.export_status_label.clear()
        try:
            self.session_images = SessionImageLog()
        except (OSError, sqlite3.Error):
            self._export_error = "Session export unavailable: temporary local storage could not be opened."
        self._refresh_export_count()

    def _refresh_export_count(self, *_args) -> None:
        total = matching = 0
        if self.session_images is not None:
            try:
                total = self.session_images.count()
                matching = self.session_images.count(self.export_threshold.value())
            except sqlite3.Error:
                self._export_error = "Session export unavailable: local image records could not be read."
        self.export_count_label.setText(
            f"{matching} of {total} unique links meet score ≥ {self.export_threshold.value():.2f}."
            + (" No qualifying links yet." if not matching else "")
        )
        self.export_csv_button.setEnabled(bool(matching) and not self._closing)
        if self._export_error:
            self.export_status_label.setText(self._export_error)

    @Slot(int, str, object)
    def _handle_event(self, session: int, kind: str, payload) -> None:
        # Match the base UI's session/Stop guards so stale worker events cannot
        # contaminate a new book. Snapshot scores before mutable candidates move on.
        if (kind == "images" and session == self._session and not self._closing
                and not self._context_frozen and self.session_images is not None):
            try:
                self.session_images.record(payload, ranking=self.settings.ranking, topic=self._feedback_topic)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                self._export_error = (
                    "Session export may be incomplete: an image batch could not be recorded. "
                    "Monitoring continues; already recorded links remain exportable."
                )
            self._refresh_export_count()
        super()._handle_event(session, kind, payload)

    def _reset_context(self, book_context: str = "", clear_seen: bool = False) -> None:
        super()._reset_context(book_context, clear_seen)
        self._new_image_log()

    @Slot()
    def export_session_csv(self) -> None:
        if self._closing or self.session_images is None:
            return
        self._refresh_export_count()
        if not self.export_csv_button.isEnabled():
            return
        # A native save dialog handles overwrite confirmation for the final .csv
        # name, rather than silently appending a suffix after that confirmation.
        dialog = QFileDialog(self, "Export high-relevance session image links")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.setDefaultSuffix("csv")
        dialog.selectFile(self.session_images.suggested_filename)
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = Path(selected[0])
        if path.suffix.lower() != ".csv":
            self.export_status_label.setText("Choose a filename ending in .csv, then export again.")
            return
        try:
            target, count = self.session_images.export_csv(path, self.export_threshold.value())
        except (OSError, sqlite3.Error, ValueError, UnicodeError):
            self.export_status_label.setText(
                "Could not save the CSV. Close it in other applications or choose a writable folder. "
                "Recorded links are retained; try exporting again."
            )
            return
        message = f"Exported {count} unique image links to {target}."
        if self._export_error:
            message += "\n" + self._export_error
        self.export_status_label.setText(message)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        # A failed story save may veto closing; do not lose the export in that case.
        if event.isAccepted() and self.session_images is not None:
            self.session_images.close()
            self.session_images = None


def run(settings: Settings, config_path: str | Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Semantic Image Stream")
    app.setOrganizationName("Semantic Image Stream")
    window = MainWindow(settings, config_path)
    window.show()
    return app.exec()
