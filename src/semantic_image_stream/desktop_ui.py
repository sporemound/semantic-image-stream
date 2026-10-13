"""Local audio monitoring and source-attributed image presentation in PySide6."""
from __future__ import annotations

from copy import deepcopy
from html import escape
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from .configuration import Settings
from .models import ImageCandidate, Topic, TranscriptSegment
from .slideshow import ImageCanvas, SlideshowController


STYLE = """
QWidget { background: #101720; color: #e4edf5; font-family: 'Segoe UI'; font-size: 13px; }
QMainWindow { background: #101720; }
QLabel#brand { font-size: 22px; font-weight: 650; color: #edf8ff; }
QLabel#subtitle, QLabel#privacy, QLabel#queue { color: #93a6b8; }
QLabel#topic { font-size: 16px; font-weight: 600; color: #a8d9d3; }
QLabel#status { color: #b4c7d6; }
QLabel#imageStatus { color: #a8d9d3; }
QLabel#error { color: #f8b9a9; }
QLabel#source { background: #0c131c; padding: 12px; }
QFrame#settings { background: #17212d; border: 1px solid #273646; border-radius: 8px; }
QPushButton { background: #243344; border: 1px solid #364b5e; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #30485d; border-color: #76b9bb; }
QPushButton:disabled { color: #718292; background: #192431; border-color: #273646; }
QPushButton#start { color: #072420; background: #86d4c4; border-color: #86d4c4; font-weight: 600; }
QPushButton#start:hover { background: #b0eadb; }
QComboBox, QDoubleSpinBox, QLineEdit { background: #101a26; border: 1px solid #36485b; border-radius: 4px; padding: 5px; }
QComboBox QAbstractItemView { background: #182638; selection-background-color: #305767; }
QTextEdit, QListWidget { background: #0e1722; border: 1px solid #2c3b4b; border-radius: 5px; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #202f40; }
QListWidget::item:selected { background: #294f5b; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: #1b2938; padding: 8px 13px; }
QTabBar::tab:selected { background: #305462; color: #c0f2e5; }
QScrollArea { border: 0; }
QSplitter::handle { background: #233343; height: 5px; width: 5px; }
QCheckBox { spacing: 7px; }
"""


def safe_source_url(value: str) -> QUrl | None:
    """Do not dispatch provider-controlled file, executable, or custom scheme links."""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password or any(ord(c) < 32 for c in value):
            return None
        url = QUrl(value)
        return url if url.isValid() else None
    except (ValueError, TypeError):
        return None


class EventRelay(QObject):
    pipeline_event = Signal(int, str, object)
    devices = Signal(int, object, str)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, config_path: str | Path | None = None):
        super().__init__()
        self.settings = deepcopy(settings)
        self.config_path = Path(config_path) if config_path else None
        self.pipeline = None
        self._session = 0
        self._device_request = 0
        self._closing = False
        self._monitoring = False
        self._stopping = False
        self._fullscreen = False
        self._normal_geometry = None
        self._last_error = ""
        self._image_error = ""
        self._retrieval_message = "Images: waiting for a topic."
        self._retrieval_seen = False
        self._displayed_candidate: ImageCandidate | None = None
        self.current_topic = ""
        self.setWindowTitle("Semantic Image Stream")
        self.resize(1360, 900)
        self.setMinimumSize(980, 690)
        self.setStyleSheet(STYLE)
        self.relay = EventRelay(self)
        self.relay.pipeline_event.connect(self._handle_event)
        self.relay.devices.connect(self._devices_ready)
        self.slideshow = SlideshowController(settings.slideshow_interval, self)
        self._build_ui()
        self.slideshow.current_changed.connect(self._show_image)
        self.slideshow.history_changed.connect(self._show_history)
        self.slideshow.queue_changed.connect(self._update_queue_label)
        self.canvas.load_failed.connect(self._image_failed)
        self._apply_presentation()
        self._load_devices()
        for key, handler in (("F11", self.toggle_fullscreen), ("Escape", self.leave_fullscreen),
                             ("Space", self.toggle_pause), ("Right", self.next_image)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(handler)

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        self.header = QWidget()
        header = QHBoxLayout(self.header)
        header.setContentsMargins(0, 0, 0, 0)
        title_column = QVBoxLayout()
        title = QLabel("Semantic Image Stream")
        title.setObjectName("brand")
        subtitle = QLabel("Spoken ideas, existing images")
        subtitle.setObjectName("subtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header.addLayout(title_column)
        header.addStretch()
        self.start_button = QPushButton("Start monitoring")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_monitoring)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)
        layout.addWidget(self.header)
        self.body = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.body, 1)
        stage = QWidget()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(8)
        self.topic_label = QLabel("Waiting for a topic")
        self.topic_label.setObjectName("topic")
        self.topic_label.setWordWrap(True)
        stage_layout.addWidget(self.topic_label)
        self.canvas = ImageCanvas()
        stage_layout.addWidget(self.canvas, 1)
        self.source_label = QLabel("Source attribution appears with each image.")
        self.source_label.setObjectName("source")
        self.source_label.setWordWrap(True)
        self.source_label.setTextFormat(Qt.TextFormat.RichText)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.source_label.setOpenExternalLinks(False)
        self.source_label.linkActivated.connect(self._open_source)
        stage_layout.addWidget(self.source_label)
        self.transport = QWidget()
        transport = QHBoxLayout(self.transport)
        transport.setContentsMargins(0, 0, 0, 0)
        self.pause_button = QPushButton("Pause images")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.next_image)
        self.save_button = QPushButton("Save image + metadata")
        self.save_button.setToolTip("Save the original image and a JSON file containing its source and attribution")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current)
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.queue_label = QLabel("0 images ready")
        self.queue_label.setObjectName("queue")
        for widget in (self.pause_button, self.next_button, self.save_button):
            transport.addWidget(widget)
        transport.addStretch()
        transport.addWidget(self.queue_label)
        transport.addWidget(self.fullscreen_button)
        stage_layout.addWidget(self.transport)
        self.body.addWidget(stage)
        self.sidebar = QWidget()
        self.sidebar.setMinimumWidth(360)
        self.sidebar.setMaximumWidth(420)
        sidebar_content = self.sidebar
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(6, 0, 0, 0)
        settings_frame = QFrame()
        settings_frame.setObjectName("settings")
        self.form = QFormLayout(settings_frame)
        self.form.setContentsMargins(12, 12, 12, 12)
        self.form.setSpacing(8)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.input_controls: list[QWidget] = []
        self.audio_mode = QComboBox()
        for title, value in (("System audio · WASAPI loopback", "loopback"),
                             ("Microphone", "microphone"), ("Audio recording", "file")):
            self.audio_mode.addItem(title, value)
        self.audio_mode.setCurrentIndex(self.audio_mode.findData(self.settings.audio_mode))
        self.form.addRow("Listen to", self.audio_mode)
        self.input_controls.append(self.audio_mode)
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_device = QComboBox()
        self.audio_device.addItem("Default device", None)
        self.refresh_devices = QPushButton("↻")
        self.refresh_devices.setToolTip("Refresh connected audio devices")
        self.refresh_devices.setFixedWidth(32)
        self.refresh_devices.clicked.connect(self._load_devices)
        device_layout.addWidget(self.audio_device, 1)
        device_layout.addWidget(self.refresh_devices)
        self.form.addRow("Device", device_row)
        self.input_controls.extend([self.audio_device, self.refresh_devices])
        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_file = QLineEdit(self.settings.audio_file)
        self.audio_file.setPlaceholderText("Choose a speech recording")
        self.choose_audio = QPushButton("…")
        self.choose_audio.setFixedWidth(32)
        self.choose_audio.clicked.connect(self._choose_audio)
        file_layout.addWidget(self.audio_file, 1)
        file_layout.addWidget(self.choose_audio)
        self.form.addRow("Recording", file_row)
        self.input_controls.extend([self.audio_file, self.choose_audio])
        self.audio_mode.currentIndexChanged.connect(self._load_devices)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(["tiny", "base", "small", "medium", "large-v3", "turbo"])
        self.model.setCurrentText(self.settings.whisper_model)
        self.form.addRow("Speech model", self.model)
        self.input_controls.append(self.model)
        self.whisper_device = QComboBox()
        for label, value in (("Automatic GPU / CPU", "auto"), ("CPU", "cpu"), ("NVIDIA GPU", "cuda")):
            self.whisper_device.addItem(label, value)
        self.whisper_device.setCurrentIndex(self.whisper_device.findData(self.settings.whisper_device))
        self.form.addRow("Speech compute", self.whisper_device)
        self.input_controls.append(self.whisper_device)
        self.provider_boxes = {}
        providers = QWidget()
        provider_layout = QVBoxLayout(providers)
        provider_layout.setContentsMargins(0, 0, 0, 0)
        for name, label in (("wikimedia", "Wikimedia Commons"), ("duckduckgo", "DuckDuckGo Images"),
                            ("openverse", "Openverse"), ("archive", "Internet Archive")):
            check = QCheckBox(label)
            check.setChecked(name in self.settings.providers)
            self.provider_boxes[name] = check
            provider_layout.addWidget(check)
            self.input_controls.append(check)
        self.form.addRow("Image sources", providers)
        self.ranking = QComboBox()
        self.ranking.addItem("Context + source metadata", "metadata")
        self.ranking.addItem("Local CLIP embeddings", "clip")
        self.ranking.setCurrentIndex(self.ranking.findData(self.settings.ranking))
        self.form.addRow("Relevance scoring", self.ranking)
        self.input_controls.append(self.ranking)
        self.interval = self._spin(1, 120, self.settings.slideshow_interval, 1, " s")
        self.form.addRow("Image interval", self.interval)
        self.sensitivity = self._spin(0, 1, self.settings.semantic_sensitivity, 0.05)
        self.form.addRow("Topic sensitivity", self.sensitivity)
        self.threshold = self._spin(0, 1, self.settings.relevance_threshold, 0.05)
        self.form.addRow("Relevance minimum", self.threshold)
        self.input_controls.extend([self.sensitivity, self.threshold])
        self.semantic_interval = self._spin(1, 60, self.settings.semantic_interval, 1, " s")
        self.search_interval = self._spin(1, 120, self.settings.search_interval, 1, " s")
        self.chunk_interval = self._spin(1, 30, self.settings.chunk_seconds, 1, " s")
        self.form.addRow("Interpret every", self.semantic_interval)
        self.form.addRow("Search cooldown", self.search_interval)
        self.form.addRow("Audio chunk", self.chunk_interval)
        self.input_controls.extend([self.semantic_interval, self.search_interval, self.chunk_interval])
        self.transition = QComboBox()
        for label, value in (("Crossfade", "crossfade"), ("Dissolve", "dissolve"),
                             ("Slow zoom", "zoom"), ("Instant cut", "cut")):
            self.transition.addItem(label, value)
        self.transition.setCurrentIndex(self.transition.findData(self.settings.transition))
        self.form.addRow("Transition", self.transition)
        self.attribution = QCheckBox("Show image attribution")
        self.attribution.setChecked(self.settings.attribution_visible)
        self.form.addRow(self.attribution)
        self.display = QComboBox()
        for index, screen in enumerate(QApplication.screens()):
            self.display.addItem(f"{index + 1} · {screen.name()} ({screen.size().width()} × {screen.size().height()})", index)
        self.display.setCurrentIndex(min(self.settings.display_index, max(0, self.display.count() - 1)))
        self.form.addRow("Fullscreen display", self.display)
        for signal in (self.interval.valueChanged, self.transition.currentIndexChanged, self.attribution.toggled):
            signal.connect(self._apply_presentation)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setMinimumHeight(180)
        settings_scroll.setWidget(settings_frame)
        sidebar_layout.addWidget(settings_scroll, 3)
        self.tabs = QTabWidget()
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.document().setMaximumBlockCount(160)
        self.transcript.setPlaceholderText("Timestamped speech appears here. Audio is transcribed locally.")
        self.image_history = QListWidget()
        self.image_history.itemClicked.connect(self._select_history)
        self.topic_history = QListWidget()
        self.tabs.addTab(self.transcript, "Transcript")
        self.tabs.addTab(self.image_history, "Images")
        self.tabs.addTab(self.topic_history, "Topics")
        self.tabs.setMinimumHeight(145)
        sidebar_layout.addWidget(self.tabs, 1)
        utility_row = QHBoxLayout()
        self.save_settings_button = QPushButton("Save settings")
        self.save_settings_button.clicked.connect(self.save_settings)
        self.clear_cache_button = QPushButton("Clear cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        utility_row.addWidget(self.save_settings_button)
        utility_row.addWidget(self.clear_cache_button)
        sidebar_layout.addLayout(utility_row)
        self.clear_history_button = QPushButton("Delete session history")
        self.clear_history_button.clicked.connect(self.clear_history)
        sidebar_layout.addWidget(self.clear_history_button)
        privacy = QLabel("Audio stays on this computer. Search queries are sent to the selected image sources. Check source licenses before reuse.")
        privacy.setObjectName("privacy")
        privacy.setWordWrap(True)
        sidebar_layout.addWidget(privacy)
        for combo in sidebar_content.findChildren(QComboBox):
            combo.setMinimumContentsLength(10)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body.addWidget(self.sidebar)
        self.body.setStretchFactor(0, 1)
        self.body.setSizes([950, 365])
        self.image_status_label = QLabel(self._retrieval_message)
        self.image_status_label.setObjectName("imageStatus")
        self.image_status_label.setWordWrap(True)
        self.image_status_label.setMaximumHeight(72)
        self.image_status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.image_status_label)
        self.status_label = QLabel("Ready · choose an input and start monitoring")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    @staticmethod
    def _spin(minimum, maximum, value, step, suffix="") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setDecimals(0 if suffix else 2)
        return widget

    def _apply_presentation(self, *args) -> None:
        self.slideshow.set_interval(self.interval.value())
        self.canvas.interval = self.interval.value()
        self.canvas.transition = self.transition.currentData()
        self.canvas.transition_seconds = self.settings.transition_seconds
        self.source_label.setVisible(self.attribution.isChecked())

    def _read_settings(self) -> Settings:
        settings = deepcopy(self.settings)
        for name, value in {
            "audio_mode": self.audio_mode.currentData(), "audio_device": self.audio_device.currentData(),
            "audio_file": self.audio_file.text().strip(), "whisper_model": self.model.currentText().strip(),
            "whisper_device": self.whisper_device.currentData(), "ranking": self.ranking.currentData(),
            "providers": [name for name, check in self.provider_boxes.items() if check.isChecked()],
            "slideshow_interval": self.interval.value(), "semantic_sensitivity": self.sensitivity.value(),
            "relevance_threshold": self.threshold.value(), "semantic_interval": self.semantic_interval.value(),
            "search_interval": self.search_interval.value(), "chunk_seconds": self.chunk_interval.value(),
            "transition": self.transition.currentData(), "attribution_visible": self.attribution.isChecked(),
            "display_index": self.display.currentData() or 0,
        }.items():
            setattr(settings, name, value)
        if settings.audio_mode == "file" and not Path(settings.audio_file).is_file():
            raise ValueError("Choose an existing audio recording first.")
        return settings.validate()

    def _load_devices(self, *args) -> None:
        mode = self.audio_mode.currentData()
        self.audio_file.setEnabled(mode == "file" and not self._monitoring)
        self.choose_audio.setEnabled(mode == "file" and not self._monitoring)
        self.audio_device.setEnabled(mode != "file" and not self._monitoring)
        self.refresh_devices.setEnabled(mode != "file" and not self._monitoring)
        self._device_request += 1
        token = self._device_request
        if mode == "file":
            return
        preferred = self.audio_device.currentData()
        if preferred is None:
            preferred = self.settings.audio_device
        self.audio_device.clear()
        self.audio_device.addItem("Default device", None)

        def work():
            try:
                from .audio_capture import list_devices
                devices, error = list_devices(mode), ""
            except Exception as exc:
                devices, error = [], str(exc)
            try:
                self.relay.devices.emit(token, (devices, preferred), error)
            except RuntimeError:
                pass  # The user may close the window while discovery completes.
        threading.Thread(target=work, daemon=True, name="audio-device-discovery").start()

    def _devices_ready(self, token: int, payload, error: str) -> None:
        if token != self._device_request or self._closing:
            return
        devices, preferred = payload
        self.audio_device.clear()
        self.audio_device.addItem("Default device", None)
        for device in devices:
            self.audio_device.addItem(device.name + (" · default" if device.is_default else ""), device.index)
        index = self.audio_device.findData(preferred)
        if index >= 0:
            self.audio_device.setCurrentIndex(index)
        elif preferred is not None:
            self.status_label.setText("Selected device is unavailable; using the default device.")
        if error:
            self.status_label.setText(f"Audio device discovery: {error}")

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose speech recording", "", "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*)")
        if path:
            self.audio_file.setText(path)

    def start_monitoring(self) -> None:
        if self._monitoring or self._stopping:
            return
        try:
            from .engine import Pipeline
            self.settings = self._read_settings()
            self._session += 1
            token = self._session

            def callback(kind, value):
                try:
                    self.relay.pipeline_event.emit(token, kind, value)
                except RuntimeError:
                    pass
            self.pipeline = Pipeline(deepcopy(self.settings), callback)
            self._monitoring = True
            self._last_error = ""
            self._retrieval_message = "Images: waiting for spoken context."
            self._retrieval_seen = False
            self._refresh_image_status()
            self._update_controls()
            self.status_label.setObjectName("status")
            self.status_label.setText("Starting local speech recognition…")
            self.pipeline.start()
        except Exception as exc:
            self._monitoring = False
            self._update_controls()
            self._error(str(exc))

    def stop_monitoring(self) -> None:
        if self.pipeline and self._monitoring:
            self._stopping = True
            self._update_controls()
            self.status_label.setText("Stopping monitoring…")
            self.pipeline.stop()

    def _update_controls(self) -> None:
        active = self._monitoring or self._stopping
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(self._monitoring and not self._stopping)
        self.clear_cache_button.setEnabled(not active)
        self.clear_history_button.setEnabled(not active)
        for control in self.input_controls:
            control.setEnabled(not active)
        if not active:
            is_file = self.audio_mode.currentData() == "file"
            self.audio_file.setEnabled(is_file)
            self.choose_audio.setEnabled(is_file)
            self.audio_device.setEnabled(not is_file)
            self.refresh_devices.setEnabled(not is_file)

    def _handle_event(self, session: int, kind: str, payload) -> None:
        if session != self._session or self._closing:
            return
        if kind == "status":
            self._last_error = ""
            self.status_label.setText(str(payload))
        elif kind == "error":
            self._error(str(payload))
        elif kind == "stopped":
            self._monitoring = self._stopping = False
            self._update_controls()
            self.status_label.setText(self._last_error or "Monitoring stopped · images remain available")
        elif kind == "transcript":
            segments: list[TranscriptSegment] = payload
            lines = []
            for segment in segments[-160:]:
                seconds = max(0, int(segment.start))
                lines.append(f"[{seconds // 60:02d}:{seconds % 60:02d}] {segment.text.strip()}")
            scrollbar = self.transcript.verticalScrollBar()
            at_end = scrollbar.value() >= scrollbar.maximum() - 10
            self.transcript.setPlainText("\n".join(lines))
            if at_end:
                scrollbar.setValue(scrollbar.maximum())
        elif kind == "topic":
            topic: Topic = payload
            if topic.query != self.current_topic:
                self.current_topic = topic.query
                self.topic_label.setText(topic.query)
                self.topic_history.insertItem(0, f"{topic.query}\n{topic.kind.replace('_', ' ').title()}")
                self.topic_history.item(0).setToolTip(topic.context)
                while self.topic_history.count() > 30:
                    self.topic_history.takeItem(self.topic_history.count() - 1)
        elif kind == "images":
            missing = sum(not candidate.path.is_file() for candidate in payload)
            if not self._retrieval_seen:
                self._retrieval_message = f"{len(payload)} relevant images available."
            self.slideshow.set_queue(payload, self.current_topic)
            # A rejected first file must not delay a later usable image by an
            # entire slideshow interval, including in progressive batches.
            while self.canvas.current_pixmap.isNull() and self.slideshow.pending and not self.slideshow.paused:
                self.slideshow.advance(manual=True)
            if payload and missing == len(payload):
                self._image_failed("The retrieved image files are unavailable. Waiting for the next search.")
        elif kind == "retrieval":
            self._retrieval_seen = True
            self._retrieval_message = self._format_retrieval(payload)
            self._refresh_image_status()

    @staticmethod
    def _format_retrieval(payload: dict) -> str:
        message = str(payload.get("message") or "Looking for relevant images.").strip()
        counts = []
        for key, label in (("found", "Found"), ("downloaded", "Loaded"), ("accepted", "Ready"), ("rejected", "Skipped")):
            if key in payload:
                counts.append(f"{label} {payload[key]}")
        detail = " · ".join(counts)
        failures = payload.get("failures") or {}
        reasons = "; ".join(f"{reason}: {count}" for reason, count in list(failures.items())[:3])
        if payload.get("accepted") == 0 and payload.get("best_score") is not None and payload.get("threshold") is not None:
            reasons = (reasons + "; " if reasons else "") + (
                f"Best relevance {payload['best_score']:.2f}; minimum {payload['threshold']:.2f}"
            )
        return "\n".join(part for part in (message, detail, reasons) if part)[:1200]

    def _refresh_image_status(self) -> None:
        message = self._image_error or self._retrieval_message
        if self.slideshow.paused:
            instruction = "Images are paused. Select Resume images or Next to show an image."
            message = f"{message}\n{instruction}"
        self.image_status_label.setText(message)
        self.image_status_label.setToolTip(message)
        self.canvas.set_empty_message(message)
        self.image_status_label.setObjectName("error" if self._image_error else "imageStatus")
        self.image_status_label.style().unpolish(self.image_status_label)
        self.image_status_label.style().polish(self.image_status_label)

    def _update_queue_label(self, count: int) -> None:
        prefix = "Paused · " if self.slideshow.paused else ""
        noun = "image" if count == 1 else "images"
        self.queue_label.setText(f"{prefix}{count} {noun} ready")
        self._refresh_image_status()

    def _image_failed(self, message: str) -> None:
        self._image_error = message
        self._refresh_image_status()

    def _error(self, message: str) -> None:
        self._last_error = message
        self.status_label.setObjectName("error")
        self.status_label.setText(message)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _show_image(self, candidate: ImageCandidate | None) -> None:
        displayed = self.canvas.set_image(candidate)
        if candidate is None:
            self._displayed_candidate = None
            self._image_error = ""
            self.save_button.setEnabled(False)
            self.source_label.setText("Source attribution appears with each image.")
            self._refresh_image_status()
            return
        if not displayed:
            self.save_button.setEnabled(False)
            return
        self._displayed_candidate = candidate
        self._image_error = ""
        self.save_button.setEnabled(True)
        self._refresh_image_status()
        if self.pipeline is not None:
            try:
                self.pipeline.record_displayed(candidate)
            except Exception as exc:
                self._error(f"Could not record image history: {exc}")
        title = escape(candidate.title or "Untitled image")
        url = safe_source_url(candidate.source_url)
        link = f'<a style="color:#90ded0" href="{escape(candidate.source_url, quote=True)}">{title} ↗</a>' if url else title
        self.source_label.setText(
            f"{link}<br>{escape(candidate.creator)} · {escape(candidate.provider)} · {escape(candidate.license)}"
            f"<br><span style='color:#93a6b8'>{escape(candidate.authenticity)} · relevance {candidate.score:.2f}</span>"
        )
        self.source_label.setToolTip(candidate.source_url)

    def _open_source(self, value: str) -> None:
        url = safe_source_url(value)
        if url is not None:
            QDesktopServices.openUrl(url)

    def _show_history(self, images: list[ImageCandidate]) -> None:
        self.image_history.clear()
        for index in range(len(images) - 1, -1, -1):
            candidate = images[index]
            item = QListWidgetItem(f"{candidate.title or 'Untitled image'}\n{candidate.provider} · relevance {candidate.score:.2f}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(candidate.source_url)
            self.image_history.addItem(item)

    def _select_history(self, item: QListWidgetItem) -> None:
        self.slideshow.select_history(item.data(Qt.ItemDataRole.UserRole))

    def toggle_pause(self) -> None:
        self.slideshow.set_paused(not self.slideshow.paused)
        self.pause_button.setText("Resume images" if self.slideshow.paused else "Pause images")

    def next_image(self) -> None:
        self.slideshow.advance(manual=True)

    def save_current(self) -> None:
        candidate = self._displayed_candidate
        if candidate is None or not self.save_button.isEnabled():
            return
        suffix = candidate.path.suffix or ".jpg"
        destination, _ = QFileDialog.getSaveFileName(self, "Save image with attribution JSON", f"image{suffix}", "Images (*.*)")
        if destination:
            try:
                from .engine import export_image
                export_image(candidate, Path(destination))
                self.status_label.setText(f"Image and attribution saved to {destination}")
            except Exception as exc:
                self._error(f"Could not save image: {exc}")

    def save_settings(self) -> None:
        try:
            settings = self._read_settings()
            destination = self.config_path
            if destination is None:
                path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "config.toml", "TOML configuration (*.toml)")
                if not path:
                    return
                destination = Path(path)
            settings.save(destination)
            self.config_path = destination
            if not self._monitoring:
                self.settings = settings
            self.status_label.setText(f"Settings saved to {destination}")
        except Exception as exc:
            self._error(f"Could not save settings: {exc}")

    def _maintenance_pipeline(self):
        if self._monitoring or self._stopping:
            raise RuntimeError("Stop monitoring before deleting stored data.")
        if self.pipeline is None:
            from .engine import Pipeline
            self.pipeline = Pipeline(deepcopy(self.settings), lambda *args: None)
        return self.pipeline

    def clear_cache(self) -> None:
        try:
            self._maintenance_pipeline().clear_cache()
            self.slideshow.clear()
            self.status_label.setText("Cached images and search results deleted")
        except Exception as exc:
            self._error(f"Could not clear cache: {exc}")

    def clear_history(self) -> None:
        try:
            self._maintenance_pipeline().clear_history()
            self.slideshow.clear_history()
            self.topic_history.clear()
            self.transcript.clear()
            self.current_topic = ""
            self.topic_label.setText("Waiting for a topic")
            self.status_label.setText("Stored session history deleted")
        except Exception as exc:
            self._error(f"Could not delete history: {exc}")

    def toggle_fullscreen(self) -> None:
        if self._fullscreen:
            self.leave_fullscreen()
            return
        self._normal_geometry = self.saveGeometry()
        self._fullscreen = True
        self.header.hide()
        self.sidebar.hide()
        self.status_label.hide()
        self.fullscreen_button.setText("Exit fullscreen")
        screens = QApplication.screens()
        index = self.display.currentData() or 0
        if 0 <= index < len(screens):
            self.showNormal()
            self.windowHandle().setScreen(screens[index])
            self.setGeometry(screens[index].geometry())
        self.showFullScreen()

    def leave_fullscreen(self) -> None:
        if not self._fullscreen:
            return
        self._fullscreen = False
        self.showNormal()
        if self._normal_geometry is not None:
            self.restoreGeometry(self._normal_geometry)
        self.header.show()
        self.sidebar.show()
        self.status_label.show()
        self.fullscreen_button.setText("Fullscreen")

    def closeEvent(self, event) -> None:
        self._closing = True
        self._device_request += 1
        self.slideshow.timer.stop()
        self.canvas.animation_timer.stop()
        if self.pipeline:
            self.pipeline.stop()
        event.accept()


def run(settings: Settings, config_path: str | Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Semantic Image Stream")
    app.setOrganizationName("Semantic Image Stream")
    window = MainWindow(settings, config_path)
    window.show()
    return app.exec()
