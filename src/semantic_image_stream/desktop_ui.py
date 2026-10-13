"""Local audio monitoring and source-attributed image presentation in PySide6."""
from __future__ import annotations

from copy import deepcopy
from html import escape
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, QSignalBlocker, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from .configuration import Settings
from .models import ImageCandidate, Topic, TranscriptSegment
from .presets import AUDIOBOOK_PRESETS, apply_audiobook_preset
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
        self._session_state = None
        self._session_book_context = settings.book_context.strip()
        self._transcript_segments: list[TranscriptSegment] = []
        self._context_payload: dict = {}
        self._context_frozen = False
        self._playback_key = ""
        self._explicit_playback_key = False
        self._session = 0
        self._device_request = 0
        self._devices_loading = False
        self._device_selection_error = ""
        self._device_mode = ""
        self._device_preferences = {settings.audio_mode: settings.audio_device}
        self._device_preferred = settings.audio_device
        self._device_choices: set[int] = set()
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
        self.stage = stage
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
        self.fullscreen_button.setToolTip("Show images across the selected display. F11 toggles; Escape returns to controls.")
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
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        self.audiobook_preset = QComboBox()
        self.audiobook_preset.addItem("Choose audiobook preset", None)
        for key, preset in AUDIOBOOK_PRESETS.items():
            self.audiobook_preset.addItem(preset.name, key)
            self.audiobook_preset.setItemData(self.audiobook_preset.count() - 1, preset.description,
                                             Qt.ItemDataRole.ToolTipRole)
        self.apply_preset_button = QPushButton("Apply")
        self.apply_preset_button.setEnabled(False)
        self.apply_preset_button.clicked.connect(self.apply_preset)
        self.audiobook_preset.currentIndexChanged.connect(self._preset_selected)
        preset_layout.addWidget(self.audiobook_preset, 1)
        preset_layout.addWidget(self.apply_preset_button)
        self.form.addRow("Audiobooks", preset_row)
        self.input_controls.extend([self.audiobook_preset, self.apply_preset_button])
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
        self.audio_device.currentIndexChanged.connect(self._device_selected)
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
        self.interpreter = QComboBox()
        for label, value in (("Full session · local model", "ollama"), ("Basic phrase analysis", "contextual"),
                             ("spaCy phrases (optional)", "spacy")):
            self.interpreter.addItem(label, value)
        self.interpreter.setCurrentIndex(self.interpreter.findData(self.settings.semantic_provider))
        self.interpreter.currentIndexChanged.connect(self._update_context_controls)
        self.form.addRow("Interpreter", self.interpreter)
        self.context_model = QLineEdit(self.settings.context_model)
        self.context_model.setPlaceholderText("qwen2.5:7b")
        self.context_model.setToolTip("Name of an installed local Ollama model. Narration stays on this computer.")
        self.form.addRow("Local context model", self.context_model)
        self.book_context = QLineEdit(self.settings.book_context)
        self.book_context.setPlaceholderText("Optional title, setting, or background")
        self.book_context.setToolTip("Optional background for local interpretation. Changes re-read the retained transcript next time monitoring starts. New book clears recorded text.")
        self.form.addRow("Book or context", self.book_context)
        self.input_controls.extend([self.interpreter, self.context_model, self.book_context])
        self.ranking = QComboBox()
        self.ranking.addItem("Full context + image embeddings", "context_clip")
        self.ranking.addItem("Source metadata (basic)", "metadata")
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
        self.transcript.setPlaceholderText("The complete recorded session appears here. Transcription stays local.")
        self.context_panel = QWidget()
        context_layout = QVBoxLayout(self.context_panel)
        context_layout.setContentsMargins(0, 0, 0, 0)
        self.context_status = QLabel("Waiting for recorded speech · context stays local.")
        self.context_status.setWordWrap(True)
        self.context_status.setObjectName("status")
        self.context_text = QTextEdit()
        self.context_text.setReadOnly(True)
        self.context_text.setPlaceholderText("The local interpreter's narrative summary, current scene, and visual subjects appear here.")
        context_layout.addWidget(self.context_status)
        context_layout.addWidget(self.context_text, 1)
        self.image_history = QListWidget()
        self.image_history.itemClicked.connect(self._select_history)
        self.topic_history = QListWidget()
        self.tabs.addTab(self.transcript, "Transcript")
        self._context_tab_index = self.tabs.addTab(self.context_panel, "Context")
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
        self.new_book_button = QPushButton("New book / clear context")
        self.new_book_button.setToolTip("Start a fresh local transcript and interpretation. Image cache remains available.")
        self.new_book_button.clicked.connect(self.new_book)
        sidebar_layout.addWidget(self.new_book_button)
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
        self._update_context_controls()

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
        self.source_label.setVisible(self.attribution.isChecked() and not self._fullscreen)

    def _read_settings(self, require_recording: bool = True) -> Settings:
        settings = deepcopy(self.settings)
        for name, value in {
            "audio_mode": self.audio_mode.currentData(),
            "audio_device": self._device_preferred if self._devices_loading else self.audio_device.currentData(),
            "audio_file": self.audio_file.text().strip(), "whisper_model": self.model.currentText().strip(),
            "whisper_device": self.whisper_device.currentData(), "ranking": self.ranking.currentData(),
            "semantic_provider": self.interpreter.currentData(), "context_model": self.context_model.text().strip(),
            "book_context": self.book_context.text().strip(),
            "providers": [name for name, check in self.provider_boxes.items() if check.isChecked()],
            "slideshow_interval": self.interval.value(), "semantic_sensitivity": self.sensitivity.value(),
            "relevance_threshold": self.threshold.value(), "semantic_interval": self.semantic_interval.value(),
            "search_interval": self.search_interval.value(), "chunk_seconds": self.chunk_interval.value(),
            "transition": self.transition.currentData(), "attribution_visible": self.attribution.isChecked(),
            "display_index": self.display.currentData() or 0,
        }.items():
            setattr(settings, name, value)
        if require_recording and settings.audio_mode == "file" and not Path(settings.audio_file).is_file():
            raise ValueError("Choose an existing audio recording first.")
        return settings.validate()

    def _preset_selected(self, *args) -> None:
        preset = AUDIOBOOK_PRESETS.get(self.audiobook_preset.currentData())
        self.apply_preset_button.setEnabled(bool(preset) and not (self._monitoring or self._stopping))
        self.audiobook_preset.setToolTip(preset.description if preset else "Choose a profile, then select Apply.")

    def apply_preset(self) -> None:
        if self._monitoring or self._stopping or self.audiobook_preset.currentData() is None:
            return
        try:
            key = self.audiobook_preset.currentData()
            self.settings = apply_audiobook_preset(self._read_settings(require_recording=False), key)
            for control, value in ((self.interval, self.settings.slideshow_interval),
                                   (self.sensitivity, self.settings.semantic_sensitivity),
                                   (self.threshold, self.settings.relevance_threshold),
                                   (self.semantic_interval, self.settings.semantic_interval),
                                   (self.search_interval, self.settings.search_interval),
                                   (self.chunk_interval, self.settings.chunk_seconds)):
                with QSignalBlocker(control):
                    control.setValue(value)
            self.ranking.setCurrentIndex(self.ranking.findData(self.settings.ranking))
            self.interpreter.setCurrentIndex(self.interpreter.findData(self.settings.semantic_provider))
            self.context_model.setText(self.settings.context_model)
            self.book_context.setText(self.settings.book_context)
            for provider, check in self.provider_boxes.items():
                check.setChecked(provider in self.settings.providers)
            self._apply_presentation()
            self._device_status(f"{AUDIOBOOK_PRESETS[key].name} preset applied. Select Save settings to keep it.")
        except Exception as exc:
            self._error(f"Could not apply audiobook preset: {exc}")

    def _update_context_controls(self, *args) -> None:
        if hasattr(self, "context_model"):
            self.context_model.setEnabled(not (self._monitoring or self._stopping)
                                          and self.interpreter.currentData() == "ollama")

    def _reset_context(self, book_context: str = "") -> None:
        """Start a new in-memory book without letting old workers update its panels."""
        self._session += 1
        self._session_state = None
        self.pipeline = None
        self._session_book_context = book_context
        self._transcript_segments.clear()
        self._context_payload.clear()
        self._context_frozen = False
        self._playback_key = ""
        self._explicit_playback_key = False
        self.transcript.clear()
        self.context_text.clear()
        self.context_status.setText("Waiting for recorded speech · context stays local.")
        self.context_status.setObjectName("status")
        self.context_status.style().unpolish(self.context_status)
        self.context_status.style().polish(self.context_status)
        self.tabs.setTabText(self._context_tab_index, "Context")
        self.topic_history.clear()
        self.current_topic = ""
        self.topic_label.setText("Waiting for a topic")
        self.slideshow.clear()

    def new_book(self) -> None:
        if self._monitoring or self._stopping:
            return
        self.book_context.clear()
        self.settings.book_context = ""
        self._reset_context()
        self._retrieval_message = "Images: waiting for the new book's spoken context."
        self._retrieval_seen = False
        self._refresh_image_status()
        self._device_status("New book started. Previous transcript and interpretation cleared locally.")

    def _load_devices(self, *args) -> None:
        if self._monitoring or self._stopping:
            return
        mode = self.audio_mode.currentData()
        self._device_request += 1
        token = self._device_request
        self._device_mode = mode
        self._device_selection_error = ""
        if mode == "file":
            self._devices_loading = False
            self._update_controls()
            self._device_status("Choose an audio recording, then start monitoring.")
            return
        preferred = self._device_preferences.get(mode)
        self._device_preferred = preferred
        self._devices_loading = True
        self._device_choices.clear()
        with QSignalBlocker(self.audio_device):
            self.audio_device.clear()
            self.audio_device.addItem("Loading audio devices…", preferred)
        self._update_controls()
        self._device_status("Loading audio devices. Your selected input is preserved.")

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
        self._devices_loading = False
        self._device_choices = {device.index for device in devices}
        self._device_selection_error = ""
        with QSignalBlocker(self.audio_device):
            self.audio_device.clear()
            self.audio_device.addItem("Default device", None)
            for device in devices:
                self.audio_device.addItem(device.name + (" · default" if device.is_default else ""), device.index)
            index = self.audio_device.findData(preferred)
            if preferred is not None and index < 0:
                self.audio_device.addItem(f"Unavailable selected device ({preferred})", preferred)
                index = self.audio_device.count() - 1
                self._device_selection_error = "Selected audio input is unavailable. Refresh devices or choose another input."
            if index >= 0:
                self.audio_device.setCurrentIndex(index)
        if error:
            self._device_selection_error = f"Could not load audio devices: {error}. Select Refresh to try again."
        elif not devices:
            self._device_selection_error = "No compatible audio inputs found. Connect an input and refresh devices."
        self._update_controls()
        if self._device_selection_error:
            self._error(self._device_selection_error)
        elif not self._monitoring and not self._stopping:
            self._device_status("Audio input ready · start monitoring when ready")

    def _device_selected(self, *args) -> None:
        if self._devices_loading or self._monitoring or self._stopping:
            return
        selected = self.audio_device.currentData()
        self._device_preferences[self._device_mode or self.audio_mode.currentData()] = selected
        if self._device_choices and (selected is None or selected in self._device_choices):
            self._device_selection_error = ""
            self._device_status("Audio input ready · start monitoring when ready")
        self._update_controls()

    def _device_status(self, message: str) -> None:
        self._last_error = ""
        self.status_label.setObjectName("status")
        self.status_label.setText(message)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose speech recording", "", "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*)")
        if path:
            self.audio_file.setText(path)

    def start_monitoring(self) -> None:
        if self._monitoring or self._stopping:
            return
        if self.audio_mode.currentData() != "file":
            if self._devices_loading:
                self._device_status("Loading audio devices. Wait for your selected input to be ready.")
                return
            if self._device_selection_error:
                self._error(self._device_selection_error)
                return
        try:
            from .engine import Pipeline, SessionState
            self.settings = self._read_settings()
            changed_background = self.settings.book_context != self._session_book_context
            self._session_book_context = self.settings.book_context
            if self._session_state is None:
                self._session_state = SessionState()
            self._session += 1
            token = self._session

            def callback(kind, value):
                try:
                    self.relay.pipeline_event.emit(token, kind, value)
                except RuntimeError:
                    pass
            self.pipeline = Pipeline(deepcopy(self.settings), callback, session_state=self._session_state)
            if not self._transcript_segments:
                self.current_topic = ""
                self._playback_key = ""
                self._explicit_playback_key = False
                self.topic_label.setText("Waiting for a topic")
                self.slideshow.reset_playback()
            self._context_frozen = False
            self._monitoring = True
            self._last_error = ""
            self._retrieval_message = "Images: waiting for spoken context."
            self._retrieval_seen = False
            self._refresh_image_status()
            self._update_controls()
            self.status_label.setObjectName("status")
            self.status_label.setText("Book/context updated · re-reading the retained transcript locally…" if changed_background
                                      else "Starting local speech recognition…")
            self.pipeline.start()
        except Exception as exc:
            self._monitoring = False
            self._update_controls()
            self._error(str(exc))

    def stop_monitoring(self) -> None:
        if self.pipeline and self._monitoring:
            # Queued context/model callbacks can arrive after the click. Stop
            # retains the accepted interpretation until monitoring resumes.
            self._context_frozen = True
            self._stopping = True
            self._update_controls()
            self.status_label.setText("Stopping monitoring…")
            self.pipeline.stop()

    def _update_controls(self) -> None:
        active = self._monitoring or self._stopping
        device_ready = self.audio_mode.currentData() == "file" or not (self._devices_loading or self._device_selection_error)
        self.start_button.setEnabled(not active and device_ready)
        self.stop_button.setEnabled(self._monitoring and not self._stopping)
        self.clear_cache_button.setEnabled(not active)
        self.clear_history_button.setEnabled(not active)
        self.new_book_button.setEnabled(not active)
        for control in self.input_controls:
            control.setEnabled(not active)
        self._preset_selected()
        self._update_context_controls()
        if not active:
            is_file = self.audio_mode.currentData() == "file"
            self.audio_file.setEnabled(is_file)
            self.choose_audio.setEnabled(is_file)
            self.audio_device.setEnabled(not is_file and not self._devices_loading)
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
            self._context_frozen = True
            self._update_controls()
            self.status_label.setText(self._last_error or "Monitoring stopped · transcript and context retained")
        elif kind == "transcript":
            self._show_transcript(payload)
        elif kind == "context":
            if not self._context_frozen:
                self._show_context(payload)
        elif kind == "topic":
            if self._context_frozen:
                return
            topic: Topic = payload
            if topic.summary or topic.scene or topic.visual_subjects or topic.context_used_segments:
                self._show_context({"phase": "ready", "summary": topic.summary, "scene": topic.scene,
                                    "visual_subjects": topic.visual_subjects,
                                    "segments_used": topic.context_used_segments,
                                    "segments_recorded": len(self._transcript_segments),
                                    "method": self.settings.semantic_provider,
                                    "model": self.settings.context_model if self.settings.semantic_provider == "ollama" else ""})
            if topic.query != self.current_topic:
                self.current_topic = topic.query
                if not self._explicit_playback_key:
                    self._playback_key = topic.query
                    self.slideshow.set_topic(self._playback_key)
                self.topic_label.setText(topic.query)
                self.topic_history.insertItem(0, f"{topic.query}\n{topic.kind.replace('_', ' ').title()}")
                self.topic_history.item(0).setToolTip(topic.context)
                while self.topic_history.count() > 30:
                    self.topic_history.takeItem(self.topic_history.count() - 1)
        elif kind == "images":
            missing = sum(not candidate.path.is_file() for candidate in payload)
            if not self._retrieval_seen:
                self._retrieval_message = f"{len(payload)} relevant images available."
            self._retrieval_seen = True
            self.slideshow.set_queue(payload, self._playback_key)
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

    def _show_transcript(self, segments: list[TranscriptSegment]) -> None:
        """Append a full-session event's new lines without rebuilding an audiobook."""
        count = len(self._transcript_segments)
        extends = len(segments) >= count and segments[:count] == self._transcript_segments
        if count and (len(segments) < count or self._context_frozen and not extends):
            # Full-session events grow monotonically. A delayed empty/short
            # snapshot must not erase already recorded text around Stop.
            return
        scrollbar = self.transcript.verticalScrollBar()
        position = scrollbar.value()
        at_end = position >= scrollbar.maximum() - 10
        if not extends:
            self.transcript.clear()
            count = 0
        lines = []
        for segment in segments[count:]:
            seconds = max(0, int(segment.start))
            lines.append(f"[{seconds // 60:02d}:{seconds % 60:02d}] {segment.text.strip()}")
        if lines:
            cursor = QTextCursor(self.transcript.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(("\n" if count else "") + "\n".join(lines))
        self._transcript_segments = list(segments)
        if at_end:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(position)
        self._show_context({"segments_recorded": len(segments)})

    def _show_context(self, payload: dict) -> None:
        payload = dict(payload)
        if payload.get("phase") in {"updating", "error"}:
            for field in ("summary", "scene", "visual_subjects"):
                if not payload.get(field) and self._context_payload.get(field):
                    payload.pop(field, None)
        if payload.get("playback_key") is not None:
            playback_key = "context:" + str(payload["playback_key"])
            self._explicit_playback_key = True
            if playback_key != self._playback_key:
                self._playback_key = playback_key
                self.slideshow.set_topic(playback_key)
        self._context_payload.update(payload)
        context = self._context_payload
        recorded = max(len(self._transcript_segments), int(context.get("segments_recorded") or 0))
        used = max(0, int(context.get("segments_used") or 0))
        phase = str(context.get("phase") or "waiting")
        if used >= recorded and recorded:
            coverage = f"Using {recorded} recorded segments · full session"
        else:
            coverage = f"Using {used} of {recorded} recorded segments"
            if recorded > used:
                coverage += f" · {recorded - used} awaiting context update"
        message = str(context.get("message") or ("Updating local interpretation…" if phase == "updating" else ""))
        method = str(context.get("method") or self.interpreter.currentText())
        method = {"ollama": "Full session · local model", "contextual": "Basic phrase analysis",
                  "spacy": "spaCy phrase analysis"}.get(method, method)
        model = str(context.get("model") or (self.context_model.text().strip()
                    if self.interpreter.currentData() == "ollama" else ""))
        checked = ("Details checked against recorded passages"
                   if context.get("summary_method") == "model-selected recorded evidence" else "")
        self.context_status.setText("\n".join(part for part in (coverage, message, checked,
                                   f"{method} · {model}" if model else method) if part))
        self.context_status.setObjectName("error" if phase == "error" else "status")
        self.context_status.style().unpolish(self.context_status)
        self.context_status.style().polish(self.context_status)
        self.tabs.setTabText(self._context_tab_index, "Context · error" if phase == "error" else "Context")
        sections = []
        for key, title in (("summary", "Narrative context"), ("scene", "Current scene")):
            if context.get(key):
                sections.append(f"{title}\n{context[key]}")
        subjects = context.get("visual_subjects") or ()
        if subjects:
            sections.append("Visual subjects\n" + "\n".join(str(subject) for subject in subjects))
        self.context_text.setPlainText("\n\n".join(sections))
        if not self._retrieval_seen:
            if phase == "error":
                heading = "Images: local story understanding needs attention."
            elif phase == "updating":
                heading = ("Images: the local model is reading the recorded story." if model
                           else "Images: interpreting the recorded speech.")
            elif phase == "ready":
                heading = ("Images: no visual subject found in the recorded passage yet."
                           if context.get("method") == "ollama" and not subjects
                           else "Images: context ready; waiting for image search.")
            else:
                heading = ("Images: recorded speech is waiting for interpretation." if recorded
                           else "Images: waiting for recorded speech.")
            progress = f"{used} of {recorded} passages processed"
            if model:
                progress = f"{model} · {progress}"
            self._retrieval_message = "\n".join(part for part in (heading, progress, message) if part)[:1200]
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
        context_failed = not self._retrieval_seen and self._context_payload.get("phase") == "error"
        self.image_status_label.setObjectName("error" if self._image_error or context_failed else "imageStatus")
        self.image_status_label.style().unpolish(self.image_status_label)
        self.image_status_label.style().polish(self.image_status_label)

    def _update_queue_label(self, count: int) -> None:
        prefix = "Paused · " if self.slideshow.paused else ""
        noun = "image" if count == 1 else "images"
        if count == 0 and self.slideshow.replay_count >= 2:
            self.queue_label.setText(f"{prefix}Cycling {self.slideshow.replay_count} current-topic images")
        else:
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
            self.slideshow.discard(candidate, self._displayed_candidate)
            self.save_button.setEnabled(self._displayed_candidate is not None and self._displayed_candidate.path.is_file())
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
            self._reset_context(self.book_context.text().strip())
            self.status_label.setText("Stored session history deleted")
        except Exception as exc:
            self._error(f"Could not delete history: {exc}")

    def toggle_fullscreen(self) -> None:
        if self._fullscreen:
            self.leave_fullscreen()
            return
        self._normal_geometry = self.saveGeometry()
        self._normal_rect = self.geometry()
        self._was_maximized = self.isMaximized()
        self._normal_minimum = self.minimumSize()
        self._canvas_minimum = self.canvas.minimumSize()
        outer = self.centralWidget().layout()
        stage_layout = self.stage.layout()
        self._fullscreen_layout = (outer.getContentsMargins(), outer.spacing(), stage_layout.spacing(), self.body.handleWidth())
        chrome = (self.header, self.sidebar, self.topic_label, self.source_label, self.transport,
                  self.status_label, self.image_status_label)
        self._fullscreen_visibility = [(widget, not widget.isHidden()) for widget in chrome]
        self._fullscreen = True
        for widget, _ in self._fullscreen_visibility:
            widget.hide()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        stage_layout.setSpacing(0)
        self.body.setHandleWidth(0)
        self.setMinimumSize(0, 0)
        self.canvas.setMinimumSize(0, 0)
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
        outer = self.centralWidget().layout()
        margins, spacing, stage_spacing, handle_width = self._fullscreen_layout
        outer.setContentsMargins(*margins)
        outer.setSpacing(spacing)
        self.stage.layout().setSpacing(stage_spacing)
        self.body.setHandleWidth(handle_width)
        self.canvas.setMinimumSize(self._canvas_minimum)
        self.setMinimumSize(self._normal_minimum)
        for widget, visible in self._fullscreen_visibility:
            widget.setVisible(visible)
        if self._normal_geometry is not None:
            self.restoreGeometry(self._normal_geometry)
        if self._was_maximized:
            self.showMaximized()
        else:
            self.setGeometry(self._normal_rect)
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
