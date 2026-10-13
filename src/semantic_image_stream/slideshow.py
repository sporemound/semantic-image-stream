"""Bounded slideshow playback and genuine Qt image transitions."""
from __future__ import annotations

from collections import deque
import random
import time

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImageReader, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QWidget

from .models import ImageCandidate


class SlideshowController(QObject):
    current_changed = Signal(object)
    history_changed = Signal(object)
    queue_changed = Signal(int)

    def __init__(self, interval: float = 9, parent: QObject | None = None):
        super().__init__(parent)
        self.current: ImageCandidate | None = None
        self.history: deque[ImageCandidate] = deque(maxlen=60)
        self.pending: deque[ImageCandidate] = deque(maxlen=60)
        self._seen: set[str] = set()
        self._generation = ""
        self.paused = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.set_interval(interval)
        self.timer.start()

    @staticmethod
    def identity(candidate: ImageCandidate) -> str:
        return candidate.image_url or candidate.local_path

    def set_interval(self, seconds: float) -> None:
        self.timer.setInterval(max(1000, int(seconds * 1000)))

    def set_queue(self, images: list[ImageCandidate], generation: str = "") -> None:
        """Replace future images; preserve the current frame during topic changes."""
        if generation != self._generation:
            self._seen.clear()
            self._generation = generation
        unique: dict[str, ImageCandidate] = {}
        current_id = self.identity(self.current) if self.current else None
        for candidate in images:
            identity = self.identity(candidate)
            if identity != current_id and identity not in self._seen and candidate.path.is_file():
                unique.setdefault(identity, candidate)
            if len(unique) == 60:
                break
        self.pending = deque(unique.values(), maxlen=60)
        self.queue_changed.emit(len(self.pending))
        if self.current is None and not self.paused:
            self.advance()

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.timer.stop()
        else:
            self.timer.start()
            if self.current is None:
                self.advance()
        self.queue_changed.emit(len(self.pending))

    def advance(self, manual: bool = False) -> None:
        if self.paused and not manual:
            return
        while self.pending:
            candidate = self.pending.popleft()
            if not candidate.path.is_file():
                continue
            self.current = candidate
            self._seen.add(self.identity(candidate))
            self.history.append(candidate)
            self.current_changed.emit(candidate)
            self.history_changed.emit(list(self.history))
            if not self.paused:
                self.timer.start()
            break
        self.queue_changed.emit(len(self.pending))

    def select_history(self, index: int) -> None:
        images = list(self.history)
        if 0 <= index < len(images) and images[index].path.is_file():
            self.current = images[index]
            self.current_changed.emit(self.current)
            if not self.paused:
                self.timer.start()

    def clear_history(self) -> None:
        self.history.clear()
        self.history_changed.emit([])

    def clear(self) -> None:
        self.current = None
        self.pending.clear()
        self._seen.clear()
        self.clear_history()
        self.current_changed.emit(None)
        self.queue_changed.emit(0)


class ImageCanvas(QWidget):
    """Paint transitions at a bounded frame rate without changing source files."""
    load_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.setObjectName("imageCanvas")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.current_pixmap = QPixmap()
        self.previous_pixmap = QPixmap()
        self.transition = "crossfade"
        self.transition_seconds = 0.8
        self.interval = 9.0
        self.empty_message = "Images will appear as speech reveals a topic.\nStart monitoring or play a recording."
        self._started = time.monotonic()
        self._tiles = list(range(96))
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(33)
        self.animation_timer.timeout.connect(self._frame)

    def set_empty_message(self, message: str) -> None:
        self.empty_message = message[:1200]
        if self.current_pixmap.isNull():
            self.update()

    def set_image(self, candidate: ImageCandidate | None) -> bool:
        if candidate is None:
            self.animation_timer.stop()
            self.previous_pixmap = QPixmap()
            self.current_pixmap = QPixmap()
            self.update()
            return True
        reader = QImageReader(str(candidate.path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.width() > 4096 or size.height() > 4096:
            reader.setScaledSize(size.scaled(QSize(4096, 4096), Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.load_failed.emit(f"Could not display image: {reader.errorString()}")
            return False
        self.previous_pixmap = self.current_pixmap
        self.current_pixmap = QPixmap.fromImage(image)
        self._started = time.monotonic()
        random.Random(candidate.image_url).shuffle(self._tiles)
        if self.transition != "cut" and (self.transition == "zoom" or self.transition_seconds > 0):
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            self.previous_pixmap = QPixmap()
        self.update()
        return True

    def _frame(self) -> None:
        elapsed = time.monotonic() - self._started
        duration = self.interval if self.transition == "zoom" else self.transition_seconds
        if elapsed >= duration:
            self.animation_timer.stop()
            self.previous_pixmap = QPixmap()
        self.update()

    def _draw(self, painter: QPainter, pixmap: QPixmap, zoom: float = 1) -> None:
        if pixmap.isNull():
            return
        fitted = pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        width, height = fitted.width() * zoom, fitted.height() * zoom
        rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080c11"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.current_pixmap.isNull():
            painter.setPen(QColor("#8697a9"))
            painter.drawText(self.rect().adjusted(30, 30, -30, -30), Qt.AlignmentFlag.AlignCenter,
                             self.empty_message)
            return
        elapsed = time.monotonic() - self._started
        progress = min(1, elapsed / max(0.001, self.transition_seconds))
        if self.transition == "zoom":
            self._draw(painter, self.current_pixmap, 1 + 0.055 * min(1, elapsed / max(1, self.interval)))
        elif self.transition == "crossfade" and not self.previous_pixmap.isNull() and progress < 1:
            self._draw(painter, self.previous_pixmap)
            painter.setOpacity(progress)
            self._draw(painter, self.current_pixmap)
        elif self.transition == "dissolve" and not self.previous_pixmap.isNull() and progress < 1:
            self._draw(painter, self.previous_pixmap)
            path = QPainterPath()
            for tile in self._tiles[:int(progress * len(self._tiles))]:
                row, col = divmod(tile, 12)
                path.addRect(QRectF(col * self.width() / 12, row * self.height() / 8,
                                    self.width() / 12 + 1, self.height() / 8 + 1))
            painter.setClipPath(path)
            self._draw(painter, self.current_pixmap)
        else:
            self._draw(painter, self.current_pixmap)
