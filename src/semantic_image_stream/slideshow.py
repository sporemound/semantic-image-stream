"""Bounded slideshow playback and genuine Qt image transitions."""
from __future__ import annotations

from collections import deque
import random
import time

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImageReader, QLinearGradient, QPainter, QPainterPath, QPixmap, QTextLayout, QTextOption
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
        self._seen_hashes: set[int] = set()
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
        milliseconds = max(1000, int(seconds * 1000))
        if milliseconds != self.timer.interval():
            self.timer.setInterval(milliseconds)

    @staticmethod
    def _fingerprint(value: str) -> int | None:
        if (not isinstance(value, str) or len(value) != 16
                or any(character not in "0123456789abcdefABCDEF" for character in value)):
            return None
        try:
            return int(value, 16)
        except ValueError:
            return None

    def _already_seen(self, candidate: ImageCandidate) -> bool:
        fingerprint = self._fingerprint(candidate.phash)
        return (self.identity(candidate) in self._seen or
                (fingerprint is not None and any((fingerprint ^ previous).bit_count() <= 6
                                                for previous in self._seen_hashes)))

    def restore_seen(self, urls, hashes) -> None:
        """Merge durable displayed identities without loading old image files."""
        self._seen.update(url for url in urls if isinstance(url, str) and url)
        self._seen_hashes.update(value for item in hashes if (value := self._fingerprint(item)) is not None)
        self.pending = deque((image for image in self.pending if not self._already_seen(image)), maxlen=60)
        self.queue_changed.emit(len(self.pending))

    def set_topic(self, generation: str) -> None:
        """Discard pending results for a prior scene while retaining displayed identities."""
        if generation != self._generation:
            self.reset_playback(generation)

    def reset_playback(self, generation: str = "") -> None:
        """Keep the displayed frame and history, but discard future subject playback."""
        self.pending.clear()
        self._generation = generation
        self.queue_changed.emit(0)

    def set_queue(self, images: list[ImageCandidate], generation: str = "") -> None:
        """Merge unseen progressive results into the bounded pending queue."""
        self.set_topic(generation)
        pending_ids = {self.identity(candidate) for candidate in self.pending}
        for candidate in images:
            if len(self.pending) >= 60:
                break
            identity = self.identity(candidate)
            if candidate.path.is_file() and not self._already_seen(candidate) and identity not in pending_ids:
                self.pending.append(candidate)
                pending_ids.add(identity)
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

    def discard(self, candidate: ImageCandidate, previous: ImageCandidate | None = None) -> None:
        """Remove a file Qt could not decode, preserving the actual displayed frame."""
        identity = self.identity(candidate)
        self.pending = deque((image for image in self.pending if self.identity(image) != identity), maxlen=60)
        self.history = deque((image for image in self.history if self.identity(image) != identity), maxlen=60)
        if self.current is not None and self.identity(self.current) == identity:
            self.current = previous
        self.history_changed.emit(list(self.history))
        self.queue_changed.emit(len(self.pending))

    def advance(self, manual: bool = False) -> None:
        if self.paused and not manual:
            return
        candidate = None
        while self.pending and candidate is None:
            next_candidate = self.pending.popleft()
            if next_candidate.path.is_file() and not self._already_seen(next_candidate):
                candidate = next_candidate
        if candidate is not None:
            self.current = candidate
            self.history.append(candidate)
            self.current_changed.emit(candidate)
            # The canvas may synchronously reject decoding and restore the
            # previous frame. Only a successfully displayed candidate counts.
            if self.current is candidate:
                self._seen.add(self.identity(candidate))
                fingerprint = self._fingerprint(candidate.phash)
                if fingerprint is not None:
                    self._seen_hashes.add(fingerprint)
            self.history_changed.emit(list(self.history))
            if not self.paused:
                self.timer.start()
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

    def clear(self, clear_seen: bool = True) -> None:
        self.current = None
        self.reset_playback()
        if clear_seen:
            self._seen.clear()
            self._seen_hashes.clear()
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
        self.current_title = ""
        self.title_overlay_visible = False
        self._title_layout_cache = None
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

    def set_title_overlay(self, visible: bool) -> None:
        self.title_overlay_visible = bool(visible)
        self.update()

    def set_image(self, candidate: ImageCandidate | None) -> bool:
        if candidate is None:
            self.animation_timer.stop()
            self.previous_pixmap = QPixmap()
            self.current_pixmap = QPixmap()
            self.current_title = ""
            self._title_layout_cache = None
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
        self.current_title = " ".join((candidate.title or "Untitled image").split())[:1200]
        self._title_layout_cache = None
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

    def _title_layout(self):
        key = self.width(), self.height(), self.current_title
        if self._title_layout_cache is not None and self._title_layout_cache[0] == key:
            return self._title_layout_cache[1]
        margin = max(18, min(96, int(self.width() * .055), int(self.height() * .07)))
        width = max(1, self.width() - 2 * margin)
        size = max(20, min(76, int(self.height() * .075), int(self.width() * .055)))
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        for pixels in range(size, 11, -2):
            font = QFont("Georgia")
            font.setStyleHint(QFont.StyleHint.Serif)
            font.setItalic(True)
            font.setPixelSize(pixels)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, .4)
            layout = QTextLayout(self.current_title, font)
            layout.setTextOption(option)
            layout.beginLayout()
            lines, starts = [], []
            for _ in range(4):
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                starts.append(line.textStart())
                lines.append(self.current_title[line.textStart():line.textStart() + line.textLength()].strip())
            layout.endLayout()
            if len(lines) <= 3:
                break
        metrics = QFontMetricsF(font)
        if len(lines) > 3:
            lines = lines[:2] + [metrics.elidedText(self.current_title[starts[2]:], Qt.TextElideMode.ElideRight, width)]
        spacing = metrics.lineSpacing() * 1.05
        bounds = QRectF(margin, self.height() - margin - spacing * len(lines), width, spacing * len(lines))
        result = font, lines, bounds
        self._title_layout_cache = key, result
        return result

    def _draw_title(self, painter: QPainter) -> None:
        if not self.title_overlay_visible or not self.current_title:
            return
        font, lines, bounds = self._title_layout()
        gradient_top = max(0, bounds.top() - self.height() * .15)
        gradient = QLinearGradient(0, gradient_top, 0, self.height())
        gradient.setColorAt(0, QColor(4, 7, 10, 0))
        gradient.setColorAt(.45, QColor(4, 7, 10, 55))
        gradient.setColorAt(1, QColor(4, 7, 10, 210))
        painter.fillRect(QRectF(0, gradient_top, self.width(), self.height() - gradient_top), gradient)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setFont(font)
        painter.setClipRect(bounds.adjusted(-4, -8, 4, 8))
        metrics = QFontMetricsF(font)
        baseline = bounds.top() + metrics.ascent()
        for line in lines:
            painter.setPen(QColor(0, 0, 0, 170))
            painter.drawText(QPointF(bounds.left() + 1, baseline + 2), line)
            painter.setPen(QColor("#f5e8d0"))
            painter.drawText(QPointF(bounds.left(), baseline), line)
            baseline += metrics.lineSpacing() * 1.05
        painter.restore()

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
        painter.setOpacity(1)
        painter.setClipping(False)
        self._draw_title(painter)
