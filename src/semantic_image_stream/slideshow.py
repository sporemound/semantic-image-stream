"""Bounded slideshow playback and genuine Qt image transitions."""
from __future__ import annotations

from collections import deque
from html import escape
import random
import time

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImageReader, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QTextLayout, QTextOption
from PySide6.QtWidgets import QFrame, QTextEdit, QVBoxLayout, QWidget

from .models import ImageCandidate


_PROVIDER_LABELS = {
    "archive": "Internet Archive",
    "deviantart": "DeviantArt",
    "duckduckgo": "DuckDuckGo Images",
    "openverse": "Openverse",
    "wikimedia": "Wikimedia Commons",
}


def _overlay_text(value, limit: int = 1200) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _provider_label(value: str) -> str:
    text = _overlay_text(value, 200)
    return _PROVIDER_LABELS.get(text.casefold(), text or "Unknown source")


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
        self.current_credit = ""
        self.current_transcript = ""
        self._transcript_layout_cache = None
        self._transcript_pending = deque()
        self._transcript_idle = True
        self._transcript_page_end = 0
        self._transcript_timer = QTimer(self)
        self._transcript_timer.setSingleShot(True)
        self._transcript_timer.timeout.connect(self._advance_transcript)
        self.title_overlay_visible = False
        self._title_layout_cache = None
        self.context_overlay_visible = False
        self._fullscreen_info = dict.fromkeys(
            ("narrative", "scene", "subjects", "title", "credits", "transcript"), True
        )
        self._context_content = None
        self._context_rendered_html = None
        self._context_has_sections = False
        self._credit_has_sections = False
        self.context_panel = QFrame(self)
        self.context_panel.setObjectName("fullscreenContext")
        self.context_panel.setStyleSheet(
            "QFrame#fullscreenContext { background: rgba(8, 14, 22, 220); "
            "border: 1px solid rgba(168, 217, 211, 65); border-radius: 10px; }"
            "QTextEdit { background: transparent; color: #e4edf5; border: 0; }"
        )
        context_layout = QVBoxLayout(self.context_panel)
        context_layout.setContentsMargins(18, 14, 18, 14)
        self.context_text = QTextEdit(self.context_panel)
        self.context_text.setReadOnly(True)
        self.context_text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.context_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.context_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.context_text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        context_layout.addWidget(self.context_text, 1)
        self.credit_text = QTextEdit(self.context_panel)
        self.credit_text.setReadOnly(True)
        self.credit_text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.credit_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.credit_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.credit_text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.credit_text.setMinimumHeight(0)
        self.credit_text.hide()
        context_layout.addWidget(self.credit_text)
        self.context_panel.hide()
        self.set_context("", "", ())
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
        self._position_context_panel()
        self.update()

    def set_context_overlay(self, visible: bool) -> None:
        self.context_overlay_visible = bool(visible)
        self._position_context_panel()
        if visible:
            self.context_panel.raise_()
            self._schedule_transcript()
        else:
            self._transcript_timer.stop()
        self.update()

    def set_fullscreen_info(self, selections: dict[str, bool]) -> None:
        """Choose individual details for the context view without changing source data."""
        updated = {
            name: bool(selections.get(name, enabled))
            for name, enabled in self._fullscreen_info.items()
        }
        if updated == self._fullscreen_info:
            return
        transcript_changed = updated["transcript"] != self._fullscreen_info["transcript"]
        self._fullscreen_info = updated
        self._refresh_context_text()
        self._refresh_credit_panel()
        self._position_context_panel()
        if transcript_changed:
            self._transcript_timer.stop()
            self._schedule_transcript()
        self.update()

    def set_context(self, summary: str, scene: str, visual_subjects) -> None:
        """Show accepted live story context as text, with no provider markup or links."""
        content = (str(summary).strip(), str(scene).strip(), tuple(str(subject) for subject in visual_subjects))
        if content == self._context_content:
            return
        self._context_content = content
        self._refresh_context_text()
        self._position_context_panel()

    def _refresh_context_text(self) -> None:
        content = self._context_content or ("", "", ())
        sections = []
        for key, heading, text in (("narrative", "Narrative context", content[0]),
                                   ("scene", "Current scene", content[1]),
                                   ("subjects", "Visual subjects", "\n".join(content[2]))):
            if self._fullscreen_info[key] and text:
                sections.append(
                    '<p style="color:#a8d9d3; margin-top:14px; margin-bottom:5px;">'
                    f'<b>{heading}</b></p><p style="margin-top:0; margin-bottom:0;">'
                    f'{escape(text).replace(chr(10), "<br>")}</p>'
                )
        if not sections and self._fullscreen_info["narrative"]:
            sections.append('<p><b>Narrative context</b></p><p>Waiting for interpreted speech.</p>')
        self._context_has_sections = bool(sections)
        rendered_html = "".join(sections)
        if rendered_html != self._context_rendered_html:
            self._context_rendered_html = rendered_html
            self.context_text.setHtml(rendered_html)
            self.context_text.verticalScrollBar().setValue(0)
        self.context_text.setVisible(self._context_has_sections)

    def set_transcript(self, text: str) -> None:
        """Replace the caption stream, for a restored passage or a new book."""
        self._transcript_timer.stop()
        self._transcript_pending.clear()
        self._transcript_idle = not bool(text.strip())
        self._set_transcript_page(" ".join(text.split()))
        self._schedule_transcript()

    def enqueue_transcript(self, texts) -> None:
        """Show every new live passage in order, without truncating long speech."""
        passages = [" ".join(text.split()) for text in texts if text.strip()]
        if not passages:
            return
        if not self.context_overlay_visible or not self._fullscreen_info["transcript"]:
            # Retain only the latest incoming passage while captions are hidden,
            # so enabling them does not replay a growing hidden backlog.
            self.set_transcript(passages[-1])
            return
        if self._transcript_idle or not self.current_transcript:
            self._transcript_idle = False
            self._set_transcript_page(passages.pop(0))
        self._transcript_pending.extend(passages)
        if not self._transcript_timer.isActive():
            self._schedule_transcript()

    def _set_transcript_page(self, text: str) -> None:
        self.current_transcript = text
        self._transcript_layout_cache = None
        self._position_context_panel()
        self.update()

    def _schedule_transcript(self) -> None:
        if not self.context_overlay_visible or not self._fullscreen_info["transcript"] or self._transcript_idle:
            return
        layout = self._transcript_layout()
        if layout is not None:
            words = len(" ".join(layout[1]).split())
            self._transcript_timer.start(max(1000, min(4000, words * 200)))

    def _advance_transcript(self) -> None:
        self._transcript_timer.stop()
        if self._transcript_idle or not self.context_overlay_visible or not self._fullscreen_info["transcript"]:
            return
        self._transcript_layout()
        remaining = self.current_transcript[self._transcript_page_end:].lstrip()
        if remaining:
            self._set_transcript_page(remaining)
        elif self._transcript_pending:
            self._set_transcript_page(self._transcript_pending.popleft())
        else:
            self._transcript_idle = True
            return
        self._schedule_transcript()

    def _refresh_credit_panel(self) -> None:
        sections = []
        if self._fullscreen_info["title"] and self.current_title:
            sections.append(f"Image title\n{self.current_title}")
        if self._fullscreen_info["credits"] and self.current_credit:
            sections.append(f"Image credits\n{self.current_credit}")
        self._credit_has_sections = bool(sections)
        text = "\n\n".join(sections)
        if text != self.credit_text.toPlainText():
            self.credit_text.setPlainText(text)
        self.credit_text.setVisible(self._credit_has_sections)

    def _position_context_panel(self) -> None:
        self.context_panel.setVisible(
            self.context_overlay_visible and (self._context_has_sections or self._credit_has_sections)
        )
        if not self.context_overlay_visible:
            return
        margin = max(12, min(64, int(self.width() * .04), int(self.height() * .05)))
        width = max(1, min(720, self.width() - 2 * margin,
                           max(340, int(self.width() * .48))))
        bottom = self.height() - margin
        if self.title_overlay_visible and self.current_title and not self.context_overlay_visible:
            bottom = min(bottom, int(self._title_layout()[2].top()) - margin)
        transcript_layout = self._transcript_layout()
        if transcript_layout is not None:
            bottom = min(bottom, int(transcript_layout[2].top()) - margin)
        height = max(1, min(int(self.height() * .60), bottom - margin))
        font = QFont("Segoe UI")
        font.setPixelSize(max(14, min(24, int(self.height() * .024))))
        if self.context_text.document().defaultFont() != font:
            self.context_text.setFont(font)
            self.context_text.document().setDefaultFont(font)
        credit_font = QFont(font)
        credit_font.setPixelSize(max(12, min(18, int(self.height() * .021))))
        if self.credit_text.document().defaultFont() != credit_font:
            self.credit_text.setFont(credit_font)
            self.credit_text.document().setDefaultFont(credit_font)
        self.credit_text.setMaximumHeight(max(48, min(200, int(height * .35))))
        self.context_panel.setGeometry(margin, margin, width, height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_context_panel()

    def set_image(self, candidate: ImageCandidate | None) -> bool:
        if candidate is None:
            self.animation_timer.stop()
            self.previous_pixmap = QPixmap()
            self.current_pixmap = QPixmap()
            self.current_title = ""
            self.current_credit = ""
            self._refresh_credit_panel()
            self._title_layout_cache = None
            self._position_context_panel()
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
        self.current_title = _overlay_text(candidate.title) or "Untitled image"
        creator = _overlay_text(candidate.creator, 300)
        provider = _provider_label(candidate.provider)
        rights = _overlay_text(candidate.license, 500)
        credit = []
        if creator and creator.casefold() != "unknown":
            credit.append(f"By {creator}")
        if provider:
            credit.append(provider)
        if rights and not rights.casefold().startswith("unknown"):
            credit.append(rights)
        self.current_credit = " · ".join(credit)[:1200]
        self._refresh_credit_panel()
        self._title_layout_cache = None
        self._position_context_panel()
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

    def _overlay_margin(self) -> int:
        return max(18, min(96, int(self.width() * .055), int(self.height() * .07)))

    def _credit_font(self) -> QFont:
        font = QFont("Segoe UI")
        font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPixelSize(max(12, min(24, int(self.height() * .026), int(self.width() * .018))))
        return font

    def _title_layout(self):
        key = self.width(), self.height(), self.current_title, self.current_credit, self.context_overlay_visible
        if self._title_layout_cache is not None and self._title_layout_cache[0] == key:
            return self._title_layout_cache[1]
        margin = self._overlay_margin()
        width = max(1, self.width() - 2 * margin)
        size = max(20, min(76, int(self.height() * .075), int(self.width() * .055)))
        show_credit = self.current_credit and not self.context_overlay_visible
        credit_height = QFontMetricsF(self._credit_font()).height() if show_credit else 0
        credit_gap = max(7, int(self.height() * .012)) if show_credit else 0
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
        bottom = self.height() - margin - credit_height - credit_gap
        bounds = QRectF(margin, bottom - spacing * len(lines), width, spacing * len(lines))
        result = font, lines, bounds
        self._title_layout_cache = key, result
        return result

    def _credit_layout(self):
        if not self.current_credit or self.context_overlay_visible:
            return None
        margin = self._overlay_margin()
        width = max(1, self.width() - 2 * margin)
        font = self._credit_font()
        metrics = QFontMetricsF(font)
        text = metrics.elidedText(self.current_credit, Qt.TextElideMode.ElideRight, width)
        bounds = QRectF(margin, self.height() - margin - metrics.height(), width, metrics.height())
        return font, text, bounds

    def _transcript_layout(self):
        if not self.context_overlay_visible or not self._fullscreen_info["transcript"] or not self.current_transcript:
            return None
        key = self.width(), self.height(), self.current_transcript
        if self._transcript_layout_cache is not None and self._transcript_layout_cache[0] == key:
            return self._transcript_layout_cache[1]
        margin = self._overlay_margin()
        width = max(1, self.width() - 2 * margin)
        font = QFont("Arial")
        font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPixelSize(max(16, min(48, int(self.height() * .047), int(self.width() * .04))))
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout = QTextLayout(self.current_transcript, font)
        layout.setTextOption(option)
        layout.beginLayout()
        lines = []
        encoded = self.current_transcript.encode("utf-16-le")
        page_end = 0
        for _ in range(2):
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            page_end = line.textStart() + line.textLength()
            lines.append(encoded[line.textStart() * 2:page_end * 2].decode("utf-16-le").strip())
        layout.endLayout()
        self._transcript_page_end = len(encoded[:page_end * 2].decode("utf-16-le"))
        metrics = QFontMetricsF(font)
        spacing = metrics.lineSpacing() * 1.05
        bottom = self.height() - margin
        bounds = QRectF(margin, bottom - spacing * len(lines), width, spacing * len(lines))
        result = font, lines, bounds
        self._transcript_layout_cache = key, result
        return result

    def _draw_transcript(self, painter: QPainter) -> None:
        layout = self._transcript_layout()
        if layout is None:
            return
        font, lines, bounds = layout
        metrics = QFontMetricsF(font)
        baseline = bounds.top() + metrics.ascent()
        path = QPainterPath()
        for line in lines:
            left = (self.width() - metrics.horizontalAdvance(line)) / 2
            path.addText(QPointF(left, baseline), font, line)
            baseline += metrics.lineSpacing() * 1.05
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = QPen(QColor("#000000"), max(2, font.pixelSize() * .09))
        outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, outline)
        painter.fillPath(path, QColor("#ffffff"))
        painter.restore()

    def _draw_title(self, painter: QPainter) -> None:
        if not self.title_overlay_visible or not self.current_title or self.context_overlay_visible:
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
        credit_layout = self._credit_layout()
        if credit_layout is not None:
            credit_font, credit_text, credit_bounds = credit_layout
            credit_metrics = QFontMetricsF(credit_font)
            credit_baseline = credit_bounds.top() + credit_metrics.ascent()
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.setFont(credit_font)
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(QPointF(credit_bounds.left() + 1, credit_baseline + 2), credit_text)
            painter.setPen(QColor("#c7d7df"))
            painter.drawText(QPointF(credit_bounds.left(), credit_baseline), credit_text)
            painter.restore()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080c11"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.current_pixmap.isNull():
            painter.setPen(QColor("#8697a9"))
            painter.drawText(self.rect().adjusted(30, 30, -30, -30), Qt.AlignmentFlag.AlignCenter,
                             self.empty_message)
            self._draw_transcript(painter)
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
        self._draw_transcript(painter)
