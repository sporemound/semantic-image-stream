"""Two keyboard-accessible handles with linear calendar and logarithmic deep-time scales."""
from __future__ import annotations

import math
from datetime import date

from PySide6.QtCore import QPointF, Qt, Signal, QSignalBlocker
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSlider, QWidget

from .temporal_filter import MIN_YEAR


class _Handle(QSlider):
    def __init__(self, owner, index):
        super().__init__(Qt.Orientation.Horizontal, owner)
        self.owner, self.index = owner, index
        self.setRange(0, 10_000)
        self.setSingleStep(1)
        self.setPageStep(100)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Start of time period" if index == 0 else "End of time period")
        self.setAccessibleDescription("Use arrow keys to move this handle. Tab selects the other handle.")
        self.valueChanged.connect(lambda value: owner._handle_changed(index, value))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#eaf6ee") if self.hasFocus() else QColor("#6fb9b1"), 2))
        painter.setBrush(QColor("#86d4c4") if self.isEnabled() else QColor("#536574"))
        painter.drawEllipse(self.rect().adjusted(3, 3, -3, -3))

    def focusInEvent(self, event):
        self.owner._active_handle = self.index
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.update()
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.owner._begin_drag(self.index)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.owner._drag_handle == self.index:
            self.owner._drag_to(self.mapTo(self.owner, event.position().toPoint()).x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.owner._end_drag()
            event.accept()


class RangeSlider(QWidget):
    valuesChanged = Signal(float, float)
    sliderPressed = Signal()
    sliderReleased = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 34)
        self.setAccessibleName("Time period range")
        self._values = (1000.0, 2026.0)
        self._scale = "calendar"
        self._geological_domain = (1950.0 - 4.567e9, float(date.today().year))
        self._active_handle = 0
        self._drag_handle = None
        self._handles = (_Handle(self, 0), _Handle(self, 1))
        self._sync_handles()

    @property
    def values(self):
        return self._values

    @property
    def is_dragging(self):
        return self._drag_handle is not None

    def set_values(self, start, end):
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("Start must be no later than end")
        # Only explicit values widen the scale; dragging keeps a stable domain.
        older, newer = self._geological_domain
        self._geological_domain = min(older, max(MIN_YEAR, start)), newer
        self._values = start, end
        self._sync_handles()

    def set_scale(self, scale):
        if scale not in {"calendar", "geological"}:
            raise ValueError("Choose calendar or geological time")
        self._scale = scale
        self._sync_handles()

    def set_handle_labels(self, start, end):
        self._handles[0].setAccessibleName(f"Start of time period: {start}")
        self._handles[1].setAccessibleName(f"End of time period: {end}")

    def _domain(self):
        return (-9999.0, 2100.0) if self._scale == "calendar" else self._geological_domain

    @staticmethod
    def _signed_log(value):
        return math.copysign(math.log1p(abs(value)), value)

    def _position(self, year):
        start, end = self._domain()
        if self._scale == "geological":
            start, end, year = (self._signed_log(value - 1950) for value in (start, end, year))
        return max(0.0, min(1.0, (year - start) / (end - start)))

    def _year(self, position):
        start, end = self._domain()
        if self._scale == "geological":
            start, end = self._signed_log(start - 1950), self._signed_log(end - 1950)
            value = start + position * (end - start)
            return 1950 + math.copysign(math.expm1(abs(value)), value)
        return float(round(start + position * (end - start)))

    def _sync_handles(self):
        for index, handle in enumerate(self._handles):
            with QSignalBlocker(handle):
                handle.setValue(round(self._position(self._values[index]) * 10_000))
            handle.setAccessibleDescription("Use arrow keys to move this handle. Tab selects the other handle. Enter an exact date in the year field.")
        self._place_handles()
        self.update()

    def _place_handles(self):
        width = max(1, self.width() - 24)
        for index, handle in enumerate(self._handles):
            handle.setGeometry(round(self._position(self._values[index]) * width), (self.height() - 24) // 2, 24, 24)
        self._handles[self._active_handle].raise_()

    def _handle_changed(self, index, position):
        start, end = self._values
        year = self._year(position / 10_000)
        values = (min(year, end), end) if index == 0 else (start, max(year, start))
        changed = values != self._values
        self._values = values
        self._sync_handles()
        if changed:
            self.valuesChanged.emit(*values)

    def _begin_drag(self, index):
        self._active_handle = self._drag_handle = index
        self.sliderPressed.emit()

    def _drag_to(self, x):
        position = max(0, min(10_000, round((x - 12) / max(1, self.width() - 24) * 10_000)))
        self._handles[self._drag_handle].setValue(position)

    def _end_drag(self):
        self._drag_handle = None
        self.sliderReleased.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            positions = [12 + self._position(year) * (self.width() - 24) for year in self._values]
            distances = [abs(position - event.position().x()) for position in positions]
            index = self._active_handle if distances[0] == distances[1] else distances.index(min(distances))
            self._handles[index].setFocus(Qt.FocusReason.MouseFocusReason)
            self._begin_drag(index)
            self._drag_to(event.position().x())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_handle is not None:
            self._drag_to(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_handle is not None:
            self._end_drag()
            event.accept()

    def resizeEvent(self, event):
        self._place_handles()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        painter.setPen(QPen(QColor("#36485b"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(12, y), QPointF(self.width() - 12, y))
        painter.setPen(QPen(QColor("#86d4c4") if self.isEnabled() else QColor("#536574"), 4))
        left, right = [12 + self._position(year) * (self.width() - 24) for year in self._values]
        painter.drawLine(QPointF(left, y), QPointF(right, y))
