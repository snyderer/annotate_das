# annotate/panels/navigation_panel.py
from __future__ import annotations

from datetime import datetime, timezone

from PyQt6.QtCore import QDate, QTime, QDateTime, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QDoubleSpinBox,
    QDateTimeEdit,
)

from annotate.config import UserSettings


class NavigationPanel(QWidget):
    """Controls for absolute-time DAS navigation."""

    def __init__(self, parent=None):
        super().__init__(parent)

        defaults = UserSettings()
        layout = QFormLayout(self)

        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setTimeSpec(Qt.TimeSpec.UTC)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setValue(defaults.duration_s)
        self.duration_spin.setSuffix(" s")

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 3600.0)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(defaults.navigation_step_s)
        self.step_spin.setSuffix(" s")

        self.btn_back = QPushButton("<")
        self.btn_forward = QPushButton(">")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_back)
        button_layout.addWidget(self.btn_forward)

        layout.addRow("Start time (UTC)", self.start_time_edit)
        layout.addRow("Plot duration", self.duration_spin)
        layout.addRow("Navigation step", self.step_spin)
        layout.addRow(button_layout)

        self.step_spin.valueChanged.connect(self._update_button_labels)
        self._update_button_labels()

    def _update_button_labels(self) -> None:
        step_s = self.step_spin.value()
        self.btn_back.setText(f"< {step_s:g}s")
        self.btn_forward.setText(f"> {step_s:g}s")

    def get_settings(self) -> dict:
        return {
            "duration_s": self.duration_spin.value(),
            "navigation_step_s": self.step_spin.value(),
        }

    def selected_start_time(self) -> datetime:
        """Return the selected UTC time as a timezone-aware datetime."""
        qdt = self.start_time_edit.dateTime().toUTC()

        return datetime(
            qdt.date().year(),
            qdt.date().month(),
            qdt.date().day(),
            qdt.time().hour(),
            qdt.time().minute(),
            qdt.time().second(),
            qdt.time().msec() * 1000,
            tzinfo=timezone.utc,
        )

    def set_start_time(self, value: datetime) -> None:
        """Update the displayed navigation time from a UTC datetime."""
        if value.tzinfo is None:
            raise ValueError("start time must be timezone-aware")

        value = value.astimezone(timezone.utc)

        qdt = QDateTime(
            QDate(value.year, value.month, value.day),
            QTime(
                value.hour,
                value.minute,
                value.second,
                value.microsecond // 1000,
            ),
            Qt.TimeSpec.UTC,
        )

        self.start_time_edit.setDateTime(qdt)