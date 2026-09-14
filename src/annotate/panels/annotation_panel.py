from __future__ import annotations
import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QComboBox,
    QSlider,
    QPushButton,
    QLabel,
    QFileDialog,
)
from html import escape
from annotate.config import DEFAULT_LABEL_MAPPING, DEFAULT_SAVE_PATH


class AnnotationPanel(QWidget):
    """
    Controls and status fields for the annotation workflow.

    This panel does not contain annotation logic. MainWindow owns the
    annotation state machine and updates this panel through its public methods.
    """

    confirm_label_requested = pyqtSignal(int)
    distance_to_cable_changed = pyqtSignal(float)
    toggle_existing_labels_requested = pyqtSignal(bool)
    
    DISTANCE_TO_CABLE_MIN_M = 0
    DISTANCE_TO_CABLE_MAX_M = 10_000
    DISTANCE_TO_CABLE_STEP_M = 10

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout(self)
        
        self.workflow_label = QLabel()
        self.workflow_label.setWordWrap(True)
        self.workflow_label.setTextFormat(Qt.TextFormat.RichText)
        self.workflow_label.setStyleSheet(
            "padding: 4px; background: #eef6ff; border: 1px solid #aaccee;"
        )

        self.workflow_label.setText(
            "<b>Annotation workflow</b><br>"
            "<u>A</u>: select / confirm apex<br>"
            "<u>H</u>: fit / confirm distance to cable<br>"
            "<u>E</u>: select / confirm endpoint<br>"
            "<u>D</u>: select / confirm distance range<br>"
            "<u>F</u>: edit / finish F-X boxes<br>"
            "<u>Esc</u>: cancel annotation"
        )

        layout.addRow(self.workflow_label)

        # ----------------------------------------------------------
        # Label selection
        # ----------------------------------------------------------
        self.label_dropdown = QComboBox()

        for label_number in sorted(DEFAULT_LABEL_MAPPING):
            if label_number == 0:
                continue

            label_name = DEFAULT_LABEL_MAPPING[label_number]

            self.label_dropdown.addItem(
                f"{label_number}: {label_name}",
                userData=label_number,
            )

        layout.addRow("Selected label", self.label_dropdown)

        # Show all possible key labels simultaneously.
        self.label_legend = QLabel()
        self.label_legend.setWordWrap(True)
        self.label_legend.setTextFormat(Qt.TextFormat.RichText)
        self.label_legend.setStyleSheet(
            "padding: 4px; background: #f5f5f5; border: 1px solid #cccccc;"
        )

        self.label_legend.setText(self._build_label_legend())

        layout.addRow("Number shortcuts", self.label_legend)

        # ----------------------------------------------------------
        # Read-only annotation values
        # ----------------------------------------------------------
        self.apex_time_display = self._readonly_field()
        self.apex_dist_display = self._readonly_field()
        self.duration_display = self._readonly_field()

        self.distance_to_cable_display = self._readonly_field()

        self.dist_min_display = self._readonly_field()
        self.dist_max_display = self._readonly_field()

        self.freq_min_display = self._readonly_field()
        self.freq_max_display = self._readonly_field()

        layout.addRow("Apex time (s)", self.apex_time_display)
        layout.addRow("Apex distance (m)", self.apex_dist_display)
        layout.addRow("Duration (s)", self.duration_display)

        # ----------------------------------------------------------
        # Distance-to-cable slider
        # ----------------------------------------------------------
        self.distance_to_cable_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_to_cable_slider.setRange(
            self.DISTANCE_TO_CABLE_MIN_M,
            self.DISTANCE_TO_CABLE_MAX_M,
        )
        self.distance_to_cable_slider.setSingleStep(
            self.DISTANCE_TO_CABLE_STEP_M
        )
        self.distance_to_cable_slider.setPageStep(100)
        self.distance_to_cable_slider.setValue(0)
        self.distance_to_cable_slider.setEnabled(False)

        self.distance_to_cable_slider.valueChanged.connect(
            self._on_distance_to_cable_changed
        )

        distance_layout = QVBoxLayout()
        distance_layout.addWidget(self.distance_to_cable_display)
        distance_layout.addWidget(self.distance_to_cable_slider)

        layout.addRow("Distance to cable (m)", distance_layout)

        # ----------------------------------------------------------
        # Distance range
        # ----------------------------------------------------------
        distance_range_layout = QHBoxLayout()
        distance_range_layout.addWidget(QLabel("Min"))
        distance_range_layout.addWidget(self.dist_min_display)
        distance_range_layout.addWidget(QLabel("Max"))
        distance_range_layout.addWidget(self.dist_max_display)

        layout.addRow("Distance range (m)", distance_range_layout)

        # ----------------------------------------------------------
        # Frequency range
        # ----------------------------------------------------------
        frequency_range_layout = QHBoxLayout()
        frequency_range_layout.addWidget(QLabel("Min"))
        frequency_range_layout.addWidget(self.freq_min_display)
        frequency_range_layout.addWidget(QLabel("Max"))
        frequency_range_layout.addWidget(self.freq_max_display)

        layout.addRow("Frequency range (Hz)", frequency_range_layout)

        # ----------------------------------------------------------
        # Save path
        # ----------------------------------------------------------
        self.labels_path_edit = QLineEdit(DEFAULT_SAVE_PATH)
        self.browse_button = QPushButton("Browse...")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.labels_path_edit)
        file_layout.addWidget(self.browse_button)

        layout.addRow("Annotation file", file_layout)

        self.browse_button.clicked.connect(self.select_save_path)

        # ----------------------------------------------------------
        # Existing labels visibility
        # ----------------------------------------------------------
        self.toggle_labels_button = QPushButton("Show Existing Labels")
        self.toggle_labels_button.setCheckable(True)
        self.toggle_labels_button.toggled.connect(
            self.toggle_existing_labels_requested.emit
        )

        layout.addRow("Show Existing Labels", self.toggle_labels_button)

        # ----------------------------------------------------------
        # Confirm and save
        # ----------------------------------------------------------
        self.confirm_save_button = QPushButton("Confirm and Save")
        self.confirm_save_button.clicked.connect(
            self._emit_confirm_label
        )

        layout.addRow("Confirm and Save", self.confirm_save_button)

    def select_save_path(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Annotation Save File",
            self.labels_path_edit.text() or DEFAULT_SAVE_PATH,

            "CSV files (*.csv);;All Files (*)",
        )

        if filepath:
            self.labels_path_edit.setText(filepath)

    def get_settings(self) -> dict:
        """
        Return only settings that should become part of DataManager state.
        Annotation preview fields are not settings and are intentionally
        excluded.
        """
        return {
            "labels_file_path": self.labels_path_edit.text().strip(),

            # Fixed mapping from config, not user-editable.
            "label_mapping": DEFAULT_LABEL_MAPPING.copy(),
        }

    def set_distance_to_cable_enabled(self, enabled: bool) -> None:
        self.distance_to_cable_slider.setEnabled(enabled)

    def set_distance_to_cable(self, distance_m: float) -> None:
        """Set slider/display while safely clamping to configured range."""
        value = int(round(distance_m))

        value = max(
            self.distance_to_cable_slider.minimum(),
            min(value, self.distance_to_cable_slider.maximum()),
        )

        self.distance_to_cable_slider.setValue(value)

    def clear_distance_to_cable(self) -> None:
        self.distance_to_cable_slider.blockSignals(True)
        self.distance_to_cable_slider.setValue(0)
        self.distance_to_cable_slider.blockSignals(False)

        self.distance_to_cable_display.clear()
        self.distance_to_cable_slider.setEnabled(False)

    def clear_preview(self) -> None:
        for field in (
            self.apex_time_display,
            self.apex_dist_display,
            self.duration_display,
            self.distance_to_cable_display,
            self.dist_min_display,
            self.dist_max_display,
            self.freq_min_display,
            self.freq_max_display,
        ):
            field.clear()

        self.clear_distance_to_cable()

    def update_preview(self, values: dict) -> None:
        """Populate read-only annotation summary fields."""
        self._set_float(
            self.apex_time_display,
            values.get("apex_time_local"),
            decimals=3,
        )
        self._set_float(
            self.apex_dist_display,
            values.get("apex_dist"),
            decimals=2,
        )
        # self._set_float(
        #     self.duration_display,
        #     values.get("duration"),
        #     decimals=3,
        # )
        
        distance_to_cable = values.get("distance_to_cable")
        self._set_float(
            self.distance_to_cable_display,
            distance_to_cable,
            decimals=1,
        )
        if distance_to_cable is not None:
            self.distance_to_cable_slider.blockSignals(True)
            self.set_distance_to_cable(float(distance_to_cable))
            self.distance_to_cable_slider.blockSignals(False)
            
        self._set_float(
            self.dist_min_display,
            values.get("dist_min"),
            decimals=2,
        )
        self._set_float(
            self.dist_max_display,
            values.get("dist_max"),
            decimals=2,
        )
        self._set_float(
            self.freq_min_display,
            values.get("f_min"),
            decimals=2,
        )
        self._set_float(
            self.freq_max_display,
            values.get("f_max"),
            decimals=2,
        )
        
    def apex_time_to_utc_text(self, apex_time_offset_s: float) -> str:
        """Format a current-window-relative time coordinate as UTC."""
        window_start_utc = float(
            self.data_manager.loaded_data["time_stamps"][0]
        )

        apex_time_utc = window_start_utc + float(apex_time_offset_s)

        return datetime.fromtimestamp(
            apex_time_utc,
            tz=datetime.timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"

    def _build_label_legend(self) -> str:
        """
        Render keyboard label shortcuts.

        Number keys 1–9 assign labels. Key 0 is reserved and does not create
        a new annotation.
        """
        rows = []

        for number in sorted(DEFAULT_LABEL_MAPPING):
            if number == 0:
                continue

            name = DEFAULT_LABEL_MAPPING[number]
            safe_name = escape(name or "(unassigned)")

            rows.append(
                f"<b><u>{number}</u></b>: {safe_name}"
            )

        rows.append("<b><u>0</u></b>: cancel / reserved")

        return "<br>".join(rows)

    @staticmethod
    def _readonly_field() -> QLineEdit:
        field = QLineEdit()
        field.setReadOnly(True)
        return field

    def _emit_confirm_label(self) -> None:
        label_number = int(self.label_dropdown.currentData())
        self.confirm_label_requested.emit(label_number)

    def _on_distance_to_cable_changed(self, value: int) -> None:
        distance_m = float(value)

        self.distance_to_cable_display.setText(
            f"{distance_m:.1f}"
        )

        self.distance_to_cable_changed.emit(distance_m)
        
    @staticmethod
    def _set_float(
        field: QLineEdit,
        value,
        *,
        decimals: int,
    ) -> None:
        if value is None:
            field.clear()
            return

        field.setText(f"{float(value):.{decimals}f}")