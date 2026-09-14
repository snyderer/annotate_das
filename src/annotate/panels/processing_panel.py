# annotate/panels/processing_panel.py
from PyQt6.QtWidgets import (
    QComboBox,
    QWidget,
    QFormLayout,
    QCheckBox,
    QDoubleSpinBox,
)

from annotate.config import UserSettings, BANDPASS_PRESETS


class ProcessingPanel(QWidget):
    """Controls affecting loaded/processed DAS signal data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        defaults = UserSettings()
        layout = QFormLayout(self)

        # File-boundary stitching
        self.boundary_correction_check = QCheckBox(
            "Correct file-boundary offsets"
        )
        self.boundary_correction_check.setChecked(
            getattr(defaults, "file_boundary_correction_enabled", True)
        )

        self.boundary_edge_spin = QDoubleSpinBox()
        self.boundary_edge_spin.setRange(0.01, 5.0)
        self.boundary_edge_spin.setDecimals(3)
        self.boundary_edge_spin.setSingleStep(0.05)
        self.boundary_edge_spin.setValue(
            getattr(defaults, "file_boundary_edge_duration_s", 0.25)
        )
        self.boundary_edge_spin.setSuffix(" s")

        # Bandpass
        self.bandpass_check = QCheckBox("Enable bandpass")
        self.bandpass_check.setChecked(defaults.bandpass_enabled)

        self.bandpass_preset_combo = QComboBox()

        for name in BANDPASS_PRESETS:
            self.bandpass_preset_combo.addItem(name)

        self.bandpass_preset_combo.setCurrentText("Whale calls (5–70 Hz)")

        self.f_lo_spin = QDoubleSpinBox()
        self.f_lo_spin.setRange(0.0, 100_000.0)
        self.f_lo_spin.setDecimals(2)
        self.f_lo_spin.setValue(defaults.f_lo_hz)
        self.f_lo_spin.setSuffix(" Hz")

        self.f_hi_spin = QDoubleSpinBox()
        self.f_hi_spin.setRange(0.0, 100_000.0)
        self.f_hi_spin.setDecimals(2)
        self.f_hi_spin.setValue(defaults.f_hi_hz)
        self.f_hi_spin.setSuffix(" Hz")

        self.fk_filter_check = QCheckBox("Enable F-K filter")
        self.fk_filter_check.setChecked(defaults.fk_filter_enabled)

        self.c_min_spin = QDoubleSpinBox()
        self.c_min_spin.setRange(0.0, 20_000.0)
        self.c_min_spin.setDecimals(1)
        self.c_min_spin.setValue(defaults.c_min_mps)
        self.c_min_spin.setSuffix(" m/s")

        self.c_max_spin = QDoubleSpinBox()
        self.c_max_spin.setRange(0.0, 20_000.0)
        self.c_max_spin.setDecimals(1)
        self.c_max_spin.setValue(defaults.c_max_mps)
        self.c_max_spin.setSuffix(" m/s")

        # Display downsampling.
        self.max_frequency_spin = QDoubleSpinBox()
        self.max_frequency_spin.setRange(1.0, 100_000.0)
        self.max_frequency_spin.setDecimals(2)
        self.max_frequency_spin.setValue(defaults.max_display_freq_hz)
        self.max_frequency_spin.setSuffix(" Hz")

        layout.addRow(self.boundary_correction_check)
        layout.addRow("Boundary estimate window", self.boundary_edge_spin)

        layout.addRow(self.bandpass_check)
        layout.addRow("Bandpass preset", self.bandpass_preset_combo)
        layout.addRow("Bandpass low", self.f_lo_spin)
        layout.addRow("Bandpass high", self.f_hi_spin)

        layout.addRow(self.fk_filter_check)
        layout.addRow("F-K minimum velocity", self.c_min_spin)
        layout.addRow("F-K maximum velocity", self.c_max_spin)

        layout.addRow("Maximum display frequency", self.max_frequency_spin)
       
        self.bandpass_preset_combo.currentTextChanged.connect(
            self._apply_bandpass_preset
        )
        self.bandpass_check.toggled.connect(self._update_enabled_state)
        self.fk_filter_check.toggled.connect(self._update_enabled_state)

        self._update_enabled_state()

    def _apply_bandpass_preset(self, preset_name: str) -> None:
        f_lo, f_hi = BANDPASS_PRESETS[preset_name]

        if f_lo is not None:
            self.f_lo_spin.setValue(f_lo)

        if f_hi is not None:
            self.f_hi_spin.setValue(f_hi)


    def _update_enabled_state(self) -> None:
        bandpass_enabled = self.bandpass_check.isChecked()

        self.bandpass_preset_combo.setEnabled(bandpass_enabled)
        self.f_lo_spin.setEnabled(bandpass_enabled)
        self.f_hi_spin.setEnabled(bandpass_enabled)

        fk_enabled = self.fk_filter_check.isChecked()

        self.c_min_spin.setEnabled(fk_enabled)
        self.c_max_spin.setEnabled(fk_enabled)
        
    def get_settings(self) -> dict:
        if (
            self.bandpass_check.isChecked()
            and self.f_lo_spin.value() >= self.f_hi_spin.value()
        ):
            raise ValueError(
                "Bandpass low frequency must be less than high frequency."
            )

        if (
            self.fk_filter_check.isChecked()
            and self.c_min_spin.value() >= self.c_max_spin.value()
        ):
            raise ValueError(
                "F-K minimum velocity must be less than maximum velocity."
            )

        return {
            "file_boundary_correction_enabled": (
                self.boundary_correction_check.isChecked()
            ),
            "file_boundary_edge_duration_s": (
                self.boundary_edge_spin.value()
            ),

            "bandpass_enabled": self.bandpass_check.isChecked(),
            "bandpass_preset": self.bandpass_preset_combo.currentText(),
            "f_lo_hz": self.f_lo_spin.value(),
            "f_hi_hz": self.f_hi_spin.value(),

            "fk_filter_enabled": self.fk_filter_check.isChecked(),
            "c_min_mps": self.c_min_spin.value(),
            "c_max_mps": self.c_max_spin.value(),

            "max_display_freq_hz": self.max_frequency_spin.value(),
        }