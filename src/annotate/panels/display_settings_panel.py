# annotate/panels/display_settings_panel.py
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
)

from annotate.config import UserSettings


class DisplaySettingsPanel(QWidget):
    LABEL_WIDTH = 105
    display_settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        defaults = UserSettings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --------------------------------------------------------------
        # T-X
        # --------------------------------------------------------------
        tx_box = QGroupBox("T-X Display")
        tx_form = QFormLayout(tx_box)

        self.tx_vmin_slider = self._make_level_slider(defaults.tx_vmin)
        self.tx_vmax_slider = self._make_level_slider(defaults.tx_vmax)

        tx_form.addRow(self._label("vmin"), self.tx_vmin_slider)
        tx_form.addRow(self._label("vmax"), self.tx_vmax_slider)

        layout.addWidget(tx_box)

        self.tx_vmin_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )
        self.tx_vmax_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )

        # --------------------------------------------------------------
        # F-X
        # --------------------------------------------------------------
        fx_box = QGroupBox("F-X Display")
        fx_form = QFormLayout(fx_box)

        self.fx_win_s_spin = QDoubleSpinBox()
        self.fx_win_s_spin.setRange(0.1, 60.0)
        self.fx_win_s_spin.setDecimals(2)
        self.fx_win_s_spin.setValue(defaults.fx_win_s)
        self.fx_win_s_spin.setSuffix(" s")

        self.fx_nfft_spin = QSpinBox()
        self.fx_nfft_spin.setRange(32, 2**16)
        self.fx_nfft_spin.setSingleStep(256)
        self.fx_nfft_spin.setValue(defaults.fx_nfft)

        self.fx_vmin_slider = self._make_level_slider(defaults.fx_vmin)
        self.fx_vmax_slider = self._make_level_slider(defaults.fx_vmax)

        fx_form.addRow(self._label("Window length"), self.fx_win_s_spin)
        fx_form.addRow(self._label("F-X NFFT"), self.fx_nfft_spin)
        fx_form.addRow(self._label("vmin"), self.fx_vmin_slider)
        fx_form.addRow(self._label("vmax"), self.fx_vmax_slider)

        layout.addWidget(fx_box)

        self.fx_vmin_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )
        self.fx_vmax_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )
        
        # --------------------------------------------------------------
        # Spectrogram
        # --------------------------------------------------------------
        spec_box = QGroupBox("Spectrogram Display")
        spec_form = QFormLayout(spec_box)

        self.spec_nfft_spin = QSpinBox()
        self.spec_nfft_spin.setRange(32, 2**13)
        self.spec_nfft_spin.setSingleStep(32)
        self.spec_nfft_spin.setValue(defaults.spec_nfft)

        self.spec_overlap_spin = QSpinBox()
        self.spec_overlap_spin.setRange(0, 99)
        self.spec_overlap_spin.setValue(int(defaults.spec_overlap))
        
        self.spec_vmin_slider = self._make_level_slider(
            defaults.spec_vmin
        )
        self.spec_vmax_slider = self._make_level_slider(
            defaults.spec_vmax
        )

        spec_form.addRow(self._label("NFFT"), self.spec_nfft_spin)
        spec_form.addRow(self._label("Overlap (%)"), self.spec_overlap_spin)
        spec_form.addRow(self._label("vmin"), self.spec_vmin_slider)
        spec_form.addRow(self._label("vmax"), self.spec_vmax_slider)

        layout.addWidget(spec_box)
        
        self.spec_vmin_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )
        self.spec_vmax_slider.valueChanged.connect(
            self.display_settings_changed.emit
        )
        
        display_settings_changed = pyqtSignal()

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setFixedWidth(self.LABEL_WIDTH)
        return label

    @staticmethod
    def _make_level_slider(value: float) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(value * 100))
        return slider

    def get_settings(self) -> dict:
        return {
            # F-X
            "win_s": self.fx_win_s_spin.value(),
            "fx_nfft": self.fx_nfft_spin.value(),
            "fx_vmin": self.fx_vmin_slider.value(),
            "fx_vmax": self.fx_vmax_slider.value(),

            # Spectrogram
            "spec_nfft": self.spec_nfft_spin.value(),
            "spec_overlap": self.spec_overlap_spin.value(),
            "spec_vmin": self.spec_vmin_slider.value(),
            "spec_vmax": self.spec_vmax_slider.value(),

            # T-X
            "tx_vmin": self.tx_vmin_slider.value(),
            "tx_vmax": self.tx_vmax_slider.value(),
        }