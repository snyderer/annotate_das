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
            
    def set_settings(self, settings: dict) -> None:
        """
        Populate display widgets from profile/session settings.

        Missing settings intentionally leave the current widget values unchanged.
        Since widgets are initialized from UserSettings(), omitted profile values
        naturally fall back to config.py defaults.
        """
        self.blockSignals(True)

        widgets = (
            self.tx_vmin_slider,
            self.tx_vmax_slider,
            self.fx_vmin_slider,
            self.fx_vmax_slider,
            self.spec_vmin_slider,
            self.spec_vmax_slider,
            self.fx_win_s_spin,
            self.fx_nfft_spin,
            self.spec_nfft_spin,
            self.spec_overlap_spin,
        )

        for widget in widgets:
            widget.blockSignals(True)

        try:
            def set_level_slider(slider: QSlider, key: str) -> None:
                """
                Set a 0--100 display slider.

                Supported input forms:
                    normalized profile/config value:
                        0.4 -> slider value 40

                    current GUI/session slider value:
                        40 -> slider value 40
                """
                value = settings.get(key)

                if value is None:
                    return

                value = float(value)

                if 0.0 <= value <= 1.0:
                    slider_value = int(round(value * 100.0))
                else:
                    slider_value = int(round(value))

                slider.setValue(
                    max(
                        slider.minimum(),
                        min(slider_value, slider.maximum()),
                    )
                )

            # T-X display levels.
            set_level_slider(self.tx_vmin_slider, "tx_vmin")
            set_level_slider(self.tx_vmax_slider, "tx_vmax")

            # F-X display levels.
            set_level_slider(self.fx_vmin_slider, "fx_vmin")
            set_level_slider(self.fx_vmax_slider, "fx_vmax")

            # Spectrogram display levels.
            set_level_slider(self.spec_vmin_slider, "spec_vmin")
            set_level_slider(self.spec_vmax_slider, "spec_vmax")

            # F-X transform settings.
            if settings.get("win_s") is not None:
                self.fx_win_s_spin.setValue(
                    float(settings["win_s"])
                )

            if settings.get("fx_nfft") is not None:
                self.fx_nfft_spin.setValue(
                    int(settings["fx_nfft"])
                )

            # Spectrogram settings.
            if settings.get("spec_nfft") is not None:
                self.spec_nfft_spin.setValue(
                    int(settings["spec_nfft"])
                )

            if settings.get("spec_overlap") is not None:
                self.spec_overlap_spin.setValue(
                    int(settings["spec_overlap"])
                )

        finally:
            for widget in widgets:
                widget.blockSignals(False)

            self.blockSignals(False)

        # One intentional signal after all display controls are synchronized.
        self.display_settings_changed.emit()