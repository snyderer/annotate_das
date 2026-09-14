from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
)

from annotate.config import UserSettings


class DataSourcePanel(QWidget):
    """
    Controls needed to interpret and load raw DAS source files.

    GUI cable distances are displayed in kilometres, while returned settings
    use metres:

        cable_start_m
        cable_end_m
        target_dx_m
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout(self)
        defaults = UserSettings()

        self.interrogator_combo = QComboBox()
        self.interrogator_combo.addItem("Auto detect", "auto")
        self.interrogator_combo.addItem("ASN", "asn")
        self.interrogator_combo.addItem("OptaSense", "optasense")
        self.interrogator_combo.addItem("Onyx", "onyx")
        self.interrogator_combo.addItem("Fosina DxS", "fosina_dxs")
        self.interrogator_combo.addItem("Silixa", "silixa")

        self.use_full_cable_check = QCheckBox("Use full cable length")
        self.use_full_cable_check.setChecked(defaults.use_full_cable_length)

        # These widgets display kilometres.
        self.start_distance_spin = QDoubleSpinBox()
        self.start_distance_spin.setRange(-1_000.0, 100_000.0)
        self.start_distance_spin.setDecimals(3)
        self.start_distance_spin.setSuffix(" km")
        self.start_distance_spin.setValue(defaults.cable_start_m / 1000.0 or 0.0)

        self.end_distance_spin = QDoubleSpinBox()
        self.end_distance_spin.setRange(-1_000.0, 10_000.0)
        self.end_distance_spin.setDecimals(3)
        self.end_distance_spin.setSuffix(" km")
        self.end_distance_spin.setValue(defaults.cable_end_m / 1000.0 or 80.0)

        # This remains metres, because it is spatial sample/channel spacing.
        self.channel_spacing_spin = QDoubleSpinBox()
        self.channel_spacing_spin.setRange(1.0, 1_000.0)
        self.channel_spacing_spin.setDecimals(2)
        self.channel_spacing_spin.setSuffix(" m")
        self.channel_spacing_spin.setValue(defaults.target_dx_m or 16.0)

        layout.addRow("Interrogator", self.interrogator_combo)
        layout.addRow(self.use_full_cable_check)
        layout.addRow("Start distance", self.start_distance_spin)
        layout.addRow("End distance", self.end_distance_spin)
        layout.addRow("Channel spacing", self.channel_spacing_spin)

        self.use_full_cable_check.toggled.connect(
            self._update_distance_controls
        )

        self._update_distance_controls(
            self.use_full_cable_check.isChecked()
        )

    def _update_distance_controls(self, use_full_cable: bool) -> None:
        self.start_distance_spin.setEnabled(not use_full_cable)
        self.end_distance_spin.setEnabled(not use_full_cable)

    def get_settings(self) -> dict:
        """
        Return settings using metres internally.

        The GUI widgets display km, so values are multiplied by 1000 here.
        """
        use_full = self.use_full_cable_check.isChecked()

        cable_start_m = self.start_distance_spin.value() * 1000.0
        cable_end_m = self.end_distance_spin.value() * 1000.0

        if not use_full and cable_end_m <= cable_start_m:
            raise ValueError(
                "End distance must be greater than start distance."
            )

        return {
            "interrogator": self.interrogator_combo.currentData(),

            "use_full_cable_length": use_full,
            "cable_start_m": cable_start_m,
            "cable_end_m": cable_end_m,

            # Requested output spatial spacing, in metres.
            "target_dx_m": self.channel_spacing_spin.value(),
        }

    def set_cable_metadata(
        self,
        *,
        start_distance_m: float,
        end_distance_m: float,
        native_dx_m: float,
    ) -> None:
        """
        Update valid cable range after indexing a source dataset.

        The GUI displays km, while source metadata is metres.

        Existing requested distances are retained when possible. If they fall
        outside the cable extent, they are clamped and snapped to the nearest
        valid native channel position.
        """
        if end_distance_m <= start_distance_m:
            raise ValueError(
                "Dataset cable end distance must be greater than start distance."
            )

        if native_dx_m <= 0:
            raise ValueError(
                "Dataset native channel spacing must be positive."
            )

        # Current user/default requested positions, converted from GUI km to m.
        requested_start_m = self.start_distance_spin.value() * 1000.0
        requested_end_m = self.end_distance_spin.value() * 1000.0

        # Clamp requested distances to actual cable coverage.
        requested_start_m = min(
            max(requested_start_m, start_distance_m),
            end_distance_m,
        )
        requested_end_m = min(
            max(requested_end_m, start_distance_m),
            end_distance_m,
        )

        # Snap a physical location to closest available native channel position.
        def snap_to_channel(distance_m: float) -> float:
            channel_index = round(
                (distance_m - start_distance_m) / native_dx_m
            )

            snapped = start_distance_m + channel_index * native_dx_m

            return min(
                max(snapped, start_distance_m),
                end_distance_m,
            )

        resolved_start_m = snap_to_channel(requested_start_m)
        resolved_end_m = snap_to_channel(requested_end_m)

        # Ensure start/end remain valid after clamping/snapping.
        if resolved_end_m <= resolved_start_m:
            resolved_end_m = min(
                resolved_start_m + native_dx_m,
                end_distance_m,
            )

        # Convert physical bounds back to the km values displayed by the GUI.
        start_distance_km = start_distance_m / 1000.0
        end_distance_km = end_distance_m / 1000.0

        self.start_distance_spin.setRange(
            start_distance_km,
            end_distance_km,
        )
        self.end_distance_spin.setRange(
            start_distance_km,
            end_distance_km,
        )

        self.start_distance_spin.setValue(resolved_start_m / 1000.0)
        self.end_distance_spin.setValue(resolved_end_m / 1000.0)

        # Do not permit requested output spacing smaller than native spacing.
        self.channel_spacing_spin.setMinimum(native_dx_m)

        if self.channel_spacing_spin.value() < native_dx_m:
            self.channel_spacing_spin.setValue(native_dx_m)

    def set_interrogator(self, interrogator: str) -> None:
        """Select a known interrogator option."""
        index = self.interrogator_combo.findData(interrogator)

        if index >= 0:
            self.interrogator_combo.setCurrentIndex(index)