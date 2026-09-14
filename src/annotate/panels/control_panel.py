# annotate/panels/control_panel.py
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
)

from annotate.panels.annotation_panel import AnnotationPanel
from annotate.panels.data_source_panel import DataSourcePanel
from annotate.panels.display_settings_panel import DisplaySettingsPanel
from annotate.panels.navigation_panel import NavigationPanel
from annotate.panels.processing_panel import ProcessingPanel


class ControlPanel(QWidget):
    """
    Top-level container for application controls.

    Child panels own their local widgets and settings. ControlPanel:
    - arranges child panels;
    - merges settings;
    - forwards signals expected by MainWindow;
    - exposes small compatibility helpers for annotation workflow code.
    """

    refresh_requested = pyqtSignal()
    toggle_labels_requested = pyqtSignal(bool)
    confirm_label_requested = pyqtSignal(int)
    distance_to_cable_changed = pyqtSignal(float)
    display_settings_changed = pyqtSignal()

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.data_manager = data_manager

        layout = QVBoxLayout(self)

        # ------------------------------------------------------------------
        # Child panels
        # ------------------------------------------------------------------
        self.data_source_panel = DataSourcePanel()
        self.navigation_panel = NavigationPanel()
        self.annotation_panel = AnnotationPanel()
        self.processing_panel = ProcessingPanel()
        self.display_settings_panel = DisplaySettingsPanel()

        # Annotation is directly below Navigation for quick access.
        layout.addWidget(
            self._group("Data Source", self.data_source_panel)
        )
        layout.addWidget(
            self._group("Navigation", self.navigation_panel)
        )
        layout.addWidget(
            self._group("Annotation", self.annotation_panel)
        )
        layout.addWidget(
            self._group("Processing", self.processing_panel)
        )
        layout.addWidget(
            self._group("Display", self.display_settings_panel)
        )
        
        self.display_settings_panel.display_settings_changed.connect(
            self.display_settings_changed.emit
        )

        # ------------------------------------------------------------------
        # Apply/reload button
        # ------------------------------------------------------------------
        self.refresh_button = QPushButton("Apply Changes")
        self.refresh_button.clicked.connect(
            self.refresh_requested.emit
        )
        layout.addWidget(self.refresh_button)

        layout.addStretch()

        # ------------------------------------------------------------------
        # Compatibility aliases used by MainWindow
        # ------------------------------------------------------------------
        self.btn_back = self.navigation_panel.btn_back
        self.btn_forward = self.navigation_panel.btn_forward

        # The existing-label control lives in AnnotationPanel.
        self.toggle_labels_button = (
            self.annotation_panel.toggle_labels_button
        )

        # ------------------------------------------------------------------
        # Forward AnnotationPanel signals through ControlPanel
        # ------------------------------------------------------------------
        self.annotation_panel.confirm_label_requested.connect(
            self.confirm_label_requested.emit
        )

        self.annotation_panel.distance_to_cable_changed.connect(
            self.distance_to_cable_changed.emit
        )

        self.annotation_panel.toggle_existing_labels_requested.connect(
            self.toggle_labels_requested.emit
        )

    @staticmethod
    def _group(title: str, widget: QWidget) -> QGroupBox:
        """Put a child control widget inside a titled group box."""
        group = QGroupBox(title)

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 8, 6, 6)
        group_layout.addWidget(widget)

        return group

    # ======================================================================
    # Settings
    # ======================================================================

    def get_settings(self) -> dict:
        """Merge current settings from all child panels."""
        settings: dict = {}

        settings.update(self.data_source_panel.get_settings())
        settings.update(self.navigation_panel.get_settings())
        settings.update(self.annotation_panel.get_settings())
        settings.update(self.processing_panel.get_settings())
        settings.update(self.display_settings_panel.get_settings())

        return settings

    # ======================================================================
    # Navigation compatibility helpers
    # ======================================================================

    def selected_start_time(self):
        return self.navigation_panel.selected_start_time()

    def set_start_time(self, value) -> None:
        self.navigation_panel.set_start_time(value)

    # ======================================================================
    # Annotation compatibility helpers
    # ======================================================================

    def clear_annotation_preview(self) -> None:
        """Clear read-only annotation summary fields and slider state."""
        self.annotation_panel.clear_preview()

    def clear_distance_to_cable(self) -> None:
        self.annotation_panel.clear_distance_to_cable()

    def set_distance_to_cable_enabled(self, enabled: bool) -> None:
        self.annotation_panel.set_distance_to_cable_enabled(enabled)

    def set_distance_to_cable(self, distance_m: float) -> None:
        self.annotation_panel.set_distance_to_cable(distance_m)
