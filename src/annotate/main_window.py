# annotate/main_window.py
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QWidget,
)

from annotate.config import DEFAULT_DATASET_PATH, DEFAULT_LABEL_MAPPING, SOUND_SPEED_MPS
from annotate.data_manager import DataManager
from annotate.label_saver import LabelSaver
from annotate.panels.control_panel import ControlPanel
from annotate.panels.fx_plot_panel import FXPlotPanel
from annotate.panels.fx_series_panel import FXSeriesPanel
from annotate.panels.spectrogram_panel import SpectrogramPanel
from annotate.panels.text_display_panel import TextDisplayPanel
from annotate.panels.tx_plot_panel import TXPlotPanel


class MainWindow(QMainWindow):
    """
    Main GUI coordinator.

    Responsibilities
    ----------------
    - Construct and connect GUI panels.
    - Delegate data loading/navigation to DataManager.
    - Own annotation workflow state.
    - Coordinate interactions between T-X, F-X, spectrogram, and controls.
    - Save/display/delete annotations.

    Does not
    --------
    - Read DAS source files directly.
    - Perform signal processing directly.
    """

    # ========================================================================
    # Cursor modes
    # ========================================================================

    MODE_NORMAL = ""
    MODE_SPECTROGRAM = "s"
    MODE_ANNOTATION = "annotation"
    MODE_FX_BOXES = "fx_boxes"

    # ========================================================================
    # Annotation stages
    # ========================================================================

    STAGE_NONE = ""

    STAGE_APEX = "select_apex"
    STAGE_APEX_DONE = "apex_confirmed"

    STAGE_HYPERBOLA = "hyperbola_fitting"
    STAGE_HYPERBOLA_DONE = "hyperbola_confirmed"

    STAGE_ENDPOINT = "select_endpoint"
    STAGE_ENDPOINT_DONE = "endpoint_confirmed"

    STAGE_DIST_POINTS = "select_distance_points"
    STAGE_DIST_DONE = "distance_complete"

    STAGE_FX = "fx_boxes"
    STAGE_FX_DONE = "fx_complete"

    # ========================================================================
    # Construction
    # ========================================================================

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Annotate DAS Data")
        self.resize(1400, 900)
        self.showMaximized()

        self.data_manager = DataManager()

        self.cursor_mode = self.MODE_NORMAL
        self.annotation_stage = self.STAGE_NONE
        self.show_labels = False

        self._initialize_annotation_state()
        self._build_layout()
        self._connect_signals()
        self._create_menu()

    def _initialize_annotation_state(self) -> None:
        """Initialize temporary state for one in-progress annotation."""
        self.apex_point: tuple[float, float] | None = None
        self.apex_row_idx: int | None = None

        self.endpoint_point: tuple[float, float] | None = None

        self.distance_point_1: tuple[float, float] | None = None
        self.distance_point_2: tuple[float, float] | None = None

        self.dist_min: float | None = None
        self.dist_max: float | None = None

        self.distance_to_cable: float | None = None
        self.hyperbola_active = False

        self.active_fx_slice_indices: list[int] = []
        self.f_min: float | None = None
        self.f_max: float | None = None

        # Straight-cable acoustic approximation.
        self.sound_speed = SOUND_SPEED_MPS

        # Maximum temporal difference between click and predicted hyperbola.
        self.hyperbola_click_tolerance_s = 1.0

    def _build_layout(self) -> None:
        """Build the control, plot, and text-panel layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(5)

        self.horizontal_splitter = QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        outer_layout.addWidget(self.horizontal_splitter)

        # --------------------------------------------------------------------
        # Left: scrollable control panel
        # --------------------------------------------------------------------
        self.control_panel = ControlPanel(self.data_manager)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(self.control_panel)
        control_scroll.setMinimumWidth(280)
        control_scroll.setMaximumWidth(430)

        self.horizontal_splitter.addWidget(control_scroll)

        # --------------------------------------------------------------------
        # Centre: T-X / F-X above spectrogram / information display
        # --------------------------------------------------------------------
        middle_splitter = QSplitter(QtCore.Qt.Orientation.Vertical)

        top_splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.tx_plot_panel = TXPlotPanel(self.data_manager)
        self.fx_plot_panel = FXPlotPanel(self.data_manager)

        top_splitter.addWidget(self.tx_plot_panel)
        top_splitter.addWidget(self.fx_plot_panel)

        middle_splitter.addWidget(top_splitter)

        bottom_splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.spectrogram_panel = SpectrogramPanel(self.data_manager)
        bottom_splitter.addWidget(self.spectrogram_panel)

        self.text_display_panel = TextDisplayPanel()

        text_scroll = QScrollArea()
        text_scroll.setWidgetResizable(True)
        text_scroll.setWidget(self.text_display_panel)

        bottom_splitter.addWidget(text_scroll)

        middle_splitter.addWidget(bottom_splitter)

        # Keep T-X / F-X widths synchronized with spectrogram / text widths.
        top_splitter.splitterMoved.connect(
            lambda _position, _index: bottom_splitter.setSizes(
                top_splitter.sizes()
            )
        )
        bottom_splitter.splitterMoved.connect(
            lambda _position, _index: top_splitter.setSizes(
                bottom_splitter.sizes()
            )
        )

        self.horizontal_splitter.addWidget(middle_splitter)

        # --------------------------------------------------------------------
        # Right: F-X thumbnail series
        # --------------------------------------------------------------------
        self.fx_series_panel = FXSeriesPanel(self.data_manager)
        self.fx_series_panel.setMinimumWidth(300)
        self.fx_series_panel.setMaximumWidth(300)

        self.horizontal_splitter.addWidget(self.fx_series_panel)

        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 4)
        self.horizontal_splitter.setStretchFactor(2, 0)

    def _connect_signals(self) -> None:
        """Connect DataManager, panel, control, and annotation signals."""

        # ------------------------------------------------------------------
        # DataManager -> GUI
        # ------------------------------------------------------------------
        self.data_manager.file_loaded.connect(
            self.text_display_panel.update_file_info
        )
        self.data_manager.dataset_opened.connect(
            self.on_dataset_opened
        )
        self.data_manager.dataset_opened.connect(
            self.text_display_panel.update_dataset_metadata
        )
        self.data_manager.window_changed.connect(
            self.control_panel.set_start_time
        )

        # ------------------------------------------------------------------
        # Plot interactions
        # ------------------------------------------------------------------
        self.fx_series_panel.slice_selected.connect(
            self.on_fx_slice_selected
        )

        # Link distance/cable Y axes between large T-X and F-X views.
        self.fx_plot_panel.plot_widget.getPlotItem().setYLink(
            self.tx_plot_panel.plot_widget
        )

        # T-X clicks are used for apex, endpoint, distance-point, and
        # spectrogram-row selection.
        self.tx_plot_panel.point_clicked.connect(
            self.on_point_clicked
        )

        # Current F-X panel only emits point_clicked for spectrogram row
        # selection, which is fine.
        self.fx_plot_panel.point_clicked.connect(
            self.on_point_clicked
        )

        # Ctrl-click near existing saved T-X apex.
        self.tx_plot_panel.label_delete_requested.connect(
            self.on_delete_label
        )

        # F-X ROI creation/deletion/movement refreshes thumbnail borders.
        self.fx_plot_panel.roi_changed.connect(
            self.on_fx_roi_changed
        )

        # Make apex and endpoints draggable in T-X panel. 
        self.tx_plot_panel.apex_dragged.connect(
            self.on_apex_dragged
        )
        self.tx_plot_panel.endpoint_dragged.connect(
            self.on_endpoint_dragged
        )

        self.fx_plot_panel.distance_boundary_dragged.connect(
            self.on_fx_distance_boundary_dragged
        )
        
        # ------------------------------------------------------------------
        # Navigation / settings
        # ------------------------------------------------------------------
        self.control_panel.btn_back.clicked.connect(
            self.navigate_backward
        )
        self.control_panel.btn_forward.clicked.connect(
            self.navigate_forward
        )
        self.control_panel.refresh_requested.connect(
            self.on_apply_changes
        )
        
        self.control_panel.display_settings_changed.connect(
            self.on_display_settings_changed
        )

        # ------------------------------------------------------------------
        # Annotation controls
        # ------------------------------------------------------------------
        self.control_panel.toggle_labels_requested.connect(
            self.on_toggle_labels
        )
        self.control_panel.confirm_label_requested.connect(
            self.save_annotation
        )
        self.control_panel.distance_to_cable_changed.connect(
            self.on_distance_to_cable_changed
        )

    def _create_menu(self) -> None:
        """Create application menus."""
        file_menu = self.menuBar().addMenu("File")

        open_dataset_action = file_menu.addAction(
            "Open DAS Dataset File..."
        )
        open_dataset_action.triggered.connect(
            self.select_dataset
        )

        # Add legacy preprocessed support later as a separate DataManager
        # implementation. Do not expose a currently broken menu item.
        #
        # open_preprocessed_action = file_menu.addAction(
        #     "Open Legacy Preprocessed Dataset..."
        # )

    # ========================================================================
    # Dataset opening, navigation, and processing settings
    # ========================================================================

    def select_dataset(self) -> None:
        """Choose and open a raw DAS source file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select DAS Dataset File",
            DEFAULT_DATASET_PATH,
            "DAS Files (*.h5 *.hdf5 *.tdms);;All Files (*)",
        )

        if not filepath:
            return

        try:
            settings = self.control_panel.get_settings()

            self.data_manager.open_raw_dataset(
                filepath=filepath,
                settings=settings,
            )

            self._configure_label_saver(settings)

            self.control_panel.set_start_time(
                self.data_manager.current_start_time
            )

            self.statusBar().showMessage(
                f"Opened {Path(filepath).name}"
            )

        except Exception as exc:
            self._show_exception(
                "Could not open dataset",
                exc,
            )

    def on_dataset_opened(self, record) -> None:
        """Update source controls after indexing selected raw DAS dataset."""
        metadata = record.metadata

        self.control_panel.data_source_panel.set_cable_metadata(
            start_distance_m=float(
                metadata.get("start_distance_m", 0.0)
            ),
            end_distance_m=float(
                metadata.get(
                    "end_distance_m",
                    record.n_channels * record.dx_m,
                )
            ),
            native_dx_m=float(record.dx_m),
        )

    def on_apply_changes(self) -> None:
        """
        Apply settings.

        Re-load source data only when spatial/time-window/boundary settings
        change. Reprocess the existing raw window for filter/display changes.
        """
        try:
            settings = self.control_panel.get_settings()
            old_settings = self.data_manager.get_user_settings()

            self.data_manager.apply_user_settings(settings)
            self._configure_label_saver(settings)

            raw_reload_keys = {
                "duration_s",
                "use_full_cable_length",
                "cable_start_m",
                "cable_end_m",
                "target_dx_m",
                "file_boundary_correction_enabled",
                "file_boundary_edge_duration_s",
            }

            raw_reload_needed = any(
                old_settings.get(key) != settings.get(key)
                for key in raw_reload_keys
            )

            if raw_reload_needed:
                self.data_manager.load_current_window(
                    recompute_fx=True
                )
            else:
                self.data_manager.reprocess_current_window(
                    recompute_fx=True
                )

            self.statusBar().showMessage("Settings applied.")

        except Exception as exc:
            self._show_exception(
                "Could not apply settings",
                exc,
            )

    def on_display_settings_changed(self) -> None:
        """
        Apply contrast/appearance changes without modifying data.
        """
        settings = self.control_panel.get_settings()

        # Update DataManager's current settings dictionary, but avoid forcing
        # reprocessing. This lets plot panels read current slider values.
        self.data_manager.apply_user_settings(
            settings,
            emit_signal=False,
        )

        self.tx_plot_panel.update_settings()
        self.tx_plot_panel.update_plot()

        self.fx_plot_panel.update_settings()

        if self.fx_plot_panel.current_slice_idx is not None:
            self.fx_plot_panel.show_slice(
                self.fx_plot_panel.current_slice_idx
            )

        self.fx_series_panel.update_settings()

        dataset = self.data_manager.fx_manager.get_dataset()
        self.fx_series_panel.set_plot_data(dataset)

        if getattr(self.spectrogram_panel, "last_row_idx", None) is not None:
            self.spectrogram_panel.update_settings()
            self.spectrogram_panel.update_plot(
                self.spectrogram_panel.last_row_idx
            )

    def navigate_backward(self) -> None:
        self._navigate("backward")

    def navigate_forward(self) -> None:
        self._navigate("forward")

    def _navigate(self, direction: str) -> None:
        """Navigate one configured time step."""
        try:
            self.data_manager.navigate(direction)
        except Exception as exc:
            self._show_exception(
                "Navigation failed",
                exc,
            )

    def _configure_label_saver(self, settings: dict) -> None:
        """Create/update the annotation repository for the configured path."""
        labels_path = settings.get("labels_file_path", "").strip()

        if not labels_path:
            return

        current_saver = self.data_manager.label_saver

        if (
            current_saver is None
            or getattr(current_saver, "csv_path", None)
            != labels_path
        ):
            self.data_manager.label_saver = LabelSaver(labels_path)

    # ========================================================================
    # General GUI / annotation state helpers
    # ========================================================================

    def notify(
        self,
        *,
        cursor_mode: str | None = None,
        prompt: str | None = None,
        status: str | None = None,
    ) -> None:
        """Update cursor mode text, annotation prompt, and status bar."""
        if cursor_mode is not None:
            self.text_display_panel.update_cursor_mode(cursor_mode)

        if prompt is not None:
            if hasattr(self.text_display_panel, "set_annotation_prompt"):
                self.text_display_panel.set_annotation_prompt(prompt)
            else:
                self.text_display_panel.update_info_text(prompt)

        if status is not None:
            self.statusBar().showMessage(status)

    def set_mode(
        self,
        mode: str,
        stage: str | None = None,
        *,
        tx_annotation_active: bool = False,
    ) -> None:
        """Set interaction mode/stage consistently."""
        self.cursor_mode = mode
        self.data_manager.set_cursor_mode(mode)
        self.tx_plot_panel.annotation_mode_active = tx_annotation_active

        if stage is not None:
            self.annotation_stage = stage

    def reset_annotation(
        self,
        *,
        clear_control_panel: bool = True,
    ) -> None:
        """Clear temporary annotation state and all unsaved overlays."""
        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_NONE,
            tx_annotation_active=False,
        )

        self._initialize_annotation_state()
        self.clear_all_annotation_overlays()

        if clear_control_panel:
            self.control_panel.clear_annotation_preview()

        self.control_panel.clear_distance_to_cable()

        if hasattr(
            self.fx_series_panel,
            "clear_annotation_slice_highlights",
        ):
            self.fx_series_panel.clear_annotation_slice_highlights()

    def clear_all_annotation_overlays(self) -> None:
        """Remove all temporary T-X/F-X annotation graphics."""
        self.tx_plot_panel.clear_annotation_overlays()
        self.fx_plot_panel.clear_annotation_overlays()
        self.fx_series_panel.clear_annotation_overlays()

        if hasattr(self.data_manager, "annotation_rois_per_slice"):
            self.data_manager.annotation_rois_per_slice.clear()

    def apex_time_to_utc_text(
        self,
        time_offset_s: float,
    ) -> str:
        """
        Format a current-window-relative T-X time as an absolute UTC timestamp.
        """
        window_start_utc = float(
            self.data_manager.loaded_data["time_stamps"][0]
        )

        apex_time_utc = window_start_utc + float(time_offset_s)

        return datetime.fromtimestamp(
            apex_time_utc,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"

    def _show_exception(
        self,
        context: str,
        exc: Exception,
    ) -> None:
        """Display a concise GUI failure and print full traceback for debug."""
        self.statusBar().showMessage(f"{context}: {exc}")

        import traceback
        traceback.print_exc()

    # ========================================================================
    # Plot interactions
    # ========================================================================

    def on_fx_slice_selected(self, idx: int) -> None:
        """Show selected F-X thumbnail in the main F-X panel."""
        self.fx_plot_panel.show_slice_from_series(idx)
        self.fx_series_panel.highlight_slice(idx)

        dataset = self.data_manager.fx_manager.get_dataset()
        slice_times = dataset.get("t")

        if slice_times is None or idx >= len(slice_times):
            return

        win_s = float(
            self.data_manager.get_user_settings("win_s") or 2.0
        )

        start_s = float(slice_times[idx])
        end_s = start_s + win_s

        self.tx_plot_panel.highlight_time_window(start_s, end_s)

        if self.spectrogram_panel.last_row_idx is not None:
            self.spectrogram_panel.highlight_time_window(
                start_s,
                end_s,
            )

    def on_fx_roi_changed(self, slice_idx: int) -> None:
        """Update thumbnail border state after an F-X ROI changes."""
        self.fx_series_panel.refresh_annotation_slice_styles()

        boxes = self.data_manager.annotation_rois_per_slice.get(
            slice_idx,
            [],
        )

        if boxes:
            self.statusBar().showMessage(
                f"F-X slice {slice_idx}: {len(boxes)} box(es)."
            )
        else:
            self.statusBar().showMessage(
                f"F-X slice {slice_idx}: no boxes."
            )
            
    def on_point_clicked(
        self,
        row_idx: int,
        col_idx: int,
        clicked_distance: float | None = None,
    ) -> None:
        """Dispatch a T-X/F-X click according to current interaction state."""
        if self.cursor_mode == self.MODE_SPECTROGRAM:
            self.select_spectrogram_row(row_idx)
            return

        if self.cursor_mode != self.MODE_ANNOTATION:
            return

        t_val = float(self.data_manager.loaded_data["t"][col_idx])
        x_val = float(self.data_manager.loaded_data["x"][row_idx])

        if self.annotation_stage == self.STAGE_APEX:
            self.select_apex(
                row_idx=row_idx,
                t_val=t_val,
                x_val=x_val,
            )

        elif self.annotation_stage == self.STAGE_ENDPOINT:
            self.select_endpoint(
                row_idx=row_idx,
                t_val=t_val,
                x_val=x_val,
                clicked_distance=clicked_distance,
            )

        elif self.annotation_stage == self.STAGE_DIST_POINTS:
            self.select_distance_point_on_hyperbola(
                clicked_time=t_val,
                clicked_distance=x_val,
            )

    def start_spectrogram_selection(self) -> None:
        """Enter row-selection mode for the spectrogram."""
        self.set_mode(self.MODE_SPECTROGRAM)

        self.notify(
            cursor_mode="Select spectrogram row",
            prompt="Click a T-X or F-X plot to select a spectrogram row.",
            status="Spectrogram row selection active.",
        )

    def select_spectrogram_row(self, row_idx: int) -> None:
        """Update spectrogram from the selected cable-distance row."""
        distance_m = float(
            self.data_manager.loaded_data["x"][row_idx]
        )

        self.spectrogram_panel.update_plot(row_idx)
        self.tx_plot_panel.mark_distance(distance_m)
        self.fx_plot_panel.mark_distance(distance_m)

        self.set_mode(self.MODE_NORMAL)

        self.notify(
            cursor_mode="Normal",
            prompt=(
                "Spectrogram row selected.\n"
                f"Distance: {distance_m:.2f} m"
            ),
            status=f"Spectrogram updated for row {row_idx}.",
        )
        
    def on_apex_dragged(
        self,
        new_time: float,
        new_distance: float,
    ) -> None:
        """
        Update apex state while its marker is dragged.

        The apex remains continuous while editing. It is snapped to the nearest
        real DAS channel only when H confirms distance-to-cable fitting.
        """
        if self.apex_point is None:
            return

        new_time = float(new_time)
        new_distance = float(new_distance)

        self.apex_point = (new_time, new_distance)

        duration_s = None

        # Endpoint remains tied to the current apex distance.
        if self.endpoint_point is not None:
            endpoint_time = float(self.endpoint_point[0])

            self.endpoint_point = (
                endpoint_time,
                new_distance,
            )

            self.tx_plot_panel.set_endpoint_position(
                endpoint_time,
                new_distance,
            )

            duration_s = abs(endpoint_time - new_time)

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": new_distance,
            "duration": duration_s,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "f_min": self.f_min,
            "f_max": self.f_max,
        })
        
        apex_time_s = (
            float(self.apex_point[0])
            if self.apex_point is not None
            else None
        )
                
        self.control_panel.annotation_panel.apex_time_display.setText(
            self.apex_time_to_utc_text(apex_time_s)
        )

        if self.distance_to_cable is not None:
            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

            if self.endpoint_point is not None:
                self.update_hyperbola(
                    anchor="endpoint",
                    curve_name="endpoint",
                )

    def on_endpoint_dragged(
        self,
        new_time: float,
        new_distance: float,
    ) -> None:
        """
        Update endpoint state after dragging.

        Endpoint distance is always constrained by TXPlotPanel to the current
        apex channel. Only endpoint time is truly editable.
        """
        if self.apex_point is None or self.endpoint_point is None:
            return

        apex_time = float(self.apex_point[0])
        apex_distance = float(self.apex_point[1])

        new_time = float(new_time)

        # Keep endpoint tied to the current apex channel.
        self.endpoint_point = (
            new_time,
            apex_distance,
        )

        self.tx_plot_panel.set_endpoint_position(
            new_time,
            apex_distance,
        )

        duration_s = abs(new_time - apex_time)

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": apex_distance,
            "duration": duration_s,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "f_min": self.f_min,
            "f_max": self.f_max,
        })
        
        apex_time_s = (
            float(self.apex_point[0])
            if self.apex_point is not None
            else None
        )
        
        self.control_panel.annotation_panel.apex_time_display.setText(
            self.apex_time_to_utc_text(apex_time_s)
        )

        if self.distance_to_cable is not None:
            self.update_hyperbola(
                anchor="endpoint",
                curve_name="endpoint",
            )

    def on_fx_distance_boundary_dragged(
        self,
        boundary_name: str,
        requested_distance_m: float,
    ) -> None:
        """
        Update one selected distance boundary from a dragged F-X line.

        The new boundary is snapped to the nearest loaded DAS channel and then
        projected onto the apex hyperbola, preserving annotation geometry.
        """
        if self.annotation_stage not in {
            self.STAGE_DIST_POINTS,
            self.STAGE_DIST_DONE,
            self.STAGE_FX,
            self.STAGE_FX_DONE,
        }:
            return

        x = self.data_manager.loaded_data.get("x")

        if x is None or len(x) == 0:
            return

        row_idx = int(
            np.argmin(np.abs(x - requested_distance_m))
        )

        snapped_distance_m = float(x[row_idx])

        time_s, x_values = self.calculate_hyperbola(anchor="apex")

        if time_s is None or x_values is None:
            return

        snapped_time_s = float(time_s[row_idx])
        new_point = (
            snapped_time_s,
            snapped_distance_m,
        )

        # Keep independently tracked lower/upper boundaries.
        if boundary_name == "min":
            if self.distance_point_1 is None:
                self.distance_point_1 = new_point
            elif self.distance_point_2 is None:
                self.distance_point_2 = new_point
            else:
                # Replace whichever current point has the smaller distance.
                if self.distance_point_1[1] <= self.distance_point_2[1]:
                    self.distance_point_1 = new_point
                else:
                    self.distance_point_2 = new_point

        elif boundary_name == "max":
            if self.distance_point_1 is None:
                self.distance_point_1 = new_point
            elif self.distance_point_2 is None:
                self.distance_point_2 = new_point
            else:
                # Replace whichever current point has the larger distance.
                if self.distance_point_1[1] >= self.distance_point_2[1]:
                    self.distance_point_1 = new_point
                else:
                    self.distance_point_2 = new_point

        self._refresh_distance_range_geometry()

    def clear_call_geometry(self) -> None:
        """Remove all temporary hyperbola overlays for the current annotation."""
        self.tx_plot_panel.clear_hyperbola("apex")
        self.tx_plot_panel.clear_hyperbola("endpoint")
        self.tx_plot_panel.clear_hyperbola("confirmed")

    def _refresh_distance_range_geometry(self) -> None:
        """
        Recalculate min/max distance state and update T-X/F-X overlays.

        Call this after selecting, dragging, or otherwise modifying either
        distance-side point.
        """
        points = [
            point
            for point in (
                self.distance_point_1,
                self.distance_point_2,
            )
            if point is not None
        ]

        self.tx_plot_panel.set_distance_annotation_points(points)

        if len(points) < 2:
            return

        distances = [float(point[1]) for point in points]

        self.dist_min = min(distances)
        self.dist_max = max(distances)

        self.tx_plot_panel.set_distance_boundaries(
            self.dist_min,
            self.dist_max,
        )

        self.fx_plot_panel.set_distance_boundaries(
            self.dist_min,
            self.dist_max,
            movable=(
                self.annotation_stage in {
                    self.STAGE_DIST_POINTS,
                    self.STAGE_DIST_DONE,
                    self.STAGE_FX,
                    self.STAGE_FX_DONE,
                }
            ),
        )

        apex_time_s = (
            float(self.apex_point[0])
            if self.apex_point is not None
            else None
        )

        apex_distance_m = (
            float(self.apex_point[1])
            if self.apex_point is not None
            else None
        )

        duration_s = None

        if self.endpoint_point is not None and apex_time_s is not None:
            duration_s = abs(
                float(self.endpoint_point[0]) - apex_time_s
            )

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": apex_distance_m,
            "duration": duration_s,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "f_min": self.f_min,
            "f_max": self.f_max,
        })

        if apex_time_s is not None:
            self.control_panel.annotation_panel.apex_time_display.setText(
                self.apex_time_to_utc_text(apex_time_s)
            )

    # ========================================================================
    # Keyboard shortcut dispatch
    # ========================================================================

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if key == QtCore.Qt.Key.Key_S:
            self.start_spectrogram_selection()

        elif key == QtCore.Qt.Key.Key_A:
            self.handle_apex_key()

        elif key == QtCore.Qt.Key.Key_H:
            self.handle_hyperbola_key()

        elif key == QtCore.Qt.Key.Key_E:
            self.handle_endpoint_key()

        elif key == QtCore.Qt.Key.Key_D:
            self.handle_distance_key()

        elif key == QtCore.Qt.Key.Key_F:
            if self.cursor_mode == self.MODE_FX_BOXES:
                self.finish_fx_box_labeling()
            else:
                self.start_fx_box_labeling()

        elif key == QtCore.Qt.Key.Key_Space:
            if self.cursor_mode == self.MODE_FX_BOXES:
                self.select_next_fx_slice()
            else:
                super().keyPressEvent(event)

        elif QtCore.Qt.Key.Key_0 <= key <= QtCore.Qt.Key.Key_9:
            self.handle_label_shortcut(
                key - QtCore.Qt.Key.Key_0
            )

        elif key == QtCore.Qt.Key.Key_Escape:
            self.reset_annotation()

            self.notify(
                cursor_mode="Normal",
                prompt="Annotation cancelled.",
                status="Normal mode.",
            )

        else:
            self.distance_point_1 = self.distance_point_2
            self.distance_point_2 = new_point

    def handle_label_shortcut(self, label_num: int) -> None:
        """Select label in UI and attempt to save the current annotation."""
        annotation_panel = self.control_panel.annotation_panel

        index = annotation_panel.label_dropdown.findData(label_num)

        if index >= 0:
            annotation_panel.label_dropdown.setCurrentIndex(index)

        if label_num == 0:
            self.notify(
                status="Label 0 is reserved; annotation was not saved."
            )
            return

        self.save_annotation(label_num)

    # ========================================================================
    # Apex workflow: A -> select apex -> A confirm
    # ========================================================================

    def handle_apex_key(self) -> None:
        """
        A workflow:

        A -> enter apex selection mode
        A -> confirm apex selection
        """
        if self.cursor_mode != self.MODE_ANNOTATION:
            self.reset_annotation()

            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_APEX,
                tx_annotation_active=True,
            )

            self.notify(
                cursor_mode="Apex labeling",
                prompt=(
                    "Apex labeling:\n\n"
                    "Click the apex in the T-X plot.\n"
                    "Drag the red apex marker to refine its position.\n"
                    "Click again to replace it.\n\n"
                    "Press A to confirm the apex."
                ),
                status="Select apex.",
            )
            return

        if self.annotation_stage != self.STAGE_APEX:
            return

        if self.apex_point is None:
            self.notify(
                prompt="No apex selected yet. Click an apex first.",
                status="Select an apex first.",
            )
            return

        # Apex remains movable through hyperbola fitting. It is snapped and
        # locked only after the second H confirms distance-to-cable.
        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_APEX_DONE,
            tx_annotation_active=False,
        )

        self.notify(
            cursor_mode="Apex confirmed",
            prompt=(
                "Apex confirmed.\n\n"
                "Press H to fit the hyperbola and select "
                "distance to cable."
            ),
            status="Apex confirmed.",
        )

    def select_apex(
        self,
        *,
        row_idx: int,
        t_val: float,
        x_val: float,
    ) -> None:
        """Store a new apex and clear geometry derived from any prior apex."""
        self.apex_row_idx = row_idx
        self.apex_point = (float(t_val), float(x_val))

        # A changed apex invalidates all downstream annotation geometry.
        self.endpoint_point = None

        self.distance_point_1 = None
        self.distance_point_2 = None
        self.dist_min = None
        self.dist_max = None

        self.f_min = None
        self.f_max = None

        self.clear_call_geometry()

        self.tx_plot_panel.clear_endpoint_point()
        self.tx_plot_panel.set_distance_annotation_points([])
        self.tx_plot_panel.clear_distance_boundaries()

        self.fx_plot_panel.clear_distance_boundaries()

        self.tx_plot_panel.mark_apex_point(t_val, x_val)

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": x_val,
            "duration": None,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": None,
            "dist_max": None,
            "f_min": None,
            "f_max": None,
        })

        self.control_panel.annotation_panel.apex_time_display.setText(
            self.apex_time_to_utc_text(t_val)
        )

        self.notify(
            prompt=(
                "Apex selected:\n\n"
                f"Time offset: {t_val:.3f} s\n"
                f"Distance: {x_val:.2f} m\n\n"
                "Click again to replace it.\n"
                "Press A to confirm apex."
            ),
            status="Apex updated.",
        )

    # ========================================================================
    # Hyperbola workflow: H -> adjust slider -> H confirm
    # ========================================================================

    def handle_hyperbola_key(self) -> None:
        if self.annotation_stage != self.STAGE_HYPERBOLA:
            if self.annotation_stage != self.STAGE_APEX_DONE:
                self.notify(
                    prompt=(
                        "Confirm the apex before hyperbola fitting.\n\n"
                        "Press A to select and confirm the apex."
                    ),
                    status="Apex confirmation is required first.",
                )
                return

            self.hyperbola_active = True
            self.annotation_stage = self.STAGE_HYPERBOLA

            if self.distance_to_cable is None:
                self.distance_to_cable = 0.0

            self.control_panel.set_distance_to_cable_enabled(True)
            self.control_panel.set_distance_to_cable(
                self.distance_to_cable
            )

            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

            self.notify(
                cursor_mode="Hyperbola fitting",
                prompt=(
                    "Hyperbola fitting mode.\n\n"
                    "Adjust Distance to cable using the slider.\n\n"
                    "Press H to confirm the fit."
                ),
                status="Adjust distance to cable.",
            )
            return

        self.hyperbola_active = False
        self.annotation_stage = self.STAGE_HYPERBOLA_DONE

        self.control_panel.set_distance_to_cable_enabled(False)
        self.snap_apex_and_endpoint_to_channel()
        self.tx_plot_panel.set_apex_movable(False)

        self.notify(
            cursor_mode="Distance to cable confirmed",
            prompt=(
                "Distance to cable confirmed.\n\n"
                f"Distance: {self.distance_to_cable:.1f} m\n\n"
                "Press E to label the endpoint."
            ),
            status="Distance to cable confirmed.",
        )

    def on_distance_to_cable_changed(self, distance_m: float) -> None:
        self.distance_to_cable = float(distance_m)

        if self.hyperbola_active:
            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

    def calculate_hyperbola(
        self,
        *,
        anchor: str = "apex",
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Calculate a straight-cable hyperbola for apex or endpoint."""
        anchor_point = (
            self.endpoint_point
            if anchor == "endpoint"
            else self.apex_point
        )

        if anchor_point is None or self.distance_to_cable is None:
            return None, None

        x = self.data_manager.loaded_data.get("x")

        if x is None or len(x) == 0:
            return None, None

        anchor_time, anchor_distance = map(float, anchor_point)
        distance_to_cable = float(self.distance_to_cable)

        range_m = np.sqrt(
            (x - anchor_distance) ** 2
            + distance_to_cable ** 2
        )

        time_s = anchor_time + (
            range_m - distance_to_cable
        ) / self.sound_speed

        return time_s, x

    def update_hyperbola(
        self,
        *,
        anchor: str = "apex",
        curve_name: str | None = None,
    ) -> None:
        """Calculate and draw a named hyperbola overlay."""
        time_s, distance_m = self.calculate_hyperbola(anchor=anchor)

        if time_s is None or distance_m is None:
            return

        curve_name = curve_name or anchor

        self.tx_plot_panel.show_hyperbola(
            time_values=time_s,
            distance_values=distance_m,
            curve_name=curve_name,
            color="white" if anchor == "apex" else "cyan",
            style=QtCore.Qt.PenStyle.DashLine,
            width=2,
        )

    def snap_apex_and_endpoint_to_channel(self) -> None:
        """Snap apex and endpoint to nearest loaded physical DAS channel."""
        if self.apex_point is None:
            return

        x = self.data_manager.loaded_data.get("x")

        if x is None or len(x) == 0:
            return

        apex_time, apex_distance = map(float, self.apex_point)

        self.apex_row_idx = int(
            np.argmin(np.abs(x - apex_distance))
        )

        snapped_distance = float(x[self.apex_row_idx])

        self.apex_point = (apex_time, snapped_distance)

        self.tx_plot_panel.mark_apex_point(
            apex_time,
            snapped_distance,
        )

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": snapped_distance,
            "duration": None,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": None,
            "dist_max": None,
            "f_min": None,
            "f_max": None,
        })

        if self.endpoint_point is not None:
            endpoint_time = float(self.endpoint_point[0])

            self.endpoint_point = (
                endpoint_time,
                snapped_distance,
            )

            self.tx_plot_panel.mark_endpoint_point(
                endpoint_time,
                snapped_distance,
            )

    # ========================================================================
    # Endpoint workflow: E -> select endpoint -> E confirm
    # ========================================================================

    def handle_endpoint_key(self) -> None:
        if self.annotation_stage == self.STAGE_HYPERBOLA_DONE:
            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_ENDPOINT,
                tx_annotation_active=True,
            )

            self.tx_plot_panel.set_endpoint_movable(True)
            
            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

            self.notify(
                cursor_mode="Endpoint labeling",
                prompt=(
                    "Endpoint labeling:\n\n"
                    "Click near the endpoint on the apex channel.\n"
                    "The endpoint snaps to the apex channel.\n\n"
                    "Press E to confirm the endpoint."
                ),
                status="Select endpoint near apex channel.",
            )
            return

        if self.annotation_stage != self.STAGE_ENDPOINT:
            self.notify(
                prompt=(
                    "Confirm distance-to-cable before endpoint labeling.\n\n"
                    "Press H to fit and confirm the hyperbola."
                ),
                status="Hyperbola confirmation is required.",
            )
            return

        if self.endpoint_point is None:
            self.notify(
                prompt="No endpoint selected yet. Click an endpoint first.",
                status="Select an endpoint first.",
            )
            return

        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_ENDPOINT_DONE,
            tx_annotation_active=False,
        )
        
        self.tx_plot_panel.set_endpoint_movable(False)

        self.update_hyperbola(
            anchor="endpoint",
            curve_name="endpoint",
        )

        self.notify(
            cursor_mode="Endpoint confirmed",
            prompt=(
                "Endpoint confirmed.\n\n"
                "Press D to select two distance-side points."
            ),
            status="Endpoint confirmed.",
        )

    def select_endpoint(
        self,
        *,
        row_idx: int,
        t_val: float,
        x_val: float,
        clicked_distance: float | None = None,
    ) -> None:
        """Store endpoint time while constraining distance to apex channel."""
        if self.apex_point is None:
            return

        apex_time, apex_distance = map(float, self.apex_point)
        clicked_distance = (
            float(clicked_distance)
            if clicked_distance is not None
            else x_val
        )

        dx_m = float(
            self.data_manager.loaded_segment.dx_m
            if self.data_manager.loaded_segment is not None
            else 1.0
        )

        # Permit clicks within roughly 100 selected channels.
        tolerance_m = 100.0 * dx_m

        if abs(clicked_distance - apex_distance) > tolerance_m:
            self.notify(
                prompt=(
                    "Endpoint click is too far from apex channel.\n\n"
                    f"Apex: {apex_distance:.2f} m\n"
                    f"Clicked: {clicked_distance:.2f} m"
                ),
                status="Endpoint must be near apex channel.",
            )
            return

        self.endpoint_point = (float(t_val), apex_distance)

        self.tx_plot_panel.mark_endpoint_point(
            float(t_val),
            apex_distance,
        )

        duration_s = abs(float(t_val) - apex_time)

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": apex_distance,
            "duration": duration_s,
            "distance_to_cable": self.distance_to_cable,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "f_min": self.f_min,
            "f_max": self.f_max,
        })

        if self.distance_to_cable is not None:
            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )
            self.update_hyperbola(
                anchor="endpoint",
                curve_name="endpoint",
            )

        self.notify(
            prompt=(
                "Endpoint selected.\n\n"
                f"Time: {t_val:.3f} s\n"
                f"Distance: {apex_distance:.2f} m\n"
                f"Duration: {duration_s:.3f} s\n\n"
                "Click again to replace it, or press E to confirm."
            ),
            status="Endpoint updated.",
        )

    # ========================================================================
    # Distance-point workflow: D -> two points -> D confirm
    # ========================================================================

    def handle_distance_key(self) -> None:
        if self.annotation_stage not in {
            self.STAGE_ENDPOINT_DONE,
            self.STAGE_DIST_POINTS,
            self.STAGE_DIST_DONE,
        }:
            self.notify(
                prompt=(
                    "Confirm endpoint before distance labeling.\n\n"
                    "Press E to select and confirm endpoint."
                ),
                status="Endpoint confirmation required.",
            )
            return

        if self.annotation_stage in {
            self.STAGE_ENDPOINT_DONE,
            self.STAGE_DIST_DONE,
        }:
    
            self.tx_plot_panel.clear_hyperbola("confirmed")
            self.tx_plot_panel.set_distance_annotation_points([])
            self.tx_plot_panel.clear_distance_boundaries()
            self.fx_plot_panel.clear_distance_boundaries()
            
            self.distance_point_1 = None
            self.distance_point_2 = None
            self.dist_min = None
            self.dist_max = None
            

            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_DIST_POINTS,
                tx_annotation_active=True,
            )

            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

            self.update_hyperbola(
                anchor="endpoint",
                curve_name="endpoint",
            )

            self.notify(
                cursor_mode="Distance labeling",
                prompt=(
                    "Click two points near the white apex hyperbola.\n\n"
                    "Points will snap to the hyperbola.\n"
                    "Press D when finished."
                ),
                status="Select two distance-side points.",
            )
            return

        if (
            self.distance_point_1 is None
            or self.distance_point_2 is None
        ):
            self.notify(
                prompt="Select two distance-side points first.",
                status="Two points required.",
            )
            return

        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_DIST_DONE,
            tx_annotation_active=False,
        )

        self.update_hyperbola(
            anchor="apex",
            curve_name="apex",
        )

        self.update_hyperbola(
            anchor="endpoint",
            curve_name="endpoint",
        )

        self.notify(
            cursor_mode="Distance label complete",
            prompt=(
                "Distance range complete.\n\n"
                f"Minimum: {self.dist_min:.2f} m\n"
                f"Maximum: {self.dist_max:.2f} m\n\n"
                "Review the white apex and cyan endpoint hyperbolas, "
                "plus yellow distance boundaries.\n\n"
                "Press F for F-X box labeling, or select a label "
                "and press its number to save."
            ),
            status="Distance range confirmed.",
        )

    def select_distance_point_on_hyperbola(
        self,
        *,
        clicked_time: float,
        clicked_distance: float,
    ) -> None:
        """Snap selected T-X point to the apex hyperbola."""
        snapped = self.snap_point_to_hyperbola(
            clicked_time=clicked_time,
            clicked_distance=clicked_distance,
        )

        if snapped is None:
            self.notify(
                prompt=(
                    "Click closer to the white apex hyperbola.\n\n"
                    f"Allowed time tolerance: "
                    f"±{self.hyperbola_click_tolerance_s:.2f} s"
                ),
                status="Point is too far from hyperbola.",
            )
            return

        if self.distance_point_1 is None:
            self.distance_point_1 = snapped

        elif self.distance_point_2 is None:
            self.distance_point_2 = snapped

        else:
            self.distance_point_1 = self.distance_point_2
            self.distance_point_2 = snapped
            
        self._refresh_distance_range_geometry()


    def snap_point_to_hyperbola(
        self,
        *,
        clicked_time: float,
        clicked_distance: float,
    ) -> tuple[float, float] | None:
        """Return nearest channel point on apex hyperbola if close enough."""
        time_s, x = self.calculate_hyperbola(anchor="apex")

        if time_s is None or x is None:
            return None

        row_idx = int(np.argmin(np.abs(x - clicked_distance)))

        predicted_time = float(time_s[row_idx])
        snapped_distance = float(x[row_idx])

        if abs(clicked_time - predicted_time) > self.hyperbola_click_tolerance_s:
            return None

        return predicted_time, snapped_distance

    # ========================================================================
    # F-X box workflow
    # ========================================================================

    def get_fx_call_time_range(self) -> tuple[float | None, float | None]:
        """Return [apex time, latest selected distance-point time]."""
        if (
            self.apex_point is None
            or self.distance_point_1 is None
            or self.distance_point_2 is None
        ):
            return None, None

        apex_time = float(self.apex_point[0])
        end_time = max(
            float(self.distance_point_1[0]),
            float(self.distance_point_2[0]),
        )

        return min(apex_time, end_time), max(apex_time, end_time)

    def get_call_fx_slice_indices(self) -> list[int]:
        """Return F-X slices overlapping the selected call time range."""
        call_start, call_end = self.get_fx_call_time_range()

        if call_start is None or call_end is None:
            return []

        fx_dataset = self.data_manager.fx_manager.get_dataset()
        slice_times = fx_dataset.get("t")

        if slice_times is None:
            return []

        win_s = float(
            self.data_manager.get_user_settings("win_s") or 2.0
        )

        return [
            index
            for index, slice_start in enumerate(slice_times)
            if float(slice_start) <= call_end
            and float(slice_start) + win_s >= call_start
        ]

    def start_fx_box_labeling(self) -> None:
        """Enter manual F-X ROI annotation mode."""
        if (
            self.apex_point is None
            or self.distance_point_1 is None
            or self.distance_point_2 is None
        ):
            self.notify(
                prompt=(
                    "Complete apex and distance-point labeling before "
                    "F-X box labeling."
                ),
                status="Apex and distance points are required.",
            )
            return

        self.active_fx_slice_indices = self.get_call_fx_slice_indices()

        if not self.active_fx_slice_indices:
            self.notify(
                prompt=(
                    "No F-X windows overlap the selected call range.\n\n"
                    "Check apex and distance-point selections."
                ),
                status="No F-X windows found.",
            )
            return

        self.data_manager.annotation_rois_per_slice = {}

        self.f_min = None
        self.f_max = None

        self.set_mode(
            self.MODE_FX_BOXES,
            self.STAGE_FX,
            tx_annotation_active=False,
        )

        if hasattr(
            self.fx_series_panel,
            "set_annotation_slice_indices",
        ):
            self.fx_series_panel.set_annotation_slice_indices(
                self.active_fx_slice_indices
            )

        first_idx = self.active_fx_slice_indices[0]
        self.on_fx_slice_selected(first_idx)

        # Scroll relevant first thumbnail into view.
        if first_idx < len(self.fx_series_panel.plot_widgets):
            first_thumbnail = self.fx_series_panel.plot_widgets[first_idx]

            QTimer.singleShot(
                0,
                lambda: self.fx_series_panel.scroll_area.verticalScrollBar().setValue(
                    max(0, int(first_thumbnail.pos().y()) - 5)
                ),
            )

        call_start, call_end = self.get_fx_call_time_range()

        self.notify(
            cursor_mode="F-X box labeling",
            prompt=(
                "F-X box labeling mode.\n\n"
                f"Call range: {call_start:.2f}–{call_end:.2f} s\n"
                f"Relevant windows: {len(self.active_fx_slice_indices)}\n\n"
                "Double-click main F-X plot: add a box\n"
                "Ctrl + click box: delete\n"
                "Drag box/handles: move or resize\n"
                "Space: next relevant F-X window\n"
                "F: finish F-X boxes"
            ),
            status="F-X box labeling active.",
        )

    def finish_fx_box_labeling(self) -> None:
        """Finish ROI editing and calculate global F-X frequency bounds."""
        rois_by_slice = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        )

        all_boxes = [
            box
            for boxes in rois_by_slice.values()
            for box in boxes
        ]

        if not all_boxes:
            self.notify(
                prompt=(
                    "No F-X boxes were created.\n\n"
                    "Double-click the main F-X plot to add a box."
                ),
                status="No F-X boxes created.",
            )
            return

        self.f_min = min(float(box[0]) for box in all_boxes)
        self.f_max = max(float(box[2]) for box in all_boxes)

        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_FX_DONE,
            tx_annotation_active=False,
        )

        if hasattr(
            self.fx_series_panel,
            "clear_annotation_slice_highlights",
        ):
            self.fx_series_panel.clear_annotation_slice_highlights()

        self.control_panel.annotation_panel.update_preview({
            "apex_dist": self.apex_point[1],
            "duration": abs(self.endpoint_point[0] - self.apex_point[0]),
            "distance_to_cable": self.distance_to_cable,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "f_min": self.f_min,
            "f_max": self.f_max,
        })

        self.notify(
            cursor_mode="F-X labeling complete",
            prompt=(
                "F-X labeling complete.\n\n"
                f"Frequency range: {self.f_min:.2f}–{self.f_max:.2f} Hz\n\n"
                "Select a label and press its number, or use Confirm & Save."
            ),
            status="F-X annotation complete.",
        )

    def select_next_fx_slice(self) -> None:
        """Advance to the next F-X slice relevant to the selected call."""
        if not self.active_fx_slice_indices:
            return

        current_idx = getattr(
            self.fx_plot_panel,
            "current_slice_idx",
            None,
        )

        relevant_indices = sorted(self.active_fx_slice_indices)

        try:
            current_position = relevant_indices.index(current_idx)
        except ValueError:
            self.on_fx_slice_selected(relevant_indices[0])
            return

        if current_position < len(relevant_indices) - 1:
            self.on_fx_slice_selected(
                relevant_indices[current_position + 1]
            )
            return

        self.notify(
            prompt=(
                "Already at the final relevant F-X window.\n\n"
                "Press F when F-X box labeling is complete."
            ),
            status="Final relevant F-X window.",
        )

    # ========================================================================
    # Saving and existing-label display
    # ========================================================================

    def save_annotation(self, label_num: int) -> None:
        """
        Save current annotation.

        Requires apex, endpoint, and two distance-side points.
        F-X boxes are optional.
        """
        if self.apex_point is None:
            self.notify(
                prompt="Cannot save: select an apex first.",
                status="Apex required.",
            )
            return

        if self.endpoint_point is None:
            self.notify(
                prompt="Cannot save: select an endpoint first.",
                status="Endpoint required.",
            )
            return

        if (
            self.distance_point_1 is None
            or self.distance_point_2 is None
            or self.dist_min is None
            or self.dist_max is None
        ):
            self.notify(
                prompt=(
                    "Cannot save: select and confirm two distance-side "
                    "points first."
                ),
                status="Distance range required.",
            )
            return

        if label_num == 0:
            self.notify(
                prompt="Label 0 is reserved and cannot save an annotation.",
                status="Annotation not saved.",
            )
            return

        settings = self.control_panel.get_settings()
        labels_path = settings.get("labels_file_path", "").strip()

        if not labels_path:
            self.notify(
                prompt="Choose an annotation CSV file before saving.",
                status="Annotation file required.",
            )
            return

        self._configure_label_saver(settings)

        apex_time_s, apex_distance_m = map(float, self.apex_point)
        endpoint_time_s = float(self.endpoint_point[0])

        duration_s = abs(endpoint_time_s - apex_time_s)
        
        window_start_utc = float(
            self.data_manager.loaded_data["time_stamps"][0]
        )

        apex_time_offset_s = float(self.apex_point[0])
        apex_time_utc = window_start_utc + apex_time_offset_s

        apex_time_text = datetime.fromtimestamp(
            apex_time_utc,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"

        source_file_path, apex_time_s = (
            self.data_manager.source_file_time_offset_s(
                apex_time_utc
            )
        )

        dataset_name = Path(self.data_manager.directory).name
        
        label_name = settings.get(
            "label_mapping",
            DEFAULT_LABEL_MAPPING,
        ).get(int(label_num), "")

        tx_id = self.data_manager.label_saver.save_tx_label(
            uid=str(uuid.uuid4()),
            apex_time_utc=apex_time_utc,
            apex_time_str=apex_time_text,
            apex_time_s=apex_time_s,
            apex_dist=apex_distance_m,
            x_m=[self.dist_min, self.dist_max],
            t_s=[0.0, duration_s],
            dataset=dataset_name,
            source_file=str(source_file_path),
            label=int(label_num),
            label_name=label_name,
            distance_to_cable=self.distance_to_cable,
            sound_speed_mps=self.sound_speed,
        )

        fx_box_count = self._save_fx_boxes(
            tx_id=tx_id,
            label_num=int(label_num),
            label_name=label_name,
            dataset_name=dataset_name,
        )

        if self.show_labels:
            self.on_toggle_labels(True)

        self.notify(
            cursor_mode="Normal",
            prompt=(
                f"Saved label {label_num}: {label_name}\n"
                f"TX ID: {tx_id}\n"
                f"F-X boxes: {fx_box_count}\n\n"
                "Press A to begin another annotation."
            ),
            status=f"Saved annotation {tx_id}.",
        )

        self.reset_annotation()

    def _save_fx_boxes(
        self,
        *,
        tx_id: int,
        label_num: int,
        label_name: str,
        dataset_name: str,
    ) -> int:
        """Persist all current F-X boxes and return saved box count."""
        rois_by_slice = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        )

        fx_dataset = self.data_manager.fx_manager.get_dataset()
        fx_times = fx_dataset.get("t")

        win_s = float(
            self.data_manager.get_user_settings("win_s") or 2.0
        )

        count = 0

        for slice_idx, boxes in rois_by_slice.items():
            slice_start_s = 0.0

            if fx_times is not None and slice_idx < len(fx_times):
                slice_start_s = float(fx_times[slice_idx])

            for f_min, x_min, f_max, x_max in boxes:
                self.data_manager.label_saver.save_fx_label(
                    tx_id=tx_id,
                    f_min_hz=float(f_min),
                    f_max_hz=float(f_max),
                    x_min_m=float(x_min),
                    x_max_m=float(x_max),
                    t=slice_start_s,
                    win_length_s=win_s,
                    dataset=dataset_name,
                    label=label_num,
                    label_name=label_name,
                )

                count += 1

        return count

    def on_toggle_labels(self, show: bool) -> None:
        """Show or hide labels whose apex time overlaps current window."""
        if not show:
            self.tx_plot_panel.hide_existing_labels()
            self.control_panel.toggle_labels_button.setText(
                "Show Existing Labels"
            )
            self.show_labels = False

            self.statusBar().showMessage("Existing labels hidden.")
            return

        labels = self.data_manager.get_labels_in_current_window()

        for label in labels:
            label["hyperbola_segment"] = (
                self.get_existing_label_hyperbola_segment(label)
            )

        self.tx_plot_panel.show_existing_labels(labels)

        self.control_panel.toggle_labels_button.setText(
            "Hide Existing Labels"
        )
        self.show_labels = True

        self.statusBar().showMessage(
            f"Showing {len(labels)} existing labels."
        )

    def on_delete_label(self, tx_id: int) -> None:
        """Confirm and remove one annotation."""
        if self.data_manager.label_saver is None:
            self.statusBar().showMessage("No annotation file is open.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Annotation",
            (
                f"Delete annotation ID {tx_id}?\n\n"
                "This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Delete cancelled.")
            return

        self.data_manager.label_saver.remove_label_by_id(tx_id)

        if self.show_labels:
            self.on_toggle_labels(True)

        self.statusBar().showMessage(
            f"Removed annotation {tx_id}."
        )

    def calculate_hyperbola_for_label(
        self,
        *,
        apex_time: float,
        apex_distance: float,
        distance_to_cable: float,
        sound_speed_mps: float,
    ):
        """Calculate a displayed hyperbola from saved annotation values."""
        x = self.data_manager.loaded_data.get("x")

        if x is None or len(x) == 0:
            return None, None

        try:
            apex_time = float(apex_time)
            apex_distance = float(apex_distance)
            distance_to_cable = float(distance_to_cable)
        except (TypeError, ValueError):
            return None, None

        if not np.isfinite(distance_to_cable):
            return None, None

        range_m = np.sqrt(
            (x - apex_distance) ** 2
            + distance_to_cable ** 2
        )

        try:
            sound_speed_mps = float(sound_speed_mps)
        except (TypeError, ValueError):
            return None, None

        if not np.isfinite(sound_speed_mps) or sound_speed_mps <= 0:
            return None, None

        time_s = apex_time + (
            range_m - distance_to_cable
        ) / sound_speed_mps

        return time_s, x

    def get_existing_label_hyperbola_segment(self, label: dict):
        """Return saved-label hyperbola segment for the current display window."""
        try:
            distance_to_cable = float(
                label.get("distance_to_cable")
            )
            dist_min = float(label.get("dist_min"))
            dist_max = float(label.get("dist_max"))

            window_start_utc = float(
                self.data_manager.loaded_data["time_stamps"][0]
            )

            apex_time_offset_s = (
                float(label["apex_time_utc"])
                - window_start_utc
            )

            apex_distance = float(label["apex_dist"])

        except (TypeError, ValueError, KeyError):
            return None

        if not (
            np.isfinite(distance_to_cable)
            and np.isfinite(dist_min)
            and np.isfinite(dist_max)
        ):
            return None

        try:
            sound_speed_mps = float(
                label.get("sound_speed_mps")
            )
        except (TypeError, ValueError):
            sound_speed_mps = self.sound_speed

        if not np.isfinite(sound_speed_mps) or sound_speed_mps <= 0:
            sound_speed_mps = self.sound_speed

        time_s, x = self.calculate_hyperbola_for_label(
            apex_time=apex_time_offset_s,
            apex_distance=apex_distance,
            distance_to_cable=distance_to_cable,
            sound_speed_mps=sound_speed_mps,
        )

        if time_s is None or x is None:
            return None

        lower_x, upper_x = sorted((dist_min, dist_max))
        mask = (x >= lower_x) & (x <= upper_x)

        if not np.any(mask):
            return None

        min_idx = int(np.argmin(np.abs(x - lower_x)))
        max_idx = int(np.argmin(np.abs(x - upper_x)))

        return {
            "times": time_s[mask],
            "distances": x[mask],
            "side_points": [
                (float(time_s[min_idx]), float(x[min_idx])),
                (float(time_s[max_idx]), float(x[max_idx])),
            ],
        }
