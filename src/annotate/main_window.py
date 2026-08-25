from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QFileDialog, QScrollArea, QMessageBox
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer
from datetime import datetime, timezone
import os, uuid
import numpy as np

from annotate.data_manager import PreprocessedDataManager
from annotate.panels.control_panel import ControlPanel
from annotate.panels.tx_plot_panel import TXPlotPanel
from annotate.panels.spectrogram_panel import SpectrogramPanel
from annotate.panels.fx_plot_panel import FXPlotPanel
from annotate.panels.fx_series_panel import FXSeriesPanel
from annotate.panels.text_display_panel import TextDisplayPanel
from annotate.config import DEFAULT_DATASET_PATH, DEFAULT_LABEL_MAPPING


class MainWindow(QMainWindow):
    # ==============================================================
    # Modes and annotation stages
    # ==============================================================
    MODE_NORMAL = ""
    MODE_SPECTROGRAM = "s"
    MODE_ANNOTATION = "annotation"
    MODE_FX_BOXES = "fx_boxes"

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

    # ==============================================================
    # Construction and layout
    # ==============================================================
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotate DAS Data")
        self.resize(1400, 900)
        self.showMaximized()
        self.cursor_mode = ''
        self.annotation_stage = ''
        self.show_labels = False # show existing labels toggle

        # --- Core data manager ---
        self.data_manager = PreprocessedDataManager()

        # --- Central layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # === Top-level horizontal splitter: Control | Middle grid | FXSeries ===
        hsplit = QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(hsplit)

        # --- Control panel (left) ---
        self.control_panel = ControlPanel(self.data_manager)
        self.control_panel.setMinimumWidth(200)
        self.control_panel.setMaximumWidth(400)
        hsplit.addWidget(self.control_panel)

        # --- Middle column: vertical splitter (top vs bottom) ---
        middle_vsplit = QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top horizontal split: TX plot | FX plot
        top_hsplit = QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.tx_plot_panel = TXPlotPanel(self.data_manager)
        top_hsplit.addWidget(self.tx_plot_panel)

        self.fx_plot_panel = FXPlotPanel(self.data_manager)
        top_hsplit.addWidget(self.fx_plot_panel)
        middle_vsplit.addWidget(top_hsplit)

        # Bottom horizontal split: Spectrogram | Text Panel
        bottom_hsplit = QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.spectrogram_panel = SpectrogramPanel(self.data_manager)
        bottom_hsplit.addWidget(self.spectrogram_panel)

        self.text_display_panel = TextDisplayPanel()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_display_panel)
        scroll_area.setMinimumWidth(220)
        scroll_area.setMaximumWidth(320)

        bottom_hsplit.addWidget(scroll_area)

        middle_vsplit.addWidget(bottom_hsplit)

        # (Removed duplicate scroll area creation and addition)
        # Connect to update file info display
        self.data_manager.file_loaded.connect(self.text_display_panel.update_file_info)

        # Mirror horizontal sizes between top and bottom splitters
        top_hsplit.splitterMoved.connect(lambda pos, index: bottom_hsplit.setSizes(top_hsplit.sizes()))
        bottom_hsplit.splitterMoved.connect(lambda pos, index: top_hsplit.setSizes(bottom_hsplit.sizes()))

        hsplit.addWidget(middle_vsplit)

        # --- Right column: FX Series Panel ---
        self.fx_series_panel = FXSeriesPanel(self.data_manager)
        self.fx_series_panel.setMinimumWidth(300)
        self.fx_series_panel.setMaximumWidth(300)
        hsplit.addWidget(self.fx_series_panel)

        # Stretch: control fixed, middle grows, FX fixed
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 4)
        hsplit.setStretchFactor(2, 0)

        # === Cross-panel interactions ===
        self.fx_series_panel.slice_selected.connect(self.on_fx_slice_selected)
        self.fx_plot_panel.plot_widget.getPlotItem().setYLink(self.tx_plot_panel.plot_widget)
        self.tx_plot_panel.point_clicked.connect(self.on_point_clicked)
        self.fx_plot_panel.point_clicked.connect(self.on_point_clicked)
        self.tx_plot_panel.label_delete_requested.connect(self.on_delete_label)
        self.fx_plot_panel.roi_changed.connect(self.on_fx_roi_changed)
        self.tx_plot_panel.apex_dragged.connect(self.on_apex_dragged)
        self.tx_plot_panel.endpoint_dragged.connect(self.on_endpoint_dragged)

        # === Navigation buttons ===
        self.control_panel.btn_back.clicked.connect(lambda: self.data_manager.navigate('backward'))
        self.control_panel.btn_forward.clicked.connect(lambda: self.data_manager.navigate('forward'))

        # === Apply Changes button and Add Labels button ===
        self.control_panel.toggle_labels_requested.connect(self.on_toggle_labels)
        self.control_panel.refresh_requested.connect(self.on_apply_changes)
        self.control_panel.confirm_label_requested.connect(self.save_annotation)

        self.control_panel.distance_to_cable_changed.connect(self.on_distance_to_cable_changed)

        # === T-X annotation state ===
        self.apex_point = None
        self.apex_row_idx = None
        self.endpoint_point = None
        self.distance_point_1 = None
        self.distance_point_2 = None
        self.dist_min = None
        self.dist_max = None

        # F-X bounding-box annotation state
        self.active_fx_slice_indices = []
        self.f_min = None
        self.f_max = None

        # Hyperbola / distance-to-cable fitting state
        self.hyperbola_active = False
        self.distance_to_cable = None

        # Acoustic propagation speed in water, m/s.
        self.sound_speed = 1500.0
        # Maximum allowed time difference (in seconds) between click and fitted hyperbola.
        self.hyperbola_click_tolerance_s = 1.0

        # Geometry assumptions for a straight 3-D cable.
        # Replace these later with actual cable geometry/metadata if available.
        self.cable_origin_3d = np.array([0.0, 0.0, 0.0])
        self.cable_direction_3d = np.array([1.0, 0.0, 0.0])

        # Direction perpendicular to the cable in which the whale is placed.
        # Here this means "downward" in Z. Change if your coordinate system differs.
        self.cable_normal_3d = np.array([0.0, 0.0, -1.0])

        # Build menu
        self.create_menu()

    # ==============================================================
    # Dataset and settings
    # ==============================================================
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        select_dataset_action = file_menu.addAction("Select Preprocessed Dataset")
        select_dataset_action.triggered.connect(self.select_preprocessed_dataset)

    def select_preprocessed_dataset(self):
        filepath, _ = QFileDialog.getOpenFileName(self,
                                                  "Select Preprocessed Dataset",
                                                  DEFAULT_DATASET_PATH,
                                                  "Data Files (*.h5);;All Files (*)")
        if filepath:
            settings = self.control_panel.get_settings()
            self.data_manager.apply_user_settings(settings)

            labels_path = settings.get('labels_file_path')
            print("labels_path:", labels_path)
            if labels_path:
                from annotate.data_manager import LabelSaver
                self.data_manager.label_saver = LabelSaver(labels_path)

            self.data_manager.new_file_selected(filepath)

    def on_apply_changes(self):
        """When user clicks Apply Changes: store settings + recompute FX."""
        settings = self.control_panel.get_settings()
        self.data_manager.apply_user_settings(settings)
        self.data_manager.load_current_window(recompute_fx=True)
        labels_path = settings.get('labels_file_path')
        if labels_path:
            from annotate.data_manager import LabelSaver
            self.data_manager.label_saver = LabelSaver(labels_path)

    # ==============================================================
    # Shared UI and annotation-state helpers
    # ==============================================================
    def notify(self, cursor_mode=None, prompt=None, status=None):
        """Update the text panel and status bar in one place."""
        if cursor_mode is not None:
            self.text_display_panel.update_cursor_mode(cursor_mode)

        if prompt is not None:
            self.text_display_panel.set_annotation_prompt(prompt)

        if status is not None:
            self.statusBar().showMessage(status)

    def set_mode(self, mode, stage=None, tx_annotation_active=False):
        """Set application annotation mode consistently."""
        self.cursor_mode = mode
        self.data_manager.set_cursor_mode(mode)
        self.tx_plot_panel.annotation_mode_active = tx_annotation_active

        if stage is not None:
            self.annotation_stage = stage

    def reset_annotation(self, clear_control_panel=True):
        """Clear temporary annotation data, overlays, and F-X state."""
        self.set_mode(self.MODE_NORMAL, self.STAGE_NONE, False)

        self.apex_point = None
        self.apex_row_idx = None
        self.endpoint_point = None

        self.distance_point_1 = None
        self.distance_point_2 = None
        self.dist_min = None
        self.dist_max = None

        self.active_fx_slice_indices = []
        self.f_min = None
        self.f_max = None
        self.hyperbola_active = False
        self.distance_to_cable = None   

        self.clear_all_annotation_overlays()

        if clear_control_panel:
            self.control_panel.clear_annotation_preview()
        self.control_panel.clear_distance_to_cable()

        if hasattr(self.fx_series_panel, "clear_annotation_slice_highlights"):
            self.fx_series_panel.clear_annotation_slice_highlights()

    def clear_all_annotation_overlays(self):
        """Remove unsaved T-X markers, F-X ROIs, and thumbnail annotations."""
        self.tx_plot_panel.clear_annotation_overlays()
        self.fx_plot_panel.clear_annotation_overlays()
        self.fx_series_panel.clear_annotation_overlays()
        if hasattr(self.data_manager, "annotation_rois_per_slice"):
            self.data_manager.annotation_rois_per_slice.clear()

    def update_distance_markers(self):
        """Show the latest one or two distance points on the T-X plot."""
        points = []

        if self.distance_point_1 is not None:
            points.append(self.distance_point_1)

        if self.distance_point_2 is not None:
            points.append(self.distance_point_2)

        self.tx_plot_panel.set_distance_annotation_points(points)

    # ==============================================================
    # Plot and panel interactions
    # ==============================================================
    def on_fx_slice_selected(self, idx):
        """User clicked an F-X series thumbnail."""

        # In F-X box mode, only allow windows overlapping the call.
        if self.cursor_mode == "fx_boxes":
            if idx not in self.active_fx_slice_indices:
                message = (
                    "This F-X window is outside the selected call time range.\n\n"
                    "Choose one of the highlighted call windows."
                )
                self.statusBar().showMessage(
                    "Selected F-X window is outside the call time range."
                )
                self.text_display_panel.set_annotation_prompt(message)
                return

        self.fx_plot_panel.show_slice_from_series(idx)
        self.fx_series_panel.highlight_slice(idx)

        dataset = self.data_manager.fx_manager.get_dataset()
        times = dataset["t"]

        if times is None or idx >= len(times):
            return

        win_s = self.data_manager.get_user_settings("win_s") or 2.0
        t_start = times[idx]
        t_end = t_start + win_s

        self.tx_plot_panel.highlight_time_window(t_start, t_end)

        if (
            hasattr(self.spectrogram_panel, "last_row_idx")
            and self.spectrogram_panel.last_row_idx is not None
        ):
            self.spectrogram_panel.highlight_time_window(t_start, t_end)

        if self.cursor_mode == "fx_boxes":
            box_count = len(
                getattr(self.data_manager, "annotation_rois_per_slice", {}).get(idx, [])
            )

            self.text_display_panel.set_annotation_prompt(
                "F-X box labeling mode.\n\n"
                f"Selected window: {t_start:.2f}–{t_end:.2f} s\n"
                f"Boxes in this window: {box_count}\n\n"
                "Double-click: add a box\n"
                "Ctrl + Click: delete a box\n"
                "Drag a box to move or resize it.\n\n"
                "Press Space for the next relevant F-X window.\n\n"
                "Select another relevant F-X window, or press F when done."
            )

    def select_next_fx_slice(self):
        """
        In F-X box labeling mode, move to the next relevant F-X slice.

        Space advances through the F-X windows that overlap
        the currently annotated whale call.
        """
        if not self.active_fx_slice_indices:
            message = "No relevant F-X windows are available."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # The large F-X panel stores the currently displayed slice.
        current_idx = self.fx_plot_panel.current_slice_idx

        # Ensure indices are sorted in time order.
        relevant_indices = sorted(self.active_fx_slice_indices)

        try:
            current_position = relevant_indices.index(current_idx)
        except ValueError:
            # If the current plot is not one of the relevant windows,
            # begin with the first relevant one.
            self.on_fx_slice_selected(relevant_indices[0])
            return

        # Move to the next relevant F-X plot.
        if current_position < len(relevant_indices) - 1:
            next_idx = relevant_indices[current_position + 1]
            self.on_fx_slice_selected(next_idx)
            return

        # Already at the final F-X plot.
        last_idx = relevant_indices[-1]
        rois_by_slice = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {}
        )
        box_count = len(rois_by_slice.get(last_idx, []))

        message = (
            "Already at the final relevant F-X window.\n\n"
            f"Boxes in this window: {box_count}\n\n"
            "Press F when all F-X boxes are complete."
        )
        self.statusBar().showMessage(
            "Already at final relevant F-X window. Press F when done."
        )
        self.text_display_panel.set_annotation_prompt(message)

    def on_fx_roi_changed(self, slice_idx):
        """Refresh thumbnail styling after an F-X ROI is added, changed, or removed."""
        self.fx_series_panel.highlight_slice(slice_idx)

        if self.cursor_mode != self.MODE_FX_BOXES:
            return

        boxes = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        ).get(slice_idx, [])

        if boxes:
            count = len(boxes)
            self.statusBar().showMessage(
                f"F-X window {slice_idx} labeled ({count} box{'es' if count != 1 else ''})."
            )
        else:
            self.statusBar().showMessage(
                f"F-X window {slice_idx} has no bounding boxes."
            )

    def on_point_clicked(self, row_idx, col_idx, clicked_distance=None):
        if self.cursor_mode == self.MODE_SPECTROGRAM:
            self.select_spectrogram_row(row_idx)
            return

        if self.cursor_mode != self.MODE_ANNOTATION:
            return

        t_val = float(self.data_manager.loaded_data["t"][col_idx])
        x_val = float(self.data_manager.loaded_data["x"][row_idx])

        if self.annotation_stage == self.STAGE_APEX:
            self.select_apex(row_idx, t_val, x_val)

        elif self.annotation_stage == self.STAGE_ENDPOINT:
            self.select_endpoint(row_idx, t_val, x_val, clicked_distance)

        elif self.annotation_stage == self.STAGE_DIST_POINTS:
            self.select_distance_point_on_hyperbola(t_val, x_val)

    def select_spectrogram_row(self, row_idx):
        dist_val = float(self.data_manager.loaded_data["x"][row_idx])

        self.spectrogram_panel.update_plot(row_idx)
        self.tx_plot_panel.mark_distance(dist_val)
        self.fx_plot_panel.mark_distance(dist_val)

        self.set_mode(self.MODE_NORMAL)

        self.notify(
            cursor_mode="Normal",
            prompt=f"Spectrogram row selected.\nDistance: {dist_val:.2f} m",
            status=f"Spectrogram updated for row {row_idx}.",
        )

    def start_spectrogram_selection(self):
        self.set_mode(self.MODE_SPECTROGRAM)

        self.notify(
            cursor_mode="Select spectrogram row",
            prompt="Click a T-X or F-X plot to select the spectrogram row.",
            status="Spectrogram selection active.",
        )


    # ==============================================================
    # Keyboard shortcuts
    # ==============================================================
    def keyPressEvent(self, event):
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

        elif key == QtCore.Qt.Key.Key_Escape:
            self.reset_annotation(clear_control_panel=False)
            self.notify(
                cursor_mode="Normal",
                prompt="Annotation cancelled.",
                status="Normal mode",
            )

        else:
            super().keyPressEvent(event)

    # ==============================================================
    # T-X apex and endpoint workflow
    # ==============================================================
    def handle_apex_key(self):
        """
        A workflow:

        First A:
            enter apex selection mode.

        Second A:
            confirm apex selection.
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
                    "Click again to replace it.\n\n"
                    "Press A to confirm the apex."
                ),
                status="Select the apex.",
            )
            return

        if self.annotation_stage == self.STAGE_APEX:
            if self.apex_point is None:
                self.notify(
                    prompt="No apex selected yet.\nClick an apex in the T-X plot.",
                    status="Select an apex first.",
                )
                return

            self.annotation_stage = self.STAGE_APEX_DONE
            self.tx_plot_panel.annotation_mode_active = False

            self.notify(
                cursor_mode="Apex confirmed",
                prompt=(
                    "Apex confirmed.\n\n"
                    "Press H to fit the arrival hyperbola and select "
                    "Distance to cable."
                ),
                status="Apex confirmed.",
            )

    def handle_endpoint_key(self):
        """
        E workflow:

        First E:
            enter endpoint-selection mode.

        Second E:
            confirm endpoint.
        """
        if self.annotation_stage == self.STAGE_HYPERBOLA_DONE:
            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_ENDPOINT,
                tx_annotation_active=True,
            )

            # Keep apex-fitted hyperbola visible during endpoint selection.
            self.update_hyperbola(
                anchor="apex",
                curve_name="apex",
            )

            self.notify(
                cursor_mode="Endpoint labeling",
                prompt=(
                    "Endpoint labeling:\n\n"
                    "Click near the endpoint on the apex channel.\n"
                    "The endpoint will snap to the apex channel.\n\n"
                    "White dashed line: apex hyperbola\n"
                    "Cyan dashed line: endpoint hyperbola\n\n"
                    "Press E to confirm endpoint."
                ),
                status="Select endpoint near the apex channel.",
            )
            return

        if self.annotation_stage == self.STAGE_ENDPOINT:
            if self.endpoint_point is None:
                self.notify(
                    prompt="No endpoint selected yet.\nClick an endpoint first.",
                    status="Select an endpoint first.",
                )
                return

            self.annotation_stage = self.STAGE_ENDPOINT_DONE
            self.tx_plot_panel.annotation_mode_active = False

            # Redraw final endpoint-anchored full hyperbola.
            self.update_hyperbola(anchor="endpoint")

            self.notify(
                cursor_mode="Endpoint confirmed",
                prompt=(
                    "Endpoint confirmed.\n\n"
                    "Press D to select two distance-side points "
                    "along the hyperbola."
                ),
                status="Endpoint confirmed.",
            )
            return

        self.notify(
            prompt=(
                "Confirm Distance to cable before endpoint labeling.\n\n"
                "Press H to fit and confirm the hyperbola."
            ),
            status="Hyperbola fitting must be confirmed first.",
        )

    def select_apex(self, row_idx, t_val, x_val):
        self.apex_row_idx = row_idx
        self.apex_point = (t_val, x_val)
        self.endpoint_point = None

        self.tx_plot_panel.mark_apex_point(t_val, x_val)
        self.tx_plot_panel.clear_endpoint_point()

        self.control_panel.apex_time_display.setText(f"{t_val:.3f}")
        self.control_panel.apex_dist_display.setText(f"{x_val:.2f}")
        self.control_panel.duration_display.clear()

        self.notify(
            prompt=(
                f"Apex selected:\n\n"
                f"Time: {t_val:.3f} s\n"
                f"Distance: {x_val:.2f} m\n\n"
                "Click again to replace it.\n"
                "Press A for endpoint labeling."
            ),
            status="Apex updated.",
        )

    def select_endpoint(self, row_idx, t_val, x_val, clicked_distance=None):
        """
        Select endpoint time while forcing endpoint distance to the
        current apex channel/distance.
        """
        if self.apex_point is None:
            self.notify(
                prompt="No apex selected. Press A to select an apex first.",
                status="No apex selected.",
            )
            return

        # Always use the current displayed/stored apex distance.
        apex_distance = float(self.apex_point[1])

        if clicked_distance is None:
            clicked_distance = float(x_val)

        dx = float(self.data_manager.h5settings.get("dx") or 1.0)
        tolerance = 100.0 * dx

        if abs(float(clicked_distance) - apex_distance) > tolerance:
            self.notify(
                prompt=(
                    "Endpoint click is too far from the apex channel.\n\n"
                    f"Apex channel: {apex_distance:.2f} m\n"
                    f"Clicked: {float(clicked_distance):.2f} m\n\n"
                    "Click closer to the apex channel."
                ),
                status="Endpoint click is too far from apex channel.",
            )
            return

        # Keep clicked time, but use EXACTLY the same distance as apex.
        self.endpoint_point = (float(t_val), apex_distance)

        duration = abs(float(t_val) - float(self.apex_point[0]))

        self.tx_plot_panel.mark_endpoint_point(
            float(t_val),
            apex_distance,
        )

        self.control_panel.duration_display.setText(f"{duration:.3f}")

        # Keep both hyperbolas visible during endpoint labeling.
        if self.distance_to_cable is not None:
            self.update_hyperbola(anchor="apex", curve_name="apex")
            self.update_hyperbola(anchor="endpoint", curve_name="endpoint")

        self.notify(
            prompt=(
                "Endpoint selected.\n\n"
                f"Endpoint time: {t_val:.3f} s\n"
                f"Endpoint channel: {apex_distance:.2f} m\n"
                f"Duration: {duration:.3f} s\n\n"
                "Click again to replace endpoint, or press E to confirm."
            ),
            status="Endpoint updated and snapped to apex channel.",
        )

    # ==============================================================
    # T-X distance workflow
    # ==============================================================
    def handle_distance_key(self):
        """
        D workflow:

        First D:
            enter distance-side point selection mode.

        Clicks:
            keep only the latest two points snapped to the
            endpoint-anchored hyperbola.

        Second D:
            confirm distance range and show only the selected
            hyperbola segment.
        """

        # Distance labeling is available only after endpoint confirmation.
        if self.annotation_stage not in (
            self.STAGE_ENDPOINT_DONE,
            self.STAGE_DIST_POINTS,
            self.STAGE_DIST_DONE,
        ):
            self.notify(
                prompt=(
                    "Confirm the endpoint before distance labeling.\n\n"
                    "Press E to select and confirm the endpoint."
                ),
                status="Endpoint confirmation is required.",
            )
            return

        # Start or restart distance labeling.
        if self.annotation_stage in (
            self.STAGE_ENDPOINT_DONE,
            self.STAGE_DIST_DONE,
        ):
            self.distance_point_1 = None
            self.distance_point_2 = None
            self.dist_min = None
            self.dist_max = None

            self.control_panel.dist_min_display.clear()
            self.control_panel.dist_max_display.clear()

            # Remove old point markers and restore the full hyperbola.
            self.tx_plot_panel.set_distance_annotation_points([])

            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_DIST_POINTS,
                tx_annotation_active=True,
            )

            self.tx_plot_panel.clear_hyperbola("endpoint")
            self.update_hyperbola(anchor="apex", curve_name="apex")

            self.notify(
                cursor_mode="Distance labeling",
                prompt=(
                    "Distance labeling:\n\n"
                    "Click two points near the endpoint-anchored hyperbola.\n"
                    "Selected points snap to the hyperbola.\n\n"
                    "Only the latest two points are retained.\n"
                    "Press D when distance labeling is complete."
                ),
                status="Select two distance points near the hyperbola.",
            )
            return

        # Confirm selected distance points.
        if self.annotation_stage == self.STAGE_DIST_POINTS:
            if self.distance_point_1 is None or self.distance_point_2 is None:
                self.notify(
                    prompt=(
                        "Select two distance points before finishing.\n\n"
                        "The latest two selected points define "
                        "the distance range."
                    ),
                    status="Two distance points are required.",
                )
                return

            self.annotation_stage = self.STAGE_DIST_DONE
            self.tx_plot_panel.annotation_mode_active = False

            # Replace full curve with only the curve between the two points.
            self.show_confirmed_hyperbola_segment(anchor="apex")

            self.notify(
                cursor_mode="Distance label done",
                prompt=(
                    "Distance label done.\n\n"
                    f"Minimum distance: {self.dist_min:.2f} m\n"
                    f"Maximum distance: {self.dist_max:.2f} m\n\n"
                    "Press F for F-X box labeling,\n"
                    "or select a label and click Confirm & Save."
                ),
                status="Distance labeling complete.",
            )

    def select_distance_point_on_hyperbola(self, clicked_time, clicked_distance):
        """
        Select a distance-side point constrained to the fitted hyperbola.

        The clicked point may be near the curve. If it is within the configured
        time tolerance, the stored point is snapped exactly onto the hyperbola.

        Only the two most recent distance points are retained.
        """
        snapped_point = self.snap_point_to_hyperbola(
            clicked_time,
            clicked_distance,
        )

        if snapped_point is None:
            self.notify(
                prompt=(
                    "Distance point is too far from the fitted hyperbola.\n\n"
                    f"Allowed time tolerance: ±{self.hyperbola_click_tolerance_s:.2f} s\n\n"
                    "Click closer to the white dashed hyperbola."
                ),
                status="Distance point must be selected near the hyperbola.",
            )
            return

        snapped_time, snapped_distance = snapped_point
        new_point = (snapped_time, snapped_distance)

        # First click.
        if self.distance_point_1 is None:
            self.distance_point_1 = new_point

            self.update_distance_markers()

            self.notify(
                prompt=(
                    "First distance point selected on hyperbola.\n\n"
                    f"Time: {snapped_time:.3f} s\n"
                    f"Distance: {snapped_distance:.2f} m\n\n"
                    "Click a second point on the hyperbola.\n"
                    "Press D when the two selected points are correct."
                ),
                status="First hyperbola distance point selected.",
            )
            return

        # Second click.
        if self.distance_point_2 is None:
            self.distance_point_2 = new_point

        # Third and later clicks: retain only latest two points.
        else:
            self.distance_point_1 = self.distance_point_2
            self.distance_point_2 = new_point

        d1 = float(self.distance_point_1[1])
        d2 = float(self.distance_point_2[1])

        self.dist_min = min(d1, d2)
        self.dist_max = max(d1, d2)

        self.control_panel.dist_min_display.setText(
            f"{self.dist_min:.2f}"
        )
        self.control_panel.dist_max_display.setText(
            f"{self.dist_max:.2f}"
        )

        self.update_distance_markers()

        self.notify(
            prompt=(
                "Distance points selected on hyperbola.\n\n"
                f"Point 1: {self.distance_point_1[1]:.2f} m\n"
                f"Point 2: {self.distance_point_2[1]:.2f} m\n"
                f"Minimum distance: {self.dist_min:.2f} m\n"
                f"Maximum distance: {self.dist_max:.2f} m\n\n"
                "Click again to replace the oldest point.\n"
                "Press D when distance labeling is done."
            ),
            status="Distance range updated from hyperbola points.",
        )
    
    # ==============================================================
    # F-X bounding-box workflow
    # ==============================================================
    def get_fx_call_time_range(self):
        """
        F-X windows use:
        start = apex time
        end = later time of the two distance points
        """
        if (
            self.apex_point is None
            or self.distance_point_1 is None
            or self.distance_point_2 is None
        ):
            return None, None

        apex_time = float(self.apex_point[0])

        distance_end_time = max(
            float(self.distance_point_1[0]),
            float(self.distance_point_2[0]),
        )

        return min(apex_time, distance_end_time), max(apex_time, distance_end_time)

    def get_call_fx_slice_indices(self):
        """Return F-X windows that overlap the selected whale-call interval."""
        call_start, call_end = self.get_fx_call_time_range()

        if call_start is None or call_end is None:
            return []

        dataset = self.data_manager.fx_manager.get_dataset()
        slice_times = dataset.get("t")

        if slice_times is None:
            return []

        win_s = float(self.data_manager.get_user_settings("win_s") or 2.0)

        return [
            idx
            for idx, slice_start in enumerate(slice_times)
            if float(slice_start) <= call_end
            and float(slice_start) + win_s >= call_start
        ]
    
    def start_fx_box_labeling(self):
        """Enter manual F-X bounding-box labeling mode."""

        if (self.apex_point is None
            or self.distance_point_1 is None
            or self.distance_point_2 is None
        ):
            message = (
                "Complete apex labeling and both distance-side points "
                "before F-X box labeling."
            )
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        self.active_fx_slice_indices = self.get_call_fx_slice_indices()

        if not self.active_fx_slice_indices:
            message = (
                "No F-X windows overlap the selected call duration.\n\n"
                "Check the apex and distance point selections."
            )
            self.statusBar().showMessage("No overlapping F-X windows found.")
            self.text_display_panel.set_annotation_prompt(message)
            return

        # Start a fresh set of F-X boxes for this annotation.
        self.data_manager.annotation_rois_per_slice = {}

        self.f_min = None
        self.f_max = None
        self.control_panel.freq_min_display.clear()
        self.control_panel.freq_max_display.clear()
        self.set_mode(
            self.MODE_FX_BOXES,
            self.STAGE_FX,
            tx_annotation_active=False,
        )

        # Optional visual indication in the thumbnail panel.
        if hasattr(self.fx_series_panel, "set_annotation_slice_indices"):
            self.fx_series_panel.set_annotation_slice_indices(
                self.active_fx_slice_indices
            )

        first_idx = self.active_fx_slice_indices[0]
        self.on_fx_slice_selected(first_idx)

        # Scroll the first relevant F-X thumbnail to the top only when
        # entering F-X box labeling mode.

        first_thumbnail = self.fx_series_panel.plot_widgets[first_idx]

        QTimer.singleShot(
            0,
            lambda: self.fx_series_panel.scroll_area.verticalScrollBar().setValue(
                max(0, first_thumbnail.pos().y() - 5)
            )
        )

        call_start, call_end = self.get_fx_call_time_range()  
        self.notify(
            cursor_mode="F-X box labeling",
            prompt=(
                "F-X box labeling mode.\n\n"
                f"Call time range: {call_start:.2f}–{call_end:.2f} s\n"
                f"Relevant F-X windows: {len(self.active_fx_slice_indices)}\n\n"
                "Click a relevant F-X thumbnail on the right.\n"
                "Double-click: add a box\n"
                "Ctrl + Click: delete a box\n"
                "Drag boxes to move or resize them.\n\n"
                "Press Space to move to the next F-X window.\n\n"
                "Press F when all F-X boxes are complete."
            ),
            status="F-X box mode active. Select relevant windows and draw boxes.",
        )      

    def finish_fx_box_labeling(self):
        """Leave F-X box mode and calculate global frequency minimum/maximum."""

        rois_by_slice = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {}
        )

        all_boxes = [
            box
            for boxes in rois_by_slice.values()
            for box in boxes
        ]

        if not all_boxes:
            message = (
                "No F-X boxes have been added.\n\n"
                "Select a relevant F-X window and use double-click "
                "to add a bounding box."
            )
            self.statusBar().showMessage("No F-X boxes created.")
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ROI tuple format:
        # (f_min_hz, dist_min_m, f_max_hz, dist_max_m)
        self.f_min = min(float(box[0]) for box in all_boxes)
        self.f_max = max(float(box[2]) for box in all_boxes)

        self.control_panel.freq_min_display.setText(f"{self.f_min:.2f}")
        self.control_panel.freq_max_display.setText(f"{self.f_max:.2f}")

        # Leave F-X editing mode but keep ROIs until Confirm & Save.
        self.set_mode(
            self.MODE_NORMAL,
            self.STAGE_FX_DONE,
            tx_annotation_active=False,
        )

        if hasattr(self.fx_series_panel, "clear_annotation_slice_highlights"):
            self.fx_series_panel.clear_annotation_slice_highlights()

        self.notify(
            cursor_mode="F-X labeling done",
            prompt=(
                "F-X labeling done.\n\n"
                f"Frequency minimum: {self.f_min:.2f} Hz\n"
                f"Frequency maximum: {self.f_max:.2f} Hz\n"
                f"F-X boxes: {len(all_boxes)}\n\n"
                "Select a label in the left panel,\n"
                "then click Confirm & Save."
            ),
            status="F-X labeling complete. Select a label and click Confirm & Save.",
        )

    # ==============================================================
    # Hyperbola / distance-to-cable fitting
    # ==============================================================
    def handle_hyperbola_key(self):
        """
        H workflow:

        First H:
            enter hyperbola fitting mode and enable slider.

        Second H:
            confirm distance-to-cable fit.
        """
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
            self.control_panel.set_distance_to_cable(self.distance_to_cable)

            # At this stage, use apex as a temporary hyperbola anchor.
            self.update_hyperbola(anchor="apex")

            self.notify(
                cursor_mode="Hyperbola fitting",
                prompt=(
                    "Hyperbola fitting mode.\n\n"
                    "Adjust the Distance to cable slider.\n"
                    "The hyperbola is temporarily anchored at the apex.\n\n"
                    "Press H to confirm Distance to cable."
                ),
                status="Adjust Distance to cable.",
            )
            return

        # Second H confirms the distance-to-cable fit.
        self.hyperbola_active = False
        self.annotation_stage = self.STAGE_HYPERBOLA_DONE
        self.control_panel.set_distance_to_cable_enabled(False)
        
        # Snap the smoothly dragged apex to one actual DAS channel only now.
        self.snap_apex_and_endpoint_to_channel()

        self.notify(
            cursor_mode="Distance to cable confirmed",
            prompt=(
                "Distance to cable confirmed.\n\n"
                f"Distance to cable: {self.distance_to_cable:.1f} m\n\n"
                "Press E to label the endpoint."
            ),
            status="Distance to cable confirmed.",
        )

    def start_hyperbola_fitting(self):
        """
        Enable the distance-to-cable slider and show a predicted
        arrival-time hyperbola anchored at the selected apex.
        """
        if self.apex_point is None:
            self.notify(
                prompt=(
                    "Select an apex before entering hyperbola fitting mode."
                ),
                status="Apex selection is required for hyperbola fitting.",
            )
            return

        self.hyperbola_active = True

        # Use previous value if available; otherwise start at zero.
        if self.distance_to_cable is None:
            self.distance_to_cable = 0.0

        self.control_panel.set_distance_to_cable_enabled(True)
        self.control_panel.set_distance_to_cable(self.distance_to_cable)

        self.update_hyperbola()

        self.notify(
            cursor_mode="Hyperbola fitting",
            prompt=(
                "Hyperbola fitting mode.\n\n"
                "Use the Distance to cable slider on the left.\n"
                "The predicted arrival curve updates live in the T-X plot.\n\n"
                "Press H when the fit is satisfactory."
            ),
            status="Hyperbola fitting mode active.",
        )

    def finish_hyperbola_fitting(self):
        """Leave fitting mode while retaining the selected cable distance."""
        if not self.hyperbola_active:
            return

        self.hyperbola_active = False
        self.control_panel.set_distance_to_cable_enabled(False)

        distance_text = (
            f"{self.distance_to_cable:.1f} m"
            if self.distance_to_cable is not None
            else "not selected"
        )

        self.notify(
            cursor_mode="Normal",
            prompt=(
                "Hyperbola fitting complete.\n\n"
                f"Distance to cable: {distance_text}\n\n"
                "You may continue labeling or click Confirm & Save."
            ),
            status="Hyperbola fitting complete.",
        )

    def on_distance_to_cable_changed(self, distance_m):
        """Update the visible hyperbola while the user moves the slider."""
        self.distance_to_cable = float(distance_m)

        if self.hyperbola_active:
            self.update_hyperbola()

    def update_hyperbola(self, anchor="apex", curve_name=None):
        t_pred, x = self.calculate_hyperbola(anchor=anchor)

        if t_pred is None or x is None:
            return

        if curve_name is None:
            curve_name = anchor

        color = "white" if anchor == "apex" else "cyan"

        self.tx_plot_panel.show_hyperbola(
            time_values=t_pred,
            distance_values=x,
            curve_name=curve_name,
            color=color,
            style=QtCore.Qt.PenStyle.DashLine,
            width=2,
        )

    def on_apex_dragged(self, new_time, new_distance):
        """
        Update apex position smoothly while dragging.

        Cable-channel snapping is intentionally delayed until the user
        confirms hyperbola fitting by pressing H.
        """
        if self.apex_point is None:
            return

        new_time = float(new_time)
        new_distance = float(new_distance)

        # Apex remains continuous/smooth during dragging.
        self.apex_point = (new_time, new_distance)

        self.control_panel.apex_time_display.setText(f"{new_time:.3f}")
        self.control_panel.apex_dist_display.setText(f"{new_distance:.2f}")

        # Keep endpoint aligned with the current apex distance.
        if self.endpoint_point is not None:
            endpoint_time = float(self.endpoint_point[0])

            self.endpoint_point = (
                endpoint_time,
                new_distance,
            )

            self.tx_plot_panel.mark_endpoint_point(
                endpoint_time,
                new_distance,
            )

            duration = abs(endpoint_time - new_time)
            self.control_panel.duration_display.setText(f"{duration:.3f}")

        # Update hyperbolas continuously while dragging.
        if self.distance_to_cable is not None:
            self.update_hyperbola(anchor="apex", curve_name="apex")

            if self.endpoint_point is not None:
                self.update_hyperbola(
                    anchor="endpoint",
                    curve_name="endpoint",
                )

    def on_endpoint_dragged(self, new_time, new_distance):
        """
        Drag endpoint horizontally in time.

        Endpoint distance is always forced to the current apex channel.
        The endpoint-anchored hyperbola updates continuously.
        """
        if self.apex_point is None or self.endpoint_point is None:
            return

        new_time = float(new_time)

        # Endpoint must remain on the same cable channel as the apex.
        apex_distance = float(self.apex_point[1])

        self.endpoint_point = (
            new_time,
            apex_distance,
        )

        # Force visual marker back to the apex channel without recreating it.
        self.tx_plot_panel.set_endpoint_position(
            new_time,
            apex_distance,
        )

        duration = abs(new_time - float(self.apex_point[0]))
        self.control_panel.duration_display.setText(f"{duration:.3f}")

        # Redraw only the endpoint curve while dragging.
        if self.distance_to_cable is not None:
            self.update_hyperbola(
                anchor="endpoint",
                curve_name="endpoint",
            )

        self.notify(
            prompt=(
                "Endpoint moved.\n\n"
                f"Endpoint time: {new_time:.3f} s\n"
                f"Endpoint channel: {apex_distance:.2f} m\n"
                f"Duration: {duration:.3f} s\n\n"
                "Endpoint remains constrained to the apex channel.\n"
                "Press E to confirm endpoint."
            ),
            status="Endpoint position updated.",
        )

    def snap_apex_and_endpoint_to_channel(self):
        """
        Snap the apex to the nearest physical DAS channel.

        If an endpoint exists, keep its time unchanged but move it to the
        exact same cable channel as the snapped apex.
        """
        if self.apex_point is None:
            return

        x_vec = self.data_manager.loaded_data.get("x")

        if x_vec is None or len(x_vec) == 0:
            return

        apex_time = float(self.apex_point[0])
        apex_distance = float(self.apex_point[1])

        # Nearest actual DAS channel.
        self.apex_row_idx = int(np.argmin(np.abs(x_vec - apex_distance)))
        snapped_distance = float(x_vec[self.apex_row_idx])

        self.apex_point = (
            apex_time,
            snapped_distance,
        )

        # Recreate marker only once, at confirmation time.
        self.tx_plot_panel.mark_apex_point(
            apex_time,
            snapped_distance,
        )

        self.control_panel.apex_time_display.setText(f"{apex_time:.3f}")
        self.control_panel.apex_dist_display.setText(f"{snapped_distance:.2f}")

        # Endpoint uses its selected time but shares the apex channel.
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

            duration = abs(endpoint_time - apex_time)
            self.control_panel.duration_display.setText(f"{duration:.3f}")

        # The hyperbola changes slightly after snapping, so prior D points
        # no longer necessarily lie exactly on it.
        self.distance_point_1 = None
        self.distance_point_2 = None
        self.dist_min = None
        self.dist_max = None

        self.control_panel.dist_min_display.clear()
        self.control_panel.dist_max_display.clear()
        self.tx_plot_panel.set_distance_annotation_points([])

        # Update hyperbolas after the final snapped channel is known.
        if self.distance_to_cable is not None:
            self.update_hyperbola(anchor="apex", curve_name="apex")

            if self.endpoint_point is not None:
                self.update_hyperbola(
                    anchor="endpoint",
                    curve_name="endpoint",
                )

    def calculate_hyperbola(self, anchor="apex"):
        """
        Calculate a hyperbola anchored exactly at either the apex or endpoint.

        The minimum of the curve is guaranteed to be at:
            (anchor_time, anchor_distance)
        """
        if self.distance_to_cable is None:
            return None, None

        if anchor == "endpoint":
            anchor_point = self.endpoint_point
        else:
            anchor_point = self.apex_point

        if anchor_point is None:
            return None, None

        x = self.data_manager.loaded_data.get("x")

        if x is None or len(x) == 0:
            return None, None

        anchor_time = float(anchor_point[0])
        anchor_distance = float(anchor_point[1])
        distance_to_cable = float(self.distance_to_cable)

        # Distance along cable from the apex/endpoint channel.
        along_cable_distance = x - anchor_distance

        # Source-to-cable range:
        #
        # rng_dir = sqrt((distance along cable)^2 + (distance to cable)^2)
        #
        # This is equivalent to the straight 3-D cable/source geometry,
        # but guarantees the plotted curve uses the same coordinate system
        # as the T-X plot.
        rng_dir = np.sqrt(
            along_cable_distance**2 + distance_to_cable**2
        )

        # Shift relative travel times so the minimum is exactly at the
        # selected anchor time.
        #
        # At x == anchor_distance:
        # rng_dir == distance_to_cable
        # t_pred == anchor_time
        t_pred = anchor_time + (
            rng_dir - distance_to_cable
        ) / self.sound_speed

        return t_pred, x

    def snap_point_to_hyperbola(self, clicked_time, clicked_distance):
        """
        Snap a clicked T-X point to the nearest cable channel on the hyperbola.

        Returns
        -------
        tuple | None
            (snapped_time, snapped_distance) if click is close enough.
            None if the click is too far away from the hyperbola.
        """
        t_pred, x = self.calculate_hyperbola(anchor="apex")

        if t_pred is None or x is None:
            return None

        # Find nearest cable channel to the clicked distance.
        row_idx = int(np.argmin(np.abs(x - clicked_distance)))

        predicted_time = float(t_pred[row_idx])
        snapped_distance = float(x[row_idx])

        time_error = abs(float(clicked_time) - predicted_time)

        if time_error > self.hyperbola_click_tolerance_s:
            return None

        return predicted_time, snapped_distance

    def show_confirmed_hyperbola_segment(self, anchor="endpoint"):
        """
        Show only the hyperbola segment between the two confirmed
        distance-side points.
        """
        if self.distance_point_1 is None or self.distance_point_2 is None:
            return

        t_pred, x = self.calculate_hyperbola(anchor=anchor)

        if t_pred is None or x is None:
            return

        x_1 = float(self.distance_point_1[1])
        x_2 = float(self.distance_point_2[1])

        x_min = min(x_1, x_2)
        x_max = max(x_1, x_2)

        mask = (x >= x_min) & (x <= x_max)

        if not np.any(mask):
            return

        self.tx_plot_panel.show_hyperbola(
            time_values=t_pred[mask],
            distance_values=x[mask],
        )

    # ==============================================================
    # Saving, displaying, and deleting labels
    # ==============================================================
    def save_annotation(self, label_num):
        """
        Save the current T-X annotation and all manually drawn F-X ROI boxes.

        Required:
            - apex point
            - endpoint point
            - two distance boundary points
            - non-zero label
            - valid CSV output path

        F-X boxes are optional. If present, each box is sent to save_fx_label().
        That method expands the saved TX row's f_min / f_max across all boxes.
        """

        # ----------------------------------------------------------
        # Validate required T-X annotation values.
        # ----------------------------------------------------------
        if self.apex_point is None:
            message = "Cannot save: select an apex first."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        if self.endpoint_point is None:
            message = "Cannot save: select an endpoint first."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        if self.distance_point_1 is None or self.distance_point_2 is None:
            message = "Cannot save: select both distance boundary points first."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        if self.dist_min is None or self.dist_max is None:
            message = "Cannot save: distance minimum/maximum values are missing."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # `0: Remove` is intentionally not a new annotation label.
        if label_num == 0:
            message = "Label '0: Remove' selected. Annotation was not saved."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ----------------------------------------------------------
        # Read current settings and validate output CSV path.
        # ----------------------------------------------------------
        settings = self.control_panel.get_settings()
        labels_path = settings.get("labels_file_path", "").strip()

        if not labels_path:
            message = "Cannot save: choose a CSV file in Save Labels File."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # Create/recreate saver if the selected CSV location has changed.
        if (
            self.data_manager.label_saver is None
            or os.path.abspath(self.data_manager.label_saver.csv_path)
            != os.path.abspath(labels_path)
        ):
            from annotate.data_manager import LabelSaver
            self.data_manager.label_saver = LabelSaver(labels_path)

        # ----------------------------------------------------------
        # Build T-X annotation values.
        # ----------------------------------------------------------
        apex_time_local, apex_dist = self.apex_point
        endpoint_time_local, _ = self.endpoint_point

        apex_time_local = float(apex_time_local)
        apex_dist = float(apex_dist)
        endpoint_time_local = float(endpoint_time_local)

        duration = abs(endpoint_time_local - apex_time_local)

        # The first timestamp is the Unix timestamp at the start
        # of the currently loaded data window.
        window_start_unix = float(self.data_manager.loaded_data["time_stamps"][0])
        apex_unix = window_start_unix + apex_time_local

        apex_time_str = datetime.fromtimestamp(
            apex_unix,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        label_mapping = settings.get("label_mapping", DEFAULT_LABEL_MAPPING)
        label_name = label_mapping.get(int(label_num), "")

        dataset_name = os.path.basename(self.data_manager.directory)
        source_file = self.data_manager.filepath

        # LabelSaver.save_tx_label() currently calculates:
        # duration = max(t_s) - min(t_s)
        # dist_min / dist_max = min(x_m) / max(x_m)
        #
        # These compact arrays preserve that existing API.
        t_s = [0.0, duration]
        x_m = [float(self.dist_min), float(self.dist_max)]

        annotation_uid = str(uuid.uuid4())

        # ----------------------------------------------------------
        # Save T-X label first, then obtain its tx_id.
        # ----------------------------------------------------------
        try:
            tx_id = self.data_manager.label_saver.save_tx_label(
                uid=annotation_uid,
                apex_time_global=float(apex_unix),
                apex_time_str=apex_time_str,
                apex_time_local=apex_time_local,
                apex_dist=apex_dist,
                x_m=x_m,
                t_s=t_s,
                dataset=dataset_name,
                source_file=source_file,
                label=int(label_num),
                label_name=label_name,
                distance_to_cable=(
                    float(self.distance_to_cable)
                    if self.distance_to_cable is not None
                    else None
                ),
            )
        except Exception as exc:
            message = f"Failed to save T-X annotation: {exc}"
            print(message)
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ----------------------------------------------------------
        # Save all F-X boxes BEFORE clearing annotation overlays.
        #
        # Dictionary layout:
        # {
        #   slice_index: [
        #       (f_min_hz, dist_min_m, f_max_hz, dist_max_m),
        #       ...
        #   ],
        # }
        #
        # save_fx_label() updates the TX row so:
        # f_min = global lowest F-X-box frequency
        # f_max = global highest F-X-box frequency
        # ----------------------------------------------------------
        rois_by_slice = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        )

        fx_dataset = self.data_manager.fx_manager.get_dataset()
        fx_times = fx_dataset.get("t")
        win_s = self.data_manager.get_user_settings("win_s") or 2.0

        fx_box_count = 0

        try:
            for slice_idx, boxes in rois_by_slice.items():
                for f_min_hz, dist_min_m, f_max_hz, dist_max_m in boxes:
                    slice_start_time = 0.0

                    if fx_times is not None and slice_idx < len(fx_times):
                        slice_start_time = float(fx_times[slice_idx])

                    self.data_manager.label_saver.save_fx_label(
                        tx_id=tx_id,
                        uid=annotation_uid,
                        f_min_hz=float(f_min_hz),
                        f_max_hz=float(f_max_hz),
                        x_min_m=float(dist_min_m),
                        x_max_m=float(dist_max_m),
                        t=slice_start_time,
                        win_length_s=float(win_s),
                        dataset=dataset_name,
                        label=int(label_num),
                        label_name=label_name,
                    )

                    fx_box_count += 1

        except Exception as exc:
            # TX label has already been saved. Warn instead of deleting it.
            message = (
                f"Saved T-X label (ID {tx_id}), but failed to save "
                f"one or more F-X boxes: {exc}"
            )
            print(message)
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ----------------------------------------------------------
        # Update existing-label overlay if it is currently visible.
        # ----------------------------------------------------------
        if self.show_labels:
            labels = self.data_manager.get_labels_in_current_window()
            self.tx_plot_panel.show_existing_labels(labels)

        # ----------------------------------------------------------
        # Tell the user the annotation was saved.
        # ----------------------------------------------------------
        message = (
            f"Saved label {label_num}: {label_name} "
            f"(TX ID {tx_id}, F-X boxes: {fx_box_count})."
        )

        self.statusBar().showMessage(message)
        self.text_display_panel.update_cursor_mode("Normal")
        self.text_display_panel.set_annotation_prompt(
            f"{message}\n\n"
            "Press A to begin another annotation."
        )

        # ----------------------------------------------------------
        # End annotation mode and clear temporary graphics/state.
        # This must occur only AFTER all F-X boxes have been saved.
        # ----------------------------------------------------------
        self.reset_annotation(clear_control_panel=True)

    def on_toggle_labels(self, show):
        """Show or hide existing labels in the current T-X data window."""
        if not show:
            self.tx_plot_panel.hide_existing_labels()
            self.control_panel.toggle_labels_button.setText(
                "Show Existing Labels"
            )
            self.statusBar().showMessage("Existing labels hidden")
            self.show_labels = False
            return

        labels = self.data_manager.get_labels_in_current_window()

        # Add reconstructed hyperbola information where possible.
        for label in labels:
            label["hyperbola_segment"] = self.get_existing_label_hyperbola_segment(
                label
            )

        self.tx_plot_panel.show_existing_labels(labels)

        self.control_panel.toggle_labels_button.setText(
            "Hide Existing Labels"
        )
        self.statusBar().showMessage(
            f"Showing {len(labels)} existing labels in window"
        )
        self.show_labels = True
    
    def on_delete_label(self, tx_id):
        """Ask for confirmation before permanently deleting a saved label."""

        if self.data_manager.label_saver is None:
            self.statusBar().showMessage("No label file is currently loaded.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Annotation",
            (
                f"Do you really want to delete annotation ID {tx_id}?\n\n"
                "This action permanently removes the label from the CSV file."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Delete cancelled.")
            return

        self.data_manager.label_saver.remove_label_by_id(tx_id)

        # Refresh visible existing-label overlays.
        if self.show_labels:
            refreshed_labels = self.data_manager.get_labels_in_current_window()
            self.tx_plot_panel.show_existing_labels(refreshed_labels)

        self.statusBar().showMessage(f"Removed label {tx_id}.")
        
    def calculate_hyperbola_for_label(
        self,
        apex_time,
        apex_distance,
        distance_to_cable,
    ):
        """
        Calculate the full predicted arrival curve for a saved label.

        Returns:
            t_pred: predicted arrival times
            x: cable-distance vector
        """
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

        cable_direction = np.asarray(
            self.cable_direction_3d,
            dtype=float,
        )
        cable_direction /= np.linalg.norm(cable_direction)

        cable_normal = np.asarray(
            self.cable_normal_3d,
            dtype=float,
        )
        cable_normal /= np.linalg.norm(cable_normal)

        cable_origin = np.asarray(
            self.cable_origin_3d,
            dtype=float,
        )

        # 3-D coordinates of every cable segment/channel.
        seg = cable_origin + np.outer(x, cable_direction)

        # Closest point on the cable to the whale source.
        closest_cable_location = (
            cable_origin + apex_distance * cable_direction
        )

        # Source location perpendicular to cable at selected distance.
        source_location = (
            closest_cable_location
            + distance_to_cable * cable_normal
        )

        rng_dir = np.linalg.norm(seg - source_location, axis=1)

        # Anchor arrival curve to saved apex time.
        t_pred = apex_time + (
            rng_dir - distance_to_cable
        ) / self.sound_speed

        return t_pred, x

    def get_existing_label_hyperbola_segment(self, label):
        """
        Return a hyperbola segment for an existing label.

        Returns:
            {
                "times": np.ndarray,
                "distances": np.ndarray,
                "side_points": [(time_1, dist_1), (time_2, dist_2)]
            }

        Returns None when the label has no valid distance-to-cable annotation.
        """
        try:
            distance_to_cable = float(label.get("distance_to_cable"))
            dist_min = float(label.get("dist_min"))
            dist_max = float(label.get("dist_max"))
        except (TypeError, ValueError):
            return None

        # No fitted hyperbola exists if any required value is missing.
        if not (
            np.isfinite(distance_to_cable)
            and np.isfinite(dist_min)
            and np.isfinite(dist_max)
        ):
            return None

        t_pred, x = self.calculate_hyperbola_for_label(
            apex_time=label["apex_time_local"],
            apex_distance=label["apex_dist"],
            distance_to_cable=distance_to_cable,
        )

        if t_pred is None or x is None:
            return None

        lower_distance = min(dist_min, dist_max)
        upper_distance = max(dist_min, dist_max)

        mask = (x >= lower_distance) & (x <= upper_distance)

        if not np.any(mask):
            return None

        segment_times = t_pred[mask]
        segment_distances = x[mask]

        # Make endpoint markers precisely at nearest available cable channels.
        min_idx = int(np.argmin(np.abs(x - lower_distance)))
        max_idx = int(np.argmin(np.abs(x - upper_distance)))

        side_points = [
            (float(t_pred[min_idx]), float(x[min_idx])),
            (float(t_pred[max_idx]), float(x[max_idx])),
        ]

        return {
            "times": segment_times,
            "distances": segment_distances,
            "side_points": side_points,
        }


