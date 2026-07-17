from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QFileDialog, QScrollArea
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer
from datetime import datetime, timezone
import os, uuid

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
    STAGE_ENDPOINT = "select_endpoint"
    STAGE_APEX_DONE = "apex_complete"
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

        # === Navigation buttons ===
        self.control_panel.btn_back.clicked.connect(lambda: self.data_manager.navigate('backward'))
        self.control_panel.btn_forward.clicked.connect(lambda: self.data_manager.navigate('forward'))

        # === Apply Changes button and Add Labels button ===
        self.control_panel.toggle_labels_requested.connect(self.on_toggle_labels)
        self.control_panel.refresh_requested.connect(self.on_apply_changes)
        self.control_panel.confirm_label_requested.connect(self.save_annotation)

        # === Apex, Endpoint, and Distance labeling buttons ===
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

        self.clear_all_annotation_overlays()

        if clear_control_panel:
            self.control_panel.clear_annotation_preview()

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
            self.select_endpoint(t_val, x_val, clicked_distance)

        elif self.annotation_stage == self.STAGE_DIST_POINTS:
            self.select_distance_point(t_val, x_val)

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
        """A key: apex -> endpoint -> apex complete."""

        if self.cursor_mode != self.MODE_ANNOTATION:
            self.reset_annotation()
            self.set_mode(self.MODE_ANNOTATION, self.STAGE_APEX, True)

            self.notify(
                cursor_mode="Apex labeling",
                prompt=(
                    "Apex labeling:\n"
                    "Click the apex in the T-X plot.\n\n"
                    "Click again to replace it.\n"
                    "Press A for endpoint labeling."
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

            self.annotation_stage = self.STAGE_ENDPOINT

            self.notify(
                cursor_mode="Endpoint labeling",
                prompt=(
                    "Endpoint labeling:\n"
                    "Click near the call endpoint.\n\n"
                    "The endpoint will snap to the apex channel.\n"
                    "Press A when done."
                ),
                status="Select the endpoint.",
            )
            return

        if self.annotation_stage == self.STAGE_ENDPOINT:
            if self.endpoint_point is None:
                self.notify(
                    prompt="No endpoint selected yet.\nClick near the call endpoint.",
                    status="Select an endpoint first.",
                )
                return

            self.annotation_stage = self.STAGE_APEX_DONE

            self.notify(
                cursor_mode="Apex label done",
                prompt="Apex label done.\n\nPress D to select distance boundaries.",
                status="Apex labeling complete.",
            )
            return

        if self.annotation_stage == self.STAGE_APEX_DONE:
            self.annotation_stage = self.STAGE_APEX

            self.notify(
                cursor_mode="Apex editing",
                prompt="Click a new apex location.\n\nPress A for endpoint labeling.",
                status="Apex editing active.",
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

    def select_endpoint(self, t_val, x_val, clicked_distance):
        if self.apex_point is None or self.apex_row_idx is None:
            self.notify(
                prompt="No apex selected.\nPress A to return to apex labeling.",
                status="No apex selected.",
            )
            return

        apex_distance = float(
            self.data_manager.loaded_data["x"][self.apex_row_idx]
        )

        if clicked_distance is None:
            clicked_distance = x_val

        dx = float(self.data_manager.h5settings.get("dx") or 1.0)
        tolerance = 100.0 * dx

        if abs(float(clicked_distance) - apex_distance) > tolerance:
            self.notify(
                prompt=(
                    "Endpoint click is too far from the apex channel.\n\n"
                    f"Apex channel: {apex_distance:.2f} m\n"
                    f"Clicked: {clicked_distance:.2f} m\n\n"
                    "Click closer to the apex channel."
                ),
                status="Endpoint click is too far from apex channel.",
            )
            return

        self.endpoint_point = (t_val, apex_distance)

        duration = abs(t_val - float(self.apex_point[0]))

        self.tx_plot_panel.mark_endpoint_point(t_val, apex_distance)
        self.control_panel.duration_display.setText(f"{duration:.3f}")

        self.notify(
            prompt=(
                f"Endpoint selected:\n\n"
                f"Time: {t_val:.3f} s\n"
                f"Duration: {duration:.3f} s\n\n"
                "Click again to replace it.\n"
                "Press A when done."
            ),
            status="Endpoint updated.",
        )

    # ==============================================================
    # T-X distance workflow
    # ==============================================================
    def handle_distance_key(self):
        """
        D key workflow:

        First D:
            enter distance-point selection mode.

        During selection:
            clicks retain only the latest two points.

        Second D:
            finish distance labeling.
        """
        if self.apex_point is None or self.endpoint_point is None:
            self.notify(
                prompt="Complete apex and endpoint labeling before distance labeling.",
                status="Apex and endpoint are required first.",
            )
            return

        # Start/restart distance labeling.
        if self.annotation_stage in (
            self.STAGE_APEX_DONE,
            self.STAGE_DIST_DONE,
            self.STAGE_FX_DONE,
            self.STAGE_NONE,
        ):
            self.distance_point_1 = None
            self.distance_point_2 = None
            self.dist_min = None
            self.dist_max = None

            self.control_panel.dist_min_display.clear()
            self.control_panel.dist_max_display.clear()

            self.set_mode(
                self.MODE_ANNOTATION,
                self.STAGE_DIST_POINTS,
                tx_annotation_active=True,
            )

            self.notify(
                cursor_mode="Distance labeling",
                prompt=(
                    "Distance labeling:\n\n"
                    "Click two points on the lower side of the call.\n\n"
                    "If you click more than two points, only the latest two "
                    "points are retained.\n\n"
                    "Press D when the distance range is correct."
                ),
                status="Select two distance points.",
            )
            return

        # Finish distance labeling.
        if self.annotation_stage == self.STAGE_DIST_POINTS:
            if self.distance_point_1 is None or self.distance_point_2 is None:
                self.notify(
                    prompt=(
                        "Select two distance points before finishing.\n\n"
                        "The latest two selected points define distance min/max."
                    ),
                    status="Two distance points are required.",
                )
                return

            self.annotation_stage = self.STAGE_DIST_DONE

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

    def select_distance_point(self, t_val, x_val):
        """
        Keep only the newest two distance points.

        Click 1: point_1
        Click 2: point_1 + point_2
        Click 3+: discard oldest point, retain newest two.
        """
        new_point = (t_val, x_val)

        if self.distance_point_1 is None:
            self.distance_point_1 = new_point

            # Show the first yellow marker immediately.
            self.update_distance_markers()

            self.notify(
                prompt=(
                    "First distance point selected.\n\n"
                    f"Distance: {x_val:.2f} m\n\n"
                    "Click a second point to calculate min/max.\n"
                    "Press D when both points are correct."
                ),
                status="First distance point selected.",
            )
            return

        if self.distance_point_2 is None:
            self.distance_point_2 = new_point
        else:
            # Keep only the latest two clicked points.
            self.distance_point_1 = self.distance_point_2
            self.distance_point_2 = new_point

        d1 = float(self.distance_point_1[1])
        d2 = float(self.distance_point_2[1])

        self.dist_min = min(d1, d2)
        self.dist_max = max(d1, d2)

        self.control_panel.dist_min_display.setText(f"{self.dist_min:.2f}")
        self.control_panel.dist_max_display.setText(f"{self.dist_max:.2f}")

        # Redraw yellow markers using the latest two points.
        self.update_distance_markers()

        self.notify(
            prompt=(
                "Distance points selected.\n\n"
                f"Point 1: {self.distance_point_1[1]:.2f} m\n"
                f"Point 2: {self.distance_point_2[1]:.2f} m\n"
                f"Minimum distance: {self.dist_min:.2f} m\n"
                f"Maximum distance: {self.dist_max:.2f} m\n\n"
                "Click again to replace the oldest point.\n"
                "Press D when distance labeling is done."
            ),
            status="Distance range updated.",
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
        """Show or hide existing TX labels in current time window."""
        if show:
            labels = self.data_manager.get_labels_in_current_window()
            self.tx_plot_panel.show_existing_labels(labels)
            self.statusBar().showMessage(f"Showing {len(labels)} existing labels in window")
            self.control_panel.toggle_labels_button.setText("Hide Existing Labels")
            self.show_labels = True
        else:
            self.tx_plot_panel.hide_existing_labels()
            self.statusBar().showMessage("Existing labels hidden")
            self.control_panel.toggle_labels_button.setText("Show Existing Labels")
            self.show_labels = False
    
    def on_delete_label(self, tx_id):
        """Remove label from DB and refresh overlays."""
        if self.data_manager.label_saver:
            self.data_manager.label_saver.remove_label_by_id(tx_id)
            refreshed = self.data_manager.get_labels_in_current_window()
            self.tx_plot_panel.show_existing_labels(refreshed)
            self.statusBar().showMessage(f"Removed label {tx_id}")
    





