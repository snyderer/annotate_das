from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QFileDialog, QScrollArea
from PyQt6 import QtCore
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
        bottom_hsplit.addWidget(self.text_display_panel)
        middle_vsplit.addWidget(bottom_hsplit)

        # Make text display scrollable:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Make width adjust with splitter
        scroll_area.setWidget(self.text_display_panel)

        bottom_hsplit.addWidget(scroll_area)
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

        # Build menu
        self.create_menu()

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

    def on_fx_slice_selected(self, idx):
        """User clicked an FX series thumbnail."""
        self.fx_plot_panel.show_slice_from_series(idx)

        # Highlight in FX series panel
        self.fx_series_panel.highlight_slice(idx)

        dataset = self.data_manager.fx_manager.get_dataset()
        times = dataset["t"]
        if times is None or idx >= len(times):
            return
        win_s = self.data_manager.get_user_settings('win_s') or 2.0
        t_start = times[idx]
        t_end = t_start + win_s
        self.tx_plot_panel.highlight_time_window(t_start, t_end)
        if hasattr(self.spectrogram_panel, "last_row_idx") and self.spectrogram_panel.last_row_idx is not None:
            self.spectrogram_panel.highlight_time_window(t_start, t_end)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_S:
            # Spectrogram row selection mode
            self.cursor_mode = 's'
            self.data_manager.set_cursor_mode(self.cursor_mode)
            self.text_display_panel.update_cursor_mode("select spectrogram row")
            self.statusBar().showMessage("Spectrogram selection: Click a T-X or F-X plot")
        elif event.key() == QtCore.Qt.Key.Key_A:
            # ----------------------------------------------------------
            # Start apex selection
            # ----------------------------------------------------------
            if self.cursor_mode != "annotation":
                self.clear_all_annotation_overlays()
                self.control_panel.clear_annotation_preview()

                self.apex_point = None
                self.apex_row_idx = None
                self.endpoint_point = None

                self.cursor_mode = "annotation"
                self.data_manager.set_cursor_mode(self.cursor_mode)

                self.annotation_stage = "select_apex"
                self.tx_plot_panel.annotation_mode_active = True

                self.text_display_panel.update_cursor_mode("Apex labeling")
                self.text_display_panel.set_annotation_prompt(
                    "Apex labeling:\n"
                    "Click the apex in the T-X plot.\n\n"
                    "You may click again to replace it.\n"
                    "Press A when ready to select the endpoint."
                )
                self.statusBar().showMessage(
                    "Select or adjust the apex. Press A when ready for endpoint selection."
                )

            # ----------------------------------------------------------
            # Apex selection -> endpoint selection
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_apex":
                if self.apex_point is None:
                    self.statusBar().showMessage(
                        "Select an apex point before switching to endpoint labeling."
                    )
                    self.text_display_panel.set_annotation_prompt(
                        "No apex selected yet.\n"
                        "Click an apex in the T-X plot first."
                    )
                    return

                self.annotation_stage = "select_endpoint"

                self.text_display_panel.update_cursor_mode("Endpoint labeling")
                self.text_display_panel.set_annotation_prompt(
                    "Endpoint labeling:\n"
                    "Click the end of the call on the same channel as the apex.\n\n"
                    "You may click again to replace it.\n"
                    "Press A when the endpoint is final."
                )
                self.statusBar().showMessage(
                    "Select or adjust endpoint on the same channel. Press A when done."
                )

            # ----------------------------------------------------------
            # Endpoint selection -> apex/endpoint complete
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_endpoint":
                if self.endpoint_point is None:
                    self.statusBar().showMessage(
                        "Select an endpoint before completing apex labeling."
                    )
                    self.text_display_panel.set_annotation_prompt(
                        "No endpoint selected yet.\n"
                        "Click an endpoint on the same channel as the apex."
                    )
                    return

                self.annotation_stage = "apex_complete"

                self.text_display_panel.update_cursor_mode("Apex label done")
                self.text_display_panel.set_annotation_prompt(
                    "Apex label done.\n\n"
                    "Press D to label the minimum and maximum distances."
                )
                self.statusBar().showMessage(
                    "Apex label done. Press D to label distance range."
                )

            # ----------------------------------------------------------
            # Allow restart/edit after completion
            # ----------------------------------------------------------
            elif self.annotation_stage == "apex_complete":
                self.annotation_stage = "select_apex"

                self.text_display_panel.update_cursor_mode("Apex labeling")
                self.text_display_panel.set_annotation_prompt(
                    "Apex editing:\n"
                    "Click a new apex location.\n\n"
                    "Press A when ready to select the endpoint."
                )
                self.statusBar().showMessage("Editing apex. Click a new apex point.")
        elif event.key() == QtCore.Qt.Key.Key_D:
            # Distance labeling requires a completed apex + endpoint selection.
            if self.apex_point is None or self.endpoint_point is None:
                message = "Complete apex and endpoint labeling before labeling distances."
                self.statusBar().showMessage(message)
                self.text_display_panel.set_annotation_prompt(message)
                return

            # ----------------------------------------------------------
            # Start / restart distance point 1 selection
            # ----------------------------------------------------------
            if self.annotation_stage in ("apex_complete", "distance_complete", ""):
                self.distance_point_1 = None
                self.distance_point_2 = None
                self.dist_min = None
                self.dist_max = None

                self.control_panel.dist_min_display.clear()
                self.control_panel.dist_max_display.clear()

                self.tx_plot_panel.set_distance_annotation_points([])

                self.cursor_mode = "annotation"
                self.data_manager.set_cursor_mode(self.cursor_mode)
                self.tx_plot_panel.annotation_mode_active = True

                self.annotation_stage = "select_distance_1"

                self.text_display_panel.update_cursor_mode(
                    "Distance labeling: first point"
                )
                self.text_display_panel.set_annotation_prompt(
                    "Distance labeling:\n\n"
                    "Click the first point on the lower side of the call.\n\n"
                    "Click again to replace it.\n"
                    "Press D to select the second point."
                )
                self.statusBar().showMessage(
                    "Select the first distance point. Press D for the second point."
                )

            # ----------------------------------------------------------
            # Distance point 1 -> distance point 2
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_distance_1":
                if self.distance_point_1 is None:
                    message = "Select the first distance point before pressing D."
                    self.statusBar().showMessage(message)
                    self.text_display_panel.set_annotation_prompt(message)
                    return

                self.annotation_stage = "select_distance_2"

                self.text_display_panel.update_cursor_mode(
                    "Distance labeling: second point"
                )
                self.text_display_panel.set_annotation_prompt(
                    "Select the second distance point.\n\n"
                    "Click the second point on the lower side of the call.\n\n"
                    "Click again to replace it.\n"
                    "Press D when distance labeling is done."
                )
                self.statusBar().showMessage(
                    "Select the second distance point. Press D when done."
                )

            # ----------------------------------------------------------
            # Distance point 2 -> complete
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_distance_2":
                if self.distance_point_2 is None:
                    message = "Select the second distance point before pressing D."
                    self.statusBar().showMessage(message)
                    self.text_display_panel.set_annotation_prompt(message)
                    return

                self.annotation_stage = "distance_complete"

                self.text_display_panel.update_cursor_mode("Distance label done")
                self.text_display_panel.set_annotation_prompt(
                    "Distance label done.\n\n"
                    f"Minimum distance: {self.dist_min:.2f} m\n"
                    f"Maximum distance: {self.dist_max:.2f} m\n\n"
                    "Select a label in the left panel,\n"
                    "then click Confirm & Save."
                )
                self.statusBar().showMessage(
                    "Distance label done. Select a label and click Confirm & Save."
                )
        elif event.key() == QtCore.Qt.Key.Key_Escape:
            # Cancel any active mode
            self.cursor_mode = ''
            self.data_manager.set_cursor_mode(self.cursor_mode)
            self.annotation_stage = ''
            self.tx_plot_panel.annotation_mode_active = False
            self.text_display_panel.update_cursor_mode("Normal")
            self.statusBar().showMessage("Normal mode")
            self.clear_all_annotation_overlays()           
        else:
            super().keyPressEvent(event)

    def start_apex_labeling(self):
        self.clear_all_annotation_overlays()
        self.control_panel.clear_annotation_preview()

        self.cursor_mode = "annotation"
        self.data_manager.set_cursor_mode(self.cursor_mode)

        self.annotation_stage = "select_apex"
        self.tx_plot_panel.annotation_mode_active = True

        self.apex_row_idx = None
        self.apex_point = None
        self.endpoint_point = None

        self.text_display_panel.update_cursor_mode(
            "Apex labeling: click the call apex in the T-X plot."
        )
        self.statusBar().showMessage("Click the apex of the call.")


    def on_point_clicked(self, row_idx, col_idx, clicked_distance=None):
        if self.cursor_mode == '':
            return
        elif self.cursor_mode == 's':
            dist_val = self.data_manager.loaded_data['x'][row_idx]
            self.spectrogram_panel.update_plot(row_idx)
            self.tx_plot_panel.mark_distance(dist_val)
            self.fx_plot_panel.mark_distance(dist_val)

            self.cursor_mode = ''
            self.text_display_panel.update_cursor_mode("Normal")
            self.statusBar().showMessage(
                f"Spectrogram updated for row {row_idx} ({dist_val:.2f} m)"
            )
        elif self.cursor_mode == "annotation":
            t_val = float(self.data_manager.loaded_data["t"][col_idx])
            x_val = float(self.data_manager.loaded_data["x"][row_idx])

            # ----------------------------------------------------------
            # Apex selection: every click replaces the current apex.
            # ----------------------------------------------------------
            if self.annotation_stage == "select_apex":
                self.apex_row_idx = row_idx
                self.apex_point = (t_val, x_val)

                self.tx_plot_panel.mark_apex_point(t_val, x_val)

                self.control_panel.apex_time_display.setText(f"{t_val:.3f}")
                self.control_panel.apex_dist_display.setText(f"{x_val:.2f}")

                # An old endpoint is no longer valid if apex changes.
                self.endpoint_point = None
                self.control_panel.duration_display.clear()
                self.tx_plot_panel.clear_endpoint_point()

                self.text_display_panel.set_annotation_prompt(
                    f"Apex selected:\n"
                    f"Time: {t_val:.3f} s\n"
                    f"Distance: {x_val:.2f} m\n\n"
                    "Click again to replace apex, or press A for endpoint labeling."
                )

                self.statusBar().showMessage(
                    "Apex updated. Click again to replace, or press A for endpoint."
                )

            # ----------------------------------------------------------
            # Endpoint selection: every valid click replaces endpoint.
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_endpoint":
                if self.apex_point is None or self.apex_row_idx is None:
                    message = "No apex selected. Press A to return to apex labeling."
                    self.statusBar().showMessage(message)
                    self.text_display_panel.set_annotation_prompt(message)
                    return

                # The endpoint must be stored on the same channel as the apex.
                apex_distance = float(
                    self.data_manager.loaded_data["x"][self.apex_row_idx]
                )

                # Channel spacing from settings.h5. Fallback is used if unavailable.
                dx = float(self.data_manager.h5settings.get("dx") or 1.0)

                # Allow clicks within two channels of the apex channel.
                # Increase/decrease 2.0 if desired.
                snap_tolerance_m = 100.0 * dx

                # `clicked_distance` is raw cursor location, before nearest-row snapping.
                # If click came from FXPlotPanel, fall back to the nearest TX row value.
                if clicked_distance is None:
                    clicked_distance = x_val

                distance_from_apex_channel = abs(clicked_distance - apex_distance)

                # Reject only clicks that are genuinely far away from the apex channel.
                if distance_from_apex_channel > snap_tolerance_m:
                    message = (
                        "Endpoint click is too far from the apex channel.\n\n"
                        f"Apex channel: {apex_distance:.2f} m\n"
                        f"Clicked: {clicked_distance:.2f} m\n"
                        f"Allowed distance: ±{snap_tolerance_m:.2f} m\n\n"
                        "Click closer to the apex channel."
                    )
                    self.statusBar().showMessage(
                        "Endpoint click is too far from the apex channel."
                    )
                    self.text_display_panel.set_annotation_prompt(message)
                    return

                # Snap endpoint to apex channel.
                # Keep time from the user click, but use the apex row/distance.
                row_idx = self.apex_row_idx
                x_val = apex_distance

                self.endpoint_point = (t_val, x_val)
                duration = abs(t_val - self.apex_point[0])

                self.tx_plot_panel.mark_endpoint_point(t_val, x_val)
                self.control_panel.duration_display.setText(f"{duration:.3f}")

                self.text_display_panel.set_annotation_prompt(
                    "Endpoint selected and snapped to apex channel.\n\n"
                    f"Endpoint time: {t_val:.3f} s\n"
                    f"Channel: {x_val:.2f} m\n"
                    f"Duration: {duration:.3f} s\n\n"
                    "Click again to replace endpoint, or press A when done."
                )

                self.statusBar().showMessage(
                    f"Endpoint selected at {t_val:.3f} s and snapped to apex channel."
                )
            # ----------------------------------------------------------
            # Distance point 1 selection.
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_distance_1":
                self.distance_point_1 = (t_val, x_val)

                # Point 2 remains invalid if point 1 changes.
                self.distance_point_2 = None
                self.dist_min = None
                self.dist_max = None

                self.control_panel.dist_min_display.clear()
                self.control_panel.dist_max_display.clear()
                self._update_distance_markers()

                self.text_display_panel.set_annotation_prompt(
                    "First distance point selected.\n\n"
                    f"Distance: {x_val:.2f} m\n\n"
                    "Click again to replace this point.\n"
                    "Press D to select the second distance point."
                )
                self.statusBar().showMessage(
                    "First distance point updated. Press D to select the second point."
                )

            # ----------------------------------------------------------
            # Distance point 2 selection.
            # ----------------------------------------------------------
            elif self.annotation_stage == "select_distance_2":
                if self.distance_point_1 is None:
                    message = "First distance point is missing. Press D to restart."
                    self.statusBar().showMessage(message)
                    self.text_display_panel.set_annotation_prompt(message)
                    return

                self.distance_point_2 = (t_val, x_val)

                d1 = float(self.distance_point_1[1])
                d2 = float(self.distance_point_2[1])

                self.dist_min = min(d1, d2)
                self.dist_max = max(d1, d2)

                self.control_panel.dist_min_display.setText(f"{self.dist_min:.2f}")
                self.control_panel.dist_max_display.setText(f"{self.dist_max:.2f}")
                self._update_distance_markers()

                self.text_display_panel.set_annotation_prompt(
                    "Second distance point selected.\n\n"
                    f"Minimum distance: {self.dist_min:.2f} m\n"
                    f"Maximum distance: {self.dist_max:.2f} m\n\n"
                    "Click again to replace this point.\n"
                    "Press D when distance labeling is done."
                )
                self.statusBar().showMessage(
                    "Second distance point updated. Press D when distance labeling is done."
                )

    def _update_distance_markers(self):
        """Show the current distance point 1 and point 2 on the T-X plot."""
        points = []

        if self.distance_point_1 is not None:
            points.append(self.distance_point_1)

        if self.distance_point_2 is not None:
            points.append(self.distance_point_2)

        self.tx_plot_panel.set_distance_annotation_points(points)

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
    
    def clear_all_annotation_overlays(self):
        self.tx_plot_panel.clear_annotation_overlays()
        self.fx_plot_panel.clear_annotation_overlays()
        self.fx_series_panel.clear_annotation_overlays()
        if hasattr(self.data_manager, "annotation_rois_per_slice"):
            self.data_manager.annotation_rois_per_slice.clear()

    def save_annotation(self, label_num):
        """
        Save the currently selected apex, endpoint, and distance annotations.

        Called by ControlPanel's 'Confirm & Save' button.
        """

        # ----------------------------------------------------------
        # Validate that the annotation is complete.
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
            message = "Cannot save: select both minimum and maximum distance points."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        if self.dist_min is None or self.dist_max is None:
            message = "Cannot save: distance minimum/maximum values are missing."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # Label 0 is currently named "Remove" in DEFAULT_LABEL_MAPPING.
        # Treat it as cancel / do not save.
        if label_num == 0:
            message = "Label '0: Remove' selected. Annotation was not saved."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ----------------------------------------------------------
        # Ensure LabelSaver exists.
        # ----------------------------------------------------------
        settings = self.control_panel.get_settings()
        labels_path = settings.get("labels_file_path", "").strip()

        if not labels_path:
            message = "Cannot save: choose a CSV file in 'Save Labels File'."
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # Recreate saver if none exists or the selected CSV path changed.
        if (
            self.data_manager.label_saver is None
            or os.path.abspath(self.data_manager.label_saver.csv_path)
            != os.path.abspath(labels_path)
        ):
            from annotate.data_manager import LabelSaver
            self.data_manager.label_saver = LabelSaver(labels_path)

        # ----------------------------------------------------------
        # Build values for the existing LabelSaver API.
        # ----------------------------------------------------------
        apex_time_local, apex_dist = self.apex_point
        endpoint_time_local, _ = self.endpoint_point

        duration = abs(endpoint_time_local - apex_time_local)

        # Unix timestamp for apex.
        window_start_unix = float(self.data_manager.loaded_data["time_stamps"][0])
        apex_unix = window_start_unix + float(apex_time_local)

        apex_time_str = datetime.fromtimestamp(
            apex_unix,
            tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        label_mapping = settings.get("label_mapping", DEFAULT_LABEL_MAPPING)
        label_name = label_mapping.get(label_num, "")

        dataset_name = os.path.basename(self.data_manager.directory)
        source_file = self.data_manager.filepath

        # LabelSaver currently derives:
        # duration = max(t_s) - min(t_s)
        # dist_min/max = min/max(x_m)
        #
        # These small arrays preserve compatibility without changing LabelSaver.
        t_s = [0.0, float(duration)]
        x_m = [float(self.dist_min), float(self.dist_max)]

        # Each saved annotation gets a fresh UUID.
        annotation_uid = str(uuid.uuid4())

        # ----------------------------------------------------------
        # Save the TX annotation.
        # ----------------------------------------------------------
        try:
            tx_id = self.data_manager.label_saver.save_tx_label(
                uid=annotation_uid,
                apex_time_global=float(apex_unix),
                apex_time_str=apex_time_str,
                apex_time_local=float(apex_time_local),
                apex_dist=float(apex_dist),
                x_m=x_m,
                t_s=t_s,
                dataset=dataset_name,
                source_file=source_file,
                label=int(label_num),
                label_name=label_name,
            )
        except Exception as exc:
            message = f"Failed to save annotation: {exc}"
            print(message)
            self.statusBar().showMessage(message)
            self.text_display_panel.set_annotation_prompt(message)
            return

        # ----------------------------------------------------------
        # Optional: save F-X ranges, if your existing F-X workflow
        # has supplied annotation_fx_boxes_per_plot.
        # ----------------------------------------------------------
        fx_boxes = getattr(
            self.data_manager,
            "annotation_fx_boxes_per_plot",
            []
        )

        if fx_boxes:
            for coords in fx_boxes:
                if coords is None:
                    continue

                f_min_hz, x_min_m, f_max_hz, x_max_m = coords

                self.data_manager.label_saver.save_fx_label(
                    tx_id=tx_id,
                    uid=annotation_uid,
                    f_min_hz=float(f_min_hz),
                    f_max_hz=float(f_max_hz),
                    x_min_m=float(x_min_m),
                    x_max_m=float(x_max_m),
                    dataset=dataset_name,
                    label=int(label_num),
                    label_name=label_name,
                )

        # ----------------------------------------------------------
        # Refresh displayed existing labels if enabled.
        # ----------------------------------------------------------
        if self.show_labels:
            labels = self.data_manager.get_labels_in_current_window()
            self.tx_plot_panel.show_existing_labels(labels)

        message = (
            f"Saved label {label_num}: {label_name} "
            f"(TX ID {tx_id})."
        )

        self.statusBar().showMessage(message)
        self.text_display_panel.update_cursor_mode("Normal")
        self.text_display_panel.set_annotation_prompt(
            f"{message}\n\n"
            "Press A to begin another annotation."
        )

        # End active annotation mode and remove temporary overlays.
        self.cursor_mode = ""
        self.data_manager.set_cursor_mode("")
        self.annotation_stage = ""
        self.tx_plot_panel.annotation_mode_active = False

        self.clear_all_annotation_overlays()

        # Clear values so it is obvious that a new annotation must begin.
        self.control_panel.clear_annotation_preview()

        # Reset values held by MainWindow.
        self.apex_point = None
        self.apex_row_idx = None
        self.endpoint_point = None
        self.distance_point_1 = None
        self.distance_point_2 = None
        self.dist_min = None
        self.dist_max = None