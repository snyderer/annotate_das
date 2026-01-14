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
            # Annotation mode
            settings = self.control_panel.get_settings()
            labels_path = settings.get('labels_file_path')
            if labels_path:
                from annotate.data_manager import LabelSaver
                self.data_manager.label_saver = LabelSaver(labels_path)
            if self.cursor_mode != 'annotation':
                self.clear_all_annotation_overlays()
                # Start annotation mode
                self.cursor_mode = 'annotation'
                self.data_manager.set_cursor_mode(self.cursor_mode)
                self.annotation_stage = 'draw_tx_contour'
                self.tx_plot_panel.annotation_mode_active = True
                self.statusBar().showMessage("Select points along the T-X plot contour")
                self.tx_contour_points = []
                self.tx_apex_point = None
                self.text_display_panel.update_cursor_mode(
                    "Annotation: Draw T-X contour. Click along path. Press 'a' when completed"
                )
                self.statusBar().showMessage("Drawing T-X contour...")
            else:
                # Advance stage
                if self.annotation_stage == 'draw_tx_contour':
                    self.annotation_stage = 'select_apex'
                    self.statusBar().showMessage("Select apex arrival point...")
                    self.text_display_panel.update_cursor_mode(
                        "Annotation: Click T-X energy apex. \n" 
                        "Press 'a' when completed"
                    )
                elif self.annotation_stage == 'select_apex':
                    self.annotation_stage = 'box_fx_energy'
                    self.text_display_panel.update_cursor_mode(
                        "Annotation: Box F-X energy.\n Drag/resize box. \n"
                        "Press 'a' when done."
                    )
                    self.fx_plot_panel.start_fx_box_annotation()
                elif self.annotation_stage == 'box_fx_energy':
                    box_coords = self.fx_plot_panel.get_fx_box_coordinates()
                    self.data_manager.annotation_fx_box = box_coords
                    self.tx_contour_points = self.tx_plot_panel.interpolate_contour()  # or interpolated
                    # Auto-place in all subplots based on contour
                    self.fx_series_panel.add_adjustable_rois_to_all(box_coords, self.tx_contour_points)
                    self.annotation_stage = 'adjust_fx_energy'
                    self.text_display_panel.update_cursor_mode(
                        "Annotation: Adjust F-X boxes per plot. \n" 
                        "Drag/resize each if needed. \n" 
                        "Press 'a' when done."
                    )
                    
                elif self.annotation_stage == 'adjust_fx_energy':
                    all_coords = self.fx_series_panel.get_all_rois_coordinates()
                    self.data_manager.annotation_fx_boxes_per_plot = all_coords
                    # proceed to label assignment:
                    self.annotation_stage = 'assign_label'
                    self.text_display_panel.update_cursor_mode(
                        "Annotation: Assign label (1-9)."
                    )
                elif self.annotation_stage == 'assign_label':
                    self.annotation_stage = 'complete'

                elif self.annotation_stage == 'complete':
                    self.annotation_stage = ''
                    self.tx_plot_panel.annotation_mode_active = False   # turn off annotation click handling
                    self.annotation_stage = ''
                    self.cursor_mode = ''
                    self.data_manager.set_cursor_mode(self.cursor_mode)
                    self.text_display_panel.update_cursor_mode("Normal")
                    self.clear_all_annotation_overlays()
                    if self.show_labels:
                        self.on_toggle_labels(False)  # refresh existing labels display
                        self.on_toggle_labels(True)  # refresh existing labels display

        elif event.key() in [
            QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3,
            QtCore.Qt.Key.Key_4, QtCore.Qt.Key.Key_5, QtCore.Qt.Key.Key_6,
            QtCore.Qt.Key.Key_7, QtCore.Qt.Key.Key_8, QtCore.Qt.Key.Key_9
        ]:
            # Number key pressed. Assign label if in annotation mode
            if self.cursor_mode == 'annotation' and self.annotation_stage == 'assign_label':
                label_num = event.key() - QtCore.Qt.Key.Key_0
                label_mapping = self.data_manager.get_user_settings('label_mapping')
                if label_mapping is None:
                    label_mapping = DEFAULT_LABEL_MAPPING
                label_name = label_mapping.get(label_num, "")
                self.data_manager.annotation_label = label_num

                # ===== Generate UID for this detection =====
                if not hasattr(self.data_manager, 'annotation_uid'):
                    self.data_manager.annotation_uid = str(uuid.uuid4())
                self.data_manager.label_saver
                if self.data_manager.label_saver:
                    # ===== Save TX label =====
                    # Apex time absolute Unix timestamp from apex selection
                    apex_unix = float(self.data_manager.loaded_data['time_stamps'][0]) + self.tx_apex_point[0]
                    apex_str = datetime.fromtimestamp(apex_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    rel_t_s = list(map(float, [t - self.tx_apex_point[0] for t, x in self.tx_contour_points]))
                    dist_vec = list(map(float, [x for t, x in self.tx_contour_points]))

                    dataset_name = os.path.basename(self.data_manager.directory)
                    source_file = self.data_manager.filepath

                    # Save TX label and get tx_id (DB PK)
                    tx_id = self.data_manager.label_saver.save_tx_label(
                        uid=self.data_manager.annotation_uid,
                        apex_time=apex_unix,
                        apex_time_str=apex_str,
                        apex_distance=float(self.tx_apex_point[1]),
                        x_m=dist_vec,
                        t_s=rel_t_s,
                        dataset=dataset_name,
                        source_file=source_file,
                        label=label_num,
                        label_name=label_name
                    )

                    # Save FX labels linked by tx_id and with same uid for external reference
                    if hasattr(self.data_manager, 'annotation_fx_boxes_per_plot'):
                        for idx, coords in enumerate(self.data_manager.annotation_fx_boxes_per_plot):
                            f_min_hz, f_max_hz, x_min_m, x_max_m = coords
                            fx_dataset = self.data_manager.fx_manager.get_dataset()
                            start_t = fx_dataset["t"][idx] if idx < len(fx_dataset["t"]) else 0.0
                            win_s = self.data_manager.get_user_settings('win_s') or 2.0

                            self.data_manager.label_saver.save_fx_label(
                                tx_id=tx_id,
                                uid=self.data_manager.annotation_uid,
                                f_min_hz=f_min_hz,
                                f_max_hz=f_max_hz,
                                x_min_m=x_min_m,
                                x_max_m=x_max_m,
                                t=start_t,
                                win_length_s=win_s,
                                dataset=dataset_name,
                                label=label_num,
                                label_name=label_name
                            )
                            
                self.statusBar().showMessage(
                    f"Assigned label {label_num} and saved TX + FX annotations"
                )

                # End annotation mode
                self.annotation_stage = ''
                self.tx_plot_panel.annotation_mode_active = False
                self.cursor_mode = ''
                self.data_manager.set_cursor_mode(self.cursor_mode)
                self.text_display_panel.update_cursor_mode("Normal")
                self.clear_all_annotation_overlays()
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

    def on_point_clicked(self, row_idx, col_idx):
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
        elif self.cursor_mode == 'annotation':
            t_val = self.data_manager.loaded_data['t'][col_idx]
            x_val = self.data_manager.loaded_data['x'][row_idx]
            if self.annotation_stage == 'draw_tx_contour':
                self.tx_contour_points.append((t_val, x_val))
                self.tx_plot_panel.mark_annotation_point(t_val, x_val)
            elif self.annotation_stage == 'select_apex':
                self.tx_apex_point = (t_val, x_val)
                self.tx_plot_panel.mark_apex_point(t_val, x_val)

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