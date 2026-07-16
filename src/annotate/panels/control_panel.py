from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QSpinBox, QSlider, QLineEdit, QLabel,
    QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from annotate.config import (
    DEFAULT_LABEL_MAPPING, DEFAULT_DATASET_PATH,
    DEFAULT_SAVE_PATH, UserSettings
)


class ControlPanel(QWidget):
    refresh_requested = pyqtSignal()
    toggle_labels_requested = pyqtSignal(bool)
    confirm_label_requested = pyqtSignal(int)   # emits label_num (0 = cancel/remove)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        defaults = UserSettings()
        main_layout = QVBoxLayout(self)

        # --- Navigation ---
        nav_box = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        self.btn_back = QPushButton("< 30s")
        self.btn_forward = QPushButton("> 30s")
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_forward)
        nav_box.setLayout(nav_layout)
        main_layout.addWidget(nav_box)

        # --- T-X Settings ---
        tx_box = QGroupBox("T-X Settings")
        tx_form = QFormLayout()
        self.tx_vmin_slider = QSlider(Qt.Orientation.Horizontal)
        self.tx_vmax_slider = QSlider(Qt.Orientation.Horizontal)
        for slider, val in [(self.tx_vmin_slider, defaults.tx_vmin),
                            (self.tx_vmax_slider, defaults.tx_vmax)]:
            slider.setRange(0, 100)
            slider.setValue(int(val * 100))
        tx_form.addRow("vmin", self.tx_vmin_slider)
        tx_form.addRow("vmax", self.tx_vmax_slider)
        tx_box.setLayout(tx_form)
        main_layout.addWidget(tx_box)

        # --- F-X Settings ---
        fx_box = QGroupBox("F-X Settings")
        fx_form = QFormLayout()
        self.fx_win_s_spin = QDoubleSpinBox()
        self.fx_win_s_spin.setMinimum(1)
        self.fx_win_s_spin.setValue(defaults.fx_win_s)
        fx_form.addRow("FX window size (s)", self.fx_win_s_spin)

        self.fx_vmin_slider = QSlider(Qt.Orientation.Horizontal)
        self.fx_vmax_slider = QSlider(Qt.Orientation.Horizontal)
        for slider, val in [(self.fx_vmin_slider, defaults.fx_vmin),
                            (self.fx_vmax_slider, defaults.fx_vmax)]:
            slider.setRange(0, 100)
            slider.setValue(int(val * 100))
        fx_form.addRow("vmin", self.fx_vmin_slider)
        fx_form.addRow("vmax", self.fx_vmax_slider)
        fx_box.setLayout(fx_form)
        main_layout.addWidget(fx_box)

        # --- Spectrogram Settings ---
        spec_box = QGroupBox("Spectrogram Settings")
        spec_form = QFormLayout()

        self.nfft_spin = QSpinBox()
        self.nfft_spin.setMinimum(32)
        self.nfft_spin.setMaximum(2**13)
        self.nfft_spin.setSingleStep(10)
        self.nfft_spin.setValue(defaults.nfft)
        spec_form.addRow("Nfft", self.nfft_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 99)
        self.overlap_spin.setValue(defaults.overlap)
        spec_form.addRow("% Overlap", self.overlap_spin)

        self.spec_vmin_slider = QSlider(Qt.Orientation.Horizontal)
        self.spec_vmax_slider = QSlider(Qt.Orientation.Horizontal)
        self.spec_vmin_slider.setRange(0, 100)
        self.spec_vmax_slider.setRange(0, 100)
        self.spec_vmin_slider.setValue(int(defaults.spec_vmin * 100))
        self.spec_vmax_slider.setValue(int(defaults.spec_vmax * 100))
        spec_form.addRow("Spec vmin", self.spec_vmin_slider)
        spec_form.addRow("Spec vmax", self.spec_vmax_slider)

        spec_box.setLayout(spec_form)
        main_layout.addWidget(spec_box)

        # =====================================================
        # --- Annotation Section (replaces old "Labels" box) ---
        # =====================================================
        annot_box = QGroupBox("Annotation")
        annot_form = QFormLayout()

        # Label dropdown (fixed classes from config)
        self.label_dropdown = QComboBox()
        self._label_keys = list(DEFAULT_LABEL_MAPPING.keys())
        for key in self._label_keys:
            name = DEFAULT_LABEL_MAPPING[key]
            self.label_dropdown.addItem(f"{key}: {name}", userData=key)
        annot_form.addRow("Label:", self.label_dropdown)

        # Read-only computed fields
        self.apex_time_display = QLineEdit()
        self.apex_dist_display = QLineEdit()
        self.duration_display = QLineEdit()
        self.dist_to_cable_display = QLineEdit()
        self.dist_max_display = QLineEdit()
        self.dist_min_display = QLineEdit()
        self.freq_max_display = QLineEdit()
        self.freq_min_display = QLineEdit()

        self._readonly_fields = [
            self.apex_time_display, self.apex_dist_display,
            self.duration_display, self.dist_to_cable_display,
            self.dist_max_display, self.dist_min_display,
            self.freq_max_display, self.freq_min_display
        ]
        for field in self._readonly_fields:
            field.setReadOnly(True)   # read-only for now, editable later

        annot_form.addRow("Apex time (s):", self.apex_time_display)
        annot_form.addRow("Apex dist (m):", self.apex_dist_display)
        annot_form.addRow("Duration (s):", self.duration_display)
        annot_form.addRow("Dist. to cable (m):", self.dist_to_cable_display)

        # Combine dist max/min into one row
        dist_row_layout = QHBoxLayout()
        dist_row_layout.addWidget(QLabel("max:"))
        dist_row_layout.addWidget(self.dist_max_display)
        dist_row_layout.addWidget(QLabel("min:"))
        dist_row_layout.addWidget(self.dist_min_display)
        annot_form.addRow("Dist (m):", dist_row_layout)

        # Combine freq max/min into one row
        freq_row_layout = QHBoxLayout()
        freq_row_layout.addWidget(QLabel("max:"))
        freq_row_layout.addWidget(self.freq_max_display)
        freq_row_layout.addWidget(QLabel("min:"))
        freq_row_layout.addWidget(self.freq_min_display)
        annot_form.addRow("Freq (Hz):", freq_row_layout)

        # Confirm & Save button
        self.confirm_save_button = QPushButton("Confirm && Save")
        self.confirm_save_button.clicked.connect(self._on_confirm_save_clicked)
        annot_form.addRow(self.confirm_save_button)

        annot_box.setLayout(annot_form)
        main_layout.addWidget(annot_box)

        # --- Save labels file path ---
        path_box = QGroupBox("Save Labels File")
        path_layout = QHBoxLayout()
        self.labels_path_edit = QLineEdit()
        self.labels_path_edit.setText(DEFAULT_SAVE_PATH)
        self.labels_browse_button = QPushButton('Browse...')
        path_layout.addWidget(self.labels_path_edit)
        path_layout.addWidget(self.labels_browse_button)
        path_box.setLayout(path_layout)
        main_layout.addWidget(path_box)
        self.labels_browse_button.clicked.connect(self.select_labels_save_path)

        # --- Toggle Existing Labels ---
        self.toggle_labels_button = QPushButton("Show Existing Labels")
        self.toggle_labels_button.setCheckable(True)
        main_layout.addWidget(self.toggle_labels_button)
        self.toggle_labels_button.toggled.connect(self._toggle_labels_clicked)

        # --- Apply Changes ---
        self.refresh_button = QPushButton("Apply Changes")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        main_layout.addWidget(self.refresh_button)

        main_layout.addStretch()

    #####################################################################
    # Signal handlers
    #####################################################################
    def _toggle_labels_clicked(self, checked):
        self.toggle_labels_requested.emit(checked)

    def _on_confirm_save_clicked(self):
        label_num = self.label_dropdown.currentData()
        self.confirm_label_requested.emit(label_num)

    def select_labels_save_path(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Label Save File",
            DEFAULT_SAVE_PATH,
            "CSV Files (*.csv);;All Files (*)"
        )
        if filepath:
            self.labels_path_edit.setText(filepath)

    #####################################################################
    # Populate preview fields (called externally by MainWindow)
    #####################################################################
    def update_annotation_preview(self, values: dict):
        """
        values keys expected:
        apex_time_local, apex_dist, duration, distance_to_cable,
        dist_max, dist_min, f_max, f_min
        """
        self.apex_time_display.setText(f"{values.get('apex_time_local', ''):.3f}"
                                        if values.get('apex_time_local') is not None else "")
        self.apex_dist_display.setText(f"{values.get('apex_dist', ''):.2f}"
                                        if values.get('apex_dist') is not None else "")
        self.duration_display.setText(f"{values.get('duration', ''):.3f}"
                                       if values.get('duration') is not None else "")
        self.dist_to_cable_display.setText(f"{values.get('distance_to_cable', ''):.2f}"
                                            if values.get('distance_to_cable') is not None else "")
        self.dist_max_display.setText(f"{values.get('dist_max', ''):.2f}"
                                       if values.get('dist_max') is not None else "")
        self.dist_min_display.setText(f"{values.get('dist_min', ''):.2f}"
                                       if values.get('dist_min') is not None else "")
        self.freq_max_display.setText(f"{values.get('f_max', ''):.2f}"
                                       if values.get('f_max') is not None else "")
        self.freq_min_display.setText(f"{values.get('f_min', ''):.2f}"
                                       if values.get('f_min') is not None else "")

    def clear_annotation_preview(self):
        for field in self._readonly_fields:
            field.clear()

    #####################################################################
    # Settings dict (label_mapping now comes from fixed config, not edits)
    #####################################################################
    def get_settings(self):
        """Return current UI values as a dict."""
        settings = {
            'win_s': self.fx_win_s_spin.value(),
            'nfft': self.nfft_spin.value(),
            'overlap': self.overlap_spin.value(),
            'tx_vmin': self.tx_vmin_slider.value(),
            'tx_vmax': self.tx_vmax_slider.value(),
            'fx_vmin': self.fx_vmin_slider.value(),
            'fx_vmax': self.fx_vmax_slider.value(),
            'spec_vmin': self.spec_vmin_slider.value(),
            'spec_vmax': self.spec_vmax_slider.value(),
            'labels_file_path': self.labels_path_edit.text().strip()
        }
        # Label mapping is now fixed (from config), not user-editable text boxes
        settings['label_mapping'] = DEFAULT_LABEL_MAPPING
        return settings