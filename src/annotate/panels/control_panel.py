from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QSpinBox, QSlider, QLineEdit, QLabel,
    QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from annotate.config import (
    DEFAULT_LABEL_MAPPING, DEFAULT_DATASET_PATH,
    DEFAULT_SAVE_PATH, UserSettings
)

class ControlPanel(QWidget):
    refresh_requested = pyqtSignal()
    toggle_labels_requested = pyqtSignal(bool)

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

        # Add spec_vmin/spec_vmax sliders back
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

        # --- Labels Section ---
        labels_box = QGroupBox("Labels")
        labels_form = QFormLayout()
        self.label_edits = {}
        for i in range(1, 10):
            edit = QLineEdit()
            edit.setText(DEFAULT_LABEL_MAPPING.get(i, ""))
            labels_form.addRow(f"{i}:", edit)
            self.label_edits[i] = edit
        labels_form.addRow(QLabel("0 : remove label"))

        # Add save labels path selection
        labels_path_layout = QHBoxLayout()
        self.labels_path_edit = QLineEdit()
        self.labels_path_edit.setText(DEFAULT_SAVE_PATH)
        self.labels_browse_button = QPushButton('Browse...')
        labels_path_layout.addWidget(self.labels_path_edit)
        labels_path_layout.addWidget(self.labels_browse_button)

        labels_form.addRow("Save labels file:", labels_path_layout)
        self.labels_browse_button.clicked.connect(self.select_labels_save_path) # connect browse button to file dialog

        labels_box.setLayout(labels_form)
        main_layout.addWidget(labels_box)

        # --- Toggle Existing Labels ---
        self.toggle_labels_button = QPushButton("Show Existing Labels")
        self.toggle_labels_button.setCheckable(True)  # so it can be toggled
        main_layout.addWidget(self.toggle_labels_button)
        self.toggle_labels_button.toggled.connect(self._toggle_labels_clicked)

        # --- Apply Changes ---
        self.refresh_button = QPushButton("Apply Changes")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        main_layout.addWidget(self.refresh_button)

        main_layout.addStretch()

    def _toggle_labels_clicked(self, checked):
        """Emit signal when user toggles label visibility."""
        self.toggle_labels_requested.emit(checked)
        
    def select_labels_save_path(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Label Save File",
            DEFAULT_SAVE_PATH,
            "SQL database files (*.db);;All Files (*)"
            #"CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            self.labels_path_edit.setText(filepath)
            
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
        # Label mapping
        label_mapping = {i: edit.text().strip() for i, edit in self.label_edits.items()}
        settings['label_mapping'] = label_mapping
        return settings
    