from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QFileDialog
from PyQt6 import QtCore
from annotate.data_manager import DataManager
from annotate.panels.settings_panel import SettingsPanel
from annotate.panels.tx_plot_panel import TXPlotPanel
from annotate.panels.spectrogram_panel import SpectrogramPanel
from annotate.panels.fx_plot_panel import FXPlotPanel
from annotate.panels.fx_series_panel import FXSeriesPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotate DAS Data")
        self.resize(1400, 900)
        self.showMaximized()

        # Central data manager
        self.data_manager = DataManager()

        # Central widget + layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        hsplit = QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left
        self.settings_panel = SettingsPanel()
        hsplit.addWidget(self.settings_panel.widget)

        # Middle
        middle_split = QSplitter(QtCore.Qt.Orientation.Vertical)
        self.tx_plot_panel = TXPlotPanel()
        middle_split.addWidget(self.tx_plot_panel.widget)
        
        self.spectrogram_panel = SpectrogramPanel()
        middle_split.addWidget(self.spectrogram_panel.widget)
        hsplit.addWidget(middle_split)

        # Right
        right_split = QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fx_plot_panel = FXPlotPanel()
        right_split.addWidget(self.fx_plot_panel.widget)
        
        self.fx_series_panel = FXSeriesPanel()
        right_split.addWidget(self.fx_series_panel.widget)
        hsplit.addWidget(right_split)

        layout.addWidget(hsplit)

        # Connect panels to central data manager signal
        self.data_manager.dataset_loaded.connect(self.tx_plot_panel.update_plot)
        self.data_manager.dataset_loaded.connect(self.fx_plot_panel.update_plot)
        self.data_manager.dataset_loaded.connect(self.fx_series_panel.update_plot)
        self.data_manager.dataset_loaded.connect(self.spectrogram_panel.update_plot)

        # Menu bar
        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        select_dataset_action = file_menu.addAction("Select Preprocessed Dataset")
        select_dataset_action.triggered.connect(self.select_dataset)

    def select_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Preprocessed Dataset",
            "",
            "Data Files (*.csv *.txt *.h5 *.npy);;All Files (*)"
        )
        if file_path:
            self.data_manager.load_dataset(file_path)