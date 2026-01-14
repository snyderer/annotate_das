from PyQt6.QTCore import QObject, pyqtSignal
import numpy as np
from preprocess_DAS.data_io import load_rehydrate_preprocessed_h5, load_settings_preprocessed_h5

class DataManager(QObject):
    dataset_loaded = pyqtSignal(object)

    def __init__(self):
        super().__init()
        self.dataset = None

    def load_dataset(self, file_path):
        self.h5settings = load_settings_preprocessed_h5(file_path)
        self.tx_data = load_rehydrate_preprocessed_h5(file_path)

    