from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg

class SpectrogramPanel:
    def __init__(self):
        self.widget = pg.PlotWidget()
        self.widget.setBackground('w')
        self.widget.setTitle("Spectrogram")

    def plot_sample_data(self):
        import numpy as np
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.widget.plot(x, y, pen='b')