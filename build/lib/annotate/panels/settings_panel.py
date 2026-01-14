from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox

class SettingsPanel:
    def __init__(self):
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        layout.addWidget(QLabel("Settings:"))
        layout.addWidget(QDoubleSpinBox())
        layout.addWidget(QLineEdit())

        layout.addWidget(QLabel("Spectrogram Settings:"))
        nfft_box = QSpinBox(); nfft_box.setPrefix("Nfft = ")
        overlap_box = QSpinBox(); overlap_box.setPrefix("Overlap = ")
        layout.addWidget(nfft_box)
        layout.addWidget(overlap_box)

    def get_values(self):
        # Return all current settings as a dict (you can expand later)
        return {}