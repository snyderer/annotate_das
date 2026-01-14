import sys
from PyQt6.QtWidgets import QApplication
from annotate.main_window import MainWindow

def run():
    """Entry point for `annotate` command."""
    app = QApplication(sys.argv)
    win = MainWindow()
    # Example: populate T-X plot
    win.tx_plot_panel.plot_sample_data()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()