import sys
from annotate.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

def run():
    """Entry point for `annotate` command."""
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()