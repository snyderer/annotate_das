# annotate/panels/text_display_panel.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class TextDisplayPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File info section
        self.filename_label = QLabel("File: Not loaded")
        self.filename_label.setStyleSheet("font-weight: bold; font-size: 12px; color: black;")
        
        self.timestamp_label = QLabel("Start Time: --")
        self.timestamp_label.setStyleSheet("font-size: 12px; color: black;")
       
        # Add some spacing
        separator = QLabel("─" * 20)
        separator.setStyleSheet("color: gray;")

        # Cursor mode label
        self.cursor_mode_label = QLabel("Cursor Mode: Normal")
        self.cursor_mode_label.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")

        # Add some spacing
        separator = QLabel("─" * 20)
        separator.setStyleSheet("color: gray;")

        # Add metadata label for dataset information
        self.metadata_label = QLabel("Dataset metadata: --")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet("font-size: 11px; color: black;")
        layout.addWidget(self.metadata_label)

        # Layout order
        layout.addWidget(self.filename_label)
        layout.addWidget(self.timestamp_label)
        layout.addWidget(separator)
        layout.addWidget(self.cursor_mode_label)

        # Another place to add more status/info text later
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        layout.addStretch()
    
    def update_file_info(self, filename, timestamp_str):
        """Update the file and timestamp information."""
        self.filename_label.setText(f"File: {filename}")
        self.timestamp_label.setText(f"Start Time: {timestamp_str}")

    def update_cursor_mode(self, mode_text):
        """Change the text for cursor mode."""
        self.cursor_mode_label.setText(f"Cursor Mode: {mode_text}")

    def update_info_text(self, info_text):
        """Change the content of the info label."""
        self.info_label.setText(info_text)
        
    def update_dataset_metadata(self, record) -> None:
        """
        Display source metadata from DatasetService.FileRecord.
        """
        metadata = record.metadata

        gauge_length = metadata.get("gauge_length_m", "unknown")

        self.metadata_label.setText(
            "<b>Dataset metadata</b><br>"
            f"Interrogator: {record.interrogator}<br>"
            f"Sampling rate: {record.fs_hz:g} Hz<br>"
            f"Channel spacing: {record.dx_m:.3f} m<br>"
            f"Cable range: {metadata['start_distance_m']:.1f} – "
            f"{metadata['end_distance_m']:.1f} m<br>"
            f"Samples/file: {record.n_samples}<br>"
            f"Channels/file: {record.n_channels}<br>"
            f"Gauge length: {gauge_length} m"
        )