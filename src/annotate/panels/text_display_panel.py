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

        # Annotation workflow prompt
        self.annotation_prompt_label = QLabel("No annotation in progress.")
        self.annotation_prompt_label.setWordWrap(True)
        self.annotation_prompt_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #0033cc; "
            "background-color: #f4f7ff; "
            "border: 1px solid #c8d5f0; "
            "padding: 6px;"
        )


        # Layout order
        layout.addWidget(self.filename_label)
        layout.addWidget(self.timestamp_label)
        layout.addWidget(separator)
        layout.addWidget(self.cursor_mode_label)
        layout.addWidget(self.annotation_prompt_label)

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

    def set_annotation_prompt(self, text):
        """Show the current annotation instruction beside the spectrogram."""
        self.annotation_prompt_label.setText(text)