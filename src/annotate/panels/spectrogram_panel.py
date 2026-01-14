from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from annotate.config import PLOTCOLOR_LUT, UserSettings
import numpy as np

class SpectrogramPanel(QWidget):
    """
    Spectrogram panel shows frequency vs time for a single row of TX or FX data.
    """

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.vmin = UserSettings.spec_vmin
        self.vmax = UserSettings.spec_vmax
        self.use_db = False
        self.last_row_idx = None  # will be set on first click

        self.data_manager.settings_changed.connect(self.on_settings_changed)

        # Layout and single plot
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        layout.addWidget(self.plot_widget)

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('bottom', 'time', units='s')
        plot_item.setLabel('left', 'frequency', units='Hz')
        self.plot_widget.setTitle("Spectrogram")

        # Image for spectrogram
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.img_item.setLookupTable(PLOTCOLOR_LUT)
        self.img_item.setVisible(False)
        self.plot_widget.addItem(self.img_item)

        

    def update_plot(self, row_idx=None):
        """
        Update spectrogram for the selected cable distance row.
        Called when user clicks in TXPlotPanel or FXPlotPanel.
        """
        if row_idx is None:
            # If no row passed, use the last selected row
            if self.last_row_idx is None:
                return
            row_idx = self.last_row_idx
        else:
            self.last_row_idx = row_idx

        self.update_settings()

        # Use SpectrogramHandle to calculate the data
        freqs, times, Sxx = self.data_manager.spectrogram_manager.calc_spectrogram(row_idx)

        if self.use_db:
            Sxx = 20 * np.log10(np.maximum(Sxx, 1e-12))

        # Validate levels vs data range
        levels = (self.vmin, self.vmax)
        if self.vmin is None or self.vmax is None:
            levels = (np.nanmin(Sxx), np.nanmax(Sxx))

        # Debug output to help tune sliders
        print(f"SPECTROGRAM: vmin={self.vmin}, vmax={self.vmax}, data min={np.min(Sxx)}, data max={np.max(Sxx)}")

        self.img_item.setImage(Sxx, levels=levels)
        self.img_item.setVisible(True)

        # Transform coords for correct frequency/time axes
        f0, f1 = freqs[0], freqs[-1]
        t0, t1 = times[0], times[-1]
        width = Sxx.shape[1]
        height = Sxx.shape[0]
        tr = pg.QtGui.QTransform()
        tr.scale((t1 - t0) / width, (f1 - f0) / height)
        tr.translate(t0, f0)
        self.img_item.setTransform(tr)

        dist_val = self.data_manager.loaded_data['x'][row_idx]
        self.plot_widget.setTitle(f"Spectrogram (row {row_idx}: {dist_val:.2f} m)")

    def on_settings_changed(self):
        self.update_settings()
        self.refresh_with_current_row()

    def update_settings(self):
        """Get user-defined settings from Control Panel via data_manager."""
        user_settings = self.data_manager.get_user_settings()
        self.vmin = user_settings['spec_vmin']/100
        self.vmax = user_settings['spec_vmax']/100
        self.use_db = user_settings.get('spec_use_db', self.use_db)
        self.data_manager.spectrogram_manager.nfft = user_settings['nfft']
        self.data_manager.spectrogram_manager.percent_overlap = user_settings['overlap']

    def refresh_with_current_row(self):
        if hasattr(self, "last_row_idx") and self.last_row_idx is not None:
            self.update_plot(self.last_row_idx)

    def highlight_time_window(self, t_start, t_end):
        """Draw dashed vertical lines to indicate the time window."""
        if hasattr(self, 'highlight_lines'):
            for line in self.highlight_lines:
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass

        pen = QPen(Qt.GlobalColor.darkGray)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(0.2)

        line1 = pg.InfiniteLine(pos=t_start, angle=90, pen=pen)
        line2 = pg.InfiniteLine(pos=t_end,   angle=90, pen=pen)
        self.plot_widget.addItem(line1)
        self.plot_widget.addItem(line2)

        self.highlight_lines = [line1, line2]