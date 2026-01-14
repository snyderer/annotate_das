from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGraphicsRectItem
from PyQt6.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
import numpy as np
from annotate.config import PLOTCOLOR_LUT, UserSettings

class FXSeriesPanel(QWidget):
    slice_selected = pyqtSignal(int)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.vmin = UserSettings.fx_vmin
        self.vmax = UserSettings.fx_vmax
        self.use_db = False

        self.data_manager.dataset_loaded.connect(self.on_dataset_loaded)

        self.plot_widgets = []
        self.plot_img_items = []
        self.fx_rois = []  # keep track of adjustable ROIs
        self.highlight_idx = None

        outer_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self.scroll_area)

        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.scroll_area.setWidget(self.container_widget)

    def on_dataset_loaded(self):
        self.update_settings()
        dataset = self.data_manager.fx_manager.get_dataset()
        self.set_plot_data(dataset)

    def update_settings(self):
        user_settings = self.data_manager.get_user_settings()
        self.vmin = user_settings.get('fx_vmin', self.vmin)
        self.vmax = user_settings.get('fx_vmax', self.vmax)
        self.use_db = user_settings.get('fx_use_db', self.use_db)

    def set_plot_data(self, dataset):
        if not dataset or dataset["amp"] is None:
            return

        amp = dataset["amp"]
        freq = dataset["freq"]
        x = dataset["x"]
        times = dataset["t"]
        nt = amp.shape[0]

        # Clear existing
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.plot_widgets.clear()
        self.plot_img_items.clear()
        self.fx_rois.clear()
        self.highlight_idx = None

        # Determine levels
        if self.vmin is not None and self.vmax is not None:
            levels = (self.vmin, self.vmax)
        else:
            levels = (np.nanmin(amp), np.nanmax(amp))

        if self.use_db:
            amp = 20*np.log10(np.maximum(amp, 1e-12))
            levels = (20*np.log10(max(levels[0], 1e-12)),
                      20*np.log10(max(levels[1], 1e-12)))

        # Build per-slice plot widgets
        for idx in range(nt):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.setMinimumHeight(180)
            plot_item = plot_widget.getPlotItem()
            plot_item.setLabel('bottom', 'frequency', units='Hz')
            plot_item.setLabel('left', 'cable distance', units='m')
            plot_item.setTitle(f"T = {times[idx]:.2f} s")

            img_item = pg.ImageItem(axisOrder='row-major')
            img_item.setLookupTable(PLOTCOLOR_LUT)
            img_data = amp[idx, :, :]

            img_item.setImage(img_data, levels=levels)

            # Coordinate transform
            f0, f1 = freq[0], freq[-1]
            x0, x1 = x[0], x[-1]
            width, height = img_data.shape[1], img_data.shape[0]
            tr = pg.QtGui.QTransform()
            tr.scale((f1 - f0) / width, (x1 - x0) / height)
            tr.translate(f0, x0)
            img_item.setTransform(tr)

            plot_widget.addItem(img_item)

            plot_widget.scene().sigMouseClicked.connect(
                lambda evt, idx=idx: self.slice_selected.emit(idx)
            )

            self.container_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_img_items.append(img_item)

    def highlight_slice(self, idx):
        if self.highlight_idx is not None:
            self.plot_widgets[self.highlight_idx].setStyleSheet("")
        self.plot_widgets[idx].setStyleSheet("border: 3px solid red;")
        self.highlight_idx = idx

    def add_adjustable_rois_to_all(self, box_coords, tx_contour_points):
        """
        Place movable/resizable RectROIs on each FX subplot for 'adjust_fx_energy' stage.
        Vertically position based on T-X contour distances for that time window.
        If contour has multiple separate vertical sections in the window,
        make one ROI per section.
        """
        self.fx_rois.clear()
        dataset = self.data_manager.fx_manager.get_dataset()
        times = np.array(dataset["t"])  # FX slice start times
        win_s = self.data_manager.get_user_settings('win_s') or 2.0

        if not box_coords or tx_contour_points is None:
            return

        freq_min, dist_min, freq_max, dist_max = box_coords
        w = freq_max - freq_min
        h = dist_max - dist_min

        contour_times = np.array([p[0] for p in tx_contour_points])
        contour_dists = np.array([p[1] for p in tx_contour_points])

        min_ct = contour_times.min()
        max_ct = contour_times.max()

        # Ensure central store exists
        if not hasattr(self.data_manager, "annotation_rois_per_slice"):
            self.data_manager.annotation_rois_per_slice = {}

        for idx, plot_widget in enumerate(self.plot_widgets):
            # FX slice time window
            t_start = times[idx]
            t_end = t_start + win_s

            if t_end < min_ct or t_start > max_ct:
                continue  # skip if no contour overlaps

            # Indices of contour points in this time window
            idx_in_window = np.where((contour_times >= t_start) & (contour_times <= t_end))[0]
            if len(idx_in_window) == 0:
                continue

            # Breaks in contiguous contour within this FX window
            breaks = np.where(np.diff(idx_in_window) > 1)[0]
            # Add start and end markers for segment slicing
            segment_bounds = [0] + (breaks + 1).tolist() + [len(idx_in_window)]

            # List to store ROI coords for this slice
            slice_coords = []

            for sb in range(len(segment_bounds) - 1):
                seg_idx = idx_in_window[segment_bounds[sb]:segment_bounds[sb + 1]]
                mean_dist = np.mean(contour_dists[seg_idx])

                # Position ROI so vertical center is mean_dist
                roi_y = mean_dist - h / 2

                roi = pg.RectROI(
                    [freq_min, roi_y],  # x = freq_min, y = distance start
                    [w, h],             # width = frequency span, height = distance span
                    pen={'color': 'r', 'width': 2},
                    movable=True, resizable=True
                )
                plot_widget.getPlotItem().addItem(roi)
                self.fx_rois.append(roi)

                # Save coordinates for this ROI
                slice_coords.append((freq_min, roi_y, freq_min + w, roi_y + h))

            # Store all ROI coords for this slice index
            if slice_coords:
                self.data_manager.annotation_rois_per_slice[idx] = slice_coords

    def get_all_rois_coordinates(self):
        """
        Get list of (freq_min, dist_min, freq_max, dist_max) for each adjustable ROI.
        """
        coords = []
        for roi in self.fx_rois:
            pos = roi.pos()
            size = roi.size()
            coords.append((
                pos.x(),
                pos.y(),
                pos.x() + size[0],
                pos.y() + size[1]
            ))
        return coords
    
    def clear_annotation_overlays(self):
        for roi in self.fx_rois:
            roi.scene().removeItem(roi)
        self.fx_rois = []