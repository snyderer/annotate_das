from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
import numpy as np
from annotate.config import PLOTCOLOR_LUT


class FXPlotPanel(QWidget):
    """
    Large single FX plot view.
    """
    point_clicked = pyqtSignal(int, int)  # row index, col index

    def __init__(self, data_manager, vmin=0, vmax=0.4):
        super().__init__()
        self.data_manager = data_manager
        self.vmin = vmin
        self.vmax = vmax
        self.use_db = False
        self.current_slice_idx = 0

        self.data_manager.dataset_loaded.connect(self.on_dataset_loaded)
        self.data_manager.settings_changed.connect(self.on_settings_changed)

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        layout.addWidget(self.plot_widget)

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('bottom', 'frequency', units='Hz')
        plot_item.setLabel('left', 'cable distance', units='m')
        self.plot_widget.setTitle("FX plot")

        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.img_item.setLookupTable(PLOTCOLOR_LUT)
        self.plot_widget.addItem(self.img_item)

        # Track ROIs for current slice
        self.fx_slice_rois = []

        # Connect mouse clicks
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)

    #############################
    # Plotting and updating
    #############################
    def on_settings_changed(self):
        self.update_settings()
        idx = getattr(self.data_manager, 'current_fx_index', 0)
        self.show_slice(idx)

    def on_dataset_loaded(self):
        self.update_settings()
        self.data_manager.current_fx_index = 0
        self.show_slice(0)

    def update_settings(self):
        user_settings = self.data_manager.get_user_settings()
        self.vmin = user_settings.get('fx_vmin', self.vmin)
        self.vmax = user_settings.get('fx_vmax', self.vmax)
        self.use_db = user_settings.get('fx_use_db', self.use_db)

    def show_slice_from_series(self, idx):
        self.current_slice_idx = idx
        self.show_slice(idx)

    def show_slice(self, idx):
        dataset = self.data_manager.fx_manager.get_dataset()
        amp = dataset["amp"]
        freq = dataset["freq"]
        x = dataset["x"]
        times = dataset["t"]

        if amp is None or idx >= amp.shape[0]:
            return

        # Display FX image
        img_data = amp[idx, :, :]
        levels = (self.vmin, self.vmax)
        if self.use_db:
            img_data = 20 * np.log10(np.maximum(img_data, 1e-12))
            levels = (20 * np.log10(max(levels[0], 1e-12)),
                      20 * np.log10(max(levels[1], 1e-12)))

        self.img_item.setImage(img_data, levels=levels)
        f0, f1 = freq[0], freq[-1]
        x0, x1 = x[0], x[-1]
        width, height = img_data.shape[1], img_data.shape[0]
        tr = pg.QtGui.QTransform()
        tr.scale((f1 - f0) / width, (x1 - x0) / height)
        tr.translate(f0, x0)
        self.freq_band = [f0, f1]
        self.img_item.setTransform(tr)

        self.plot_widget.setTitle(f"FX plot @ T={times[idx]:.2f}s")

        # Clear the single fx_roi if it exists from start_fx_box_annotation
        if hasattr(self, 'fx_roi'):
            try:
                self.plot_widget.removeItem(self.fx_roi)
            except Exception:
                pass
            del self.fx_roi

        # Clear any ROIs from previous slice
        for roi in self.fx_slice_rois:
            try:
                self.plot_widget.removeItem(roi)
            except Exception:
                pass
        self.fx_slice_rois = []

        # Add ROIs for this slice from stored coordinates
        coords_list = getattr(self.data_manager, "annotation_rois_per_slice", {}).get(idx, [])
        for (fmin, dmin, fmax, dmax) in coords_list:
            w = fmax - fmin
            h = dmax - dmin
            roi = pg.RectROI(
                [fmin, dmin],
                [w, h],
                pen={'color': 'r', 'width': 2},
                movable=True, resizable=True
            )
            roi.sigRegionChanged.connect(lambda r=roi: self._update_roi_coordinates(r))
            self.plot_widget.getPlotItem().addItem(roi)
            self.fx_slice_rois.append(roi)

    ################################
    # ROI Update
    ################################
    def _update_roi_coordinates(self, roi):
        """Update data_manager storage when an ROI is moved/resized."""
        pos = roi.pos()
        size = roi.size()
        coords_list = self.data_manager.annotation_rois_per_slice.get(self.current_slice_idx, [])
        for i, r in enumerate(self.fx_slice_rois):
            if r is roi:
                coords_list[i] = (pos.x(), pos.y(),
                                  pos.x() + size[0], pos.y() + size[1])
                break

    ################################
    # Mouse Interactions
    ################################
    def _roi_at_click(self, freq_click, dist_click):
        """Return index of ROI that contains the click point, else None."""
        if not self.fx_slice_rois:
            return None
        for i, roi in enumerate(self.fx_slice_rois):
            pos = roi.pos()
            size = roi.size()
            f0, d0 = pos.x(), pos.y()
            f1, d1 = f0 + size[0], d0 + size[1]
            if f0 <= freq_click <= f1 and d0 <= dist_click <= d1:
                return i
        return None


    def on_mouse_click(self, ev):
        mp = self.plot_widget.getPlotItem().vb.mapSceneToView(ev.scenePos())
        f_click, d_click = mp.x(), mp.y()

        # Ctrl-click = delete
        if ev.modifiers() == Qt.KeyboardModifier.ControlModifier:
            idx = self._roi_at_click(f_click, d_click)
            if idx is not None:
                roi = self.fx_slice_rois.pop(idx)
                try:
                    self.plot_widget.removeItem(roi)
                except Exception:
                    pass
                coords_list = self.data_manager.annotation_rois_per_slice.get(self.current_slice_idx, [])
                if idx < len(coords_list):
                    coords_list.pop(idx)

        # Ctrl+Shift-click = add
        elif ev.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            w = 10.0  # default width in Hz
            h = 20.0  # default height in m
            roi_pos = [f_click - w / 2, d_click - h / 2]
            roi = pg.RectROI(
                roi_pos, [w, h],
                pen={'color': 'r', 'width': 2},
                movable=True, resizable=True
            )
            roi.sigRegionChanged.connect(lambda r=roi: self._update_roi_coordinates(r))
            self.plot_widget.getPlotItem().addItem(roi)
            self.fx_slice_rois.append(roi)
            coords_list = self.data_manager.annotation_rois_per_slice.setdefault(self.current_slice_idx, [])
            coords_list.append((roi_pos[0], roi_pos[1], roi_pos[0] + w, roi_pos[1] + h))

    ###############################
    # Other
    ###############################
    def mark_distance(self, distance_value):
        if hasattr(self, 'distance_arrow') and self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass
        self.distance_arrow = pg.ScatterPlotItem(
            x=self.freq_band,
            y=[distance_value, distance_value],
            symbol='d', size=12,
            brush='red', pen='red'
        )
        self.plot_widget.addItem(self.distance_arrow)

    def start_fx_box_annotation(self):
        plot_item = self.plot_widget.getPlotItem()
        freq_range = plot_item.viewRange()[0]
        dist_range = plot_item.viewRange()[1]
        w = (freq_range[1] - freq_range[0]) * 0.3
        h = (dist_range[1] - dist_range[0]) * 0.3
        roi_pos = [freq_range[0] + w * 0.5, dist_range[0] + h * 0.5]
        self.fx_roi = pg.RectROI(
            roi_pos, [w, h],
            pen={'color': 'r', 'width': 2},
            movable=True, resizable=True
        )
        plot_item.addItem(self.fx_roi)

    def get_fx_box_coordinates(self):
        if not hasattr(self, 'fx_roi'):
            return None
        pos = self.fx_roi.pos()
        size = self.fx_roi.size()
        freq_min = pos.x()
        freq_max = pos.x() + size[0]
        dist_min = pos.y()
        dist_max = pos.y() + size[1]
        return freq_min, dist_min, freq_max, dist_max

    def clear_fx_box_annotation(self):
        if hasattr(self, 'fx_roi'):
            try:
                self.plot_widget.removeItem(self.fx_roi)
            except Exception:
                pass
            del self.fx_roi

    def clear_annotation_overlays(self):
        """remove all annotation ROIs from the plot."""
        self.clear_fx_box_annotation()
        for roi in self.fx_slice_rois:
            try:
                self.plot_widget.removeItem(roi)
            except Exception:
                pass
        self.fx_slice_rois = []

        if hasattr(self, 'distance_arrow') and self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass
            self.distance_arrow = None