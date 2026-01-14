from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np
import scipy.signal as sp
from annotate.config import PLOTCOLOR_LUT, UserSettings


class TXPlotPanel(QWidget):
    point_clicked = pyqtSignal(int, int)  # Emits (row index, col index)
    label_delete_requested = pyqtSignal(int)      # Emits tx_id for DB deletion

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.vmin = UserSettings.tx_vmin
        self.vmax = UserSettings.tx_vmax

        # Hook data manager signals
        self.data_manager.dataset_loaded.connect(self.on_dataset_loaded)
        self.data_manager.settings_changed.connect(self.on_settings_changed)

        # Layout / Plot
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(background='w', title="T-X plot")
        layout.addWidget(self.plot_widget)

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('bottom', 'time', units='s')
        plot_item.setLabel('left', 'cable distance', units='m')

        # Heatmap image
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.img_item.setLookupTable(PLOTCOLOR_LUT)
        self.img_item.setVisible(False)
        self.plot_widget.addItem(self.img_item)

        # Annotation points
        self.annotation_points_list = []    # list of (time, dist)
        self.annotation_points_item = None
        self.annotation_line_item = None
        self.apex_point_item = None

        # Toggle from MainWindow when annotation stage is active
        self.annotation_mode_active = False

        # Mouse click connection
        self.plot_widget.scene().sigMouseClicked.connect(self._plot_scene_click)

    #####################################################################
    # Data Manager events
    #####################################################################
    def on_dataset_loaded(self):
        self.set_plot_data({
            "amp": self.data_manager.loaded_data['amp'],
            "t": self.data_manager.loaded_data['t'],
            "x": self.data_manager.loaded_data['x']
        })

    def set_plot_data(self, dataset):
        if dataset is None or dataset['amp'] is None:
            return

        amp, t_vec, x_vec = dataset['amp'], dataset['t'], dataset['x']
        levels = (self.vmin, self.vmax) if self.vmin is not None else (np.nanmin(amp), np.nanmax(amp))
        amp = np.clip(amp, self.vmin, self.vmax)

        self.img_item.setImage(amp, levels=levels)
        t0, t1 = t_vec[0], t_vec[-1]
        x0, x1 = x_vec[0], x_vec[-1]
        width, height = amp.shape[1], amp.shape[0]
        tr = pg.QtGui.QTransform()
        tr.scale((t1 - t0) / width, (x1 - x0) / height)
        tr.translate(t0, x0)
        self.img_item.setTransform(tr)
        self.img_item.setVisible(True)

    def on_settings_changed(self):
        self.update_settings()
        self.update_plot()

    def update_settings(self):
        self.vmin = self.data_manager.get_user_settings('tx_vmin') / 100
        self.vmax = self.data_manager.get_user_settings('tx_vmax') / 100

    def update_plot(self):
        amp = self.data_manager.loaded_data['amp']
        if amp is None:
            return
        amp = np.abs(sp.hilbert(amp, axis=1))
        self.set_plot_data({
            "amp": amp,
            "t": self.data_manager.loaded_data['t'],
            "x": self.data_manager.loaded_data['x']
        })

    def show_existing_labels(self, labels):
        self.hide_existing_labels()
        self.existing_label_items = []
        self.existing_labels_metadata = []

        for label in labels:
            # absolute times for contour = apex_time_abs + relative t_s
            t_abs = np.array(label['t_s']) + (label['apex_time'] - self.data_manager.loaded_data['time_stamps'][0])
            x_vec = np.array(label['x_m'])
            apex_time_rel = label['apex_time'] - self.data_manager.loaded_data['time_stamps'][0]

            # Draw contour as a line
            line_item = pg.PlotDataItem(
                x=t_abs,
                y=x_vec,
                pen=pg.mkPen(color='c', width=2)
            )
            self.plot_widget.addItem(line_item)
            self.existing_label_items.append(line_item)

            # Draw smaller filled apex marker
            apex_time_rel = label['apex_time'] - self.data_manager.loaded_data['time_stamps'][0]
            apex_item = pg.ScatterPlotItem(
                x=[apex_time_rel],
                y=[label['apex_distance']],
                symbol='o',
                size=6,                          # small
                pen=pg.mkPen('m', width=1),       # magenta outline
                brush=pg.mkBrush('m')             # magenta fill
            )
            self.plot_widget.addItem(apex_item)
            self.existing_label_items.append(apex_item)

            # store metadata for removal lookup
            self.existing_labels_metadata.append({
                'tx_id': label['tx_id'],
                'uid': label['uid'],
                'apex_time_rel': apex_time_rel,
                'apex_distance': label['apex_distance']
            })

    def hide_existing_labels(self):
        """Remove existing label markers from the plot."""
        if hasattr(self, 'existing_label_items'):
            for item in self.existing_label_items:
                try:
                    self.plot_widget.removeItem(item)
                except Exception:
                    pass
            self.existing_label_items.clear()
    #####################################################################
    # Plot extras
    #####################################################################
    def highlight_time_window(self, t_start, t_end):
        if hasattr(self, 'highlight_lines'):
            for l in self.highlight_lines:
                try:
                    self.plot_widget.removeItem(l)
                except Exception:
                    pass
        pen = QPen(Qt.GlobalColor.darkGray)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(0.2)
        l1 = pg.InfiniteLine(pos=t_start, angle=90, pen=pen)
        l2 = pg.InfiniteLine(pos=t_end, angle=90, pen=pen)
        self.plot_widget.addItem(l1)
        self.plot_widget.addItem(l2)
        self.highlight_lines = [l1, l2]

    def mark_distance(self, distance_value):
        if hasattr(self, 'distance_arrow') and self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass
        t0 = self.data_manager.loaded_data['t'][0]
        tend = self.data_manager.loaded_data['t'][-1]
        self.distance_arrow = pg.ScatterPlotItem(
            x=[t0, tend],
            y=[distance_value, distance_value],
            symbol='d', size=12,
            brush='red', pen='red'
        )
        self.plot_widget.addItem(self.distance_arrow)

    #####################################################################
    # Annotation points
    #####################################################################
    def mark_annotation_point(self, time_val, dist_val):
        """Add a new annotation point."""
        self.annotation_points_list.append((time_val, dist_val))
        self.update_annotation_polyline()

    def update_annotation_polyline(self):
        if not self.annotation_points_list:
            return

        sorted_pts = sorted(self.annotation_points_list, key=lambda p: p[1])
        pts_t = [p[0] for p in sorted_pts]
        pts_x = [p[1] for p in sorted_pts]

        if self.annotation_points_item is not None:
            try:
                self.plot_widget.removeItem(self.annotation_points_item)
            except Exception:
                pass
        self.annotation_points_item = pg.ScatterPlotItem(
            x=pts_t,
            y=pts_x,
            symbol='o',
            size=8,
            brush='yellow',
            pen='black'
        )
        self.plot_widget.addItem(self.annotation_points_item)

        if self.annotation_line_item is not None:
            try:
                self.plot_widget.removeItem(self.annotation_line_item)
            except Exception:
                pass
        self.annotation_line_item = pg.PlotDataItem(
            x=pts_t,
            y=pts_x,
            pen=pg.mkPen(color='y', width=2)
        )
        self.plot_widget.addItem(self.annotation_line_item)

    def _nearest_point_index(self, t_click, x_click, threshold=0.1):
        """Find closest annotation point using threshold scaled by zoom."""
        if not self.annotation_points_list:
            return None

        # Get current view limits
        view_x_range, view_y_range = self.plot_widget.getPlotItem().vb.viewRange()
        time_threshold = 0.02 * (view_x_range[1] - view_x_range[0])      # 2% of current time range
        dist_threshold = 0.02 * (view_y_range[1] - view_y_range[0])      # 2% of current distance range

        pts = np.array(self.annotation_points_list)
        time_diffs = np.abs(pts[:, 0] - t_click)
        dist_diffs = np.abs(pts[:, 1] - x_click)

        # Mask points within threshold in both dimensions
        valid_mask = (time_diffs <= time_threshold) & (dist_diffs <= dist_threshold)
        if not np.any(valid_mask):
            return None

        # Among valid points, pick the one with smallest Euclidean distance
        dists = np.sqrt(time_diffs**2 + dist_diffs**2)
        idx = np.argmin(dists)
        return idx

    def _nearest_existing_label(self, t_click, x_click, threshold=0.1):
        if not getattr(self, 'existing_labels_metadata', None):
            return None
        # measure Euclidean distance to each apex point
        distances = [
            (i, np.hypot(meta['apex_time_rel'] - t_click,
                        meta['apex_distance'] - x_click))
            for i, meta in enumerate(self.existing_labels_metadata)
        ]
        # pick smallest distance if within a reasonable threshold
        min_i, min_dist = min(distances, key=lambda p: p[1])
        threshold_t = threshold * (self.plot_widget.getPlotItem().vb.viewRange()[0][1] -
                            self.plot_widget.getPlotItem().vb.viewRange()[0][0])
        threshold_x = threshold * (self.plot_widget.getPlotItem().vb.viewRange()[1][1] -
                            self.plot_widget.getPlotItem().vb.viewRange()[1][0])
        if (abs(self.existing_labels_metadata[min_i]['apex_time_rel'] - t_click) <= threshold_t and
            abs(self.existing_labels_metadata[min_i]['apex_distance'] - x_click) <= threshold_x):
            return min_i
        return None

    def _plot_scene_click(self, ev):
        mp = self.plot_widget.getPlotItem().vb.mapSceneToView(ev.scenePos())

        if self.annotation_mode_active:
            # annotation mode active
            # Ctrl-click near point → delete
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                idx = self._nearest_point_index(mp.x(), mp.y())
                if idx is not None:
                    self.annotation_points_list.pop(idx)
                    self.update_annotation_polyline()
                    return
            # Otherwise, add a new point
            x_vec = self.data_manager.loaded_data['x']
            t_vec = self.data_manager.loaded_data['t']
            row_idx = int(np.argmin(np.abs(x_vec - mp.y())))
            col_idx = int(np.argmin(np.abs(t_vec - mp.x())))
            self.point_clicked.emit(row_idx, col_idx)
        elif self.data_manager.cursor_mode == 's':
            # Spectrogram select mode active
            if self.data_manager.loaded_data['amp'] is None:
                return
            x_vec = self.data_manager.loaded_data['x']
            t_vec = self.data_manager.loaded_data['t']
            row_idx = int(np.argmin(np.abs(x_vec - mp.y())))
            col_idx = int(np.argmin(np.abs(t_vec - mp.x())))
            self.point_clicked.emit(row_idx, col_idx)
        else:
            # no specific mode active
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier
                and hasattr(self, 'existing_labels_metadata')
                and bool(self.existing_labels_metadata)):
                # Ctrl-click near existing label → emit its tx_id for deletion
                idx = self._nearest_existing_label(mp.x(), mp.y())
                if idx is not None:
                    meta = self.existing_labels_metadata[idx]
                    self.label_delete_requested.emit(meta['tx_id'])  # send deletion request
                    return
            

    #####################################################################
    # Apex
    #####################################################################
    def mark_apex_point(self, time_val, dist_val):
        if self.apex_point_item is not None:
            try:
                self.plot_widget.removeItem(self.apex_point_item)
            except Exception:
                pass
        self.apex_point_item = pg.ScatterPlotItem(
            x=[time_val],
            y=[dist_val],
            symbol='star',
            size=12,
            brush='red',
            pen='red'
        )
        self.plot_widget.addItem(self.apex_point_item)

    #####################################################################
    # Clear overlays
    #####################################################################
    def clear_annotation_overlays(self):
        for item in (self.annotation_points_item, self.annotation_line_item, self.apex_point_item):
            if item is not None:
                try:
                    self.plot_widget.removeItem(item)
                except Exception:
                    pass
        self.annotation_points_list.clear()

    #####################################################################
    # Interpolation
    #####################################################################
    def interpolate_contour(self):
        if len(self.annotation_points_list) < 2:
            return self.annotation_points_list
        sorted_pts = sorted(self.annotation_points_list, key=lambda p: p[1])
        pts_t = [p[0] for p in sorted_pts]
        pts_x = [p[1] for p in sorted_pts]
        all_x = self.data_manager.loaded_data['x']
        min_x, max_x = min(pts_x), max(pts_x)
        mask = (all_x >= min_x) & (all_x <= max_x)
        interp_t = np.interp(all_x[mask], pts_x, pts_t)
        return list(zip(interp_t, all_x[mask]))