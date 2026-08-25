from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np
import scipy.signal as sp
from annotate.config import PLOTCOLOR_LUT, UserSettings


class TXPlotPanel(QWidget):
    point_clicked = pyqtSignal(int, int, float)  # row index, time-column index, raw clicked distance in metres
    label_delete_requested = pyqtSignal(int)      # Emits tx_id for DB deletion
    apex_dragged = pyqtSignal(float, float)  # time_s, distance_m
    endpoint_dragged = pyqtSignal(float, float)  # time_s, distance_m

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

        # Multiple hyperbola overlays can be displayed simultaneously.
        self.hyperbola_items = {
            "apex": None,
            "endpoint": None,
            "segment": None,
        }

        # Annotation points
        self.annotation_points_list = []    # list of (time, dist)
        self.annotation_points_item = None
        self.annotation_line_item = None
        
        # New apex / endpoint / distance markers
        self.apex_point_item = None
        self.endpoint_point_item = None
        self._updating_endpoint_position = False  # Prevent recursive updates when dragging endpoint
        self.distance_annotation_item = None
        self.distance_annotation_points = []

        # Existing saved labels currently drawn on the T-X plot
        self.existing_label_items = []
        self.existing_labels_metadata = []

        self.hyperbola_item = None

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
        """
        Show saved labels.

        Colors:
        - Red dots: apex and distance-side points.
        - Yellow dashed line: saved/reconstructed hyperbola segment.
        """
        self.hide_existing_labels()

        for label in labels:
            apex_time = float(label["apex_time_local"])
            apex_distance = float(label["apex_dist"])

            # Always show saved apex as a red dot.
            apex_item = pg.ScatterPlotItem(
                x=[apex_time],
                y=[apex_distance],
                symbol="o",
                size=8,
                pen=pg.mkPen("red", width=1),
                brush=pg.mkBrush("red"),
            )

            self.plot_widget.addItem(apex_item)
            self.existing_label_items.append(apex_item)

            self.existing_labels_metadata.append(
                {
                    "tx_id": int(label["tx_id"]),
                    "uid": label["uid"],
                    "apex_time_rel": apex_time,
                    "apex_distance": apex_distance,
                }
            )

            # Labels without distance-to-cable metadata show apex only.
            segment = label.get("hyperbola_segment")
            if not segment:
                continue

            times = segment.get("times")
            distances = segment.get("distances")
            side_points = segment.get("side_points", [])

            if times is None or distances is None or len(times) == 0:
                continue

            # Yellow dashed hyperbola segment.
            hyperbola_item = pg.PlotDataItem(
                x=times,
                y=distances,
                pen=pg.mkPen(
                    color="yellow",
                    width=2,
                    style=Qt.PenStyle.DashLine,
                ),
            )

            self.plot_widget.addItem(hyperbola_item)
            self.existing_label_items.append(hyperbola_item)

            # Red dots at the two selected distance-side points.
            if side_points:
                side_item = pg.ScatterPlotItem(
                    x=[point[0] for point in side_points],
                    y=[point[1] for point in side_points],
                    symbol="o",
                    size=8,
                    pen=pg.mkPen("red", width=1),
                    brush=pg.mkBrush("red"),
                )

                self.plot_widget.addItem(side_item)
                self.existing_label_items.append(side_item)

    def hide_existing_labels(self):
        """Remove all displayed existing-label markers and hyperbola segments."""
        for item in self.existing_label_items:
            self.plot_widget.removeItem(item)  # Corrected method call

        self.existing_label_items.clear()
        self.existing_labels_metadata.clear()

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
            self.point_clicked.emit(row_idx, col_idx, float(mp.y())) 
        elif self.data_manager.cursor_mode == 's':
            # Spectrogram select mode active
            if self.data_manager.loaded_data['amp'] is None:
                return
            x_vec = self.data_manager.loaded_data['x']
            t_vec = self.data_manager.loaded_data['t']
            row_idx = int(np.argmin(np.abs(x_vec - mp.y())))    
            col_idx = int(np.argmin(np.abs(t_vec - mp.x())))
            self.point_clicked.emit(row_idx, col_idx, float(mp.y()))  
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
    # Apex / Endpoint / Distance Annotations
    #####################################################################
    def mark_apex_point(self, time_s, distance_m):
        """
        Show the apex as a draggable red star.

        Dragging emits apex_dragged(time_s, distance_m).
        """
        self.clear_apex_point()

        self.apex_point_item = pg.TargetItem(
            pos=(time_s, distance_m),
            symbol="star",
            size=14,
            pen=pg.mkPen("red", width=2),
            brush=pg.mkBrush("red"),
            movable=True,
        )

        self.apex_point_item.sigPositionChanged.connect(
            self._on_apex_position_changed
        )

        self.plot_widget.addItem(self.apex_point_item)

    def _on_apex_position_changed(self, target_item):
        """Emit the new apex position while the user drags the apex marker."""
        pos = target_item.pos()

        self.apex_dragged.emit(
            float(pos.x()),
            float(pos.y()),
        )

    def clear_apex_point(self):
        """Remove the current apex marker."""
        if self.apex_point_item is not None:
            try:
                self.plot_widget.removeItem(self.apex_point_item)
            except Exception:
                pass

        self.apex_point_item = None

    def mark_endpoint_point(self, time_s, distance_m):
        """Show endpoint as a draggable cyan X."""
        self.clear_endpoint_point()

        self.endpoint_point_item = pg.TargetItem(
            pos=(time_s, distance_m),
            symbol="x",
            size=14,
            pen=pg.mkPen("cyan", width=2),
            brush=pg.mkBrush("cyan"),
            movable=True,
        )

        self.endpoint_point_item.sigPositionChanged.connect(
            self._on_endpoint_position_changed
        )

        self.plot_widget.addItem(self.endpoint_point_item)

    def _on_endpoint_position_changed(self, target_item):
        """Emit endpoint position while dragging."""
        if self._updating_endpoint_position:
            return

        pos = target_item.pos()

        self.endpoint_dragged.emit(
            float(pos.x()),
            float(pos.y()),
        )


    def set_endpoint_position(self, time_s, distance_m):
        """
        Move the existing endpoint marker without recreating it.

        Used while dragging so the marker remains smooth and is constrained
        to the apex cable channel.
        """
        if self.endpoint_point_item is None:
            self.mark_endpoint_point(time_s, distance_m)
            return

        self._updating_endpoint_position = True
        try:
            self.endpoint_point_item.setPos(time_s, distance_m)
        finally:
            self._updating_endpoint_position = False

    def clear_endpoint_point(self):
        """Remove the current endpoint marker."""
        if self.endpoint_point_item is not None:
            try:
                self.plot_widget.removeItem(self.endpoint_point_item)
            except Exception:
                pass

        self.endpoint_point_item = None

    def set_distance_annotation_points(self, points):
        """
        Replace visible yellow distance markers with the supplied points.

        points is a list of:
            [(time_s, distance_m), ...]
        """
        if not hasattr(self, "distance_annotation_item"):
            self.distance_annotation_item = None

        if self.distance_annotation_item is not None:
            try:
                self.plot_widget.removeItem(self.distance_annotation_item)
            except Exception:
                pass

        self.distance_annotation_item = None

        if not points:
            return

        self.distance_annotation_item = pg.ScatterPlotItem(
            x=[time_s for time_s, _ in points],
            y=[distance_m for _, distance_m in points],
            symbol="o",
            size=10,
            brush=pg.mkBrush("yellow"),
            pen=pg.mkPen("black", width=1),
        )

        self.plot_widget.addItem(self.distance_annotation_item)

    #####################################################################
    # Clear overlays
    #####################################################################
    def clear_annotation_overlays(self):
        """Remove temporary apex, endpoint, distance annotation markers, and hyperbola."""

        # Legacy contour items; safe to retain while old code remains.
        for item in (
            self.annotation_points_item,
            self.annotation_line_item,
        ):
            if item is not None:
                try:
                    self.plot_widget.removeItem(item)
                except Exception:
                    pass

        self.annotation_points_item = None
        self.annotation_line_item = None
        self.annotation_points_list.clear()

        # New annotation markers.
        self.clear_apex_point()
        self.clear_endpoint_point()
        self.set_distance_annotation_points([])
        self.clear_hyperbola()

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
    
    def show_hyperbola(self, time_values, distance_values, curve_name="apex", color="white", style=Qt.PenStyle.DashLine, width=2):
        """
        Draw or update a named hyperbola curve.

        curve_name:
            "apex"     - hyperbola anchored at apex
            "endpoint" - hyperbola anchored at endpoint
            "segment"  - final confirmed hyperbola segment
        """
        if curve_name not in self.hyperbola_items:
            self.hyperbola_items[curve_name] = None

        old_item = self.hyperbola_items[curve_name]

        if old_item is not None:
            self.plot_widget.removeItem(old_item)

        curve = pg.PlotDataItem(
            x=time_values,
            y=distance_values,
            pen=pg.mkPen(
                color=color,
                width=width,
                style=style,
            ),
        )

        self.plot_widget.addItem(curve)
        self.hyperbola_items[curve_name] = curve


    def clear_hyperbola(self, curve_name=None):
        """
        Remove one named hyperbola or all hyperbola overlays.

        curve_name=None removes every hyperbola.
        """
        if curve_name is not None:
            item = self.hyperbola_items.get(curve_name)

            if item is not None:
                self.plot_widget.removeItem(item)

            self.hyperbola_items[curve_name] = None
            return

        for name, item in self.hyperbola_items.items():
            if item is not None:
                self.plot_widget.removeItem(item)

            self.hyperbola_items[name] = None