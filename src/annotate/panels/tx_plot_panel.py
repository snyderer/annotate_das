from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QPen
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
import pyqtgraph as pg
import numpy as np
import scipy.signal as sp
from annotate.config import PLOTCOLOR_LUT, UserSettings


class TXPlotPanel(QWidget):
    point_clicked = pyqtSignal(int, int)
    label_delete_requested = pyqtSignal(int)

    apex_dragged = pyqtSignal(float, float)
    endpoint_dragged = pyqtSignal(float, float)

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
        self.apex_point_item = None
        self.endpoint_point_item = None
        self.distance_points_item = None
        self.distance_min_line = None
        self.distance_max_line = None
        
        self._endpoint_distance_m: float | None = None
        self._updating_endpoint_position = False

        # "apex", "endpoint", "confirmed", etc.
        self.hyperbola_items: dict[str, pg.PlotDataItem] = {}

        # Existing saved annotation overlays.
        self.existing_label_items: list = []
        self.existing_labels_metadata: list[dict] = []

        # MainWindow enables this for T-X click annotation stages.
        self.annotation_mode_active = False

        # Mouse click connection
        self.plot_widget.scene().sigMouseClicked.connect(self._plot_scene_click)
        self.plot_widget.setTitle("T-X plot (nanostrain)")

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
        amp = np.abs(sp.hilbert(amp, axis=1))  # take envelope
        amp = np.clip(amp, self.vmin, self.vmax)

        self.img_item.setImage(amp, levels=levels)
        t0, t1 = float(t_vec[0]), float(t_vec[-1])
        x0, x1 = float(x_vec[0]), float(x_vec[-1])

        self.img_item.setRect(
            QRectF(
                t0,
                x0,
                t1 - t0,
                x1 - x0,
            )
        )
        self.img_item.setVisible(True)

    def on_settings_changed(self):
        self.update_settings()
        self.update_plot()

    def update_settings(self):
        settings = self.data_manager.get_user_settings()

        self.vmin = float(
            settings.get("tx_vmin", 0)
        ) / 100.0

        self.vmax = float(
            settings.get("tx_vmax", 40)
        ) / 100.0
        
    def update_plot(self):
        amp = self.data_manager.loaded_data["amp"]

        if amp is None:
            return

        self.set_plot_data({
            "amp": amp,
            "t": self.data_manager.loaded_data["t"],
            "x": self.data_manager.loaded_data["x"],
        })

    def show_existing_labels(self, labels: list[dict]) -> None:
        """
        Draw saved annotation overlays.

        Labels without a valid fitted hyperbola display only a red apex marker.

        Labels with valid hyperbola_segment display:
        - red apex marker;
        - yellow dashed hyperbola segment;
        - red markers at saved distance-side endpoints.
        """
        self.hide_existing_labels()

        window_start_unix = float(
            self.data_manager.loaded_data["time_stamps"][0]
        )

        for label in labels:
            apex_utc = label.get("apex_time_utc")
            apex_distance = label.get("apex_dist")

            if apex_utc is None or apex_distance is None:
                continue

            apex_time_relative = float(apex_utc) - window_start_unix
            apex_distance = float(apex_distance)

            # --------------------------------------------------------------
            # Saved apex marker
            # --------------------------------------------------------------
            apex_item = pg.ScatterPlotItem(
                x=[apex_time_relative],
                y=[apex_distance],
                symbol="o",
                size=8,
                pen=pg.mkPen("r", width=1),
                brush=pg.mkBrush("r"),
            )

            self.plot_widget.addItem(apex_item)
            self.existing_label_items.append(apex_item)

            self.existing_labels_metadata.append({
                "tx_id": int(label["tx_id"]),
                "uid": label.get("uid"),
                "apex_time_rel": apex_time_relative,
                "apex_distance": apex_distance,
            })

            # --------------------------------------------------------------
            # Saved hyperbola segment and distance-side markers
            # --------------------------------------------------------------
            segment = label.get("hyperbola_segment")

            if not segment:
                continue

            time_values = np.asarray(
                segment.get("times", []),
                dtype=float,
            )
            distance_values = np.asarray(
                segment.get("distances", []),
                dtype=float,
            )

            if time_values.size > 0 and distance_values.size > 0:
                line_item = pg.PlotDataItem(
                    x=time_values,
                    y=distance_values,
                    pen=pg.mkPen(
                        color="y",
                        style=Qt.PenStyle.DashLine,
                        width=2,
                    ),
                )

                self.plot_widget.addItem(line_item)
                self.existing_label_items.append(line_item)

            side_points = segment.get("side_points", [])

            if side_points:
                side_times = [point[0] for point in side_points]
                side_distances = [point[1] for point in side_points]

                side_item = pg.ScatterPlotItem(
                    x=side_times,
                    y=side_distances,
                    symbol="o",
                    size=8,
                    pen=pg.mkPen("r", width=1),
                    brush=pg.mkBrush("r"),
                )

                self.plot_widget.addItem(side_item)
                self.existing_label_items.append(side_item)

    def hide_existing_labels(self) -> None:
        """Remove all saved-label overlays."""
        for item in self.existing_label_items:
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass

        self.existing_label_items = []
        self.existing_labels_metadata = []
        
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

    def set_distance_boundaries(
        self,
        distance_min_m: float | None,
        distance_max_m: float | None,
    ) -> None:
        """
        Draw full-width horizontal min/max cable-distance boundaries.

        These are visual guides for the active annotation. They are not
        draggable in T-X; F-X owns draggable distance-boundary lines.
        """
        self.clear_distance_boundaries()

        if distance_min_m is None or distance_max_m is None:
            return

        common_options = {
            "angle": 0,
            "movable": False,
            "pen": pg.mkPen("yellow", width=2),
            "hoverPen": pg.mkPen("orange", width=3),
        }

        self.distance_min_line = pg.InfiniteLine(
            pos=float(distance_min_m),
            label="Min distance",
            labelOpts={
                "position": 0.98,
                "color": "yellow",
                "fill": (0, 0, 0, 120),
            },
            **common_options,
        )

        self.distance_max_line = pg.InfiniteLine(
            pos=float(distance_max_m),
            label="Max distance",
            labelOpts={
                "position": 0.98,
                "color": "yellow",
                "fill": (0, 0, 0, 120),
            },
            **common_options,
        )

        self.plot_widget.addItem(self.distance_min_line)
        self.plot_widget.addItem(self.distance_max_line)

    def clear_distance_boundaries(self) -> None:
        """Remove T-X min/max cable-distance boundary lines."""
        for line in (
            self.distance_min_line,
            self.distance_max_line,
        ):
            if line is not None:
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass

        self.distance_min_line = None
        self.distance_max_line = None


    def clear_distance_boundaries(self) -> None:
        """Remove T-X min/max cable-distance boundary lines."""
        for line in (
            self.distance_min_line,
            self.distance_max_line,
        ):
            if line is not None:
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass

        self.distance_min_line = None
        self.distance_max_line = None

    #####################################################################
    # Annotations
    #####################################################################

    def _nearest_existing_label(
        self,
        time_s: float,
        distance_m: float,
    ) -> int | None:
        """Return index of a nearby existing saved-label apex."""
        if not self.existing_labels_metadata:
            return None

        x_range, y_range = (
            self.plot_widget.getPlotItem()
            .vb
            .viewRange()
        )

        time_tolerance = 0.03 * abs(x_range[1] - x_range[0])
        distance_tolerance = 0.03 * abs(y_range[1] - y_range[0])

        candidates = []

        for index, metadata in enumerate(self.existing_labels_metadata):
            dt = abs(metadata["apex_time_rel"] - time_s)
            dx = abs(metadata["apex_distance"] - distance_m)

            if dt <= time_tolerance and dx <= distance_tolerance:
                candidates.append((
                    index,
                    np.hypot(
                        dt / max(time_tolerance, 1e-12),
                        dx / max(distance_tolerance, 1e-12),
                    ),
                ))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[1])[0]

    def _plot_scene_click(self, event):
        """Convert a scene click to nearest displayed data row/column."""
        if self.data_manager.loaded_data["amp"] is None:
            return

        mouse_point = (
            self.plot_widget.getPlotItem()
            .vb
            .mapSceneToView(event.scenePos())
        )

        x_vec = self.data_manager.loaded_data["x"]
        t_vec = self.data_manager.loaded_data["t"]

        row_idx = int(np.argmin(np.abs(x_vec - mouse_point.y())))
        col_idx = int(np.argmin(np.abs(t_vec - mouse_point.x())))

        # Annotation stage clicks are handled in MainWindow.on_point_clicked().
        if self.annotation_mode_active:
            self.point_clicked.emit(row_idx, col_idx)
            return

        # Spectrogram row-selection mode.
        if self.data_manager.cursor_mode == "s":
            self.point_clicked.emit(row_idx, col_idx)
            return

        # Ctrl-click near an existing saved apex deletes a saved annotation.
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.existing_labels_metadata
        ):
            idx = self._nearest_existing_label(
                mouse_point.x(),
                mouse_point.y(),
            )

            if idx is not None:
                tx_id = self.existing_labels_metadata[idx]["tx_id"]
                self.label_delete_requested.emit(tx_id)
            
    #####################################################################
    # Apex / Endpoint / Distance Annotations
    #####################################################################
    def _remove_apex_point(self) -> None:
        """Remove the current editable apex marker."""
        if self.apex_point_item is not None:
            try:
                self.plot_widget.removeItem(self.apex_point_item)
            except Exception:
                pass

        self.apex_point_item = None
        
    def _on_apex_position_changed(self, target_item) -> None:
        """
        Forward dragged apex coordinates to MainWindow.

        The position is kept within the currently loaded T-X data bounds.
        MainWindow decides when to snap it to an actual DAS channel.
        """
        if self.apex_point_item is None:
            return

        amp = self.data_manager.loaded_data.get("amp")
        t_vec = self.data_manager.loaded_data.get("t")
        x_vec = self.data_manager.loaded_data.get("x")

        if amp is None or t_vec is None or x_vec is None:
            return

        position = target_item.pos()

        time_s = float(position.x())
        distance_m = float(position.y())

        # Keep the marker inside the currently loaded plot area.
        time_s = float(np.clip(time_s, t_vec[0], t_vec[-1]))
        distance_m = float(np.clip(distance_m, x_vec[0], x_vec[-1]))

        # TargetItem can be dragged outside the image; visually clamp it back.
        if (
            not np.isclose(position.x(), time_s)
            or not np.isclose(position.y(), distance_m)
        ):
            target_item.setPos(time_s, distance_m)

        self.apex_dragged.emit(time_s, distance_m)
        
    def mark_apex_point(
        self,
        time_s: float,
        distance_m: float,
    ) -> None:
        """
        Draw or move the editable apex marker.

        Uses pyqtgraph.TargetItem because it supports dragging and emits
        position-change signals.
        """
        self._remove_apex_point()

        self.apex_point_item = pg.TargetItem(
            pos=(float(time_s), float(distance_m)),
            size=14,
            pen=pg.mkPen("red", width=2),
            brush=pg.mkBrush("red"),
            movable=True,
        )

        self.apex_point_item.sigPositionChanged.connect(
            self._on_apex_position_changed
        )

        self.plot_widget.addItem(self.apex_point_item)

    def set_apex_movable(self, movable: bool) -> None:
        """Enable or disable dragging of the editable TargetItem apex."""
        if self.apex_point_item is None:
            return

        movable = bool(movable)
        self.apex_point_item.movable = movable

        if movable:
            self.apex_point_item.setCursor(
                Qt.CursorShape.SizeAllCursor
            )
        else:
            self.apex_point_item.setCursor(
                Qt.CursorShape.ArrowCursor
            )

    def set_endpoint_movable(self, movable: bool) -> None:
        """Enable or disable dragging of the endpoint TargetItem."""
        if self.endpoint_point_item is not None:
            self.endpoint_point_item.movable = bool(movable)

    def _on_endpoint_position_changed(self, target_item) -> None:
        """
        Constrain endpoint dragging to the apex channel.

        The user may drag freely, but the marker is immediately forced back to
        the stored apex-distance coordinate. MainWindow receives the updated
        endpoint time and updates annotation state/hyperbola geometry.
        """
        if self._updating_endpoint_position:
            return

        if self._endpoint_distance_m is None:
            return

        t_vec = self.data_manager.loaded_data.get("t")

        if t_vec is None or len(t_vec) == 0:
            return

        position = target_item.pos()

        time_s = float(position.x())
        fixed_distance_m = float(self._endpoint_distance_m)

        # Keep endpoint time inside the displayed T-X window.
        time_s = float(np.clip(time_s, t_vec[0], t_vec[-1]))

        # TargetItem may have been dragged vertically; constrain it back to apex
        # channel while preventing recursive signal handling.
        if (
            not np.isclose(position.x(), time_s)
            or not np.isclose(position.y(), fixed_distance_m)
        ):
            self._updating_endpoint_position = True
            try:
                target_item.setPos(time_s, fixed_distance_m)
            finally:
                self._updating_endpoint_position = False

        self.endpoint_dragged.emit(
            time_s,
            fixed_distance_m,
        )

    #####################################################################
    # Clear overlays
    #####################################################################
    def clear_annotation_overlays(self) -> None:
        """Remove all unsaved annotation markers and hyperbola overlays."""
        self._remove_apex_point()
        self.clear_endpoint_point()
        self.clear_distance_boundaries()

        if self.distance_points_item is not None:
            try:
                self.plot_widget.removeItem(self.distance_points_item)
            except Exception:
                pass

        self.distance_points_item = None
        self.clear_hyperbola()

    def mark_endpoint_point(
        self,
        time_s: float,
        distance_m: float,
    ) -> None:
        """
        Draw or move an editable endpoint marker.

        The endpoint can move horizontally in time, but its distance remains
        constrained to the current apex channel distance.
        """
        self.clear_endpoint_point()

        self._endpoint_distance_m = float(distance_m)

        self.endpoint_point_item = pg.TargetItem(
            pos=(float(time_s), float(distance_m)),
            size=14,
            pen=pg.mkPen("cyan", width=2),
            brush=pg.mkBrush("cyan"),
            movable=True,
        )

        self.endpoint_point_item.sigPositionChanged.connect(
            self._on_endpoint_position_changed
        )

        self.plot_widget.addItem(self.endpoint_point_item)

    def clear_endpoint_point(self) -> None:
        """Remove temporary editable endpoint marker."""
        if self.endpoint_point_item is not None:
            try:
                self.plot_widget.removeItem(self.endpoint_point_item)
            except Exception:
                pass

        self.endpoint_point_item = None
        self._endpoint_distance_m = None

    def set_endpoint_position(
        self,
        time_s: float,
        distance_m: float,
    ) -> None:
        """
        Move the endpoint marker programmatically.

        Called by MainWindow after apex movement or endpoint dragging.
        """
        self._endpoint_distance_m = float(distance_m)

        if self.endpoint_point_item is None:
            self.mark_endpoint_point(time_s, distance_m)
            return

        self._updating_endpoint_position = True
        try:
            self.endpoint_point_item.setPos(
                float(time_s),
                float(distance_m),
            )
        finally:
            self._updating_endpoint_position = False
            
    def set_distance_annotation_points(
        self,
        points: list[tuple[float, float]],
    ) -> None:
        """
        Draw one or two selected points snapped to the hyperbola.

        Parameters
        ----------
        points:
            List of (time_s, distance_m) tuples.
        """
        if self.distance_points_item is not None:
            try:
                self.plot_widget.removeItem(self.distance_points_item)
            except Exception:
                pass

            self.distance_points_item = None

        if not points:
            return

        time_values = [point[0] for point in points]
        distance_values = [point[1] for point in points]

        self.distance_points_item = pg.ScatterPlotItem(
            x=time_values,
            y=distance_values,
            symbol="o",
            size=9,
            pen=pg.mkPen("black", width=1),
            brush=pg.mkBrush("yellow"),
        )

        self.plot_widget.addItem(self.distance_points_item)        
        
    def show_hyperbola(
        self,
        *,
        time_values: np.ndarray,
        distance_values: np.ndarray,
        curve_name: str = "apex",
        color: str = "white",
        style=Qt.PenStyle.DashLine,
        width: int = 2,
    ) -> None:
        """
        Draw or replace a named hyperbola overlay.

        curve_name examples:
            "apex"
            "endpoint"
            "confirmed"
        """
        self.clear_hyperbola(curve_name)

        line_item = pg.PlotDataItem(
            x=np.asarray(time_values, dtype=float),
            y=np.asarray(distance_values, dtype=float),
            pen=pg.mkPen(
                color=color,
                style=style,
                width=width,
            ),
        )

        self.plot_widget.addItem(line_item)
        self.hyperbola_items[curve_name] = line_item


    def clear_hyperbola(self, curve_name: str | None = None) -> None:
        """
        Remove one named hyperbola or all temporary hyperbola overlays.
        """
        if curve_name is not None:
            item = self.hyperbola_items.pop(curve_name, None)

            if item is not None:
                try:
                    self.plot_widget.removeItem(item)
                except Exception:
                    pass

            return

        for item in self.hyperbola_items.values():
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass

        self.hyperbola_items.clear()        
