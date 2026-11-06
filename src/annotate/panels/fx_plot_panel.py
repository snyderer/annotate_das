# annotate/panels/fx_plot_panel.py
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from annotate.config import PLOTCOLOR_LUT

# editable FX ROI function
class FXAnnotationViewBox(pg.ViewBox):
    """
    ViewBox that emits F-X plot-coordinate drag events while F-X ROI annotation
    mode is active.

    Normal panning/zooming behavior remains unchanged outside fx_boxes mode.
    """

    box_drag_started = pyqtSignal(float, float)
    box_dragged = pyqtSignal(float, float)
    box_drag_finished = pyqtSignal(float, float)

    def __init__(self, panel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.panel = panel

    def mouseDragEvent(self, event, axis=None) -> None:
        """
        Convert left-button drags in F-X box mode into ROI drawing gestures.
        """
        if (
            self.panel.data_manager.cursor_mode == "fx_boxes"
            and event.button() == Qt.MouseButton.LeftButton
            and not (
                event.modifiers()
                & Qt.KeyboardModifier.ControlModifier
            )
        ):
            point = self.mapToView(event.pos())

            frequency_hz = float(point.x())
            distance_m = float(point.y())

            if event.isStart():
                self.box_drag_started.emit(
                    frequency_hz,
                    distance_m,
                )

            elif event.isFinish():
                self.box_drag_finished.emit(
                    frequency_hz,
                    distance_m,
                )

            else:
                self.box_dragged.emit(
                    frequency_hz,
                    distance_m,
                )

            event.accept()
            return

        super().mouseDragEvent(event, axis=axis)

class FXPlotPanel(QWidget):
    """
    Large F-X plot for one selected temporal F-X slice.

    Supports:
    - viewing F-X slices from FXSeriesPanel;
    - selecting a cable-distance row for spectrogram display;
    - F-X bounding-box annotation in ``fx_boxes`` mode.
    """

    point_clicked = pyqtSignal(int, int)
    cursor_moved = pyqtSignal(float, float)
    roi_changed = pyqtSignal(int)
    
    distance_boundary_dragged = pyqtSignal(str, float)

    def __init__(self, data_manager, vmin: float = 0.0, vmax: float = 0.4):
        super().__init__()

        self.data_manager = data_manager

        self.vmin = vmin
        self.vmax = vmax
        self.use_db = False

        self.current_slice_idx: int | None = None
        self.freq_band: tuple[float, float] | None = None

        # RectROI objects currently displayed for current_slice_idx.
        self.fx_slice_rois: list[pg.RectROI] = []
        
        # Temporary state while the user is click-dragging a new F-X ROI.
        self._draw_roi: pg.RectROI | None = None
        self._draw_start_frequency_hz: float | None = None
        self._draw_start_distance_m: float | None = None

        self.distance_arrow = None  
        
        self.frequency_annotation_box = None # Frequency box (non-editable) for annotation display.

        self.distance_min_line = None
        self.distance_max_line = None

        self._updating_distance_boundaries = False

        layout = QVBoxLayout(self)

        self.view_box = FXAnnotationViewBox(self)

        self.plot_widget = pg.PlotWidget(
            viewBox=self.view_box,
        )
        
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("F-X plot")

        self.view_box.box_drag_started.connect(
            self._begin_draw_roi
        )

        self.view_box.box_dragged.connect(
            self._update_draw_roi
        )

        self.view_box.box_drag_finished.connect(
            self._finish_draw_roi
        )

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel("bottom", "frequency", units="Hz")
        plot_item.setLabel("left", "cable distance", units="m")

        layout.addWidget(self.plot_widget)

        self.img_item = pg.ImageItem(axisOrder="row-major")
        self.img_item.setLookupTable(PLOTCOLOR_LUT)
        self.plot_widget.addItem(self.img_item)

        self.plot_widget.scene().sigMouseClicked.connect(
            self._on_mouse_click
        )

        self.data_manager.dataset_loaded.connect(
            self.on_dataset_loaded
        )
        self.data_manager.settings_changed.connect(
            self.on_settings_changed
        )
        
        self.plot_widget.scene().sigMouseMoved.connect(
            self._on_scene_mouse_moved
        )
        
        # Cursor setup
        crosshair_pen = pg.mkPen(
            color=(90, 90, 90, 140),
            width=1,
        )

        self.cursor_vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=crosshair_pen,
        )

        self.cursor_hline = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=crosshair_pen,
        )

        self.cursor_vline.setVisible(False)
        self.cursor_hline.setVisible(False)

        self.plot_widget.addItem(
            self.cursor_vline,
            ignoreBounds=True,
        )
        self.plot_widget.addItem(
            self.cursor_hline,
            ignoreBounds=True,
        )
        
    # =====================================================================
    # Dataset / display updates
    # =====================================================================

    def on_dataset_loaded(self) -> None:
        """Display the first available F-X slice after a new data load."""
        self.update_settings()

        dataset = self.data_manager.fx_manager.get_dataset()
        amp = dataset.get("amp")

        if amp is None or len(amp) == 0:
            return

        self.show_slice(0)

    def reset_view_to_data_extent(self) -> None:
        """Show the complete frequency and cable-distance extent of F-X data."""
        dataset = self.data_manager.fx_manager.get_dataset()

        freq = dataset.get("freq")
        distance_m = dataset.get("x")

        if (
            freq is None
            or distance_m is None
            or len(freq) == 0
            or len(distance_m) == 0
        ):
            return

        f_min = float(np.min(freq))
        f_max = float(np.max(freq))

        x_min = float(np.min(distance_m))
        x_max = float(np.max(distance_m))

        self.plot_widget.getPlotItem().vb.setRange(
            xRange=(f_min, f_max),
            yRange=(x_min, x_max),
            padding=0.02,
        )

    def on_settings_changed(self) -> None:
        """Update contrast settings and redraw current F-X slice."""
        self.update_settings()

        if self.current_slice_idx is not None:
            self.show_slice(self.current_slice_idx)

    def update_settings(self) -> None:
        """Read F-X display settings from DataManager."""
        settings = self.data_manager.get_user_settings()

        # Display sliders are 0-100; plotted amplitude levels are 0.0-1.0.
        self.vmin = float(settings.get("fx_vmin", 0)) / 100.0
        self.vmax = float(settings.get("fx_vmax", 40)) / 100.0

        self.use_db = bool(
            settings.get("fx_use_db", False)
        )

    def show_slice_from_series(self, slice_idx: int) -> None:
        """Public entry point used by MainWindow/FXSeriesPanel."""
        self.show_slice(slice_idx)

    def show_slice(self, slice_idx: int) -> None:
        """Draw one F-X slice and restore any stored annotation ROIs."""
        dataset = self.data_manager.fx_manager.get_dataset()

        amp = dataset.get("amp")
        freq = dataset.get("freq")
        distance_m = dataset.get("x")
        times = dataset.get("t")

        if (
            amp is None
            or freq is None
            or distance_m is None
            or times is None
        ):
            return

        if not (0 <= slice_idx < amp.shape[0]):
            return

        self.clear_frequency_annotation_box()
        self.current_slice_idx = int(slice_idx)
        self.data_manager.current_fx_index = self.current_slice_idx

        image_data = np.asarray(
            amp[self.current_slice_idx, :, :],
            dtype=float,
        )

        levels = (self.vmin, self.vmax)

        if self.use_db:
            image_data = 20.0 * np.log10(
                np.maximum(image_data, 1e-12)
            )

            levels = (
                20.0 * np.log10(max(self.vmin, 1e-12)),
                20.0 * np.log10(max(self.vmax, 1e-12)),
            )

        self.img_item.setImage(
            image_data,
            levels=levels,
        )

        f0 = float(freq[0])
        f1 = float(freq[-1])

        x0 = float(distance_m[0])
        x1 = float(distance_m[-1])

        self.img_item.setRect(
            QRectF(
                f0,
                x0,
                f1 - f0,
                x1 - x0,
            )
        )

        self.freq_band = (f0, f1)

        self.plot_widget.setTitle(
            f"F-X plot @ T={float(times[self.current_slice_idx]):.2f} s "
            "(|nanostrain|)"
        )

        self._restore_rois_for_current_slice()

    def set_distance_boundaries(
        self,
        distance_min_m: float | None,
        distance_max_m: float | None,
        *,
        movable: bool = False,
    ) -> None:
        """Draw/update draggable min/max cable-distance lines."""
        self._updating_distance_boundaries = True

        try:
            self.clear_distance_boundaries()

            if distance_min_m is None or distance_max_m is None:
                return

            self.distance_min_line = pg.InfiniteLine(
                pos=float(distance_min_m),
                angle=0,
                movable=movable,
                pen=pg.mkPen("yellow", width=2),
                hoverPen=pg.mkPen("orange", width=3),
                label="Min distance",
                labelOpts={
                    "position": 0.95,
                    "color": "yellow",
                    "fill": (0, 0, 0, 120),
                },
            )

            self.distance_max_line = pg.InfiniteLine(
                pos=float(distance_max_m),
                angle=0,
                movable=movable,
                pen=pg.mkPen("yellow", width=2),
                hoverPen=pg.mkPen("orange", width=3),
                label="Max distance",
                labelOpts={
                    "position": 0.95,
                    "color": "yellow",
                    "fill": (0, 0, 0, 120),
                },
            )

            self.plot_widget.addItem(self.distance_min_line)
            self.plot_widget.addItem(self.distance_max_line)

            # Emit only after the user releases the line. Emitting continuously
            # would cause MainWindow to rebuild the lines during dragging.
            self.distance_min_line.sigPositionChangeFinished.connect(
                lambda line: self._on_distance_boundary_changed(
                    "min",
                    line,
                )
            )

            self.distance_max_line.sigPositionChangeFinished.connect(
                lambda line: self._on_distance_boundary_changed(
                    "max",
                    line,
                )
            )

        finally:
            self._updating_distance_boundaries = False
        
    def clear_distance_boundaries(self) -> None:
        """Remove F-X distance-range lines."""
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

    def _on_distance_boundary_changed(
        self,
        boundary_name: str,
        line: pg.InfiniteLine,
    ) -> None:
        """Inform MainWindow that one F-X distance boundary was dragged."""
        if self._updating_distance_boundaries:
            return

        self.distance_boundary_dragged.emit(
            boundary_name,
            float(line.value()),
        )
        
    def set_distance_boundaries_movable(
        self,
        movable: bool,
    ) -> None:
        """Enable/disable drag interaction for visible distance lines."""
        for line in (
            self.distance_min_line,
            self.distance_max_line,
        ):
            if line is not None:
                line.setMovable(bool(movable))

    # =====================================================================
    # ROI storage / restoration
    # =====================================================================

    def _clear_displayed_rois(self) -> None:
        """Remove only ROI graphics currently shown in the large F-X plot."""
        for roi in self.fx_slice_rois:
            try:
                self.plot_widget.removeItem(roi)
            except Exception:
                pass

        self.fx_slice_rois = []

    def _restore_rois_for_current_slice(self) -> None:
        """Draw ROIs stored for the currently shown F-X slice."""
        self._clear_displayed_rois()

        if self.current_slice_idx is None:
            return

        roi_store = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        )

        coordinates = roi_store.get(
            self.current_slice_idx,
            [],
        )

        for f_min, x_min, f_max, x_max in coordinates:
            roi = self._create_roi(
                f_min=float(f_min),
                x_min=float(x_min),
                f_max=float(f_max),
                x_max=float(x_max),
            )

            self.fx_slice_rois.append(roi)

    def _create_roi(
        self,
        *,
        f_min: float,
        x_min: float,
        f_max: float,
        x_max: float,
    ) -> pg.RectROI:
        """Create one editable F-X ROI and connect it to persistent storage."""
        width = max(float(f_max - f_min), 1e-6)
        height = max(float(x_max - x_min), 1e-6)

        roi = pg.RectROI(
            [f_min, x_min],
            [width, height],
            pen=pg.mkPen("r", width=2),
            movable=True,
            resizable=True,
        )

        # Update stored coordinates continuously while moving/resizing.
        roi.sigRegionChanged.connect(
            lambda *_args, current_roi=roi: self._update_roi_coordinates(
                current_roi
            )
        )

        # Signal MainWindow when a change is complete.
        roi.sigRegionChangeFinished.connect(
            lambda *_args: self._emit_roi_changed()
        )

        self.plot_widget.getPlotItem().addItem(roi)

        return roi

    def _begin_draw_roi(
        self,
        frequency_hz: float,
        distance_m: float,
    ) -> None:
        """
        Begin drawing a new ROI from the current F-X mouse location.

        A nearly zero-sized ROI is created immediately and resized while the
        user drags.
        """
        if self._roi_at_position(
            frequency_hz,
            distance_m,
        ) is not None:
            return
        
        if self.current_slice_idx is None:
            return

        # Do not begin a second draw while one is still active.
        if self._draw_roi is not None:
            return

        self._draw_start_frequency_hz = float(frequency_hz)
        self._draw_start_distance_m = float(distance_m)

        epsilon = 1e-6

        roi = self._create_roi(
            f_min=float(frequency_hz),
            x_min=float(distance_m),
            f_max=float(frequency_hz) + epsilon,
            x_max=float(distance_m) + epsilon,
        )

        self.fx_slice_rois.append(roi)

        coordinates = self.data_manager.annotation_rois_per_slice.setdefault(
            self.current_slice_idx,
            [],
        )

        coordinates.append((
            float(frequency_hz),
            float(distance_m),
            float(frequency_hz) + epsilon,
            float(distance_m) + epsilon,
        ))

        self._draw_roi = roi

    def _update_draw_roi(
        self,
        frequency_hz: float,
        distance_m: float,
    ) -> None:
        """Resize the currently drawn ROI as the user drags."""
        if (
            self._draw_roi is None
            or self._draw_start_frequency_hz is None
            or self._draw_start_distance_m is None
        ):
            return

        start_f = self._draw_start_frequency_hz
        start_x = self._draw_start_distance_m

        current_f = float(frequency_hz)
        current_x = float(distance_m)

        f_min = min(start_f, current_f)
        f_max = max(start_f, current_f)

        x_min = min(start_x, current_x)
        x_max = max(start_x, current_x)

        minimum_width = 1e-6
        minimum_height = 1e-6

        self._draw_roi.setPos(
            [f_min, x_min],
            update=True,
        )

        self._draw_roi.setSize(
            [
                max(f_max - f_min, minimum_width),
                max(x_max - x_min, minimum_height),
            ],
            update=True,
        )

        self._update_roi_coordinates(
            self._draw_roi
        )

    def _finish_draw_roi(
        self,
        frequency_hz: float,
        distance_m: float,
    ) -> None:
        """
        Finalize one click-drag ROI.

        Very small drags are discarded so ordinary clicks do not create tiny,
        unusable ROI rectangles.
        """
        if self._draw_roi is None:
            return

        roi = self._draw_roi

        # Clear drawing state first; deletion/update logic may run below.
        self._draw_roi = None
        self._draw_start_frequency_hz = None
        self._draw_start_distance_m = None

        size = roi.size()

        width_hz = float(size.x())
        height_m = float(size.y())

        minimum_frequency_width_hz = 0.5
        minimum_distance_height_m = 5.0

        if (
            width_hz < minimum_frequency_width_hz
            or height_m < minimum_distance_height_m
        ):
            try:
                roi_index = self.fx_slice_rois.index(roi)
            except ValueError:
                return

            self.fx_slice_rois.pop(roi_index)

            try:
                self.plot_widget.removeItem(roi)
            except Exception:
                pass

            if self.current_slice_idx is not None:
                coordinates = (
                    self.data_manager.annotation_rois_per_slice.get(
                        self.current_slice_idx,
                        [],
                    )
                )

                if roi_index < len(coordinates):
                    coordinates.pop(roi_index)

            return

        self._update_roi_coordinates(roi)
        self._emit_roi_changed()

    def promote_frequency_annotation_box_to_editable(self) -> None:
        """
        Convert the displayed aggregate cyan frequency guide into an editable
        red F-X ROI for the currently selected slice.

        This is used when the user enters F-X box annotation mode while
        editing or pasting an existing annotation that only has aggregate
        f_min/f_max bounds rather than persisted individual ROI rectangles.
        """
        if (
            self.frequency_annotation_box is None
            or self.current_slice_idx is None
        ):
            return

        pos = self.frequency_annotation_box.pos()
        size = self.frequency_annotation_box.size()

        f_min = float(pos.x())
        x_min = float(pos.y())

        f_max = f_min + float(size.x())
        x_max = x_min + float(size.y())

        # Remove the non-editable cyan guide.
        self.clear_frequency_annotation_box()

        # Create a normal editable stored ROI.
        roi = self._create_roi(
            f_min=f_min,
            x_min=x_min,
            f_max=f_max,
            x_max=x_max,
        )

        self.fx_slice_rois.append(roi)

        coordinates = self.data_manager.annotation_rois_per_slice.setdefault(
            self.current_slice_idx,
            [],
        )

        coordinates.append(
            (f_min, x_min, f_max, x_max)
        )

        self._emit_roi_changed()

    def _update_roi_coordinates(self, roi: pg.RectROI) -> None:
        """Write the current ROI coordinates into DataManager storage."""
        if self.current_slice_idx is None:
            return

        try:
            roi_index = self.fx_slice_rois.index(roi)
        except ValueError:
            return

        pos = roi.pos()
        size = roi.size()

        f_min = float(pos.x())
        x_min = float(pos.y())
        f_max = f_min + float(size[0])
        x_max = x_min + float(size[1])

        roi_store = self.data_manager.annotation_rois_per_slice
        coordinates = roi_store.setdefault(
            self.current_slice_idx,
            [],
        )

        # This can occur when a graphics ROI was created but storage has
        # not yet been initialized.
        while len(coordinates) <= roi_index:
            coordinates.append(
                (f_min, x_min, f_max, x_max)
            )

        coordinates[roi_index] = (
            f_min,
            x_min,
            f_max,
            x_max,
        )

    def _emit_roi_changed(self) -> None:
        """Notify MainWindow that current F-X slice ROI state changed."""
        if self.current_slice_idx is not None:
            self.roi_changed.emit(self.current_slice_idx)

    # =====================================================================
    # Mouse interactions
    # =====================================================================
    def set_cursor_guides(
        self,
        *,
        frequency_hz: float | None = None,
        distance_m: float | None = None,
    ) -> None:
        """Show/update crosshair guides in F-X coordinates."""
        if frequency_hz is None:
            self.cursor_vline.setVisible(False)
        else:
            self.cursor_vline.setPos(float(frequency_hz))
            self.cursor_vline.setVisible(True)

        if distance_m is None:
            self.cursor_hline.setVisible(False)
        else:
            self.cursor_hline.setPos(float(distance_m))
            self.cursor_hline.setVisible(True)

    def clear_cursor_guides(self) -> None:
        self.cursor_vline.setVisible(False)
        self.cursor_hline.setVisible(False)
        
    def _on_scene_mouse_moved(self, scene_pos) -> None:
        """Emit F-X cursor coordinates while pointer is over the plot."""
        plot_item = self.plot_widget.getPlotItem()
        view_box = plot_item.vb

        if not view_box.sceneBoundingRect().contains(scene_pos):
            return

        point = view_box.mapSceneToView(scene_pos)

        self.cursor_moved.emit(
            float(point.x()),
            float(point.y()),
        )
        
    def _roi_at_position(
        self,
        frequency_hz: float,
        distance_m: float,
    ) -> int | None:
        """Return index of ROI containing supplied F-X position."""
        for index, roi in enumerate(self.fx_slice_rois):
            pos = roi.pos()
            size = roi.size()

            f_min = float(pos.x())
            x_min = float(pos.y())
            f_max = f_min + float(size[0])
            x_max = x_min + float(size[1])

            if (
                f_min <= frequency_hz <= f_max
                and x_min <= distance_m <= x_max
            ):
                return index

        return None

    def _on_mouse_click(self, event) -> None:
        """
        Handle F-X mouse interactions.

        Normal mode:
            click selects a distance row for the spectrogram.

        F-X box mode:
            double-click -> add ROI
            Ctrl-click ROI -> delete ROI
        """
        if self.current_slice_idx is None:
            return

        point = (
            self.plot_widget.getPlotItem()
            .vb
            .mapSceneToView(event.scenePos())
        )

        f_click = float(point.x())
        x_click = float(point.y())

        # -----------------------------------------------------------------
        # F-X bounding box mode
        # -----------------------------------------------------------------
        if self.data_manager.cursor_mode == "fx_boxes":
            modifiers = event.modifiers()

            # Ctrl-click deletes ROI under click.
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                self._delete_roi_at_position(
                    frequency_hz=f_click,
                    distance_m=x_click,
                )
                return

            # Double-click creates a new ROI.
            #
            # pyqtgraph MouseClickEvent has double() for double-click state.
            # if event.double():
            #     self._add_default_roi(
            #         frequency_hz=f_click,
            #         distance_m=x_click,
            #     )
            #     return

            return

        # -----------------------------------------------------------------
        # Spectrogram row selection
        # -----------------------------------------------------------------
        if self.data_manager.cursor_mode == "s":
            dataset = self.data_manager.fx_manager.get_dataset()

            freq = dataset.get("freq")
            distance = dataset.get("x")

            if freq is None or distance is None:
                return

            row_idx = int(
                np.argmin(np.abs(distance - x_click))
            )
            col_idx = int(
                np.argmin(np.abs(freq - f_click))
            )

            self.point_clicked.emit(row_idx, col_idx)

    def _add_default_roi(
        self,
        *,
        frequency_hz: float,
        distance_m: float,
    ) -> None:
        """Create a default F-X ROI centered at a double-click."""
        if self.current_slice_idx is None:
            return

        # Reasonable initial annotation ROI size.
        frequency_width_hz = 10.0

        # Use selected channel spacing to make a spatially meaningful box.
        dx_m = 20.0

        loaded_segment = getattr(
            self.data_manager,
            "loaded_segment",
            None,
        )

        if loaded_segment is not None:
            dx_m = max(
                float(loaded_segment.dx_m) * 10.0,
                20.0,
            )

        f_min = frequency_hz - frequency_width_hz / 2.0
        f_max = frequency_hz + frequency_width_hz / 2.0

        x_min = distance_m - dx_m / 2.0
        x_max = distance_m + dx_m / 2.0

        roi = self._create_roi(
            f_min=f_min,
            x_min=x_min,
            f_max=f_max,
            x_max=x_max,
        )

        self.fx_slice_rois.append(roi)

        coordinates = self.data_manager.annotation_rois_per_slice.setdefault(
            self.current_slice_idx,
            [],
        )

        coordinates.append(
            (f_min, x_min, f_max, x_max)
        )

        self._emit_roi_changed()

    def _delete_roi_at_position(
        self,
        *,
        frequency_hz: float,
        distance_m: float,
    ) -> None:
        """Delete the current-slice ROI under a Ctrl-click, if any."""
        if self.current_slice_idx is None:
            return

        roi_index = self._roi_at_position(
            frequency_hz,
            distance_m,
        )

        if roi_index is None:
            return

        roi = self.fx_slice_rois.pop(roi_index)

        try:
            self.plot_widget.removeItem(roi)
        except Exception:
            pass

        coordinates = self.data_manager.annotation_rois_per_slice.get(
            self.current_slice_idx,
            [],
        )

        if roi_index < len(coordinates):
            coordinates.pop(roi_index)

        self._emit_roi_changed()

    # =====================================================================
    # Plot overlays / cleanup
    # =====================================================================

    def mark_distance(
        self,
        distance_m: float,
    ) -> None:
        """
        Show the currently selected spectrogram cable-distance row.

        InfiniteLine spans the visible F-X view, so it remains visible while
        zooming or panning in frequency.
        """
        if self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(
                    self.distance_arrow
                )
            except Exception:
                pass

        self.distance_arrow = pg.InfiniteLine(
            pos=float(distance_m),
            angle=0,
            movable=False,
            pen=pg.mkPen(
                color="red",
                width=2,
                style=Qt.PenStyle.DashLine,
            ),
            label="Spectrogram row",
            labelOpts={
                "position": 0.98,
                "color": "red",
                "fill": (255, 255, 255, 150),
            },
        )

        self.plot_widget.addItem(
            self.distance_arrow,
            ignoreBounds=True,
        )
        
    def clear_frequency_annotation_box(self) -> None:
        """Remove the non-editable aggregate frequency annotation guide box."""
        if self.frequency_annotation_box is not None:
            try:
                self.plot_widget.removeItem(
                    self.frequency_annotation_box
                )
            except Exception:
                pass

        self.frequency_annotation_box = None

    def set_frequency_annotation_box(
        self,
        *,
        f_min_hz: float,
        f_max_hz: float,
        x_min_m: float,
        x_max_m: float,
    ) -> None:
        """
        Draw a non-editable box showing aggregate frequency/distance bounds.

        This is intentionally separate from editable F-X ROI boxes.
        """
        self.clear_frequency_annotation_box()

        f_min_hz = float(f_min_hz)
        f_max_hz = float(f_max_hz)
        x_min_m = float(x_min_m)
        x_max_m = float(x_max_m)

        if f_max_hz <= f_min_hz or x_max_m <= x_min_m:
            return

        self.frequency_annotation_box = pg.RectROI(
            [f_min_hz, x_min_m],
            [
                f_max_hz - f_min_hz,
                x_max_m - x_min_m,
            ],
            pen=pg.mkPen(
                color="cyan",
                width=3,
                style=Qt.PenStyle.DashLine,
            ),
            movable=False,
            resizable=False,
        )

        # Do not let this guide box intercept mouse events intended for the
        # plot or editable ROI logic.
        self.frequency_annotation_box.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton
        )

        self.plot_widget.addItem(
            self.frequency_annotation_box
        )

    def clear_annotation_overlays(self) -> None:
        """Remove visible F-X annotation graphics."""
        self._clear_displayed_rois()
        self.clear_distance_boundaries()
        self.clear_frequency_annotation_box()

        if self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass

        self.distance_arrow = None