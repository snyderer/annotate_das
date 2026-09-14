# annotate/panels/fx_plot_panel.py
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from annotate.config import PLOTCOLOR_LUT


class FXPlotPanel(QWidget):
    """
    Large F-X plot for one selected temporal F-X slice.

    Supports:
    - viewing F-X slices from FXSeriesPanel;
    - selecting a cable-distance row for spectrogram display;
    - F-X bounding-box annotation in ``fx_boxes`` mode.
    """

    point_clicked = pyqtSignal(int, int)
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

        self.distance_arrow = None  

        self.distance_min_line = None
        self.distance_max_line = None

        self._updating_distance_boundaries = False

        layout = QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("F-X plot")

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
            if event.double():
                self._add_default_roi(
                    frequency_hz=f_click,
                    distance_m=x_click,
                )
                return

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

    def mark_distance(self, distance_m: float) -> None:
        """Mark one selected distance row across the displayed F-X plot."""
        if self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass

        if self.freq_band is None:
            return

        self.distance_arrow = pg.ScatterPlotItem(
            x=list(self.freq_band),
            y=[distance_m, distance_m],
            symbol="d",
            size=12,
            brush="red",
            pen="red",
        )

        self.plot_widget.addItem(self.distance_arrow)

    def clear_annotation_overlays(self) -> None:
        """Remove visible F-X annotation graphics."""
        self._clear_displayed_rois()
        self.clear_distance_boundaries()

        if self.distance_arrow is not None:
            try:
                self.plot_widget.removeItem(self.distance_arrow)
            except Exception:
                pass

        self.distance_arrow = None