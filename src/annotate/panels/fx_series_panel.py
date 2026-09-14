from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import QRectF, pyqtSignal
from PyQt6.QtWidgets import (
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from annotate.config import PLOTCOLOR_LUT, UserSettings


class FXSeriesPanel(QWidget):
    """
    Scrollable F-X thumbnail series.

    Responsibilities
    ----------------
    - Display F-X thumbnails for each temporal F-X slice.
    - Emit slice_selected when a thumbnail is clicked.
    - Highlight annotation-relevant F-X windows.
    - Indicate whether relevant windows contain F-X annotation boxes.

    Does not
    --------
    - Create/edit F-X ROIs.
    - Store ROI coordinates.
    """

    slice_selected = pyqtSignal(int)

    # Border styles used during F-X annotation.
    BORDER_NONE = ""
    BORDER_SELECTED_EMPTY = "border: 3px solid red;"
    BORDER_SELECTED_BOXED = "border: 3px solid limegreen;"
    BORDER_RELEVANT_EMPTY = "border: 2px solid orange;"
    BORDER_RELEVANT_BOXED = "border: 2px solid lightgreen;"

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.data_manager = data_manager

        defaults = UserSettings()

        self.vmin = defaults.fx_vmin
        self.vmax = defaults.fx_vmax
        self.use_db = False

        self.plot_widgets: list[pg.PlotWidget] = []
        self.plot_img_items: list[pg.ImageItem] = []

        self.highlight_idx: int | None = None

        # F-X slice indices relevant to the current annotation.
        self.annotation_slice_indices: set[int] = set()

        # F-X thumbnail indices that overlap the currently annotated call.
        # Empty unless F-X box annotation mode is active.
        self.annotation_slice_indices = set()

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self.scroll_area)

        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setContentsMargins(2, 2, 2, 2)
        self.container_layout.setSpacing(4)

        self.scroll_area.setWidget(self.container_widget)

        self.data_manager.dataset_loaded.connect(
            self.on_dataset_loaded
        )
        self.data_manager.settings_changed.connect(
            self.on_settings_changed
        )

    # ======================================================================
    # Data / display updates
    # ======================================================================

    def on_dataset_loaded(self) -> None:
        """Rebuild all F-X thumbnails after data are loaded."""
        self.update_settings()

        dataset = self.data_manager.fx_manager.get_dataset()
        self.set_plot_data(dataset)

    def on_settings_changed(self) -> None:
        """Rebuild thumbnails when F-X contrast/settings change."""
        self.update_settings()

        dataset = self.data_manager.fx_manager.get_dataset()
        self.set_plot_data(dataset)

    def update_settings(self) -> None:
        """Read F-X display settings from DataManager."""
        settings = self.data_manager.get_user_settings()

        # GUI sliders use integer range 0–100.
        self.vmin = float(settings.get("fx_vmin", 0)) / 100.0
        self.vmax = float(settings.get("fx_vmax", 40)) / 100.0

        self.use_db = bool(settings.get("fx_use_db", False))

    def set_plot_data(self, dataset: dict) -> None:
        """Build a thumbnail widget for every F-X temporal slice."""
        if not dataset or dataset.get("amp") is None:
            self._clear_thumbnails()
            return

        amp = dataset["amp"]
        freq = dataset["freq"]
        distance_m = dataset["x"]
        times = dataset["t"]

        if (
            amp is None
            or freq is None
            or distance_m is None
            or times is None
        ):
            self._clear_thumbnails()
            return

        self._clear_thumbnails()

        levels = (self.vmin, self.vmax)

        # Convert full array once if using dB display.
        display_amp = amp

        if self.use_db:
            display_amp = 20.0 * np.log10(
                np.maximum(amp, 1e-12)
            )

            levels = (
                20.0 * np.log10(max(self.vmin, 1e-12)),
                20.0 * np.log10(max(self.vmax, 1e-12)),
            )

        f0 = float(freq[0])
        f1 = float(freq[-1])

        x0 = float(distance_m[0])
        x1 = float(distance_m[-1])

        for slice_idx in range(display_amp.shape[0]):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("w")
            plot_widget.setMinimumHeight(180)

            plot_item = plot_widget.getPlotItem()
            plot_item.setLabel("bottom", "frequency", units="Hz")
            plot_item.setLabel("left", "cable distance", units="m")

            plot_item.setTitle(
                f"T = {float(times[slice_idx]):.2f} s"
            )

            img_item = pg.ImageItem(axisOrder="row-major")
            img_item.setLookupTable(PLOTCOLOR_LUT)

            image_data = display_amp[slice_idx, :, :]

            img_item.setImage(
                image_data,
                levels=levels,
            )

            img_item.setRect(
                QRectF(
                    f0,
                    x0,
                    f1 - f0,
                    x1 - x0,
                )
            )

            plot_widget.addItem(img_item)

            # Capture slice_idx as a default argument. Without this,
            # every lambda would emit the final loop value.
            plot_widget.scene().sigMouseClicked.connect(
                lambda _event, idx=slice_idx: self.slice_selected.emit(idx)
            )

            self.container_layout.addWidget(plot_widget)

            self.plot_widgets.append(plot_widget)
            self.plot_img_items.append(img_item)

        self._refresh_thumbnail_borders()

    def _clear_thumbnails(self) -> None:
        """Delete all existing thumbnail widgets."""
        while self.container_layout.count():
            layout_item = self.container_layout.takeAt(0)
            widget = layout_item.widget()

            if widget is not None:
                widget.deleteLater()

        self.plot_widgets = []
        self.plot_img_items = []
        self.highlight_idx = None

    # ======================================================================
    # Thumbnail selection / annotation highlighting
    # ======================================================================

    def highlight_slice(self, slice_idx: int) -> None:
        """Highlight the selected thumbnail and update all border colors."""
        if not (0 <= slice_idx < len(self.plot_widgets)):
            return

        self.highlight_idx = int(slice_idx)
        self._refresh_thumbnail_borders()

    def set_annotation_slice_indices(
        self,
        slice_indices: list[int] | set[int],
    ) -> None:
        """
        Mark F-X thumbnails relevant to the currently annotated call.

        Relevant-but-unselected thumbnails become orange or light green.
        """
        self.annotation_slice_indices = {
            int(index)
            for index in slice_indices
        }

        self._refresh_thumbnail_borders()

    def clear_annotation_slice_highlights(self) -> None:
        """Remove call-specific annotation highlighting from thumbnails."""
        self.annotation_slice_indices.clear()

        # Keep current selected thumbnail if it exists, but remove special
        # annotation colors.
        self._refresh_thumbnail_borders()

    def refresh_annotation_slice_styles(self) -> None:
        """
        Public method used by MainWindow after F-X ROI changes.

        The main F-X panel owns ROI editing. This panel only reads
        annotation_rois_per_slice to decide thumbnail border color.
        """
        self._refresh_thumbnail_borders()

    def _refresh_thumbnail_borders(self) -> None:
        """Apply selected/relevant/boxed border styles to all thumbnails."""
        roi_store = getattr(
            self.data_manager,
            "annotation_rois_per_slice",
            {},
        )

        for slice_idx, plot_widget in enumerate(self.plot_widgets):
            has_boxes = bool(roi_store.get(slice_idx, []))
            is_selected = slice_idx == self.highlight_idx
            is_relevant = slice_idx in self.annotation_slice_indices

            if is_selected and is_relevant:
                style = (
                    self.BORDER_SELECTED_BOXED
                    if has_boxes
                    else self.BORDER_SELECTED_EMPTY
                )

            elif is_relevant:
                style = (
                    self.BORDER_RELEVANT_BOXED
                    if has_boxes
                    else self.BORDER_RELEVANT_EMPTY
                )

            else:
                style = self.BORDER_NONE

            plot_widget.setStyleSheet(style)

    # ======================================================================
    # Cleanup
    # ======================================================================

    def clear_annotation_overlays(self) -> None:
        """
        Clear annotation highlighting.

        Actual F-X ROIs belong to FXPlotPanel and are removed there.
        """
        self.clear_annotation_slice_highlights()