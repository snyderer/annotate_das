from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from annotate.data.dataset_service import DatasetService
from annotate.data.processing import apply_processing
from annotate.data.window_loading import (
    PreparedWindow,
    WindowLoadRequest,
)

from time import perf_counter # for testing timer


class WindowLoaderWorker(QObject):
    """
    Background worker that owns its own DatasetService.

    All DAS source reads, cache access, filtering, and F-X computation occur
    in this worker thread. It never touches Qt widgets.
    """

    window_ready = pyqtSignal(object)   # PreparedWindow
    window_failed = pyqtSignal(int, str)
    loading_started = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()

        self.dataset_service = DatasetService()

        self._opened_file: Path | None = None
        self._opened_interrogator: str | None = None

    @pyqtSlot(object)
    def prepare_window(self, request: WindowLoadRequest) -> None:
        """
        Load and process one requested window in the background.

        Timing output is intentionally detailed while performance work is in
        progress. It can be converted to logging or removed later.
        """
        self.loading_started.emit(request.request_id)

        total_start = perf_counter()

        try:
            # --------------------------------------------------------------
            # Dataset/index setup.
            # --------------------------------------------------------------
            ensure_start = perf_counter()

            self._ensure_dataset_open(request)

            ensure_elapsed = perf_counter() - ensure_start

            # --------------------------------------------------------------
            # Raw source-file loading, cropping, stitching, concatenation.
            # --------------------------------------------------------------
            raw_start = perf_counter()

            raw_segment = self.dataset_service.load_window(
                start_time=request.start_time,
                duration_s=request.duration_s,
                selected_channels=request.selected_channels,
                correct_file_boundaries=request.correct_file_boundaries,
                boundary_edge_duration_s=request.boundary_edge_duration_s,
            )

            raw_elapsed = perf_counter() - raw_start

            # --------------------------------------------------------------
            # Bandpass / F-K / downsampling / processing.
            # --------------------------------------------------------------
            processing_start = perf_counter()

            amp, processed_fs_hz = apply_processing(
                amp=raw_segment.data,
                fs=raw_segment.fs_hz,
                dx_m=raw_segment.dx_m,
                settings=request.processing_settings,
            )

            processing_elapsed = perf_counter() - processing_start

            # --------------------------------------------------------------
            # Display array/time-vector setup.
            # --------------------------------------------------------------
            display_start = perf_counter()

            amp_display = amp * 1e9

            time_s = (
                np.arange(
                    amp_display.shape[1],
                    dtype=float,
                )
                / processed_fs_hz
            )

            display_elapsed = perf_counter() - display_start

            # --------------------------------------------------------------
            # F-X computation.
            # --------------------------------------------------------------
            fx_start = perf_counter()

            fx_amp, fx_freq, fx_distance_m, fx_time_s = (
                self._compute_fx(
                    amp=amp_display,
                    distance_m=raw_segment.distance_m,
                    fs_hz=processed_fs_hz,
                    settings=request.processing_settings,
                )
            )

            fx_elapsed = perf_counter() - fx_start

            # --------------------------------------------------------------
            # Package result.
            # --------------------------------------------------------------
            result_start = perf_counter()

            result = PreparedWindow(
                request_id=request.request_id,
                kind=request.kind,
                requested_start_time=request.start_time,

                raw_segment=raw_segment,

                amp_display=amp_display,
                time_s=time_s,
                distance_m=raw_segment.distance_m,
                processed_fs_hz=processed_fs_hz,

                fx_amp=fx_amp,
                fx_freq=fx_freq,
                fx_distance_m=fx_distance_m,
                fx_time_s=fx_time_s,
            )

            result_elapsed = perf_counter() - result_start
            total_elapsed = perf_counter() - total_start

            print(
                "\n"
                "Async window timing:\n"
                f"  request:       {request.start_time.isoformat()}\n"
                f"  dataset setup: {ensure_elapsed:.3f}s\n"
                f"  raw window:    {raw_elapsed:.3f}s\n"
                f"  processing:    {processing_elapsed:.3f}s\n"
                f"  display setup: {display_elapsed:.3f}s\n"
                f"  F-X:           {fx_elapsed:.3f}s\n"
                f"  result pack:   {result_elapsed:.3f}s\n"
                f"  total:         {total_elapsed:.3f}s\n"
            )

            self.window_ready.emit(result)

        except Exception:
            self.window_failed.emit(
                request.request_id,
                traceback.format_exc(),
            )

    def _ensure_dataset_open(
        self,
        request: WindowLoadRequest,
    ) -> None:
        """
        Open/index the worker-owned service only when dataset/interrogator
        changes. Its source-file RAM cache then persists across requests.
        """
        selected_file = request.selected_file.resolve()

        if (
            self._opened_file == selected_file
            and self._opened_interrogator == request.interrogator
            and self.dataset_service.is_open
        ):
            return

        self.dataset_service.clear()

        self.dataset_service.open_dataset(
            selected_file=selected_file,
            requested_interrogator=request.interrogator,
        )

        self._opened_file = selected_file
        self._opened_interrogator = request.interrogator

    @staticmethod
    def _compute_fx(
        *,
        amp: np.ndarray,
        distance_m: np.ndarray,
        fs_hz: float,
        settings: dict,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """
        Worker-only equivalent of FXHandle.update_data().

        This deliberately does not access DataManager.
        """
        fx_win_s = float(settings.get("fx_win_s", 2.0))
        fx_win_samples = int(round(fx_win_s * fs_hz))

        if fx_win_samples < 2 or amp.shape[1] < fx_win_samples:
            return None, None, distance_m, np.array([])

        requested_nfft = int(
            settings.get("fx_nfft", fx_win_samples)
        )
        fx_nfft = max(fx_win_samples, requested_nfft)

        freqs = np.fft.rfftfreq(
            fx_nfft,
            d=1.0 / fs_hz,
        )

        slices: list[np.ndarray] = []
        slice_times: list[float] = []

        for start in range(
            0,
            amp.shape[1] - fx_win_samples + 1,
            fx_win_samples,
        ):
            segment = amp[:, start:start + fx_win_samples]

            spectrum = np.fft.rfft(
                segment,
                n=fx_nfft,
                axis=1,
            )

            spectrum = (
                2.0 * np.abs(spectrum) / fx_win_samples
            )

            spectrum[:, 0] *= 0.5

            if fx_nfft % 2 == 0:
                spectrum[:, -1] *= 0.5

            slices.append(spectrum)
            slice_times.append(start / fs_hz)

        return (
            np.stack(slices, axis=0),
            freqs,
            distance_m,
            np.asarray(slice_times, dtype=float),
        )