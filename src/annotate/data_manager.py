from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.signal as sp
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from annotate.config import UserSettings

from annotate.data.dataset_service import DatasetService, FileRecord
from annotate.data.processing import apply_processing
from annotate.data.window_loading import (
    PreparedWindow,
    WindowLoadRequest,
)
from annotate.data.window_loader_worker import (
    WindowLoaderWorker,
)

from time import perf_counter # For debugging performance

class DataManager(QObject):

    dataset_loaded = pyqtSignal()
    settings_changed = pyqtSignal()
    file_loaded = pyqtSignal(str, str)

    dataset_opened = pyqtSignal(object)   # FileRecord
    window_changed = pyqtSignal(object)   # datetime
    
    window_loading_started = pyqtSignal(object)  # datetime
    window_loading_failed = pyqtSignal(str)
    
    load_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.dataset_service = DatasetService()
        
        self._window_request_id = 0
        self._active_window_request_id: int | None = None

        self._loader_thread = QThread(self)
        self._loader_worker = WindowLoaderWorker()
 
        self._loader_thread.finished.connect(
            self._loader_worker.deleteLater
        )
 
        self._loader_worker.moveToThread(
            self._loader_thread
        )

        self._loader_thread.start()

        self._loader_worker.window_ready.connect(
            self._on_worker_window_ready
        )
        self._loader_worker.window_failed.connect(
            self._on_worker_window_failed
        )

        self.load_requested.connect(
            self._loader_worker.prepare_window
        )

        self.user_settings = asdict(UserSettings())
        
        # One fully processed speculative forward window.
        self._prefetched_window: PreparedWindow | None = None
        self._prefetch_key: tuple | None = None
        self._prefetch_request_id: int | None = None
        self._prefetch_pending = False
        
        # Incremented whenever a prefetch timer/result should become obsolete.
        self._prefetch_generation = 0

        # Keep the timer object so it can be stopped/replaced instead of leaving
        # several old QTimer.singleShot callbacks queued.
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.timeout.connect(
            self._request_forward_prefetch
        )

        self._prefetch_delay_ms = 3_000

        # Delay before starting a speculative prefetch. This gives a user who
        # immediately presses Forward a chance to request a normal visible load
        # before the worker begins low-priority work.
        self._prefetch_delay_ms = 3_000

        self.current_start_time: datetime | None = None
        self.selected_channels: tuple[int, int, int] | None = None
        self.dataset_record: FileRecord | None = None

        self.filepath: str | None = None
        self.directory: str | None = None

        self.loaded_segment = None
        self.loaded_fs_hz: float | None = None

        self.raw_window_segment = None
        
        self.dataset_key: str | None = None
        self.dataset_display_name: str | None = None

        self.loaded_data = {
            "amp": None,
            "t": None,
            "x": None,
            "time_stamps": None,
        }

        self.cursor_mode = ""

        # annotation-related attributes
        self.label_saver = None
        self.annotation_fx_box = None
        self.annotation_fx_boxes_per_plot = None
        self.annotation_rois_per_slice = {}
        
        self.fx_manager = FXHandle(self)
        self.spectrogram_manager = SpectrogramHandle(self)
        
        self.raw_window_segment = None
    
    def _window_cache_key(
        self,
        *,
        start_time: datetime,
    ) -> tuple:
        """
        Return a stable identifier for one fully prepared display window.

        Includes all current settings because any processing/cable/F-X setting
        may change the prepared numeric result.
        """
        return (
            Path(self.filepath).resolve()
            if self.filepath is not None
            else None,

            start_time.astimezone(timezone.utc),

            float(
                self.user_settings.get("duration_s", 40.0)
            ),

            tuple(self.selected_channels or ()),

            # Easy and conservative during development: invalidate prefetch
            # whenever any stored setting changes.
            repr(
                sorted(
                    self.user_settings.items(),
                    key=lambda item: item[0],
                )
            ),
        )
   
    def _clear_prefetch(self) -> None:
        """Discard a completed/pending speculative forward result."""
        self._prefetch_generation += 1

        if self._prefetch_timer.isActive():
            self._prefetch_timer.stop()

        self._prefetched_window = None
        self._prefetch_key = None
        self._prefetch_request_id = None
        self._prefetch_pending = False

    def _install_prepared_window(
        self,
        result: PreparedWindow,
    ) -> None:
        """
        Install a worker-produced window into GUI-visible DataManager state.

        This method is called only in the DataManager/main Qt thread.
        """
        self.raw_window_segment = result.raw_segment
        self.loaded_segment = result.raw_segment
        self.loaded_fs_hz = result.processed_fs_hz

        self.loaded_data = {
            "amp": result.amp_display,
            "t": result.time_s,
            "x": result.distance_m,
            "time_stamps": np.array([
                result.raw_segment.start_time.timestamp()
            ]),
        }

        self.current_start_time = result.raw_segment.start_time

        # The worker already generated F-X values, so do not recompute them.
        self.fx_manager.fx_series_data = result.fx_amp
        self.fx_manager.freq = result.fx_freq
        self.fx_manager.x = result.fx_distance_m
        self.fx_manager.plot_start_time = result.fx_time_s

        self.window_changed.emit(
            self.current_start_time
        )

        self.dataset_loaded.emit()
    
    def request_window_async(
        self,
        start_time: datetime,
    ) -> None:
        """
        Request a visible window asynchronously.

        If the requested window has already been prefetched, install it
        immediately. If that exact window is still being prefetched, promote
        the prefetch result into the active visible request rather than
        enqueueing duplicate expensive work.
        """
        if not self.dataset_service.is_open:
            raise RuntimeError("No raw DAS dataset is open.")

        if self.selected_channels is None:
            raise RuntimeError("No DAS channel range is selected.")

        if self.filepath is None:
            raise RuntimeError(
                "No selected DAS source file is available."
            )

        if start_time.tzinfo is None:
            raise ValueError(
                "Window start time must be timezone-aware."
            )

        start_time = start_time.astimezone(timezone.utc)

        requested_key = self._window_cache_key(
            start_time=start_time,
        )

        # --------------------------------------------------------------
        # Case 1: completed prefetch exists. Install it immediately.
        # --------------------------------------------------------------
        # A fully prepared forward window already exists. Install it immediately.
        if (
            self._prefetched_window is not None
            and self._prefetch_key == requested_key
        ):
            result = self._prefetched_window

            print(
                "FORWARD PREFETCH HIT: "
                f"{start_time.isoformat()} "
                f"(request_id={result.request_id})"
            )

            self._clear_prefetch()
            self._install_prepared_window(result)

            # Begin preparing the following forward window after the normal delay.
            self.schedule_forward_prefetch()
            
            return

        # --------------------------------------------------------------
        # Case 2: the exact desired next window is still loading as a
        # prefetch. Promote it to the active visible request.
        #
        # Do not submit duplicate disk/processing work.
        # --------------------------------------------------------------
        if (
            self._prefetch_pending
            and self._prefetch_key == requested_key
            and self._prefetch_request_id is not None
        ):
            self._active_window_request_id = (
                self._prefetch_request_id
            )

            self.window_loading_started.emit(start_time)

            return

        # --------------------------------------------------------------
        # Case 3: normal visible request.
        # --------------------------------------------------------------
        self._window_request_id += 1
        request_id = self._window_request_id

        self._active_window_request_id = request_id

        request = WindowLoadRequest(
            request_id=request_id,
            selected_file=Path(self.filepath),
            interrogator=str(
                self.dataset_service.interrogator
            ),
            start_time=start_time,
            duration_s=float(
                self.user_settings.get("duration_s", 40.0)
            ),
            selected_channels=self.selected_channels,
            correct_file_boundaries=bool(
                self.user_settings.get(
                    "file_boundary_correction_enabled",
                    True,
                )
            ),
            boundary_edge_duration_s=float(
                self.user_settings.get(
                    "file_boundary_edge_duration_s",
                    0.25,
                )
            ),
            processing_settings=self.user_settings.copy(),
            kind="visible",
        )

        # The exact requested window is already being computed as a prefetch.
        # Do not submit duplicate HDF5/processing work. Mark it as the active
        # visible request; when the worker completes, its result will display.
        if (
            self._prefetch_pending
            and self._prefetch_key == requested_key
            and self._prefetch_request_id is not None
        ):
            self._active_window_request_id = (
                self._prefetch_request_id
            )

            print(
                "FORWARD PREFETCH PROMOTED: "
                f"{start_time.isoformat()} "
                f"(request_id={self._prefetch_request_id})"
            )

            self.window_loading_started.emit(start_time)

            return
        
        self.window_loading_started.emit(start_time)
        self.load_requested.emit(request)
    
    def schedule_forward_prefetch(self) -> None:
        """
        Prepare the likely next forward window after a short idle delay.

        The delay prevents the prefetch from immediately occupying the only
        worker thread when the user may want to navigate again right away.
        """
        if (
            self.current_start_time is None
            or self.selected_channels is None
            or self.filepath is None
            or not self.dataset_service.is_open
        ):
            return

        # Replace any old delayed prefetch timer with a new one.
        if self._prefetch_timer.isActive():
            self._prefetch_timer.stop()

        self._prefetch_timer.start(
            self._prefetch_delay_ms
        )

    def _request_forward_prefetch(self) -> None:
        """
        Start a low-priority load of exactly one likely forward window.
        """
        if (
            self.current_start_time is None
            or self.selected_channels is None
            or self.filepath is None
            or not self.dataset_service.is_open
        ):
            return

        # Do not queue another prefetch while one is already being computed.
        if self._prefetch_pending:
            return

        step_s = float(
            self.user_settings.get(
                "navigation_step_s",
                25.0,
            )
        )

        next_start_time = self.current_start_time + timedelta(
            seconds=step_s
        )

        prefetch_key = self._window_cache_key(
            start_time=next_start_time,
        )

        # Already have the desired completed forward window.
        if (
            self._prefetched_window is not None
            and self._prefetch_key == prefetch_key
        ):
            return

        self._window_request_id += 1

        request_id = self._window_request_id

        self._prefetch_key = prefetch_key
        self._prefetch_request_id = request_id
        self._prefetch_pending = True

        request = WindowLoadRequest(
            request_id=request_id,

            selected_file=Path(self.filepath),
            interrogator=str(
                self.dataset_service.interrogator
            ),

            start_time=next_start_time,
            duration_s=float(
                self.user_settings.get(
                    "duration_s",
                    40.0,
                )
            ),

            selected_channels=self.selected_channels,

            correct_file_boundaries=bool(
                self.user_settings.get(
                    "file_boundary_correction_enabled",
                    True,
                )
            ),

            boundary_edge_duration_s=float(
                self.user_settings.get(
                    "file_boundary_edge_duration_s",
                    0.25,
                )
            ),

            processing_settings=self.user_settings.copy(),

            kind="prefetch",
        )

        print(
            "Starting forward prefetch: "
            f"{next_start_time.isoformat()} "
            f"(request_id={request_id})"
        )

        self.load_requested.emit(request)
    
    def shutdown_loader(self) -> None:
        """Stop the background window-loader thread during application exit."""
        if self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait()
            
    def _on_worker_window_ready(
        self,
        result: PreparedWindow,
    ) -> None:
        """
        Handle a completed visible request or speculative prefetch.

        Runs in DataManager's GUI/main Qt thread.
        """
        # --------------------------------------------------------------
        # Completed speculative forward prefetch.
        # --------------------------------------------------------------
        if result.request_id == self._prefetch_request_id:
            self._prefetch_pending = False

            # User pressed Forward while the prefetch was still loading.
            # The prefetch was promoted to the visible request.
            if result.request_id == self._active_window_request_id:
                print(
                    "FORWARD PREFETCH PROMOTED RESULT: "
                    f"{result.requested_start_time.isoformat()}"
                )

                self._clear_prefetch()
                self._install_prepared_window(result)
                self.schedule_forward_prefetch()
                return

            # Store completed speculative result for a later matching Forward.
            self._prefetched_window = result

            print(
                "Forward prefetch ready: "
                f"{result.requested_start_time.isoformat()} "
                f"(request_id={result.request_id})"
            )

            return

        # --------------------------------------------------------------
        # Completed visible request.
        # --------------------------------------------------------------
        if result.request_id != self._active_window_request_id:
            print(
                "Discarding stale visible result: "
                f"{result.requested_start_time.isoformat()} "
                f"(request_id={result.request_id})"
            )
            return

        print(
            "Visible window ready: "
            f"{result.requested_start_time.isoformat()} "
            f"(request_id={result.request_id})"
        )

        self._install_prepared_window(result)
        self.schedule_forward_prefetch()

    def _on_worker_window_failed(
        self,
        request_id: int,
        traceback_text: str,
    ) -> None:
        """Handle worker failures for visible or speculative requests."""
        if request_id == self._prefetch_request_id:
            print(
                "Forward prefetch failed:\n"
                f"{traceback_text}"
            )

            self._clear_prefetch()
            return

        if request_id != self._active_window_request_id:
            return

        self.window_loading_failed.emit(
            traceback_text
        )
    
    @staticmethod
    def _prepared_window_nbytes(
        result: PreparedWindow,
    ) -> int:
        total = 0

        arrays = (
            result.raw_segment.data,
            result.raw_segment.time_s,
            result.raw_segment.distance_m,

            result.amp_display,
            result.time_s,
            result.distance_m,

            result.fx_amp,
            result.fx_freq,
            result.fx_distance_m,
            result.fx_time_s,
        )

        for array in arrays:
            if isinstance(array, np.ndarray):
                total += int(array.nbytes)

        return total
    
    def _resolve_selected_channels(self) -> tuple[int, int, int]:
        """
        Convert user physical cable settings into DAS4Whales channel settings.

        Returns:
            (start_channel_index, end_channel_index, channel_stride)
        """
        if self.dataset_record is None:
            raise RuntimeError("No dataset metadata are available.")

        settings = self.user_settings

        n_channels = self.dataset_record.n_channels
        native_dx_m = self.dataset_record.dx_m

        start_distance_m = float(
            self.dataset_record.metadata["start_distance_m"]
        )
        end_distance_m = float(
            self.dataset_record.metadata["end_distance_m"]
        )

        if settings.get("use_full_cable_length", True):
            start_index = 0
            end_index = n_channels
        else:
            requested_start_m = float(settings["cable_start_m"])
            requested_end_m = float(settings["cable_end_m"])

            requested_start_m = max(
                start_distance_m,
                min(requested_start_m, end_distance_m),
            )
            requested_end_m = max(
                start_distance_m,
                min(requested_end_m, end_distance_m),
            )

            if requested_end_m <= requested_start_m:
                raise ValueError(
                    "Resolved cable end distance must be greater than start distance."
                )

            start_index = int(
                round((requested_start_m - start_distance_m) / native_dx_m)
            )
            end_index = int(
                round((requested_end_m - start_distance_m) / native_dx_m)
            )

            start_index = max(0, min(start_index, n_channels - 1))
            end_index = max(start_index + 1, min(end_index, n_channels))

        requested_dx_m = float(
            settings.get("target_dx_m", native_dx_m)
        )

        stride = max(1, int(round(requested_dx_m / native_dx_m)))

        return start_index, end_index, stride
       
    def open_raw_dataset(
        self,
        filepath: str | Path,
        settings: dict,
        *,
        initial_start_time: datetime | None = None,
    ) -> None:
            """
            Open/index a raw DAS dataset directory and load the first window.
            """
            
            self._invalidate_pending_window_requests()
            
            filepath = Path(filepath).resolve()

            self.apply_user_settings(settings)

            self.dataset_record = self.dataset_service.open_dataset(
                selected_file=filepath,
                requested_interrogator=self.user_settings.get(
                    "interrogator",
                    "auto",
                ),
            )

            self.filepath = str(filepath)
            self.directory = str(filepath.parent)

            if initial_start_time is None:
                self.current_start_time = self.dataset_record.start_time

            else:
                if initial_start_time.tzinfo is None:
                    raise ValueError(
                        "initial_start_time must be timezone-aware."
                    )

                requested_start_time = initial_start_time.astimezone(
                    timezone.utc
                )

                # Do not request a time before the indexed dataset begins.
                self.current_start_time = max(
                    self.dataset_record.start_time,
                    requested_start_time,
                )

            self.selected_channels = self._resolve_selected_channels()

            self.dataset_opened.emit(self.dataset_record)

            # This is lightweight metadata/UI information; the actual raw data window
            # is requested asynchronously below.
            self.file_loaded.emit(
                filepath.name,
                self.current_start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            )

            # Raw DAS load, processing, and F-X computation now happen in the worker.
            self.request_window_async(self.current_start_time)

    def _invalidate_pending_window_requests(self) -> None:
        """
        Mark worker results from the prior dataset/settings state obsolete.

        Running worker operations are not force-cancelled; results are ignored
        when they eventually return.
        """
        self._window_request_id += 1
        self._active_window_request_id = None

        self._clear_prefetch()

    def load_raw_current_window(self) -> None:
        """
        Load, crop, stitch, and assemble the current raw DAS window.

        This is the expensive disk-I/O step. It should only run when the
        requested time window, source dataset, cable range, channel spacing,
        or boundary-correction settings change.
        """
        if not self.dataset_service.is_open:
            raise RuntimeError("No raw DAS dataset is open.")

        if self.current_start_time is None:
            raise RuntimeError("No current start time is set.")

        if self.selected_channels is None:
            raise RuntimeError("No DAS channel range is selected.")

        duration_s = float(
            self.user_settings.get("duration_s", 30.0)
        )

        self.raw_window_segment = self.dataset_service.load_window(
            start_time=self.current_start_time,
            duration_s=duration_s,
            selected_channels=self.selected_channels,
            correct_file_boundaries=self.user_settings.get(
                "file_boundary_correction_enabled",
                True,
            ),
            boundary_edge_duration_s=float(
                self.user_settings.get(
                    "file_boundary_edge_duration_s",
                    0.25,
                )
            ),
        )

        self.current_start_time = self.raw_window_segment.start_time

    def reprocess_current_window(
        self,
        recompute_fx: bool = True,
    ) -> None:
        """
        Reapply bandpass/F-K/downsampling/display conversion to the currently
        loaded raw window without rereading source files.
        """
        if self.raw_window_segment is None:
            raise RuntimeError("No raw DAS window has been loaded.")

        amp, fs_processed = apply_processing(
            amp=self.raw_window_segment.data,
            fs=self.raw_window_segment.fs_hz,
            dx_m=self.raw_window_segment.dx_m,
            settings=self.user_settings,
        )

        amp_display = amp * 1e9
        
        t = np.arange(
            amp_display.shape[1],
            dtype=float,
        ) / fs_processed

        self.loaded_segment = self.raw_window_segment
        self.loaded_fs_hz = fs_processed

        self.loaded_data = {
            "amp": amp_display,
            "t": t,
            "x": self.raw_window_segment.distance_m,
            "time_stamps": np.array([
                self.raw_window_segment.start_time.timestamp()
            ]),
        }

        self.current_start_time = self.raw_window_segment.start_time
        self.window_changed.emit(self.current_start_time)

        if recompute_fx:
            self._recompute_fx_for_current_window()

        self.dataset_loaded.emit()
    
    def load_current_window(self, recompute_fx: bool = True) -> None:
        start = perf_counter()

        raw_start = perf_counter()
        self.load_raw_current_window()
        raw_elapsed = perf_counter() - raw_start

        process_start = perf_counter()
        self.reprocess_current_window(
            recompute_fx=recompute_fx,
        )
        process_elapsed = perf_counter() - process_start

        total_elapsed = perf_counter() - start

        print(
            "Window load timing: "
            f"raw={raw_elapsed:.3f}s, "
            f"processing={process_elapsed:.3f}s, "
            f"total={total_elapsed:.3f}s"
        )
    
    def navigate(
        self,
        direction: str,
    ) -> None:
        """Move the current display window by configured navigation step."""
        if self.current_start_time is None:
            raise RuntimeError(
                "Open a dataset before navigating."
            )

        step_s = float(
            self.user_settings.get(
                "navigation_step_s",
                30.0,
            )
        )

        if direction == "forward":
            delta_s = step_s
        elif direction == "backward":
            delta_s = -step_s
        else:
            raise ValueError(
                f"Unknown navigation direction: {direction!r}"
            )

        requested_start = self.current_start_time + timedelta(
            seconds=delta_s
        )

        available_time = self.dataset_service.next_available_time(
            requested_start
        )

        if available_time is None:
            raise ValueError(
                "No later assigned DAS data are available."
            )

        # Requested timestamp falls in a locally absent assignment gap.
        if available_time != requested_start:
            self.window_loading_failed.emit(
                "Assignment gap encountered.\n\n"
                f"Requested: {requested_start.isoformat()}\n"
                f"Next assigned data begin: "
                f"{available_time.isoformat()}\n\n"
                "Use the start-time control to jump to the next block."
            )
            return

        self.request_window_async(requested_start)
            
    def set_window_start_time(
        self,
        start_time: datetime,
        *,
        reload_window: bool = True,
    ) -> None:
        if not self.dataset_service.is_open:
            raise RuntimeError(
                "Open a dataset before setting a start time."
            )

        if start_time.tzinfo is None:
            raise ValueError(
                "Start time must be timezone-aware. Use UTC."
            )

        start_time = start_time.astimezone(timezone.utc)

        if reload_window:
            self.request_window_async(start_time)
        else:
            self.current_start_time = start_time
            self.window_changed.emit(start_time)

    def set_navigation_step_s(
        self,
        navigation_step_s: float,
    ) -> None:
        """
        Update navigation step immediately without loading/reprocessing data.
        """
        navigation_step_s = float(navigation_step_s)

        if navigation_step_s <= 0:
            raise ValueError(
                "Navigation step must be greater than zero."
            )

        self.user_settings["navigation_step_s"] = navigation_step_s

    def set_duration_s(
        self,
        duration_s: float,
        *,
        reload_window: bool = True,
    ) -> None:
        duration_s = float(duration_s)

        if duration_s <= 0:
            raise ValueError(
                "Plot duration must be greater than zero."
            )

        self.user_settings["duration_s"] = duration_s

        if reload_window and self.current_start_time is not None:
            self.request_window_async(
                self.current_start_time
            )
         
    def apply_user_settings(
        self,
        settings: dict,
        *,
        emit_signal: bool = True,
    ) -> None:
        """Merge GUI settings into session settings."""

        self._clear_prefetch()
        self.user_settings.update(settings)

        if self.dataset_record is not None:
            self.selected_channels = self._resolve_selected_channels()

        if emit_signal:
            self.settings_changed.emit()

    def get_user_settings(self, key: str | None = None):
        """Return all settings or one settings value."""
        if key is None:
            return self.user_settings.copy()

        return self.user_settings.get(key)

    def get_labels_in_current_window(self) -> list[dict]:
        """Return saved labels whose apex lies in the displayed time window."""
        if self.label_saver is None:
            return []

        if (
            self.loaded_data["time_stamps"] is None
            or self.loaded_data["t"] is None
            or self.directory is None
        ):
            return []

        start_unix = float(self.loaded_data["time_stamps"][0])

        t_vec = self.loaded_data["t"]

        if len(t_vec) == 0:
            return []

        end_unix = start_unix + float(t_vec[-1])

        dataset_name = Path(self.directory).name

        df = self.label_saver.df

        if df.empty:
            return []

        matching = df[
            (df["apex_time_utc"] >= start_unix)
            & (df["apex_time_utc"] <= end_unix)
            & (df["dataset"] == dataset_name)
        ]

        return [
            row.to_dict()
            for _, row in matching.iterrows()
        ]

    def source_file_time_offset_s(
        self,
        apex_time_utc: float,
    ) -> tuple[Path, float]:
        """
        Return the physical source file containing apex_time_utc and the apex
        offset in seconds from that file's UTC start time.

        This is needed because the displayed T-X time vector is relative to the
        current displayed window, which may begin partway through a source file.
        """
        if not self.dataset_service.is_open:
            raise RuntimeError("No dataset is open.")

        apex_datetime = datetime.fromtimestamp(
            float(apex_time_utc),
            tz=timezone.utc,
        )

        record = self.dataset_service.find_file(apex_datetime)

        if record is None:
            raise ValueError(
                "Could not find indexed source file containing apex time: "
                f"{apex_datetime.isoformat()}"
            )

        apex_time_s = (
            apex_datetime - record.start_time
        ).total_seconds()

        return record.path, float(apex_time_s)

    def _recompute_fx_for_current_window(self) -> None:
        if self.loaded_data["amp"] is None:
            return

        start = perf_counter()

        self.fx_manager.update_data()
        fx_elapsed = perf_counter() - start

        spec_start = perf_counter()
        self.spectrogram_manager.update_data()
        spec_elapsed = perf_counter() - spec_start

        print(
            "Derived products timing: "
            f"F-X={fx_elapsed:.3f}s, "
            f"spectrogram setup={spec_elapsed:.3f}s"
        )
        
    def set_cursor_mode(self, mode: str) -> None:
        self.cursor_mode = mode


class FXHandle:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.fx_series_data = None
        self.freq = None
        self.x = None
        self.plot_start_time = None

    def update_data(self) -> None:
        amp = self.data_manager.loaded_data["amp"]
        x = self.data_manager.loaded_data["x"]
        t = self.data_manager.loaded_data["t"]
        fs = self.data_manager.loaded_fs_hz

        if amp is None or x is None or t is None or fs is None:
            self.fx_series_data = None
            self.freq = None
            self.x = None
            self.plot_start_time = None
            return

        fx_win_s = float(
            self.data_manager.get_user_settings("fx_win_s") or 2.0
        )

        fx_win_samples = int(round(fx_win_s * fs))

        if fx_win_samples < 2:
            raise ValueError(
                f"F-X window is too short: {fx_win_s} s at {fs} Hz."
            )

        requested_nfft = int(
            self.data_manager.get_user_settings("fx_nfft") or fx_win_samples
        )

        # Do not truncate the selected F-X time window accidentally.
        # If requested NFFT is larger, this zero-pads the FFT.
        fx_nfft = max(fx_win_samples, requested_nfft)

        if amp.shape[1] < fx_win_samples:
            # Not enough time samples to make one F-X slice.
            self.fx_series_data = None
            self.freq = None
            self.x = x
            self.plot_start_time = []
            return

        step_samples = fx_win_samples
        freqs = np.fft.rfftfreq(fx_nfft, d=1.0 / fs)

        slices = []
        t_win = []

        for start in range(
            0,
            amp.shape[1] - fx_win_samples + 1,
            step_samples,
        ):
            segment = amp[:, start:start + fx_win_samples]

            t_win.append(float(t[start]))

            # Result shape:
            # (n_channels, n_frequencies)
            spectrum = np.fft.rfft(segment, n=fx_nfft, axis=1)
            spectrum = 2.0 * np.abs(spectrum) / fx_win_samples
            spectrum[:, 0] *= 0.5 # don't double DC bin

            if fx_nfft % 2 == 0:
                spectrum[:, -1] *= 0.5 # For even-length windows, don't double Nyquist bin.

            slices.append(spectrum)

        self.fx_series_data = np.stack(slices, axis=0)

        # Final shape:
        # (n_time_slices, n_channels, n_frequencies)
        self.freq = freqs
        self.x = x
        self.plot_start_time = np.asarray(t_win, dtype=float)

    def get_dataset(self) -> dict:
        return {
            "amp": self.fx_series_data,
            "freq": self.freq,
            "x": self.x,
            "t": self.plot_start_time,
        }


class SpectrogramHandle:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def update_data(self):
        pass

    def calc_spectrogram(self, row_idx: int):
        nfft = int(
            self.data_manager.get_user_settings("spec_nfft") or 256
        )

        percent_overlap = float(
            self.data_manager.get_user_settings("spec_overlap") or 50
        )

        fs = self.data_manager.loaded_fs_hz
        amp = self.data_manager.loaded_data["amp"]

        if amp is None or fs is None:
            raise RuntimeError("No data are loaded.")

        if not (0 <= row_idx < amp.shape[0]):
            raise IndexError(
                f"row_idx={row_idx} is outside valid range "
                f"[0, {amp.shape[0] - 1}]"
            )

        sig = amp[row_idx, :]

        # scipy.signal.spectrogram requires nperseg <= signal length.
        nfft = min(nfft, len(sig))

        if nfft < 2:
            raise ValueError(
                "Loaded signal is too short to calculate a spectrogram."
            )

        noverlap = int(nfft * percent_overlap / 100)
        noverlap = min(noverlap, nfft - 1)

        window = sp.windows.tukey(nfft, alpha=0.25)
        window_rms = np.sqrt(np.sum(window ** 2))

        freqs, times, sxx = sp.spectrogram(
            sig,
            fs=fs,
            window=window,
            nperseg=nfft,
            noverlap=noverlap,
            scaling="spectrum",
            mode="magnitude",
        )

        sxx_corrected = sxx * nfft / window_rms

        return freqs, times, sxx_corrected
