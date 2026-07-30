from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import datetime
import getpass
import json
import os
import sqlite3

import numpy as np
import scipy.signal as sp
from PyQt6.QtCore import QObject, pyqtSignal

import das4whales.data_handle as dh
from . import data_io as io

@dataclass
class DatasetMetadata:
    dataset_path: Path
    format_name: str

    fs_native: float
    x_native: np.ndarray

    file_paths: list[Path] = field(default_factory=list)
    file_start_times: np.ndarray | None = None
    file_end_times: np.ndarray | None = None

    extra: dict = field(default_factory=dict)


@dataclass
class LoadedWindow:
    """
    Format-independent data returned by every loader.

    amp always has shape:
        (n_channels, n_time_samples)
    """
    amp: np.ndarray
    t: np.ndarray
    x: np.ndarray

    start_timestamp: float
    end_timestamp: float
    fs: float

    source_files: list[Path] = field(default_factory=list)
    resolved_channel_indices: np.ndarray | None = None
    resolved_loading_settings: dict = field(default_factory=dict)

    @property
    def time_stamps(self) -> np.ndarray:
        return self.start_timestamp + self.t
    
    class BaseDataLoader(ABC):
    format_name = "unknown"

    @classmethod
    @abstractmethod
    def can_open(cls, filepath: str | Path) -> bool:
        pass

    @abstractmethod
    def open_dataset(self, filepath: str | Path) -> DatasetMetadata:
        pass

    @abstractmethod
    def load_window(
        self,
        *,
        start_timestamp: float,
        duration_s: float,
        channel_start_m: float | None,
        channel_end_m: float | None,
        target_dx_m: float | None,
        processing_settings: dict,
    ) -> LoadedWindow:
        pass
    
    class PreprocessedH5Loader(BaseDataLoader):
    format_name = "preprocessed"

    def __init__(self):
        self.metadata = None
        self._nonzeros_mask = None
        self._target_shape = None

    @classmethod
    def can_open(cls, filepath: str | Path) -> bool:
        filepath = Path(filepath)

        # Your legacy format is recognized by its companion settings.h5.
        return (
            filepath.suffix.lower() == ".h5"
            and (filepath.parent / "settings.h5").is_file()
        )

    def open_dataset(self, filepath: str | Path) -> DatasetMetadata:
        filepath = Path(filepath)
        settings_path = filepath.parent / "settings.h5"

        settings = io.load_settings_preprocessed_h5(settings_path)

        fs = float(settings["processing_settings"]["fs"])
        dx = float(settings["processing_settings"]["dx"])
        nx, ns = settings["rehydration_info"]["target_shape"]

        self._nonzeros_mask = settings["rehydration_info"]["nonzeros_mask"]
        self._target_shape = (nx, ns)

        file_paths = [
            filepath.parent / str(name)
            for name in settings["file_map"]["filename"]
        ]

        # This is acceptable initially. If a dataset has thousands of files,
        # build/cache this file index more efficiently later.
        file_start_times = []
        for file_path in file_paths:
            _, timestamp = io.load_preprocessed_h5(file_path)
            file_start_times.append(float(timestamp))

        file_start_times = np.asarray(file_start_times, dtype=float)
        file_end_times = file_start_times + ns / fs

        self.metadata = DatasetMetadata(
            dataset_path=filepath.parent,
            format_name=self.format_name,
            fs_native=fs,
            x_native=np.arange(nx, dtype=float) * dx,
            file_paths=file_paths,
            file_start_times=file_start_times,
            file_end_times=file_end_times,
            extra={"samples_per_file": ns},
        )
        return self.metadata

    def _read_file(self, filepath: Path) -> tuple[np.ndarray, float]:
        fk_dehyd, timestamp = io.load_preprocessed_h5(filepath)

        amp = 1e9 * io.rehydrate(
            fk_dehyd,
            self._nonzeros_mask,
            self._target_shape,
        )
        return amp, float(timestamp)

    def load_window(
        self,
        *,
        start_timestamp,
        duration_s,
        channel_start_m,
        channel_end_m,
        target_dx_m,
        processing_settings,
    ) -> LoadedWindow:
        if self.metadata is None:
            raise RuntimeError("Dataset has not been opened.")

        end_timestamp = start_timestamp + duration_s

        overlap = (
            (self.metadata.file_end_times > start_timestamp)
            & (self.metadata.file_start_times < end_timestamp)
        )
        selected_indices = np.flatnonzero(overlap)

        if selected_indices.size == 0:
            raise ValueError("No preprocessed files overlap this time window.")

        amp_parts = []
        abs_time_parts = []

        for idx in selected_indices:
            amp, file_start_timestamp = self._read_file(
                self.metadata.file_paths[idx]
            )

            file_t_abs = file_start_timestamp + (
                np.arange(amp.shape[1]) / self.metadata.fs_native
            )

            amp_parts.append(amp)
            abs_time_parts.append(file_t_abs)

        amp = np.concatenate(amp_parts, axis=1)
        absolute_time = np.concatenate(abs_time_parts)

        time_mask = (
            (absolute_time >= start_timestamp)
            & (absolute_time < end_timestamp)
        )

        amp = amp[:, time_mask]

        if amp.shape[1] == 0:
            raise ValueError("No samples remain after cropping the time window.")

        channel_indices, x = resolve_spatial_selection(
            self.metadata.x_native,
            channel_start_m,
            channel_end_m,
            target_dx_m,
        )
        amp = amp[channel_indices, :]

        amp, fs = apply_processing(
            amp=amp,
            fs=self.metadata.fs_native,
            dx_m=estimate_dx(x),
            settings=processing_settings,
        )

        t = np.arange(amp.shape[1], dtype=float) / fs

        return LoadedWindow(
            amp=amp,
            t=t,
            x=x,
            fs=fs,
            start_timestamp=start_timestamp,
            end_timestamp=start_timestamp + len(t) / fs,
            source_files=[
                self.metadata.file_paths[idx]
                for idx in selected_indices
            ],
            resolved_channel_indices=channel_indices,
            resolved_loading_settings={
                "resolved_channel_start_m": float(x[0]),
                "resolved_channel_end_m": float(x[-1]),
                "resolved_channel_dx_m": estimate_dx(x),
            },
        )
        
        
class DAS4WhalesLoader(BaseDataLoader):
    format_name = "das4whales"

    SUPPORTED_INTERROGATORS = {
        "optasense",
        "silixa",
        "mars",
        "asn",
        "onyx",
        "fosina",
        "fosina_dxs",
        "dxs",
    }

    def __init__(self, interrogator: str | None = None):
        self.interrogator = interrogator
        self.metadata = None

    @classmethod
    def can_open(cls, filepath: str | Path) -> bool:
        filepath = Path(filepath)

        # Do not mistake legacy preprocessed .h5 files for raw DAS files.
        if (filepath.parent / "settings.h5").is_file():
            return False

        return filepath.suffix.lower() in {".h5", ".hdf5", ".tdms"}

    def open_dataset(self, filepath: str | Path) -> DatasetMetadata:
        filepath = Path(filepath)

        if self.interrogator is None:
            self.interrogator = self._guess_interrogator(filepath)

        if self.interrogator not in self.SUPPORTED_INTERROGATORS:
            raise ValueError(
                "DAS interrogator could not be determined. "
                "Select it explicitly in the loading UI."
            )

        # Adjust this call and its unpacking to your installed DAS4Whales API.
        acquisition = dh.get_acquisition_parameters(
            str(filepath),
            self.interrogator,
        )

        fs = float(acquisition["fs"])
        dx = float(acquisition["dx"])
        n_channels = int(acquisition["n_channels"])

        file_paths, file_starts, file_ends = self._build_file_index(
            filepath,
            acquisition,
        )

        self.metadata = DatasetMetadata(
            dataset_path=filepath.parent,
            format_name=f"das4whales:{self.interrogator}",
            fs_native=fs,
            x_native=np.arange(n_channels, dtype=float) * dx,
            file_paths=file_paths,
            file_start_times=np.asarray(file_starts, dtype=float),
            file_end_times=np.asarray(file_ends, dtype=float),
            extra={
                "interrogator": self.interrogator,
                "acquisition": acquisition,
            },
        )
        return self.metadata

    def _guess_interrogator(self, filepath: Path) -> str | None:
        text = str(filepath).lower()

        if "optasense" in text:
            return "optasense"
        if "silixa" in text:
            return "silixa"
        if "svalbard" in text or "medsea" in text:
            return "asn"

        return None

    def _build_file_index(self, selected_file, acquisition):
        """
        DAS4Whales-specific implementation.

        Return:
            file_paths: list[Path]
            file_start_times: sequence[float]
            file_end_times: sequence[float]

        The exact implementation depends on the raw DAS source format and
        the DAS4Whales version/API you use.
        """
        raise NotImplementedError

    def _read_file(self, filepath: Path, channel_indices: np.ndarray):
        """
        Read one DAS raw file via DAS4Whales.

        Return:
            amp: ndarray shaped (n_selected_channels, n_samples)
            start_timestamp: float Unix seconds
        """
        raise NotImplementedError

    def load_window(
        self,
        *,
        start_timestamp,
        duration_s,
        channel_start_m,
        channel_end_m,
        target_dx_m,
        processing_settings,
    ) -> LoadedWindow:
        if self.metadata is None:
            raise RuntimeError("Dataset has not been opened.")

        end_timestamp = start_timestamp + duration_s

        overlap = (
            (self.metadata.file_end_times > start_timestamp)
            & (self.metadata.file_start_times < end_timestamp)
        )
        selected_indices = np.flatnonzero(overlap)

        if selected_indices.size == 0:
            raise ValueError("No DAS files overlap this time window.")

        channel_indices, x = resolve_spatial_selection(
            self.metadata.x_native,
            channel_start_m,
            channel_end_m,
            target_dx_m,
        )

        amp_parts = []
        abs_time_parts = []

        for idx in selected_indices:
            filepath = self.metadata.file_paths[idx]

            amp, file_start_timestamp = self._read_file(
                filepath,
                channel_indices,
            )

            file_t_abs = file_start_timestamp + (
                np.arange(amp.shape[1]) / self.metadata.fs_native
            )

            amp_parts.append(amp)
            abs_time_parts.append(file_t_abs)

        amp = np.concatenate(amp_parts, axis=1)
        absolute_time = np.concatenate(abs_time_parts)

        time_mask = (
            (absolute_time >= start_timestamp)
            & (absolute_time < end_timestamp)
        )
        amp = amp[:, time_mask]

        if amp.shape[1] == 0:
            raise ValueError("No samples remain after cropping the time window.")

        amp, fs = apply_processing(
            amp=amp,
            fs=self.metadata.fs_native,
            dx_m=estimate_dx(x),
            settings=processing_settings,
        )

        t = np.arange(amp.shape[1], dtype=float) / fs

        return LoadedWindow(
            amp=amp,
            t=t,
            x=x,
            fs=fs,
            start_timestamp=start_timestamp,
            end_timestamp=start_timestamp + len(t) / fs,
            source_files=[
                self.metadata.file_paths[idx]
                for idx in selected_indices
            ],
            resolved_channel_indices=channel_indices,
            resolved_loading_settings={
                "interrogator": self.interrogator,
                "resolved_channel_start_m": float(x[0]),
                "resolved_channel_end_m": float(x[-1]),
                "resolved_channel_dx_m": estimate_dx(x),
            },
        )
        
class DataManager(QObject):
    dataset_loaded = pyqtSignal()
    settings_changed = pyqtSignal()
    file_loaded = pyqtSignal(str, str)
    loading_parameters_resolved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.loader = None
        self.dataset_metadata = None

        self.filepath = ""
        self.directory = ""
        self.dataset_name = ""
        self.current_window_start_timestamp = None

        self.cursor_mode = ""
        self.label_saver = None
        self._user_settings = {}

        self.loaded_data = {
            "amp": None,
            "t": None,
            "x": None,
            "time_stamps": None,
            "start_timestamp": None,
            "end_timestamp": None,
            "fs": None,
            "dx": None,
            "source_files": [],
        }

        self.fx_manager = FXHandle(self)
        self.spectrogram_manager = SpectrogramHandle(self)

    def open_dataset(self, filepath, interrogator=None):
        filepath = Path(filepath).expanduser().resolve()

        self.filepath = str(filepath)
        self.directory = str(filepath.parent)
        self.dataset_name = filepath.parent.name

        self.loader = self._choose_loader(filepath, interrogator)
        self.dataset_metadata = self.loader.open_dataset(filepath)

        self.current_window_start_timestamp = self._get_selected_file_start(
            filepath
        )
        self.load_current_window(recompute_fx=True)

    def _choose_loader(self, filepath, interrogator):
        if PreprocessedH5Loader.can_open(filepath):
            return PreprocessedH5Loader()

        if DAS4WhalesLoader.can_open(filepath):
            return DAS4WhalesLoader(interrogator=interrogator)

        raise ValueError(f"Unsupported data format: {filepath}")

    def _get_selected_file_start(self, filepath):
        try:
            idx = self.dataset_metadata.file_paths.index(filepath)
            return float(self.dataset_metadata.file_start_times[idx])
        except ValueError:
            return float(self.dataset_metadata.file_start_times[0])

    def load_current_window(self, recompute_fx=True):
        settings = self.get_user_settings()
        duration_s = float(settings.get("duration_s", 60.0))

        window = self.loader.load_window(
            start_timestamp=self.current_window_start_timestamp,
            duration_s=duration_s,
            channel_start_m=settings.get("channel_start_m"),
            channel_end_m=settings.get("channel_end_m"),
            target_dx_m=settings.get("channel_dx_m"),
            processing_settings=settings,
        )

        self.loaded_data.update({
            "amp": window.amp,
            "t": window.t,
            "x": window.x,
            "time_stamps": window.time_stamps,
            "start_timestamp": window.start_timestamp,
            "end_timestamp": window.end_timestamp,
            "fs": window.fs,
            "dx": estimate_dx(window.x),
            "source_files": [str(p) for p in window.source_files],
        })

        self.loading_parameters_resolved.emit(
            window.resolved_loading_settings
        )

        if recompute_fx:
            self.fx_manager.update_data()
            self.spectrogram_manager.update_data()

        self._emit_file_info()
        self.dataset_loaded.emit()