"""
DAS4Whales-backed source loader.

This module adapts DAS4Whales raw data formats to the application's
format-independent BaseDataLoader / LoadedWindow interface.

The exact DAS4Whales calls may need minor adjustment for the installed
DAS4Whales version and interrogator-specific file formats.
"""

from __future__ import annotations

from pathlib import Path
import datetime as dt
import inspect

import numpy as np
import das4whales.data_handle as dh

from .data_loader import (
    BaseDataLoader,
    DatasetMetadata,
    LoadedWindow,
    apply_processing,
    estimate_dx,
    resolve_spatial_selection,
    spatial_selection_metadata,
)


class DAS4WhalesDataLoader(BaseDataLoader):
    """
    Loader for raw DAS files supported by DAS4Whales.

    Parameters
    ----------
    interrogator
        DAS4Whales interrogator name, e.g. "optasense", "silixa", "asn".

        If None, a simple path-name guess is attempted. For production use,
        it is preferable to let the user select the interrogator explicitly
        in the loading UI when auto-detection is ambiguous.
    """

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

    # Modify these based on the raw formats you actually expect to support.
    DATA_FILE_SUFFIXES = {
        ".h5",
        ".hdf5",
        ".tdms",
    }

    def __init__(self, interrogator: str | None = None):
        self.interrogator = interrogator
        self.metadata: DatasetMetadata | None = None

    # ------------------------------------------------------------------
    # Loader identification / dataset opening
    # ------------------------------------------------------------------

    @classmethod
    def can_open(cls, filepath: str | Path) -> bool:
        """
        Return True for likely raw DAS data files.

        This deliberately avoids claiming legacy preprocessed files, which
        are identified by the presence of a companion settings.h5 file.
        """
        filepath = Path(filepath)

        if not filepath.is_file():
            return False

        # Your legacy preprocessed datasets should be handled by
        # PreprocessedDataLoader instead.
        if (filepath.parent / "settings.h5").is_file():
            return False

        return filepath.suffix.lower() in cls.DATA_FILE_SUFFIXES

    def open_dataset(self, filepath: str | Path) -> DatasetMetadata:
        """
        Open/index the DAS4Whales dataset containing `filepath`.

        This reads acquisition metadata and creates a list of source files
        with their absolute start/end times.
        """
        filepath = Path(filepath).expanduser().resolve()

        if self.interrogator is None:
            self.interrogator = self._guess_interrogator(filepath)

        if self.interrogator is None:
            raise ValueError(
                "Could not determine DAS interrogator type automatically. "
                "Please select an interrogator explicitly."
            )

        if self.interrogator not in self.SUPPORTED_INTERROGATORS:
            raise ValueError(
                f"Unsupported interrogator: {self.interrogator!r}. "
                f"Supported values: {sorted(self.SUPPORTED_INTERROGATORS)}"
            )

        acquisition = self._read_acquisition_parameters(filepath)

        fs = acquisition["fs"]
        dx = acquisition["dx"]
        n_channels = acquisition["n_channels"]

        # If DAS4Whales eventually provides physical channel positions
        # directly, use them here instead of np.arange(...) * dx.
        x_native = np.arange(n_channels, dtype=float) * dx

        file_paths, file_start_times, file_end_times = self._build_file_index(
            selected_file=filepath,
            fs=fs,
            n_samples_per_file=acquisition["n_samples"],
        )

        self.metadata = DatasetMetadata(
            dataset_path=filepath.parent,
            format_name=f"das4whales:{self.interrogator}",
            fs_native=fs,
            x_native=x_native,
            file_paths=file_paths,
            file_start_times=np.asarray(file_start_times, dtype=float),
            file_end_times=np.asarray(file_end_times, dtype=float),
            extra={
                "interrogator": self.interrogator,
                "acquisition": acquisition,
            },
        )

        return self.metadata

    def _guess_interrogator(self, filepath: Path) -> str | None:
        """
        Convenience-only interrogator detection.

        This is not reliable enough to be the sole method in production.
        Prefer an explicit combobox in the GUI with:
            Auto detect
            optasense
            silixa
            asn
            ...
        """
        text = str(filepath).lower()

        if "optasense" in text:
            return "optasense"

        if "silixa" in text:
            return "silixa"

        if "svalbard" in text or "medsea" in text:
            return "asn"

        if "onyx" in text:
            return "onyx"

        if "mars" in text:
            return "mars"

        return None

    # ------------------------------------------------------------------
    # DAS4Whales API adapter methods
    # ------------------------------------------------------------------

    def _read_acquisition_parameters(self, filepath: Path) -> dict:
        """
        Read and normalize DAS4Whales acquisition parameters.

        DAS4Whales versions may return a tuple, dict, or other structure.
        Keep that version-specific interpretation in this method.

        Returns
        -------
        dict with:
            fs          : sample rate in Hz
            dx          : channel spacing in metres
            n_channels  : number of channels
            n_samples   : samples per source file
            raw         : original DAS4Whales return value
        """
        # This print is useful during initial integration. Remove later.
        #
        # print("DAS4Whales get_acquisition_parameters signature:")
        # print(inspect.signature(dh.get_acquisition_parameters))

        raw = dh.get_acquisition_parameters(
            filepath=str(filepath),
            interrogator=self.interrogator,
        )

        """
        IMPORTANT:
        Inspect `raw` with your actual data:

            print(type(raw))
            print(raw)

        A common DAS4Whales pattern has historically been something like:

            fs, dx, nx, ns, gauge_length = dh.get_acquisition_parameters(...)

        but do not assume this is identical in every package release.
        """

        if isinstance(raw, dict):
            # Adapt these candidate keys if your installed version differs.
            fs = raw.get("fs", raw.get("sampling_frequency"))
            dx = raw.get("dx", raw.get("channel_spacing"))
            n_channels = raw.get("nx", raw.get("n_channels"))
            n_samples = raw.get("ns", raw.get("n_samples"))

        elif isinstance(raw, (tuple, list)):
            # Common expected tuple arrangement:
            # fs, dx, nx, ns, gauge_length = ...
            #
            # Confirm against your DAS4Whales installation.
            if len(raw) < 4:
                raise ValueError(
                    "DAS4Whales acquisition metadata tuple has fewer than "
                    "four values. Expected at least fs, dx, nx, ns."
                )

            fs, dx, n_channels, n_samples = raw[:4]

        else:
            raise TypeError(
                "Unsupported return type from "
                "dh.get_acquisition_parameters(): "
                f"{type(raw)}"
            )

        if fs is None or dx is None or n_channels is None or n_samples is None:
            raise ValueError(
                "Could not extract fs, dx, n_channels, and n_samples from "
                f"DAS4Whales acquisition parameters: {raw!r}"
            )

        return {
            "fs": float(fs),
            "dx": float(dx),
            "n_channels": int(n_channels),
            "n_samples": int(n_samples),
            "raw": raw,
        }

    def _build_file_index(
        self,
        selected_file: Path,
        fs: float,
        n_samples_per_file: int,
    ) -> tuple[list[Path], list[float], list[float]]:
        """
        Find files belonging to the selected dataset and index their timing.

        This version assumes related files are in the same directory and have
        matching file extensions. You will likely want to customize the file
        discovery rule for each dataset convention.

        Returns
        -------
        file_paths, file_start_times, file_end_times
        """
        candidates = sorted(
            path
            for path in selected_file.parent.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == selected_file.suffix.lower()
            )
        )

        if selected_file not in candidates:
            candidates.append(selected_file)
            candidates.sort()

        file_paths = []
        file_start_times = []
        file_end_times = []

        nominal_duration_s = n_samples_per_file / fs

        for path in candidates:
            try:
                file_start = self._read_file_start_timestamp(path)
            except Exception as exc:
                # During initial development it is reasonable to skip files
                # that do not belong to the same source family. Once your
                # dataset naming/layout rules are known, make discovery stricter.
                print(f"Skipping {path.name}: unable to read timestamp ({exc})")
                continue

            file_paths.append(path)
            file_start_times.append(file_start)
            file_end_times.append(file_start + nominal_duration_s)

        if not file_paths:
            raise ValueError(
                f"No readable DAS files were found in {selected_file.parent}"
            )

        # File names are not guaranteed to be chronological. Sort by time.
        ordered = np.argsort(file_start_times)

        file_paths = [file_paths[i] for i in ordered]
        file_start_times = [file_start_times[i] for i in ordered]
        file_end_times = [file_end_times[i] for i in ordered]

        return file_paths, file_start_times, file_end_times

    def _read_file_start_timestamp(self, filepath: Path) -> float:
        """
        Read the absolute start time of one DAS file.

        This is the most likely method you will need to customize because
        timestamp fields vary by interrogator and raw file format.

        Expected return:
            Unix timestamp in seconds, UTC.
        """
        metadata = self._read_das4whales_metadata(filepath)

        timestamp = self._extract_timestamp(metadata)

        if timestamp is None:
            raise ValueError(
                "No recognizable acquisition start timestamp in metadata."
            )

        return timestamp

    def _read_das4whales_metadata(self, filepath: Path):
        """
        Read only metadata needed to identify file start time.

        Replace this implementation with the correct DAS4Whales metadata
        function for your source type/version if available.

        If `load_das_data` is the only available API, this method may need to
        read a minimal channel/sample selection.
        """
        # ------------------------------------------------------------------
        # OPTION A: If DAS4Whales provides a metadata-only function:
        #
        # return dh.get_metadata(str(filepath), self.interrogator)
        #
        # OPTION B: If metadata is returned by load_das_data:
        #
        # data, metadata = dh.load_das_data(
        #     str(filepath),
        #     selected_channels=[0, 1],
        #     selected_samples=[0, 1],
        #     interrogator=self.interrogator,
        # )
        # return metadata
        # ------------------------------------------------------------------

        raise NotImplementedError(
            "Implement _read_das4whales_metadata() for your DAS4Whales "
            "version/interrogator. Inspect the package API with:\n"
            "  import inspect\n"
            "  print(inspect.signature(dh.load_das_data))\n"
            "  print(dir(dh))"
        )

    @staticmethod
    def _extract_timestamp(metadata) -> float | None:
        """
        Convert common timestamp metadata representations to Unix seconds.

        Adapt the possible key names after inspecting your real metadata.
        """
        if metadata is None:
            return None

        # Direct number.
        if isinstance(metadata, (int, float, np.integer, np.floating)):
            return float(metadata)

        # Python datetime.
        if isinstance(metadata, dt.datetime):
            if metadata.tzinfo is None:
                metadata = metadata.replace(tzinfo=dt.timezone.utc)
            return metadata.timestamp()

        # NumPy datetime64.
        if isinstance(metadata, np.datetime64):
            return float(
                metadata.astype("datetime64[ns]").astype(np.int64) / 1e9
            )

        if isinstance(metadata, dict):
            timestamp_keys = (
                "start_timestamp",
                "start_time",
                "file_start_time",
                "acquisition_start_time",
                "timestamp",
                "time",
                "datetime",
            )

            for key in timestamp_keys:
                if key not in metadata:
                    continue

                value = metadata[key]

                if isinstance(value, (int, float, np.integer, np.floating)):
                    return float(value)

                if isinstance(value, dt.datetime):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=dt.timezone.utc)
                    return value.timestamp()

                if isinstance(value, np.datetime64):
                    return float(
                        value.astype("datetime64[ns]").astype(np.int64) / 1e9
                    )

                if isinstance(value, bytes):
                    value = value.decode()

                if isinstance(value, str):
                    # Try ISO-8601 timestamps.
                    try:
                        value = value.replace("Z", "+00:00")
                        parsed = dt.datetime.fromisoformat(value)

                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=dt.timezone.utc)

                        return parsed.timestamp()

                    except ValueError:
                        pass

        return None

    # ------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------

    def _read_file(
        self,
        filepath: Path,
        channel_indices: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Read one raw DAS file through DAS4Whales.

        Returns
        -------
        amp
            Array with shape (n_selected_channels, n_samples).
        file_start_timestamp
            Absolute Unix timestamp.

        Notes
        -----
        This is the second main method to adapt after inspecting the exact
        DAS4Whales API for your interrogator.
        """
        # ------------------------------------------------------------------
        # This is representative pseudocode, not guaranteed package syntax:
        #
        # amp, metadata = dh.load_das_data(
        #     str(filepath),
        #     selected_channels=channel_indices,
        #     selected_samples=None,
        #     interrogator=self.interrogator,
        # )
        #
        # Some versions may instead expect:
        #
        # amp, metadata = dh.load_das_data(
        #     str(filepath),
        #     channel_indices,
        #     np.arange(n_samples),
        #     self.interrogator,
        # )
        #
        # Or may provide a different function entirely.
        # ------------------------------------------------------------------

        raise NotImplementedError(
            "Implement _read_file() using the DAS4Whales read function for "
            "your installed version and interrogator."
        )

    # ------------------------------------------------------------------
    # Common window assembly
    # ------------------------------------------------------------------

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
        """
        Load a requested physical time/spatial window.

        The source-reading portion is DAS4Whales-specific. Concatenation,
        cropping, spatial selection, and display processing follow the same
        logic as the legacy preprocessed loader.
        """
        if self.metadata is None:
            raise RuntimeError("Call open_dataset() before load_window().")

        if duration_s <= 0:
            raise ValueError("duration_s must be greater than zero.")

        requested_end_timestamp = start_timestamp + duration_s

        # Select all files overlapping [start_timestamp, requested_end).
        overlap_mask = (
            (self.metadata.file_end_times > start_timestamp)
            & (self.metadata.file_start_times < requested_end_timestamp)
        )
        selected_file_indices = np.flatnonzero(overlap_mask)

        if selected_file_indices.size == 0:
            raise ValueError(
                "No DAS files overlap the requested display time window."
            )

        # Resolve requested cable range and desired dx against actual native
        # channel geometry before reading files.
        channel_indices, x = resolve_spatial_selection(
            self.metadata.x_native,
            requested_start_m=channel_start_m,
            requested_end_m=channel_end_m,
            requested_dx_m=target_dx_m,
        )

        amp_parts = []
        absolute_time_parts = []

        for file_index in selected_file_indices:
            filepath = self.metadata.file_paths[file_index]

            amp_part, file_start_timestamp = self._read_file(
                filepath=filepath,
                channel_indices=channel_indices,
            )

            amp_part = np.asarray(amp_part)

            if amp_part.ndim != 2:
                raise ValueError(
                    f"DAS4Whales read for {filepath.name} returned shape "
                    f"{amp_part.shape}; expected 2-D "
                    "(n_channels, n_samples)."
                )

            # Some readers return (n_samples, n_channels). Detect and correct
            # the common case using the requested channel count.
            if (
                amp_part.shape[0] != len(channel_indices)
                and amp_part.shape[1] == len(channel_indices)
            ):
                amp_part = amp_part.T

            if amp_part.shape[0] != len(channel_indices):
                raise ValueError(
                    f"Read {filepath.name} returned {amp_part.shape[0]} "
                    f"channels, but {len(channel_indices)} were requested."
                )

            file_time = file_start_timestamp + (
                np.arange(amp_part.shape[1], dtype=float)
                / self.metadata.fs_native
            )

            amp_parts.append(amp_part)
            absolute_time_parts.append(file_time)

        amp = np.concatenate(amp_parts, axis=1)
        absolute_time = np.concatenate(absolute_time_parts)

        # Crop exactly to [start_timestamp, requested_end_timestamp).
        crop_mask = (
            (absolute_time >= start_timestamp)
            & (absolute_time < requested_end_timestamp)
        )

        amp = amp[:, crop_mask]
        absolute_time = absolute_time[crop_mask]

        if amp.shape[1] == 0:
            raise ValueError(
                "The requested DAS window contains no samples after cropping."
            )

        # Process after assembling/cropping the source data.
        actual_dx_m = estimate_dx(x)

        amp, fs = apply_processing(
            amp=amp,
            fs=self.metadata.fs_native,
            dx_m=actual_dx_m,
            settings=processing_settings,
        )

        # The first retained timestamp is the actual window origin.
        actual_start_timestamp = float(absolute_time[0])
        relative_t = np.arange(amp.shape[1], dtype=float) / fs

        resolved = spatial_selection_metadata(
            x_native=self.metadata.x_native,
            channel_indices=channel_indices,
            requested_start_m=channel_start_m,
            requested_end_m=channel_end_m,
            requested_dx_m=target_dx_m,
        )

        resolved.update({
            "interrogator": self.interrogator,
            "requested_start_timestamp": float(start_timestamp),
            "requested_duration_s": float(duration_s),
            "actual_start_timestamp": actual_start_timestamp,
            "actual_duration_s": float(len(relative_t) / fs),
            "native_fs_hz": float(self.metadata.fs_native),
            "loaded_fs_hz": float(fs),
            "n_source_files": int(len(selected_file_indices)),
        })

        return LoadedWindow(
            amp=amp,
            t=relative_t,
            x=x,
            start_timestamp=actual_start_timestamp,
            end_timestamp=actual_start_timestamp + len(relative_t) / fs,
            fs=fs,
            source_files=[
                self.metadata.file_paths[index]
                for index in selected_file_indices
            ],
            resolved_channel_indices=channel_indices,
            resolved_loading_settings=resolved,
        )