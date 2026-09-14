from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from das4whales.data_handle import SIMPLEDAS_AVAILABLE
import numpy as np
import das4whales as dw
import h5py
from annotate.models import RawDASSegment, ReaderSettings
from nptdms import TdmsFile

class DAS4WhalesDatasetReader:
    """
    Application wrapper around DAS4Whales.

    This class is responsible for:
    - Passing the correct interrogator type to DAS4Whales.
    - Reading lightweight file metadata.
    - Obtaining the authoritative file start timestamp.
    - Loading source data.
    - Converting DAS4Whales output into the application's convention.

    This class does not:
    - Plot data.
    - Apply bandpass/F-K filtering.
    - Resample to target dx/fs.
    """

    def __init__(self, settings: ReaderSettings | None = None):
        self.settings = settings or ReaderSettings()


    def set_interrogator(self, interrogator: str) -> None:
        self.settings = ReaderSettings(interrogator=interrogator)

    def inspect_file(self, filepath: str | Path) -> dict[str, Any]:
        """
        Read lightweight metadata needed for indexing and loading.

        Do not load the full DAS array here.
        """
        filepath = Path(filepath)
        
        self._validate_file(filepath)
        
        interrogator = self.resolve_interrogator(filepath)

        metadata = dw.data_handle.get_acquisition_parameters(
            str(filepath),
            interrogator=interrogator,
        )

        normalized = self._normalize_metadata(
            filepath=filepath,
            interrogator=interrogator,
            metadata=metadata,
        )

        return normalized
            
    def resolve_interrogator(self, filepath: Path) -> str:
        """
        Return a DAS4Whales interrogator identifier.

        If user explicitly selected a type, use it.
        If "auto" is selected, attempt detection.
        """
        if self.settings.interrogator is None:
            configured = "auto"
        else:
            configured = self.settings.interrogator.lower().strip()

        if configured != "auto":
            return configured

        detected = self._detect_interrogator(filepath)

        if detected is None:
            raise ValueError(
                f"Could not automatically determine interrogator type for "
                f"{filepath.name}. Select an interrogator manually."
            )

        return detected

    def get_file_start_time(
        self,
        filepath: str | Path,
        *,
        interrogator: str | None = None,
    ) -> datetime:
        """
        Return the source-file start time as a timezone-aware UTC datetime.

        Uses interrogator-specific header metadata. Does not load the full
        DAS signal array.
        """
        filepath = Path(filepath)
        self._validate_file(filepath)

        resolved_interrogator = (
            interrogator.lower().strip()
            if interrogator is not None
            else self.resolve_interrogator(filepath)
        )

        if resolved_interrogator in {"optasense", "onyx"}:
            return self._get_optasense_style_start_time(filepath)

        if resolved_interrogator in {"fosina", "fosina_dxs", "dxs"}:
            return self._get_fosina_start_time(filepath)

        if resolved_interrogator == "asn":
            return self._get_asn_start_time(filepath)

        if resolved_interrogator == "silixa":
            return self._get_silixa_start_time(filepath)

        raise NotImplementedError(
            f"File start-time extraction is not implemented for "
            f"interrogator={resolved_interrogator!r}"
        )
        
    def load_file(
        self,
        filepath: str | Path,
        selected_channels,
    ) -> RawDASSegment:
        """
        Load data from a file through DAS4Whales.

        This example loads the selected channels for the full file.
        Add temporal slicing after confirming the relevant DAS4Whales API.
        """
        filepath = Path(filepath)

        info = self.inspect_file(filepath)
        metadata = info["das4whales_metadata"]

        tr, time, dist, file_begin_time_utc = dw.data_handle.load_das_data(
            str(filepath),
            selected_channels,
            metadata,
            interrogator=info["interrogator"]
        )

        return self._to_raw_segment(
            filepath=filepath,
            tr=tr,
            time=time,
            dist=dist,
            file_begin_time_utc=file_begin_time_utc,
            metadata=info,
        )

    def _validate_file(self, filepath: Path) -> None:
        if not filepath.is_file():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

    def _detect_interrogator(self, filepath: Path) -> str | None:
        """
        Detect only the interrogator types currently supported by this GUI:

        - silixa: TDMS files
        - asn: HDF5 files with ASN header/demodSpec/cableSpec structure
        - optasense: HDF5 files with OptaSense Acquisition/Custom attributes

        Returns None for unknown or ambiguous files. The user must then select
        an interrogator manually in the GUI.
        """
        suffix = filepath.suffix.lower()

        # Silixa datasets use TDMS files.
        if suffix == ".tdms":
            return "silixa"

        # ASN and OptaSense currently use HDF5.
        if suffix not in {".h5", ".hdf5"}:
            return None

        try:
            with h5py.File(filepath, "r") as h5:

                # ASN structure used by get_metadata_asn():
                #
                # /header/dt
                # /header/dx
                # /demodSpec/roiDec
                # /cableSpec/refractiveIndex
                if (
                    "header" in h5
                    and "demodSpec" in h5
                    and "cableSpec" in h5
                ):
                    return "asn"

                # OptaSense structure used by get_metadata_optasense():
                #
                # /Acquisition
                # /Acquisition/Raw[0]
                # /Acquisition/Custom
                #
                # /Acquisition/Custom attributes:
                # - Fibre Refractive Index
                # - Output Channel Start (CSU)
                if "Acquisition" not in h5:
                    return None

                acquisition = h5["Acquisition"]

                if "Raw[0]" not in acquisition:
                    return None

                if "Custom" not in acquisition:
                    return None

                custom_attrs = acquisition["Custom"].attrs

                has_refractive_index = (
                    "Fibre Refractive Index" in custom_attrs
                )
                has_output_channel_start = (
                    "Output Channel Start (CSU)" in custom_attrs
                )

                if has_refractive_index and has_output_channel_start:
                    return "optasense"

                # Do not infer an interrogator from generic Acquisition/Raw[0].
                # It may be another OptaSense-like HDF5 format or a future
                # unsupported format.
                return None

        except OSError:
            # File has an HDF5 extension but cannot be opened as HDF5.
            return None
        
    def _to_raw_segment(
        self,
        *,
        filepath: Path,
        tr,
        time,
        dist,
        file_begin_time_utc,
        metadata: dict[str, Any],
    ) -> RawDASSegment:
        """
        Normalize DAS4Whales output.

        Confirm DAS4Whales' array convention with your real files.
        The application standard is always (channel, time).
        """
        data = np.asarray(tr)
        time = np.asarray(time, dtype=float)
        dist = np.asarray(dist, dtype=float)

        # Check data shape and time/distance arrays for consistency. Raise ValueError if any checks fail.
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2-D DAS array, got shape {data.shape} "
                f"from {filepath.name}"
            )

        if len(time) == 0:
            raise ValueError(
                f"DAS4Whales returned no time samples for {filepath.name}"
            )

        if len(dist) == 0:
            raise ValueError(
                f"DAS4Whales returned no distance channels for {filepath.name}"
            )

        # If DAS4Whales returns (time, channel), transpose it.
        if data.shape == (len(time), len(dist)):
            data = data.T
        elif data.shape != (len(dist), len(time)):
            raise ValueError(
                "Unexpected DAS data shape returned by DAS4Whales. "
                f"data.shape={data.shape}, len(time)={len(time)}, "
                f"len(dist)={len(dist)}"
            )

        if data.shape != (len(dist), len(time)):
            raise ValueError(
                "Normalized data shape does not match coordinate arrays: "
                f"data.shape={data.shape}, len(dist)={len(dist)}, len(time)={len(time)}"
            )
            
        start_time = self._coerce_utc_datetime(file_begin_time_utc)

        fs_hz = float(metadata["fs_hz"])

        native_dx_m = float(metadata["dx_m"])
        resolved_dx_m = float(np.median(np.abs(np.diff(dist))))

        if not np.isfinite(resolved_dx_m) or resolved_dx_m <= 0:
            raise ValueError(
                f"Invalid resolved channel spacing for {filepath.name}: "
                f"{resolved_dx_m}"
            )
        
        return RawDASSegment(
            data=data.astype(np.float32, copy=False),
            time_s=time - time[0],
            distance_m=dist,
            start_time=start_time,
            fs_hz=fs_hz,
            dx_m=resolved_dx_m,
            source_file=filepath,
            metadata={
                **metadata,
                "native_dx_m": native_dx_m,
                "resolved_dx_m": resolved_dx_m,
            },
        )
        
    def _get_optasense_style_start_time(self, filepath: Path) -> datetime:
        """
        Read OptaSense/Onyx-style RawDataTime.

        Assumes RawDataTime contains Unix timestamps in microseconds.
        Confirm this assumption with a known file.
        """
        with h5py.File(filepath, "r") as fp:
            raw_data_time = fp["Acquisition"]["Raw[0]"]["RawDataTime"]

            # This appears to be an array, so [0] gets the first sample time.
            unix_time_us = float(raw_data_time[0])

        unix_time_s = unix_time_us * 1e-6

        return datetime.fromtimestamp(unix_time_s, tz=timezone.utc)
        
    def _get_asn_start_time(self, filepath: Path) -> datetime:
        """
        Read ASN header/time.

        header/time is a scalar HDF5 dataset, so use [()], not [0].
        """
        with h5py.File(filepath, "r") as fp:
            unix_time_s = float(fp["header"]["time"][()])

        return datetime.fromtimestamp(unix_time_s, tz=timezone.utc)    
    
    def _get_fosina_start_time(self, filepath: Path) -> datetime:
        """
        Read Fosina DxS PartStartTime ISO datetime attribute.
        """
        with h5py.File(filepath, "r") as fp:
            raw_value = fp["Acquisition"]["Raw[0]"]["RawDataTime"].attrs[
                "PartStartTime"
            ]

        # HDF5 attributes may arrive as bytes, np.bytes_, or str.
        if isinstance(raw_value, (bytes, np.bytes_)):
            raw_time = raw_value.decode("ascii").strip()
        else:
            raw_time = str(raw_value).strip()

        try:
            # Handles e.g.:
            # "2025-01-01T12:34:56Z"
            # "2025-01-01T12:34:56.123Z"
            # "2025-01-01T12:34:56+00:00"
            parsed = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))

        except ValueError:
            try:
                # Fallback for an example like:
                # "2025-01-01 12:34:56"
                parsed = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")

            except ValueError as exc:
                raise ValueError(
                    f"Could not parse Fosina PartStartTime {raw_time!r} "
                    f"in file {filepath.name}"
                ) from exc

        # A timestamp without timezone information is assumed to represent UTC.
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    def _get_silixa_start_time(self, filepath: Path) -> datetime:
        """
        Read Silixa start time from TDMS properties.

        The property name must be verified from your actual files.
        """
        tdms_file = TdmsFile.read(filepath)

        print(tdms_file.properties.keys())  # temporary inspection/debugging

        # Replace "StartTime" with the actual property name in your files.
        raw_time = tdms_file.properties["StartTime"]

        if isinstance(raw_time, datetime):
            if raw_time.tzinfo is None:
                return raw_time.replace(tzinfo=timezone.utc)
            return raw_time.astimezone(timezone.utc)

        if isinstance(raw_time, (int, float, np.integer, np.floating)):
            return datetime.fromtimestamp(float(raw_time), tz=timezone.utc)

        raise TypeError(
            f"Unsupported Silixa TDMS start-time type: {type(raw_time).__name__}"
        )
        
    def _normalize_metadata(
        self,
        *,
        filepath: Path,
        interrogator: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "filepath": str(filepath),
            "interrogator": interrogator,

            # Preserve DAS4Whales-native metadata unchanged. It may be needed
            # by dw.data_handle.load_das_data().
            "das4whales_metadata": metadata,

            # Normalize scalar metadata regardless of whether DAS4Whales returns
            # Python scalars, NumPy scalars, or one-element arrays.
            "fs_hz": self._as_scalar_float(metadata["fs"], "fs"),
            "dx_m": self._as_scalar_float(metadata["dx"], "dx"),
            "n_channels": self._as_scalar_int(metadata["nx"], "nx"),
            "n_samples": self._as_scalar_int(metadata["ns"], "ns"),
            "gauge_length_m": self._as_scalar_float(metadata["GL"], "GL"),
            "start_distance_m": self._as_scalar_float(
                metadata["start_dist"],
                "start_dist",
            ),
            "end_distance_m": self._as_scalar_float(
                metadata["end_dist"],
                "end_dist",
            ),

            # This currently works for ASN because its scale factor is a 1x1
            # array. If a future interrogator has a per-channel scale-factor
            # array, preserve it instead of forcing it to scalar.
            "scale_factor": self._as_scalar_float(
                metadata["scale_factor"],
                "scale_factor",
            ),
        }

    @staticmethod
    def _as_scalar_float(value, field_name: str) -> float:
        """
        Convert a Python scalar, NumPy scalar, or one-element NumPy array
        into a Python float.
        """
        array = np.asarray(value)

        if array.size != 1:
            raise ValueError(
                f"Expected metadata field {field_name!r} to have one value, "
                f"but got shape={array.shape}, size={array.size}."
            )

        return float(array.item())


    @staticmethod
    def _as_scalar_int(value, field_name: str) -> int:
        """
        Convert a Python scalar, NumPy scalar, or one-element NumPy array
        into a Python int.
        """
        array = np.asarray(value)

        if array.size != 1:
            raise ValueError(
                f"Expected metadata field {field_name!r} to have one value, "
                f"but got shape={array.shape}, size={array.size}."
            )

        return int(array.item())

    @staticmethod
    def _coerce_utc_datetime(value) -> datetime:
        """
        Adapt this depending on the type returned by DAS4Whales:
        datetime, Unix float, NumPy datetime64, etc.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float, np.integer, np.floating)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        if isinstance(value, np.datetime64):
            unix_s = value.astype("datetime64[ns]").astype(np.int64) / 1e9
            return datetime.fromtimestamp(unix_s, tz=timezone.utc)

        raise TypeError(
            f"Unsupported fileBeginTimeUTC type: {type(value).__name__}"
        )

