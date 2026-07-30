# annotate/data/dataset_reader.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import das4whales as dw
import h5py
from annotate.models import RawDASSegment, ReaderSettings


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
    def _validate_file(self, filepath: Path) -> None:
        if not filepath.is_file():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    def resolve_interrogator(self, filepath: Path) -> str:
        """
        Return a DAS4Whales interrogator identifier.

        If user explicitly selected a type, use it.
        If "auto" is selected, attempt detection.
        """
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


    def _detect_interrogator(self, filepath: Path) -> str | None:
        """
        Attempt to identify a DAS4Whales interrogator from file format and
        header structure.

        Returns
        -------
        str | None
            A DAS4Whales interrogator name, or None if detection is ambiguous
            or unsupported.
        """
        suffix = filepath.suffix.lower()

        # Based on the DAS4Whales metadata code you supplied.
        if suffix == ".tdms":
            return "silixa"

        if suffix not in {".h5", ".hdf5"}:
            return None

        try:
            with h5py.File(filepath, "r") as h5:

                # ASN structure:
                #
                # fp['header']['dt']
                # fp['header']['dx']
                # fp['demodSpec']['roiDec']
                # fp['cableSpec']
                if (
                    "header" in h5
                    and "demodSpec" in h5
                    and "cableSpec" in h5
                ):
                    return "asn"

                # OptaSense / Onyx / Fosina DxS family:
                if "Acquisition" not in h5:
                    return None

                acquisition = h5["Acquisition"]

                if "Raw[0]" not in acquisition:
                    return None

                raw = acquisition["Raw[0]"]
                raw_attrs = raw.attrs

                # Fosina DxS uses StartLocusIndex according to
                # get_metadata_fosina_dxs().
                if "StartLocusIndex" in raw_attrs:
                    return "fosina_dxs"

                # OptaSense metadata specifically uses:
                #
                # Acquisition/Custom.attrs['Fibre Refractive Index']
                # Acquisition/Custom.attrs['Output Channel Start (CSU)']
                #
                # Checking both gives stronger evidence than only checking
                # that /Acquisition exists.
                if "Custom" in acquisition:
                    custom_attrs = acquisition["Custom"].attrs

                    has_refractive_index = (
                        "Fibre Refractive Index" in custom_attrs
                    )
                    has_output_channel_start = (
                        "Output Channel Start (CSU)" in custom_attrs
                    )

                    if has_refractive_index and has_output_channel_start:
                        return "optasense"

                return None

        except OSError:
            # Not a readable HDF5 file, despite .h5/.hdf5 extension.
            return None

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
        )

        return self._to_raw_segment(
            filepath=filepath,
            tr=tr,
            time=time,
            dist=dist,
            file_begin_time_utc=file_begin_time_utc,
            metadata=info,
        )

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
                f"data={data.shape}, len(dist)={len(dist)}, len(time)={len(time)}"
            )
            
        start_time = self._coerce_utc_datetime(file_begin_time_utc)

        fs_hz = float(metadata["fs_hz"])
        dx_m = float(metadata["dx_m"])

        return RawDASSegment(
            data=data.astype(np.float32, copy=False),
            time_s=time - time[0],
            distance_m=dist,
            start_time=start_time,
            fs_hz=fs_hz,
            dx_m=dx_m,
            source_file=filepath,
            metadata=metadata,
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

            # Preserve the original DAS4Whales object/dict for load_das_data().
            "das4whales_metadata": metadata,

            # Application-owned normalized names.
            "fs_hz": float(metadata["fs"]),
            "dx_m": float(metadata["dx"]),
            "n_channels": int(metadata["nx"]),
            "n_samples": int(metadata["ns"]),
            "gauge_length_m": float(metadata["GL"]),
            "scale_factor": float(metadata["scale_factor"]),
            "start_distance_m": float(metadata["start_dist"]),
            "end_distance_m": float(metadata["end_dist"]),
        }

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