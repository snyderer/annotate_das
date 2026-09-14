from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any
import numpy as np
from collections import OrderedDict

from annotate.models import RawDASSegment
from annotate.data.dataset_reader import DAS4WhalesDatasetReader
from annotate.data.processing import stitch_file_offsets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    """
    Lightweight index entry for one raw DAS source file.

    This intentionally contains metadata only. It does not contain the
    potentially large DAS signal array.
    """

    path: Path

    # Absolute UTC coverage of the file.
    start_time: datetime
    end_time: datetime

    interrogator: str

    # Native/source metadata.
    fs_hz: float
    dx_m: float
    n_samples: int
    n_channels: int

    # Normalized metadata produced by DAS4WhalesDatasetReader.inspect_file().
    metadata: dict[str, Any]

    def contains(self, time: datetime) -> bool:
        """Return True if time falls within [start_time, end_time)."""
        if time.tzinfo is None:
            raise ValueError("Time must be timezone-aware.")

        time = time.astimezone(timezone.utc)

        return self.start_time <= time < self.end_time

    def overlaps(self, start_time: datetime, end_time: datetime) -> bool:
        """Return True if this file overlaps [start_time, end_time)."""
        return self.start_time < end_time and self.end_time > start_time


class DatasetService:
    """
    GUI-facing coordinator for a currently opened DAS dataset directory.

    Responsibilities
    ----------------
    - Store the currently opened dataset directory.
    - Resolve an interrogator once when a new dataset is opened.
    - Build and retain a metadata-only time-to-file index.
    - Find source files for requested absolute time ranges.
    - Load complete source files through DAS4Whales.
    - Cache recently loaded source files.
    - Crop, stitch, and concatenate a requested raw DAS time window.

    Does not
    --------
    - Plot.
    - Apply bandpass, F-K, or display processing.
    - Manipulate Qt widgets.
    """

    SOURCE_PATTERNS = ("*.h5", "*.hdf5", "*.tdms")

    def __init__(self):
        self.reader = DAS4WhalesDatasetReader()

        self.dataset_directory: Path | None = None

        # Always a concrete resolved value after open_dataset(), for example:
        # "optasense", "silixa", "asn", etc.
        self.interrogator: str | None = None

        # Either "auto" or "user".
        self.interrogator_source: str | None = None

        # Sorted by FileRecord.start_time.
        self.file_records: list[FileRecord] = []

        # Retain non-fatal indexing errors for display/debugging.
        self.index_failures: list[str] = []
        
        # Cache full loaded source-file segments.
        #
        # Key:
        #     (absolute filepath, selected_channels tuple)
        #
        # Keep this small because raw DAS files are large.
        self._file_cache: OrderedDict[
            tuple[Path, tuple[int, int, int]],
            RawDASSegment,
        ] = OrderedDict()

        self._max_cached_files = 2

    @property
    def is_open(self) -> bool:
        """True when a dataset directory has been successfully indexed."""
        return self.dataset_directory is not None and bool(self.file_records)

    @staticmethod
    def _file_cache_key(
        filepath: Path,
        selected_channels: tuple[int, int, int],
    ) -> tuple[Path, tuple[int, int, int]]:
        return (
            filepath.resolve(),
            tuple(int(value) for value in selected_channels),
        )

    def _load_full_file_cached(
        self,
        filepath: Path,
        selected_channels: tuple[int, int, int],
    ) -> RawDASSegment:
        """
        Load a complete source file, reusing a cached segment if available.

        The cache contains raw source-file segments before cropping and
        before boundary stitching.
        """
        key = self._file_cache_key(filepath, selected_channels)

        cached = self._file_cache.get(key)

        if cached is not None:
            # Mark this item as most recently used.
            self._file_cache.move_to_end(key)
            logger.debug("Using cached DAS file: %s", filepath.name)
            return cached

        logger.debug("Loading DAS file: %s", filepath.name)

        segment = self.reader.load_file(
            filepath=filepath,
            selected_channels=selected_channels,
        )

        self._file_cache[key] = segment
        self._file_cache.move_to_end(key)

        # Evict least-recently-used segments.
        while len(self._file_cache) > self._max_cached_files:
            evicted_key, _ = self._file_cache.popitem(last=False)
            logger.debug(
                "Evicted cached DAS file: %s",
                evicted_key[0].name,
            )

        return segment
    
    def clear(self) -> None:
        """Clear all state associated with the current dataset."""
        self.dataset_directory = None
        self.interrogator = None
        self.interrogator_source = None
        self.file_records = []
        self.index_failures = []

        # Reset reader to default behavior.
        self.reader.set_interrogator("auto")
        
        self._file_cache.clear()

    def open_dataset(
        self,
        selected_file: str | Path,
        requested_interrogator: str = "auto",
    ) -> FileRecord:
        """
        Open the selected file's parent directory as a DAS dataset.

        The directory is re-indexed only when:

        - The selected file belongs to a new directory, or
        - The user explicitly changes interrogator type.

        Parameters
        ----------
        selected_file:
            A user-selected DAS source file.

        requested_interrogator:
            Either "auto" or a concrete DAS4Whales interrogator identifier,
            such as "optasense", "silixa", "asn", "onyx", or "fosina_dxs".

        Returns
        -------
        FileRecord
            The index record corresponding to selected_file.
        """
        selected_file = Path(selected_file).expanduser().resolve()

        if not selected_file.is_file():
            raise FileNotFoundError(
                f"Selected dataset file does not exist: {selected_file}"
            )

        if requested_interrogator is None:
            requested_interrogator = "auto"
            
        requested_interrogator = requested_interrogator.lower().strip()

        if not requested_interrogator:
            requested_interrogator = "auto"

        directory = selected_file.parent

        directory_changed = directory != self.dataset_directory

        interrogator_request_changed = (
            not directory_changed
            and (
                (
                    requested_interrogator == "auto"
                    and self.interrogator_source != "auto"
                )
                or (
                    requested_interrogator != "auto"
                    and requested_interrogator != self.interrogator
                )
            )
        )

        if directory_changed or interrogator_request_changed:
            self._open_new_dataset(
                selected_file=selected_file,
                requested_interrogator=requested_interrogator,
            )

        return self.get_record(selected_file)

    def _open_new_dataset(
        self,
        *,
        selected_file: Path,
        requested_interrogator: str,
    ) -> None:
        
        """
        Resolve the interrogator once and build a new directory file index.
        """
        
        self._file_cache.clear()
        
        logger.info(
            "Opening DAS dataset directory: %s",
            selected_file.parent,
        )

        if requested_interrogator == "auto":
            # This calls reader._detect_interrogator() internally.
            # It occurs once when opening a new directory, not during
            # normal navigation/loading.
            self.reader.set_interrogator("auto")

            resolved_interrogator = self.reader.resolve_interrogator(
                selected_file
            )
            interrogator_source = "auto"

        else:
            resolved_interrogator = requested_interrogator
            interrogator_source = "user"
        
        self.reader.set_interrogator(resolved_interrogator)

        records, failures = self._build_file_index(
            directory=selected_file.parent,
            interrogator=resolved_interrogator,
        )

        self.dataset_directory = selected_file.parent
        self.interrogator = resolved_interrogator
        self.interrogator_source = interrogator_source
        self.file_records = records
        self.index_failures = failures

        logger.info(
            "Opened dataset: directory=%s interrogator=%s source=%s files=%d",
            self.dataset_directory,
            self.interrogator,
            self.interrogator_source,
            len(self.file_records),
        )

    def _build_file_index(
        self,
        *,
        directory: Path,
        interrogator: str,
    ) -> tuple[list[FileRecord], list[str]]:
        """
        Build a metadata-only index for source files in one directory.

        This must not call reader.load_file(), since that would read full
        DAS arrays. It calls:

        - reader.inspect_file() for native fs/dx/sample metadata.
        - reader.get_file_start_time() for lightweight absolute timing.

        Returns
        -------
        records, failures
            records are sorted by absolute file start time.
            failures contains human-readable non-fatal index errors.
        """
        source_files = self._find_source_files(directory)

        if not source_files:
            patterns = ", ".join(self.SOURCE_PATTERNS)
            raise FileNotFoundError(
                f"No DAS files found in {directory}. "
                f"Checked patterns: {patterns}"
            )

        records: list[FileRecord] = []
        failures: list[str] = []

        for filepath in source_files:
            try:
                info = self.reader.inspect_file(filepath)

                # Read absolute start time from interrogator-specific header metadata.
                # This must not load the full DAS signal array.
                
                start_time = self.reader.get_file_start_time(
                    filepath,
                    interrogator=interrogator,
                )

                start_time = self._as_utc(start_time)

                fs_hz = float(info["fs_hz"])
                dx_m = float(info["dx_m"])
                n_samples = int(info["n_samples"])
                n_channels = int(info["n_channels"])

                if fs_hz <= 0:
                    raise ValueError(f"Invalid sampling rate: {fs_hz}")

                if n_samples <= 0:
                    raise ValueError(f"Invalid sample count: {n_samples}")

                if n_channels <= 0:
                    raise ValueError(f"Invalid channel count: {n_channels}")

                # File coverage is represented as [start_time, end_time).
                end_time = start_time + timedelta(
                    seconds=n_samples / fs_hz
                )

                records.append(
                    FileRecord(
                        path=filepath,
                        start_time=start_time,
                        end_time=end_time,
                        interrogator=interrogator,
                        fs_hz=fs_hz,
                        dx_m=dx_m,
                        n_samples=n_samples,
                        n_channels=n_channels,
                        metadata=info,
                    )
                )

            except Exception as exc:
                message = f"{filepath.name}: {type(exc).__name__}: {exc}"
                failures.append(message)
                logger.warning("Could not index file: %s", message)

        if not records:
            details = "\n".join(f"  - {failure}" for failure in failures)
            raise RuntimeError(
                f"Could not index any DAS files in {directory}.\n{details}"
            )

        records.sort(key=lambda record: record.start_time)

        self._validate_index(records)

        return records, failures

    def _find_source_files(self, directory: Path) -> list[Path]:
        """
        Return supported DAS files directly inside directory.

        Change directory.glob(pattern) to directory.rglob(pattern) if raw
        data are organized in nested subdirectories.
        """
        files: set[Path] = set()

        for pattern in self.SOURCE_PATTERNS:
            files.update(
                path.resolve()
                for path in directory.glob(pattern)
                if path.is_file()
            )

        return sorted(files)

    def get_record(self, filepath: str | Path) -> FileRecord:
        """Return the FileRecord associated with filepath."""
        filepath = Path(filepath).expanduser().resolve()

        for record in self.file_records:
            if record.path == filepath:
                return record

        raise KeyError(
            f"File is not part of the current indexed dataset: {filepath}"
        )

    def find_file(self, requested_time: datetime) -> FileRecord | None:
        """
        Find the source file containing requested_time.

        Parameters
        ----------
        requested_time:
            Timezone-aware datetime. Converted to UTC internally.

        Returns
        -------
        FileRecord | None
            The source file containing the requested time, or None if no
            indexed file covers it.
        """
        requested_time = self._as_utc(requested_time)

        for record in self.file_records:
            if record.contains(requested_time):
                return record

        return None

    def files_overlapping(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[FileRecord]:
        """
        Return all source files overlapping [start_time, end_time).

        This will later be used by load_window() when a plot duration crosses
        one or more source file boundaries.
        """
        start_time = self._as_utc(start_time)
        end_time = self._as_utc(end_time)

        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time.")

        return [
            record
            for record in self.file_records
            if record.overlaps(start_time, end_time)
        ]

    def _validate_index(self, records: list[FileRecord]) -> None:
        """
        Report potentially problematic conditions in an index.

        This currently emits warnings rather than errors because later
        processing may normalize differing fs/dx values.
        """
        for previous, current in zip(records, records[1:]):

            if current.start_time < previous.end_time:
                logger.warning(
                    "Overlapping DAS files detected: "
                    "%s (%s to %s) overlaps %s (%s to %s)",
                    previous.path.name,
                    previous.start_time.isoformat(),
                    previous.end_time.isoformat(),
                    current.path.name,
                    current.start_time.isoformat(),
                    current.end_time.isoformat(),
                )

            if current.fs_hz != previous.fs_hz:
                logger.warning(
                    "Native sampling-rate change: %s = %s Hz, %s = %s Hz",
                    previous.path.name,
                    previous.fs_hz,
                    current.path.name,
                    current.fs_hz,
                )

            if current.dx_m != previous.dx_m:
                logger.warning(
                    "Native channel-spacing change: %s = %s m, %s = %s m",
                    previous.path.name,
                    previous.dx_m,
                    current.path.name,
                    current.dx_m,
                )
                
    def load_window(
        self,
        start_time: datetime,
        duration_s: float,
        selected_channels: tuple[int, int, int],
        *,
        correct_file_boundaries: bool = True,
        boundary_edge_duration_s: float = 0.25,
    ) -> RawDASSegment:
        """
        Load raw DAS data for an absolute requested time window.

        DAS4Whales currently loads complete source files. This service loads
        whichever files overlap the requested window, crops each loaded file
        in memory, and concatenates the requested pieces.

        Parameters
        ----------
        start_time:
            Absolute timezone-aware UTC start time.

        duration_s:
            Requested display/load duration in seconds.

        selected_channels:
            DAS4Whales channel specification:
            (start_channel, end_channel, channel_spacing).

        Returns
        -------
        RawDASSegment
            Data restricted to the requested time interval, with shape:
            (n_channels, n_time_samples).
        """
        if not self.is_open:
            raise RuntimeError("No dataset is currently open.")

        start_time = self._as_utc(start_time)

        if duration_s <= 0:
            raise ValueError("duration_s must be greater than zero.")

        if boundary_edge_duration_s <= 0:
            raise ValueError("boundary_edge_duration_s must be positive.")

        end_time = start_time + timedelta(seconds=float(duration_s))

        records = self.files_overlapping(start_time, end_time)

        if not records:
            raise ValueError(
                "No indexed DAS files overlap requested window: "
                f"{start_time.isoformat()} to {end_time.isoformat()}"
            )

        pieces: list[RawDASSegment] = []

        for record in records:
            full_segment = self._load_full_file_cached(
                filepath=record.path,
                selected_channels=selected_channels,
            )

            piece = self._crop_segment(
                segment=full_segment,
                start_time=start_time,
                end_time=end_time,
            )

            if piece is not None:
                pieces.append(piece)

        if not pieces:
            raise ValueError(
                "Source files overlapped the requested window according to the "
                "index, but no DAS samples were found in that window."
            )
        
        if correct_file_boundaries:
            pieces = stitch_file_offsets(
                pieces,
                edge_duration_s=boundary_edge_duration_s, 
            )


        return self._concatenate_segments(
            pieces=pieces,
            requested_start_time=start_time,
        )
        
    def _crop_segment(
        self,
        *,
        segment: RawDASSegment,
        start_time: datetime,
        end_time: datetime,
    ) -> RawDASSegment | None:
        """
        Return the portion of segment overlapping [start_time, end_time).

        Returns None if no samples overlap.
        """
        start_time = self._as_utc(start_time)
        end_time = self._as_utc(end_time)

        segment_start = segment.start_time
        relative_start_s = (start_time - segment_start).total_seconds()
        relative_end_s = (end_time - segment_start).total_seconds()

        sample_start = int(
            np.searchsorted(segment.time_s, relative_start_s, side="left")
        )
        sample_stop = int(
            np.searchsorted(segment.time_s, relative_end_s, side="left")
        )

        if sample_stop <= sample_start:
            return None

        cropped_time = segment.time_s[sample_start:sample_stop]
        cropped_data = segment.data[:, sample_start:sample_stop]

        actual_start_time = segment.start_time + timedelta(
            seconds=float(cropped_time[0])
        )

        return RawDASSegment(
            data=cropped_data,
            time_s=cropped_time - cropped_time[0],
            distance_m=segment.distance_m,
            start_time=actual_start_time,
            fs_hz=segment.fs_hz,
            dx_m=segment.dx_m,
            source_file=segment.source_file,
            metadata={
                **segment.metadata,
                "crop_start_sample": sample_start,
                "crop_stop_sample": sample_stop,
                "parent_file": str(segment.source_file),
            },
        )
        
    def _concatenate_segments(
        self,
        *,
        pieces: list[RawDASSegment],
        requested_start_time: datetime,
    ) -> RawDASSegment:
        """
        Concatenate time-contiguous pieces from one or more source files.

        Initially, require identical native fs, dx, and distance coordinates.
        Later, processing.py can normalize mismatched source files.
        """
        pieces = sorted(pieces, key=lambda segment: segment.start_time)

        first = pieces[0]

        for previous, current in zip(pieces, pieces[1:]):
            if not np.isclose(current.fs_hz, first.fs_hz):
                raise ValueError(
                    "Cannot concatenate source files with different native "
                    f"sampling rates: {first.fs_hz} Hz vs {current.fs_hz} Hz."
                )

            if not np.isclose(current.dx_m, first.dx_m):
                raise ValueError(
                    "Cannot concatenate source files with different native "
                    f"channel spacing: {first.dx_m} m vs {current.dx_m} m."
                )

            if not np.array_equal(current.distance_m, first.distance_m):
                raise ValueError(
                    "Cannot concatenate source files with different "
                    "distance-coordinate vectors."
                )

            # Detect a true gap. Allow roughly one sample of tolerance.
            expected_start = previous.start_time + timedelta(
                seconds=previous.data.shape[1] / previous.fs_hz
            )
            gap_s = (current.start_time - expected_start).total_seconds()

            if gap_s > (1.5 / first.fs_hz):
                raise ValueError(
                    "Gap detected between DAS source files: "
                    f"{previous.source_file.name} -> {current.source_file.name}; "
                    f"gap={gap_s:.6f} seconds."
                )

        data = np.concatenate(
            [segment.data for segment in pieces],
            axis=1,
        )

        # Use a regular, zero-based relative time vector for the output window.
        time_s = np.arange(data.shape[1], dtype=float) / first.fs_hz

        return RawDASSegment(
            data=data,
            time_s=time_s,
            distance_m=first.distance_m,
            start_time=pieces[0].start_time,
            fs_hz=first.fs_hz,
            dx_m=first.dx_m,
            source_file=first.source_file,
            metadata={
                "source_files": [str(piece.source_file) for piece in pieces],
                "n_source_files": len(pieces),
                "requested_start_time": requested_start_time.isoformat(),
                "concatenated": len(pieces) > 1,
            },
        )
        
    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        """Validate and normalize a datetime to UTC."""
        if value.tzinfo is None:
            raise ValueError(
                "Datetime must be timezone-aware. Use UTC datetimes."
            )

        return value.astimezone(timezone.utc)